"""Motor health policy: raw STS3215 readings to motor_health levels and
alert lifecycles. Pure functions and small state holders, no bus access."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from so101_description.units import MOTOR_NAMES

LEVEL_NOMINAL = 0
LEVEL_WARNING = 1
LEVEL_CRITICAL = 2
LEVEL_FAULT = 3
LEVEL_NOT_REPORTING = 4

# Read cycles the whole bus may miss before every motor reads as silent.
SILENT_AFTER_MISSED_READS = 3

# motor_health cadence; the contract mandates at least 2 Hz.
HEALTH_RATE_HZ = 5

# How often the active alert set is published again unchanged, inside the
# alert contract's 2000 ms ceiling. The floor bounds the age of the
# measurement each message carries, recovers a consumer that lost or refused
# one, and is what lets a consumer judge a producer to have gone quiet.
ALERT_FLOOR_PERIOD_S = 1.6

# STS3215 judgment constants. The servo firmware self-protects at 70 C, so
# warn/crit sit below it; load fractions are of stall torque, and the EWMA
# window matches the servo's own 2 s overload-trip hold.
TEMP_WARN_C = 55.0
TEMP_CRIT_C = 65.0
LOAD_WARN_FRACTION = 0.5
LOAD_CRIT_FRACTION = 0.8
EWMA_TAU_S = 2.0

# Release points for the four thresholds above. A reading must fall this far
# back before its level clears, so a motor parked on a threshold holds one
# level, the alert set is published on a real change, and a late consumer
# reads a set that stands.
TEMP_WARN_RELEASE_C = 52.0
TEMP_CRIT_RELEASE_C = 62.0
LOAD_WARN_RELEASE_FRACTION = 0.4
LOAD_CRIT_RELEASE_FRACTION = 0.7

# The servo's own overload protection trips at this fraction of stall torque
# (held 2 s), the contract's "effective peak": 1.0 on the wire's
# effort_fraction_peak is that trip point.
TRIP_FRACTION_OF_STALL = 0.8

# Alert severities (alert:v1): 1 warn, 2 critical, 3 fault. A nominal motor
# is not listed.
_LEVEL_SEVERITY = {
    LEVEL_WARNING: 1,
    LEVEL_CRITICAL: 2,
    LEVEL_FAULT: 3,
    # Silence lands on the fault severity, as it does for every other
    # producer of this contract: a motor that has stopped answering is at
    # least as serious as one that said it faulted, because it has not said
    # anything.
    LEVEL_NOT_REPORTING: 3,
}

# Status-register fault bits, per the Feetech protocol (scservo_sdk ERRBIT_*).
_FAULT_BIT_NAMES = (
    (1, "voltage"),
    (2, "angle sensor"),
    (4, "overheating"),
    (8, "overcurrent"),
    (32, "overload"),
)


# Every bit the table above can name. A Status value carrying anything else
# is a fault this build does not know about, and dropping it would report a
# motor as merely overheating when it also raised something unrecognised.
_NAMED_FAULT_BITS = sum(bit for bit, _ in _FAULT_BIT_NAMES)


def describe_faults(bits: int) -> str:
    """Human name(s) for a Status register value. Bits with no name are
    reported as themselves rather than dropped, so a fault this build does
    not recognise still reaches the operator."""
    names = [name for bit, name in _FAULT_BIT_NAMES if bits & bit]
    unnamed = bits & ~_NAMED_FAULT_BITS
    if unnamed:
        names.append(f"unknown bits 0x{unnamed:02x}")
    return ", ".join(names) if names else "no fault"


class _Band:
    """One latched threshold: engages at `on`, releases below `off`."""

    def __init__(self, on: float, off: float):
        assert off < on, f"release {off} must sit below engage {on}"
        self._on = on
        self._off = off
        self._engaged = False

    def update(self, value: float) -> bool:
        self._engaged = value > self._off if self._engaged else value >= self._on
        return self._engaged


class Bands:
    """Per-motor hysteresis for the temperature and load thresholds, the
    state the levels are judged against.

    Every band is updated on every call, so each threshold's state follows
    its own channel.
    """

    def __init__(self) -> None:
        self._temp_warn = [_Band(TEMP_WARN_C, TEMP_WARN_RELEASE_C) for _ in MOTOR_NAMES]
        self._temp_crit = [_Band(TEMP_CRIT_C, TEMP_CRIT_RELEASE_C) for _ in MOTOR_NAMES]
        self._load_warn = [
            _Band(LOAD_WARN_FRACTION, LOAD_WARN_RELEASE_FRACTION) for _ in MOTOR_NAMES
        ]
        self._load_crit = [
            _Band(LOAD_CRIT_FRACTION, LOAD_CRIT_RELEASE_FRACTION) for _ in MOTOR_NAMES
        ]

    def thermal_level(self, motor: int, temp: float, load_sustained: float) -> int:
        """The level this motor's temperature and sustained load put it at,
        with no regard for faults, which outrank both."""
        temp_crit = self._temp_crit[motor].update(temp)
        load_crit = self._load_crit[motor].update(load_sustained)
        temp_warn = self._temp_warn[motor].update(temp)
        load_warn = self._load_warn[motor].update(load_sustained)
        if temp_crit or load_crit:
            return LEVEL_CRITICAL
        if temp_warn or load_warn:
            return LEVEL_WARNING
        return LEVEL_NOMINAL


class SustainedLoads:
    """EWMA of per-motor stall-torque fractions, the sustained estimate the
    levels judge."""

    def __init__(self, tau_s: float):
        self._tau_s = tau_s
        self._values: tuple[float, ...] | None = None

    def update(self, loads: tuple[float, ...], dt_s: float) -> tuple[float, ...]:
        if self._values is None or len(self._values) != len(loads):
            self._values = loads
            return self._values
        alpha = 1.0 - math.exp(-max(dt_s, 0.0) / self._tau_s)
        self._values = tuple(
            prev + alpha * (now - prev) for prev, now in zip(self._values, loads, strict=True)
        )
        return self._values

    def current(self) -> tuple[float, ...]:
        return self._values if self._values is not None else ()


@dataclass(frozen=True)
class HealthReport:
    levels: tuple[int, ...]
    # Status-register value per motor; nonzero drove that motor's FAULT.
    # Always motor-count long (a judgment input, not a wire vector).
    fault_bits: tuple[int, ...]
    # Instantaneous and EWMA |load| as fractions of stall torque, the basis
    # the thresholds are tuned against.
    stall_fractions: tuple[float, ...]
    stall_fractions_sustained: tuple[float, ...]
    winding_temp_c: tuple[float, ...]

    def peak_fractions(self) -> tuple[float, ...]:
        """The wire's effort_fraction_peak: instantaneous |load| against the
        servo's overload trip point (may exceed 1.0 above it)."""
        return tuple(f / TRIP_FRACTION_OF_STALL for f in self.stall_fractions)


def assess(
    temps_c: tuple[float, ...],
    loads: tuple[float, ...],
    torque_enabled: tuple[bool, ...],
    fault_bits: tuple[int, ...],
    sustained: tuple[float, ...],
    bus_silent: bool,
    bands: Bands,
) -> HealthReport:
    """One motor_health report. With a silent bus the last readings are not
    evidence of anything, so every vector empties and every level reads 4."""
    count = len(MOTOR_NAMES)
    if bus_silent:
        return HealthReport(
            levels=(LEVEL_NOT_REPORTING,) * count,
            fault_bits=(0,) * count,
            stall_fractions=(),
            stall_fractions_sustained=(),
            winding_temp_c=(),
        )

    def level(motor: int, temp: float, load_sustained: float, enabled: bool, faults: int) -> int:
        # The bands track every reading, including a motor whose fault
        # outranks them, so a fault clearing does not leave a stale band.
        thermal = bands.thermal_level(motor, temp, load_sustained)
        # A latched servo fault or a silently disabled motor cannot drive:
        # fault outranks the thermal and load judgments, which presume a
        # live actuator. Overload protection cuts output while leaving
        # Torque_Enable at 1, so the Status bits are checked first.
        if faults != 0 or not enabled:
            return LEVEL_FAULT
        return thermal

    return HealthReport(
        levels=tuple(
            level(motor, t, s, e, f)
            for motor, (t, s, e, f) in enumerate(
                zip(temps_c, sustained, torque_enabled, fault_bits, strict=True)
            )
        ),
        fault_bits=fault_bits,
        stall_fractions=loads,
        stall_fractions_sustained=sustained,
        winding_temp_c=temps_c,
    )


@dataclass(frozen=True)
class Alert:
    source: str
    kind: str
    severity: int
    message: str


@dataclass(frozen=True)
class AlertSet:
    """The active alerts a report owes the wire, with the per-motor
    conditions to commit once it is out."""

    conditions: tuple[tuple[int, int], ...]
    alerts: tuple[Alert, ...]


class AlertTracker:
    """Motor alert lifecycle: one (source, kind) identity per motor, and the
    whole set of active alerts owed whenever any motor's condition (its level
    plus fault bits) changes and again every `ALERT_FLOOR_PERIOD_S`, so a set
    that omits a motor clears it.

    The first round always owes a set, empty while every motor is nominal, so
    the topic holds a message for a consumer that subscribes later and that
    consumer can tell a quiet follower from one that has not started.

    Two-phase: `due` proposes the set and `mark_sent` commits it once its
    publish succeeded, so a failed send is owed again."""

    def __init__(self, source_prefix: str, monotonic=time.monotonic):
        self._source_prefix = source_prefix
        self._monotonic = monotonic
        self._published: tuple[tuple[int, int], ...] | None = None
        self._sent_monotonic_s = 0.0

    def _alert(self, motor: str, level: int, fault_bits: int) -> Alert:
        return Alert(
            source=f"{self._source_prefix} {motor}",
            kind="motor_condition",
            severity=_LEVEL_SEVERITY[level],
            message=_describe(motor, level, fault_bits),
        )

    def due(self, report: HealthReport) -> AlertSet | None:
        """The set this report owes: every motor with a condition, when any
        motor's condition differs from the last published set, when the floor
        has elapsed since it went out, or when nothing has gone out yet."""
        conditions = tuple(zip(report.levels, report.fault_bits, strict=True))
        floor_elapsed = (
            self._monotonic() - self._sent_monotonic_s >= ALERT_FLOOR_PERIOD_S
        )
        if conditions == self._published and not floor_elapsed:
            return None
        alerts = tuple(
            self._alert(motor, level, bits)
            for motor, (level, bits) in zip(MOTOR_NAMES, conditions, strict=True)
            if level != LEVEL_NOMINAL
        )
        return AlertSet(conditions, alerts)

    def mark_sent(self, owed: AlertSet) -> None:
        """Records a published set."""
        self._published = owed.conditions
        self._sent_monotonic_s = self._monotonic()


def _describe(motor: str, level: int, fault_bits: int) -> str:
    if level == LEVEL_FAULT and fault_bits != 0:
        return f"{motor} servo fault latched: {describe_faults(fault_bits)}"
    return {
        LEVEL_WARNING: f"{motor} running warm or loaded",
        LEVEL_CRITICAL: f"{motor} hot or overloaded",
        LEVEL_FAULT: f"{motor} torque unexpectedly disabled",
        LEVEL_NOT_REPORTING: f"{motor} not reporting",
    }[level]
