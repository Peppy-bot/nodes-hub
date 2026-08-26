"""Motor health policy: raw STS3215 readings to motor_health levels and
alert lifecycles. Pure functions and small state holders, no bus access."""

from __future__ import annotations

import math
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

# STS3215 judgment constants. The servo firmware self-protects at 70 C, so
# warn/crit sit below it; load fractions are of stall torque, and the EWMA
# window matches the servo's own 2 s overload-trip hold.
TEMP_WARN_C = 55.0
TEMP_CRIT_C = 65.0
LOAD_WARN_FRACTION = 0.5
LOAD_CRIT_FRACTION = 0.8
EWMA_TAU_S = 2.0

# The servo's own overload protection trips at this fraction of stall torque
# (held 2 s), the contract's "effective peak": 1.0 on the wire's
# effort_fraction_peak is that trip point.
TRIP_FRACTION_OF_STALL = 0.8

# Alert severities (alert:v1): 0 clears, 1 warn, 2 critical, 3 fault.
_LEVEL_SEVERITY = {
    LEVEL_NOMINAL: 0,
    LEVEL_WARNING: 1,
    LEVEL_CRITICAL: 2,
    LEVEL_FAULT: 3,
    # A silent motor cannot prove it is healthy; treat as critical.
    LEVEL_NOT_REPORTING: 2,
}

# Status-register fault bits, per the Feetech protocol (scservo_sdk ERRBIT_*).
_FAULT_BIT_NAMES = (
    (1, "voltage"),
    (2, "angle sensor"),
    (4, "overheating"),
    (8, "overcurrent"),
    (32, "overload"),
)


def describe_faults(bits: int) -> str:
    """Human name(s) for a nonzero Status register value."""
    names = [name for bit, name in _FAULT_BIT_NAMES if bits & bit]
    return ", ".join(names) if names else f"code {bits}"


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
            prev + alpha * (now - prev) for prev, now in zip(self._values, loads)
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

    def level(temp: float, load_sustained: float, enabled: bool, faults: int) -> int:
        # A latched servo fault or a silently disabled motor cannot drive:
        # fault outranks the thermal and load judgments, which presume a
        # live actuator. Overload protection cuts output while leaving
        # Torque_Enable at 1, so the Status bits are checked first.
        if faults != 0 or not enabled:
            return LEVEL_FAULT
        if temp >= TEMP_CRIT_C or load_sustained >= LOAD_CRIT_FRACTION:
            return LEVEL_CRITICAL
        if temp >= TEMP_WARN_C or load_sustained >= LOAD_WARN_FRACTION:
            return LEVEL_WARNING
        return LEVEL_NOMINAL

    return HealthReport(
        levels=tuple(
            level(t, s, e, f)
            for t, s, e, f in zip(
                temps_c, sustained, torque_enabled, fault_bits, strict=True
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


class AlertTracker:
    """Motor alert lifecycle: one (source, kind) identity per motor, a message
    on every level change including the severity-0 clear, and periodic
    re-emission of active alerts within the contract's 2000 ms bound."""

    def __init__(self, source_prefix: str):
        self._source_prefix = source_prefix
        self._conditions: dict[str, tuple[int, int]] = {}

    def _alert(self, motor: str, level: int, fault_bits: int) -> Alert:
        return Alert(
            source=f"{self._source_prefix} {motor}",
            kind="motor_condition",
            severity=_LEVEL_SEVERITY[level],
            message=_describe(motor, level, fault_bits),
        )

    def transitions(self, report: HealthReport) -> list[Alert]:
        """Alerts for every motor whose condition changed since the last
        call, where a condition is the level plus its fault bits (a fault
        that changes cause re-alerts at the same level)."""
        changed = []
        for motor, level, bits in zip(MOTOR_NAMES, report.levels, report.fault_bits):
            if self._conditions.get(motor, (LEVEL_NOMINAL, 0)) != (level, bits):
                self._conditions[motor] = (level, bits)
                changed.append(self._alert(motor, level, bits))
        return changed

    def active(self) -> list[Alert]:
        """Every currently non-nominal alert, for periodic re-emission."""
        return [
            self._alert(motor, level, bits)
            for motor, (level, bits) in self._conditions.items()
            if level != LEVEL_NOMINAL
        ]


def _describe(motor: str, level: int, fault_bits: int) -> str:
    if level == LEVEL_FAULT and fault_bits != 0:
        return f"{motor} servo fault latched: {describe_faults(fault_bits)}"
    return {
        LEVEL_NOMINAL: f"{motor} nominal",
        LEVEL_WARNING: f"{motor} running warm or loaded",
        LEVEL_CRITICAL: f"{motor} hot or overloaded",
        LEVEL_FAULT: f"{motor} torque unexpectedly disabled",
        LEVEL_NOT_REPORTING: f"{motor} not reporting",
    }[level]
