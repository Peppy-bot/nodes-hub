"""Per-motor health telemetry, listed on the status panel.

The motor_health contract carries an evaluated severity per motor plus the
load and temperature readings behind it, one report per producing instance
(the instance is the component). The panel is a glanceable surface: an
all-nominal component compresses to one line and only off-nominal motors
get detail rows.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from xr_commander.bus import CancellationToken, Latch, log, messages

# Level encoding of the motor_health contract: nominal, then warning <
# critical < fault, and 4 for a motor whose feedback went quiet (its
# readings are last-known, not current).
NOMINAL = 0
WARNING = 1
CRITICAL = 2
FAULT = 3
NOT_REPORTING = 4

_LEVEL_WORDS = {
    NOMINAL: "nominal",
    WARNING: "WARNING",
    CRITICAL: "CRITICAL",
    FAULT: "FAULT",
    NOT_REPORTING: "NOT REPORTING",
}

# Reports age out this long after arrival: 3x the contract's 500 ms report
# cadence, so a report survives two dropped re-emits but not a dead
# producer. An aged-out report means the producer went quiet and its motors
# are in an unknown state, not a healthy one.
HEALTH_STALE_AFTER_MS = 1500

# The panel's fixed instance order, the openarm launcher family's limb
# instance ids. Instances outside it follow, sorted by name, so a producer
# this node did not anticipate is still listed.
_PANEL_INSTANCE_ORDER = (
    "left_arm_inst",
    "right_arm_inst",
    "left_grip_inst",
    "right_grip_inst",
)


def _display_name(instance: str) -> str:
    """The panel wording for an instance name: "left_grip_inst" reads
    LEFT GRIP, with any "_inst" launcher suffix dropped."""
    return instance.removesuffix("_inst").replace("_", " ").upper()


@dataclass(frozen=True)
class HealthReport:
    """One instance's parsed report: a level per motor, each reading vector
    empty (not sensed) or exactly level-length, and the arrival time."""

    instance: str
    levels: tuple[int, ...]
    effort_fraction_rated: tuple[float, ...]
    effort_fraction_rated_sustained: tuple[float, ...]
    effort_fraction_peak: tuple[float, ...]
    driver_temp_c: tuple[float, ...]
    winding_temp_c: tuple[float, ...]
    received_monotonic_s: float


@dataclass(frozen=True)
class InstanceHealth:
    """One instance as the panel reads it: its name, and the live report or
    None once its producer aged out (motors unknown, not healthy)."""

    instance: str
    report: HealthReport | None


def _levels(values: Iterable[int]) -> tuple[int, ...]:
    """The level vector as ints; ValueError on anything undefined."""
    try:
        levels = tuple(int(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError("levels must be integers") from None
    if not levels:
        raise ValueError("a report naming no motors describes nothing")
    for level in levels:
        if not NOMINAL <= level <= NOT_REPORTING:
            raise ValueError(f"undefined level {level}")
    return levels


def _readings(
    name: str, values: Iterable[float], motor_count: int
) -> tuple[float, ...]:
    """One reading vector as floats: empty (not sensed) or one finite value
    per motor; ValueError otherwise."""
    readings = tuple(float(value) for value in values)
    if readings and len(readings) != motor_count:
        raise ValueError(
            f"{name} carries {len(readings)} readings for {motor_count} motors"
        )
    for value in readings:
        if not math.isfinite(value):
            raise ValueError(f"non-finite {name} {value!r}")
    return readings


class MotorHealthReports:
    """Latest report per producing instance, loop-confined: the listener
    writes and the panel reads, with no await between a read and its purge.

    A malformed report is refused (ValueError) rather than rendered: a wrong
    vector length has no defensible motor alignment, and a non-finite
    reading is a producer bug, not a measurement. An instance stays known
    after its report ages out, so the panel can say its producer went quiet
    instead of dropping the rows as if the motors were fine.
    """

    def __init__(self, *, producers_bound: bool = True, monotonic=time.monotonic) -> None:
        self._producers_bound = producers_bound
        # Injectable so staleness tests can move time instead of sleeping.
        self._monotonic = monotonic
        self._by_instance: dict[str, HealthReport] = {}

    @property
    def producers_bound(self) -> bool:
        """Whether anything is wired to the motor_health slot.

        The slot is zero_or_more, so an unwired stack receives nothing and
        looks exactly like one whose producers all died. A surface that
        renders silence has to be able to say which of the two it is showing.
        """
        return self._producers_bound

    def update(
        self,
        instance: str,
        level: Iterable[int],
        effort_fraction_rated: Iterable[float],
        effort_fraction_rated_sustained: Iterable[float],
        effort_fraction_peak: Iterable[float],
        driver_temp_c: Iterable[float],
        winding_temp_c: Iterable[float],
    ) -> None:
        if not instance:
            raise ValueError("a health report needs a producing instance")
        levels = _levels(level)
        count = len(levels)
        self._by_instance[instance] = HealthReport(
            instance=instance,
            levels=levels,
            effort_fraction_rated=_readings(
                "effort_fraction_rated", effort_fraction_rated, count
            ),
            effort_fraction_rated_sustained=_readings(
                "effort_fraction_rated_sustained",
                effort_fraction_rated_sustained,
                count,
            ),
            effort_fraction_peak=_readings(
                "effort_fraction_peak", effort_fraction_peak, count
            ),
            driver_temp_c=_readings("driver_temp_c", driver_temp_c, count),
            winding_temp_c=_readings("winding_temp_c", winding_temp_c, count),
            received_monotonic_s=self._monotonic(),
        )

    def by_instance(self) -> tuple[InstanceHealth, ...]:
        """Every instance ever received, panel order first and the rest
        sorted; a report older than the staleness window reads as None.

        An instance never received is absent entirely: the launcher may not
        wire it, and rows for it would name motors this node knows nothing
        about.
        """
        now = self._monotonic()

        def entry(report: HealthReport) -> InstanceHealth:
            live = now - report.received_monotonic_s <= HEALTH_STALE_AFTER_MS / 1000.0
            return InstanceHealth(
                instance=report.instance, report=report if live else None
            )

        known = self._by_instance
        ordered = [name for name in _PANEL_INSTANCE_ORDER if name in known] + sorted(
            name for name in known if name not in _PANEL_INSTANCE_ORDER
        )
        return tuple(entry(known[name]) for name in ordered)


@dataclass(frozen=True)
class HealthRow:
    """One health line for the panel: its text, and the level that colours
    it (None on an instance-name heading, which carries no level of its
    own)."""

    text: str
    level: int | None


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _celsius(value: float) -> str:
    return f"{value:.0f}C"


def _motor_text(report: HealthReport, index: int) -> str:
    """One motor's state word plus whichever readings its producer senses."""
    parts = [_LEVEL_WORDS[report.levels[index]]]
    if report.effort_fraction_peak:
        parts.append(f"peak {_percent(report.effort_fraction_peak[index])}")
    if report.effort_fraction_rated:
        parts.append(f"rated {_percent(report.effort_fraction_rated[index])}")
    if report.effort_fraction_rated_sustained:
        parts.append(
            f"sust {_percent(report.effort_fraction_rated_sustained[index])}"
        )
    if report.driver_temp_c and report.winding_temp_c:
        parts.append(
            f"{_celsius(report.driver_temp_c[index])}"
            f"/{_celsius(report.winding_temp_c[index])}"
        )
    elif report.driver_temp_c:
        parts.append(f"drv {_celsius(report.driver_temp_c[index])}")
    elif report.winding_temp_c:
        parts.append(f"wind {_celsius(report.winding_temp_c[index])}")
    return " ".join(parts)


def _collapsed_text(name: str, report: HealthReport) -> str:
    """The one-line form of an all-nominal component: its name, the state
    word, and the component's highest sustained fraction and temperature
    among whatever it senses."""
    parts = [name, "nominal"]
    if report.effort_fraction_rated_sustained:
        parts.append(f"sust {_percent(max(report.effort_fraction_rated_sustained))}")
    temps = report.driver_temp_c + report.winding_temp_c
    if temps:
        parts.append(_celsius(max(temps)))
    return " ".join(parts)


def _instance_rows(entry: InstanceHealth) -> tuple[HealthRow, ...]:
    name = _display_name(entry.instance)
    if entry.report is None:
        # The producer went quiet: one row for the whole instance, since any
        # per-motor row would be describing motors nobody has heard from.
        return (HealthRow(f"{name} not reporting", NOT_REPORTING),)
    report = entry.report
    trouble = [i for i, level in enumerate(report.levels) if level != NOMINAL]
    if not trouble:
        # Nominal is the panel's steady state, so it compresses to one line
        # per component; the numbers say how close the worst motor runs.
        return (HealthRow(_collapsed_text(name, report), NOMINAL),)
    if len(report.levels) == 1:
        # A single-motor instance (a gripper) is one row carrying its name.
        return (HealthRow(f"{name} {_motor_text(report, 0)}", report.levels[0]),)
    # A troubled multi-motor instance is a heading (folding the count of
    # motors with nothing to say) over one detail row per off-nominal motor,
    # so what is wrong gets the space the healthy rows would have taken.
    calm = len(report.levels) - len(trouble)
    heading = f"{name} ({calm} nominal)" if calm else name
    return (HealthRow(heading, None),) + tuple(
        HealthRow(f"j{index + 1} {_motor_text(report, index)}", report.levels[index])
        for index in trouble
    )


def health_rows(instances: Sequence[InstanceHealth]) -> tuple[HealthRow, ...]:
    """The health section's rows for `instances`, kept in their given order."""
    return tuple(row for entry in instances for row in _instance_rows(entry))


async def drain_motor_health(
    node_runner,
    topic_module,
    reports: MotorHealthReports,
    token: CancellationToken,
) -> None:
    """Keep `reports` at every producer's newest health report."""
    try:
        subscription = await topic_module.subscribe(node_runner)
    except Exception as e:
        # Loud and fail-safe, matching the alert drain: no health rows
        # rather than a task that dies and takes its failure with it.
        log(f"motor health subscribe failed: {e!r}")
        return
    # Latched per producer: a malformed producer repeats every report, and a
    # shared latch would be cleared by any other producer's good message.
    unusable: dict[str, Latch] = {}
    async for producer, message in messages(subscription, token, "motor health"):
        latch = unusable.get(producer.instance_id)
        if latch is None:
            latch = unusable[producer.instance_id] = Latch()
        try:
            reports.update(
                producer.instance_id,
                message.level,
                message.effort_fraction_rated,
                message.effort_fraction_rated_sustained,
                message.effort_fraction_peak,
                message.driver_temp_c,
                message.winding_temp_c,
            )
            latch.clear()
        except Exception as e:
            latch.trip(f"health report unusable from {producer.instance_id}: {e!r}")
    log("motor health stream ended")
