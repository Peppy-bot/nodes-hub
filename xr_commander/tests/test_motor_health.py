import asyncio
import math
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.helpers import ClosingSubscription, FakeToken, FakeTopic
from xr_commander.motor_health import (
    HEALTH_STALE_AFTER_MS,
    NOT_REPORTING,
    WARNING,
    HealthRow,
    MotorHealthReports,
    drain_motor_health,
    health_rows,
)

STALE_AFTER_S = HEALTH_STALE_AFTER_MS / 1000.0

ARM_LEVELS = (0, 0, 0, 0, 0, 0, 0)


def report(
    store,
    instance="left_arm",
    levels=ARM_LEVELS,
    **readings,
):
    """Store one report; readings default to a full sensed set."""
    count = len(levels)
    values = {
        "effort_fraction_rated": (0.10,) * count,
        "effort_fraction_rated_sustained": (0.09,) * count,
        "effort_fraction_peak": (0.03,) * count,
        "driver_temp_c": (33.0,) * count,
        "winding_temp_c": (31.0,) * count,
    } | readings
    store.update(
        instance,
        levels,
        values["effort_fraction_rated"],
        values["effort_fraction_rated_sustained"],
        values["effort_fraction_peak"],
        values["driver_temp_c"],
        values["winding_temp_c"],
    )


def _wire(levels=ARM_LEVELS):
    count = len(levels)
    return SimpleNamespace(
        level=bytes(levels),
        effort_fraction_rated=[0.1] * count,
        effort_fraction_rated_sustained=[0.09] * count,
        effort_fraction_peak=[0.03] * count,
        driver_temp_c=[33.0] * count,
        winding_temp_c=[31.0] * count,
    )


def test_a_report_without_an_instance_is_refused():
    # A row for an unnamed component tells the operator nothing actionable.
    with pytest.raises(ValueError):
        report(MotorHealthReports(), instance="")


def test_a_report_naming_no_motors_is_refused():
    with pytest.raises(ValueError):
        report(MotorHealthReports(), levels=())


def test_an_undefined_level_is_refused():
    # The contract tops out at 4; rendering more would invent a state word.
    with pytest.raises(ValueError):
        report(MotorHealthReports(), levels=(0, 5, 0, 0, 0, 0, 0))


def test_a_mismatched_reading_vector_is_refused():
    # Six readings against seven motors has no defensible alignment; guessing
    # one would pin a warning's numbers on the wrong joint.
    with pytest.raises(ValueError, match="effort_fraction_rated "):
        report(MotorHealthReports(), effort_fraction_rated=(0.1,) * 6)
    with pytest.raises(ValueError, match="winding_temp_c"):
        report(MotorHealthReports(), winding_temp_c=(31.0,) * 8)


def test_a_non_finite_reading_is_refused():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            report(
                MotorHealthReports(),
                driver_temp_c=(33.0,) * 6 + (bad,),
            )


def test_wire_levels_arrive_as_bytes_and_are_accepted():
    # The generated module deserializes the u8 vector as bytes.
    store = MotorHealthReports()
    store.update("left_arm", bytes(ARM_LEVELS), (), (), (), (), ())
    assert len(store.by_instance()) == 1


def test_an_empty_reading_vector_means_not_sensed():
    store = MotorHealthReports()
    report(store, effort_fraction_rated=(), effort_fraction_rated_sustained=())
    (entry,) = store.by_instance()
    assert entry.report.effort_fraction_rated == ()
    assert entry.report.effort_fraction_peak == (0.03,) * 7


def test_an_instance_never_received_renders_nothing():
    assert health_rows(MotorHealthReports().by_instance()) == ()


def test_known_instances_keep_the_panel_order():
    store = MotorHealthReports()
    report(store, instance="right_gripper", levels=(0,))
    report(store, instance="left_arm")
    assert [entry.instance for entry in store.by_instance()] == [
        "left_arm",
        "right_gripper",
    ]


def test_an_unanticipated_instance_is_listed_after_the_known_ones():
    store = MotorHealthReports()
    report(store, instance="torso", levels=(0, 0))
    report(store, instance="left_arm")
    assert [entry.instance for entry in store.by_instance()] == ["left_arm", "torso"]


def test_a_stale_instance_collapses_to_one_not_reporting_row(monkeypatch):
    # A quiet producer's motors are unknown, not healthy: seven current-
    # looking rows off a dead feed would be the panel lying at a glance.
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "xr_commander.motor_health.time.monotonic", lambda: clock["now"]
    )
    store = MotorHealthReports()
    report(store)
    clock["now"] += STALE_AFTER_S
    assert len(health_rows(store.by_instance())) == 1
    clock["now"] += 0.001
    rows = health_rows(store.by_instance())
    assert [row.text for row in rows] == ["LEFT ARM not reporting"]
    assert rows[0].level == NOT_REPORTING


def test_a_re_emit_refreshes_the_staleness_clock(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "xr_commander.motor_health.time.monotonic", lambda: clock["now"]
    )
    store = MotorHealthReports()
    report(store)
    clock["now"] += STALE_AFTER_S - 0.1
    report(store)  # a contract-cadence re-emit
    clock["now"] += STALE_AFTER_S - 0.1
    (entry,) = store.by_instance()
    assert entry.report is not None


def test_an_all_nominal_instance_collapses_to_one_line():
    # Nominal is the steady state; sixteen rows of healthy numbers would
    # bury the panel. The line carries the worst sustained fraction and
    # temperature so closeness to a threshold is still visible.
    store = MotorHealthReports()
    report(store)
    (row,) = health_rows(store.by_instance())
    assert row == HealthRow("LEFT ARM nominal sust 9% 33C", 0)


def test_a_troubled_instance_details_only_its_off_nominal_motors():
    store = MotorHealthReports()
    report(store, levels=(0, 1, 0, 0, 4, 0, 0))
    rows = health_rows(store.by_instance())
    assert rows[0] == HealthRow("LEFT ARM (5 nominal)", None)
    assert [row.text.split()[0] for row in rows[1:]] == ["j2", "j5"]


def test_a_fully_troubled_instance_has_a_bare_heading():
    store = MotorHealthReports()
    report(store, levels=(1,) * 7)
    rows = health_rows(store.by_instance())
    assert rows[0] == HealthRow("LEFT ARM", None)
    assert len(rows) == 8


def test_a_single_motor_instance_is_one_row_carrying_its_name():
    store = MotorHealthReports()
    report(store, instance="left_gripper", levels=(1,))
    (row,) = health_rows(store.by_instance())
    assert row.text.startswith("LEFT GRIPPER WARNING")
    assert row.level == WARNING


def test_a_sensed_motor_row_reads_compactly():
    store = MotorHealthReports()
    store.update(
        "left_arm",
        (0, 1, 0, 0, 0, 0, 0),
        (0.10, 0.96, 0.10, 0.10, 0.10, 0.10, 0.10),
        (0.09, 0.93, 0.09, 0.09, 0.09, 0.09, 0.09),
        (0.03, 0.48, 0.03, 0.03, 0.03, 0.03, 0.03),
        (33.0, 41.2, 33.0, 33.0, 33.0, 33.0, 33.0),
        (31.0, 38.4, 31.0, 31.0, 31.0, 31.0, 31.0),
    )
    rows = health_rows(store.by_instance())
    assert rows[0].text == "LEFT ARM (6 nominal)"
    assert rows[1].text == "j2 WARNING peak 48% rated 96% sust 93% 41C/38C"
    assert rows[1].level == WARNING


def test_an_unsensed_report_shows_state_words_only():
    store = MotorHealthReports()
    store.update("left_arm", (0, 4, 0, 0, 0, 0, 0), (), (), (), (), ())
    rows = health_rows(store.by_instance())
    assert rows[0].text == "LEFT ARM (6 nominal)"
    assert rows[1].text == "j2 NOT REPORTING"
    assert rows[1].level == NOT_REPORTING


def test_an_unsensed_all_nominal_report_is_the_bare_state_word():
    store = MotorHealthReports()
    store.update("left_arm", (0,) * 7, (), (), (), (), ())
    (row,) = health_rows(store.by_instance())
    assert row == HealthRow("LEFT ARM nominal", 0)


def test_a_partially_sensed_report_shows_only_what_is_sensed():
    store = MotorHealthReports()
    report(
        store,
        instance="left_gripper",
        levels=(0,),
        effort_fraction_rated=(),
        effort_fraction_rated_sustained=(),
        effort_fraction_peak=(),
        driver_temp_c=(41.0,),
        winding_temp_c=(),
    )
    (row,) = health_rows(store.by_instance())
    assert row.text == "LEFT GRIPPER nominal 41C"


def test_drain_motor_health_survives_a_slot_it_cannot_subscribe_to():
    # The alert drain already fails this way: loud, and without taking the
    # panel down with it.
    class Refuses:
        async def subscribe(self, _runner):
            raise RuntimeError("no such slot")

    store = MotorHealthReports()
    asyncio.run(drain_motor_health(object(), Refuses(), store, FakeToken()))
    assert store.by_instance() == ()


def test_drain_motor_health_keys_reports_by_the_producing_instance():
    stream = [
        (SimpleNamespace(instance_id="left_arm"), _wire()),
        (SimpleNamespace(instance_id="left_gripper"), _wire((0,))),
    ]
    store = MotorHealthReports()
    asyncio.run(
        drain_motor_health(
            object(), FakeTopic(ClosingSubscription(stream)), store, FakeToken()
        )
    )
    assert [entry.instance for entry in store.by_instance()] == [
        "left_arm",
        "left_gripper",
    ]


def test_a_malformed_producer_is_reported_once_not_every_report():
    # A shared latch is cleared by any other producer's good message, so one
    # bad arm would log twice a second forever on a two-arm stack.
    bad = SimpleNamespace(instance_id="left_arm")
    good = SimpleNamespace(instance_id="right_arm")
    stream = []
    for _ in range(5):
        stream.append((bad, _wire(levels=(9,) * 7)))
        stream.append((good, _wire()))

    logged: list[str] = []
    store = MotorHealthReports()
    with mock.patch("xr_commander.bus.log", side_effect=lambda m: logged.append(m)):
        asyncio.run(
            drain_motor_health(
                object(), FakeTopic(ClosingSubscription(stream)), store, FakeToken()
            )
        )
    unusable = [m for m in logged if "unusable" in m]
    assert len(unusable) == 1, f"logged {len(unusable)} times: {unusable}"
    assert [entry.instance for entry in store.by_instance()] == ["right_arm"]


def test_an_unwired_health_slot_is_distinguishable_from_a_quiet_one():
    unwired = MotorHealthReports(producers_bound=False)
    wired = MotorHealthReports(producers_bound=True)
    assert unwired.by_instance() == wired.by_instance() == ()
    assert not unwired.producers_bound
    assert wired.producers_bound
