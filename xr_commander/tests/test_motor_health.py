import asyncio
import math
from types import SimpleNamespace

import pytest
from peppygen.consumed_topics.motor_health import motor_health as motor_health_topic

from tests.helpers import FakeToken, StubTopic, boot, eventually, running_drain
from xr_commander.motor_health import (
    HEALTH_STALE_AFTER_MS,
    HealthRow,
    Level,
    MotorHealthReports,
    drain_motor_health,
    health_rows,
)

STALE_AFTER_S = HEALTH_STALE_AFTER_MS / 1000.0

ARM_LEVELS = (0, 0, 0, 0, 0, 0, 0)

# The core node every store-level report comes from.
CORE = "core"


def report(
    store,
    # The launcher family's id: display assertions downstream also pin that
    # the "_inst" suffix drops from the panel wording.
    instance="left_arm_inst",
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
        (CORE, instance),
        levels,
        values["effort_fraction_rated"],
        values["effort_fraction_rated_sustained"],
        values["effort_fraction_peak"],
        values["driver_temp_c"],
        values["winding_temp_c"],
    )


def _wire(levels=ARM_LEVELS):
    """One real wire report, as a producer would publish it."""
    count = len(levels)
    return motor_health_topic.Message(
        timestamp=0.0,
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
    store.update((CORE, "left_arm"), bytes(ARM_LEVELS), (), (), (), (), ())
    assert len(store.by_instance()) == 1


def test_an_empty_reading_vector_means_not_sensed():
    store = MotorHealthReports()
    report(store, effort_fraction_rated=(), effort_fraction_rated_sustained=())
    (entry,) = store.by_instance()
    assert entry.report.effort_fraction_rated == ()
    assert entry.report.effort_fraction_peak == (0.03,) * 7


def test_an_instance_never_received_renders_nothing():
    assert health_rows(MotorHealthReports().by_instance()) == ()


@pytest.mark.parametrize("prefix", ["", "alpha_", "right_"])
def test_instances_follow_configured_producer_order(prefix):
    store = MotorHealthReports(
        bound_now=lambda: (
            (CORE, f"{prefix}right_arm_inst"),
            (CORE, f"{prefix}left_gripper_inst"),
        )
    )
    report(store, instance=f"{prefix}left_gripper_inst", levels=(0,))
    report(store, instance=f"{prefix}right_arm_inst")
    assert [entry.instance for entry in store.by_instance()] == [
        f"{prefix}right_arm_inst",
        f"{prefix}left_gripper_inst",
    ]


def test_an_unanticipated_instance_is_listed_after_the_known_ones():
    store = MotorHealthReports()
    report(store, instance="torso", levels=(0, 0))
    report(store, instance="left_arm_inst")
    assert [entry.instance for entry in store.by_instance()] == [
        "left_arm_inst",
        "torso",
    ]


def test_a_stale_instance_collapses_to_one_not_reporting_row():
    # A quiet producer's motors are unknown, not healthy: seven current-
    # looking rows off a dead feed would be the panel lying at a glance.
    clock = {"now": 100.0}
    store = MotorHealthReports(monotonic=lambda: clock["now"])
    report(store)
    clock["now"] += STALE_AFTER_S
    assert len(health_rows(store.by_instance())) == 1
    clock["now"] += 0.001
    rows = health_rows(store.by_instance())
    assert [row.text for row in rows] == ["LEFT ARM not reporting"]
    assert rows[0].level == Level.NOT_REPORTING


def test_a_re_emit_refreshes_the_staleness_clock():
    clock = {"now": 100.0}
    store = MotorHealthReports(monotonic=lambda: clock["now"])
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
    assert row.level == Level.WARNING


def test_a_sensed_motor_row_reads_compactly():
    store = MotorHealthReports()
    store.update(
        (CORE, "left_arm"),
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
    assert rows[1].level == Level.WARNING


def test_an_unsensed_report_shows_state_words_only():
    store = MotorHealthReports()
    store.update((CORE, "left_arm"), (0, 4, 0, 0, 0, 0, 0), (), (), (), (), ())
    rows = health_rows(store.by_instance())
    assert rows[0].text == "LEFT ARM (6 nominal)"
    assert rows[1].text == "j2 NOT REPORTING"
    assert rows[1].level == Level.NOT_REPORTING


def test_an_unsensed_all_nominal_report_is_the_bare_state_word():
    store = MotorHealthReports()
    store.update((CORE, "left_arm"), (0,) * 7, (), (), (), (), ())
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


async def test_drain_motor_health_keys_reports_by_the_producing_instance():
    async with boot(motor_health_instances=2) as h:
        arm, gripper = h.mocks.deps.motor_health
        arm_id, gripper_id = [
            p.instance_id for p in motor_health_topic.bound_producers(h.node_runner)
        ]
        store = MotorHealthReports()
        async with running_drain(
            lambda token: drain_motor_health(
                h.node_runner, motor_health_topic, store, token
            )
        ):
            await arm.motor_health.publish(_wire())
            await gripper.motor_health.publish(_wire((0,)))
            await eventually(
                lambda: (
                    [entry.instance for entry in store.by_instance()]
                    == sorted([arm_id, gripper_id])
                ),
                message="both producers' reports",
            )
        # Each report landed under its own transport-authenticated instance.
        by_name = {e.instance: e.report for e in store.by_instance()}
        assert len(by_name[arm_id].levels) == 7
        assert len(by_name[gripper_id].levels) == 1


async def test_a_malformed_producer_is_reported_once_not_every_report(capsys):
    # A shared latch is cleared by any other producer's good message, so one
    # bad arm would log twice a second forever on a two-arm stack. The good
    # producer's report is deliberately interleaved between the bad one's,
    # forcing exactly the arrival order that would clear a shared latch.
    # The refusals go to the captured log, read incrementally.
    async with boot(motor_health_instances=2) as h:
        bad, good = h.mocks.deps.motor_health
        bad_id, good_id = [
            p.instance_id for p in motor_health_topic.bound_producers(h.node_runner)
        ]
        logged: list[str] = []
        store = MotorHealthReports()

        def unusable():
            logged.extend(
                line
                for line in capsys.readouterr().out.splitlines()
                if "unusable" in line
            )
            return logged

        async with running_drain(
            lambda token: drain_motor_health(
                h.node_runner, motor_health_topic, store, token
            )
        ):
            for _ in range(3):
                await bad.motor_health.publish(_wire(levels=(9,) * 7))
            await eventually(lambda: len(unusable()) == 1, message="the first refusal")
            await good.motor_health.publish(_wire())
            await eventually(
                lambda: len(store.by_instance()) == 1, message="the good report"
            )
            for _ in range(2):
                await bad.motor_health.publish(_wire(levels=(9,) * 7))
            # A valid report from the bad producer marks its earlier ones
            # processed (per-producer order holds).
            await bad.motor_health.publish(_wire())
            await eventually(
                lambda: len(store.by_instance()) == 2, message="the recovery"
            )
        assert len(unusable()) == 1, f"logged {len(unusable())} times: {unusable()}"
        # The malformed reports were refused, never rendered: only the valid
        # nominal report is stored for the bad producer.
        by_name = {e.instance: e.report for e in store.by_instance()}
        assert by_name[bad_id].levels == (0,) * 7
        assert by_name[good_id].levels == (0,) * 7


async def test_one_instance_id_on_two_core_nodes_is_two_producers(capsys):
    # Two stacks on one mesh may each run a `left_arm_inst`. Every harness
    # mock shares the harness's core node, so the producers are scripted.
    # core-a's instance sends a bad report, core-b's a good one, then core-a's
    # a bad one and a good one. The test checks that both producers' reports
    # are stored and that core-a's bad reports log one refusal.
    here = SimpleNamespace(core_node="core-a", instance_id="left_arm_inst")
    there = SimpleNamespace(core_node="core-b", instance_id="left_arm_inst")
    bad = _wire(levels=(9,) * 7)
    store = MotorHealthReports()
    async with running_drain(
        lambda token: drain_motor_health(
            None,
            StubTopic(
                [(here, bad), (there, _wire()), (here, bad), (here, _wire((0,)))]
            ),
            store,
            token,
        )
    ):
        await eventually(
            lambda: len(store.by_instance()) == 2, message="both producers' reports"
        )
    assert [len(entry.report.levels) for entry in store.by_instance()] == [1, 7]
    assert capsys.readouterr().out.count("unusable") == 1


def test_an_unwired_health_slot_is_distinguishable_from_a_quiet_one():
    unwired = MotorHealthReports(bound_now=lambda: ())
    wired = MotorHealthReports(bound_now=lambda: ((CORE, "motor_inst"),))
    assert unwired.by_instance() == wired.by_instance() == ()
    assert not unwired.producers_bound
    assert wired.producers_bound
