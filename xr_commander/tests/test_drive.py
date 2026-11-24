"""The shell of the grip world: the driver steps the world at each event and
runs each step's effects, against a scripted daemon, scripted backbones and
scripted posture goals, each call recorded in the order it is made. The
world's own rules are `test_grip_world`'s."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.helpers import (
    AGREED,
    CHANGES_ENDED,
    DIRECTOR,
    DIRECTOR_LEASE,
    HALTED,
    HALTER,
    HEADSET,
    HOME_REACHED,
    POSE,
    REFUSED_GOAL,
    RETRY_S,
    ROBOT_FRAME,
    SETTINGS,
    FakeCalls,
    FakeGoals,
    FakeStops,
    ManualClock,
    NoWireSlot,
    Peer,
    ScriptedHeadset,
    fleet_view,
    limits_given,
    listed,
    postures_of,
    reported,
    scripted_driver,
    settle,
    states,
    unlisted_driver,
)
from xr_commander.base import BaseReports, NotedStops, base_tick
from xr_commander.bus import FLASH_S, Failed, Unanswered
from xr_commander.clutch import HandClutch
from xr_commander.devices import Hand, HandSample, Thumbstick
from xr_commander.drive import Driver
from xr_commander.fleet import (
    LEASE_NAME,
    DaemonLost,
    Following,
    Holder,
    Lease,
    Starting,
    Timing,
)
from xr_commander.grip_world import (
    GOAL_UNSENT,
    ROBOT_NOTE,
    Engaged,
    Pin,
    Press,
    PressInputs,
    Reading,
    Ready,
    Requesting,
    Shutdown,
    Slot,
    Tick,
    TickInputs,
)
from xr_commander.publish import MeasuredPoses, pose_tick
from xr_commander.robots import PairedRobots

TIDY = Holder("tidy_inst", "fixture-core")
CLOSED_LEASE = {"alpha": Lease(DIRECTOR, DIRECTOR_LEASE, open=False)}
MINE = {"alpha": Lease(HEADSET, LEASE_NAME, open=False)}
# A drive_states of a base that follows, with unit limits along x and y and
# of the turn.
FOLLOWING = states(limits=limits_given(x=(-1.0, 1.0), y=(-1.0, 1.0), turn=(-1.0, 1.0)))


@dataclass
class Driving:
    """A driver, the calls it makes in one log, its daemon, its backbones,
    its posture goals, its clock, and the listing at each refresh of the
    page."""

    driver: Driver
    log: list
    calls: FakeCalls
    stops: FakeStops
    goals: FakeGoals
    clock: ManualClock
    changes: list

    def squeeze(self, robot="alpha", hand=Hand.RIGHT, reading=Reading.SQUEEZE):
        """A tick of `hand`'s arm grip reading `reading`, with `robot`
        chosen and its arm pair pinned and ready."""
        pins = frozenset({Pin(Slot.ARM, Peer(f"{robot} {hand} arm"))})
        inputs = TickInputs({hand: reading}, robot, pins, pins, Ready(), Ready())
        return self.driver.step(Tick(hand, inputs))

    def press(self, robot="alpha"):
        """A press of the home posture button, no hand squeezing."""
        inputs = PressInputs(False, robot, postures_of(robot))
        return self.driver.step(Press("home", inputs))


def driving(*robots, **view) -> Driving:
    clock = ManualClock()
    log = []
    calls, stops, goals = FakeCalls(log), FakeStops(log), FakeGoals(log)
    changes = []
    driver = unlisted_driver(
        clock=clock,
        calls=calls,
        stops=stops,
        goals=goals,
        timing=Timing(retry_s=RETRY_S),
        on_change=lambda: changes.append(driver.world.fleet.view),
    )
    listed(driver, *robots, **view)
    return Driving(driver, log, calls, stops, goals, clock, changes)


# Calls, timers and notes


async def test_a_call_effect_runs_as_a_task_and_its_outcome_steps_the_world():
    r = driving("alpha")
    world = r.squeeze()
    # The step starts the call: its outcome comes as a later event.
    assert isinstance(world.grips.right.phase, Requesting)
    assert r.log == []
    await settle()
    assert r.log == [("take", "alpha"), ("stop", "alpha")]
    assert isinstance(r.driver.world.grips.right.phase, Engaged)


async def test_a_call_that_raises_steps_raised_with_its_text(capsys):
    r = driving("alpha")

    async def raising_take(robot):
        r.log.append(("take", robot))
        raise RuntimeError("x")

    r.calls.take = raising_take
    r.squeeze()
    await settle()
    # A take that raised may have reached the daemon: its lease is released.
    assert r.log == [("take", "alpha"), ("release", "alpha")]
    assert r.driver.note(Hand.RIGHT) == "squeeze refused: RuntimeError('x')"
    assert "TakeLease of alpha raised: RuntimeError('x')" in capsys.readouterr().out


async def test_a_stop_reaches_the_robots_stops_and_its_refusal_is_a_failed_call():
    r = driving("alpha")
    r.stops.refusal = "alpha did not answer"
    r.squeeze()
    await settle()
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("release", "alpha")]
    assert r.driver.note(Hand.RIGHT) == (
        "squeeze refused: stop failed: alpha did not answer"
    )


@pytest.mark.parametrize(
    ("handle", "line", "after"),
    [
        (SimpleNamespace(accepted=True), "moving to home...", []),
        (REFUSED_GOAL, "move to home refused: busy", [("release", "alpha")]),
        (
            None,
            f"move to home refused: {GOAL_UNSENT}",
            [("stop", "alpha"), ("release", "alpha")],
        ),
    ],
    ids=["accepted", "refused", "unsent"],
)
async def test_a_fired_goal_is_answered_accepted_failed_refused_or_unanswered_unsent(
    handle, line, after
):
    r = driving("alpha")
    r.goals.handle = handle
    r.press()
    await settle()
    assert r.log == [
        ("take", "alpha"),
        ("stop", "alpha"),
        ("check_move", "alpha_inst"),
        ("goal", "home"),
        *after,
    ]
    assert r.driver.posture_line() == line


async def test_a_retry_timer_steps_retry_due_after_retry_s():
    r = driving("alpha")
    r.calls.take_outcome = Unanswered()
    r.calls.release_outcome = Unanswered()
    r.squeeze()
    await settle()
    assert r.log == [("take", "alpha"), ("release", "alpha")]
    await r.clock.advance(RETRY_S / 2)
    assert r.log.count(("release", "alpha")) == 1
    await r.clock.advance(RETRY_S / 2)
    assert r.log.count(("release", "alpha")) == 2


async def test_a_note_flashes_on_its_row_for_three_seconds_and_is_logged(capsys):
    r = driving("alpha")
    r.calls.take_outcome = Failed("leased")
    r.squeeze()
    await settle()
    assert r.driver.note(Hand.RIGHT) == "squeeze refused: leased"
    assert "squeeze refused: leased" in capsys.readouterr().out
    await r.clock.advance(FLASH_S)
    assert r.driver.note(Hand.RIGHT) == ""


async def test_changed_refreshes_the_page_and_a_failing_refresh_is_logged_once(
    capsys,
):
    r = driving("alpha")
    r.changes.clear()
    r.driver.fleet_changed(fleet_view(["alpha", "bravo"]))
    assert [view.robots for view in r.changes] == [frozenset({"alpha", "bravo"})]

    def failing_refresh():
        raise RuntimeError("page gone")

    failing = unlisted_driver(clock=ManualClock(), on_change=failing_refresh)
    # The start, the first listing and the next each refresh the page.
    listed(failing, "alpha")
    failing.fleet_changed(fleet_view(["alpha", "bravo"]))
    out = capsys.readouterr().out
    assert out.count("the page's refresh failed: RuntimeError('page gone')") == 1


async def test_a_step_from_inside_an_effect_fails_loudly():
    r = driving("alpha")
    r.press()
    await settle()
    assert r.driver.world.robot("alpha").hold.goal is not None
    r.goals.cancel = lambda label, goal: r.driver.step(Shutdown())
    with pytest.raises(AssertionError, match="a step runs its effects before"):
        r.driver.fleet_changed(fleet_view(["alpha"], halted_by=HALTER))


async def test_no_event_steps_after_shutdown():
    r = driving("alpha")
    r.squeeze()
    r.driver.step(Shutdown())
    world = r.driver.world
    await settle()
    # The take answered after the shutdown, and its answer stepped nothing:
    # no release follows it.
    assert r.log == [("take", "alpha")]
    assert r.driver.world is world
    r.driver.fleet_changed(fleet_view(["alpha"], halted_by=HALTER))
    assert r.driver.world is world


# Posture goals


async def test_a_watched_goal_steps_its_result_or_its_loss(capsys):
    r = driving("alpha")
    r.press()
    await settle()
    r.goals.results.put_nowait(HOME_REACHED)
    await settle()
    assert r.driver.world.posture_moving is None
    assert r.driver.posture_line() == HOME_REACHED
    assert r.log[-1] == ("release", "alpha")
    r.press()
    await settle()
    r.goals.results.put_nowait(RuntimeError("gone"))
    await settle()
    assert "move to home: result unknown: RuntimeError('gone')" in (
        capsys.readouterr().out
    )
    assert r.driver.posture_line() == "move to home: result unknown"
    # The move that may run is stopped, and then its lease released.
    assert r.log[-2:] == [("stop", "alpha"), ("release", "alpha")]


async def test_a_failed_stop_of_a_lost_move_is_noted_and_logged(capsys):
    r = driving("alpha")
    r.press()
    await settle()
    r.stops.refusal = "alpha did not answer"
    r.goals.results.put_nowait(RuntimeError("gone"))
    await settle()
    note = "move to home: its stop failed: alpha did not answer"
    assert r.driver.posture_line() == note
    assert note in capsys.readouterr().out


async def test_a_cancelled_goal_is_cancelled_without_waiting():
    r = driving("alpha")
    r.press()
    await settle()
    r.driver.fleet_changed(fleet_view(["alpha"], halted_by=HALTER))
    # The cancel goes out in the step itself, with no await.
    assert r.log[-1] == ("cancel", "home")


async def test_the_posture_line_gives_the_note_then_the_move_in_flight():
    r = driving("alpha")
    assert r.driver.posture_line() is None
    r.press()
    assert r.driver.posture_line() == "moving to home..."
    r.press()
    assert r.driver.posture_line() == "move to home ignored: still moving to home"
    await r.clock.advance(FLASH_S)
    assert r.driver.posture_line() == "moving to home..."


# The streams' ticks


class RaisingClutch:
    """A clutch whose step raises, as a fault after the tick's step."""

    def step(self, **_):
        raise RuntimeError("clutch")


async def test_a_raise_after_a_step_still_runs_its_call():
    clock = ManualClock()
    log = []
    driver = scripted_driver("alpha", clock=clock, log=log)
    pair = Peer("alpha right arm")
    arm, no_gripper = NoWireSlot(("alpha", pair)), NoWireSlot()
    robots = PairedRobots(
        None,
        arms={Hand.RIGHT: arm},
        grippers={Hand.RIGHT: no_gripper},
        base=NoWireSlot(),
    )
    session = ScriptedHeadset(clock)
    session.press(right=(False, False, True))
    measured = MeasuredPoses(clock)
    measured.set(pair, reported(POSE))
    tick = pose_tick(
        topic_module=arm,
        gripper_module=no_gripper,
        handedness=Hand.RIGHT,
        driver=driver,
        clutch=RaisingClutch(),
        session=session,
        measured=measured,
        robots=robots,
        settings=SETTINGS,
        clock=clock,
        timestamp_s=lambda: 0.0,
    )
    with pytest.raises(RuntimeError, match="clutch"):
        tick()
    await settle()
    assert log == [("take", "alpha"), ("stop", "alpha")]


async def test_a_slow_call_holds_no_tick():
    clock = ManualClock()
    log = []
    instance_ns = {"now": 0}
    reports = BaseReports(lambda: instance_ns["now"])
    stops = NotedStops(FakeStops(log), reports)
    calls = FakeCalls(log)
    driver = scripted_driver(
        "alpha", "bravo", clock=clock, log=log, calls=calls, stops=stops
    )
    alpha_left, bravo_right = Peer("alpha left arm"), Peer("bravo right arm")
    alpha_base = Peer("alpha base")
    left_arm = NoWireSlot(("alpha", alpha_left))
    right_arm = NoWireSlot(("bravo", bravo_right))
    base_slot = NoWireSlot(("alpha", alpha_base))
    no_gripper = NoWireSlot()
    robots = PairedRobots(
        None,
        arms={Hand.LEFT: left_arm, Hand.RIGHT: right_arm},
        grippers={Hand.LEFT: no_gripper, Hand.RIGHT: no_gripper},
        base=base_slot,
    )
    session = ScriptedHeadset(clock)
    measured = MeasuredPoses(clock)
    measured.set(alpha_left, reported(POSE))
    measured.set(bravo_right, reported(POSE))

    def arm_tick(hand, slot):
        return pose_tick(
            topic_module=slot,
            gripper_module=no_gripper,
            handedness=hand,
            driver=driver,
            clutch=HandClutch(1.0),
            session=session,
            measured=measured,
            robots=robots,
            settings=SETTINGS,
            clock=clock,
            timestamp_s=lambda: 0.0,
        )

    left, right = arm_tick(Hand.LEFT, left_arm), arm_tick(Hand.RIGHT, right_arm)
    base = base_tick(
        topic_module=base_slot,
        driver=driver,
        stops=stops,
        reports=reports,
        session=session,
        robots=robots,
        settings=SETTINGS,
        clock=clock,
    )
    driving_left = HandSample(
        pose=POSE, squeezing=True, trigger=0.0, thumbstick=Thumbstick(forward=1.0)
    )
    squeezing_right = HandSample(pose=POSE, squeezing=True, trigger=0.0)
    # The left squeeze takes alpha for its arm and the base.
    session.hands = {Hand.LEFT: driving_left}
    reports.record(alpha_base, FOLLOWING)
    left()
    base()
    await settle()
    # The base follows once a report arrives more than 0.5 s after the stop.
    instance_ns["now"] = 600_000_000
    reports.record(alpha_base, FOLLOWING)
    left()
    assert base() == (alpha_base, (None, None, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    # Bravo chosen: the left arm and the base linger on alpha, and the right
    # squeeze asks for bravo, whose take does not answer yet.
    takes = calls.takes_on_cue()
    assert robots.choose("bravo")
    measured_pose = (ROBOT_FRAME, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    still = (None, None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert left() == (alpha_left, measured_pose)
    assert base() == (alpha_base, still)
    session.hands = {Hand.LEFT: driving_left, Hand.RIGHT: squeezing_right}
    assert right() is None
    await settle()
    assert takes.unanswered() == [0]
    # Every tick steps both bursts while the take waits.
    for _ in range(3):
        assert left() == (alpha_left, measured_pose)
        assert base() == (alpha_base, still)
        assert right() is None
    takes.answer(0, AGREED)
    await settle()
    assert isinstance(driver.world.grips.right.phase, Engaged)


# The listener and the status words


async def test_a_status_write_that_changes_nothing_steps_nothing():
    r = driving("alpha")
    world = r.driver.world
    r.driver.goals_status(Following())
    assert r.driver.world is world
    r.driver.goals_ended(CHANGES_ENDED)
    r.driver.goals_status(Starting())
    r.driver.goals_started(2)
    world = r.driver.world
    # The status is starting already at the start after an end.
    r.driver.goals_status(Starting())
    assert r.driver.world is world
    # After a lost daemon the start's write changes it.
    r.driver.goals_status(DaemonLost())
    r.driver.goals_started(3)
    r.driver.goals_status(Starting())
    assert r.driver.world.fleet.status == Starting()
    assert r.driver.world.fleet.generation == 3


async def test_the_fleet_listener_steps_each_event():
    log = []
    driver = unlisted_driver(clock=ManualClock(), log=log)
    driver.goals_started(1)
    assert driver.world.fleet.generation == 1
    driver.goals_status(Following())
    assert driver.world.fleet.goals_run
    driver.fleet_changed(fleet_view(["alpha"]))
    assert driver.world.fleet.view.robots == frozenset({"alpha"})
    pins = frozenset({Pin(Slot.ARM, Peer("alpha right arm"))})
    inputs = TickInputs(
        {Hand.RIGHT: Reading.SQUEEZE}, "alpha", pins, pins, Ready(), Ready()
    )
    driver.step(Tick(Hand.RIGHT, inputs))
    await settle()
    driver.own_lease_ended("alpha", "")
    await settle()
    # The daemon's yes settles against the listing, which names no holder of
    # alpha: the headset lets go.
    assert log[-1] == ("check_move", "alpha")
    assert driver.note(ROBOT_NOTE) == "let go of alpha: the lease ended"
    driver.goals_ended("the lease goal of the fleet ended")
    assert driver.world.fleet.view is None


async def test_the_robot_row_says_who_holds_the_chosen_robot():
    r = driving("alpha", leases=CLOSED_LEASE)
    line = r.driver.robot_line("alpha")
    assert (line.text, line.tone) == (
        f"alpha: leased by director_inst ({DIRECTOR_LEASE})",
        "waiting",
    )
    r.driver.fleet_changed(fleet_view(["alpha"], leases=MINE))
    line = r.driver.robot_line("alpha")
    assert (line.text, line.tone) == ("alpha: driven from this headset", "driving")
    r.driver.fleet_changed(fleet_view(["alpha"]))
    assert r.driver.robot_line("alpha").tone == "resting"
    r.driver.fleet_changed(fleet_view(["alpha"], halted_by=HALTER))
    line = r.driver.robot_line("alpha")
    assert (line.text, line.tone) == (f"alpha: {HALTED}", "waiting")
    assert r.driver.robot_line(None).text == "no robot paired"
    r.driver.goals_ended("the daemon is lost")
    r.driver.goals_status(DaemonLost())
    assert r.driver.robot_line("alpha").text == "the daemon is lost"


async def test_a_robot_reads_as_the_cause_while_the_fleet_cannot_be_asked():
    r = driving("alpha")
    r.driver.goals_ended(CHANGES_ENDED)
    r.driver.goals_status(Starting())
    assert r.driver.state_of("alpha") == "waiting for the fleet listing"


async def test_the_page_hears_each_change_a_transfer_and_a_give_back_included():
    r = driving("alpha", leases=CLOSED_LEASE)
    r.changes.clear()
    # The director transfers the lease to a run, which gives it back.
    for view in (
        fleet_view(["alpha"], leases={"alpha": Lease(TIDY, DIRECTOR_LEASE, False)}),
        fleet_view(["alpha"], leases=CLOSED_LEASE),
    ):
        r.driver.fleet_changed(view)
    assert [view.leases["alpha"].holder for view in r.changes] == [TIDY, DIRECTOR]
    assert r.driver.state_of("alpha") == f"leased by director_inst ({DIRECTOR_LEASE})"
