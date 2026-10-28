"""The lease follows the grip: the driver's holds against a scripted daemon
and scripted backbones, each call recorded in the order it is made."""

import asyncio
from dataclasses import dataclass, field

from tests.helpers import (
    DIRECTOR,
    HALTED,
    HEADSET,
    RETRY_S,
    FakeCalls,
    FakeStops,
    ManualClock,
    fleet_view,
    following_fleet,
    settle,
)
from xr_commander.bus import Answered, Failed, Unanswered
from xr_commander.drive import (
    GOALS_ENDED,
    GRIP_HELD,
    GRIP_JUST_RELEASED,
    POSTURE,
    Driver,
    Lost,
    Refusal,
    Withdrawn,
)
from xr_commander.fleet import (
    DAEMON_SILENT,
    LEASE_NAME,
    DaemonLost,
    Fleet,
    Holder,
    Lease,
    Starting,
    Timing,
)
from xr_commander.robots import JOIN_COMMAND

TIDY = Holder("tidy_inst", "fixture-core")
OPEN_LEASE = {"alpha": Lease(DIRECTOR, "tidy the table", open=True)}
CLOSED_LEASE = {"alpha": Lease(DIRECTOR, "tidy the table", open=False)}
MINE = {"alpha": Lease(HEADSET, LEASE_NAME, open=False)}
# The goal handle that a posture start returns.
GOAL = "goal"


def held() -> bool:
    return True


def released() -> bool:
    return False


@dataclass
class Driving:
    """A driver, the calls it makes in one log, its daemon, its backbones,
    its fleet, its clock, and the listings it gave the page at each
    change."""

    driver: Driver
    log: list
    calls: FakeCalls
    stops: FakeStops
    fleet: Fleet
    clock: ManualClock
    changes: list
    cancelled: list = field(default_factory=list)

    def squeeze(self, robot="alpha", hand="right", wanted=held):
        return self.driver.acquire(robot, hand, wanted)

    async def press(self, robot="alpha", start=None):
        """A posture press whose start returns `GOAL` and tells the driver
        how to cancel it, unless `start` says otherwise."""

        async def started():
            self.log.append(("goal", robot))
            self.driver.on_posture_cancel(robot, lambda: self.cancelled.append(robot))
            return GOAL

        return await self.driver.start_posture_move(robot, start or started)

    def engage(self, hand="right", robot="alpha"):
        """The hand engages on `robot`, as its pose stream does once the
        driver holds the robot for it."""
        state = self.driver.hands[hand]
        state.engaged, state.robot = True, robot

    def release_grip(self, hand="right"):
        """The hand's grip is released, as its pose stream sees it: its
        release burst starts."""
        self.driver.hands[hand].engaged = False

    def hold_stops(self) -> asyncio.Event:
        """From now on each stop answers only once the returned event is
        set."""
        answer = asyncio.Event()
        stop = self.stops.stop

        async def held_stop(robot):
            stopped = await stop(robot)
            await answer.wait()
            return stopped

        self.stops.stop = held_stop
        return answer

    async def settled(self):
        """Let every task that the driver spawned reach the daemon."""
        await self.clock.idle()

    def releases(self, robot="alpha") -> int:
        return self.log.count(("release", robot))


def driving(*robots, **view) -> Driving:
    clock = ManualClock()
    log = []
    calls = FakeCalls(log)
    stops = FakeStops(log)
    fleet = following_fleet(*robots, **view)
    changes = []
    driver = Driver(
        fleet,
        calls,
        stops,
        timing=Timing(retry_s=RETRY_S),
        on_change=lambda: changes.append(fleet.view),
        clock=clock,
    )
    return Driving(driver, log, calls, stops, fleet, clock, changes)


# A squeeze


async def test_a_squeeze_takes_the_lease_then_stops_the_moves_in_flight():
    r = driving("alpha")
    assert await r.squeeze() is None
    assert r.log == [("take", "alpha"), ("stop", "alpha")]
    assert r.driver.holds("alpha")
    assert len(r.changes) == 1


async def test_a_refused_take_stops_nothing_and_names_the_holder():
    r = driving("alpha")
    r.calls.take_outcome = Failed("robot `alpha` is leased by `director_inst`")
    assert await r.squeeze() == Refusal("robot `alpha` is leased by `director_inst`")
    await r.settled()
    assert r.log == [("take", "alpha")]
    assert not r.driver.holds("alpha")


async def test_a_take_the_daemon_does_not_answer_is_released():
    r = driving("alpha")
    r.calls.take_outcome = Unanswered()
    assert await r.squeeze() == Refusal(DAEMON_SILENT)
    await r.settled()
    # The daemon may have recorded the take: the release undoes it.
    assert r.log == [("take", "alpha"), ("release", "alpha")]


async def test_a_squeeze_released_during_the_take_stops_nothing_and_releases_the_lease():
    r = driving("alpha")
    assert await r.squeeze(wanted=released) == Withdrawn()
    await r.settled()
    assert r.log == [("take", "alpha"), ("release", "alpha")]
    assert not r.driver.holds("alpha")


async def test_a_squeeze_released_during_the_stop_releases_the_lease():
    r = driving("alpha")
    grip = [True]
    stop = r.stops.stop

    async def released_while_stopping(robot):
        grip[0] = False
        return await stop(robot)

    r.stops.stop = released_while_stopping
    assert await r.squeeze(wanted=lambda: grip[0]) == Withdrawn()
    await r.settled()
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("release", "alpha")]


async def test_the_lease_is_released_after_the_last_reason_only():
    r = driving("alpha")
    await r.squeeze(hand="right")
    r.engage("right")
    await r.squeeze(hand="left")
    # The second hand joins the hold: no take and no stop.
    assert r.log == [("take", "alpha"), ("stop", "alpha")]
    r.driver.drop("alpha", "right")
    await r.settled()
    assert r.releases() == 0
    r.driver.drop("alpha", "left")
    await r.settled()
    assert r.log[-1] == ("release", "alpha")
    assert not r.driver.holds("alpha")
    assert len(r.changes) == 2


async def test_a_squeeze_in_the_hands_own_burst_engages_with_no_take_and_no_stop():
    r = driving("alpha")
    await r.squeeze()
    # The hand's reason stays until its release burst ends: a squeeze in the
    # burst joins it.
    assert await r.squeeze() is None
    assert r.log == [("take", "alpha"), ("stop", "alpha")]


async def test_two_hands_that_squeeze_at_once_take_the_robot_once():
    r = driving("alpha")
    left, right = await asyncio.gather(r.squeeze(hand="left"), r.squeeze())
    assert (left, right) == (None, None)
    assert r.log == [("take", "alpha"), ("stop", "alpha")]


# A squeeze during a posture move


async def test_a_squeeze_during_a_posture_move_stops_it_and_keeps_the_lease():
    r = driving("alpha")
    assert await r.press() == GOAL
    assert await r.squeeze() is None
    # The squeeze stops the move in flight, the headset's own included, and
    # takes nothing: the posture move took the lease.
    assert r.log == [
        ("take", "alpha"),
        ("stop", "alpha"),
        ("goal", "alpha"),
        ("stop", "alpha"),
    ]
    # The end of the stopped move releases nothing: the hand holds it.
    r.driver.drop("alpha", POSTURE)
    await r.settled()
    assert r.releases() == 0
    r.driver.drop("alpha", "right")
    await r.settled()
    assert r.log[-1] == ("release", "alpha")


async def test_the_end_of_the_posture_move_a_squeeze_stops_releases_nothing():
    r = driving("alpha")
    await r.press()
    stop = r.stops.stop

    async def ends_the_posture_move(robot):
        # The backbone ends the posture goal before it answers the stop.
        r.driver.drop("alpha", POSTURE)
        return await stop(robot)

    r.stops.stop = ends_the_posture_move
    assert await r.squeeze() is None
    await r.settled()
    assert r.driver.holds("alpha") and r.releases() == 0


async def test_a_failed_stop_of_a_posture_move_refuses_the_squeeze_and_keeps_the_move():
    r = driving("alpha")
    await r.press()
    r.stops.refusal = "alpha did not answer"
    assert await r.squeeze() == Refusal("stop failed: alpha did not answer")
    assert not r.driver.holds_for("alpha", "right")
    # The move holds the robot to its end, then releases it.
    assert r.driver.holds_for("alpha", POSTURE)
    r.driver.drop("alpha", POSTURE)
    await r.settled()
    assert r.releases() == 1


async def test_a_squeeze_leaves_a_posture_move_on_another_robot_alone():
    r = driving("alpha", "bravo")
    await r.press("alpha")
    await r.squeeze("bravo")
    assert r.log == [
        ("take", "alpha"),
        ("stop", "alpha"),
        ("goal", "alpha"),
        ("take", "bravo"),
        ("stop", "bravo"),
    ]
    assert r.driver.holds("alpha") and r.cancelled == []


async def test_a_squeeze_during_a_posture_start_waits_and_stops_the_started_move():
    r = driving("alpha")
    starting = asyncio.Event()
    go_on = asyncio.Event()

    async def start():
        starting.set()
        await go_on.wait()
        r.log.append(("goal", "alpha"))
        return GOAL

    press = asyncio.create_task(r.press(start=start))
    await starting.wait()
    squeeze = asyncio.create_task(r.squeeze())
    # The squeeze waits for the start, which waits for the test.
    await settle()
    assert not squeeze.done() and r.log == [("take", "alpha"), ("stop", "alpha")]
    go_on.set()
    assert await press == GOAL and await squeeze is None
    # The squeeze stops the move once it runs.
    assert r.log[-2:] == [("goal", "alpha"), ("stop", "alpha")]


# A posture press


async def test_a_press_takes_the_robot_and_holds_it_for_its_move():
    r = driving("alpha")
    assert await r.press() == GOAL
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("goal", "alpha")]
    assert r.driver.holds_for("alpha", POSTURE)


async def test_a_posture_start_that_is_refused_releases_the_lease():
    r = driving("alpha")

    async def refused_start():
        return Refusal("a leader is streaming arm commands")

    assert await r.press(start=refused_start) == Refusal(
        "a leader is streaming arm commands"
    )
    await r.settled()
    assert not r.driver.holds("alpha") and r.releases() == 1


async def test_a_posture_start_that_a_let_go_overtakes_is_refused_and_its_move_cancelled():
    r = driving("alpha")

    async def start_during_a_halt():
        r.driver.fleet_changed(fleet_view(["alpha"], halted_by="peppy-cli"))
        r.driver.on_posture_cancel("alpha", lambda: r.cancelled.append("alpha"))
        return GOAL

    assert await r.press(start=start_during_a_halt) == Refusal(
        "the headset let go of alpha"
    )
    assert r.cancelled == ["alpha"]
    assert not r.driver.holds("alpha")


async def test_a_press_on_a_robot_a_hand_holds_is_refused():
    r = driving("alpha")
    await r.squeeze()
    r.engage("right")
    assert await r.press() == Refusal(GRIP_HELD)
    r.release_grip()
    assert await r.press() == Refusal(GRIP_JUST_RELEASED)
    assert r.log == [("take", "alpha"), ("stop", "alpha")]
    assert not r.driver.holds_for("alpha", POSTURE)


# The halt and the refusals


async def test_a_halt_refuses_every_squeeze_and_press_and_names_who_halted():
    r = driving("alpha", halted_by="peppy-cli")
    assert await r.squeeze() == Refusal(HALTED)
    assert await r.press() == Refusal(HALTED)
    assert r.log == []


async def test_a_halt_during_the_stop_refuses_the_squeeze_and_releases_the_lease():
    r = driving("alpha")
    stop = r.stops.stop

    async def halts_while_stopping(robot):
        answer = await stop(robot)
        r.fleet.view = fleet_view(["alpha"], halted_by="peppy-cli")
        r.driver.fleet_changed(r.fleet.view)
        return answer

    r.stops.stop = halts_while_stopping
    assert await r.squeeze() == Refusal(HALTED)
    await r.settled()
    assert not r.driver.holds("alpha") and r.releases() == 1


async def test_a_let_go_during_the_stop_of_a_posture_move_refuses_the_squeeze():
    r = driving("alpha")
    await r.press()
    stop = r.stops.stop

    async def halts_while_stopping(robot):
        r.driver.fleet_changed(fleet_view(["alpha"], halted_by="peppy-cli"))
        return await stop(robot)

    r.stops.stop = halts_while_stopping
    assert await r.squeeze() == Refusal("the headset let go of alpha")
    assert not r.driver.holds("alpha")


async def test_a_restart_of_the_goals_during_the_take_refuses_the_squeeze():
    r = driving("alpha")
    take = r.calls.take

    async def restarts_while_taking(robot):
        r.fleet.generation += 1
        return await take(robot)

    r.calls.take = restarts_while_taking
    assert await r.squeeze() == Refusal(GOALS_ENDED)
    assert not r.driver.holds("alpha")


async def test_the_end_of_the_changes_goal_during_the_take_refuses_the_squeeze():
    r = driving("alpha")
    take = r.calls.take

    async def ends_while_taking(robot):
        # The goals have not started again yet: the listing is gone.
        r.fleet.view = None
        r.fleet.status = Starting()
        return await take(robot)

    r.calls.take = ends_while_taking
    assert await r.squeeze() == Refusal(GOALS_ENDED)
    assert not r.driver.holds("alpha")


async def test_a_robot_outside_the_fleet_listing_is_not_driven_and_says_how_to_join():
    r = driving("alpha")
    assert await r.squeeze("backbone_inst") == Refusal(
        "backbone_inst is not in the fleet listing, which holds the copies of the"
        f" stack: join a robot as a copy with {JOIN_COMMAND}"
    )
    assert r.log == []


async def test_a_lost_daemon_makes_every_squeeze_and_press_do_nothing():
    r = driving("alpha")
    r.fleet.status = DaemonLost()
    assert await r.squeeze() == Refusal("the daemon is lost")
    assert await r.press() == Refusal("the daemon is lost")
    assert r.log == []


async def test_a_failed_stop_takes_nothing_and_releases_the_lease():
    r = driving("alpha")
    r.stops.refusal = "alpha did not answer"
    assert await r.squeeze() == Refusal("stop failed: alpha did not answer")
    await r.settled()
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("release", "alpha")]
    assert not r.driver.holds("alpha")


# Letting go


async def test_the_headset_lets_go_at_a_halt():
    r = driving("alpha")
    await r.press()
    await r.squeeze()
    r.engage("right")
    r.driver.fleet_changed(fleet_view(["alpha"], halted_by="peppy-cli"))
    # Let go as at a grip release: the hand disengages and waits for its
    # grip to be released, the posture move is cancelled.
    hand = r.driver.hands["right"]
    assert not hand.engaged and hand.wait_release
    assert r.cancelled == ["alpha"] and not r.driver.holds("alpha")
    assert r.driver.note.text() == f"let go of alpha: {HALTED}"
    assert r.driver.robot_line("alpha").text == f"let go of alpha: {HALTED}"
    # The halt ended the lease: nothing is released.
    await r.settled()
    assert r.releases() == 0


async def test_a_let_go_disengages_only_the_hands_on_that_robot():
    r = driving("alpha", "bravo")
    await r.squeeze("alpha", hand="right")
    r.engage("right", "alpha")
    await r.squeeze("bravo", hand="left")
    r.engage("left", "bravo")
    r.driver.let_go("alpha", "a test let go", release=True)
    assert not r.driver.hands["right"].engaged
    assert r.driver.hands["left"].engaged and not r.driver.hands["left"].wait_release
    assert r.driver.holds("bravo")


async def test_a_lease_end_the_daemon_confirms_lets_go():
    r = driving("alpha")
    await r.squeeze()
    r.calls.check_outcome = Failed("headset_inst holds no lease of robot `alpha`")
    r.driver.own_lease_ended("alpha", "")
    await r.settled()
    assert not r.driver.holds("alpha")
    assert r.driver.note.text() == "let go of alpha: the lease ended"
    # The lease ended: nothing is released.
    assert r.log[-1] == ("check_move", "alpha") and r.releases() == 0


async def test_the_end_of_its_own_release_lets_go_of_no_new_take():
    r = driving("alpha")
    await r.squeeze()
    r.driver.drop("alpha", "right")
    await r.settled()
    # The person squeezes again before the lease feedback gives the end that
    # the release caused: the daemon allows the move at a revision that the
    # listing has not reached yet.
    await r.squeeze()
    r.calls.check_outcome = Answered(2)
    r.driver.own_lease_ended("alpha", "headset_inst released it")
    await r.settled()
    assert r.driver.holds("alpha")
    assert r.log[-1] == ("check_move", "alpha")
    # The listing of that revision names the headset: it drives on.
    r.driver.fleet_changed(fleet_view(["alpha"], leases=MINE, revision=2))
    assert r.driver.holds("alpha")


async def test_a_yes_that_the_current_listing_settles_keeps_the_hold():
    r = driving("alpha")
    await r.squeeze()
    r.fleet.view = fleet_view(["alpha"], leases=MINE, revision=2)
    r.calls.check_outcome = Answered(2)
    r.driver.own_lease_ended("alpha", "headset_inst released it")
    await r.settled()
    assert r.driver.holds("alpha")


async def test_a_yes_that_the_current_listing_settles_against_the_headset_lets_go():
    r = driving("alpha")
    await r.squeeze()
    r.calls.check_outcome = Answered(3)
    r.driver.own_lease_ended("alpha", "the stack was reset")
    # The listing of the director's open lease comes before the daemon's yes.
    r.fleet.view = fleet_view(["alpha"], leases=OPEN_LEASE, revision=3)
    r.driver.fleet_changed(r.fleet.view)
    assert r.driver.holds("alpha")
    await r.settled()
    assert not r.driver.holds("alpha")
    assert r.driver.note.text() == "let go of alpha: the stack was reset"


async def test_the_earliest_yes_settles_two_lease_ends():
    r = driving("alpha")
    await r.squeeze()
    for revision in (5, 7):
        r.calls.check_outcome = Answered(revision)
        r.driver.own_lease_ended("alpha", f"the end read at {revision}")
        await r.settled()
    r.driver.fleet_changed(fleet_view(["alpha"], revision=5))
    assert not r.driver.holds("alpha")
    assert r.driver.note.text() == "let go of alpha: the end read at 5"


async def test_a_yes_under_another_clients_open_lease_lets_go_at_the_listing():
    r = driving("alpha")
    await r.squeeze()
    # A reset ends the lease, and the director takes and opens it before the
    # daemon answers: the yes is for the director's open lease.
    r.calls.check_outcome = Answered(3)
    r.driver.own_lease_ended("alpha", "the stack was reset")
    await r.settled()
    r.driver.fleet_changed(fleet_view(["alpha"], leases=OPEN_LEASE, revision=3))
    assert not r.driver.holds("alpha")
    assert r.driver.note.text() == "let go of alpha: the stack was reset"
    await r.settled()
    assert r.releases() == 0


async def test_a_lease_end_during_the_take_is_confirmed_once_the_take_is_done():
    r = driving("alpha")
    take = r.calls.take

    async def ends_while_taking(robot):
        answer = await take(robot)
        r.driver.own_lease_ended("alpha", "the stack was reset")
        return answer

    r.calls.take = ends_while_taking
    r.calls.check_outcome = Failed("headset_inst holds no lease of robot `alpha`")
    assert await r.squeeze() is None
    await r.settled()
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("check_move", "alpha")]
    assert not r.driver.holds("alpha")


async def test_a_lease_end_the_daemon_does_not_confirm_lets_go_and_releases():
    r = driving("alpha")
    await r.squeeze()
    r.calls.check_outcome = Unanswered()
    r.driver.own_lease_ended("alpha", "the stack was reset")
    await r.settled()
    assert not r.driver.holds("alpha")
    assert r.driver.note.text() == f"let go of alpha: {DAEMON_SILENT}"
    # The daemon may still hold the lease for the headset.
    assert r.log[-2:] == [("check_move", "alpha"), ("release", "alpha")]


async def test_a_lease_end_of_a_robot_the_headset_does_not_hold_asks_nothing():
    r = driving("alpha")
    r.driver.own_lease_ended("alpha", "headset_inst released it")
    await r.settled()
    assert r.log == []


async def test_a_lease_end_of_the_goals_before_a_restart_asks_nothing():
    r = driving("alpha")
    await r.squeeze()
    r.driver.own_lease_ended("alpha", "the lease goal ended")
    r.fleet.generation += 1
    await r.settled()
    assert r.driver.holds("alpha") and r.log == [("take", "alpha"), ("stop", "alpha")]


async def test_the_end_of_a_goal_of_the_fleet_lets_go_of_every_robot():
    r = driving("alpha", "bravo")
    await r.squeeze("alpha")
    await r.press("bravo")
    r.driver.let_go_of_every_robot("the changes goal of the fleet ended")
    assert not r.driver.holds("alpha") and not r.driver.holds("bravo")
    assert (
        r.driver.note.text() == "let go of bravo: the changes goal of the fleet ended"
    )
    # The leases end with the lease goal: nothing is asked of the daemon.
    await r.settled()
    assert r.releases("alpha") == 0 and r.releases("bravo") == 0


# Take-over under an open lease


async def test_under_an_open_lease_the_headset_drives_with_a_move_check_and_no_take():
    r = driving("alpha", leases=OPEN_LEASE)
    assert await r.squeeze() is None
    assert r.log == [("check_move", "alpha"), ("stop", "alpha")]
    assert r.driver.driven_under_open("alpha")
    assert (
        r.driver.state_of("alpha")
        == "driven from this headset under the open lease of director_inst"
    )
    # The lease is the run's: the end of the drive releases nothing.
    r.driver.drop("alpha", "right")
    await r.settled()
    assert r.releases() == 0


async def test_under_an_open_lease_a_squeeze_in_a_burst_asks_again_and_stops():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze("alpha", hand="right")
    # The right hand's burst runs, and a program moves the other arm under
    # the open lease: the left squeeze asks the daemon and stops it.
    assert await r.squeeze("alpha", hand="left") is None
    assert r.log == [("check_move", "alpha"), ("stop", "alpha")] * 2


async def test_under_an_open_lease_a_squeeze_beside_the_other_hand_engages_at_once():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze("alpha", hand="right")
    r.engage("right")
    assert await r.squeeze("alpha", hand="left") is None
    assert r.log == [("check_move", "alpha"), ("stop", "alpha")]


async def test_a_withdrawn_squeeze_in_its_own_burst_leaves_the_burst_its_hold():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze()
    assert await r.squeeze(wanted=released) == Withdrawn()
    assert r.driver.holds_for("alpha", "right")
    # It asked the daemon again, and stopped nothing.
    assert r.log == [
        ("check_move", "alpha"),
        ("stop", "alpha"),
        ("check_move", "alpha"),
    ]


async def test_a_squeeze_withdrawn_behind_a_posture_start_stops_nothing():
    r = driving("alpha")
    starting = asyncio.Event()
    go_on = asyncio.Event()
    grip = [True]

    async def start():
        starting.set()
        await go_on.wait()
        r.log.append(("goal", "alpha"))
        return GOAL

    press = asyncio.create_task(r.press(start=start))
    await starting.wait()
    squeeze = asyncio.create_task(r.squeeze(wanted=lambda: grip[0]))
    await settle()
    grip[0] = False
    go_on.set()
    assert await press == GOAL and await squeeze == Withdrawn()
    assert r.log[-1] == ("goal", "alpha")
    assert r.driver.holds_for("alpha", POSTURE)


async def test_an_open_lease_beside_a_posture_move_of_another_robot_takes_nothing():
    r = driving("alpha", "bravo", leases=OPEN_LEASE)
    await r.press("bravo")
    assert await r.squeeze("alpha") is None
    assert ("take", "alpha") not in r.log
    assert r.driver.holds("bravo")


async def test_the_headset_lets_go_when_the_open_lease_closes_and_drives_again_when_it_reopens():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze()
    r.driver.fleet_changed(fleet_view(["alpha"], leases=CLOSED_LEASE))
    assert not r.driver.holds("alpha")
    assert (
        r.driver.note.text()
        == "let go of alpha: the open lease of director_inst closed"
    )
    # A refused resume opens the lease again: the next squeeze drives.
    r.fleet.view = fleet_view(["alpha"], leases=OPEN_LEASE)
    assert await r.squeeze() is None
    assert r.log.count(("check_move", "alpha")) == 2
    # The lease was the run's: nothing is released.
    await r.settled()
    assert r.releases() == 0


async def test_the_headset_lets_go_when_the_open_lease_ends_or_passes_to_another_holder():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze()
    r.driver.fleet_changed(fleet_view(["alpha"]))
    assert not r.driver.holds("alpha")
    r.fleet.view = fleet_view(["alpha"], leases=OPEN_LEASE)
    await r.squeeze()
    tidy_open = {"alpha": Lease(TIDY, "tidy the table", open=True)}
    r.driver.fleet_changed(fleet_view(["alpha"], leases=tidy_open))
    assert not r.driver.holds("alpha")


async def test_under_an_open_lease_a_refused_move_check_takes_nothing_and_says_why():
    r = driving("alpha", leases=OPEN_LEASE)
    r.calls.check_outcome = Failed("the stack is halted (by peppy-cli)")
    assert await r.squeeze() == Refusal("the stack is halted (by peppy-cli)")
    assert r.log == [("check_move", "alpha")]
    assert not r.driver.holds("alpha")


async def test_under_an_open_lease_a_refused_check_in_a_burst_stops_nothing():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze("alpha", hand="right")
    r.calls.check_outcome = Failed("the stack is halted (by peppy-cli)")
    assert await r.squeeze("alpha", hand="left") == Refusal(
        "the stack is halted (by peppy-cli)"
    )
    assert r.log == [
        ("check_move", "alpha"),
        ("stop", "alpha"),
        ("check_move", "alpha"),
    ]
    assert r.driver.holds_for("alpha", "right")
    assert not r.driver.holds_for("alpha", "left")


async def test_under_an_open_lease_a_close_during_the_check_of_a_burst_stops_nothing():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.squeeze("alpha", hand="right")
    check = r.calls.check_move

    async def closes_while_checking(**target):
        answer = await check(**target)
        r.fleet.view = fleet_view(["alpha"], leases=CLOSED_LEASE)
        r.driver.fleet_changed(r.fleet.view)
        return answer

    r.calls.check_move = closes_while_checking
    assert await r.squeeze("alpha", hand="left") == Refusal(
        "the headset let go of alpha"
    )
    assert r.log == [
        ("check_move", "alpha"),
        ("stop", "alpha"),
        ("check_move", "alpha"),
    ]


async def test_a_squeeze_on_a_robot_another_client_holds_asks_to_take_it():
    r = driving("alpha", leases=CLOSED_LEASE)
    r.calls.take_outcome = Failed("robot `alpha` is leased by `director_inst`")
    assert await r.squeeze() == Refusal("robot `alpha` is leased by `director_inst`")
    assert r.log == [("take", "alpha")]


async def test_an_open_lease_that_closes_before_the_stop_sends_no_stop_and_releases_nothing():
    r = driving("alpha", leases=OPEN_LEASE)
    check = r.calls.check_move

    async def closes_while_checking(**target):
        answer = await check(**target)
        r.fleet.view = fleet_view(["alpha"], leases=CLOSED_LEASE)
        return answer

    r.calls.check_move = closes_while_checking
    assert await r.squeeze() == Refusal("the open lease of director_inst closed")
    await r.settled()
    assert r.log == [("check_move", "alpha")]
    assert not r.driver.holds("alpha")


# Lost posture moves


async def test_a_posture_start_whose_send_is_lost_stops_the_moves_before_the_release():
    r = driving("alpha")
    answer = None

    async def lost():
        nonlocal answer
        answer = r.hold_stops()
        return Lost("the goal could not be sent (see log)")

    press = asyncio.create_task(r.press(start=lost))
    await settle()
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("stop", "alpha")]
    answer.set()
    assert await press == Refusal("the goal could not be sent (see log)")
    await r.settled()
    assert r.log[-1] == ("release", "alpha")


def halt(r: Driving) -> None:
    r.fleet.view = fleet_view(["alpha"], halted_by="peppy-cli")
    r.driver.fleet_changed(r.fleet.view)


async def test_a_posture_start_lost_at_a_halt_is_stopped():
    r = driving("alpha")

    async def lost_during_a_halt():
        halt(r)
        return Lost("the goal could not be sent (see log)")

    assert await r.press(start=lost_during_a_halt) == Refusal(
        "the goal could not be sent (see log)"
    )
    # No goal handle cancels the move that the backbone may run: the stop
    # does.
    assert r.log == [("take", "alpha"), ("stop", "alpha"), ("stop", "alpha")]


async def test_a_posture_move_whose_result_is_lost_is_stopped_before_the_release():
    r = driving("alpha")
    await r.press()
    answer = r.hold_stops()
    ending = asyncio.create_task(r.driver.end_lost_posture_move("alpha"))
    await settle()
    assert r.log[-1] == ("stop", "alpha")
    answer.set()
    await ending
    await r.settled()
    assert r.log[-2:] == [("stop", "alpha"), ("release", "alpha")]
    assert not r.driver.holds("alpha")


async def test_a_posture_move_lost_at_a_halt_is_stopped():
    r = driving("alpha")
    await r.press()
    halt(r)
    await r.driver.end_lost_posture_move("alpha")
    assert r.log[-2:] == [("goal", "alpha"), ("stop", "alpha")]


async def test_a_posture_move_lost_after_its_open_lease_closed_stops_nothing():
    r = driving("alpha", leases=OPEN_LEASE)
    await r.press()
    # The run resumes and closes its lease: its own goals now move the robot.
    r.fleet.view = fleet_view(["alpha"], leases=CLOSED_LEASE)
    r.driver.fleet_changed(r.fleet.view)
    await r.driver.end_lost_posture_move("alpha")
    assert r.log == [("check_move", "alpha"), ("stop", "alpha"), ("goal", "alpha")]


# Releases


async def test_a_release_the_daemon_does_not_answer_is_asked_again_on_the_clock():
    r = driving("alpha")
    await r.squeeze()
    r.calls.release_outcome = Unanswered()
    r.driver.drop("alpha", "right")
    await r.settled()
    assert r.releases() == 1
    r.calls.release_outcome = Failed("headset_inst holds no lease of robot `alpha`")
    await r.clock.advance(RETRY_S)
    await r.settled()
    assert r.releases() == 2
    # A refused release is not asked again.
    await r.clock.advance(RETRY_S)
    await r.settled()
    assert r.releases() == 2


async def test_a_release_is_asked_again_only_under_the_goals_it_was_asked_under():
    r = driving("alpha")
    await r.squeeze()
    r.calls.release_outcome = Unanswered()
    r.driver.drop("alpha", "right")
    await r.settled()
    r.fleet.generation += 1
    await r.clock.advance(RETRY_S)
    await r.settled()
    assert r.releases() == 1


async def test_a_release_retry_waits_for_a_take_in_flight():
    r = driving("alpha")
    await r.squeeze()
    r.calls.release_outcome = Unanswered()
    r.driver.drop("alpha", "right")
    await r.settled()
    take = r.calls.take
    taking = asyncio.Event()
    go_on = asyncio.Event()

    async def slow_take(robot):
        taking.set()
        await go_on.wait()
        return await take(robot)

    r.calls.take = slow_take
    squeeze = asyncio.create_task(r.squeeze())
    await taking.wait()
    # The retry comes round while the take runs: it waits for the take, and
    # then the headset holds the robot.
    await r.clock.advance(RETRY_S)
    go_on.set()
    assert await squeeze is None
    await r.settled()
    assert r.releases() == 1 and r.driver.holds("alpha")


async def test_a_pending_release_gives_way_to_a_new_squeeze():
    r = driving("alpha")
    await r.squeeze()
    r.calls.release_outcome = Unanswered()
    r.driver.drop("alpha", "right")
    await r.settled()
    # The robot is taken again before the retry comes round: the retry must
    # not release the lease the new squeeze drives under.
    await r.squeeze()
    await r.clock.advance(RETRY_S)
    await r.settled()
    assert r.releases() == 1
    assert r.driver.holds("alpha")


# The page and the panel


async def test_the_robot_row_says_who_holds_the_chosen_robot():
    r = driving("alpha", leases=CLOSED_LEASE)
    line = r.driver.robot_line("alpha")
    assert (line.text, line.tone) == (
        "alpha: leased by director_inst (tidy the table)",
        "waiting",
    )
    r.fleet.view = fleet_view(
        ["alpha"], leases={"alpha": Lease(HEADSET, LEASE_NAME, open=False)}
    )
    line = r.driver.robot_line("alpha")
    assert (line.text, line.tone) == ("alpha: driven from this headset", "driving")
    r.fleet.view = fleet_view(["alpha"])
    assert r.driver.robot_line("alpha").tone == "resting"
    r.fleet.view = fleet_view(["alpha"], halted_by="peppy-cli")
    line = r.driver.robot_line("alpha")
    assert (line.text, line.tone) == (f"alpha: {HALTED}", "waiting")
    assert r.driver.robot_line(None).text == "no robot paired"
    r.fleet.status = DaemonLost()
    assert r.driver.robot_line("alpha").text == "the daemon is lost"


async def test_a_robot_reads_as_the_cause_while_the_fleet_cannot_be_asked():
    r = driving("alpha")
    r.fleet.view = None
    r.fleet.status = Starting()
    assert r.driver.state_of("alpha") == "waiting for the fleet listing"


async def test_the_page_hears_each_change_a_transfer_and_a_give_back_included():
    r = driving("alpha", leases=CLOSED_LEASE)
    # The director transfers the lease to a run, which gives it back.
    for view in (
        fleet_view(["alpha"], leases={"alpha": Lease(TIDY, "tidy the table", False)}),
        fleet_view(["alpha"], leases=CLOSED_LEASE),
    ):
        r.fleet.view = view
        r.driver.fleet_changed(view)
    assert [view.leases["alpha"].holder for view in r.changes] == [TIDY, DIRECTOR]
    assert r.driver.state_of("alpha") == "leased by director_inst (tidy the table)"
