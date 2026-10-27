"""The headset's side of fleet:v1: the listing, the robot states, the two
long goals and the calls of a lease holder, against the generated mock of the
daemon target."""

import asyncio
import contextlib

from peppygen.consumed_actions.fleet import changes, lease
from peppygen.consumed_services.fleet import check_move, release, take
from peppygen.consumed_services.limb_motion import stop

from tests.helpers import (
    CALL_S,
    DIRECTOR,
    HEADSET,
    RETRY_S,
    ManualClock,
    accept_both,
    boot,
    change,
    eventually,
    fleet_view,
    lease_event,
    listing,
    running,
)
from xr_commander import fleet
from xr_commander.bus import Answered, Failed, Unanswered
from xr_commander.drive import LimbStops
from xr_commander.fleet import (
    LEASE_NAME,
    DaemonLost,
    DrivenHere,
    Fleet,
    FleetCalls,
    FleetView,
    Following,
    Free,
    Lease,
    Leased,
    NotListed,
    OpenLease,
    Refused,
    Starting,
    Timing,
    robot_state,
    state_text,
    status_text,
)
from xr_commander.robots import PairedRobots

# The bound of a call that the test leaves unanswered.
SILENT_CALL_S = 0.2


def test_a_listing_parses_into_the_robots_their_leases_and_the_halt():
    view = FleetView.parse(
        listing(
            robots=["alpha", "bravo"],
            leases=[("alpha", DIRECTOR, "tidy the table", False)],
        ),
        revision=7,
    )
    assert view.revision == 7
    assert view.robots == {"alpha", "bravo"}
    assert view.leases["alpha"] == Lease(DIRECTOR, "tidy the table", open=False)
    assert view.halted_by is None
    halted = FleetView.parse(listing(robots=["alpha"], halted_by="peppy-cli"), 8)
    assert halted.halted_by == "peppy-cli"


def test_each_robot_state_reads_off_the_listing_and_says_who_holds_it():
    view = fleet_view(
        ["free", "mine", "leased", "open"],
        leases={
            "mine": Lease(HEADSET, LEASE_NAME, open=False),
            "leased": Lease(DIRECTOR, "tidy the table", open=False),
            "open": Lease(DIRECTOR, "tidy the table", open=True),
        },
    )

    def state(robot, under_open=False):
        return robot_state(view, robot, HEADSET, driven_under_open=under_open)

    assert state("free") == Free()
    assert state("mine") == DrivenHere()
    assert state("leased") == Leased("director_inst", "tidy the table")
    assert state("open") == OpenLease("director_inst")
    assert state("open", under_open=True) == DrivenHere(open_of="director_inst")
    assert state("gone") == NotListed()
    assert [
        state_text(state(robot)) for robot in ("free", "mine", "leased", "open", "gone")
    ] == [
        "free",
        "driven from this headset",
        "leased by director_inst (tidy the table)",
        "open for a take-over by director_inst",
        "not in the fleet listing",
    ]
    assert (
        state_text(state("open", under_open=True))
        == "driven from this headset under the open lease of director_inst"
    )


def test_the_goals_status_says_why_the_headset_cannot_ask_the_fleet():
    assert status_text(Following()) is None
    assert status_text(Starting()) == "waiting for the fleet listing"
    assert status_text(DaemonLost()) == "the daemon is lost"
    assert (
        status_text(Refused("no fleet scope"))
        == "the daemon refused the fleet goals until the next launch: no fleet scope"
    )


class Listener:
    """Records what the two goals hand on, each in turn."""

    def __init__(self) -> None:
        self.heard: asyncio.Queue = asyncio.Queue()

    def fleet_changed(self, view):
        self.heard.put_nowait(("changed", view.revision))

    def own_lease_ended(self, robot, why):
        self.heard.put_nowait(("lease_end", robot, why))

    def let_go_of_every_robot(self, why):
        self.heard.put_nowait(("let_go", why))

    async def next(self):
        return await asyncio.wait_for(self.heard.get(), CALL_S)


@contextlib.asynccontextmanager
async def following(node_runner, clock, changes_module=changes, lease_module=lease):
    """`follow_fleet` running: yields the fleet, the listener and the
    task."""
    the_fleet = Fleet(HEADSET)
    listener = Listener()

    def follow(token):
        return fleet.follow_fleet(
            node_runner,
            the_fleet,
            listener,
            changes_module=changes_module,
            lease_module=lease_module,
            timing=Timing(call_timeout_s=CALL_S, retry_s=RETRY_S),
            token=token,
            clock=clock,
        )

    async with running(follow) as (task,):
        yield the_fleet, listener, task


async def test_the_headset_follows_the_listing_and_hears_the_end_of_its_leases():
    async with (
        boot() as h,
        following(h.node_runner, ManualClock()) as (the_fleet, listener, _),
    ):
        follow, hold = await accept_both(h.mocks.deps.fleet)
        await eventually(lambda: the_fleet.generation == 1)
        assert the_fleet.unavailable() == "waiting for the fleet listing"
        await follow.publish_feedback(change("listing", revision=3, robots=["alpha"]))
        assert await listener.next() == ("changed", 3)
        assert the_fleet.unavailable() is None
        assert the_fleet.view.robots == {"alpha"}
        # Only the end of a lease reaches the listener: the headset gives
        # and receives no lease.
        await hold.publish_feedback(lease_event("offer"))
        await hold.publish_feedback(lease_event("lease_back"))
        await hold.publish_feedback(
            lease_event("lease_end", cause="the stack is halted")
        )
        assert await listener.next() == ("lease_end", "alpha", "the stack is halted")


async def test_an_ended_changes_goal_lets_go_of_every_robot_and_both_goals_start_again():
    async with (
        boot() as h,
        following(h.node_runner, ManualClock()) as (the_fleet, listener, _),
    ):
        mock = h.mocks.deps.fleet
        follow, hold = await accept_both(mock)
        await follow.publish_feedback(change("listing", robots=["alpha"]))
        assert await listener.next() == ("changed", 1)
        await follow.complete()
        assert await listener.next() == (
            "let_go",
            "the changes goal of the fleet ended",
        )
        assert the_fleet.view is None
        assert the_fleet.unavailable() == "waiting for the fleet listing"
        # The lease goal that still ran is ended, so no lease outlives the
        # listing the headset drives by.
        await asyncio.wait_for(hold.cancel_signal(), CALL_S)
        # Both start again at once: the daemon answers.
        follow, hold = await accept_both(mock)
        await eventually(lambda: the_fleet.generation == 2)
        await follow.publish_feedback(change("listing", revision=2, robots=["alpha"]))
        assert await listener.next() == ("changed", 2)
        assert the_fleet.unavailable() is None


async def test_an_ended_lease_goal_lets_go_of_every_robot_and_both_goals_start_again():
    async with (
        boot() as h,
        following(h.node_runner, ManualClock()) as (the_fleet, listener, _),
    ):
        mock = h.mocks.deps.fleet
        follow, hold = await accept_both(mock)
        await follow.publish_feedback(change("listing", robots=["alpha"]))
        assert await listener.next() == ("changed", 1)
        await hold.complete()
        assert await listener.next() == ("let_go", "the lease goal of the fleet ended")
        await asyncio.wait_for(follow.cancel_signal(), CALL_S)
        await accept_both(mock)


async def test_a_refused_goal_leaves_the_headset_unable_to_drive_until_the_next_launch():
    clock = ManualClock()
    async with (
        boot() as h,
        following(h.node_runner, clock) as (the_fleet, listener, task),
    ):
        mock = h.mocks.deps.fleet
        follow = await (await mock.changes.next_goal(CALL_S)).accept()
        pending = await mock.lease.next_goal(CALL_S)
        await pending.refuse(lease.FailureKind.REJECTED, "the instance has no scope")
        await eventually(lambda: isinstance(the_fleet.status, Refused))
        assert the_fleet.unavailable() == (
            "the daemon refused the fleet goals until the next launch: "
            "the instance has no scope"
        )
        # The accepted goal ends with the refused one.
        await asyncio.wait_for(follow.cancel_signal(), CALL_S)
        # Never asked again: following the fleet ended.
        await asyncio.wait_for(task, CALL_S)
        assert listener.heard.empty()


async def test_a_lease_goal_refused_while_the_last_one_ends_starts_again_on_the_clock():
    clock = ManualClock()
    async with boot() as h, following(h.node_runner, clock) as (the_fleet, _, task):
        mock = h.mocks.deps.fleet
        follow = await (await mock.changes.next_goal(CALL_S)).accept()
        pending = await mock.lease.next_goal(CALL_S)
        await pending.refuse(
            lease.FailureKind.PRECONDITION_FAILED,
            "a `lease` goal of this process of `headset_inst` runs",
        )
        await asyncio.wait_for(follow.cancel_signal(), CALL_S)
        await clock.advance(0.0, parked=[task])
        # The daemon answers: the headset waits for the listing.
        assert the_fleet.unavailable() == "waiting for the fleet listing"
        await clock.advance(RETRY_S, parked=[task])
        await accept_both(mock)
        await eventually(lambda: the_fleet.generation == 1)
        assert not isinstance(the_fleet.status, Refused)


async def test_a_lost_daemon_lets_go_of_every_robot_and_says_so():
    async with (
        boot() as h,
        following(h.node_runner, ManualClock()) as (the_fleet, listener, _),
    ):
        mock = h.mocks.deps.fleet
        follow, _ = await accept_both(mock)
        await follow.publish_feedback(change("listing", robots=["alpha"]))
        assert await listener.next() == ("changed", 1)
        # The daemon process dies.
        await mock.stop()
        assert await listener.next() == ("let_go", "the daemon is lost")
        assert the_fleet.unavailable() == "the daemon is lost"


class ScriptedGoals:
    """A goal module of the fleet whose daemon the test scripts. A start is
    unanswered until `answers`, refused with `refusal_kind` once set, and,
    for an `exclusive` module, refused with precondition_failed while an
    earlier goal runs that no cancel reached. A goal's feedback ends when the
    test sets `ended`, and the first `lost_cancels` cancels get no answer."""

    FailureKind = lease.FailureKind

    def __init__(self, name: str, *, exclusive: bool = False) -> None:
        self.TARGET_ACTION_NAME = name
        self.starts = 0
        self.answers = False
        self.refusal_kind: lease.FailureKind | None = None
        self.exclusive = exclusive
        self.lost_cancels = 0
        self.running: set = set()
        self.ended = asyncio.Event()
        module = self

        class Handle:
            def __init__(self, refusal=None) -> None:
                self.accepted = refusal is None
                self.refusal = refusal

            @classmethod
            async def fire_goal(cls, _runner, _target, _timeout):
                module.starts += 1
                if not module.answers:
                    raise ConnectionError("no producer")
                kind = module.refusal_kind
                if kind is None and module.exclusive and module.running:
                    kind = lease.FailureKind.PRECONDITION_FAILED
                if kind is not None:
                    return cls(lease.Failure(kind, f"the {name} goal is refused"))
                handle = cls()
                module.running.add(handle)
                return handle

            async def on_next_feedback_message(self):
                await module.ended.wait()
                module.ended.clear()
                raise RuntimeError(f"the {name} goal ended")

            async def cancel_goal(self, _timeout):
                if module.lost_cancels > 0:
                    module.lost_cancels -= 1
                    raise TimeoutError("the cancel got no answer")
                module.running.discard(self)

        self.ActionHandle = Handle

    def bound_producer(self, _runner):
        return "daemon"


async def test_the_goals_start_again_only_once_the_daemon_answers():
    clock = ManualClock()
    follow, hold = ScriptedGoals("changes"), ScriptedGoals("lease")
    async with following(None, clock, follow, hold) as (the_fleet, _, task):
        await clock.advance(0.0, parked=[task])
        assert the_fleet.unavailable() == "the daemon is lost"
        assert follow.starts == 1
        # Each retry asks again, on the clock.
        await clock.advance(RETRY_S, parked=[task])
        assert follow.starts == 2 and hold.starts == 0
        follow.answers = hold.answers = True
        await clock.advance(RETRY_S, parked=[task])
        assert (follow.starts, hold.starts) == (3, 1)
        assert the_fleet.unavailable() == "waiting for the fleet listing"


async def test_a_lease_goal_refused_after_a_lost_daemon_says_the_headset_waits(capsys):
    clock = ManualClock()
    follow, hold = ScriptedGoals("changes"), ScriptedGoals("lease")
    async with following(None, clock, follow, hold) as (the_fleet, _, task):
        await clock.advance(0.0, parked=[task])
        assert the_fleet.unavailable() == "the daemon is lost"
        # The daemon answers, and still ends the last lease goal of the
        # headset: the headset waits for the listing and asks again.
        follow.answers = hold.answers = True
        hold.refusal_kind = lease.FailureKind.PRECONDITION_FAILED
        await clock.advance(RETRY_S, parked=[task])
        assert the_fleet.unavailable() == "waiting for the fleet listing"
        await clock.advance(RETRY_S, parked=[task])
        assert hold.starts == 2
        assert capsys.readouterr().out.count("the fleet goals start again later") == 1
        hold.refusal_kind = None
        await clock.advance(RETRY_S, parked=[task])
        await eventually(lambda: the_fleet.generation == 1)


async def test_a_changes_goal_refused_with_precondition_failed_is_final():
    clock = ManualClock()
    follow, hold = ScriptedGoals("changes"), ScriptedGoals("lease")
    follow.answers = hold.answers = True
    follow.refusal_kind = lease.FailureKind.PRECONDITION_FAILED
    async with following(None, clock, follow, hold) as (the_fleet, _, task):
        await asyncio.wait_for(task, CALL_S)
        assert isinstance(the_fleet.status, Refused)


async def test_a_lost_cancel_of_the_last_lease_goal_is_asked_again_before_a_restart():
    clock = ManualClock()
    follow, hold = ScriptedGoals("changes"), ScriptedGoals("lease", exclusive=True)
    follow.answers = hold.answers = True
    async with following(None, clock, follow, hold) as (the_fleet, listener, _):
        await eventually(lambda: the_fleet.generation == 1)
        # The changes goal ends, and the cancel of the lease goal gets no
        # answer: the daemon would refuse every new lease goal.
        hold.lost_cancels = 1
        follow.ended.set()
        assert (await listener.next())[0] == "let_go"
        await eventually(lambda: the_fleet.generation == 2)
        assert hold.starts == 2 and len(hold.running) == 1


def calls_of(h, call_timeout_s=CALL_S) -> FleetCalls:
    return FleetCalls(
        h.node_runner,
        take_module=take,
        release_module=release,
        check_move_module=check_move,
        timing=Timing(call_timeout_s=call_timeout_s),
    )


async def test_a_take_carries_the_lease_name_and_a_refusal_names_the_holder():
    async with boot() as h:
        calls = calls_of(h)
        mock = h.mocks.deps.fleet

        async def refuse():
            request, responder = await mock.take.next_request(CALL_S)
            assert (request.robot, request.name) == ("alpha", LEASE_NAME)
            await responder.respond_failure(
                take.FailureKind.LEASED,
                "robot `alpha` is leased by `director_inst` on `core` under the "
                "name `tidy the table`",
            )

        refused, _ = await asyncio.gather(calls.take("alpha"), refuse())
        assert refused == Failed(
            "robot `alpha` is leased by `director_inst` on `core` under the name "
            "`tidy the table`"
        )

        async def agree():
            _, responder = await mock.take.next_request(CALL_S)
            await responder.respond()

        taken, _ = await asyncio.gather(calls.take("alpha"), agree())
        assert isinstance(taken, Answered)


async def test_a_move_check_asks_by_robot_or_by_instance_and_gives_the_refusal():
    async with boot() as h:
        calls = calls_of(h)
        mock = h.mocks.deps.fleet

        async def answer(expected, refusal):
            request, responder = await mock.check_move.next_request(CALL_S)
            assert (request.instance, request.robot) == expected
            await responder.respond(
                check_move.ResponseData(refusal=refusal, revision=4)
            )

        refusal = check_move.ResponseDataRefusal(
            failure_kind="halted",
            message="the stack is halted (by peppy-cli); clear the halt first",
        )
        refused, _ = await asyncio.gather(
            calls.check_move(robot="alpha"), answer((None, "alpha"), refusal)
        )
        assert refused == Failed("the stack is halted (by peppy-cli)")
        allowed, _ = await asyncio.gather(
            calls.check_move(instance="backbone_inst"),
            answer(("backbone_inst", None), None),
        )
        # A yes carries the revision of the fleet that it reads.
        assert allowed == Answered(4)

        async def fail():
            _, responder = await mock.check_move.next_request(CALL_S)
            await responder.respond_failure(
                check_move.FailureKind.REJECTED, "no instance `ghost_inst`"
            )

        failed, _ = await asyncio.gather(
            calls.check_move(instance="ghost_inst"), fail()
        )
        assert failed == Failed("no instance `ghost_inst`")


async def test_a_release_is_answered_or_refused():
    async with boot() as h:
        mock = h.mocks.deps.fleet
        calls = calls_of(h)

        async def answer(refuse: bool):
            _, responder = await mock.release.next_request(CALL_S)
            if refuse:
                await responder.respond_failure(
                    release.FailureKind.NOT_LEASED, "the lease ended at the halt"
                )
            else:
                await responder.respond()

        released, _ = await asyncio.gather(calls.release("alpha"), answer(False))
        assert isinstance(released, Answered)
        refused, _ = await asyncio.gather(calls.release("alpha"), answer(True))
        assert refused == Failed("the lease ended at the halt")


async def test_each_call_the_daemon_does_not_answer_ends_unanswered():
    async with boot() as h:
        calls = calls_of(h, call_timeout_s=SILENT_CALL_S)
        mock = h.mocks.deps.fleet
        assert await calls.take("alpha") == Unanswered()
        assert (await mock.take.next_request(CALL_S))[0].robot == "alpha"
        assert await calls.release("alpha") == Unanswered()
        assert (await mock.release.next_request(CALL_S))[0].robot == "alpha"
        assert await calls.check_move(robot="alpha") == Unanswered()
        assert (await mock.check_move.next_request(CALL_S))[0].robot == "alpha"


def stops_of(h, call_timeout_s=CALL_S) -> LimbStops:
    robots = PairedRobots(h.node_runner, arms={}, grippers={})
    return LimbStops(h.node_runner, robots, stop, Timing(call_timeout_s=call_timeout_s))


ROBOT_AND_BRAVO = {
    "limb_motion_instance_ids": ["backbone_inst"],
    "right_arm_members": {"backbone_inst": None, "bravo_backbone_inst": "bravo"},
}


async def test_a_stop_reaches_the_robots_own_backbone_with_the_reason_that_names_the_headset():
    async with boot(**ROBOT_AND_BRAVO) as h:
        stops = stops_of(h)
        mock = h.mocks.deps.limb_motion[0].limb_motion_stop

        async def answer():
            request, responder = await mock.next_request(CALL_S)
            assert request.reason == LEASE_NAME
            await responder.respond(stop.ResponseData(stopped=["right_arm"]))

        stopped, _ = await asyncio.gather(stops.stop("backbone_inst"), answer())
        assert stopped is None
        # Bravo has no limb_motion of its own: nothing to stop it with.
        assert await stops.stop("bravo") == (
            "bravo has no limb_motion: the launcher links none to the headset"
        )


async def test_a_stop_that_fails_or_gets_no_answer_says_why():
    async with boot(**ROBOT_AND_BRAVO) as h:
        mock = h.mocks.deps.limb_motion[0].limb_motion_stop

        async def fail():
            _, responder = await mock.next_request(CALL_S)
            await responder.respond_failure(
                stop.FailureKind.RESOURCE_UNAVAILABLE, "the arm's bus is down"
            )

        failed, _ = await asyncio.gather(stops_of(h).stop("backbone_inst"), fail())
        assert failed == "the arm's bus is down"
        silent = stops_of(h, call_timeout_s=SILENT_CALL_S)
        assert await silent.stop("backbone_inst") == "backbone_inst did not answer"
        assert (await mock.next_request(CALL_S))[0].reason == LEASE_NAME
