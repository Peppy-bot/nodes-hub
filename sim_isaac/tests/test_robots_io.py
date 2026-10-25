"""Who may join this stage: the name a robot stands under, the spot it is
promised, and what an admission gives back when its goal never reaches its
stay.

A robot is admitted on the node loop and stood later, on the thread that steps
the scene, so its name and its spot are held across that gap. What holds them
refuses whoever asks for them meanwhile, and gives them back the moment the
robot is on its way out, so a copy that comes straight back finds its own name
and its own spot waiting for it. A refusal and the end of a stay each give the
failure kind or the reason that the simulation_robot contract names.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from unittest.mock import Mock

from sim_robot_core.pairs import Held
from sim_robot_core.registry import Caller, Registry

import runtime_fakes  # pylint: disable=W0611  (installs the runtime robots_io imports)
from runtime_fakes import CANCEL_REASONS, FailureKind

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from edits import Edits  # noqa: E402  pylint: disable=C0413
from isaac_models import IsaacModels  # noqa: E402  pylint: disable=C0413
from robots_io import Ending, RobotsIO  # noqa: E402  pylint: disable=C0413
from world import Placement, Robot, World  # noqa: E402  pylint: disable=C0413

MODELS = IsaacModels.read()

# The core node every copy in these tests attaches from.
CORE_NODE = "sim16"
# The spot a robot in the way stands on, and the one a copy coming straight
# back is owed.
ORIGIN = Placement.of((0.0, 0.0, 0.0), 0.0)


def _stage(*standing: tuple[str, str, Placement]) -> World:
    """The stage as the thread that steps the scene leaves it, each robot
    given as the name it stands under, the model it is, and where."""
    world = World(head_camera_pack=None)
    world._robots = {  # pylint: disable=W0212
        name: Robot(instance=name, known=MODELS.of(model), placement=at)
        for name, model, at in standing
    }
    return world


class _ClosedStage:
    """A stage that answers nothing. A robot refused as it is admitted never
    reaches it."""

    def free_spot(self, promised=(), leaving=None):
        raise AssertionError("the stage laid out a spot for a robot it refused")

    def standing_within(self, placement, leaving=None):
        raise AssertionError("the stage was read for a robot it refused")


def _robots_io(world) -> RobotsIO:
    """A RobotsIO with only the parts admitting a robot touches, over the
    models this engine ships an entry for and the stage it is given."""
    io = RobotsIO.__new__(RobotsIO)
    io._models = MODELS
    io._robots = Registry()
    io._world = world
    io._loop = SimpleNamespace(time=lambda: 0.0)
    io._handovers = {}
    io._placements = {}
    io._admitted = None
    return io


def _request(
    robot: str = "alpha",
    model: str = "openarm_v2",
    placement=None,
    instance: Optional[str] = None,
):
    """What a copy attaches with: the robot it joins as, the model it is, and
    where it asks to stand. A request naming no copy runs as the robot's
    first one."""
    return SimpleNamespace(
        core_node=CORE_NODE,
        instance_id=instance or f"{robot}_init_inst",
        data=SimpleNamespace(robot=robot, model=model, placement=placement),
    )


def _accepted(decision) -> bool:
    return decision.refusal is None


def _refused(decision, kind: FailureKind) -> str:
    """The message of a refusal of `kind`."""
    assert decision.refusal is not None, "the goal was accepted"
    assert decision.refusal.kind is kind, decision.refusal
    return decision.refusal.message


# The reason the goal of a robot that its caller took out ends cancelled with.
CANCELLED_BY_ITS_CALLER = "cancelled by its caller"
# The reason the goal that held a robot ends cancelled with, once another goal
# of the same caller holds it.
HANDED_OVER = "another goal of the same caller holds the robot now"


def _asks_for(x: float, yaw: float = 0.0):
    """The placement a request carries, as the contract hands it over."""
    return SimpleNamespace(position=[x, 0.0, 0.0], yaw=yaw)


def _done() -> concurrent.futures.Future:
    """A change the thread that steps the scene has already made."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(None)
    return future


class _Answers:
    """The answers a goal context takes, and why it received its cancel
    signal: "caller" unless a test says otherwise."""

    def __init__(self, cause: str = "caller"):
        self.answers = []
        self.cause = cause

    def cancel_cause(self):
        return self.cause

    def cancel_reason(self):
        return CANCEL_REASONS.get(self.cause)

    async def complete(self):
        self.answers.append(("completed",))

    async def fail(self, kind, message):
        self.answers.append(("failed", kind, message))

    async def complete_cancelled(self, reason):
        self.answers.append(("cancelled", reason))


def test_a_name_carrying_the_prim_separator_is_refused_before_a_spot_is_looked_for():
    """Every robot is a prim of the stage under its own name, so a name
    carrying the separator nests one robot inside another's path. The refusal
    comes with the admission and names what to join as, so nothing is read off
    the stage and nothing is reserved for a name the stage was never going to
    carry."""
    io = _robots_io(_ClosedStage())

    decision = io._admit(_request(robot="left/arm"))

    assert "peppy stack join LAUNCHER -i left_arm" in _refused(decision, FailureKind.REJECTED)
    assert io._robots.robots() == []
    assert io._placements == {}


def test_a_taken_spot_is_refused_naming_who_holds_it_and_what_to_ask_for():
    """A robot standing on the spot is named as standing, where a copy only
    promised it is named as about to. Either way the refusal says what to
    type, and the name goes back so the copy may ask again."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request(robot="bravo", placement=_asks_for(3.0))))

    on_alpha = _refused(
        io._admit(_request(robot="charlo", placement=_asks_for(0.4))), FailureKind.PRECONDITION_FAILED
    )
    on_bravo = _refused(
        io._admit(_request(robot="charlo", placement=_asks_for(3.4))), FailureKind.PRECONDITION_FAILED
    )

    assert "'alpha' stands within 1.5 m of [0.4, 0.0, 0.0]" in on_alpha
    assert "'bravo' is about to stand within 1.5 m of [3.4, 0.0, 0.0]" in on_bravo
    assert "turn its `placement.auto` on" in on_alpha
    assert io._robots.of_name("charlo") is None
    assert "charlo" not in io._placements


@pytest.mark.parametrize("asked", [None, _asks_for(0.0)], ids=["auto", "the spot it had"])
def test_a_copy_that_comes_straight_back_is_promised_the_spot_it_stands_on(asked):
    """Its name went back when its stay ended, so the robot still standing
    under that name is on its way out: the spot it stands on is the one the
    copy is coming back to, whether it asks for that spot or takes the one
    this stage lays out for it."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))

    decision = io._admit(_request(instance="alpha_init_2", placement=asked))

    assert _accepted(decision)
    assert io._placements["alpha"] == ORIGIN


def test_a_copy_admitted_while_its_robot_leaves_is_stood_afresh():
    """Taking a robot out waits for the thread that steps the scene, which is
    a whole stage resolved, and a copy that comes straight back attaches
    inside that window. Everything the robot's joining held goes back before
    the stage is asked to let it go, so the copy is admitted under its own
    name, on the spot its robot is standing on, and its own stay stands it
    there."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))
    io._robots.stand("alpha")
    joined = []

    def attach_the_copy(_change):
        # The take-out reaches the thread that steps the scene here, and
        # the copy attaches while it waits for the scene to be resolved.
        joined.append(io._admit(_request(instance="alpha_init_2")))
        return _done()

    io._edits = SimpleNamespace(submit=attach_the_copy)

    asyncio.run(io._take_out(_Answers(), "alpha", Ending.LEFT, CANCELLED_BY_ITS_CALLER))

    assert [_accepted(decision) for decision in joined] == [True]
    copy = io._robots.of_name("alpha")
    assert copy.caller.instance_id == "alpha_init_2"
    assert not copy.standing()
    assert io._placements["alpha"] == ORIGIN


def test_a_goal_that_never_reached_its_stay_gives_back_its_name_and_its_spot():
    """A goal is accepted on a hop of its own after this engine has answered
    for it, and a robot with no stay has nothing watching its pair grace time:
    what its admission reserved is held by nobody, and the copy that comes after it
    takes the name."""
    io = _robots_io(_stage())

    assert _accepted(io._admit(_request()))
    io._withdraw()

    assert io._robots.of_name("alpha") is None
    assert io._placements == {}
    assert _accepted(io._admit(_request(instance="alpha_init_2")))


def test_withdrawing_again_leaves_the_copy_that_took_the_name_alone():
    """Serving the attaches withdraws whenever it fails and goes back to
    waiting, so it withdraws again having admitted nothing in between. One
    admission is given back once: the name is the next copy's by then."""
    io = _robots_io(_stage())
    assert _accepted(io._admit(_request()))
    io._withdraw()

    # The name the withdrawal freed is taken by a copy of its own.
    io._robots.admit(
        "alpha",
        MODELS.of("openarm_v2").entry,
        Caller(core_node=CORE_NODE, instance_id="alpha_init_2"),
    )
    io._robots.stand("alpha")
    io._withdraw()

    assert io._robots.of_name("alpha").standing()


class _Hop:
    """An action handle that admits one goal and loses it, the way a goal
    accepted on a hop of its own is lost once this engine has answered for
    it."""

    def __init__(self, request):
        self._request = request
        self.decisions = []

    async def handle_goal_next_request(self, admit):
        self.decisions.append(admit(self._request))
        return None


def test_a_goal_lost_on_the_way_back_frees_the_name_it_was_admitted_under():
    """The engine answers each attach and then waits for the goal it
    accepted. A goal that never arrives has nothing watching its pair grace
    time, so serving the attaches gives its name and its spot back."""
    io = _robots_io(_stage())
    handle = _Hop(_request())

    asyncio.run(io._serve_attach(handle))

    assert [_accepted(decision) for decision in handle.decisions] == [True]
    assert io._robots.of_name("alpha") is None
    assert io._placements == {}


def test_a_handover_that_never_reached_its_stay_leaves_the_robot_standing():
    """A goal admitted for the robot its own copy stands reserves nothing:
    the robot stays in the scene, on the spot it stands on, when that goal
    never reaches its stay."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))
    io._robots.stand("alpha")
    promised = io._placements["alpha"]

    assert _accepted(io._admit(_request()))
    io._withdraw()

    assert io._robots.of_name("alpha").standing()
    assert io._placements["alpha"] == promised


def test_a_copy_re_registering_its_own_robot_takes_it_over_where_it_stands():
    """Its initializer died and came back: the same copy attaching the robot
    it stands is accepted and the robot stays exactly as it is. Admitting it
    reserves nothing and ends nobody, so the robot keeps the stay hosting it
    until the stay taking it over exists."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))
    io._robots.stand("alpha")
    hosted = io._handover("alpha")

    decision = io._admit(_request())

    assert _accepted(decision)
    assert decision.response.arm_names == ["left_arm", "right_arm"]
    assert io._admitted is None
    assert not hosted.is_set()
    assert list(io._robots.standing()) == ["alpha"]


def _built(pair_grace_s: float = 2.0) -> RobotsIO:
    """A RobotsIO built the way the launch builds it, so the order its parts
    are given in is the order it reads them."""
    return RobotsIO(
        None,
        asyncio.new_event_loop(),
        MODELS,
        Registry(),
        World(head_camera_pack=None),
        Edits(),
        _HoldsEverything(),
        pair_grace_s,
    )


def test_a_robots_io_reads_each_part_as_the_launch_gives_it():
    """Its registry and its stage are two arguments of one type each, and a
    launch passes them positionally: swapped, the engine reads every robot
    off the stage and every spot off the registry."""
    io = _built()

    assert isinstance(io._robots, Registry)
    assert isinstance(io._world, World)
    assert io._models is MODELS


def test_a_pair_grace_time_no_robot_can_keep_is_refused_as_the_engine_is_built():
    """The pair grace time is how long a robot whose pairs are gone stays, so
    one that is zero or below takes every robot off the stage as it stands."""
    with pytest.raises(ValueError, match="robot_pair_grace_ms must be positive"):
        _built(pair_grace_s=0.0)


def test_another_caller_naming_the_standing_robot_is_refused():
    """The name is the sharper answer than the engine's capacity: it names
    the instance whose robot stands under it."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))

    decision = io._admit(_request(instance="other_init_inst"))

    assert "already stands as 'alpha'" in _refused(decision, FailureKind.PRECONDITION_FAILED)


def test_a_second_robot_of_the_same_caller_is_refused():
    """A caller holds one robot: attaching another while its first stands is
    a request the stage cannot take now."""
    io = _robots_io(_stage())
    assert _accepted(io._admit(_request()))

    decision = io._admit(_request(robot="charlo", instance="alpha_init_inst"))

    assert "already stands a robot in this scene, as 'alpha'" in _refused(
        decision, FailureKind.PRECONDITION_FAILED
    )


def test_a_copy_re_registering_a_robot_that_still_joins_is_busy_until_it_stands():
    """The robot the copy attached has not stood yet: a second goal for it
    is refused with `busy`, and the first goal goes on. Once the robot
    stands, the same goal takes it over."""
    io = _robots_io(_stage())
    assert _accepted(io._admit(_request()))

    decision = io._admit(_request())

    assert "already joining 'alpha'" in _refused(decision, FailureKind.BUSY)
    io._robots.stand("alpha")
    assert _accepted(io._admit(_request()))


def test_a_robot_that_joins_keeps_its_name_from_another_caller():
    """Only the caller's own robot is `busy` while it joins: once it stands,
    the name stays its own, so another caller is refused for good."""
    io = _robots_io(_stage())
    assert _accepted(io._admit(_request()))

    decision = io._admit(_request(instance="other_init_inst"))

    assert "already stands as 'alpha'" in _refused(decision, FailureKind.PRECONDITION_FAILED)


def test_a_copy_re_registering_as_another_model_is_refused():
    """A robot is the model it attached as, so the registry answers for the
    name and the goal is told."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))

    decision = io._admit(_request(model="so101"))

    assert "alpha" in _refused(decision, FailureKind.PRECONDITION_FAILED)


def test_a_goal_handed_over_leaves_its_robot_standing():
    """A copy whose initializer came back takes its own robot over: the goal
    that hosted it ends cancelled without asking the stage for anything."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    io._edits = Mock(spec=["submit"])
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.HANDED_OVER, HANDED_OVER))

    io._edits.submit.assert_not_called()
    assert goal.answers == [("cancelled", HANDED_OVER)]


def _ending_io() -> RobotsIO:
    """A RobotsIO whose robot `alpha` stands and whose stage takes a robot
    out at once."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    io._robots.admit("alpha", MODELS.of("openarm_v2").entry, Caller(CORE_NODE, "alpha_init_inst"))
    io._robots.stand("alpha")
    io._placements["alpha"] = ORIGIN
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = _done()
    io._pair_grace_s = 2.0
    io._io = _HoldsEverything()
    io._stopping = asyncio.Event()
    return io


@pytest.mark.parametrize(
    ("cause", "reason"),
    [("caller", CANCELLED_BY_ITS_CALLER), ("sender_gone", "its sender is gone")],
)
def test_a_robot_whose_goal_is_cancelled_is_taken_out_and_its_goal_ends_cancelled(cause, reason):
    """`peppy stack remove` cancels the goal, and so does peppy once the
    initializer is gone for longer than the sender grace: the robot leaves
    the stage, and the goal ends cancelled for the cause of its cancel."""
    io = _ending_io()
    goal = _Leaving(cause)

    async def let_go():
        goal.let_go.set()
        return await io._watch(goal, "alpha", io._handover("alpha"))

    ending, why = asyncio.run(let_go())
    asyncio.run(io._take_out(goal, "alpha", ending, why))

    assert (ending, why) == (Ending.LEFT, reason)
    io._edits.submit.assert_called_once()
    assert io._robots.of_name("alpha") is None
    assert goal.answers == [("cancelled", reason)]


@pytest.mark.parametrize(
    ("ending", "why"),
    [
        (Ending.LAPSED, "none of this robot's limbs were paired for 2.0s"),
        (Ending.FAILED, "the robot's holder could not be told it stands: the holder is away"),
        (Ending.STOPPED, "the engine stopped"),
    ],
)
def test_a_robot_the_engine_takes_out_fails_its_goal_as_gone(ending, why):
    """The engine took the robot out for a reason of its own: its goal fails
    with `resource_unavailable`, and the message says why."""
    io = _ending_io()
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", ending, why))

    assert io._robots.of_name("alpha") is None
    assert goal.answers == [("failed", FailureKind.RESOURCE_UNAVAILABLE, why)]


def _bare_io() -> RobotsIO:
    """A RobotsIO with only the parts a pair grace check and a forget
    touch."""
    io = RobotsIO.__new__(RobotsIO)
    io._placements = {}
    io._handovers = {}
    return io


class TestHowOftenAStayChecksItsPairGrace:
    """A stay wakes on a share of its pair grace time so a robot whose pairs
    are gone leaves within it, and no faster than the floor, because every wake is a
    turn of the node loop the whole engine shares."""

    def test_the_period_is_a_share_of_the_pair_grace_time(self):
        io = _bare_io()
        io._pair_grace_s = 8.0

        assert io._pair_grace_check_period() == 8.0 * 0.25

    def test_a_short_pair_grace_time_is_checked_no_faster_than_the_floor(self):
        """`robot_pair_grace_ms` takes any positive number, and a pair grace
        time of one millisecond would wake the node loop a thousand times a
        second."""
        io = _bare_io()
        io._pair_grace_s = 0.001

        assert io._pair_grace_check_period() == 0.05

    def test_every_pair_grace_time_is_checked_inside_itself(self):
        io = _bare_io()
        for pair_grace_s in (0.001, 0.05, 1.0, 8.0, 600.0):
            io._pair_grace_s = pair_grace_s
            assert 0.0 < io._pair_grace_check_period()
            assert io._pair_grace_check_period() <= max(pair_grace_s, 0.05)


def test_forgetting_a_robot_drops_its_spot_and_its_signal():
    """What a robot's joining held is a spot promised and a signal its own
    goal waits on, and a name given back leaves neither behind."""
    io = _bare_io()
    io._placements["alpha"] = Placement.of((0.0, 0.0, 0.0), 0.0)
    io._handover("alpha")

    io._forget("alpha")

    assert io._placements == {}
    assert io._handovers == {}


class _Leaving(_Answers):
    """A goal whose holder lets go the moment the stay waits on it."""

    def __init__(self, cause: str = "caller"):
        super().__init__(cause)
        self.let_go = asyncio.Event()

    async def cancel_signal(self):
        await self.let_go.wait()


class _Hosting(_Answers):
    """A goal whose holder follows its stay and never lets go."""

    def request(self):
        return _request()

    async def publish_feedback(self, standing):
        return None

    async def cancel_signal(self):
        await asyncio.Event().wait()


def test_the_stay_taking_a_robot_over_ends_the_one_hosting_it_and_not_itself():
    """The goal hosting the robot ends where the goal taking it over exists.
    Ending it takes the signal this stay then waits on, so the stay reads
    the one the handover leaves behind and keeps hosting the robot."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))
    io._robots.stand("alpha")
    io._pair_grace_s = 2.0
    io._io = _HoldsEverything()
    io._stopping = asyncio.Event()
    hosted = io._handover("alpha")
    goal = _Hosting()

    async def take_over() -> tuple:
        stay = asyncio.create_task(io._stay(goal))
        # Enough turns for the stay to end the goal it took the robot from
        # and to settle into watching the robot's pair grace time.
        for _ in range(50):
            await asyncio.sleep(0)
        watching = not stay.done(), list(goal.answers)
        stay.cancel()
        await asyncio.gather(stay, return_exceptions=True)
        return watching

    still_hosting, answered = asyncio.run(take_over())

    assert hosted.is_set()
    assert still_hosting
    assert answered == []


class _HoldsEverything:
    """Pairs that answer for whichever robot is asked, all of them held."""

    def held_by(self, robot: str) -> Held:
        return Held(
            arms=frozenset({"left_arm", "right_arm"}),
            grippers=frozenset({"left_gripper", "right_gripper"}),
        )


def test_a_placement_the_stage_cannot_stand_a_robot_on_is_rejected():
    """A prim placed at NaN reports no error and simulates nothing, so the
    placement is rejected as the robot is admitted, before its name is
    reserved."""
    io = _robots_io(_stage())

    decision = io._admit(_request(placement=_asks_for(float("nan"))))

    assert "a placement is finite in every coordinate" in _refused(decision, FailureKind.REJECTED)
    assert io._robots.robots() == []
    assert io._placements == {}


def test_a_model_the_engine_has_no_entry_for_is_rejected():
    io = _robots_io(_ClosedStage())

    decision = io._admit(_request(model="openarm_v9"))

    assert "unknown model 'openarm_v9'" in _refused(decision, FailureKind.REJECTED)
    assert io._robots.robots() == []


def test_the_attach_goal_keeps_the_robot_of_a_gone_sender_for_the_pair_grace_time():
    """The sender grace of the attach action is the pair grace time, set
    before the engine serves its first goal: peppy keeps the goal of an
    initializer that is gone for that long before it cancels it, so an
    initializer that comes back within it registers the robot that still
    stands."""
    io = _built(pair_grace_s=2.0)

    async def start_and_stop():
        await io.start()
        # One turn of the loop runs the attach loop up to its closed stream.
        await asyncio.sleep(0)
        await io.stop()

    asyncio.run(start_and_stop())

    handle = runtime_fakes.ATTACH_HANDLES[-1]
    assert handle.sender_grace_s == 2.0
    assert handle.sender_grace_s_when_served == 2.0


def test_an_initializer_that_comes_back_takes_its_robot_over_and_the_old_goal_ends_cancelled():
    """The first initializer of the copy is gone and its goal still runs, in
    its sender grace. The initializer that comes back attaches the robot
    again: the robot stays where it stands, the old goal ends cancelled with
    the reason the contract gives, and the new goal hosts the robot."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert _accepted(io._admit(_request()))
    io._robots.stand("alpha")
    io._pair_grace_s = 2.0
    io._io = _HoldsEverything()
    io._stopping = asyncio.Event()
    io._edits = Mock(spec=["submit"])
    gone, back = _Hosting(), _Hosting()

    async def come_back() -> tuple:
        first = asyncio.create_task(io._stay(gone))
        # Enough turns for each stay to settle into watching the robot.
        for _ in range(50):
            await asyncio.sleep(0)
        assert _accepted(io._admit(_request()))
        second = asyncio.create_task(io._stay(back))
        for _ in range(50):
            await asyncio.sleep(0)
        hosting = not second.done()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        return first.done(), hosting

    first_ended, still_hosting = asyncio.run(come_back())

    assert first_ended
    assert gone.answers == [("cancelled", HANDED_OVER)]
    assert still_hosting
    io._edits.submit.assert_not_called()
