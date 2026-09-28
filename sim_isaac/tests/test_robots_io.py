"""Who may join this stage: the name a robot stands under, the spot it is
promised, and what an admission gives back when its goal never reaches its
stay.

A robot is admitted on the node loop and stood later, on the thread that steps
the scene, so its name and its spot are held across that gap. What holds them
refuses whoever asks for them meanwhile, and gives them back the moment the
robot is on its way out, so a copy that comes straight back finds its own name
and its own spot waiting for it.
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

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime robots_io imports)

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


def _asks_for(x: float, yaw: float = 0.0):
    """The placement a request carries, as the contract hands it over."""
    return SimpleNamespace(position=[x, 0.0, 0.0], yaw=yaw)


def _done() -> concurrent.futures.Future:
    """A change the thread that steps the scene has already made."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(None)
    return future


class _Answers:
    """The answers a goal context takes."""

    def __init__(self):
        self.answers = []

    async def complete(self, success, message):
        self.answers.append(("completed", success, message))

    async def complete_cancelled(self, success, message):
        self.answers.append(("cancelled", success, message))


def test_a_name_carrying_the_prim_separator_is_refused_before_a_spot_is_looked_for():
    """Every robot is a prim of the stage under its own name, so a name
    carrying the separator nests one robot inside another's path. The refusal
    comes with the admission and names what to join as, so nothing is read off
    the stage and nothing is reserved for a name the stage was never going to
    carry."""
    io = _robots_io(_ClosedStage())

    decision = io._admit(_request(robot="left/arm"))

    assert not decision.accepted
    assert "peppy stack join LAUNCHER -i left_arm" in decision.payload
    assert io._robots.robots() == []
    assert io._placements == {}


def test_a_taken_spot_is_refused_naming_who_holds_it_and_what_to_ask_for():
    """A robot standing on the spot is named as standing, where a copy only
    promised it is named as about to. Either way the refusal says what to
    type, and the name goes back so the copy may ask again."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert io._admit(_request(robot="bravo", placement=_asks_for(3.0))).accepted

    on_alpha = io._admit(_request(robot="charlo", placement=_asks_for(0.4)))
    on_bravo = io._admit(_request(robot="charlo", placement=_asks_for(3.4)))

    assert not on_alpha.accepted
    assert not on_bravo.accepted
    assert "'alpha' stands within 1.5 m of [0.4, 0.0, 0.0]" in on_alpha.payload
    assert "'bravo' is about to stand within 1.5 m of [3.4, 0.0, 0.0]" in on_bravo.payload
    assert "turn its `placement.auto` on" in on_alpha.payload
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

    assert decision.accepted
    assert io._placements["alpha"] == ORIGIN


def test_a_copy_admitted_while_its_robot_leaves_is_stood_afresh():
    """Taking a robot out waits for the thread that steps the scene, which is
    a whole stage resolved, and a copy that comes straight back attaches
    inside that window. Everything the robot's joining held goes back before
    the stage is asked to let it go, so the copy is admitted under its own
    name, on the spot its robot is standing on, and its own stay stands it
    there."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")
    joined = []

    def attach_the_copy(_change):
        # The take-out reaches the thread that steps the scene here, and
        # the copy attaches while it waits for the scene to be resolved.
        joined.append(io._admit(_request(instance="alpha_init_2")))
        return _done()

    io._edits = SimpleNamespace(submit=attach_the_copy)

    asyncio.run(io._take_out(_Answers(), "alpha", Ending.LEFT, "the robot left the scene"))

    assert [decision.accepted for decision in joined] == [True]
    copy = io._robots.of_name("alpha")
    assert copy.caller.instance_id == "alpha_init_2"
    assert not copy.standing()
    assert io._placements["alpha"] == ORIGIN


def test_a_goal_that_never_reached_its_stay_gives_back_its_name_and_its_spot():
    """A goal is accepted on a hop of its own after this engine has answered
    for it, and a robot with no stay has nothing watching its lease: what its
    admission reserved is held by nobody, and the copy that comes after it
    takes the name."""
    io = _robots_io(_stage())

    assert io._admit(_request()).accepted
    io._withdraw()

    assert io._robots.of_name("alpha") is None
    assert io._placements == {}
    assert io._admit(_request(instance="alpha_init_2")).accepted


def test_withdrawing_again_leaves_the_copy_that_took_the_name_alone():
    """Serving the attaches withdraws whenever it fails and goes back to
    waiting, so it withdraws again having admitted nothing in between. One
    admission is given back once: the name is the next copy's by then."""
    io = _robots_io(_stage())
    assert io._admit(_request()).accepted
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
    accepted. A goal that never arrives has nothing watching its lease, so
    serving the attaches gives its name and its spot back."""
    io = _robots_io(_stage())
    handle = _Hop(_request())

    asyncio.run(io._serve_attach(handle))

    assert [decision.accepted for decision in handle.decisions] == [True]
    assert io._robots.of_name("alpha") is None
    assert io._placements == {}


def test_a_handover_that_never_reached_its_stay_leaves_the_robot_standing():
    """A goal admitted for the robot its own copy stands reserves nothing:
    the robot stays in the scene, on the spot it stands on, when that goal
    never reaches its stay."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")
    promised = io._placements["alpha"]

    assert io._admit(_request()).accepted
    io._withdraw()

    assert io._robots.of_name("alpha").standing()
    assert io._placements["alpha"] == promised


def test_a_copy_re_registering_its_own_robot_takes_it_over_where_it_stands():
    """Its initializer died and came back: the same copy attaching the robot
    it stands is accepted and the robot stays exactly as it is. Admitting it
    reserves nothing and ends nobody, so the robot keeps the stay hosting it
    until the stay taking it over exists."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")
    hosted = io._handover("alpha")

    decision = io._admit(_request())

    assert decision.accepted
    assert decision.payload.arm_names == ["left_arm", "right_arm"]
    assert io._admitted is None
    assert not hosted.is_set()
    assert list(io._robots.standing()) == ["alpha"]


def _built(lease_s: float = 2.0) -> RobotsIO:
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
        lease_s,
    )


def test_a_robots_io_reads_each_part_as_the_launch_gives_it():
    """Its registry and its stage are two arguments of one type each, and a
    launch passes them positionally: swapped, the engine reads every robot
    off the stage and every spot off the registry."""
    io = _built()

    assert isinstance(io._robots, Registry)
    assert isinstance(io._world, World)
    assert io._models is MODELS


def test_a_lease_no_robot_can_keep_is_refused_as_the_engine_is_built():
    """The lease is how long a robot whose pairs are gone stays, so one that
    is zero or below takes every robot off the stage as it stands."""
    with pytest.raises(ValueError, match="robot_lease_ms must be positive"):
        _built(lease_s=0.0)


def test_another_caller_naming_the_standing_robot_is_refused():
    """The name is the sharper answer than the engine's capacity: it names
    the instance whose robot stands under it."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert io._admit(_request()).accepted

    decision = io._admit(_request(instance="other_init_inst"))

    assert not decision.accepted
    assert "already stands as 'alpha'" in decision.payload


def test_a_copy_re_registering_as_another_model_is_refused():
    """A robot is the model it attached as, so the registry answers for the
    name and the goal is told."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    assert io._admit(_request()).accepted

    decision = io._admit(_request(model="so101"))

    assert not decision.accepted
    assert "alpha" in decision.payload


def test_a_goal_handed_over_leaves_its_robot_standing():
    """A copy whose initializer came back takes its own robot over: the goal
    that hosted it ends without asking the stage for anything."""
    io = _robots_io(_stage(("alpha", "openarm_v2", ORIGIN)))
    io._edits = Mock(spec=["submit"])
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.HANDED_OVER, "another goal hosts it"))

    io._edits.submit.assert_not_called()
    assert goal.answers == [("completed", False, "another goal hosts it")]


def _lease_io() -> RobotsIO:
    """A RobotsIO with only the parts a lease and a forget touch."""
    io = RobotsIO.__new__(RobotsIO)
    io._placements = {}
    io._handovers = {}
    return io


class TestHowOftenAStayChecksItsLease:
    """A stay wakes on a share of its lease so a robot whose pairs are gone
    leaves within it, and no faster than the floor, because every wake is a
    turn of the node loop the whole engine shares."""

    def test_the_period_is_a_share_of_the_lease(self):
        io = _lease_io()
        io._lease_s = 8.0

        assert io._lease_check_period() == 8.0 * 0.25

    def test_a_short_lease_is_checked_no_faster_than_the_floor(self):
        """`robot_lease_ms` takes any positive number, and a lease of one
        millisecond would wake the node loop a thousand times a second."""
        io = _lease_io()
        io._lease_s = 0.001

        assert io._lease_check_period() == 0.05

    def test_every_lease_is_checked_inside_itself(self):
        io = _lease_io()
        for lease_s in (0.001, 0.05, 1.0, 8.0, 600.0):
            io._lease_s = lease_s
            assert 0.0 < io._lease_check_period()
            assert io._lease_check_period() <= max(lease_s, 0.05)


def test_forgetting_a_robot_drops_its_spot_and_its_signal():
    """What a robot's joining held is a spot promised and a signal its own
    goal waits on, and a name given back leaves neither behind."""
    io = _lease_io()
    io._placements["alpha"] = Placement.of((0.0, 0.0, 0.0), 0.0)
    io._handover("alpha")

    io._forget("alpha")

    assert io._placements == {}
    assert io._handovers == {}


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
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")
    io._lease_s = 2.0
    io._io = _HoldsEverything()
    io._stopping = asyncio.Event()
    hosted = io._handover("alpha")
    goal = _Hosting()

    async def take_over() -> tuple:
        stay = asyncio.create_task(io._stay(goal))
        # Enough turns for the stay to end the goal it took the robot from
        # and to settle into watching the robot's lease.
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


def test_a_placement_the_stage_cannot_stand_a_robot_on_is_refused():
    """A prim placed at NaN reports no error and simulates nothing, so the
    placement is refused as the robot is admitted and its name goes back with
    the refusal."""
    io = _robots_io(_stage())

    decision = io._admit(_request(placement=_asks_for(float("nan"))))

    assert not decision.accepted
    assert "a placement is finite in every coordinate" in decision.payload
    assert io._robots.robots() == []
    assert io._placements == {}
