"""Standing a robot on the stage, what happens when it cannot be stood, and
how a robot leaves it.

Standing lets go of the stage before it changes it: the views stop reading and
the timeline stops. Whatever the change does, the stage has to be taken up
again, or the robots already on it are frozen for good. A robot leaves when
its goal is cancelled, or when its pairs have not been its model's for the
lease; the stage stands many robots, of different models, and each is held to
its own.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sim_robot_core.pairs import Held
from sim_robot_core.registry import Caller, Registry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime robots_io imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from edits import Edits  # noqa: E402  pylint: disable=C0413
from isaac_models import IsaacModels  # noqa: E402  pylint: disable=C0413
from robots_io import Ending, RobotsIO  # noqa: E402  pylint: disable=C0413

MODELS = IsaacModels.read()
OPENARM_V1 = MODELS.of("openarm_v1")
OPENARM_V2 = MODELS.of("openarm_v2")
SO101 = MODELS.of("so101")


def _robots_io(world):
    """A RobotsIO with only the parts standing a robot touches."""
    io = RobotsIO.__new__(RobotsIO)
    io._world = world
    io._robots = Mock(spec=["renew", "release"])
    io._io = Mock(spec=["forget"])
    io._loop = Mock()
    io._loop.time.return_value = 0.0
    io._bind = Mock()
    io._unbind_stage = Mock()
    io.binds_with(io._bind, io._unbind_stage)
    return io


def test_a_robot_that_stands_leaves_the_stage_taken_up():
    world = Mock(spec=["add", "remove"])
    io = _robots_io(world)

    spot = Mock()
    io.stand("alpha", SO101, spot)

    # The stage is handed the model the robot joined as.
    world.add.assert_called_once_with("alpha", SO101, spot)
    io._bind.assert_called_once_with()
    io._unbind_stage.assert_called_once_with()


def test_a_robot_the_stage_refuses_leaves_it_taken_up_anyway():
    """A duplicate name, a model with no stage, or any USD error raises out of
    World.add, after the stage has already been let go of."""
    world = Mock(spec=["add", "remove"])
    world.add.side_effect = ValueError("alpha already stands in the stage")
    io = _robots_io(world)

    with pytest.raises(ValueError, match="already stands"):
        io.stand("alpha", OPENARM_V2, Mock())

    io._bind.assert_called_once_with()
    world.remove.assert_not_called()


def test_a_robot_the_engine_cannot_resolve_is_taken_back_out():
    world = Mock(spec=["add", "remove"])
    io = _robots_io(world)
    io._bind.side_effect = [RuntimeError("joints are not the ones driven"), None]

    with pytest.raises(RuntimeError, match="joints"):
        io.stand("bravo", OPENARM_V1, Mock())

    world.remove.assert_called_once_with("bravo")
    assert io._bind.call_count == 2


def test_a_robot_that_cannot_be_taken_out_leaves_the_stage_taken_up():
    world = Mock(spec=["add", "remove"])
    world.remove.side_effect = KeyError("charlie")
    io = _robots_io(world)

    with pytest.raises(KeyError):
        io.unstand("charlie")

    io._bind.assert_called_once_with()


def test_taking_a_robot_out_is_all_the_stage_is_asked_for():
    """What a robot's joining held went back when its stay ended, before the
    stage was asked to let the robot go, so a copy that comes straight back
    finds its name free and is stood afresh."""
    world = Mock(spec=["add", "remove"])
    io = _robots_io(world)
    order = []
    world.remove.side_effect = lambda *_: order.append("left the stage")
    io._io.forget.side_effect = lambda *_: order.append("setpoints dropped")

    io._rebind = Mock(side_effect=lambda: order.append("scene resolved"))

    io.unstand("alpha")

    assert order == ["left the stage", "setpoints dropped", "scene resolved"]
    io._robots.release.assert_not_called()


def test_a_removal_that_fails_takes_the_stage_back_up():
    """The stage is let go of inside the change that asks for it, so a robot
    that cannot be taken out leaves the robots beside it stepping."""
    world = Mock(spec=["add", "remove"])
    world.remove.side_effect = KeyError("alpha")
    io = _robots_io(world)
    resolved = []
    io._rebind = Mock(side_effect=lambda: resolved.append("scene resolved"))

    with pytest.raises(KeyError):
        io.unstand("alpha")

    assert resolved == ["scene resolved"]


class _Goal:
    """The parts of an attach goal a robot's stand reads and answers."""

    def __init__(self, request=None):
        self.leave = asyncio.Event()
        self.answers = []
        self.feedback = []
        self._request = request

    def request(self):
        return self._request

    async def cancel_signal(self):
        await self.leave.wait()

    async def publish_feedback(self, standing):
        self.feedback.append(standing)

    async def complete(self, success, message):
        self.answers.append(("completed", success, message))

    async def complete_cancelled(self, success, message):
        self.answers.append(("cancelled", success, message))


def _joining_io():
    """A RobotsIO with one robot admitted and waiting for the thread that
    steps the scene, which takes nothing up until a test drains the edits."""
    io = RobotsIO.__new__(RobotsIO)
    io._edits = Edits()
    io._placements = {"alpha": Mock()}
    io._handovers = {}
    io._robots = Mock(spec=["release", "stand"])
    io._models = MODELS
    io._loop = Mock()
    io._loop.time.return_value = 0.0
    return io


def test_a_stand_whose_spot_was_taken_back_gives_the_name_back():
    """Whatever took the spot back took the name with it, and this goal is
    the one holding it: a name nothing gives back is a name no copy can
    ever join under again."""
    io = _joining_io()
    io._placements = {}
    goal = _Goal()

    assert asyncio.run(io._stand(goal, "alpha", "openarm_v2")) is False

    io._robots.release.assert_called_once_with("alpha")
    assert goal.answers == [
        ("completed", False, "the name was given back before the robot stood")
    ]


def test_a_stay_whose_name_went_back_before_it_began_answers_its_own_goal():
    """The stay reads the robot it was admitted for, and a name given back
    under it leaves nothing to stand."""
    io = _joining_io()
    io._robots = Mock(spec=["release", "stand", "of_name"])
    io._robots.of_name.return_value = None
    goal = _Goal(_request("alpha", "openarm_v2"))

    asyncio.run(io._stay(goal))

    assert goal.answers == [
        ("completed", False, "the name was given back before the robot stood")
    ]


def test_a_robot_that_leaves_before_the_stage_takes_it_up_never_stands():
    """Its stand still waits for the thread that steps the scene, so the
    stand is withdrawn and the name given back at once: a copy removed and
    joined straight back finds its name free."""
    io = _joining_io()

    async def leave_while_waiting():
        goal = _Goal()
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        goal.leave.set()
        return goal, await asyncio.wait_for(standing, 5)

    goal, stood = asyncio.run(leave_while_waiting())

    assert stood is False
    assert goal.answers == [("cancelled", True, "the robot left before it stood")]
    io._robots.release.assert_called_once_with("alpha")
    io._robots.stand.assert_not_called()
    assert "alpha" not in io._placements
    assert io._edits.drain() == 0


def test_a_robot_that_leaves_while_the_stage_stands_it_is_stood_first():
    """A stand the thread has taken up runs to its end; the stay that follows
    sees the robot leave and takes it out."""
    io = _joining_io()
    taken_up, finish = threading.Event(), threading.Event()

    def stand(*_):
        taken_up.set()
        finish.wait(5)

    io.stand = stand

    async def leave_while_standing():
        goal = _Goal()
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        stage = threading.Thread(target=io._edits.drain)
        stage.start()
        await asyncio.get_running_loop().run_in_executor(None, taken_up.wait, 5)
        goal.leave.set()
        await asyncio.sleep(0.05)
        finish.set()
        stood = await asyncio.wait_for(standing, 5)
        stage.join(5)
        return goal, stood

    goal, stood = asyncio.run(leave_while_standing())

    assert stood is True
    assert goal.answers == []
    assert goal.feedback == [], "the stay tells the holder, once, that it stands"
    io._robots.stand.assert_called_once_with("alpha", 0.0)
    io._robots.release.assert_not_called()


@pytest.mark.parametrize("model", ["openarm_v1", "openarm_v2", "so101"])
def test_a_stand_asks_the_stage_for_the_robots_own_model(model):
    io = _joining_io()
    asked = []
    io.stand = lambda name, known, placement: asked.append((name, known, placement))

    async def stand_once():
        goal = _Goal()
        standing = asyncio.create_task(io._stand(goal, "alpha", model))
        await asyncio.sleep(0)
        io._edits.drain()
        return await asyncio.wait_for(standing, 5)

    assert asyncio.run(stand_once()) is True
    assert asked == [("alpha", MODELS.of(model), io._placements["alpha"])]


def _done():
    """A future the thread that steps the scene has already answered."""
    future = concurrent.futures.Future()
    future.set_result(None)
    return future


class _UnreachableGoal(_Goal):
    """A goal whose holder cannot be told anything."""

    async def publish_feedback(self, standing):
        raise RuntimeError("the holder is away")


def test_a_robot_whose_holder_cannot_be_told_it_stands_is_taken_out():
    io = _joining_io()
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = _done()
    goal = _UnreachableGoal()

    assert asyncio.run(io._standing(goal, "alpha")) is False

    io._edits.submit.assert_called_once()
    assert goal.answers == [
        ("completed", False, "the robot's holder could not be told it stands: the holder is away")
    ]


def test_a_stay_ending_with_the_engine_asks_nothing_of_the_stopping_stage():
    io = _joining_io()
    io._edits = Mock(spec=["submit"])
    goal = _Goal()

    asyncio.run(io._take_out(goal, "alpha", Ending.STOPPED, "the engine stopped"))

    io._edits.submit.assert_not_called()
    io._robots.release.assert_called_once_with("alpha")
    assert "alpha" not in io._placements
    assert goal.answers == [("completed", False, "the engine stopped")]


def test_a_stay_that_lapsed_takes_the_robot_off_the_stage_first():
    io = _joining_io()
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = _done()
    goal = _Goal()

    asyncio.run(io._take_out(goal, "alpha", Ending.LAPSED, "its pairs were gone for the lease"))

    io._edits.submit.assert_called_once()
    assert goal.answers == [("completed", False, "its pairs were gone for the lease")]


# What a robot holds once every limb of its model is paired.
OPENARM_LIMBS = Held(
    arms=frozenset({"left_arm", "right_arm"}),
    grippers=frozenset({"left_gripper", "right_gripper"}),
)
SO101_LIMBS = Held(arms=frozenset({"arm"}), grippers=frozenset({"gripper"}))

# The lease every leasing test gives, in the words a stay's ending says it.
LEASE_S = 2.0
LEASE = "2.0s, the lease this scene gives"
# What each model's side of a mismatch names.
OPENARM_V2_HAS = (
    "the model openarm_v2 has arms ['left_arm', 'right_arm'], "
    "grippers ['left_gripper', 'right_gripper'], rgb_cameras ['wrist_left', 'wrist_right'], "
    "rgbd_cameras ['chest']"
)
SO101_HAS = "the model so101 has arms ['arm'], grippers ['gripper'], rgb_cameras ['front'], rgbd_cameras []"


class _Pairs:
    """What each robot holds on the four slots, as the engine reads its
    pairs."""

    def __init__(self, held: dict) -> None:
        self.held = dict(held)

    def held_by(self, robot: str) -> Held:
        return self.held.get(robot, Held())


def _caller(robot: str) -> Caller:
    return Caller(core_node="sim16", instance_id=f"{robot}_init_inst")


def _leasing(now_s: float, fleet: dict) -> RobotsIO:
    """A RobotsIO whose robots, each given as name: (model, what it holds),
    last renewed their leases at 0 s, read at `now_s` with the lease this
    scene gives."""
    io = RobotsIO.__new__(RobotsIO)
    io._models = MODELS
    io._robots = Registry()
    for name, (model, _held) in fleet.items():
        io._robots.admit(name, MODELS.of(model).entry, _caller(name), 0.0)
    io._loop = SimpleNamespace(time=lambda: now_s)
    io._io = _Pairs({name: held for name, (_model, held) in fleet.items()})
    io._lease_s = LEASE_S
    io._handovers = {}
    return io


class _Staying(_Goal):
    """A goal whose holder never cancels."""

    async def cancel_signal(self):
        await asyncio.Event().wait()


class _Withdrawing(_Goal):
    """A goal whose holder lets go while its robot still waits for the
    thread that steps the scene."""

    async def cancel_signal(self):
        return None


def test_a_robot_that_left_before_it_stood_gives_back_its_name_and_its_spot():
    """Its stand still waits for the thread that steps the scene, so the
    stand is withdrawn: nothing stood, and the next copy takes the name."""
    io = _joining_io()
    io._io = Mock(spec=["forget"])

    stood = asyncio.run(io._stand(_Withdrawing(), "alpha", "openarm_v2"))

    assert stood is False
    io._robots.release.assert_called_once_with("alpha")
    assert io._placements == {}


def test_a_robot_the_stage_refused_gives_back_its_name_and_its_spot():
    """The stage answered the stand with why it will not take the robot, so
    what the admission held goes back with the answer."""
    io = _joining_io()
    io._io = Mock(spec=["forget"])
    io._robots = Mock(spec=["release", "stand", "renew"])
    io._unbind = lambda: None
    io._rebind = lambda: None
    io._world = Mock(spec=["add", "remove"])
    io._world.add.side_effect = RuntimeError("the stage will not take it")
    goal = _Goal()

    async def stand_and_drain():
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        io._edits.drain()
        return await standing

    assert asyncio.run(stand_and_drain()) is False
    io._robots.release.assert_called_once_with("alpha")
    assert io._placements == {}
    assert "the stage will not take it" in goal.answers[0][2]


def test_a_robot_that_stood_is_recorded_as_standing():
    """What marks it standing is what tells a copy re-registering that its
    own robot is there to be handed over."""
    io = _joining_io()
    io._io = Mock(spec=["forget"])
    io._robots = Mock(spec=["release", "stand", "renew"])
    io._unbind = lambda: None
    io._rebind = lambda: None
    io._world = Mock(spec=["add", "remove"])
    goal = _Goal()

    async def stand_and_drain():
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        io._edits.drain()
        return await standing

    assert asyncio.run(stand_and_drain()) is True
    io._robots.stand.assert_called_once_with("alpha", 0.0)


def test_a_stopping_engine_ends_every_stay_and_every_loop():
    """A robot whose engine is going down is told so on its own goal, and
    nothing of this engine is left running behind it."""
    io = RobotsIO.__new__(RobotsIO)
    io._stopping = asyncio.Event()

    async def run_and_stop():
        io._tasks = [asyncio.create_task(asyncio.Event().wait())]
        io._stays = {asyncio.create_task(asyncio.Event().wait())}
        watched = list(io._tasks) + list(io._stays)
        await io.stop()
        return watched

    watched = asyncio.run(run_and_stop())

    assert io._stopping.is_set()
    assert all(task.cancelled() for task in watched)


class _Leaving(_Goal):
    """A goal whose holder lets go the moment the stay waits on it."""

    def __init__(self):
        super().__init__()
        self.let_go = asyncio.Event()

    async def cancel_signal(self):
        await self.let_go.wait()


class TestHowAStayEnds:
    """A stay waits for one of the ways it can end and says which it was.
    Every one of them answers the goal differently, and the answer is what
    the copy holding the robot reads."""

    @staticmethod
    def _watching() -> RobotsIO:
        io = _leasing(now_s=0.0, fleet={"alpha": ("openarm_v2", OPENARM_LIMBS)})
        io._stopping = asyncio.Event()
        return io

    def test_a_robot_handed_over_ends_its_stay_where_it_stands(self):
        """The signal the goal taking the robot over sets is what this stay
        is waiting on."""
        io = self._watching()
        handover = io._handover("alpha")
        handover.set()

        ending, why = asyncio.run(io._watch(_Leaving(), "alpha", handover))

        assert ending is Ending.HANDED_OVER
        assert why == "this robot is hosted by another goal of this copy"

    def test_a_holder_that_lets_go_ends_its_stay(self):
        """`peppy stack remove` cancels the goal, which is how a copy takes
        its robot off the stage."""
        io = self._watching()
        goal = _Leaving()

        async def let_go():
            goal.let_go.set()
            return await io._watch(goal, "alpha", io._handover("alpha"))

        ending, why = asyncio.run(let_go())

        assert ending is Ending.LEFT
        assert why == "the robot left the scene"

    def test_a_stopping_engine_ends_every_stay(self):
        io = self._watching()
        io._stopping.set()
        io._lease_check_period = lambda: 0.0

        ending, why = asyncio.run(io._watch(_Leaving(), "alpha", io._handover("alpha")))

        assert ending is Ending.STOPPED
        assert why == "the engine stopped"

    def test_a_name_given_back_under_a_stay_ends_it(self):
        """Whatever gave the name back took the robot with it, so the stay
        has nothing left to watch."""
        io = self._watching()
        io._robots.release("alpha")
        io._lease_check_period = lambda: 0.0

        ending, why = asyncio.run(io._watch(_Leaving(), "alpha", io._handover("alpha")))

        assert ending is Ending.STOPPED
        assert why == "the name was given back"

    def test_a_stay_that_ends_leaves_nothing_waiting_behind_it(self):
        """Its two waiters are a task each, and a stay that ends without
        them leaks one per robot that ever stood."""
        io = self._watching()
        handover = io._handover("alpha")
        handover.set()

        async def watch_and_count():
            await io._watch(_Leaving(), "alpha", handover)
            await asyncio.sleep(0)
            watching = asyncio.current_task()
            return [
                task for task in asyncio.all_tasks() if task is not watching and not task.done()
            ]

        assert asyncio.run(watch_and_count()) == []


class TestLease:
    def test_a_robot_holding_its_models_pairs_keeps_its_place_however_stale_its_lease(self):
        """The watcher's tick renews a lease from the pairs it reads, and a
        node loop the stage kept busy leaves the lease stale: a robot still
        holding its pairs is not taken out for that."""
        io = _leasing(now_s=10.0, fleet={"alpha": ("openarm_v2", OPENARM_LIMBS)})
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s == 10.0

    def test_a_camera_left_unpaired_does_not_end_a_stay(self):
        """A camera nobody views is not rendered, so an OpenArm v2 holding
        its limbs and one of its three cameras holds what its model asks."""
        held = Held(
            arms=OPENARM_LIMBS.arms,
            grippers=OPENARM_LIMBS.grippers,
            rgb_cameras=frozenset({"wrist_left"}),
        )
        io = _leasing(now_s=10.0, fleet={"alpha": ("openarm_v2", held)})
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s == 10.0

    def test_a_robot_holding_no_pair_for_the_lease_lapses(self):
        io = _leasing(now_s=10.0, fleet={"alpha": ("openarm_v2", Held())})

        assert io._lapse(io._robots.of_name("alpha")) == (
            f"none of this robot's limbs were paired for {LEASE}"
        )

    def test_a_robot_has_the_lease_to_pair_its_limbs(self):
        io = _leasing(now_s=LEASE_S, fleet={"alpha": ("openarm_v2", Held())})
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s == 0.0, "a lease still running is not a renewed one"

    def test_a_robot_missing_a_limb_lapses_naming_both_lists(self):
        held = Held(arms=OPENARM_LIMBS.arms, grippers=frozenset({"left_gripper"}))
        io = _leasing(now_s=10.0, fleet={"alpha": ("openarm_v2", held)})

        assert io._lapse(io._robots.of_name("alpha")) == (
            f"this robot's pairs were not its model's for {LEASE}: its pairs name "
            "arms ['left_arm', 'right_arm'], grippers ['left_gripper'], rgb_cameras [], "
            f"rgbd_cameras [], and {OPENARM_V2_HAS}"
        )

    def test_a_robot_streaming_a_camera_its_model_lacks_lapses_naming_both_lists(self):
        held = Held(
            arms=SO101_LIMBS.arms, grippers=SO101_LIMBS.grippers, rgbd_cameras=frozenset({"chest"})
        )
        io = _leasing(now_s=10.0, fleet={"charlo": ("so101", held)})

        assert io._lapse(io._robots.of_name("charlo")) == (
            f"this robot's pairs were not its model's for {LEASE}: its pairs name "
            "arms ['arm'], grippers ['gripper'], rgb_cameras [], rgbd_cameras ['chest'], "
            f"and {SO101_HAS}"
        )

    def test_each_robot_of_a_fleet_is_held_to_its_own_model(self):
        """An SO-101 stands beside an OpenArm on the same four slots. Each
        holds its own model's limbs, and neither is asked for the other's."""
        io = _leasing(
            now_s=10.0,
            fleet={"alpha": ("openarm_v2", OPENARM_LIMBS), "charlo": ("so101", SO101_LIMBS)},
        )

        for name in ("alpha", "charlo"):
            robot = io._robots.of_name(name)
            assert io._lapse(robot) is None
            assert robot.last_paired_s == 10.0

    def test_one_robots_mismatch_ends_its_own_stay_and_no_others(self):
        """The SO-101's backbone leads an OpenArm's limbs: its lease runs
        out naming both lists while the OpenArm beside it keeps its place."""
        io = _leasing(
            now_s=10.0,
            fleet={"alpha": ("openarm_v2", OPENARM_LIMBS), "charlo": ("so101", OPENARM_LIMBS)},
        )
        # The lease is read at once instead of after a share of it.
        io._lease_check_period = lambda: 0.0
        io._stopping = threading.Event()

        ending, why = asyncio.run(io._watch(_Staying(), "charlo", io._handover("charlo")))

        assert ending is Ending.LAPSED
        assert why == (
            f"this robot's pairs were not its model's for {LEASE}: its pairs name "
            "arms ['left_arm', 'right_arm'], grippers ['left_gripper', 'right_gripper'], "
            f"rgb_cameras [], rgbd_cameras [], and {SO101_HAS}"
        )
        assert io._robots.of_name("charlo").last_paired_s == 0.0
        assert io._lapse(io._robots.of_name("alpha")) is None

    def test_the_watcher_renews_each_lease_only_while_its_robots_pairs_are_its_models(self):
        io = _leasing(
            now_s=10.0,
            fleet={
                "alpha": ("openarm_v2", OPENARM_LIMBS),
                "bravo": ("openarm_v2", Held(arms=OPENARM_LIMBS.arms)),
                "charlo": ("so101", SO101_LIMBS),
                "delta": ("so101", OPENARM_LIMBS),
                "echo": ("openarm_v1", Held()),
            },
        )

        async def one_tick() -> None:
            watcher = asyncio.create_task(io._watch_pairs())
            # One turn of the loop runs the watcher up to the pause between
            # two ticks.
            await asyncio.sleep(0)
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

        asyncio.run(one_tick())

        renewed = {robot.name: robot.last_paired_s for robot in io._robots.robots()}
        assert renewed == {"alpha": 10.0, "bravo": 0.0, "charlo": 10.0, "delta": 0.0, "echo": 0.0}


class TestReadiness:
    @staticmethod
    def _ready(io: RobotsIO, robot: str) -> bool:
        caller = _caller(robot)
        request = SimpleNamespace(core_node=caller.core_node, instance_id=caller.instance_id)
        return io._ready(request).ready

    def _standing(self, fleet: dict) -> RobotsIO:
        io = _leasing(now_s=1.0, fleet=fleet)
        for name in fleet:
            io._robots.stand(name, now_s=1.0)
        return io

    def test_a_robot_holding_every_limb_of_its_model_is_ready(self):
        io = self._standing({"alpha": ("openarm_v2", OPENARM_LIMBS)})
        assert self._ready(io, "alpha")

    def test_a_one_arm_robot_is_ready_with_its_one_arm_and_its_one_gripper(self):
        io = self._standing({"charlo": ("so101", SO101_LIMBS)})
        assert self._ready(io, "charlo")

    def test_a_robot_missing_one_limb_is_never_ready(self):
        for missing in ("left_arm", "right_arm"):
            held = Held(arms=OPENARM_LIMBS.arms - {missing}, grippers=OPENARM_LIMBS.grippers)
            assert not self._ready(self._standing({"alpha": ("openarm_v2", held)}), "alpha")
        for missing in ("left_gripper", "right_gripper"):
            held = Held(arms=OPENARM_LIMBS.arms, grippers=OPENARM_LIMBS.grippers - {missing})
            assert not self._ready(self._standing({"alpha": ("openarm_v2", held)}), "alpha")
        jawless = Held(arms=SO101_LIMBS.arms)
        assert not self._ready(self._standing({"charlo": ("so101", jawless)}), "charlo")

    def test_a_camera_left_unpaired_does_not_hold_readiness_back(self):
        io = self._standing({"charlo": ("so101", SO101_LIMBS)})
        assert self._ready(io, "charlo")

    def test_each_robot_of_a_fleet_is_judged_on_its_own_pairs_and_its_own_model(self):
        io = self._standing(
            {
                "alpha": ("openarm_v2", Held(arms=OPENARM_LIMBS.arms)),
                "charlo": ("so101", SO101_LIMBS),
            }
        )

        assert not self._ready(io, "alpha")
        assert self._ready(io, "charlo")

    def test_a_robot_that_does_not_stand_yet_is_not_ready(self):
        io = _leasing(now_s=1.0, fleet={"alpha": ("openarm_v2", OPENARM_LIMBS)})
        assert not self._ready(io, "alpha")

    def test_an_instance_with_no_robot_is_not_ready(self):
        io = self._standing({"alpha": ("openarm_v2", OPENARM_LIMBS)})
        assert not self._ready(io, "stranger")


class _Spots:
    """The stage's spots, every one of them free."""

    def free_spot(self, promised=(), leaving=None):
        return SimpleNamespace(position=(1.5 * len(promised), 0.0, 0.0), yaw=0.0)


def _admitting() -> RobotsIO:
    """A RobotsIO with only the parts admitting a robot touches, over the
    models this engine ships an entry for."""
    io = RobotsIO.__new__(RobotsIO)
    io._models = MODELS
    io._robots = Registry()
    io._world = _Spots()
    io._loop = SimpleNamespace(time=lambda: 0.0)
    io._handovers = {}
    io._placements = {}
    return io


def _request(robot: str, model: str):
    caller = _caller(robot)
    return SimpleNamespace(
        core_node=caller.core_node,
        instance_id=caller.instance_id,
        data=SimpleNamespace(robot=robot, model=model, placement=None),
    )


class TestAdmission:
    @pytest.mark.parametrize(
        ("model", "arm_names", "arm_joints", "gripper_names"),
        [
            ("openarm_v1", ["left_arm", "right_arm"], [7, 7], ["left_gripper", "right_gripper"]),
            ("openarm_v2", ["left_arm", "right_arm"], [7, 7], ["left_gripper", "right_gripper"]),
            ("so101", ["arm"], [5], ["gripper"]),
        ],
    )
    def test_a_robot_is_admitted_with_the_limbs_of_its_model_under_its_robots_names(
        self, model, arm_names, arm_joints, gripper_names
    ):
        io = _admitting()

        decision = io._admit(_request("alpha", model))

        assert decision.accepted
        answer = decision.payload
        assert (answer.arm_names, answer.arm_joints, answer.gripper_names) == (
            arm_names,
            arm_joints,
            gripper_names,
        )
        assert io._robots.of_name("alpha").model == model

    def test_a_model_the_engine_has_no_entry_for_is_refused_with_the_ones_it_stands(self):
        io = _admitting()

        decision = io._admit(_request("alpha", "openarm_v9"))

        assert not decision.accepted
        assert decision.payload == (
            "unknown model 'openarm_v9': this engine stands openarm_v1, openarm_v2, so101"
        )
        assert io._robots.robots() == []
        assert io._placements == {}

    def test_an_so101_is_admitted_beside_an_openarm_each_on_a_spot_of_its_own(self):
        io = _admitting()

        assert io._admit(_request("alpha", "openarm_v2")).accepted
        assert io._admit(_request("charlo", "so101")).accepted

        assert {robot.name: robot.model for robot in io._robots.robots()} == {
            "alpha": "openarm_v2",
            "charlo": "so101",
        }
        spots = [placement.position for placement in io._placements.values()]
        assert len(set(spots)) == 2
