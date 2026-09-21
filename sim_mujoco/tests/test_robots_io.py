"""Who may join this engine's scene: any robot of a model the engine has an
entry for, standing on the spot it asked for or one of the engine's own,
staying for as long as its pairs are its model's, and ready once it holds
every limb."""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
from sim_robot_core.pairs import Held
from sim_robot_core.registry import Caller, Registry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime robots_io imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from mujoco_models import MujocoModels  # noqa: E402  pylint: disable=C0413
from robots_io import Ending, RobotsIO  # noqa: E402  pylint: disable=C0413
from edits import Edits  # noqa: E402  pylint: disable=C0413
from world import Placement, World  # noqa: E402  pylint: disable=C0413

MODELS = MujocoModels.read()
CALLER = Caller(core_node="sim16", instance_id="alpha_init_inst")

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


def _robots_io() -> RobotsIO:
    """A RobotsIO with only the parts admitting a robot touches, over the
    models this engine ships an entry for."""
    io = RobotsIO.__new__(RobotsIO)
    io._models = MODELS
    io._robots = Registry()
    io._world = World(head_camera_pack=None, renders=False)
    io._loop = SimpleNamespace(time=lambda: 0.0)
    io._handovers = {}
    io._placements = {}
    io._admitted = None
    return io


def _request(robot: str = "alpha", model: str = "openarm_v2", placement=None, instance: str = "alpha_init_inst"):
    return SimpleNamespace(
        core_node="sim16",
        instance_id=instance,
        data=SimpleNamespace(robot=robot, model=model, placement=placement),
    )


@pytest.mark.parametrize(
    ("model", "arm_names", "arm_joints", "gripper_names"),
    [
        ("openarm_v1", ["left_arm", "right_arm"], [7, 7], ["left_gripper", "right_gripper"]),
        ("openarm_v2", ["left_arm", "right_arm"], [7, 7], ["left_gripper", "right_gripper"]),
        ("so101", ["arm"], [5], ["gripper"]),
    ],
)
def test_a_robot_of_a_model_the_engine_stands_is_admitted_with_its_limbs(
    model, arm_names, arm_joints, gripper_names
):
    io = _robots_io()
    decision = io._admit(_request(model=model))
    assert decision.accepted
    assert (decision.payload.arm_names, decision.payload.arm_joints, decision.payload.gripper_names) == (
        arm_names,
        arm_joints,
        gripper_names,
    )
    assert io._robots.of_name("alpha").model == model


def test_a_model_the_engine_has_no_entry_for_is_refused_with_the_ones_it_stands():
    io = _robots_io()
    decision = io._admit(_request(model="openarm_v9"))
    assert not decision.accepted
    assert decision.payload == (
        "unknown model 'openarm_v9': this engine stands openarm_v1, openarm_v2, so101"
    )
    assert io._robots.robots() == []


def test_a_robot_that_asks_for_no_spot_is_promised_one_of_the_engines_own():
    io = _robots_io()

    assert io._admit(_request()).accepted

    assert io._placements["alpha"] == Placement.of((0.0, 0.0, 0.0), 0.0)


def test_a_robot_stands_on_the_spot_it_asked_for():
    io = _robots_io()
    asked = SimpleNamespace(position=[1.0, -2.0, 0.0], yaw=0.5)

    assert io._admit(_request(placement=asked)).accepted

    assert io._placements["alpha"] == Placement.of((1.0, -2.0, 0.0), 0.5)


def test_a_spot_another_robot_was_promised_is_refused_with_what_to_ask_instead():
    io = _robots_io()
    assert io._admit(_request()).accepted

    taken = SimpleNamespace(position=[0.0, 0.0, 0.0], yaw=0.0)
    decision = io._admit(_request(robot="charlo", instance="charlo_init_inst", placement=taken))

    assert not decision.accepted
    assert "'alpha' is about to stand within 1.5 m of [0.0, 0.0, 0.0]" in decision.payload
    assert "turn its `placement.auto` on" in decision.payload
    # The name goes back with the refusal, so the copy may ask again.
    assert io._robots.of_name("charlo") is None


def test_a_placement_that_is_not_finite_is_refused():
    io = _robots_io()
    nowhere = SimpleNamespace(position=[float("nan"), 0.0, 0.0], yaw=0.0)

    decision = io._admit(_request(placement=nowhere))

    assert not decision.accepted
    assert "finite in every coordinate" in decision.payload
    assert io._robots.robots() == []


def test_a_second_robot_joins_beside_the_first_on_a_spot_of_its_own():
    io = _robots_io()
    assert io._admit(_request()).accepted

    decision = io._admit(_request(robot="charlo", model="so101", instance="charlo_init_inst"))

    assert decision.accepted
    assert {robot.name for robot in io._robots.robots()} == {"alpha", "charlo"}
    assert io._placements["charlo"] != io._placements["alpha"]


def test_the_name_is_free_again_once_the_robot_left():
    io = _robots_io()
    assert io._admit(_request()).accepted
    io._robots.release("alpha")
    assert io._admit(_request(robot="charlo", model="so101", instance="charlo_init_inst")).accepted


def test_a_robot_with_no_name_is_refused():
    decision = _robots_io()._admit(_request(robot=""))
    assert not decision.accepted
    assert "the copy it runs as" in decision.payload


def test_another_caller_naming_the_standing_robot_is_refused():
    """The name is the sharper answer than the engine's capacity: it names
    the instance whose robot stands under it."""
    io = _robots_io()
    assert io._admit(_request()).accepted
    decision = io._admit(_request(instance="other_init_inst"))
    assert not decision.accepted
    assert "already stands as 'alpha'" in decision.payload


def test_a_copy_re_registering_its_own_robot_is_admitted():
    """Its initializer died and came back: the same instance attaching the
    robot it stands is accepted, and the robot is never stood again."""
    io = _robots_io()
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")

    decision = io._admit(_request())

    assert decision.accepted
    assert decision.payload.arm_names == ["left_arm", "right_arm"]
    assert list(io._robots.standing()) == ["alpha"], "one entity, not two"
    assert io._handover("alpha").is_set() is False, "the new goal's own signal is fresh"


def test_a_re_registration_naming_another_model_is_refused():
    io = _robots_io()
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")
    decision = io._admit(_request(model="so101"))
    assert not decision.accepted
    assert "stands 'alpha' as openarm_v2" in decision.payload


def _built(lease_s: float = LEASE_S) -> RobotsIO:
    """A RobotsIO built the way the launch builds it, so the order its parts
    are given in is the order it reads them."""
    return RobotsIO(
        None,
        asyncio.new_event_loop(),
        MODELS,
        Registry(),
        World(head_camera_pack=None, renders=False),
        Edits(),
        Mock(spec=["forget", "held_by"]),
        lease_s,
    )


def test_a_robots_io_reads_each_part_as_the_launch_gives_it():
    """Its registry and its scene are two arguments of one type each, and a
    launch passes them positionally: swapped, the engine reads every robot
    off the scene and every spot off the registry."""
    io = _built()

    assert isinstance(io._robots, Registry)
    assert isinstance(io._world, World)
    assert io._models is MODELS


def test_a_lease_no_robot_can_keep_is_refused_as_the_engine_is_built():
    """The lease is how long a robot whose pairs are gone stays, so one that
    is zero or below takes every robot out as it stands."""
    with pytest.raises(ValueError, match="robot_lease_ms must be positive"):
        _built(lease_s=0.0)


class _Pairs:
    """What each robot holds on the four slots, as the engine reads its
    pairs."""

    def __init__(self, held: dict) -> None:
        self.held = dict(held)

    def held_by(self, robot: str) -> Held:
        return self.held.get(robot, Held())


def _caller(robot: str) -> Caller:
    return Caller(core_node="sim16", instance_id=f"{robot}_init_inst")


def _leasing(
    now_s: float, held: Held, model: str = "openarm_v2", reached_s: Optional[float] = 0.0
) -> RobotsIO:
    """A RobotsIO whose robot `alpha` of `model` holds `held`, read at `now_s`
    with the lease this scene gives. Its lease runs from `reached_s`, when a
    limb first reached it, and not at all when that is None."""
    return _leasing_fleet(now_s, {"alpha": (model, held)}, reached_s)


def _leasing_fleet(now_s: float, fleet: dict, reached_s: Optional[float] = 0.0) -> RobotsIO:
    """A RobotsIO whose robots, each given as name: (model, what it holds),
    are read at `now_s` with the lease this scene gives. Their leases run
    from `reached_s`, when a limb first reached each, and not at all when
    that is None."""
    io = _robots_io()
    for name, (model, _held) in fleet.items():
        io._robots.admit(name, MODELS.of(model).entry, _caller(name))
        if reached_s is not None:
            io._robots.note_limbs_reached(name, reached_s)
    io._loop = SimpleNamespace(time=lambda: now_s)
    io._io = _Pairs({name: held for name, (_model, held) in fleet.items()})
    io._lease_s = LEASE_S
    return io


class _Answers:
    """The answers a goal context takes."""

    def __init__(self):
        self.answers = []

    async def complete(self, success, message):
        self.answers.append(("completed", success, message))

    async def complete_cancelled(self, success, message):
        self.answers.append(("cancelled", success, message))


def _done(result=None) -> concurrent.futures.Future:
    """A change the thread that steps the scene has already made."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(result)
    return future


def _ending_io() -> RobotsIO:
    io = _robots_io()
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = _done()
    io._io = Mock(spec=["forget"])
    io._placements["alpha"] = Placement.of((0.0, 0.0, 0.0), 0.0)
    return io


class _Goal(_Answers):
    """A goal context carrying `request`, whose holder never cancels."""

    def __init__(self, request):
        super().__init__()
        self._request = request

    def request(self):
        return self._request

    async def cancel_signal(self):
        await asyncio.Event().wait()


@pytest.mark.parametrize("model", ["openarm_v2", "so101"])
def test_a_scene_that_cannot_stand_the_robot_answers_its_own_goal(model):
    """The stand runs on the thread that steps the scene; a scene that
    cannot take the robot says so on that robot's goal, and its name goes
    back."""
    io = _robots_io()
    io._io = Mock(spec=["forget"])
    io._robots.admit("alpha", MODELS.of(model).entry, CALLER)
    io._placements["alpha"] = Placement.of((0.0, 0.0, 0.0), 0.0)
    refused: concurrent.futures.Future = concurrent.futures.Future()
    refused.set_exception(RuntimeError("the test scene stands nothing"))
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = refused
    goal = _Goal(_request(model=model))

    asyncio.run(io._stay(goal))

    io._edits.submit.assert_called_once()
    assert goal.answers == [
        ("completed", False, "the scene could not stand the robot: the test scene stands nothing")
    ]
    assert io._robots.of_name("alpha") is None, "the name goes back"
    assert "alpha" not in io._placements, "and so does its spot"


def test_standing_a_robot_composes_the_scene_around_it_and_takes_it_back_out_on_a_refusal():
    """The scene is let go of, changed and taken up again, and a scene that
    will not compile leaves the robots that were standing where they were."""
    io = _robots_io()
    io._io = Mock(spec=["forget"])
    io._robots = Mock(spec=["renew", "release", "stand"])
    composed = []
    io.binds_with(lambda: composed.append("rebind"), lambda: composed.append("unbind"))
    known = MODELS.of("openarm_v2")

    io.stand("alpha", known, Placement.of((0.0, 0.0, 0.0), 0.0))

    assert composed == ["unbind", "rebind"]
    assert [robot.instance for robot in io._world.robots()] == ["alpha"]

    def refuse():
        raise RuntimeError("this scene compiles nothing")

    io.binds_with(refuse, lambda: composed.append("unbind"))
    with pytest.raises(RuntimeError, match="this scene compiles nothing"):
        io.stand("charlo", known, Placement.of((3.0, 0.0, 0.0), 0.0))

    assert [robot.instance for robot in io._world.robots()] == ["alpha"], "the fleet is unchanged"


def test_taking_a_robot_out_composes_the_scene_around_the_ones_that_remain():
    io = _robots_io()
    io._io = Mock(spec=["forget"])
    io._robots = Mock(spec=["renew", "release", "stand"])
    composed = []
    io.binds_with(lambda: composed.append("rebind"), lambda: composed.append("unbind"))
    io.stand("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))
    composed.clear()

    io.unstand("alpha")

    assert composed == ["unbind", "rebind"]
    assert io._world.robots() == []
    io._io.forget.assert_called_once_with("alpha")


class _Unreachable(_Answers):
    """A goal context whose holder cannot be told anything."""

    async def publish_feedback(self, standing):
        raise RuntimeError("the holder is away")


def test_a_robot_whose_holder_cannot_be_told_it_stands_is_taken_out():
    io = _ending_io()
    goal = _Unreachable()

    assert asyncio.run(io._standing(goal, "alpha")) is False

    io._edits.submit.assert_called_once()
    assert "alpha" not in io._placements
    assert goal.answers == [
        ("completed", False, "the robot's holder could not be told it stands: the holder is away")
    ]


def test_a_stay_ending_with_the_engine_asks_nothing_of_the_stopping_scene():
    io = _ending_io()
    io._robots = Mock(spec=["release"])
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.STOPPED, "the engine stopped"))

    io._edits.submit.assert_not_called()
    io._robots.release.assert_called_once_with("alpha")
    assert "alpha" not in io._placements
    assert goal.answers == [("completed", False, "the engine stopped")]


def test_a_stay_that_lapsed_takes_the_robot_out_of_the_scene_first():
    io = _ending_io()
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.LAPSED, "its pairs were gone for the lease"))

    io._edits.submit.assert_called_once()
    assert "alpha" not in io._placements
    assert goal.answers == [("completed", False, "its pairs were gone for the lease")]


class TestLease:
    def test_a_robot_holding_its_models_pairs_keeps_its_place_however_stale_its_lease(self):
        """The watcher's tick renews a lease from the pairs it reads, and a
        node loop too busy to run it leaves the lease stale: a robot still
        holding its pairs is not taken out for that."""
        io = _leasing(now_s=10.0, held=OPENARM_LIMBS)
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
        io = _leasing(now_s=10.0, held=held)
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s == 10.0

    def test_a_robot_holding_no_pair_for_the_lease_lapses(self):
        io = _leasing(now_s=10.0, held=Held())

        assert io._lapse(io._robots.of_name("alpha")) == (
            f"none of this robot's limbs were paired for {LEASE}"
        )

    def test_a_robot_whose_limbs_left_keeps_its_place_for_the_lease(self):
        io = _leasing(now_s=LEASE_S, held=Held())
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s == 0.0, "a lease still running is not a renewed one"

    def test_a_robot_no_limb_reached_keeps_its_place_however_long_its_nodes_take(self):
        """Its stay is its goal's until a limb reaches it, so a backbone slow
        to come up finds the robot standing."""
        io = _leasing(now_s=1_000_000.0, held=Held(), reached_s=None)
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s is None

    def test_the_first_limb_to_reach_a_robot_starts_its_lease(self):
        """Limbs of another model reach it: not its own, so from their
        arrival on it has the lease to be paired as its model."""
        io = _leasing(now_s=10.0, held=OPENARM_LIMBS, model="so101", reached_s=None)
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s == 10.0
        io._loop = SimpleNamespace(time=lambda: 10.0 + LEASE_S + 0.1)
        assert io._lapse(robot).startswith(
            f"this robot's pairs were not its model's for {LEASE}"
        )

    def test_a_camera_reaching_a_robot_starts_no_lease(self):
        io = _leasing(
            now_s=10.0, held=Held(rgb_cameras=frozenset({"wrist_left"})), reached_s=None
        )
        robot = io._robots.of_name("alpha")

        assert io._lapse(robot) is None
        assert robot.last_paired_s is None

    def test_a_robot_missing_a_limb_lapses_naming_both_lists(self):
        held = Held(arms=OPENARM_LIMBS.arms, grippers=frozenset({"left_gripper"}))
        io = _leasing(now_s=10.0, held=held)

        assert io._lapse(io._robots.of_name("alpha")) == (
            f"this robot's pairs were not its model's for {LEASE}: its pairs name "
            "arms ['left_arm', 'right_arm'], grippers ['left_gripper'], rgb_cameras [], "
            f"rgbd_cameras [], and {OPENARM_V2_HAS}"
        )

    def test_a_robot_streaming_a_camera_its_model_lacks_lapses_naming_both_lists(self):
        held = Held(
            arms=SO101_LIMBS.arms, grippers=SO101_LIMBS.grippers, rgbd_cameras=frozenset({"chest"})
        )
        io = _leasing(now_s=10.0, held=held, model="so101")

        assert io._lapse(io._robots.of_name("alpha")) == (
            f"this robot's pairs were not its model's for {LEASE}: its pairs name "
            "arms ['arm'], grippers ['gripper'], rgb_cameras [], rgbd_cameras ['chest'], "
            f"and {SO101_HAS}"
        )

    def test_a_robot_paired_as_another_model_has_its_stay_ended_naming_both_lists(self):
        """It joined as an SO-101 and its backbone leads an OpenArm's limbs,
        which reached it at 0 s: no pair of its own ever drives it, so its
        lease is never renewed."""
        io = _leasing(now_s=10.0, held=OPENARM_LIMBS, model="so101")
        # The lease is read at once instead of after a share of it.
        io._lease_check_period = lambda: 0.0
        io._stopping = asyncio.Event()
        goal = _Goal(_request(model="so101"))

        ending, why = asyncio.run(io._watch(goal, "alpha", io._handover("alpha")))

        assert ending is Ending.LAPSED
        assert why == (
            f"this robot's pairs were not its model's for {LEASE}: its pairs name "
            "arms ['left_arm', 'right_arm'], grippers ['left_gripper', 'right_gripper'], "
            f"rgb_cameras [], rgbd_cameras [], and {SO101_HAS}"
        )
        assert io._robots.of_name("alpha").last_paired_s == 0.0

    @pytest.mark.parametrize(
        ("model", "held", "renewed_s"),
        [
            ("openarm_v2", OPENARM_LIMBS, 10.0),
            ("so101", SO101_LIMBS, 10.0),
            ("so101", OPENARM_LIMBS, 0.0),
            ("openarm_v2", Held(arms=OPENARM_LIMBS.arms), 0.0),
            ("openarm_v2", Held(), 0.0),
        ],
    )
    def test_the_watcher_renews_a_lease_only_while_the_pairs_are_the_models(
        self, model, held, renewed_s
    ):
        io = _leasing(now_s=10.0, held=held, model=model)

        async def one_tick() -> None:
            watcher = asyncio.create_task(io._watch_pairs())
            # One turn of the loop runs the watcher up to the pause between
            # two ticks.
            await asyncio.sleep(0)
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

        asyncio.run(one_tick())

        assert io._robots.of_name("alpha").last_paired_s == renewed_s


class TestAFleetsLeases:
    """Every robot of a fleet holds its own pairs on the four slots this
    engine declares, so each one's lease and each one's readiness is read
    off its own."""

    def test_each_robot_is_held_to_its_own_model(self):
        """An SO-101 stands beside an OpenArm on the same four slots. Each
        holds its own model's limbs, and neither is asked for the other's."""
        io = _leasing_fleet(
            now_s=10.0,
            fleet={"alpha": ("openarm_v2", OPENARM_LIMBS), "charlo": ("so101", SO101_LIMBS)},
        )

        for name in ("alpha", "charlo"):
            robot = io._robots.of_name(name)
            assert io._lapse(robot) is None
            assert robot.last_paired_s == 10.0

    def test_one_robots_mismatch_ends_its_own_stay_and_no_others(self):
        """The SO-101's backbone leads an OpenArm's limbs, so its lease runs
        out while the OpenArm beside it keeps its place."""
        io = _leasing_fleet(
            now_s=10.0,
            fleet={"alpha": ("openarm_v2", OPENARM_LIMBS), "charlo": ("so101", OPENARM_LIMBS)},
        )

        assert io._lapse(io._robots.of_name("charlo")) is not None
        assert io._lapse(io._robots.of_name("alpha")) is None

    def test_each_lease_is_read_off_its_own_robots_pairs(self):
        """Reading one robot's lease renews that robot's and no other's."""
        io = _leasing_fleet(
            now_s=10.0,
            fleet={
                "alpha": ("openarm_v2", OPENARM_LIMBS),
                "bravo": ("openarm_v2", Held(arms=OPENARM_LIMBS.arms)),
            },
        )

        io._lapse(io._robots.of_name("alpha"))

        renewed = {robot.name: robot.last_paired_s for robot in io._robots.robots()}
        assert renewed == {"alpha": 10.0, "bravo": 0.0}

    def test_the_watcher_renews_each_lease_only_while_its_own_pairs_are_its_models(self):
        """The watcher reads every robot on one tick, and a robot renews on
        that tick only while its own pairs are the ones its model asks for:
        one robot holding its limbs renews nobody else's lease."""
        io = _leasing_fleet(
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
        assert renewed == {
            "alpha": 10.0,
            "bravo": 0.0,
            "charlo": 10.0,
            "delta": 0.0,
            "echo": 0.0,
        }

    def test_each_robot_is_ready_on_its_own_pairs(self):
        """Readiness answers for the robot its caller runs, so a robot whose
        limbs are all held is ready beside one whose are not."""
        io = _leasing_fleet(
            now_s=10.0,
            fleet={"alpha": ("openarm_v2", OPENARM_LIMBS), "bravo": ("openarm_v2", Held())},
        )
        for name in ("alpha", "bravo"):
            io._robots.stand(name)

        assert self._ready(io, "alpha") is True
        assert self._ready(io, "bravo") is False

    @staticmethod
    def _ready(io: RobotsIO, robot: str) -> bool:
        caller = _caller(robot)
        request = SimpleNamespace(core_node=caller.core_node, instance_id=caller.instance_id)
        return io._ready(request).ready


class _Leaving(_Answers):
    """A goal whose holder lets go the moment the stay waits on it."""

    def __init__(self):
        super().__init__()
        self.let_go = asyncio.Event()

    def request(self):
        return _request()

    async def publish_feedback(self, standing):
        return None

    async def cancel_signal(self):
        await self.let_go.wait()


class _Withdrawing(_Answers):
    """A goal whose holder lets go while its robot still waits for the
    thread that steps the scene."""

    def request(self):
        return _request()

    async def cancel_signal(self):
        return None


def _standing_io() -> RobotsIO:
    """A RobotsIO with a robot admitted and its stand still waiting."""
    io = _robots_io()
    io._edits = Edits()
    io._io = Mock(spec=["forget"])
    io._robots.admit("alpha", MODELS.of("openarm_v2").entry, CALLER)
    io._placements["alpha"] = Placement.of((0.0, 0.0, 0.0), 0.0)
    io._admitted = None
    return io


def test_a_robot_that_left_before_it_stood_gives_back_its_name_and_its_spot():
    """Its stand still waits for the thread that steps the scene, so the
    stand is withdrawn: nothing stood, and the next copy takes the name."""
    io = _standing_io()

    stood = asyncio.run(io._stand(_Withdrawing(), "alpha", "openarm_v2"))

    assert stood is False
    assert io._robots.of_name("alpha") is None
    assert io._placements == {}


def test_a_robot_the_scene_refused_gives_back_its_name_and_its_spot():
    """The scene answered the stand with why it will not take the robot, so
    what the admission held goes back with the answer."""
    io = _standing_io()
    io.binds_with(lambda: None, lambda: None)
    io._world.add("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))
    goal = _Goal(_request())

    async def stand_and_drain():
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        io._edits.drain()
        return await standing

    assert asyncio.run(stand_and_drain()) is False
    assert io._robots.of_name("alpha") is None
    assert io._placements == {}
    assert "the scene could not stand the robot" in goal.answers[0][2]


def test_a_robot_that_stood_is_recorded_as_standing():
    """What marks it standing is what tells a copy re-registering that its
    own robot is there to be handed over."""
    io = _standing_io()
    io.binds_with(lambda: None, lambda: None)
    goal = _Goal(_request())

    async def stand_and_drain():
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        io._edits.drain()
        return await standing

    assert asyncio.run(stand_and_drain()) is True
    assert io._robots.of_name("alpha").standing()


def test_a_stopping_engine_ends_every_stay_and_every_loop():
    """A robot whose engine is going down is told so on its own goal, and
    nothing of this engine is left running behind it."""
    io = _robots_io()
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


class TestHowOftenAStayChecksItsLease:
    """A stay wakes on a share of its lease so a robot whose pairs are gone
    leaves within it, and no faster than the floor, because every wake is a
    turn of the node loop the whole engine shares."""

    def test_the_period_is_a_share_of_the_lease(self):
        io = _robots_io()
        io._lease_s = 8.0

        assert io._lease_check_period() == 8.0 * 0.25

    def test_a_short_lease_is_checked_no_faster_than_the_floor(self):
        """`robot_lease_ms` takes any positive number, and a lease of one
        millisecond would wake the node loop a thousand times a second."""
        io = _robots_io()
        io._lease_s = 0.001

        assert io._lease_check_period() == 0.05

    def test_every_lease_is_checked_inside_itself(self):
        io = _robots_io()
        for lease_s in (0.001, 0.05, 1.0, 8.0, 600.0):
            io._lease_s = lease_s
            assert 0.0 < io._lease_check_period()
            assert io._lease_check_period() <= max(lease_s, 0.05)


def test_forgetting_a_robot_drops_its_spot_and_its_signal():
    """What a robot's joining held is a spot promised and a signal its own
    goal waits on, and a name given back leaves neither behind."""
    io = _robots_io()
    io._placements["alpha"] = Placement.of((0.0, 0.0, 0.0), 0.0)
    io._handover("alpha")

    io._forget("alpha")

    assert io._placements == {}
    assert io._handovers == {}


class TestHowAStayEnds:
    """A stay waits for one of the ways it can end and says which it was.
    Every one of them answers the goal differently, and the answer is what
    the copy holding the robot reads."""

    @staticmethod
    def _watching() -> RobotsIO:
        io = _leasing_fleet(now_s=0.0, fleet={"alpha": ("openarm_v2", OPENARM_LIMBS)})
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
        its robot out of the scene."""
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


def test_a_robot_that_left_is_answered_as_cancelled():
    """A goal its holder cancelled is completed as cancelled, which is what
    tells the copy its robot left because it asked."""
    io = _ending_io()
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.LEFT, "the robot left the scene"))

    assert goal.answers == [("cancelled", True, "the robot left the scene")]


def test_a_robot_that_lapsed_is_answered_as_completed():
    io = _ending_io()
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.LAPSED, "its pairs were not its model's"))

    assert goal.answers == [("completed", False, "its pairs were not its model's")]


class TestReadiness:
    @staticmethod
    def _ready(held: Held, model: str = "openarm_v2", standing: bool = True, caller: Caller = CALLER) -> bool:
        io = _leasing(now_s=1.0, held=held, model=model)
        if standing:
            io._robots.stand("alpha")
        request = SimpleNamespace(core_node=caller.core_node, instance_id=caller.instance_id)
        return io._ready(request).ready

    def test_a_robot_holding_every_limb_is_ready(self):
        assert self._ready(OPENARM_LIMBS)

    def test_a_robot_missing_one_limb_is_never_ready(self):
        for missing in ("left_arm", "right_arm"):
            assert not self._ready(Held(arms=OPENARM_LIMBS.arms - {missing}, grippers=OPENARM_LIMBS.grippers))
        for missing in ("left_gripper", "right_gripper"):
            assert not self._ready(Held(arms=OPENARM_LIMBS.arms, grippers=OPENARM_LIMBS.grippers - {missing}))

    def test_a_camera_left_unpaired_does_not_hold_readiness_back(self):
        assert self._ready(OPENARM_LIMBS)
        assert self._ready(
            Held(
                arms=OPENARM_LIMBS.arms,
                grippers=OPENARM_LIMBS.grippers,
                rgbd_cameras=frozenset({"chest"}),
            )
        )

    def test_a_one_arm_so101_is_ready_with_its_arm_and_its_gripper(self):
        assert self._ready(SO101_LIMBS, model="so101")
        assert not self._ready(Held(arms=SO101_LIMBS.arms), model="so101")
        assert not self._ready(Held(grippers=SO101_LIMBS.grippers), model="so101")

    def test_limbs_of_another_model_make_no_robot_ready(self):
        assert not self._ready(OPENARM_LIMBS, model="so101")
        assert not self._ready(SO101_LIMBS, model="openarm_v2")

    def test_a_robot_the_scene_has_not_stood_yet_is_not_ready(self):
        assert not self._ready(OPENARM_LIMBS, standing=False)

    def test_an_instance_with_no_robot_in_the_scene_is_not_ready(self):
        stranger = Caller(core_node="sim16", instance_id="bravo_init_inst")
        assert not self._ready(OPENARM_LIMBS, caller=stranger)


def test_a_goal_handed_over_leaves_its_robot_standing():
    """A copy whose initializer came back takes its own robot over: the goal
    that hosted it ends without asking the scene for anything."""
    io = _ending_io()
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.HANDED_OVER, "another goal hosts it"))

    io._edits.submit.assert_not_called()
    assert goal.answers == [("completed", False, "another goal hosts it")]


@pytest.mark.parametrize("change", ["stand", "unstand"])
def test_a_scene_that_will_not_let_go_is_taken_up_again(change):
    """Letting go of the scene is part of the change that asked for it, so a
    scene that will not let go leaves the engine holding one it can step."""
    io = _robots_io()
    io._io = Mock(spec=["forget"])
    composed = []

    def refuse_to_let_go():
        composed.append("unbind refused")
        raise RuntimeError("this scene will not let go")

    io.binds_with(lambda: None, lambda: None)
    io.stand("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))
    composed.clear()
    io.binds_with(lambda: composed.append("rebind"), refuse_to_let_go)

    with pytest.raises(RuntimeError, match="this scene will not let go"):
        if change == "stand":
            io.stand("bravo", MODELS.of("openarm_v2"), Placement.of((3.0, 0.0, 0.0), 0.0))
        else:
            io.unstand("alpha")

    assert composed == ["unbind refused", "rebind"]


def test_a_stand_that_cannot_compose_the_scene_puts_the_previous_one_back():
    """The scene is let go of before it is changed, so a robot the scene will
    not take leaves the robots that were standing stepping as they were."""
    io = _robots_io()
    io._io = Mock(spec=["forget"])
    io._robots = Mock(spec=["renew", "release", "stand"])
    composed = []
    refusals = iter([RuntimeError("this scene compiles nothing")])

    def rebind():
        refusal = next(refusals, None)
        composed.append("refused" if refusal else "rebind")
        if refusal:
            raise refusal

    io.binds_with(lambda: None, lambda: composed.append("unbind"))
    io.stand("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))
    composed.clear()
    io.binds_with(rebind, lambda: composed.append("unbind"))

    with pytest.raises(RuntimeError, match="this scene compiles nothing"):
        io.stand("charlo", MODELS.of("openarm_v2"), Placement.of((3.0, 0.0, 0.0), 0.0))

    # The scene was composed again without the robot that could not join.
    assert composed == ["unbind", "refused", "rebind"]
    assert [robot.instance for robot in io._world.robots()] == ["alpha"]


def test_a_copy_that_comes_straight_back_keeps_the_spot_it_was_promised():
    """Taking a robot out gives its name back while the scene is composed
    again, so the copy is admitted inside that window: what its own goal was
    promised must survive the goal that is leaving."""
    io = _ending_io()
    goal = _Answers()
    promised = Placement.of((1.5, 0.0, 0.0), 0.0)

    def admit_the_copy_again(_change):
        # The copy is admitted while the take-out waits for the thread that
        # steps the scene, and is promised a spot of its own.
        io._placements["alpha"] = promised
        return _done()

    io._edits.submit = admit_the_copy_again

    asyncio.run(io._take_out(goal, "alpha", Ending.LEFT, "the robot left the scene"))

    assert io._placements["alpha"] == promised


def test_a_copy_that_comes_straight_back_is_stood_afresh():
    """Taking a robot out waits for the thread that steps the scene, which is
    a whole scene composed and compiled. A copy admitted inside that window
    decides what happens to it by what it finds: a name still taken is its
    own robot to be handed over, and this robot is already on its way out, so
    everything its joining held goes back before the scene is asked."""
    io = _ending_io()
    io._robots.admit("alpha", MODELS.of("openarm_v2").entry, CALLER)
    held_when_the_scene_was_asked = []

    def take_the_robot_out(_change):
        held_when_the_scene_was_asked.append(io._robots.of_name("alpha"))
        return _done()

    io._edits.submit = take_the_robot_out

    asyncio.run(io._take_out(_Answers(), "alpha", Ending.LEFT, "the robot left the scene"))

    assert held_when_the_scene_was_asked == [None]
    assert io._placements == {}


def test_a_name_carrying_the_scene_separator_is_refused_without_touching_the_scene():
    """Every name a robot answers to in the scene is its own name and that
    separator. The refusal comes with the admission, so no robot standing is
    torn down and composed again for a name the scene was never going to
    carry."""
    io = _robots_io()
    composed = []
    io.binds_with(lambda: composed.append("rebind"), lambda: composed.append("unbind"))

    decision = io._admit(_request(robot="left/arm"))

    assert not decision.accepted
    assert "peppy stack join LAUNCHER -i left_arm" in decision.payload
    assert composed == []
    assert io._world.robots() == []


def test_a_spot_a_robot_stands_on_is_refused_naming_that_robot():
    """A robot standing on the spot is named as standing, where a copy only
    promised it is named as about to."""
    io = _robots_io()
    io.binds_with(lambda: None, lambda: None)
    io.stand("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))

    decision = io._admit(
        _request(
            robot="charlo",
            instance="charlo_init_inst",
            placement=SimpleNamespace(position=[0.1, 0.0, 0.0], yaw=0.0),
        )
    )

    assert not decision.accepted
    assert "'alpha' stands within 1.5 m" in decision.payload


def test_a_copy_that_comes_straight_back_takes_the_spot_it_had():
    """Its name went back when its stay ended, so the robot still standing
    under that name is on its way out: the spot it stands on is the one the
    copy is coming back to, and nothing else may be told to move for it."""
    io = _robots_io()
    io.binds_with(lambda: None, lambda: None)
    io.stand("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))

    # The take-out gave the name back; the robot has yet to leave the scene.
    asked_again = io._admit(_request(instance="alpha_init_2"))
    io._forget("alpha")
    io._robots.release("alpha")
    where_it_stands = io._admit(
        _request(
            instance="alpha_init_3",
            placement=SimpleNamespace(position=[0.0, 0.0, 0.0], yaw=0.0),
        )
    )

    assert asked_again.accepted
    assert where_it_stands.accepted
    assert io._placements["alpha"] == Placement.of((0.0, 0.0, 0.0), 0.0)


def test_a_goal_that_never_reached_its_stay_gives_back_what_it_reserved():
    """A goal is accepted on a hop of its own after this engine has answered
    for it, and a robot with no stay has nothing watching its lease."""
    io = _robots_io()

    assert io._admit(_request()).accepted
    io._withdraw()

    assert io._robots.of_name("alpha") is None
    assert io._placements == {}
    # A copy coming after it is admitted, under the name and on the spot.
    assert io._admit(_request(instance="alpha_init_2")).accepted


def test_withdrawing_again_leaves_the_copy_that_took_the_name_alone():
    """Serving the attaches withdraws whenever it fails and goes back to
    waiting, so it withdraws again having admitted nothing in between. One
    admission is given back once: the name is the next copy's by then."""
    io = _robots_io()
    assert io._admit(_request()).accepted
    io._withdraw()

    # The name the withdrawal freed is taken by a copy of its own.
    io._robots.admit("alpha", MODELS.of("openarm_v2").entry, _caller("alpha_2"))
    io._robots.stand("alpha")
    io._withdraw()

    assert io._robots.of_name("alpha").standing()


def test_a_goal_lost_on_its_way_back_frees_the_name_it_reserved():
    """The engine answers for a goal and the runtime accepts it on a hop of
    its own, which can fail: what the admission reserved is given back by
    the loop that finds the accept unfinished."""
    io = _robots_io()

    class _Lost:
        async def handle_goal_next_request(self, admit):
            admit(_request())
            raise RuntimeError("the peer went away before it was accepted")

    async def serve_once():
        server = asyncio.create_task(io._serve_attach(_Lost()))
        await asyncio.sleep(0)
        server.cancel()

    asyncio.run(serve_once())

    assert io._robots.of_name("alpha") is None
    assert io._placements == {}


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
    io = _leasing_fleet(now_s=0.0, fleet={"alpha": ("openarm_v2", OPENARM_LIMBS)})
    io._robots.stand("alpha")
    io._stopping = asyncio.Event()
    hosted = io._handover("alpha")
    goal = _Hosting()

    async def take_over() -> tuple:
        stay = asyncio.create_task(io._stay(goal))
        # Enough turns for the stay to end the goal it took the robot from
        # and to settle into watching the robot's lease.
        for _ in range(50):
            await asyncio.sleep(0)
        # What the stay had done by the time it was watching.
        watching = not stay.done(), list(goal.answers)
        stay.cancel()
        await asyncio.gather(stay, return_exceptions=True)
        return watching

    still_hosting, answered = asyncio.run(take_over())

    assert hosted.is_set()
    assert still_hosting
    assert answered == []


def test_adopting_a_robot_reserves_nothing_and_ends_nobody():
    """Admitting an adoption takes nothing and ends nobody: the robot keeps
    the stay hosting it until the stay taking it over exists, so the host it
    has is the one watching its lease throughout."""
    io = _robots_io()
    assert io._admit(_request()).accepted
    io._robots.stand("alpha")
    hosted = io._handover("alpha")

    assert io._admit(_request()).accepted

    assert io._admitted is None
    assert not hosted.is_set()
    assert io._robots.of_name("alpha") is not None


def test_a_spot_the_robot_coming_back_stands_on_is_its_own():
    """Its name went back when its stay ended and the scene has yet to let
    it go, so the lattice offers it the spot it is standing on."""
    io = _robots_io()
    io.binds_with(lambda: None, lambda: None)
    io.stand("alpha", MODELS.of("openarm_v2"), Placement.of((0.0, 0.0, 0.0), 0.0))
    world = io._world

    assert world.occupied(Placement.of((0.0, 0.0, 0.0), 0.0))
    assert not world.occupied(Placement.of((0.0, 0.0, 0.0), 0.0), leaving="alpha")
    assert world.free_spot((), "alpha").position == (0.0, 0.0, 0.0)


def test_a_goal_whose_spot_was_taken_back_gives_its_name_back():
    io = _robots_io()
    io._io = Mock(spec=["forget"])
    io._robots.admit("alpha", MODELS.of("openarm_v2").entry, CALLER)
    goal = _Goal(_request())

    asyncio.run(io._stay(goal))

    assert io._robots.of_name("alpha") is None
    assert goal.answers == [("completed", False, "the name was given back before the robot stood")]
