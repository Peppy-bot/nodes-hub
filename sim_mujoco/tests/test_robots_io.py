"""Who may join this engine's scene: one robot at a time, of a model the
engine has an entry for, standing where the scene puts it, staying for as long
as its pairs are its model's, and ready once it holds every limb."""

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
    io._loop = SimpleNamespace(time=lambda: 0.0)
    io._handovers = {}
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


def test_a_placement_is_refused_because_the_scene_places_the_robot():
    io = _robots_io()
    decision = io._admit(_request(placement=SimpleNamespace(position=[1.0, 0.0, 0.0], yaw=0.0)))
    assert not decision.accepted
    assert "placement { auto: true }" in decision.payload
    assert io._robots.robots() == []


def test_a_second_robot_is_refused_while_one_stands():
    io = _robots_io()
    assert io._admit(_request()).accepted
    decision = io._admit(_request(robot="charlo", model="so101", instance="charlo_init_inst"))
    assert not decision.accepted
    assert "stands one robot" in decision.payload
    assert "'alpha' of alpha_init_inst@sim16 stands already" in decision.payload


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


class _Pairs:
    """What the robot holds on the four slots, as the engine reads its
    pairs."""

    def __init__(self, held: Held) -> None:
        self.held = held
        self.forgotten = []

    def held_by(self, _robot: str) -> Held:
        return self.held

    def forget(self, robot: str) -> None:
        self.forgotten.append(robot)


def _leasing(
    now_s: float, held: Held, model: str = "openarm_v2", reached_s: Optional[float] = 0.0
) -> RobotsIO:
    """A RobotsIO whose robot `alpha` of `model` holds `held`, read at `now_s`
    with the lease this scene gives. Its lease runs from `reached_s`, when a
    limb first reached it, and not at all when that is None."""
    io = _robots_io()
    io._robots.admit("alpha", MODELS.of(model).entry, CALLER)
    if reached_s is not None:
        io._robots.note_limbs_reached("alpha", reached_s)
    io._loop = SimpleNamespace(time=lambda: now_s)
    io._io = _Pairs(held)
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


def _ending_io() -> RobotsIO:
    io = _robots_io()
    io._stands = Mock(spec=["unstand"])
    io._io = Mock(spec=["forget"])
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
def test_a_stand_asks_the_scene_for_the_robots_model(model):
    io = _robots_io()
    io._robots.admit("alpha", MODELS.of(model).entry, CALLER)
    refused = concurrent.futures.Future()
    refused.set_exception(RuntimeError("the test scene stands nothing"))
    io._stands = Mock(spec=["stand"])
    io._stands.stand.return_value = refused
    goal = _Goal(_request(model=model))

    asyncio.run(io._stay(goal))

    io._stands.stand.assert_called_once_with(MODELS.of(model), "alpha")
    assert goal.answers == [
        ("completed", False, "the scene could not stand the robot: the test scene stands nothing")
    ]
    assert io._robots.of_name("alpha") is None, "the name goes back"


class _Unreachable(_Answers):
    """A goal context whose holder cannot be told anything."""

    async def publish_feedback(self, standing):
        raise RuntimeError("the holder is away")


def test_a_robot_whose_holder_cannot_be_told_it_stands_is_taken_out():
    io = _ending_io()
    taken_out = concurrent.futures.Future()
    taken_out.set_result(None)
    io._stands.unstand.return_value = taken_out
    goal = _Unreachable()

    assert asyncio.run(io._standing(goal, "alpha")) is False

    io._stands.unstand.assert_called_once_with()
    io._io.forget.assert_called_once_with("alpha")
    assert goal.answers == [
        ("completed", False, "the robot's holder could not be told it stands: the holder is away")
    ]


def test_a_stay_ending_with_the_engine_asks_nothing_of_the_stopping_scene():
    io = _ending_io()
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.STOPPED, "the engine stopped"))

    io._stands.unstand.assert_not_called()
    io._io.forget.assert_called_once_with("alpha")
    assert goal.answers == [("completed", False, "the engine stopped")]


def test_a_stay_that_lapsed_takes_the_robot_out_of_the_scene_first():
    io = _ending_io()
    taken_out = concurrent.futures.Future()
    taken_out.set_result(None)
    io._stands.unstand.return_value = taken_out
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.LAPSED, "its pairs were gone for the lease"))

    io._stands.unstand.assert_called_once_with()
    io._io.forget.assert_called_once_with("alpha")
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
