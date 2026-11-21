"""Every robot on the stage is driven as the model it joined as: its limbs
under the names its own robot answers to, its drives under its own model's
gains, its gripper opening mapped onto its own fingers' travel, its base's
wheels on their rate loops, and the rig of its own model's cameras. An SO-101
stands beside an OpenArm here, and a botshare base rolls beside them, on
stand-in views, so nothing of one leaks into another.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from control_core_py.geometry import quaternion_of_yaw
from robot_description_py import model as bundled
from sim_robot_core.models import shipped_entry
from sim_robot_core.setpoints import JointSetpoint

import _world

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))

from wheel_loop import RateLoopGains  # noqa: E402  pylint: disable=C0413


def _finger(model: str, joint: str):
    """The finger of `model`'s description that `joint` moves."""
    arms = bundled(model).description().arms
    return next(finger for arm in arms for finger in arm.gripper.fingers if finger.joint == joint)


# Where each finger closes and opens, as each model's description gives it:
# the SO-101's jaw closes below its joint's zero, and an OpenArm v2's fingers
# close at zero, the left ones opening toward positive by as much as the
# right ones open toward negative.
_SO101_JAW = (_finger("so101", "gripper").span.closed, _finger("so101", "gripper").span.open)
_OPENARM_V2_FINGER_OPEN = _finger("openarm_v2", "openarm_left_finger_joint1").span.open


class _Views:
    """Stand-ins for the PhysX-backed views, recording how each was built and
    what was written through it."""

    def __init__(self):
        self.articulations = {}
        self.actuators = {}
        self.sensors = {}
        self.rigs = []
        # Per articulation prim: the dof names it reports, in its own order.
        self.dofs = {}
        self.positions = {}
        self.velocities = {}
        # Per articulation prim: its root's state as PhysX reads it, and where
        # each set down stood it.
        self.roots = {}
        self.set_downs = []

    def articulation(self, prim, name):
        views = self

        class Articulation:
            def setup(self):
                return True

            def teardown(self):
                pass

            def get_joint_names(self):
                return list(views.dofs[prim])

            def get_joint_states(self):
                positions = [views.positions.get((prim, dof), 0.0) for dof in views.dofs[prim]]
                velocities = [views.velocities.get((prim, dof), 0.0) for dof in views.dofs[prim]]
                return positions, velocities

            def get_root_state(self):
                return views.roots.get(prim)

            def set_down_at(self, position, orientation):
                views.set_downs.append((prim, tuple(position), tuple(orientation)))
                return True

        built = Articulation()
        self.articulations[name] = prim
        return built

    def actuator(self, prim, joint_names, params, name):
        built = Mock(spec=["setup", "teardown", "write_targets", "set_force_limit", "write_velocity_targets"])
        built.setup.return_value = True
        built.set_force_limit.return_value = True
        self.actuators[name] = (prim, joint_names, params, built)
        return built

    def sensor(self, prim, finger_joints):
        views = self
        built = Mock(spec=["setup", "teardown", "get_gripper_state"])
        built.setup.return_value = True
        built.get_gripper_state.side_effect = lambda: {
            "positions": [views.positions.get((prim, joint), 0.0) for joint in finger_joints],
            "applied_forces": [0.0] * len(finger_joints),
        }
        self.sensors[(prim, tuple(finger_joints))] = built
        return built

    def rig(self, robot, root_prim, known, io):
        built = Mock(spec=["setup", "teardown", "step"])
        self.rigs.append((robot, root_prim, known))
        return built


@pytest.fixture(name="engine")
def _engine(monkeypatch):
    # The typed transport is not under test; the module only needs its names.
    topics = ModuleType("sim_topics")
    topics.SimTopicIO = object
    monkeypatch.setitem(sys.modules, "sim_topics", topics)
    monkeypatch.syspath_prepend(str(_ENGINE_DIR))
    spec = importlib.util.spec_from_file_location(
        "_bridge_models_under_test", _ENGINE_DIR / "bridge_extension.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    views = _Views()
    monkeypatch.setattr(module, "IsaacArticulation", views.articulation)
    monkeypatch.setattr(module, "IsaacActuatorCtrl", views.actuator)
    monkeypatch.setattr(module, "IsaacGripperSensor", views.sensor)
    monkeypatch.setattr(module, "IsaacCameraSensor", views.rig)
    module.views = views
    return module


def _robot(engine, instance: str, known):
    """A robot on the stage, its articulation reporting every joint of its
    model, fingers first so no limb's rows are its entry's own order."""
    import world  # pylint: disable=C0415

    robot = world.Robot(instance=instance, known=known, placement=_world.placement_at(0.0))
    entry = known.entry
    engine.views.dofs[robot.articulation()] = [
        *entry.finger_joints(),
        *entry.arm_joints(),
        *entry.wheel_joints(),
    ]
    return robot


def _limbs(engine, instance: str, model: str):
    limbs = engine.RobotLimbs(_robot(engine, instance, _world.known(model)))
    assert limbs.setup()
    return limbs


class TestLimbs:
    def test_an_openarm_is_driven_limb_by_limb_under_its_robots_names(self, engine):
        limbs = _limbs(engine, "alpha", "openarm_v2")

        assert sorted(limbs.arm_actuators) == ["left_arm", "right_arm"]
        assert sorted(limbs.gripper_actuators) == ["left_gripper", "right_gripper"]
        assert sorted(limbs.gripper_sensors) == ["left_gripper", "right_gripper"]
        # Isaac registers a view under a name: each is its limb's and its robot's.
        assert sorted(engine.views.actuators) == [
            "left_arm_alpha",
            "left_gripper_alpha",
            "right_arm_alpha",
            "right_gripper_alpha",
        ]
        # A gripper drives its first finger: the second one mimics it.
        for side in ("left", "right"):
            joints = engine.views.actuators[f"{side}_gripper_alpha"][1]
            assert joints == [f"openarm_{side}_finger_joint1"]

    def test_an_so101_is_driven_by_its_one_arm_and_its_one_jaw(self, engine):
        limbs = _limbs(engine, "charlo", "so101")

        assert list(limbs.arm_actuators) == ["arm"]
        assert list(limbs.gripper_actuators) == ["gripper"]
        prim, joints, _, _ = engine.views.actuators["arm_charlo"]
        assert prim == "/World/charlo"
        assert joints == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        assert engine.views.actuators["gripper_charlo"][1] == ["gripper"]

    def test_each_robots_drives_take_the_gains_of_its_own_model(self, engine):
        _limbs(engine, "alpha", "openarm_v2")
        _limbs(engine, "charlo", "so101")
        params = {name: built[2] for name, built in engine.views.actuators.items()}

        gains = _world.known("openarm_v2").arm_gains
        for arm in ("left_arm_alpha", "right_arm_alpha"):
            assert params[arm]["kp"] == list(gains.kp)
            assert params[arm]["kd"] == list(gains.kd)
            assert params[arm]["max_efforts"] == [40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0]
        assert params["left_gripper_alpha"]["kp"] == [80.0]
        assert params["left_gripper_alpha"]["max_efforts"] == [7.0]
        assert params["arm_charlo"] == {
            "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            "kp": [998.22] * 5,
            "kd": [2.731] * 5,
            "max_efforts": [2.94] * 5,
        }
        assert params["gripper_charlo"] == {
            "joint_names": ["gripper"],
            "kp": [998.22],
            "kd": [2.731],
            "max_efforts": [2.94],
        }

    def test_a_joint_its_models_entry_names_and_its_articulation_lacks_is_refused(self, engine):
        robot = _robot(engine, "charlo", _world.known("so101"))
        engine.views.dofs[robot.articulation()].remove("wrist_roll")

        with pytest.raises(
            RuntimeError,
            match=r"the so101 entry names joints not on the articulation of 'charlo': \['wrist_roll'\]",
        ):
            engine.RobotLimbs(robot).setup()

    def test_the_views_read_the_articulation_where_the_models_stage_keeps_it(self, engine):
        import isaac_models  # pylint: disable=C0415

        known = isaac_models.stood(
            "so101",
            shipped_entry("so101", bundled("so101").description()),
            isaac_models.parse("so101", {"stage": "so101/so101.usd", "articulation_root": "base_link"}),
            {},
            {},
            "base_link",
        )
        robot = _robot(engine, "charlo", known)

        limbs = engine.RobotLimbs(robot)

        assert robot.prim() == "/World/charlo"
        assert engine.views.articulations == {"articulation_charlo": "/World/charlo/base_link"}
        assert engine.views.actuators["arm_charlo"][0] == "/World/charlo/base_link"
        assert list(engine.views.sensors) == [("/World/charlo/base_link", ("gripper",))]
        assert limbs.setup()


class TestFingerSpans:
    def test_the_so101_jaw_opens_from_its_closed_state_to_its_open_one(self, engine):
        (span,) = _limbs(engine, "charlo", "so101").spans["gripper"]
        closed, opened = _SO101_JAW

        assert span.position_at(0.0) == pytest.approx(closed)
        assert span.position_at(1.0) == pytest.approx(opened)
        assert span.position_at(0.5) == pytest.approx((closed + opened) / 2.0)
        assert span.opening_at(opened) == pytest.approx(1.0)
        # The jaw's zero is not its closed pose.
        assert span.opening_at(0.0) == pytest.approx(-closed / (opened - closed))

    def test_an_openarms_fingers_open_from_zero_each_toward_its_own_side(self, engine):
        spans = _limbs(engine, "alpha", "openarm_v2").spans

        assert _OPENARM_V2_FINGER_OPEN > 0.0
        assert [span.position_at(1.0) for span in spans["left_gripper"]] == [_OPENARM_V2_FINGER_OPEN] * 2
        assert [span.position_at(1.0) for span in spans["right_gripper"]] == [-_OPENARM_V2_FINGER_OPEN] * 2
        for gripper in ("left_gripper", "right_gripper"):
            assert [span.position_at(0.0) for span in spans[gripper]] == [0.0, 0.0]


@pytest.fixture(name="fleet")
def _fleet(engine, monkeypatch):
    """An OpenArm v2 and an SO-101 standing on one stage."""
    robots = [
        _robot(engine, "alpha", _world.known("openarm_v2")),
        _robot(engine, "charlo", _world.known("so101")),
    ]
    world, io, registry = Mock(), Mock(), Mock()
    world.robots.return_value = robots
    world.take_set_down.return_value = set()
    registry.standing.return_value = {"alpha": Mock(), "charlo": Mock()}
    io.camera_robots.return_value = set()
    io.latest_arm_command.return_value = None
    io.latest_gripper_command.return_value = None
    objects = Mock()
    objects.capture_object_states.return_value = None
    bridge = engine.IsaacBridgeExtension(world, io, registry, objects, state_rate_hz=60, renders=True)
    monkeypatch.setattr(type(bridge), "_engine_time_s", lambda self: 0.0)
    bridge._state_pacer = Mock(take_if_due=Mock(return_value=True))
    bridge.bind()
    return bridge


def _writes(engine, name: str):
    return engine.views.actuators[name][3].write_targets


class TestDrivingAFleet:
    def test_each_robot_is_asked_for_the_setpoints_of_its_own_limbs(self, engine, fleet):
        fleet.step()

        asked = {
            (call.args[0], call.args[1])
            for call in (
                *fleet._io.latest_arm_command.call_args_list,
                *fleet._io.latest_gripper_command.call_args_list,
            )
        }
        assert asked == {
            ("alpha", "left_arm"),
            ("alpha", "right_arm"),
            ("alpha", "left_gripper"),
            ("alpha", "right_gripper"),
            ("charlo", "arm"),
            ("charlo", "gripper"),
        }

    def test_a_setpoint_reaches_the_drives_of_the_limb_it_names(self, engine, fleet):
        commands = {
            ("charlo", "arm"): JointSetpoint(positions=(0.1, 0.2, 0.3, 0.4, 0.5), velocities=()),
            ("alpha", "right_arm"): JointSetpoint(positions=(0.7,) * 7, velocities=(0.01,) * 7),
        }
        fleet._io.latest_arm_command.side_effect = lambda robot, arm: commands.get((robot, arm))

        fleet.step()

        _writes(engine, "arm_charlo").assert_called_once_with(
            {
                "shoulder_pan": 0.1,
                "shoulder_lift": 0.2,
                "elbow_flex": 0.3,
                "wrist_flex": 0.4,
                "wrist_roll": 0.5,
            },
            None,
        )
        right = engine.views.actuators["right_arm_alpha"][1]
        _writes(engine, "right_arm_alpha").assert_called_once_with(
            dict(zip(right, [0.7] * 7)), dict(zip(right, [0.01] * 7))
        )
        _writes(engine, "left_arm_alpha").assert_not_called()

    def test_an_opening_is_mapped_onto_the_fingers_of_its_own_gripper(self, engine, fleet):
        commands = {("charlo", "gripper"): (0.5, 0.0), ("alpha", "right_gripper"): (1.0, 0.0)}
        fleet._io.latest_gripper_command.side_effect = (
            lambda robot, gripper: commands.get((robot, gripper))
        )

        fleet.step()

        closed, opened = _SO101_JAW
        (jaw,) = _writes(engine, "gripper_charlo").call_args.args
        assert jaw == {"gripper": pytest.approx((closed + opened) / 2.0)}
        # The first finger takes the target; the second one follows it.
        (fingers,) = _writes(engine, "right_gripper_alpha").call_args.args
        assert fingers == {"openarm_right_finger_joint1": pytest.approx(-_OPENARM_V2_FINGER_OPEN)}
        _writes(engine, "left_gripper_alpha").assert_not_called()

    def test_each_robots_state_is_published_under_its_own_limbs(self, engine, fleet):
        positions = engine.views.positions
        positions[("/World/charlo", "shoulder_lift")] = -1.72
        _, opened = _SO101_JAW
        positions[("/World/charlo", "gripper")] = opened
        positions[("/World/alpha", "openarm_left_joint7")] = 0.3
        positions[("/World/alpha", "openarm_right_finger_joint1")] = -_OPENARM_V2_FINGER_OPEN / 2.0
        positions[("/World/alpha", "openarm_right_finger_joint2")] = -_OPENARM_V2_FINGER_OPEN / 2.0

        fleet.step()

        arms = {
            (call.args[0], call.args[1]): call.args[2]
            for call in fleet._io.publish_arm_states.call_args_list
        }
        assert arms == {
            ("alpha", "left_arm"): [0.0] * 6 + [0.3],
            ("alpha", "right_arm"): [0.0] * 7,
            ("charlo", "arm"): [0.0, -1.72, 0.0, 0.0, 0.0],
        }
        grippers = {
            (call.args[0], call.args[1]): call.args[2]
            for call in fleet._io.publish_gripper_states.call_args_list
        }
        assert grippers == {
            ("alpha", "left_gripper"): pytest.approx(0.0),
            ("alpha", "right_gripper"): pytest.approx(0.5),
            ("charlo", "gripper"): pytest.approx(1.0),
        }


class TestRigs:
    def test_a_robot_renders_the_rig_of_the_model_it_joined_as(self, engine, fleet):
        fleet._io.camera_robots.return_value = {"alpha", "charlo"}

        fleet.step()

        rigs = {robot: (root, known) for robot, root, known in engine.views.rigs}
        assert sorted(rigs) == ["alpha", "charlo"]
        assert rigs["alpha"][0] == "/World/alpha"
        assert [camera.name for camera in rigs["alpha"][1].entry.cameras] == [
            "wrist_left",
            "wrist_right",
            "chest",
        ]
        assert rigs["charlo"][0] == "/World/charlo"
        assert [camera.name for camera in rigs["charlo"][1].entry.cameras] == ["wrist"]

    def test_a_robot_pairing_no_camera_renders_none_whatever_its_model_carries(self, engine, fleet):
        fleet._io.camera_robots.return_value = {"charlo"}

        fleet.step()

        assert [robot for robot, _, _ in engine.views.rigs] == ["charlo"]

    def test_a_model_with_no_camera_renders_none(self, engine, fleet):
        fleet._world.robots.return_value = [_robot(engine, "bravo", _world.known("openarm_v1"))]
        fleet._io.camera_robots.return_value = {"bravo"}

        fleet._reconcile_rigs()

        assert engine.views.rigs == []
        assert fleet._camera_sensors == {}

    def test_an_engine_that_renders_no_camera_mounts_no_rig(self, engine, fleet):
        fleet._renders = False
        fleet._io.camera_robots.reset_mock()
        fleet._io.camera_robots.return_value = {"alpha", "charlo"}

        fleet.step()

        assert engine.views.rigs == []
        fleet._io.camera_robots.assert_not_called()


# Each wheel's rate loop as a stage authors it: other gains on each wheel, so
# nothing of one wheel's loop reaches the other.
_LOOPS = (RateLoopGains(kp=128.0, ki=64.0, max_torque=16.0), RateLoopGains(kp=96.0, ki=32.0, max_torque=12.0))


@pytest.fixture(name="rolling")
def _rolling(engine, monkeypatch):
    """A botshare base on the stage, its wheels' loops as its stage authors
    them, its engine clock under the test's hand."""
    import dataclasses  # pylint: disable=C0415

    robot = dataclasses.replace(
        _robot(engine, "delta", _world.known("botshare_base")), wheel_loops=_LOOPS
    )
    world, io, registry = Mock(), Mock(), Mock()
    world.robots.return_value = [robot]
    world.take_set_down.return_value = set()
    registry.standing.return_value = {"delta": Mock()}
    io.camera_robots.return_value = set()
    io.latest_base_command.return_value = None
    objects = Mock()
    objects.capture_object_states.return_value = None
    bridge = engine.IsaacBridgeExtension(world, io, registry, objects, state_rate_hz=60, renders=False)
    clock = {"now_s": 0.0}
    monkeypatch.setattr(type(bridge), "_engine_time_s", lambda self: clock["now_s"])
    bridge.clock = clock
    bridge._state_pacer = Mock(take_if_due=Mock(return_value=True))
    bridge.bind()
    return bridge


_WHEELS = ("left_wheel_joint", "right_wheel_joint")
_STEP_S = 1.0 / 60.0


def _wheel_targets(engine):
    return engine.views.actuators["base_delta"][3].write_velocity_targets


def _step_at(rolling, now_s, left=0.0, right=0.0):
    """Steps the bridge at `now_s` with the wheels read at these angles."""
    rolling.clock["now_s"] = now_s
    positions = sys.modules[type(rolling).__module__].views.positions
    positions[("/World/delta", "left_wheel_joint")] = left
    positions[("/World/delta", "right_wheel_joint")] = right
    rolling.step()


class TestDrivingABase:
    def test_the_wheels_keep_the_drives_their_stage_authors(self, engine, rolling):
        """The engine applies no gain of its own to a wheel's drive."""
        prim, joints, params, _ = engine.views.actuators["base_delta"]
        assert (prim, joints) == ("/World/delta", list(_WHEELS))
        assert params == {"joint_names": list(_WHEELS), "kp": [], "kd": [], "max_efforts": []}

    def test_a_stalled_wheels_target_holds_at_its_cap_over_its_proportional_gain(self, engine, rolling):
        """The integral clamps at the cap over ki, so the target velocity
        stands at most the cap over kp above the rate, each wheel by its own
        loop."""
        rolling._io.latest_base_command.return_value = JointSetpoint(positions=(0.0, 0.0), velocities=(2.0, -2.0))
        for step in range(120):
            _step_at(rolling, step * _STEP_S)

        (targets,) = _wheel_targets(engine).call_args.args
        assert targets == {
            "left_wheel_joint": pytest.approx(2.0 + 16.0 / 128.0),
            "right_wheel_joint": pytest.approx(-2.0 - 12.0 / 96.0),
        }

    def test_each_wheel_follows_the_rate_of_its_setpoint_through_its_loop(self, engine, rolling):
        rolling._io.latest_base_command.return_value = JointSetpoint(positions=(0.0, 0.0), velocities=(2.0, -1.0))

        _step_at(rolling, 0.0)
        # A first read anchors each loop at rest: the drive's target is the rate.
        _wheel_targets(engine).assert_called_with(dict(zip(_WHEELS, (2.0, -1.0))))

        _step_at(rolling, _STEP_S, left=0.02, right=-0.02)
        # The integral: the angle the rates carried each wheel to less its
        # turn, through each wheel's own gains.
        left = 2.0 * _STEP_S - 0.02
        right = -1.0 * _STEP_S + 0.02
        (targets,) = _wheel_targets(engine).call_args.args
        assert targets == {
            "left_wheel_joint": pytest.approx(2.0 + 64.0 / 128.0 * left),
            "right_wheel_joint": pytest.approx(-1.0 + 32.0 / 96.0 * right),
        }

    def test_a_base_with_no_setpoint_holds_its_wheels_at_rate_0(self, engine, rolling):
        _step_at(rolling, 0.0)

        _wheel_targets(engine).assert_called_with(dict.fromkeys(_WHEELS, 0.0))

    def test_the_wheels_state_is_their_counted_angle_and_measured_rate(self, engine, rolling):
        engine.views.velocities[("/World/delta", "left_wheel_joint")] = 1.5
        _step_at(rolling, 0.0, left=3.1)
        _step_at(rolling, _STEP_S, left=-3.1)

        robot, base, angles, rates = rolling._io.publish_base_states.call_args.args
        assert (robot, base) == ("delta", "base")
        assert angles == [pytest.approx(2 * 3.141592653589793 - 6.2), 0.0]
        assert rates == [1.5, 0.0]

    def test_the_robot_is_recorded_where_its_wheels_carried_it_each_step(self, engine, rolling):
        yaw = quaternion_of_yaw(0.5)
        engine.views.roots["/World/delta"] = ((1.0, 2.0, 0.01), yaw, (0.2, 0.0, 0.0), (0.0, 0.0, 0.1))
        rolling._state_pacer = Mock(take_if_due=Mock(return_value=False))

        _step_at(rolling, 0.0)

        name, placement, linear, angular = rolling._world.roll.call_args.args
        assert name == "delta"
        assert placement.position.position == (1.0, 2.0, 0.01)
        assert placement.orientation.quaternion == pytest.approx(yaw)
        assert (linear, angular) == ((0.2, 0.0, 0.0), (0.0, 0.0, 0.1))

    def test_a_robot_moved_since_the_last_step_is_set_down_at_rest(self, engine, rolling):
        rolling._io.latest_base_command.return_value = JointSetpoint(positions=(0.0, 0.0), velocities=(2.0, 2.0))
        _step_at(rolling, 0.0)
        _step_at(rolling, _STEP_S, left=0.01, right=0.01)
        rolling._world.take_set_down.return_value = {"delta"}

        _step_at(rolling, 2 * _STEP_S, left=0.04, right=0.04)

        (where,) = engine.views.set_downs
        assert where == ("/World/delta", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        rolling._io.drop_base_command.assert_called_once_with("delta")
        # Each loop reads the stage afresh: no turn counted, its integral clear.
        assert [loop.integral for loop in rolling._wheels["delta"]] == [0.0, 0.0]
        assert [loop.angle for loop in rolling._wheels["delta"]] == [pytest.approx(0.01)] * 2

    def test_a_stopping_timeline_stands_the_robot_where_it_rolled_and_its_wheels_count_on(self, engine, rolling):
        _step_at(rolling, 0.0, left=0.5)
        _step_at(rolling, _STEP_S, left=0.7)

        rolling.unbind()
        rolling.hold_through_stop()
        rolling._world.stand_where_rolled.assert_called_once_with()
        rolling.bind()
        # The timeline set the wheel back to its stage's angle.
        _step_at(rolling, 0.0, left=0.0)
        _step_at(rolling, _STEP_S, left=0.1)

        assert rolling._wheels["delta"][0].angle == pytest.approx(0.3)

    def test_views_taken_again_while_the_timeline_plays_keep_each_wheels_count_and_integral(
        self, engine, rolling
    ):
        """A prim removal drops the views and takes them again with the
        timeline playing: the wheels stand where they turned."""
        rolling._io.latest_base_command.return_value = JointSetpoint(positions=(0.0, 0.0), velocities=(2.0, 2.0))
        _step_at(rolling, 0.0, left=0.5)
        _step_at(rolling, _STEP_S, left=0.51)
        integral = rolling._wheels["delta"][0].integral

        rolling.unbind()
        rolling.bind()
        _step_at(rolling, 2 * _STEP_S, left=0.53)

        rolling._world.stand_where_rolled.assert_not_called()
        assert rolling._wheels["delta"][0].angle == pytest.approx(0.03)
        assert rolling._wheels["delta"][0].integral == pytest.approx(integral + 2.0 * _STEP_S - 0.02)
