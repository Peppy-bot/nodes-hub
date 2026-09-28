"""Every robot on the stage is driven as the model it joined as: its limbs
under the names its own robot answers to, its drives under its own model's
gains, its gripper opening mapped onto its own fingers' travel, and the rig
of its own model's cameras. An SO-101 stands beside an OpenArm here, on
stand-in views, so nothing of one leaks into the other.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from sim_robot_core.models import EngineModel, shipped_entry

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"

# Joint limits as each model's stage carries them: an OpenArm's fingers close
# at zero, the left ones opening toward positive and the right toward
# negative, and the SO-101's jaw closes at its lower limit.
_OPENARM_FINGER_LIMITS = {
    "openarm_left_finger_joint1": (0.0, 0.044),
    "openarm_left_finger_joint2": (0.0, 0.044),
    "openarm_right_finger_joint1": (-0.044, 0.0),
    "openarm_right_finger_joint2": (-0.044, 0.0),
}
_SO101_JAW_LIMITS = (-0.174533, 1.74533)


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
        self.limits = {}
        self.positions = {}

    def articulation(self, prim, name):
        views = self

        class Articulation:
            def setup(self):
                return True

            def teardown(self):
                pass

            def get_joint_names(self):
                return list(views.dofs[prim])

            def get_joint_limits(self):
                pairs = [views.limits.get(dof, (-3.0, 3.0)) for dof in views.dofs[prim]]
                return [lo for lo, _ in pairs], [hi for _, hi in pairs]

            def get_joint_states(self):
                positions = [views.positions.get((prim, dof), 0.0) for dof in views.dofs[prim]]
                return positions, [0.0] * len(positions)

        built = Articulation()
        self.articulations[name] = prim
        return built

    def actuator(self, prim, joint_names, params, name):
        built = Mock(spec=["setup", "teardown", "write_targets", "set_force_limit"])
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


def _models():
    import isaac_models  # pylint: disable=C0415

    return isaac_models.IsaacModels.read()


def _robot(engine, instance: str, known):
    """A robot on the stage, its articulation reporting every joint of its
    model, fingers first so no limb's rows are its entry's own order."""
    import world  # pylint: disable=C0415

    robot = world.Robot(instance=instance, known=known, placement=world.Placement.of([0, 0, 0], 0.0))
    entry = known.entry
    engine.views.dofs[robot.articulation()] = [*entry.finger_joints(), *entry.arm_joints()]
    engine.views.limits.update(_OPENARM_FINGER_LIMITS)
    engine.views.limits["gripper"] = _SO101_JAW_LIMITS
    return robot


def _limbs(engine, instance: str, model: str):
    limbs = engine.RobotLimbs(_robot(engine, instance, _models().of(model)))
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

        for arm in ("left_arm_alpha", "right_arm_alpha"):
            assert params[arm]["kp"] == [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0]
            assert params[arm]["kd"] == [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2]
            assert params[arm]["max_efforts"] == [40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0]
        assert params["left_gripper_alpha"]["kp"] == [80.0, 80.0]
        assert params["left_gripper_alpha"]["max_efforts"] == [5.0, 5.0]
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
        robot = _robot(engine, "charlo", _models().of("so101"))
        engine.views.dofs[robot.articulation()].remove("wrist_roll")

        with pytest.raises(
            RuntimeError,
            match=r"the so101 entry names joints not on the articulation of 'charlo': \['wrist_roll'\]",
        ):
            engine.RobotLimbs(robot).setup()

    def test_the_views_read_the_articulation_where_the_models_stage_keeps_it(self, engine):
        import isaac_models  # pylint: disable=C0415

        known = isaac_models.parse(
            EngineModel(
                entry=shipped_entry("so101"),
                engine={"stage": "so101/so101.usd", "articulation_root": "base_link"},
            )
        )
        robot = _robot(engine, "charlo", known)

        limbs = engine.RobotLimbs(robot)

        assert robot.prim() == "/World/charlo"
        assert engine.views.articulations == {"articulation_charlo": "/World/charlo/base_link"}
        assert engine.views.actuators["arm_charlo"][0] == "/World/charlo/base_link"
        assert list(engine.views.sensors) == [("/World/charlo/base_link", ("gripper",))]
        assert limbs.setup()


class TestFingerSpans:
    def test_the_so101_jaw_opens_from_its_lower_limit_to_its_upper_one(self, engine):
        (span,) = _limbs(engine, "charlo", "so101").spans["gripper"]
        lower, upper = _SO101_JAW_LIMITS

        assert span.position(0.0) == pytest.approx(lower)
        assert span.position(1.0) == pytest.approx(upper)
        assert span.position(0.5) == pytest.approx((lower + upper) / 2.0)
        assert span.opening(upper) == pytest.approx(1.0)
        # The jaw's zero is not its closed pose.
        assert span.opening(0.0) == pytest.approx(-lower / (upper - lower))

    def test_an_openarms_fingers_open_from_zero_each_toward_its_own_side(self, engine):
        spans = _limbs(engine, "alpha", "openarm_v2").spans

        assert [span.position(1.0) for span in spans["left_gripper"]] == [0.044, 0.044]
        assert [span.position(1.0) for span in spans["right_gripper"]] == [-0.044, -0.044]
        for gripper in ("left_gripper", "right_gripper"):
            assert [span.position(0.0) for span in spans[gripper]] == [0.0, 0.0]


@pytest.fixture(name="fleet")
def _fleet(engine, monkeypatch):
    """An OpenArm v2 and an SO-101 standing on one stage."""
    models = _models()
    robots = [
        _robot(engine, "alpha", models.of("openarm_v2")),
        _robot(engine, "charlo", models.of("so101")),
    ]
    world, io, registry = Mock(), Mock(), Mock()
    world.robots.return_value = robots
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
            ("charlo", "arm"): ([0.1, 0.2, 0.3, 0.4, 0.5], []),
            ("alpha", "right_arm"): ([0.7] * 7, [0.01] * 7),
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

    def test_a_setpoint_of_another_models_width_is_not_written(self, engine, fleet):
        """Seven positions are an OpenArm's arm, not the SO-101's five."""
        fleet._io.latest_arm_command.side_effect = lambda robot, arm: (
            ([0.7] * 7, []) if robot == "charlo" else None
        )

        fleet.step()

        _writes(engine, "arm_charlo").assert_not_called()

    def test_an_opening_is_mapped_onto_the_fingers_of_its_own_gripper(self, engine, fleet):
        commands = {("charlo", "gripper"): (0.5, 0.0), ("alpha", "right_gripper"): (1.0, 0.0)}
        fleet._io.latest_gripper_command.side_effect = (
            lambda robot, gripper: commands.get((robot, gripper))
        )

        fleet.step()

        lower, upper = _SO101_JAW_LIMITS
        (jaw,) = _writes(engine, "gripper_charlo").call_args.args
        assert jaw == {"gripper": pytest.approx((lower + upper) / 2.0)}
        (fingers,) = _writes(engine, "right_gripper_alpha").call_args.args
        assert fingers == {
            "openarm_right_finger_joint1": pytest.approx(-0.044),
            "openarm_right_finger_joint2": pytest.approx(-0.044),
        }
        _writes(engine, "left_gripper_alpha").assert_not_called()

    def test_each_robots_state_is_published_under_its_own_limbs(self, engine, fleet):
        positions = engine.views.positions
        positions[("/World/charlo", "shoulder_lift")] = -1.72
        lower, upper = _SO101_JAW_LIMITS
        positions[("/World/charlo", "gripper")] = upper
        positions[("/World/alpha", "openarm_left_joint7")] = 0.3
        positions[("/World/alpha", "openarm_right_finger_joint1")] = -0.022
        positions[("/World/alpha", "openarm_right_finger_joint2")] = -0.022

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
        assert [camera.name for camera in rigs["charlo"][1].entry.cameras] == ["front"]

    def test_a_robot_pairing_no_camera_renders_none_whatever_its_model_carries(self, engine, fleet):
        fleet._io.camera_robots.return_value = {"charlo"}

        fleet.step()

        assert [robot for robot, _, _ in engine.views.rigs] == ["charlo"]

    def test_a_model_with_no_camera_renders_none(self, engine, fleet):
        fleet._world.robots.return_value = [_robot(engine, "bravo", _models().of("openarm_v1"))]
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
