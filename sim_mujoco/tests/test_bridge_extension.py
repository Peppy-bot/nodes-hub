"""The physics tick of the scene: the posture each robot starts in, the limbs
its setpoints drive under the names its pairs carry, how a gripper's opening
maps onto its finger joints, the rate loop each wheel of a base runs, and the
state it publishes back. Every name a
robot answers to carries its own prefix in the composed scene, so two robots
of one model are driven apart. The scenes are real (tiny) MJCF composed and
compiled by MuJoCo, and the transport is a fake that keeps what the bridge
read and published.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from entries import (
    WHEEL_CAP_NM,
    WHEEL_INTEGRAL_GAIN,
    WHEELED,
    WHEELED_ENTRY,
    WHEELED_MOUNT,
    camera,
    entry,
    known as known_model,
    placement_at,
)
from sim_robot_core import rate_loop
from sim_robot_core.models import Arm, ModelEntry
from sim_robot_core.setpoints import MAX_VALUE, GripperSetpoint, JointSetpoint

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import bridge_extension  # noqa: E402  pylint: disable=C0413
import mujoco_models  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from exts.actuator_ctrl import home_inertia  # noqa: E402  pylint: disable=C0413
from world import World  # noqa: E402  pylint: disable=C0413

ROBOT = "charlo"
_STATE_RATE_HZ = 50
_JAW_RANGE = (-0.2, 1.7)
_POSTURE = {"lift": 0.4, "flex": -0.6}

# A two-joint arm on a fixed base carrying a single jaw. Nothing pulls on it,
# so it moves only where its actuators drive it.
_JAW_ARM = f"""<mujoco model="jaw_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="upper" pos="0 0 0.08">
        <joint name="lift" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        <body name="fore" pos="0 0 0.2">
          <joint name="flex" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.015" mass="0.1"/>
          <body name="jaw" pos="0 0 0.15">
            <joint name="jaw" type="hinge" axis="1 0 0" range="{_JAW_RANGE[0]} {_JAW_RANGE[1]}"/>
            <geom type="box" size="0.01 0.003 0.03" pos="0 0 0.03" mass="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
    <position name="flex" joint="flex" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
    <position name="jaw" joint="jaw" kp="5" kv="0.1" forcelimited="true" forcerange="-3 3"
      ctrlrange="{_JAW_RANGE[0]} {_JAW_RANGE[1]}"/>
  </actuator>
</mujoco>"""
_JAW_ARM_ENTRY = entry(
    "jaw_arm",
    arms={"arm": ["lift", "flex"]},
    grippers={"gripper": {"jaw": _JAW_RANGE}},
    start_posture=_POSTURE,
)

# Two mirrored prismatic fingers that close at zero, each joint's range the
# travel and the symmetric slack an importer adds around it.
_FINGER_TRAVEL = 0.044
_PINCHER = f"""<mujoco model="pincher">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="palm">
      <geom type="box" size="0.03 0.03 0.01"/>
      <body name="finger_up" pos="0 0.01 0.03">
        <joint name="finger_up" type="slide" axis="0 1 0" range="-0.001 {_FINGER_TRAVEL + 0.001}"/>
        <geom type="box" size="0.005 0.005 0.02" mass="0.01"/>
      </body>
      <body name="finger_down" pos="0 -0.01 0.03">
        <joint name="finger_down" type="slide" axis="0 1 0" range="{-_FINGER_TRAVEL - 0.001} 0.001"/>
        <geom type="box" size="0.005 0.005 0.02" mass="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="finger_up" joint="finger_up" kp="100" forcelimited="true" forcerange="-5 5"/>
    <position name="finger_down" joint="finger_down" kp="100" forcelimited="true" forcerange="-5 5"/>
  </actuator>
</mujoco>"""
_PINCHER_ENTRY = entry(
    "pincher",
    grippers={"hand": {"finger_up": (0.0, _FINGER_TRAVEL), "finger_down": (0.0, -_FINGER_TRAVEL)}},
)


class FakeSetpoints:
    """The setpoints as the bridge reads them: the latest of each limb and
    base of each robot, the robots whose wheels a dropped setpoint
    released, and the engine time of each read of a base."""

    def __init__(self) -> None:
        self.arms: dict[tuple[str, str], JointSetpoint] = {}
        self.grippers: dict[tuple[str, str], GripperSetpoint] = {}
        self.bases: dict[str, JointSetpoint] = {}
        self.released: set[str] = set()
        self.base_reads: list[float] = []

    def arm(self, robot, arm):
        return self.arms.get((robot, arm))

    def gripper(self, robot, gripper):
        return self.grippers.get((robot, gripper))

    def base(self, robot, now_s):
        self.base_reads.append(now_s)
        return self.bases.get(robot)

    def take_base_release(self, robot):
        released = robot in self.released
        self.released.discard(robot)
        return released


class FakeIO:
    """The transport as the bridge meets it: the setpoints it took, and
    everything the bridge recorded and published."""

    def __init__(self) -> None:
        self.setpoints = FakeSetpoints()
        self.engine_times: list[float] = []
        self.clock_ticks = 0
        self.arm_states: list[tuple] = []
        self.gripper_states: list[tuple] = []
        self.base_states: list[tuple] = []
        # Each capture of object_state: the robots' records.
        self.object_states: list[list] = []

    def publish_base_states(self, robot, positions, velocities):
        self.base_states.append((robot, positions, velocities))

    def publish_object_states(self, robots):
        self.object_states.append(robots)

    def record_engine_time(self, engine_time_s):
        self.engine_times.append(engine_time_s)

    def publish_clock_tick(self):
        self.clock_ticks += 1

    def publish_arm_states(self, robot, arm, positions, velocities):
        self.arm_states.append((robot, arm, positions, velocities))

    def publish_gripper_states(self, robot, gripper, opening):
        self.gripper_states.append((robot, gripper, opening))


@pytest.fixture(name="clock")
def clock_fixture(monkeypatch):
    """A monotonic clock the test sets, so a state publish lands on the step
    that should carry it whatever the host's speed."""

    class Clock:
        def __init__(self):
            self.now = 0.0

        def set(self, seconds):
            self.now = seconds

    clock = Clock()
    monkeypatch.setattr(bridge_extension.time, "monotonic", lambda: clock.now)
    return clock


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(tmp_path, monkeypatch):
    """Where a model's MJCF is read from, so these scenes stand as the baked
    ones do."""
    monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
    return tmp_path


# The body of each scene that carries its robot.
_MOUNTS = {"jaw_arm": "base", "pincher": "palm", "wheeled": WHEELED_MOUNT}


def _known(model: str, robot: ModelEntry, xml: str, **release):
    scene = mujoco_models.ASSETS_DIR / f"{model}.xml"
    scene.write_text(xml, encoding="utf-8")
    return known_model(model, robot, scene, _MOUNTS[model], **release)


def _stand(xml: str, known, io=None, time_base_s: float = 0.0, names=(ROBOT,), world=None):
    """A scene composed and started the way the launcher stands it, with one
    robot of `known` under each name, in `world` or a world of its own."""
    world = world or World(renders=False)
    for name in names:
        world.add(name, known, world.free_spot())
    model = world.compose().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    io = io or FakeIO()
    bridge = MujocoBridgeExtension(
        model, data, io, world, _STATE_RATE_HZ, renders=False, time_base_s=time_base_s
    )
    bridge.startup(posture_for=world.robots())
    mujoco.mj_forward(model, data)
    return model, data, io, bridge


def _jaw_arm(io=None, robot=None, time_base_s: float = 0.0, names=(ROBOT,)):
    known = _known("jaw_arm", robot or _JAW_ARM_ENTRY, _JAW_ARM)
    return _stand(_JAW_ARM, known, io, time_base_s, names)


def _qpos(model, data, joint: str, robot: str = ROBOT) -> float:
    return float(data.qpos[model.joint(f"{robot}/{joint}").qposadr[0]])


def _ctrl(model, data, actuator: str, robot: str = ROBOT) -> float:
    return float(data.ctrl[model.actuator(f"{robot}/{actuator}").id])


class TestStartPosture:
    def test_the_robot_starts_in_its_models_posture_and_is_held_there(self):
        model, data, _, _ = _jaw_arm()

        for joint, position in _POSTURE.items():
            assert _qpos(model, data, joint) == position
            assert _ctrl(model, data, joint) == position
        # The posture names the arm alone: the jaw starts where the file puts it.
        assert _qpos(model, data, "jaw") == 0.0
        assert _ctrl(model, data, "jaw") == 0.0

    def test_the_posture_reaches_the_kinematics_before_the_first_step(self):
        """A viewer or a camera reading the scene before physics first steps
        sees the robot where its joints were placed."""
        model, data, _, _ = _jaw_arm()
        standing = data.body(f"{ROBOT}/jaw").xpos.copy()

        posed = mujoco.MjData(model)
        posed.qpos[:] = data.qpos
        mujoco.mj_forward(model, posed)
        unposed = mujoco.MjData(model)
        mujoco.mj_forward(model, unposed)

        assert standing == pytest.approx(posed.body(f"{ROBOT}/jaw").xpos)
        assert abs(standing - unposed.body(f"{ROBOT}/jaw").xpos).max() > 0.01

    def test_the_arm_stands_still_in_its_posture_until_its_first_setpoint(self, clock):
        """Its actuators target the posture, so they pull it nowhere else."""
        io = FakeIO()
        model, data, _, bridge = _jaw_arm(io)

        for _ in range(200):
            bridge.step()
        for joint, position in _POSTURE.items():
            assert _qpos(model, data, joint) == pytest.approx(position, abs=1e-9)

        io.setpoints.arms[(ROBOT, "arm")] = JointSetpoint((0.0, 0.0), ())
        for _ in range(200):
            bridge.step()
        assert abs(_qpos(model, data, "lift")) < _POSTURE["lift"] / 2


class TestAJawClosedAtItsLowerLimit:
    @pytest.mark.parametrize(
        ("opening", "position"),
        [(0.0, _JAW_RANGE[0]), (1.0, _JAW_RANGE[1]), (0.5, sum(_JAW_RANGE) / 2)],
    )
    def test_an_opening_maps_onto_the_whole_range_of_the_jaw(self, clock, opening, position):
        io = FakeIO()
        io.setpoints.grippers[(ROBOT, "gripper")] = GripperSetpoint(opening, 0.0)
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert _ctrl(model, data, "jaw") == pytest.approx(position)

    @pytest.mark.parametrize(
        ("position", "opening"),
        [(_JAW_RANGE[0], 0.0), (_JAW_RANGE[1], 1.0), (sum(_JAW_RANGE) / 2, 0.5)],
    )
    def test_the_measured_opening_inverts_the_mapping(self, position, opening):
        model, data, io, bridge = _jaw_arm()
        data.qpos[model.joint(f"{ROBOT}/jaw").qposadr[0]] = position

        bridge._publish_state()  # pylint: disable=W0212

        assert io.gripper_states == [(ROBOT, "gripper", pytest.approx(opening))]

    def test_a_force_cap_reaches_the_jaws_actuator_and_is_lifted_by_zero(self, clock):
        io = FakeIO()
        model, _, _, bridge = _jaw_arm(io)
        jaw = model.actuator(f"{ROBOT}/jaw").id

        io.setpoints.grippers[(ROBOT, "gripper")] = GripperSetpoint(0.5, 1.5)
        bridge.step()
        assert model.actuator_forcerange[jaw].tolist() == [-1.5, 1.5]

        io.setpoints.grippers[(ROBOT, "gripper")] = GripperSetpoint(0.5, 0.0)
        bridge.step()
        assert model.actuator_forcerange[jaw].tolist() == [-3.0, 3.0]


# The pincher with its lower finger coupled to the upper one by an equality,
# the way an upstream gripper mirrors its second finger: one actuator drives
# the gripper.
_COUPLED_PINCHER = _PINCHER.replace(
    '    <position name="finger_down" joint="finger_down" kp="100" forcelimited="true" forcerange="-5 5"/>\n',
    "",
).replace(
    "  <actuator>",
    '  <equality>\n    <joint joint1="finger_down" joint2="finger_up" polycoef="0 -1 0 0 0"/>\n  </equality>\n  <actuator>',
)


class TestAFingerAnEqualityCouples:
    def test_the_opening_drives_the_actuated_finger_and_reads_back_from_both(self, clock):
        io = FakeIO()
        io.setpoints.grippers[(ROBOT, "hand")] = GripperSetpoint(1.0, 2.0)
        model, data, _, bridge = _stand(
            _COUPLED_PINCHER, _known("pincher", _PINCHER_ENTRY, _COUPLED_PINCHER), io
        )

        bridge.step()

        assert _ctrl(model, data, "finger_up") == pytest.approx(_FINGER_TRAVEL)
        assert model.actuator_forcerange[model.actuator(f"{ROBOT}/finger_up").id].tolist() == [-2.0, 2.0]
        data.qpos[model.joint(f"{ROBOT}/finger_up").qposadr[0]] = _FINGER_TRAVEL
        data.qpos[model.joint(f"{ROBOT}/finger_down").qposadr[0]] = -_FINGER_TRAVEL / 2
        bridge._publish_state()  # pylint: disable=W0212
        assert io.gripper_states[-1] == (ROBOT, "hand", pytest.approx(0.75))

    def test_the_start_posture_places_the_coupled_finger_and_targets_the_driven_one_alone(self, caplog):
        posture = {"finger_up": 0.02, "finger_down": -0.02}
        coupled = replace(_PINCHER_ENTRY, start_posture=posture)
        with caplog.at_level(logging.WARNING):
            model, data, _, _ = _stand(_COUPLED_PINCHER, _known("pincher", coupled, _COUPLED_PINCHER))

        assert _qpos(model, data, "finger_up") == 0.02
        assert _qpos(model, data, "finger_down") == -0.02
        assert _ctrl(model, data, "finger_up") == 0.02
        assert caplog.messages == []

    def test_a_gripper_no_actuator_drives_is_refused_by_name(self):
        unactuated = _COUPLED_PINCHER.replace(
            '    <position name="finger_up" joint="finger_up" kp="100" forcelimited="true" forcerange="-5 5"/>\n',
            "",
        )
        with pytest.raises(RuntimeError, match="no actuator of the scene standing .* drives a finger of gripper 'hand'"):
            _stand(unactuated, _known("pincher", _PINCHER_ENTRY, unactuated))


class TestFingersClosedAtZero:
    def test_an_opening_drives_each_finger_toward_its_own_far_end(self, clock):
        io = FakeIO()
        io.setpoints.grippers[(ROBOT, "hand")] = GripperSetpoint(1.0, 0.0)
        model, data, _, bridge = _stand(_PINCHER, _known("pincher", _PINCHER_ENTRY, _PINCHER), io)

        bridge.step()

        assert _ctrl(model, data, "finger_up") == pytest.approx(_FINGER_TRAVEL)
        assert _ctrl(model, data, "finger_down") == pytest.approx(-_FINGER_TRAVEL)

    def test_the_measured_opening_is_the_mean_travel_of_the_fingers(self):
        model, data, io, bridge = _stand(_PINCHER, _known("pincher", _PINCHER_ENTRY, _PINCHER))
        data.qpos[model.joint(f"{ROBOT}/finger_up").qposadr[0]] = _FINGER_TRAVEL
        data.qpos[model.joint(f"{ROBOT}/finger_down").qposadr[0]] = -_FINGER_TRAVEL / 2

        bridge._publish_state()  # pylint: disable=W0212

        assert io.gripper_states == [(ROBOT, "hand", pytest.approx(0.75))]


class TestLimbs:
    def test_a_setpoint_drives_the_joints_of_the_limb_it_names(self, clock):
        io = FakeIO()
        io.setpoints.arms[(ROBOT, "arm")] = JointSetpoint((0.1, -0.2), ())
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == (0.1, -0.2)

    def test_another_robots_setpoint_and_another_models_limb_drive_nothing(self, clock):
        io = FakeIO()
        io.setpoints.arms[("alpha", "arm")] = JointSetpoint((0.1, -0.2), ())
        io.setpoints.arms[(ROBOT, "left_arm")] = JointSetpoint((0.3, 0.3), ())
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == tuple(_POSTURE.values())

    def test_state_goes_back_under_the_limbs_own_names_in_joint_order(self):
        model, data, io, bridge = _jaw_arm()
        data.qvel[model.joint(f"{ROBOT}/flex").dofadr[0]] = 0.25

        bridge._publish_state()  # pylint: disable=W0212

        assert io.arm_states == [(ROBOT, "arm", [0.4, -0.6], [0.0, 0.25])]
        assert [(robot, gripper) for robot, gripper, _ in io.gripper_states] == [(ROBOT, "gripper")]


class TestStartup:
    def test_a_joint_of_the_entry_the_scene_lacks_is_refused_by_name(self):
        robot = replace(_JAW_ARM_ENTRY, arms=(Arm(name="arm", joints=("lift", "flex", "wrist")),))
        with pytest.raises(
            RuntimeError,
            match=rf"jaw_arm entry names joints not in the scene standing '{ROBOT}': \['wrist'\]",
        ):
            _jaw_arm(robot=robot)

    def test_a_finger_whose_actuator_cannot_be_force_capped_is_refused(self):
        uncapped = _JAW_ARM.replace('forcelimited="true" forcerange="-3 3"', "")
        with pytest.raises(RuntimeError, match=rf"lack forcelimited=\"true\".*\['{ROBOT}/jaw'\]"):
            _stand(uncapped, _known("jaw_arm", _JAW_ARM_ENTRY, uncapped))

    def test_a_state_rate_is_positive(self):
        model = mujoco.MjModel.from_xml_string(_JAW_ARM)
        with pytest.raises(ValueError, match="state_rate_hz must be positive"):
            MujocoBridgeExtension(
                model,
                mujoco.MjData(model),
                FakeIO(),
                World(renders=False),
                0,
                renders=False,
                time_base_s=0.0,
            )


class TestTheTick:
    def test_the_engine_clock_runs_on_from_the_scene_before_it(self, clock):
        model, _, io, bridge = _jaw_arm(time_base_s=12.0)

        bridge.step()
        bridge.step()

        timestep = model.opt.timestep
        assert io.engine_times == pytest.approx([12.0 + timestep, 12.0 + 2 * timestep])
        assert bridge.engine_time_s() == pytest.approx(12.0 + 2 * timestep)

    def test_physics_steps_every_tick_and_state_rides_the_state_rate(self, clock):
        _, data, io, bridge = _jaw_arm()

        # Five physics ticks inside one state period publish once.
        for _ in range(5):
            bridge.step()
        assert (len(io.arm_states), io.clock_ticks) == (1, 1)

        clock.set(1.0 / _STATE_RATE_HZ)
        bridge.step()
        assert (len(io.arm_states), io.clock_ticks) == (2, 2)
        assert len(io.engine_times) == 6
        assert data.time == pytest.approx(6 * 0.002)


class TestTheInertiaGainsAreScaledBy:
    """A servo's damping is raised to critical against the dof it drives, so
    what each gain is scaled by is that dof's own inertia at the scene's home
    configuration. The scene builds one mass matrix for every robot standing
    in it."""

    def test_each_dof_carries_its_own_inertia(self):
        model, *_ = _jaw_arm()

        inertia = home_inertia(model)

        assert len(inertia) == model.nv
        expected = np.zeros((model.nv, model.nv))
        home = mujoco.MjData(model)
        mujoco.mj_forward(model, home)
        mujoco.mj_fullM(model, home, expected)
        assert inertia == [expected[dof, dof] for dof in range(model.nv)]
        # The dofs of one arm differ from each other, so an index mistaken for
        # another dof's is a different number.
        assert len(set(inertia)) > 1

    def test_a_robot_that_joins_leaves_the_others_dofs_alone(self):
        """A robot carried onto a scene composed around a robot that joined
        keeps the gains it was tuned with."""
        alone, *_ = _jaw_arm()
        beside, *_ = _jaw_arm(names=(ROBOT, "delta"))

        standing = home_inertia(alone)
        joined = home_inertia(beside)

        assert joined[: len(standing)] == standing
        assert len(joined) == 2 * len(standing)


class TestTheGainsAServoRuns:
    def test_each_gain_is_damped_against_the_dof_its_own_actuator_drives(self):
        """Raising a servo's damping to critical needs the inertia of the dof
        it drives. Two joints of one arm carry different inertias, so a gain
        damped against another joint's is a different servo."""
        kp = (30.0, 12.0)
        known = _known(
            "jaw_arm", _JAW_ARM_ENTRY, _JAW_ARM, arm_gains={"arm": [(gain, 0.0) for gain in kp]}
        )
        model, _, _, _ = _stand(_JAW_ARM, known, None, 0.0, (ROBOT,))
        inertia = home_inertia(model)

        for joint, gain in zip(("lift", "flex"), kp):
            actuator = model.actuator(f"{ROBOT}/{joint}")
            dof = int(model.jnt_dofadr[int(actuator.trnid[0])])
            critical = 2.0 * (gain * inertia[dof]) ** 0.5

            assert float(actuator.gainprm[0]) == pytest.approx(gain)
            assert float(actuator.biasprm[1]) == pytest.approx(-gain)
            assert float(actuator.biasprm[2]) == pytest.approx(-critical)


class TestAControlMuJoCoCannotHold:
    """A setpoint within what a simulation holds can still ask a control past
    it once its velocities' feedforward, (kd/kp) times each velocity, is
    added to its positions: it is dropped, as one that is not a number is,
    and the arm holds the setpoint before it."""

    @staticmethod
    def _gained(io):
        known = _known(
            "jaw_arm", _JAW_ARM_ENTRY, _JAW_ARM, arm_gains={"arm": [(20.0, 4.0), (20.0, 4.0)]}
        )
        return _stand(_JAW_ARM, known, io, 0.0, (ROBOT,))

    def test_the_setpoint_is_dropped_and_the_arm_holds_the_one_before_it(self, caplog):
        io = FakeIO()
        model, data, _, bridge = self._gained(io)
        io.setpoints.arms[(ROBOT, "arm")] = JointSetpoint((0.3, -0.3), ())
        bridge.step()
        held = (_ctrl(model, data, "lift"), _ctrl(model, data, "flex"))
        # Each value lies at the bound, so the setpoint parses, and the
        # feedforward carries the control past it.
        io.setpoints.arms[(ROBOT, "arm")] = JointSetpoint((MAX_VALUE, 0.0), (MAX_VALUE, 0.0))

        with caplog.at_level(logging.WARNING, logger="bridge_extension"):
            bridge.step()
            bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == held == (0.3, -0.3)
        dropped = [record.getMessage() for record in caplog.records if "dropping" in record.getMessage()]
        assert len(dropped) == 1, "the arm is named once"
        assert dropped[0].startswith(
            f"dropping the setpoints for '{ROBOT}' arm until one fits: with the feedforward of its "
            "velocities it asks a control of "
        )
        assert dropped[0].endswith("and a control is a number from -1e+10 to 1e+10")

    def test_a_setpoint_that_fits_again_drives_the_arm_and_is_warned_of_anew(self, caplog):
        io = FakeIO()
        model, data, _, bridge = self._gained(io)
        beyond = JointSetpoint((MAX_VALUE, 0.0), (MAX_VALUE, 0.0))
        with caplog.at_level(logging.WARNING, logger="bridge_extension"):
            for setpoint in (beyond, JointSetpoint((0.2, 0.1), ()), beyond):
                io.setpoints.arms[(ROBOT, "arm")] = setpoint
                bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == (0.2, 0.1)
        assert sum("dropping" in record.getMessage() for record in caplog.records) == 2


def test_every_view_of_a_scene_closes_whatever_the_one_before_it_did():
    """A scene the engine has stopped reading keeps no view on the model
    about to be replaced, so the first that will not close does not strand
    the ones after it."""
    _, _, _, bridge = _jaw_arm()
    closed = []
    bridge._articulation.teardown = lambda: (_ for _ in ()).throw(RuntimeError("stuck"))
    for limbs in bridge._limbs.values():
        limbs.teardown = lambda: closed.append("limbs")

    bridge.shutdown()

    assert closed == ["limbs"]


class TestAFleetOfOneModel:
    """Two robots of one model carry the same limb and camera names, and the
    scene tells them apart by the prefix each stands under."""

    OTHER = "delta"

    def test_a_setpoint_drives_the_robot_it_names_and_no_other(self, clock):
        io = FakeIO()
        io.setpoints.arms[(ROBOT, "arm")] = JointSetpoint((0.9, -0.9), ())
        model, data, _, bridge = _jaw_arm(io, names=(ROBOT, self.OTHER))

        bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == (0.9, -0.9)
        assert (
            _ctrl(model, data, "lift", self.OTHER),
            _ctrl(model, data, "flex", self.OTHER),
        ) == tuple(_POSTURE.values())

    def test_each_robot_publishes_its_own_state_under_its_own_name(self):
        model, data, io, bridge = _jaw_arm(names=(ROBOT, self.OTHER))
        data.qpos[model.joint(f"{ROBOT}/lift").qposadr[0]] = 1.1
        data.qpos[model.joint(f"{self.OTHER}/lift").qposadr[0]] = -1.1

        bridge._publish_state()  # pylint: disable=W0212

        published = {robot: positions for robot, _, positions, _ in io.arm_states}
        assert published[ROBOT][0] == pytest.approx(1.1)
        assert published[self.OTHER][0] == pytest.approx(-1.1)

    def test_a_gripper_command_closes_the_jaw_of_the_robot_it_names(self, clock):
        io = FakeIO()
        io.setpoints.grippers[(self.OTHER, "gripper")] = GripperSetpoint(1.0, 0.0)
        model, data, _, bridge = _jaw_arm(io, names=(ROBOT, self.OTHER))

        bridge.step()

        assert _ctrl(model, data, "jaw", self.OTHER) == pytest.approx(_JAW_RANGE[1])
        # The robot nothing commanded holds the jaw where the scene left it:
        # the posture its model starts in names the arm alone.
        assert _ctrl(model, data, "jaw", ROBOT) == 0.0


_CAMERA = camera("wrist", "fore", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0), 60.0, 8, 4, 10)


class TestARenderingScene:
    """What the bridge hands the render thread: one rig per robot whose model
    carries cameras, each under the prefix that robot's cameras answer to in
    the composed scene."""

    def test_each_rig_carries_the_prefix_of_the_robot_it_renders_for(self):
        robot = replace(_JAW_ARM_ENTRY, cameras=(_CAMERA,))
        known = _known("jaw_arm", robot, _JAW_ARM)
        world = World(renders=True)
        for name in (ROBOT, "delta"):
            world.add(name, known, world.free_spot())
        model = world.compose().compile()
        bridge = MujocoBridgeExtension(
            model,
            mujoco.MjData(model),
            FakeIO(),
            world,
            _STATE_RATE_HZ,
            renders=True,
            time_base_s=0.0,
        )

        rigs = bridge._rigs()  # pylint: disable=W0212

        assert [(rig.robot, rig.prefix) for rig in rigs] == [
            (ROBOT, f"{ROBOT}/"),
            ("delta", "delta/"),
        ]
        # Every camera each rig names is a camera of the composed scene, so
        # the render thread finds what it renders.
        for rig in rigs:
            for rendered in rig.cameras:
                assert model.camera(f"{rig.prefix}{rendered.name}") is not None

    def test_a_scene_the_engine_renders_nothing_in_has_no_rig(self):
        robot = replace(_JAW_ARM_ENTRY, cameras=(_CAMERA,))
        _, _, _, bridge = _stand(_JAW_ARM, _known("jaw_arm", robot, _JAW_ARM))

        assert bridge._camera_sensor is None  # pylint: disable=W0212

    def test_a_camera_keeps_counting_its_frames_across_a_scene_composed_again(self):
        """A consumer pairs a colour frame with its depth by the frame id, so
        the count carries onto the scene the robot is carried into."""
        robot = replace(_JAW_ARM_ENTRY, cameras=(_CAMERA,))
        known = _known("jaw_arm", robot, _JAW_ARM)
        world = World(renders=True)
        world.add(ROBOT, known, world.free_spot())
        model = world.compose().compile()
        first = MujocoBridgeExtension(
            model, mujoco.MjData(model), FakeIO(), world, _STATE_RATE_HZ,
            renders=True, time_base_s=0.0,
        )
        carried = first.shutdown()
        for _ in range(3):
            carried.camera_counters[(ROBOT, "wrist")].next()

        world.add("delta", known, world.free_spot())
        composed = world.compose().compile()
        second = MujocoBridgeExtension(
            composed, mujoco.MjData(composed), FakeIO(), world, _STATE_RATE_HZ,
            renders=True, time_base_s=0.0, carried_sensors=carried,
        )

        counted = second.shutdown()
        assert counted.camera_counters[(ROBOT, "wrist")].next() == 3
        # The robot that just joined starts its own count.
        assert counted.camera_counters[("delta", "wrist")].next() == 0


def _wheeled(io=None, xml: str = WHEELED, world=None):
    return _stand(xml, _known("wheeled", WHEELED_ENTRY, xml), io, world=world)


def _wheel_ctrls(model, data) -> list[float]:
    """The controls of the wheels' rate loops: left proportional and
    integral, then right's."""
    return [
        _ctrl(model, data, f"{side}_wheel_{part}")
        for side in ("left", "right")
        for part in ("proportional", "integral")
    ]


def test_the_wheel_rule_reads_mujocos_numbers():
    """sim_robot_core's rule names MuJoCo's enumerations by their numbers,
    which this MuJoCo gives them."""
    assert (
        rate_loop.TRN_JOINT,
        rate_loop.DYN_NONE,
        rate_loop.DYN_INTEGRATOR,
        rate_loop.GAIN_FIXED,
        rate_loop.BIAS_AFFINE,
    ) == (
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtDyn.mjDYN_NONE),
        int(mujoco.mjtDyn.mjDYN_INTEGRATOR),
        int(mujoco.mjtGain.mjGAIN_FIXED),
        int(mujoco.mjtBias.mjBIAS_AFFINE),
    )


class TestTheWheelsOfABase:
    def test_each_wheel_runs_at_its_rate_of_the_latest_setpoint_and_its_position_goes_unread(self):
        io = FakeIO()
        io.setpoints.bases[ROBOT] = JointSetpoint((5.0, -5.0), (1.0, -2.0))
        model, data, _, bridge = _wheeled(io)

        bridge.step()

        assert _wheel_ctrls(model, data) == [1.0, 1.0, -2.0, -2.0]

    def test_the_wheels_hold_at_rate_0_without_a_setpoint(self):
        io = FakeIO()
        model, data, _, bridge = _wheeled(io)
        io.setpoints.bases[ROBOT] = JointSetpoint((0.0, 0.0), (1.0, 1.0))
        bridge.step()
        del io.setpoints.bases[ROBOT]

        bridge.step()

        assert _wheel_ctrls(model, data) == [0.0, 0.0, 0.0, 0.0]

    def test_released_wheels_are_held_where_they_stand_their_integrals_cleared(self):
        """A base whose leader is gone stalled against something: its loop
        stored the integral its cap holds. Released, its wheels push nothing
        and hold where they stand."""
        io = FakeIO()
        io.setpoints.bases[ROBOT] = JointSetpoint((0.0, 0.0), (1.0, 1.0))
        model, data, _, bridge = _wheeled(io)
        bridge.step()
        integrals = [
            int(model.actuator_actadr[model.actuator(f"{ROBOT}/{side}_wheel_integral").id])
            for side in ("left", "right")
        ]
        for act in integrals:
            data.act[act] += WHEEL_CAP_NM / WHEEL_INTEGRAL_GAIN
        del io.setpoints.bases[ROBOT]
        io.setpoints.released.add(ROBOT)

        angles = [_qpos(model, data, "left_wheel"), _qpos(model, data, "right_wheel")]
        bridge.step()

        assert _wheel_ctrls(model, data) == [0.0, 0.0, 0.0, 0.0]
        assert [float(data.act[act]) for act in integrals] == pytest.approx(angles)
        assert not io.setpoints.released, "a release is held once"

    def test_each_step_reads_the_bases_setpoint_at_the_engine_time_it_steps_from(self):
        """The leader's window runs on the engine's clock, the simulation
        clock, which carries on from the scenes before this one."""
        io = FakeIO()
        model, _, _, bridge = _stand(WHEELED, _known("wheeled", WHEELED_ENTRY, WHEELED), io, 12.0)

        bridge.step()
        bridge.step()

        assert io.setpoints.base_reads == pytest.approx([12.0, 12.0 + float(model.opt.timestep)])

    def test_the_integral_of_each_wheel_is_clamped_at_what_its_torque_cap_holds(self):
        model, data, _, bridge = _wheeled()
        integral = model.actuator(f"{ROBOT}/left_wheel_integral")
        act = int(model.actuator_actadr[integral.id])
        angle = _qpos(model, data, "left_wheel")
        data.act[act] = angle + 10.0

        bridge.step()

        # The step integrates a rate of 0 after the clamp, so the activation
        # stands where the clamp left it.
        windup = WHEEL_CAP_NM / WHEEL_INTEGRAL_GAIN
        assert data.act[act] == pytest.approx(angle + windup)

    def test_the_wheel_state_goes_back_left_then_right_on_the_state_rate(self, clock):
        io = FakeIO()
        io.setpoints.bases[ROBOT] = JointSetpoint((0.0, 0.0), (2.0, 4.0))
        model, data, _, bridge = _wheeled(io)

        bridge.step()

        ((robot, positions, velocities),) = io.base_states
        assert robot == ROBOT
        assert positions == [_qpos(model, data, "left_wheel"), _qpos(model, data, "right_wheel")]
        assert velocities == [
            float(data.qvel[model.joint(f"{ROBOT}/{wheel}").dofadr[0]])
            for wheel in ("left_wheel", "right_wheel")
        ]

    def test_where_the_wheels_carried_the_base_becomes_its_placement(self, clock):
        io = FakeIO()
        world = World(renders=False)
        model, data, _, bridge = _wheeled(io, world=world)
        free = model.joint(ROBOT).qposadr[0]
        data.qpos[free : free + 7] = [0.75, -0.25, 0.01, math.cos(0.3), 0.0, 0.0, math.sin(0.3)]

        bridge.step()

        (robot,) = world.robots()
        x, y, z = robot.placement.position.position
        assert (x, y) == pytest.approx((0.75, -0.25), abs=1e-3)
        assert z == pytest.approx(0.01, abs=1e-3)
        assert robot.placement.orientation.xyzw == pytest.approx(
            placement_at(0.0, yaw=0.6).orientation.xyzw, abs=1e-3
        )

    @pytest.mark.parametrize(
        ("change", "refusal"),
        [
            (('actuatorfrcrange="-4.0 4.0"', ""), "no range caps its actuator force"),
            (
                (f'<intvelocity name="left_wheel_integral" joint="left_wheel" kp="{WHEEL_INTEGRAL_GAIN}"/>', ""),
                "1 actuators drive it",
            ),
            (
                ('<velocity name="left_wheel_proportional" joint="left_wheel" kv="10"/>',
                 '<velocity name="left_wheel_proportional" joint="left_wheel" kv="10" gear="2"/>'),
                "has gear 2",
            ),
            (
                (f'<intvelocity name="left_wheel_integral" joint="left_wheel" kp="{WHEEL_INTEGRAL_GAIN}"/>',
                 f'<intvelocity name="left_wheel_integral" joint="left_wheel" kp="{WHEEL_INTEGRAL_GAIN}" actrange="-1 1"/>'),
                "clamps its activation",
            ),
            (('<joint name="left_wheel" class="wheel"/>', '<joint name="left_wheel" class="wheel" type="slide"/>'),
             "is not a hinge"),
        ],
    )
    def test_a_wheel_its_mjcf_does_not_run_as_a_rate_loop_is_refused_saying_what_to_give(self, change, refusal):
        old, new = change
        assert WHEELED.count(old) >= 1, old
        broken = WHEELED.replace(old, new, 1) if "left" in old else WHEELED.replace(old, new)
        with pytest.raises(RuntimeError, match=rf"wheel '{ROBOT}/left_wheel'.*{refusal}"):
            _wheeled(xml=broken)


@pytest.mark.parametrize(
    ("mount", "placed"),
    [
        ('pos="0.1 0 0"', "at [0.1, 0.0, 0.0], turned by [1.0, 0.0, 0.0, 0.0]"),
        ('quat="0 0 0 1"', "at [0.0, 0.0, 0.0], turned by [0.0, 0.0, 0.0, 1.0]"),
        ('euler="0 0 1.57"', "at [0.0, 0.0, 0.0], turned by [0.70738"),
    ],
    ids=["pos", "quat", "euler"],
)
def test_a_rolling_robot_whose_mount_body_stands_off_its_free_body_is_refused(mount, placed):
    """Its free joint is its robot frame, which the engine reports and
    places it by: it stands where the mount body does only while that body
    sits at the free body's origin unturned, however its MJCF writes it."""
    xml = WHEELED.replace(f'<body name="{WHEELED_MOUNT}">', f'<body name="{WHEELED_MOUNT}" {mount}>')
    assert xml != WHEELED

    with pytest.raises(RuntimeError, match=f"places its mount body '{WHEELED_MOUNT}' {re.escape(placed)}"):
        _wheeled(xml=xml)


class TestTheRecordsOnObjectState:
    """Each telemetry tick captures one record per standing robot, under the
    stamp of that tick's states: a robot fixed to the world at its placement,
    at rest, and a robot that rolls where its free joint stands, its
    velocities in world axes."""

    def test_a_robot_fixed_to_the_world_stands_at_its_placement_at_rest(self, clock):
        io = FakeIO()
        world = World(renders=False)
        world.add(ROBOT, _known("jaw_arm", _JAW_ARM_ENTRY, _JAW_ARM), placement_at(1.5, yaw=0.5))
        model = world.compose().compile()
        data = mujoco.MjData(model)
        bridge = MujocoBridgeExtension(model, data, io, world, _STATE_RATE_HZ, renders=False, time_base_s=0.0)
        bridge.startup(posture_for=world.robots())

        bridge.step()

        ((record,),) = io.object_states
        placement = placement_at(1.5, yaw=0.5)
        assert (record.robot, record.model) == (ROBOT, "jaw_arm")
        assert record.position == placement.position.position
        assert record.orientation == placement.orientation.xyzw
        assert (record.linear_velocity, record.angular_velocity) == ((0.0,) * 3, (0.0,) * 3)

    def test_a_robot_that_rolls_is_where_its_free_joint_stands_moving_in_world_axes(self, clock):
        io = FakeIO()
        model, data, _, bridge = _wheeled(io)
        free = model.joint(ROBOT)
        at, dof = free.qposadr[0], free.dofadr[0]
        # Facing +y, turned a quarter about +z, and rolling about its own +x,
        # which is the world's +y.
        data.qpos[at : at + 7] = [0.5, 0.25, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        data.qvel[dof : dof + 6] = [0.2, -0.1, 0.0, 0.3, 0.0, 0.4]
        io.object_states.clear()
        captured = {}

        def capture(robots):
            # The state the step left, which is the one the record is stamped with.
            captured["qpos"] = list(data.qpos[at : at + 7])
            captured["qvel"] = list(data.qvel[dof : dof + 6])
            io.object_states.append(robots)

        io.publish_object_states = capture
        bridge.step()

        ((record,),) = io.object_states
        assert record.position == pytest.approx(captured["qpos"][:3])
        w, x, y, z = captured["qpos"][3:]
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        assert record.orientation == pytest.approx((x / norm, y / norm, z / norm, w / norm))
        assert record.linear_velocity == pytest.approx(captured["qvel"][:3])
        # About a quarter turn about +z, which carries the body's +x onto the
        # world's +y: MuJoCo's own rotation of the body's axes into the world's.
        in_world = np.zeros(3)
        mujoco.mju_rotVecQuat(in_world, np.array(captured["qvel"][3:]), np.array(captured["qpos"][3:]) / norm)
        assert record.angular_velocity == pytest.approx(tuple(in_world), abs=1e-9)
        assert record.angular_velocity == pytest.approx((0.0, 0.3, 0.4), abs=1e-3)

    def test_an_empty_scene_captures_no_robot(self, clock):
        io = FakeIO()
        world = World(renders=False)
        model = world.compose().compile()
        bridge = MujocoBridgeExtension(
            model, mujoco.MjData(model), io, world, _STATE_RATE_HZ, renders=False, time_base_s=0.0
        )
        bridge.startup()

        bridge.step()

        assert io.object_states == [[]]

    def test_a_capture_rides_the_state_rate(self, clock):
        io = FakeIO()
        _, _, _, bridge = _jaw_arm(io)
        for _ in range(5):
            bridge.step()
        assert len(io.object_states) == 1
        clock.set(1.0 / _STATE_RATE_HZ)
        bridge.step()
        assert len(io.object_states) == 2
