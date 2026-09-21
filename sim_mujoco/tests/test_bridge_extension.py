"""The physics tick of one robot's scene: the posture it starts in, the limbs
its setpoints drive under the names its pairs carry, how a gripper's opening
maps onto its finger joints, and the state it publishes back. The scenes are
real (tiny) MJCF compiled by MuJoCo, and the transport is a fake that keeps
what the bridge read and published.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import pytest
from sim_robot_core.models import EngineModel, parse_entry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import bridge_extension  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from mujoco_models import parse  # noqa: E402  pylint: disable=C0413

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
_JAW_ARM_ENTRY = {
    "arms": [{"name": "arm", "joints": ["lift", "flex"]}],
    "grippers": [{"name": "gripper", "fingers": ["jaw"], "closed_at": "lower_limit"}],
    "start_posture": _POSTURE,
}

# Two mirrored prismatic fingers that close at zero, with the symmetric slack
# an importer adds around the nominal travel.
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
_PINCHER_ENTRY = {
    "grippers": [{"name": "hand", "fingers": ["finger_up", "finger_down"], "closed_at": "zero"}],
}


class FakeIO:
    """The transport as the bridge meets it: the latest setpoint of each limb
    of each robot, and everything the bridge recorded and published."""

    def __init__(self) -> None:
        self.arm_commands: dict[tuple[str, str], tuple] = {}
        self.gripper_commands: dict[tuple[str, str], tuple] = {}
        self.engine_times: list[float] = []
        self.clock_ticks = 0
        self.arm_states: list[tuple] = []
        self.gripper_states: list[tuple] = []

    def latest_arm_command(self, robot, arm):
        return self.arm_commands.get((robot, arm))

    def latest_gripper_command(self, robot, gripper):
        return self.gripper_commands.get((robot, gripper))

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


def _known(model: str, entry: dict):
    parsed = parse_entry(model, f"{model}.json5", entry)
    return parse(EngineModel(entry=parsed, engine={"scene": f"{model}.xml"}))


def _stand(xml: str, known, io=None, time_base_s: float = 0.0):
    """A scene compiled and started the way the launcher stands it."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    io = io or FakeIO()
    bridge = MujocoBridgeExtension(
        model, data, io, ROBOT, known, _STATE_RATE_HZ, False, time_base_s
    )
    bridge.startup()
    return model, data, io, bridge


def _jaw_arm(io=None, entry=None, time_base_s: float = 0.0):
    return _stand(_JAW_ARM, _known("jaw_arm", entry or _JAW_ARM_ENTRY), io, time_base_s)


def _qpos(model, data, joint: str) -> float:
    return float(data.qpos[model.joint(joint).qposadr[0]])


def _ctrl(model, data, actuator: str) -> float:
    return float(data.ctrl[model.actuator(actuator).id])


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
        standing = data.body("jaw").xpos.copy()

        posed = mujoco.MjData(model)
        posed.qpos[:] = data.qpos
        mujoco.mj_forward(model, posed)
        unposed = mujoco.MjData(model)
        mujoco.mj_forward(model, unposed)

        assert standing == pytest.approx(posed.body("jaw").xpos)
        assert abs(standing - unposed.body("jaw").xpos).max() > 0.01

    def test_the_arm_stands_still_in_its_posture_until_its_first_setpoint(self, clock):
        """Its actuators target the posture, so they pull it nowhere else."""
        io = FakeIO()
        model, data, _, bridge = _jaw_arm(io)

        for _ in range(200):
            bridge.step()
        for joint, position in _POSTURE.items():
            assert _qpos(model, data, joint) == pytest.approx(position, abs=1e-9)

        io.arm_commands[(ROBOT, "arm")] = ([0.0, 0.0], [])
        for _ in range(200):
            bridge.step()
        assert abs(_qpos(model, data, "lift")) < _POSTURE["lift"] / 2

    def test_a_model_whose_entry_names_no_posture_starts_where_its_file_puts_it(self):
        entry = {key: value for key, value in _JAW_ARM_ENTRY.items() if key != "start_posture"}
        model, data, _, _ = _jaw_arm(entry=entry)

        assert data.qpos.tolist() == [0.0, 0.0, 0.0]
        assert data.ctrl.tolist() == [0.0, 0.0, 0.0]


class TestAJawClosedAtItsLowerLimit:
    @pytest.mark.parametrize(
        ("opening", "position"),
        [(0.0, _JAW_RANGE[0]), (1.0, _JAW_RANGE[1]), (0.5, sum(_JAW_RANGE) / 2)],
    )
    def test_an_opening_maps_onto_the_whole_range_of_the_jaw(self, clock, opening, position):
        io = FakeIO()
        io.gripper_commands[(ROBOT, "gripper")] = (opening, 0.0)
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert _ctrl(model, data, "jaw") == pytest.approx(position)

    @pytest.mark.parametrize(
        ("position", "opening"),
        [(_JAW_RANGE[0], 0.0), (_JAW_RANGE[1], 1.0), (sum(_JAW_RANGE) / 2, 0.5)],
    )
    def test_the_measured_opening_inverts_the_mapping(self, position, opening):
        model, data, io, bridge = _jaw_arm()
        data.qpos[model.joint("jaw").qposadr[0]] = position

        bridge._publish_state()  # pylint: disable=W0212

        assert io.gripper_states == [(ROBOT, "gripper", pytest.approx(opening))]

    def test_a_force_cap_reaches_the_jaws_actuator_and_is_lifted_by_zero(self, clock):
        io = FakeIO()
        model, _, _, bridge = _jaw_arm(io)
        jaw = model.actuator("jaw").id

        io.gripper_commands[(ROBOT, "gripper")] = (0.5, 1.5)
        bridge.step()
        assert model.actuator_forcerange[jaw].tolist() == [-1.5, 1.5]

        io.gripper_commands[(ROBOT, "gripper")] = (0.5, 0.0)
        bridge.step()
        assert model.actuator_forcerange[jaw].tolist() == [-3.0, 3.0]


class TestFingersClosedAtZero:
    def test_an_opening_drives_each_finger_toward_its_own_far_end(self, clock):
        io = FakeIO()
        io.gripper_commands[(ROBOT, "hand")] = (1.0, 0.0)
        model, data, _, bridge = _stand(_PINCHER, _known("pincher", _PINCHER_ENTRY), io)

        bridge.step()

        assert _ctrl(model, data, "finger_up") == pytest.approx(_FINGER_TRAVEL)
        assert _ctrl(model, data, "finger_down") == pytest.approx(-_FINGER_TRAVEL)

    def test_the_measured_opening_is_the_mean_travel_of_the_fingers(self):
        model, data, io, bridge = _stand(_PINCHER, _known("pincher", _PINCHER_ENTRY))
        data.qpos[model.joint("finger_up").qposadr[0]] = _FINGER_TRAVEL
        data.qpos[model.joint("finger_down").qposadr[0]] = -_FINGER_TRAVEL / 2

        bridge._publish_state()  # pylint: disable=W0212

        assert io.gripper_states == [(ROBOT, "hand", pytest.approx(0.75))]


class TestLimbs:
    def test_a_setpoint_drives_the_joints_of_the_limb_it_names(self, clock):
        io = FakeIO()
        io.arm_commands[(ROBOT, "arm")] = ([0.1, -0.2], [])
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == (0.1, -0.2)

    def test_another_robots_setpoint_and_another_models_limb_drive_nothing(self, clock):
        io = FakeIO()
        io.arm_commands[("alpha", "arm")] = ([0.1, -0.2], [])
        io.arm_commands[(ROBOT, "left_arm")] = ([0.3, 0.3], [])
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == tuple(_POSTURE.values())

    def test_a_setpoint_of_another_joint_count_is_left_alone(self, clock):
        io = FakeIO()
        io.arm_commands[(ROBOT, "arm")] = ([0.1] * 7, [])
        model, data, _, bridge = _jaw_arm(io)

        bridge.step()

        assert (_ctrl(model, data, "lift"), _ctrl(model, data, "flex")) == tuple(_POSTURE.values())

    def test_state_goes_back_under_the_limbs_own_names_in_joint_order(self):
        model, data, io, bridge = _jaw_arm()
        data.qvel[model.joint("flex").dofadr[0]] = 0.25

        bridge._publish_state()  # pylint: disable=W0212

        assert io.arm_states == [(ROBOT, "arm", [0.4, -0.6], [0.0, 0.25])]
        assert [(robot, gripper) for robot, gripper, _ in io.gripper_states] == [(ROBOT, "gripper")]


class TestStartup:
    def test_a_joint_of_the_entry_the_scene_lacks_is_refused_by_name(self):
        entry = dict(_JAW_ARM_ENTRY, arms=[{"name": "arm", "joints": ["lift", "flex", "wrist"]}])
        with pytest.raises(RuntimeError, match=r"jaw_arm entry names joints not in its MuJoCo model: \['wrist'\]"):
            _jaw_arm(entry=entry)

    def test_a_finger_whose_actuator_cannot_be_force_capped_is_refused(self):
        uncapped = _JAW_ARM.replace('forcelimited="true" forcerange="-3 3"', "")
        with pytest.raises(RuntimeError, match=r"lack forcelimited=\"true\".*\['jaw'\]"):
            _stand(uncapped, _known("jaw_arm", _JAW_ARM_ENTRY))

    def test_a_state_rate_is_positive(self):
        model = mujoco.MjModel.from_xml_string(_JAW_ARM)
        with pytest.raises(ValueError, match="state_rate_hz must be positive"):
            MujocoBridgeExtension(
                model, mujoco.MjData(model), FakeIO(), ROBOT, _known("jaw_arm", _JAW_ARM_ENTRY), 0, False, 0.0
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
