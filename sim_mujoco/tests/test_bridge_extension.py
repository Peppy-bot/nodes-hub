"""The physics tick of the scene: the posture each robot starts in, the limbs
its setpoints drive under the names its pairs carry, how a gripper's opening
maps onto its finger joints, and the state it publishes back. Every name a
robot answers to carries its own prefix in the composed scene, so two robots
of one model are driven apart. The scenes are real (tiny) MJCF composed and
compiled by MuJoCo, and the transport is a fake that keeps what the bridge
read and published.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from sim_robot_core.models import EngineModel, parse_entry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import bridge_extension  # noqa: E402  pylint: disable=C0413
import mujoco_models  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from mujoco_models import parse  # noqa: E402  pylint: disable=C0413
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


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(tmp_path, monkeypatch):
    """Where a model's MJCF is read from, so these scenes stand as the baked
    ones do."""
    monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
    return tmp_path


def _known(model: str, entry: dict, xml: str, **engine):
    parsed = parse_entry(model, f"{model}.json5", entry)
    (mujoco_models.ASSETS_DIR / f"{model}.xml").write_text(xml, encoding="utf-8")
    return parse(EngineModel(entry=parsed, engine={"scene": f"{model}.xml", **engine}))


def _stand(xml: str, known, io=None, time_base_s: float = 0.0, names=(ROBOT,)):
    """A scene composed and started the way the launcher stands it, with one
    robot of `known` under each name."""
    world = World(head_camera_pack=None, renders=False)
    for name in names:
        world.add(name, known, world.free_spot())
    model = world.compose().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    io = io or FakeIO()
    robots = world.robots()
    bridge = MujocoBridgeExtension(
        model, data, io, robots, _STATE_RATE_HZ, renders=False, time_base_s=time_base_s
    )
    bridge.startup(posture_for=robots)
    mujoco.mj_forward(model, data)
    return model, data, io, bridge


def _jaw_arm(io=None, entry=None, time_base_s: float = 0.0, names=(ROBOT,)):
    known = _known("jaw_arm", entry or _JAW_ARM_ENTRY, _JAW_ARM)
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
        data.qpos[model.joint(f"{ROBOT}/jaw").qposadr[0]] = position

        bridge._publish_state()  # pylint: disable=W0212

        assert io.gripper_states == [(ROBOT, "gripper", pytest.approx(opening))]

    def test_a_force_cap_reaches_the_jaws_actuator_and_is_lifted_by_zero(self, clock):
        io = FakeIO()
        model, _, _, bridge = _jaw_arm(io)
        jaw = model.actuator(f"{ROBOT}/jaw").id

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
        data.qvel[model.joint(f"{ROBOT}/flex").dofadr[0]] = 0.25

        bridge._publish_state()  # pylint: disable=W0212

        assert io.arm_states == [(ROBOT, "arm", [0.4, -0.6], [0.0, 0.25])]
        assert [(robot, gripper) for robot, gripper, _ in io.gripper_states] == [(ROBOT, "gripper")]


class TestStartup:
    def test_a_joint_of_the_entry_the_scene_lacks_is_refused_by_name(self):
        entry = dict(_JAW_ARM_ENTRY, arms=[{"name": "arm", "joints": ["lift", "flex", "wrist"]}])
        with pytest.raises(
            RuntimeError,
            match=rf"jaw_arm entry names joints not in the scene standing '{ROBOT}': \['wrist'\]",
        ):
            _jaw_arm(entry=entry)

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
                [],
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
            "jaw_arm", _JAW_ARM_ENTRY, _JAW_ARM, arm_gains={"kp": list(kp), "kd": [0.0, 0.0]}
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
        io.arm_commands[(ROBOT, "arm")] = ([0.9, -0.9], [])
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
        io.gripper_commands[(self.OTHER, "gripper")] = (1.0, 0.0)
        model, data, _, bridge = _jaw_arm(io, names=(ROBOT, self.OTHER))

        bridge.step()

        assert _ctrl(model, data, "jaw", self.OTHER) == pytest.approx(_JAW_RANGE[1])
        # The robot nothing commanded holds the jaw where the scene left it:
        # the posture its model starts in names the arm alone.
        assert _ctrl(model, data, "jaw", ROBOT) == 0.0


_CAMERA = {
    "name": "wrist",
    "parent_link": "fore",
    "pos": [0.0, 0.0, 0.1],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "fovy_deg": 60.0,
    "color": {"width": 8, "height": 4},
    "fps": 10,
}


class TestARenderingScene:
    """What the bridge hands the render thread: one rig per robot whose model
    carries cameras, each under the prefix that robot's cameras answer to in
    the composed scene."""

    def test_each_rig_carries_the_prefix_of_the_robot_it_renders_for(self):
        entry = dict(_JAW_ARM_ENTRY, cameras=[_CAMERA])
        known = _known("jaw_arm", entry, _JAW_ARM)
        world = World(head_camera_pack=None, renders=True)
        for name in (ROBOT, "delta"):
            world.add(name, known, world.free_spot())
        model = world.compose().compile()
        bridge = MujocoBridgeExtension(
            model,
            mujoco.MjData(model),
            FakeIO(),
            world.robots(),
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
            for camera in rig.cameras:
                assert model.camera(f"{rig.prefix}{camera.name}") is not None

    def test_a_scene_the_engine_renders_nothing_in_has_no_rig(self):
        entry = dict(_JAW_ARM_ENTRY, cameras=[_CAMERA])
        _, _, _, bridge = _stand(_JAW_ARM, _known("jaw_arm", entry, _JAW_ARM))

        assert bridge._camera_sensor is None  # pylint: disable=W0212

    def test_a_camera_keeps_counting_its_frames_across_a_scene_composed_again(self):
        """A consumer pairs a colour frame with its depth by the frame id, so
        the count carries onto the scene the robot is carried into."""
        entry = dict(_JAW_ARM_ENTRY, cameras=[_CAMERA])
        known = _known("jaw_arm", entry, _JAW_ARM)
        world = World(head_camera_pack=None, renders=True)
        world.add(ROBOT, known, world.free_spot())
        model = world.compose().compile()
        first = MujocoBridgeExtension(
            model, mujoco.MjData(model), FakeIO(), world.robots(), _STATE_RATE_HZ,
            renders=True, time_base_s=0.0,
        )
        carried = first.shutdown()
        for _ in range(3):
            carried[(ROBOT, "wrist")].next()

        world.add("delta", known, world.free_spot())
        composed = world.compose().compile()
        second = MujocoBridgeExtension(
            composed, mujoco.MjData(composed), FakeIO(), world.robots(), _STATE_RATE_HZ,
            renders=True, time_base_s=0.0, camera_counters=carried,
        )

        counted = second.shutdown()
        assert counted[(ROBOT, "wrist")].next() == 3
        # The robot that just joined starts its own count.
        assert counted[("delta", "wrist")].next() == 0
