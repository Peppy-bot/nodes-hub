"""The scene a stand loads: the model's MJCF with the joint ranges and site
poses its entry corrects to the robot's description, the robot's weight
compensated where its entry asks, and the cameras of its rig when the engine
renders. The scene is a real (tiny) MJCF compiled by MuJoCo.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import mujoco
import pytest
from sim_robot_core.models import EngineModel, parse_entry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the launcher imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import mujoco_models  # noqa: E402  pylint: disable=C0413
from _launcher import SimLauncher, compile_spec  # noqa: E402  pylint: disable=C0413
from mujoco_models import parse  # noqa: E402  pylint: disable=C0413
from stands import Stands  # noqa: E402  pylint: disable=C0413

_SCENE = "arm/arm.xml"
_POSTURE = {"lift": 0.5, "flex": -0.8}

# A two-joint arm on a fixed base with a jointless tool below its last joint,
# beside a free prop: every kind of body gravity compensation has to tell
# apart. `lift` is driven by a position servo clamped to the joint's range
# and by a torque motor that carries no control range.
_ARM = """<mujoco model="arm">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="upper" pos="0 0 0.08">
        <joint name="lift" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        <body name="fore" pos="0 0 0.2">
          <joint name="flex" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.015" mass="0.1"/>
          <body name="tool" pos="0 0 0.15">
            <geom type="box" size="0.01 0.01 0.02" pos="0 0 0.02" mass="0.05"/>
            <site name="tool_point" pos="0 0 0.04"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.4 0 0.3">
      <freejoint name="cube"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
    <position name="flex" joint="flex" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
    <motor name="lift_torque" joint="lift"/>
  </actuator>
</mujoco>"""
_ARM_ENTRY = {"arms": [{"name": "arm", "joints": ["lift", "flex"]}]}
_FRONT_CAMERA = {
    "name": "front",
    "parent_link": "base_link",
    "pos": [0.5, 0.0, 0.3],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "fovy_deg": 60.0,
    "color": {"width": 8, "height": 4},
    "fps": 10,
}


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(tmp_path, monkeypatch):
    """The baked assets, holding the one scene these tests load."""
    monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
    scene = tmp_path / _SCENE
    scene.parent.mkdir()
    scene.write_text(_ARM)
    return tmp_path


def _known(entry=None, **engine):
    parsed = parse_entry("arm", "arm.json5", entry or _ARM_ENTRY)
    return parse(EngineModel(entry=parsed, engine={"scene": _SCENE, **engine}))


def _compile(**engine):
    return compile_spec(_known(**engine)).compile()


def _compensated(model) -> set:
    return {model.body(body).name for body in range(model.nbody) if model.body_gravcomp[body]}


def _hold(model, steps: int = 200):
    """The scene stepped from the arm's posture with its servos targeting it."""
    data = mujoco.MjData(model)
    for joint, position in _POSTURE.items():
        data.qpos[model.joint(joint).qposadr[0]] = position
        data.ctrl[model.actuator(joint).id] = position
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def _sag(model, data) -> float:
    return max(
        abs(float(data.qpos[model.joint(joint).qposadr[0]]) - position)
        for joint, position in _POSTURE.items()
    )


class TestCorrections:
    def test_an_entry_that_corrects_nothing_compiles_the_file_as_it_is(self, assets):
        baked = mujoco.MjModel.from_xml_path(str(assets / _SCENE))
        model = _compile()

        assert model.jnt_range.tolist() == baked.jnt_range.tolist()
        assert model.actuator_ctrlrange.tolist() == baked.actuator_ctrlrange.tolist()
        assert model.site_pos.tolist() == baked.site_pos.tolist()

    def test_a_joint_range_reaches_the_joint_and_the_servo_clamped_to_it(self):
        model = _compile(joint_ranges={"lift": [-1.0, 1.2]})

        assert model.joint("lift").range.tolist() == [-1.0, 1.2]
        assert model.actuator("lift").ctrlrange.tolist() == [-1.0, 1.2]
        assert model.actuator("lift").ctrllimited
        # The joint beside it keeps its file's range.
        assert model.joint("flex").range.tolist() == [-1.5, 1.5]
        assert model.actuator("flex").ctrlrange.tolist() == [-1.5, 1.5]

    def test_an_actuator_with_no_control_range_is_given_none(self):
        """A torque motor's control is no joint position, so the joint's range
        says nothing about it."""
        model = _compile(joint_ranges={"lift": [-1.0, 1.2]})

        assert not model.actuator("lift_torque").ctrllimited
        assert model.actuator("lift_torque").ctrlrange.tolist() == [0.0, 0.0]

    def test_a_site_pose_reaches_the_compiled_site(self):
        pose = {"pos": [0.01, -0.02, 0.05], "quat_wxyz": [0.0, 0.0, 1.0, 0.0]}
        model = _compile(site_poses={"tool_point": pose})

        assert model.site("tool_point").pos.tolist() == pytest.approx(pose["pos"])
        assert model.site("tool_point").quat.tolist() == pytest.approx(pose["quat_wxyz"])

    def test_a_joint_range_naming_a_joint_the_file_lacks_is_refused(self):
        entry = {"arms": [{"name": "arm", "joints": ["lift", "flex", "wrist"]}]}
        known = _known(entry, joint_ranges={"wrist": [-1.0, 1.0]})
        with pytest.raises(RuntimeError, match=f"arm: joint_ranges names 'wrist', not in {_SCENE}"):
            compile_spec(known)

    def test_a_site_pose_naming_a_site_the_file_lacks_is_refused(self):
        pose = {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
        known = _known(site_poses={"gripperframe": pose})
        with pytest.raises(RuntimeError, match=f"arm: site_poses names 'gripperframe', not in {_SCENE}"):
            compile_spec(known)

    def test_a_scene_missing_from_the_image_is_reported(self, assets):
        (assets / _SCENE).unlink()
        with pytest.raises(FileNotFoundError, match="baked into the container image"):
            compile_spec(_known())


class TestGravityCompensation:
    def test_every_body_the_models_joints_move_is_compensated_and_no_other(self):
        """The fixed base carries no joint of the model and the prop's free
        joint is not one of the model's, so neither is compensated; the tool
        carries no joint of its own and hangs below one that does."""
        model = _compile(gravity_compensation=True)

        assert _compensated(model) == {"upper", "fore", "tool"}
        assert not {"world", "base", "cube"} & _compensated(model)

    def test_compensation_starts_at_the_bodies_the_models_joints_move(self):
        entry = {"arms": [{"name": "arm", "joints": ["flex"]}]}
        model = compile_spec(_known(entry, gravity_compensation=True)).compile()

        assert _compensated(model) == {"fore", "tool"}

    def test_a_model_that_asks_for_none_compensates_no_body(self):
        model = _compile()

        assert _compensated(model) == set()
        assert model.ngravcomp == 0

    def test_a_compensated_arm_holds_its_posture_against_its_own_weight(self):
        compensated = _compile(gravity_compensation=True)
        loaded = _compile()

        assert _sag(compensated, _hold(compensated)) < 1e-9
        # The same servos alone give way under the arm's weight.
        assert _sag(loaded, _hold(loaded)) > 1e-3

    def test_the_prop_still_falls_beside_a_compensated_arm(self):
        model = _compile(gravity_compensation=True)
        dropped_from = float(mujoco.MjData(model).qpos[model.joint("cube").qposadr[0] + 2])

        data = _hold(model, steps=100)

        assert float(data.body("cube").xpos[2]) < dropped_from - 0.05

    def test_a_joint_the_file_lacks_is_refused(self):
        entry = {"arms": [{"name": "arm", "joints": ["lift", "flex", "wrist"]}]}
        known = _known(entry, gravity_compensation=True)
        with pytest.raises(
            RuntimeError, match=f"arm: gravity compensation names joint 'wrist', not in {_SCENE}"
        ):
            compile_spec(known)


def _launcher(stands=None, stop=None, renders: bool = False) -> SimLauncher:
    return SimLauncher(
        stands,
        stop or threading.Event(),
        None,
        100,
        True,
        "0.0.0.0",
        8080,
        renders=renders,
        head_camera_pack=None,
    )


class TestLoadingAScene:
    def test_a_rendering_engine_hangs_the_models_cameras_in_its_scene(self):
        entry = dict(_ARM_ENTRY, cameras=[_FRONT_CAMERA])
        known = _known(entry, link_bodies={"base_link": "base"}, gravity_compensation=True)

        rendered = _launcher(renders=True)._load_model(known)  # pylint: disable=W0212

        camera = rendered.camera("front")
        assert rendered.body(int(camera.bodyid[0])).name == "base"
        assert _compensated(rendered) == {"upper", "fore", "tool"}

    def test_an_engine_that_renders_none_loads_the_scene_without_them(self):
        entry = dict(_ARM_ENTRY, cameras=[_FRONT_CAMERA])
        known = _known(entry, link_bodies={"base_link": "base"})

        assert _launcher(renders=False)._load_model(known).ncam == 0  # pylint: disable=W0212

    def test_a_model_with_no_camera_renders_none(self):
        assert _launcher(renders=True)._load_model(_known()).ncam == 0  # pylint: disable=W0212

    def test_a_model_that_draws_the_head_camera_needs_its_pack_staged(self):
        known = _known(head_camera=True)
        with pytest.raises(RuntimeError, match="arm draws the head camera, and no pack was staged"):
            _launcher()._load_model(known)  # pylint: disable=W0212


class TestTickingWhileStanding:
    """A burst of ticks ends at the tick after which the scene is over, so a
    robot leaving a scene stepped behind real time is answered within one
    tick of the ask, not one burst."""

    def test_a_burst_runs_whole_while_the_robot_stands(self):
        launcher = _launcher(Stands(), threading.Event())
        ticks = []

        launcher._tick_while_standing(lambda: ticks.append(1), 200)  # pylint: disable=W0212

        assert len(ticks) == 200

    def test_an_unstand_asked_during_a_burst_ends_it_at_the_next_tick(self):
        stands = Stands()
        launcher = _launcher(stands, threading.Event())
        ticks = []

        def tick():
            ticks.append(1)
            if len(ticks) == 3:
                stands.unstand()

        launcher._tick_while_standing(tick, 200)  # pylint: disable=W0212

        assert len(ticks) == 3

    def test_an_engine_stopping_ticks_no_further(self):
        stop = threading.Event()
        launcher = _launcher(Stands(), stop)
        ticks = []
        stop.set()

        launcher._tick_while_standing(lambda: ticks.append(1), 200)  # pylint: disable=W0212

        assert ticks == []


def test_a_stand_whose_scene_cannot_load_fails_that_stand_alone():
    """The thread loads each stand's model, answers a scene that cannot load
    on that stand's own future, and goes on serving."""
    stands = Stands()
    stop = threading.Event()
    launcher = _launcher(stands, stop)
    asked = []

    def load_model(known):
        asked.append(known)
        stop.set()
        raise RuntimeError("this test loads no scene")

    launcher._load_model = load_model  # pylint: disable=W0212
    known = _known()
    standing = stands.stand(known, "alpha")

    launcher.run()

    assert asked == [known]
    with pytest.raises(RuntimeError, match="this test loads no scene"):
        standing.result(timeout=0)
