"""The scene the robots standing compose: each one's MJCF with the joint
ranges and site poses its entry corrects to the robot's description, its
weight compensated where its entry asks, and the cameras of its rig when the
engine renders, attached under its own prefix at its own placement. The
scenes are real (tiny) MJCF compiled by MuJoCo.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import pytest
from sim_robot_core.models import EngineModel, parse_entry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the world imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import mujoco_models  # noqa: E402  pylint: disable=C0413
from exts import camera_sensor  # noqa: E402  pylint: disable=C0413
from mujoco_models import parse  # noqa: E402  pylint: disable=C0413
import world as world_module  # noqa: E402  pylint: disable=C0413
from world import SPOT_PITCH_M, Placement, World, compile_spec  # noqa: E402  pylint: disable=C0413

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


def _world(renders: bool = False, pack=None) -> World:
    return World(head_camera_pack=pack, renders=renders)


def _standing(names, known=None, renders: bool = False, placements=None):
    """A scene of one robot per name, each at the spot given for it or the
    next free one."""
    world = _world(renders)
    known = known or _known()
    for name in names:
        placement = (placements or {}).get(name) or world.free_spot()
        world.add(name, known, placement)
    return world, world.compose().compile()


def _named(model, kind, count) -> list[str]:
    return [mujoco.mj_id2name(model, kind, index) for index in range(count)]


class TestComposingTheScene:
    def test_an_empty_scene_compiles_and_holds_nothing(self):
        model = _world().compose().compile()

        assert model.njnt == 0
        assert model.nbody == 1  # the world body alone

    def test_each_robot_carries_its_own_prefix_on_every_name_it_answers_to(self):
        _, model = _standing(["alpha", "bravo"])

        joints = _named(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        actuators = _named(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
        assert joints == ["alpha/lift", "alpha/flex", "alpha/cube", "bravo/lift", "bravo/flex", "bravo/cube"]
        assert [name.split("/")[0] for name in actuators] == ["alpha"] * 3 + ["bravo"] * 3

    def test_a_robot_stands_where_its_placement_puts_it(self):
        spot = Placement.of((2.0, -1.0, 0.0), 0.0)
        world, model = _standing(["alpha"], placements={"alpha": spot})
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        (standing,) = world.robots()
        assert standing.placement == spot
        assert data.body("alpha/base").xpos.tolist() == pytest.approx([2.0, -1.0, 0.05])

    def test_a_robot_turned_about_z_stands_turned(self):
        """The prop this model carries sits out along +x in the robot's own
        frame, so a quarter turn about +z puts it on +y."""
        world = _world()
        world.add("alpha", _known(), Placement.of((0.0, 0.0, 0.0), math.pi / 2))
        model = world.compose().compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        prop = data.body("alpha/cube").xpos
        assert prop.tolist() == pytest.approx([0.0, 0.4, 0.3], abs=1e-9)

    def test_taking_a_robot_out_composes_the_scene_without_it(self):
        world, _ = _standing(["alpha", "bravo"])

        world.remove("alpha")
        model = world.compose().compile()

        assert [robot.instance for robot in world.robots()] == ["bravo"]
        assert all(
            name.startswith("bravo/")
            for name in _named(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        )

    def test_a_robot_of_the_same_model_stands_beside_its_twin(self):
        """Two robots of one model keep their own joints, so a setpoint for
        one never reaches the other."""
        _, model = _standing(["alpha", "bravo"])

        assert model.njnt == 2 * mujoco.MjModel.from_xml_string(_ARM).njnt


class TestTheSpotsRobotsTake:
    def test_the_first_robot_stands_on_the_origin(self):
        assert _world().free_spot().position == (0.0, 0.0, 0.0)

    def test_the_next_robot_takes_a_spot_of_its_own(self):
        world, _ = _standing(["alpha"])

        spot = world.free_spot()

        assert spot.position != (0.0, 0.0, 0.0)
        assert not world.occupied(spot)

    def test_a_spot_promised_to_a_robot_that_has_not_stood_is_taken(self):
        world = _world()
        promised = world.free_spot()

        assert world.free_spot(promised=(promised,)) != promised
        assert world.occupied(promised, promised=(promised,))

    def test_the_robot_standing_within_a_spot_is_the_one_it_names(self):
        world, _ = _standing(["alpha"], placements={"alpha": Placement.of((0.0, 0.0, 0.0), 0.0)})

        assert world.standing_within(Placement.of((0.1, 0.0, 0.0), 0.0)).instance == "alpha"
        assert world.standing_within(Placement.of((SPOT_PITCH_M, 0.0, 0.0), 0.0)) is None

    def test_a_spot_is_taken_at_any_height(self):
        """A robot's placement carries its height, and two robots standing
        one above the other are still one inside the other."""
        world, _ = _standing(["alpha"], placements={"alpha": Placement.of((0.0, 0.0, 0.0), 0.0)})

        assert world.occupied(Placement.of((0.0, 0.0, SPOT_PITCH_M / 2), 0.0))
        assert not world.occupied(Placement.of((0.0, 0.0, SPOT_PITCH_M), 0.0))

    def test_a_full_floor_is_refused_the_way_a_taken_spot_is(self, monkeypatch):
        """Admission answers the goal with the reason its robot was refused,
        and reads every one of those reasons off a ValueError. Two rings put
        a robot on the origin and on the eight spots around it."""
        monkeypatch.setattr(world_module, "_MAX_RINGS", 2)
        taken = tuple(
            Placement.of((row * SPOT_PITCH_M, column * SPOT_PITCH_M, 0.0), 0.0)
            for row in (-1, 0, 1)
            for column in (-1, 0, 1)
        )

        with pytest.raises(ValueError, match=r"every spot this scene lays out is taken"):
            _world().free_spot(taken)

    def test_a_robot_nearer_than_the_lattice_leaves_them_counts_as_on_the_spot(self):
        """It is the distance that counts, not which lattice square a spot
        falls in: two robots nearer than the lattice leaves them resolve
        their overlap by throwing each other."""
        world, _ = _standing(["alpha"], placements={"alpha": Placement.of((0.0, 0.0, 0.0), 0.0)})

        assert world.occupied(Placement.of((SPOT_PITCH_M / 4, 0.0, 0.0), 0.0))
        # A hand-picked spot just the other side of a lattice line is still
        # within reach of the robot standing.
        assert world.occupied(Placement.of((SPOT_PITCH_M * 0.51, 0.0, 0.0), 0.0))
        assert not world.occupied(Placement.of((SPOT_PITCH_M, 0.0, 0.0), 0.0))

    @pytest.mark.parametrize("position", [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)])
    def test_a_placement_that_is_not_finite_is_refused(self, position):
        with pytest.raises(ValueError, match="finite in every coordinate"):
            Placement.of(position, 0.0)

    def test_a_placement_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="3 coordinates"):
            Placement.of((0.0, 0.0), 0.0)

    def test_a_yaw_that_is_not_finite_is_refused(self):
        with pytest.raises(ValueError, match="finite in every coordinate"):
            Placement.of((0.0, 0.0, 0.0), float("nan"))


class TestTheSettingsTheModelsShare:
    def test_the_scene_runs_on_the_settings_its_models_ask_for(self):
        """MuJoCo drops a model's own `<option>` when it is attached, so the
        scene takes the settings of the models standing in it."""
        world = _world()
        world.add("alpha", _known(), world.free_spot())

        model = world.compose().compile()

        baked = mujoco.MjModel.from_xml_string(_ARM)
        assert model.opt.timestep == baked.opt.timestep
        assert model.opt.integrator == baked.opt.integrator

    def test_two_models_asking_for_different_settings_are_refused_by_name(self):
        other = _ARM.replace('timestep="0.002"', 'timestep="0.004"')
        (mujoco_models.ASSETS_DIR / "arm" / "other.xml").write_text(other)
        slower = parse(
            EngineModel(
                entry=parse_entry("slow_arm", "slow_arm.json5", _ARM_ENTRY),
                engine={"scene": "arm/other.xml"},
            )
        )
        world = _world()
        world.add("alpha", _known(), world.free_spot())
        world.add("bravo", slower, world.free_spot())

        with pytest.raises(
            RuntimeError,
            match=r"models 'arm' and 'slow_arm' ask for different simulation settings "
            r"\(option.timestep\).*give both models the same settings, or stand the "
            r"slow_arm in a simulation of its own",
        ):
            world.compose()

    def test_two_robots_of_one_model_ask_for_the_same_settings(self):
        """A statistic a model leaves out reads as NaN, which is not equal to
        itself: two robots of one model still agree."""
        _, model = _standing(["alpha", "bravo"])

        assert model.njnt > 0


class TestFittingOutTheScene:
    def test_a_rendering_engine_hangs_each_robots_cameras_under_its_prefix(self):
        entry = dict(_ARM_ENTRY, cameras=[_FRONT_CAMERA])
        known = _known(entry, link_bodies={"base_link": "base"}, gravity_compensation=True)

        _, model = _standing(["alpha", "bravo"], known=known, renders=True)

        assert _named(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam) == [
            "alpha/front",
            "bravo/front",
        ]
        assert model.body(int(model.camera("alpha/front").bodyid[0])).name == "alpha/base"
        assert _compensated(model) == {
            f"{robot}/{body}" for robot in ("alpha", "bravo") for body in ("upper", "fore", "tool")
        }

    def test_an_engine_that_renders_none_composes_the_scene_without_them(self):
        entry = dict(_ARM_ENTRY, cameras=[_FRONT_CAMERA])
        known = _known(entry, link_bodies={"base_link": "base"})

        _, model = _standing(["alpha"], known=known, renders=False)

        assert model.ncam == 0

    def test_a_model_with_no_camera_renders_none(self):
        _, model = _standing(["alpha"], renders=True)

        assert model.ncam == 0

    def test_the_light_rig_lights_the_scene_once_however_many_robots_stand(self):
        entry = dict(_ARM_ENTRY, cameras=[_FRONT_CAMERA])
        known = _known(entry, link_bodies={"base_link": "base"}, camera_lights=True)

        _, model = _standing(["alpha", "bravo"], known=known, renders=True)

        assert model.nlight == len(camera_sensor._LIGHT_DIRECTIONS)  # pylint: disable=W0212

    def test_a_model_that_draws_the_head_camera_needs_its_pack_staged(self):
        world = _world()
        world.add("alpha", _known(head_camera=True), world.free_spot())

        with pytest.raises(RuntimeError, match="arm draws the head camera, and no pack was staged"):
            world.compose()

    def test_the_floor_is_laid_once_for_every_robot_that_works_against_one(self):
        """Two robots of a model that works against a floor work against the
        same one: the scene lays it, so nothing rests on two planes at once.
        A robot's own name carries its prefix, so the scene's floor is the
        one that carries none."""
        _, model = _standing(["alpha", "bravo"], known=_known(floor=True))

        floors = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(model.ngeom)
            if model.geom(index).type[0] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        assert floors.count("floor") == 1
        assert model.light("floor_light").type[0] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    def test_a_scene_no_model_works_against_a_floor_in_lays_none(self):
        _, model = _standing(["alpha"])

        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1


class TestTheRecordOfWhoStands:
    def test_a_name_standing_twice_is_refused(self):
        world, _ = _standing(["alpha"])

        with pytest.raises(ValueError, match=r"'alpha' is still in the scene"):
            world.add("alpha", _known(), world.free_spot())

    def test_a_robot_with_no_name_is_refused(self):
        with pytest.raises(ValueError, match="stands under the name of the copy it runs as"):
            _world().add("", _known(), Placement.of((0.0, 0.0, 0.0), 0.0))

    def test_a_name_carrying_the_prefix_separator_is_refused(self):
        """Every name a robot answers to in the scene is its own name and
        that separator, so a name carrying one names nothing it owns."""
        with pytest.raises(ValueError, match="carrying no '/'"):
            _world().add("alpha/bravo", _known(), Placement.of((0.0, 0.0, 0.0), 0.0))

    def test_a_name_carrying_the_separator_is_told_what_to_join_under(self):
        with pytest.raises(ValueError, match=r"peppy stack join LAUNCHER -i left_arm"):
            _world().add("left/arm", _known(), Placement.of((0.0, 0.0, 0.0), 0.0))

    def test_taking_out_a_robot_that_stands_nowhere_changes_nothing(self):
        world, _ = _standing(["alpha"])

        world.remove("ghost")

        assert [robot.instance for robot in world.robots()] == ["alpha"]
