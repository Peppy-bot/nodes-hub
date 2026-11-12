"""The robots of a staged release of Waldo's catalogue, as robots.json gives
them: what each one's MuJoCo model is, and what a bad entry is refused for."""

from __future__ import annotations

import json

import entries  # noqa: F401  pylint: disable=W0611  (puts the engine on the path)
import pytest
import robot_description_py
from sim_robot_core.errors import ModelEntryError

import catalogue_robots
from catalogue_robots import ServoGains

SO101 = robot_description_py.model("so101")
ARM = {
    "id": "so101",
    "source": {"base": "https://example.invalid/so101/", "scene": "scene.xml"},
    "tree": "robots/so101/" + "1" * 64 + "/index.json",
    "description": {"urdf": "so101.urdf", "srdf": "so101.srdf"},
    "wrapper": "mounted.xml",
    "mount": {"body": "arm_mount", "stance": "mounted"},
    "linkBodies": {"gripper_link": "gripper"},
    "gravityCompensation": True,
    "ik": {
        "arms": [
            {
                "name": "arm",
                "gains": {"kp": [10.0, 20.0], "kd": [1.0, 2.0]},
                "gripper": {"actuator": "jaw", "gains": {"kp": 5.0, "kd": 0.5}},
            }
        ]
    },
}


def _read(tmp_path, *robots, stood=("so101",), urdf=SO101.urdf, srdf=SO101.srdf):
    """The robots `stood` of a release of `robots`, which ships `urdf` and
    `srdf` as the SO-101's description."""
    directory = tmp_path / "so101"
    directory.mkdir()
    (directory / "so101.urdf").write_bytes(urdf.encode())
    (directory / "so101.srdf").write_bytes(srdf.encode())
    (tmp_path / "robots.json").write_text(json.dumps(list(robots)))
    return catalogue_robots.read(tmp_path, list(stood))


def test_a_robot_is_its_wrapper_its_mount_its_bodies_and_its_gains(tmp_path):
    robot = _read(tmp_path, ARM)["so101"]
    assert robot.file == tmp_path / "so101" / "mounted.xml"
    assert robot.mount_body == "arm_mount"
    assert robot.body_of("gripper_link") == "gripper"
    assert robot.body_of("base_link") == "base_link"
    assert robot.gravity_compensation
    assert robot.arm_gains == {"arm": (ServoGains(10.0, 1.0), ServoGains(20.0, 2.0))}
    assert robot.gripper_gains == {"arm": ServoGains(5.0, 0.5)}


def test_a_robot_without_a_wrapper_stands_from_its_source_scene(tmp_path):
    robot = _read(tmp_path, {**ARM, "wrapper": None})["so101"]
    assert robot.file == tmp_path / "so101" / "scene.xml"


def test_a_robot_without_gains_runs_its_files_actuators(tmp_path):
    robot = _read(tmp_path, {**ARM, "ik": {"arms": [{"name": "arm"}]}, "gravityCompensation": False})["so101"]
    assert (robot.arm_gains, robot.gripper_gains, robot.gravity_compensation) == ({}, {}, False)


@pytest.mark.parametrize(
    "change, refusal",
    [
        ({"description": None}, "names no description, the URDF and SRDF its MJCF stands for, got None"),
        ({"description": {"urdf": "so101.urdf"}}, "names no description"),
        ({"mount": None}, "mount must be an object"),
        ({"mount": {"stance": "mounted"}}, "names no mount body"),
        ({"wrapper": None, "source": {}}, "names neither a wrapper nor a source scene"),
        ({"wrapper": None, "source": "scene.xml"}, "source must be an object"),
        ({"wrapper": 7}, "wrapper must be a name"),
        ({"linkBodies": {"gripper_link": 7}}, "linkBodies must map link names to body names"),
        ({"gravityCompensation": 1}, "gravityCompensation must be true or false"),
        ({"ik": []}, "ik must be an object"),
        ({"ik": {"arms": {"name": "arm"}}}, "ik.arms must be a list of the robot's arm groups"),
        ({"ik": {"arms": [{"gains": {}}]}}, "an arm of ik names its arm group"),
        ({"ik": {"arms": [{"name": "arm", "gripper": 5}]}}, "arm 'arm' gripper must be an object"),
        ({"ik": {"arms": [{"name": "arm", "gains": {"kp": [1.0], "kd": []}}]}}, "arm 'arm' gains need one kd per kp"),
        ({"ik": {"arms": [{"name": "arm", "gains": {"kp": [0.0], "kd": [1.0]}}]}}, r"arm 'arm' joint 0 gains need kp > 0"),
        ({"ik": {"arms": [{"name": "arm", "gains": {"kp": [1.0], "kd": [-1.0]}}]}}, r"arm 'arm' joint 0 gains need kp > 0 and kd >= 0"),
        ({"ik": {"arms": [{"name": "arm", "gripper": {"gains": {"kp": float("nan"), "kd": 1.0}}}]}}, "arm 'arm' gripper gains need"),
        ({"ik": {"arms": [{"name": "arm", "gains": [240.0]}]}}, "arm 'arm' gains need one kd per kp"),
        ({"ik": {"arms": [{"name": "arm", "gripper": {"gains": 100.0}}]}}, "arm 'arm' gripper gains need kp and kd"),
    ],
)
def test_a_bad_robot_is_refused_naming_it(tmp_path, change, refusal):
    with pytest.raises(ModelEntryError, match=f"robots.json: robot 'so101': {refusal}"):
        _read(tmp_path, {**ARM, **change})


def test_a_robot_that_names_no_description_is_refused_naming_it(tmp_path):
    undescribed = {key: value for key, value in ARM.items() if key != "description"}
    with pytest.raises(ModelEntryError, match="robots.json: robot 'so101': names no description"):
        _read(tmp_path, undescribed)


@pytest.mark.parametrize(
    "files, named",
    [
        ({"srdf": SO101.srdf + "\n"}, "so101.srdf"),
        ({"urdf": SO101.urdf + "\n", "srdf": SO101.srdf + "\n"}, "so101.urdf and so101.srdf"),
    ],
)
def test_a_description_that_is_not_robot_descriptions_is_refused_naming_what_to_pack(tmp_path, files, named):
    with pytest.raises(
        ModelEntryError,
        match=f"robots.json: robot 'so101': the release ships another {named} than "
        "robot_description's so101: pack the catalogue again with Waldo's catalogue.mjs pack",
    ):
        _read(tmp_path, ARM, **files)


def test_a_model_robot_description_does_not_bundle_is_refused_naming_it(tmp_path):
    with pytest.raises(
        ModelEntryError,
        match="robots.json: robot 'aloha': robot_description bundles no such model: "
        'no robot model is named "aloha"',
    ):
        _read(tmp_path, {**ARM, "id": "aloha"}, stood=("aloha",))


def test_only_the_robots_the_engine_stands_are_read(tmp_path):
    """A model no robot stands leaves its mount out, as the release format
    allows, and an engine that does not stand it reads nothing of it."""
    shown = {"id": "display", "source": {"base": "https://example.invalid/display/", "scene": "scene.xml"}}
    assert list(_read(tmp_path, ARM, shown)) == ["so101"]


def test_a_model_the_release_does_not_stand_is_refused_naming_the_ones_it_does(tmp_path):
    with pytest.raises(ModelEntryError, match=r"this engine stands \['aloha'\], which the release of the assets does not: it stands \['so101'\]"):
        _read(tmp_path, ARM, stood=("aloha",))


def test_a_listing_that_is_not_a_list_is_refused(tmp_path):
    (tmp_path / "robots.json").write_text(json.dumps({"robots": [ARM]}))
    with pytest.raises(ModelEntryError, match="robots.json must list the release's robots"):
        catalogue_robots.read(tmp_path, ["so101"])


def test_a_release_without_its_robots_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="the base image bakes the robots' models"):
        catalogue_robots.read(tmp_path, ["so101"])
