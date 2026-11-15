"""The robots of a staged release of Waldo's catalogue, as robots.json gives
them: what each one's MuJoCo model is, and what a bad entry is refused for."""

from __future__ import annotations

import json

import entries  # noqa: F401  pylint: disable=W0611  (puts the engine on the path)
import pytest
from sim_robot_core.errors import ModelEntryError

import catalogue_robots
from catalogue_robots import ServoGains

ARM = {
    "id": "so101",
    "source": {"base": "https://example.invalid/so101/", "scene": "scene.xml"},
    "tree": "robots/so101/" + "1" * 64 + "/index.json",
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


def _read(tmp_path, *robots, stood=("so101",)):
    """The robots `stood` of a release of `robots`."""
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
