"""The MuJoCo base image, the node's def and these suites hold one MuJoCo: the
version sim_base_images/requirements.mujoco.txt pins.

build_base_images.sh reads the version of the image's tag out of that file,
so the publisher runs here against a copy of the tree, with a docker that
records what it is asked for and builds nothing.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_NODE_DIR = Path(__file__).resolve().parents[1]
# Where the base images of both simulation nodes are built from.
_IMAGES_DIR = _NODE_DIR.parent / "sim_base_images"
_REQUIREMENTS = (_IMAGES_DIR / "requirements.mujoco.txt").read_text()
_IMAGE = "peppybot/sim-mujoco"

# What the publisher reads back from docker: the host-network entitlement of
# its builder, and the platforms that builder offers.
_FAKE_DOCKER = r"""import json
import os
import sys

with open(os.environ["MOCK_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
print("network.host linux/amd64 linux/arm64")
"""


def _from_tags(definition: str) -> list[str]:
    return re.findall(r"^From:\s+(.*)$", definition, re.MULTILINE)


def _publish(tmp_path, requirements: str = _REQUIREMENTS):
    """Runs the publisher for the MuJoCo image on a copy of the tree that
    holds `requirements`. Returns its result, the tags it asked docker to
    build and push, and the node's def as it left it."""
    images_dir = tmp_path / "sim_base_images"
    images_dir.mkdir()
    script = images_dir / "build_base_images.sh"
    shutil.copyfile(_IMAGES_DIR / script.name, script)
    (images_dir / "requirements.mujoco.txt").write_text(requirements)
    definition = tmp_path / "sim_mujoco" / "apptainer.def"
    definition.parent.mkdir()
    shutil.copyfile(_NODE_DIR / definition.name, definition)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f"#!{sys.executable}\n{_FAKE_DOCKER}")
    docker.chmod(0o755)
    docker_log = tmp_path / "docker.jsonl"
    result = subprocess.run(
        ["bash", str(script), "--mujoco-only"],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "PEPPY_R2_DATA01_ACCESS_KEY_ID": "mujoco-key",
            "PEPPY_R2_DATA01_SECRET_ACCESS_KEY": "mujoco-secret",
            "MOCK_DOCKER_LOG": str(docker_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in docker_log.read_text().splitlines()]
        if docker_log.exists()
        else []
    )
    tags = [
        call[call.index("-t") + 1] for call in calls if call[:2] == ["buildx", "build"]
    ]
    return result, tags, definition.read_text()


def test_image_tag_names_the_mujoco_the_image_installs(tmp_path):
    # Dockerfile.mujoco installs requirements.mujoco.txt, and the tag names
    # the version that file pins.
    result, tags, _ = _publish(tmp_path)
    assert result.returncode == 0, result.stderr
    [tag] = tags
    version = re.fullmatch(rf"{_IMAGE}:(.+)-[1-9][0-9]*", tag).group(1)
    assert f"mujoco=={version}" in _REQUIREMENTS.splitlines()


def test_node_builds_on_the_image_the_build_script_publishes(tmp_path):
    # The publisher stamps the tag it pushed into the node's def; a def that
    # drifts from it builds the node on another image.
    result, tags, stamped = _publish(tmp_path)
    assert result.returncode == 0, result.stderr
    assert _from_tags(stamped) == tags
    assert stamped == (_NODE_DIR / "apptainer.def").read_text()


def test_mujoco_bump_in_the_requirements_moves_the_tag_and_the_def(tmp_path):
    bumped = re.sub(r"^mujoco==.*$", "mujoco==9.8.7", _REQUIREMENTS, flags=re.MULTILINE)
    result, tags, stamped = _publish(tmp_path, bumped)
    assert result.returncode == 0, result.stderr
    [tag] = tags
    assert re.fullmatch(rf"{_IMAGE}:9\.8\.7-[1-9][0-9]*", tag)
    assert _from_tags(stamped) == [tag]


@pytest.mark.parametrize(
    "pin",
    [
        pytest.param("", id="no pin"),
        pytest.param("mujoco>=4.0.0", id="a range"),
        pytest.param("mujoco==4.0.0\nmujoco==4.0.1", id="two pins"),
    ],
)
def test_requirements_without_one_mujoco_version_stop_the_publisher(tmp_path, pin):
    # A tag needs one version to name: the publisher stops before it asks
    # docker for anything, and leaves the def as it was.
    requirements = re.sub(r"^mujoco==.*$", pin, _REQUIREMENTS, flags=re.MULTILINE)
    result, tags, stamped = _publish(tmp_path, requirements)
    assert result.returncode != 0
    assert "requirements.mujoco.txt must pin one mujoco version" in result.stderr
    assert tags == []
    assert not (tmp_path / "docker.jsonl").exists()
    assert stamped == (_NODE_DIR / "apptainer.def").read_text()


def test_suites_run_on_what_the_image_installs():
    # The scene these suites compile is compiled by the mujoco the node runs
    # on, with the numpy, the parser and the sim_robot_core the image carries.
    project = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())
    installed = _REQUIREMENTS.splitlines()
    for package in ("mujoco", "numpy", "pyjson5"):
        [pin] = [
            dependency
            for dependency in project["project"]["dependencies"]
            if dependency.startswith(f"{package}==")
        ]
        assert pin in installed, package
    rev = project["tool"]["uv"]["sources"]["sim_robot_core"]["rev"]
    assert [line for line in installed if line.startswith("sim_")] == [
        "sim_robot_core @ git+https://github.com/Peppy-bot/public-peppy-libs.git"
        f"@{rev}"
        "#subdirectory=sim_robot_core"
    ]
