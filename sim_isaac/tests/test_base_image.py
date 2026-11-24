"""The base image carries pinned robot assets and the render runtime."""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

from _base_images import IMAGES_DIR, shell_value

_NODE_DIR = Path(__file__).resolve().parents[1]
_R2_ENDPOINT = "https://b9abcee11c090aef5279f874ff078826.r2.cloudflarestorage.com"


def test_node_builds_on_the_image_revision_the_build_script_publishes():
    # build_base_images.sh owns the tag and restamps the def when it publishes;
    # a def that drifts from it builds the node on a stale image.
    script = (IMAGES_DIR / "build_base_images.sh").read_text()
    image = (
        f'{shell_value(script, "IMAGE_NAMESPACE")}/sim-isaac:'
        f'{shell_value(script, "ISAAC_VERSION")}-{shell_value(script, "ISAAC_IMAGE_REV")}'
    )
    definition = (_NODE_DIR / "apptainer.def").read_text()
    assert re.findall(r"^From:\s+(.*)$", definition, re.MULTILINE) == [image]


def test_base_image_carries_the_ngx_core_library_dlss_runs_on():
    # Kit opens libnvidia-ngx.so.1 by soname; Peppy's `--nv` binding does not
    # bring the host's copy in, so the image ships one pinned by checksum, and
    # the README states the driver version it came from as the host minimum.
    dockerfile = (IMAGES_DIR / "Dockerfile.isaac").read_text()
    version = re.search(r"^ARG NGX_DRIVER_VERSION=(\S+)$", dockerfile, re.MULTILINE).group(1)
    url = re.search(r"^ARG NGX_DEB_URL=(https://\S+\.deb)$", dockerfile, re.MULTILINE).group(1)
    assert version in url
    assert re.search(r"^ARG NGX_DEB_SHA256=[0-9a-f]{64}$", dockerfile, re.MULTILINE)
    assert "sha256sum -c" in dockerfile
    assert "libnvidia-ngx.so.${NGX_DRIVER_VERSION}" in dockerfile
    assert (
        'ln -s "libnvidia-ngx.so.${NGX_DRIVER_VERSION}" '
        "/usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1"
    ) in dockerfile
    assert f"{version} or newer" in (_NODE_DIR / "README.md").read_text()


def test_base_image_copies_asset_pin_before_the_download_layer():
    dockerfile = (IMAGES_DIR / "Dockerfile.isaac").read_text()
    copy = "COPY download_assets.sh isaac_assets.env /tmp/"
    assert dockerfile.index(copy) < dockerfile.index(
        "bash /tmp/download_assets.sh"
    )
    pins = (IMAGES_DIR / "isaac_assets.env").read_text()
    assert shell_value(pins, "ISAAC_ASSETS_BUCKET") == "isaac-sim-assets"
    assert shell_value(pins, "ISAAC_ASSETS_ENDPOINT") == (
        "https://b9abcee11c090aef5279f874ff078826.r2.cloudflarestorage.com"
    )
    checksum = shell_value(pins, "ISAAC_ASSETS_SHA256")
    assert re.fullmatch(r"[0-9a-f]{64}", checksum)
    assert re.fullmatch(
        rf"openarm/[1-9][0-9]*/{checksum}\.tar\.gz",
        shell_value(pins, "ISAAC_ASSETS_KEY"),
    )


def test_base_image_bakes_one_directory_per_robot_where_the_node_looks_stages_up():
    # A model's entry names its stage relative to PEPPY_ROBOT_ASSETS_DIR, one
    # directory per robot under it: the download stages each robot's pinned
    # bundle into a directory of its name, the image bakes them all under
    # /opt/robot_assets, and the node's default points there.
    dockerfile = (IMAGES_DIR / "Dockerfile.isaac").read_text()
    assert "&& cp -r /tmp/.peppy_isaac_bundles/. /opt/robot_assets/ \\\n" in dockerfile
    download = (IMAGES_DIR / "download_assets.sh").read_text()
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-/tmp/.peppy_isaac_bundles}"' in download
    assert "ROBOTS=(openarm so101 botshare_base)" in download
    pins = (IMAGES_DIR / "isaac_assets.env").read_text()
    for robot, (pin, _) in _BUNDLES.items():
        key = f"{pin}_KEY"
        assert f'"${{{key}:?{key} must be set}}"' in download, robot
        # Each bundle is published under its robot and its own checksum.
        assert shell_value(pins, key).startswith(f"{robot}/"), robot
        checksum = shell_value(pins, f"{pin}_SHA256")
        assert shell_value(pins, key).endswith(f"/{checksum}.tar.gz"), robot
    definition = (_NODE_DIR / "apptainer.def").read_text()
    assert (
        "export PEPPY_ROBOT_ASSETS_DIR=${PEPPY_ROBOT_ASSETS_DIR:-/opt/robot_assets}\n"
        in definition
    )


def test_base_image_installs_the_packages_the_engines_share():
    # The registry, the six slots, the model entries, the camera and lidar
    # configurations and world come from sim_robot_core, the robots'
    # descriptions from robot_description_py, and the geometry types from
    # control_core_py, each pinned to the commit these suites run the engine
    # against.
    pinned = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())
    sources = pinned["tool"]["uv"]["sources"]
    requirements = (IMAGES_DIR / "requirements.isaac.txt").read_text().splitlines()
    assert [line for line in requirements if "public-peppy-libs" in line] == [
        f"{package} @ git+https://github.com/Peppy-bot/public-peppy-libs.git"
        f"@{sources[package]['rev']}"
        f"#subdirectory={package}"
        for package in ("sim_robot_core", "robot_description_py", "control_core_py")
    ]


def test_so101_stage_is_built_with_the_mujoco_the_mujoco_node_runs_on():
    # scripts/build_so101.py writes the SO-101's stage from the model as
    # MuJoCo compiles it, so the suites pin the mujoco the MuJoCo base image
    # installs: one bump moves both engines.
    project = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())
    [pin] = [
        dependency
        for dependency in project["dependency-groups"]["dev"]
        if dependency.startswith("mujoco==")
    ]
    assert pin in (IMAGES_DIR / "requirements.mujoco.txt").read_text().splitlines()


def test_node_installs_runtime_dependencies_without_asset_conversion():
    definition = (_NODE_DIR / "apptainer.def").read_text()
    for dependency in (
        "/opt/sim_isaac/scripts",
        "build_visuals",
        "requirements-visuals",
        "openarm_visuals_build",
        "venv",
        "pycollada",
        "usd-core",
        "omni.usd.libs",
    ):
        assert dependency not in definition
    assert "pycapnp" in definition
    assert "pyjson5" in definition


def test_node_stages_the_head_camera_pack_where_launch_reads_it():
    # The robot bundle is upstream's robot without its head camera; the node
    # stages Waldo's head camera pack at image build, after the %files copy
    # that brings head_camera.py in, at the directory launch.py reads at
    # setup. The staged pack is generated, so git ignores it.
    definition = (_NODE_DIR / "apptainer.def").read_text()
    module = "/opt/sim_isaac/engine/head_camera.py"
    directory = "/opt/sim_isaac/engine/assets/head_camera"
    fetch = re.search(
        rf"^\s*/isaac-sim/python\.sh {re.escape(module)} fetch \\\n\s*{re.escape(directory)}$",
        definition,
        re.MULTILINE,
    )
    assert fetch is not None
    assert definition.index("%files") < definition.index("%post") < fetch.start()
    assert re.search(r"^%post\n\s*set -e$", definition, re.MULTILINE)
    assert (_NODE_DIR / "engine" / "head_camera.py").is_file()
    launch = (_NODE_DIR / "engine" / "launch.py").read_text()
    assert '_HEAD_CAMERA_DIR = Path(__file__).parent / "assets" / "head_camera"' in launch
    ignored = (_NODE_DIR.parent / ".gitignore").read_text().splitlines()
    assert "sim_isaac/engine/assets/head_camera/" in ignored


# Small opaque USD payloads keep the installer tests independent of USD: the
# OpenArm bundle's files, the SO-101's and the botshare base's.
_OPENARM_FILES = {
    "openarm_bimanual.usd": b"original v1 stage",
    "openarm_bimanual_v2.usd": b"prepared v2 stage",
    "configuration/openarm_bimanual_base.usd": b"original base configuration",
    "configuration/openarm_bimanual_physics.usd": b"original physics configuration",
    "configuration/openarm_bimanual_sensor.usd": b"original sensor configuration",
    "openarm_v2_visuals.usdc": b"prepared material-bearing visuals",
    "openarm_visual_sources.json": b'{"fixture": "visual sources"}\n',
    "openarm_description.LICENSE.txt": (
        _NODE_DIR / "scripts" / "openarm_description.LICENSE.txt"
    ).read_bytes(),
    "bundle_manifest.json": b'{"fixture": "bundle provenance"}\n',
}
_SO101_FILES = {
    "so101.usd": b"built so101 stage",
    "LICENSE": b"upstream license",
    "bundle_manifest.json": b'{"fixture": "so101 provenance"}\n',
}
_BOTSHARE_BASE_FILES = {
    "botshare_base.usd": b"built botshare base stage",
    "LICENSE": b"meshes license",
    "LICENSING.md": b"meshes licensing",
    "NOTICE.md": b"meshes notice",
    "bundle_manifest.json": b'{"fixture": "botshare base provenance"}\n',
}

# Each bundle the installer fetches, in its order, by its robot: what the
# names of its pins in isaac_assets.env start with, and its files.
_BUNDLES = {
    "openarm": ("ISAAC_ASSETS", _OPENARM_FILES),
    "so101": ("SO101_ASSETS", _SO101_FILES),
    "botshare_base": ("BOTSHARE_BASE_ASSETS", _BOTSHARE_BASE_FILES),
}
# Each bundle the installer checks the stage of, in its order, by its robot:
# the stage's file in the bundle and the pin of its checksum.
_STAGES = {
    "so101": ("so101.usd", "SO101_STAGE_SHA256"),
    "botshare_base": ("botshare_base.usd", "BOTSHARE_BASE_STAGE_SHA256"),
}


def _fixture_key(robot: str) -> str:
    return f"{robot}/bundles/fixture.tar.gz"


def _stage_digest(robot: str) -> str:
    """The SHA-256 of the stage the fixture's bundle of `robot` holds."""
    stage, _ = _STAGES[robot]
    _, files = _BUNDLES[robot]
    return hashlib.sha256(files[stage]).hexdigest()


def _bundle(path: Path, files: dict[str, bytes]) -> str:
    """Packs `files` into the archive at `path` and returns its SHA-256."""
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            bundle.addfile(entry, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The R2 token of the isaac-sim-assets bucket.
_R2_CREDENTIALS = {"WALDO_R2_ACCESS_KEY_ID": "isaac-key", "WALDO_R2_SECRET_ACCESS_KEY": "isaac-secret"}


def _run_installer(
    tmp_path,
    *,
    corrupt=False,
    download_fails=False,
    pins=None,
    credentials=True,
):
    """Runs the installer on the fixture's bundles, the OpenArm archive
    corrupted when `corrupt`, with the pins of isaac_assets.env set to the
    fixture's and `pins` over them."""
    # The bundles the installer fetches: what the output holds after a run,
    # one directory per robot.
    archives = {robot: tmp_path / f"{robot}-bundle.tar.gz" for robot in _BUNDLES}
    digests = {robot: _bundle(archives[robot], bundle) for robot, (_, bundle) in _BUNDLES.items()}
    files = {
        f"{robot}/{name}": payload
        for robot, (_, bundle) in _BUNDLES.items()
        for name, payload in bundle.items()
    }
    if corrupt:
        archives["openarm"].write_bytes(archives["openarm"].read_bytes() + b"corruption")

    script_dir = tmp_path / "sim_base_images"
    script_dir.mkdir()
    script = script_dir / "download_assets.sh"
    shutil.copyfile(IMAGES_DIR / script.name, script)
    env = (IMAGES_DIR / "isaac_assets.env").read_text()
    for name, value in {
        **{f"{pin}_KEY": _fixture_key(robot) for robot, (pin, _) in _BUNDLES.items()},
        **{f"{pin}_SHA256": digests[robot] for robot, (pin, _) in _BUNDLES.items()},
        **{pin: _stage_digest(robot) for robot, (_, pin) in _STAGES.items()},
        **(pins or {}),
    }.items():
        env = re.sub(rf"^{name}=.*$", f'{name}="{value}"', env, flags=re.MULTILINE)
    (script_dir / "isaac_assets.env").write_text(env)

    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.usd").write_bytes(b"existing staging contents")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rclone = bin_dir / "rclone"
    rclone.write_text(
        f"#!{sys.executable}\n"
        r"""import json
import os
import shutil
import sys

with open(os.environ["MOCK_RCLONE_LOG"], "a") as log:
    log.write(json.dumps({
        "args": sys.argv[1:],
        "endpoint": os.environ["RCLONE_CONFIG_R2_ENDPOINT"],
        "access_key_id": os.environ["RCLONE_CONFIG_R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"],
    }) + "\n")
if os.environ["MOCK_RCLONE_FAIL"] == "1":
    sys.exit(9)
if sys.argv[1] == "copyto":
    shutil.copyfile(json.loads(os.environ["MOCK_BUNDLES"])[sys.argv[2]], sys.argv[3])
else:
    sys.exit("Unexpected rclone command")
"""
    )
    rclone.chmod(0o755)
    tar = bin_dir / "tar"
    tar.write_text(
        f"#!{sys.executable}\n"
        r"""import json
import os
import sys

with open(os.environ["MOCK_TAR_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
os.execv(os.environ["REAL_TAR"], ["tar", *sys.argv[1:]])
"""
    )
    tar.chmod(0o755)
    rclone_log = tmp_path / "rclone.jsonl"
    result = subprocess.run(
        ["bash", str(script), "--output-dir", str(output)],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "TMPDIR": str(scratch),
            **(_R2_CREDENTIALS if credentials else {}),
            "MOCK_RCLONE_LOG": str(rclone_log),
            "MOCK_RCLONE_FAIL": "1" if download_fails else "0",
            "MOCK_BUNDLES": json.dumps({
                f"r2:isaac-sim-assets/{_fixture_key(robot)}": str(archive)
                for robot, archive in archives.items()
            }),
            "MOCK_TAR_LOG": str(tmp_path / "tar.jsonl"),
            "REAL_TAR": shutil.which("tar", path=os.defpath),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in rclone_log.read_text().splitlines()]
        if rclone_log.exists()
        else []
    )
    return result, calls, files


def _tar_calls(tmp_path) -> list[list[str]]:
    return [json.loads(line) for line in (tmp_path / "tar.jsonl").read_text().splitlines()]


def _assert_reaches_r2_with_the_isaac_credentials(call):
    assert call["endpoint"] == _R2_ENDPOINT
    assert call["access_key_id"] == _R2_CREDENTIALS["WALDO_R2_ACCESS_KEY_ID"]
    assert call["secret_access_key"] == _R2_CREDENTIALS["WALDO_R2_SECRET_ACCESS_KEY"]


def test_isaac_installer_fetches_only_pinned_keys_and_extracts_each_bundle_under_its_robot(tmp_path):
    result, calls, files = _run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert [call["args"][:2] for call in calls] == [
        ["copyto", f"r2:isaac-sim-assets/{_fixture_key(robot)}"] for robot in _BUNDLES
    ]
    for call in calls:
        assert call["args"][3:] == ["--progress"]
        assert Path(call["args"][2]).is_relative_to(tmp_path / "scratch")
        _assert_reaches_r2_with_the_isaac_credentials(call)
    output = tmp_path / "output"
    assert {
        str(path.relative_to(output)): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == files
    # The SO-101's stage and the botshare base's are read out of their
    # archives to be checked, then each bundle is extracted.
    assert [call[0] for call in _tar_calls(tmp_path)] == ["-xzOf", "-xzOf", "-xzf", "-xzf", "-xzf"]
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("failure", ["checksum", "download"])
def test_isaac_installer_failure_does_not_extract_or_fall_back(tmp_path, failure):
    result, calls, _ = _run_installer(
        tmp_path, corrupt=failure == "checksum", download_fails=failure == "download"
    )
    assert result.returncode != 0
    if failure == "checksum":
        assert "FAILED" in result.stdout
    # The OpenArm bundle is fetched first and fails; nothing is fetched
    # after it and nothing extracted.
    assert len(calls) == 1
    assert calls[0]["args"][:2] == [
        "copyto",
        f"r2:isaac-sim-assets/{_fixture_key('openarm')}",
    ]
    assert not (tmp_path / "tar.jsonl").exists()
    assert list((tmp_path / "scratch").iterdir()) == []
    output = tmp_path / "output"
    assert list(output.iterdir()) == [output / "stale.usd"]
    assert (output / "stale.usd").read_bytes() == b"existing staging contents"


@pytest.mark.parametrize(
    ("pin", "refusal"),
    [
        (
            "ISAAC_ASSETS_SHA256",
            "the openarm bundle needs a pinned .tar.gz key and SHA-256 checksum: set ISAAC_ASSETS_KEY to "
            "its <robot>/<version>/<sha256>.tar.gz and ISAAC_ASSETS_SHA256 to its SHA-256 in isaac_assets.env",
        ),
        (
            "BOTSHARE_BASE_ASSETS_SHA256",
            "the botshare_base bundle needs a pinned .tar.gz key and SHA-256 checksum: set "
            "BOTSHARE_BASE_ASSETS_KEY to its <robot>/<version>/<sha256>.tar.gz and "
            "BOTSHARE_BASE_ASSETS_SHA256 to its SHA-256 in isaac_assets.env",
        ),
        (
            "SO101_STAGE_SHA256",
            "the so101 bundle needs the pinned SHA-256 checksum of its stage: set SO101_STAGE_SHA256 to the "
            "SHA-256 of its so101.usd in isaac_assets.env",
        ),
        (
            "BOTSHARE_BASE_STAGE_SHA256",
            "the botshare_base bundle needs the pinned SHA-256 checksum of its stage: set "
            "BOTSHARE_BASE_STAGE_SHA256 to the SHA-256 of its botshare_base.usd in isaac_assets.env",
        ),
    ],
)
@pytest.mark.parametrize("value", ["", "REPLACE_WITH_PREPARED_BUNDLE_SHA256"])
def test_isaac_installer_refuses_missing_or_placeholder_checksum(tmp_path, pin, refusal, value):
    """An empty pin is refused naming it, and a pin that is no checksum
    naming the pins to set."""
    result, calls, _ = _run_installer(tmp_path, pins={pin: value})
    assert result.returncode != 0
    assert (f"{pin} must be set" if value == "" else f"ERROR: {refusal}\n") in result.stderr
    assert calls == []
    assert not (tmp_path / "tar.jsonl").exists()
    assert (tmp_path / "output" / "stale.usd").exists()


# Each bundle whose stage is checked, and the number of stages the installer
# reads, up to and including the one it refuses.
@pytest.mark.parametrize(("robot", "stages_read"), [("so101", 1), ("botshare_base", 2)])
def test_isaac_installer_refuses_a_bundle_whose_stage_is_not_the_pinned_one(
    tmp_path, robot, stages_read
):
    # The archive is the pinned one and its stage is not the pinned stage:
    # the state a new bundle leaves when the pin of its stage does not move
    # with it.
    pinned = "0" * 64
    _, pin = _STAGES[robot]
    result, calls, _ = _run_installer(tmp_path, pins={pin: pinned})
    assert result.returncode != 0
    held = _stage_digest(robot)
    bundle = pin.removesuffix("_STAGE_SHA256")
    assert (
        f"the {robot} bundle holds the stage {held}, and the pinned stage is {pinned}: pin "
        f"{bundle}_ASSETS_KEY, {bundle}_ASSETS_SHA256 and {pin} in isaac_assets.env from one build of the bundle\n"
    ) in result.stderr
    # Every archive arrives and passes its own checksum, the stages are read
    # up to the one refused, and nothing is extracted.
    assert len(calls) == len(_BUNDLES)
    assert [call[0] for call in _tar_calls(tmp_path)] == ["-xzOf"] * stages_read
    assert list((tmp_path / "scratch").iterdir()) == []
    output = tmp_path / "output"
    assert list(output.iterdir()) == [output / "stale.usd"]


def test_installer_refuses_to_run_without_the_token_of_its_bucket(tmp_path):
    result, calls, _ = _run_installer(tmp_path, credentials=False)
    assert result.returncode != 0
    assert "WALDO_R2_ACCESS_KEY_ID must be set" in result.stderr
    assert calls == []
    output = tmp_path / "output"
    assert list(output.iterdir()) == [output / "stale.usd"]
