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
from pathlib import Path

import pytest

_NODE_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _NODE_DIR.parent / "robot_initializer" / "scripts"


def _shell_value(script: str, name: str) -> str:
    return re.search(rf'^{name}="([^"]*)"', script, re.MULTILINE).group(1)


def test_node_builds_on_the_image_revision_the_build_script_publishes():
    # build_base_images.sh owns the tag and restamps the def when it publishes;
    # a def that drifts from it builds the node on a stale image.
    script = (_SCRIPTS_DIR / "build_base_images.sh").read_text()
    image = (
        f'{_shell_value(script, "IMAGE_NAMESPACE")}/openarm-isaac-sim:'
        f'{_shell_value(script, "ISAAC_VERSION")}-{_shell_value(script, "ISAAC_IMAGE_REV")}'
    )
    definition = (_NODE_DIR / "apptainer.def").read_text()
    assert re.findall(r"^From:\s+(.*)$", definition, re.MULTILINE) == [image]


def test_base_image_carries_the_ngx_core_library_dlss_runs_on():
    # Kit opens libnvidia-ngx.so.1 by soname; Peppy's `--nv` binding does not
    # bring the host's copy in, so the image ships one pinned by checksum, and
    # the README states the driver version it came from as the host minimum.
    dockerfile = (_SCRIPTS_DIR / "Dockerfile.isaac").read_text()
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
    dockerfile = (_SCRIPTS_DIR / "Dockerfile.isaac").read_text()
    copy = "COPY scripts/download_assets.sh scripts/isaac_assets.env /tmp/"
    assert dockerfile.index(copy) < dockerfile.index(
        "bash /tmp/download_assets.sh --variant isaac"
    )
    pins = (_SCRIPTS_DIR / "isaac_assets.env").read_text()
    assert _shell_value(pins, "ISAAC_ASSETS_BUCKET") == "isaac-sim-assets"
    assert _shell_value(pins, "ISAAC_ASSETS_ENDPOINT") == (
        "https://b9abcee11c090aef5279f874ff078826.r2.cloudflarestorage.com"
    )
    checksum = _shell_value(pins, "ISAAC_ASSETS_SHA256")
    assert re.fullmatch(r"[0-9a-f]{64}", checksum)
    assert re.fullmatch(
        rf"openarm/[1-9][0-9]*/{checksum}\.tar\.gz",
        _shell_value(pins, "ISAAC_ASSETS_KEY"),
    )


def test_node_installs_runtime_dependencies_without_asset_conversion():
    definition = (_NODE_DIR / "apptainer.def").read_text()
    for dependency in (
        "/opt/openarm_sim_isaac/scripts",
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


_BUNDLE_KEY = "openarm/bundles/fixture.tar.gz"


def _run_installer(
    tmp_path, *, variant="isaac", corrupt=False, download_fails=False, checksum=None
):
    # Small opaque USD payloads keep the installer tests independent of USD.
    files = {
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
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            bundle.addfile(entry, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if corrupt:
        archive.write_bytes(archive.read_bytes() + b"corruption")

    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "download_assets.sh"
    shutil.copyfile(_SCRIPTS_DIR / script.name, script)
    if variant == "isaac":
        pins = (_SCRIPTS_DIR / "isaac_assets.env").read_text()
        for name, value in (
            ("ISAAC_ASSETS_KEY", _BUNDLE_KEY),
            ("ISAAC_ASSETS_SHA256", digest if checksum is None else checksum),
        ):
            pins = re.sub(rf"^{name}=.*$", f'{name}="{value}"', pins, flags=re.MULTILINE)
        (script_dir / "isaac_assets.env").write_text(pins)

    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.usd").write_bytes(b"existing staging contents")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    mujoco_source = tmp_path / "mujoco-source"
    mujoco_source.mkdir()
    (mujoco_source / "robot.xml").write_bytes(b"MuJoCo robot")
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
    }) + "\n")
if os.environ["MOCK_RCLONE_FAIL"] == "1":
    sys.exit(9)
if sys.argv[1] == "copyto":
    shutil.copyfile(os.environ["MOCK_BUNDLE"], sys.argv[3])
elif sys.argv[1] == "copy":
    shutil.copytree(os.environ["MOCK_MUJOCO_SOURCE"], sys.argv[3], dirs_exist_ok=True)
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
        ["bash", str(script), "--variant", variant, "--output-dir", str(output)],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "TMPDIR": str(scratch),
            "RCLONE_S3_ACCESS_KEY_ID": "test-key",
            "RCLONE_S3_SECRET_ACCESS_KEY": "test-secret",
            "MOCK_RCLONE_LOG": str(rclone_log),
            "MOCK_RCLONE_FAIL": "1" if download_fails else "0",
            "MOCK_BUNDLE": str(archive),
            "MOCK_MUJOCO_SOURCE": str(mujoco_source),
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


def test_isaac_installer_fetches_only_pinned_key_and_extracts_complete_bundle(tmp_path):
    result, calls, files = _run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    command = calls[0]["args"]
    assert command[:2] == ["copyto", f"r2:isaac-sim-assets/{_BUNDLE_KEY}"]
    assert command[3:] == ["--progress"]
    assert Path(command[2]).is_relative_to(tmp_path / "scratch")
    assert calls[0]["endpoint"] == (
        "https://b9abcee11c090aef5279f874ff078826.r2.cloudflarestorage.com"
    )
    output = tmp_path / "output"
    assert {
        str(path.relative_to(output)): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == files
    assert len((tmp_path / "tar.jsonl").read_text().splitlines()) == 1
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("failure", ["checksum", "download"])
def test_isaac_installer_failure_does_not_extract_or_fall_back(tmp_path, failure):
    result, calls, _ = _run_installer(
        tmp_path, corrupt=failure == "checksum", download_fails=failure == "download"
    )
    assert result.returncode != 0
    if failure == "checksum":
        assert "FAILED" in result.stdout
    assert len(calls) == 1
    assert calls[0]["args"][:2] == [
        "copyto",
        f"r2:isaac-sim-assets/{_BUNDLE_KEY}",
    ]
    assert not (tmp_path / "tar.jsonl").exists()
    assert list((tmp_path / "scratch").iterdir()) == []
    output = tmp_path / "output"
    assert list(output.iterdir()) == [output / "stale.usd"]
    assert (output / "stale.usd").read_bytes() == b"existing staging contents"


@pytest.mark.parametrize("checksum", ["", "REPLACE_WITH_PREPARED_BUNDLE_SHA256"])
def test_isaac_installer_refuses_missing_or_placeholder_checksum(tmp_path, checksum):
    result, calls, _ = _run_installer(tmp_path, checksum=checksum)
    assert result.returncode != 0
    assert calls == []
    assert not (tmp_path / "tar.jsonl").exists()
    assert (tmp_path / "output" / "stale.usd").exists()


def test_mujoco_installer_keeps_its_directory_copy_without_isaac_pin(tmp_path):
    result, calls, _ = _run_installer(tmp_path, variant="mujoco")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0]["args"] == [
        "copy",
        "r2:peppy-data01/openarm01/mujoco/assets/",
        f"{tmp_path}/output/",
        "--progress",
    ]
    assert calls[0]["endpoint"] == (
        "https://b9abcee11c090aef5279f874ff078826.r2.cloudflarestorage.com"
    )
    output = tmp_path / "output"
    assert list(output.iterdir()) == [output / "robot.xml"]
    assert (output / "robot.xml").read_bytes() == b"MuJoCo robot"
    assert not (tmp_path / "tar.jsonl").exists()
