"""The render profile runs on what the base image carries; pin that contract."""

import re
from pathlib import Path

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
