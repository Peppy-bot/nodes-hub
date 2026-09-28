#!/usr/bin/env python3
"""The OpenArm v2 head camera the MuJoCo robot carries: the ZED Mini on its
bracket under the head cover, as Enactic's OpenArm_2.0_w_Head_Camera CAD
assembly seats it on the pedestal.

Upstream's openarm_mujoco, which the base image's scene is baked from, has
no head camera. Waldo's tools/openarm_head_camera (private-nodes-hub) reads
it off the CAD into an immutable pack under the public waldo-assets store:
three OBJ meshes and one convex collision hull, all in the frame of the CAD
mount's origin, the stereo rig read off the lens barrels, and provenance. A
pack is keyed by the SHA-256 of its inventory, so the digest pinned here
names one exact set of files for good.

`fetch` stages that pack into the node image (apptainer.def runs it at
build), `load` reads the staged pack once at node setup, and `attach` adds
its meshes to a scene's spec on the pedestal, the same body Waldo's
openarm_v2 wrapper carries, so the MuJoCo robot draws and collides with the
head camera the way Waldo's does. The chest camera of a model that draws the
pack, whose pose the camera sensor renders from, is checked to sit at the
pack's left lens front, the eye the real ZED's rectified stream comes from.

    python3 head_camera.py fetch <directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Optional
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)

STORE_URL = "https://waldo-assets.peppy.bot"
# The store sits behind Cloudflare, which answers urllib's default agent with
# 403; a named one gets through.
USER_AGENT = "sim_mujoco/head_camera.py"
PACK_PREFIX = "sources/openarm_v2_head_camera"
# The derivation of 2026-09-17 from the STEP at openarm_hardware eefc9fa: the
# pack Waldo's openarm_v2 robot names as the overlay of its head_camera/
# directory, in the catalogue release its catalogue.lock.json pins.
PACK_DIGEST = "78054986a1cb6f0725bb2b8fd23e3d3078fc52f06ef4c01fe12c67c1a53dd71b"
INDEX_FILE = "index.json"
RIG_FILE = "cameras.json"
VISUAL_MESHES = ("zed_mini", "mount", "cover")
COLLISION_FILE = "collision.stl"
# The wrapper's body the pack's meshes are framed in, seated on the pedestal
# link, which is also the chest camera's parent. The MJCF compiler folds the
# fixed pedestal link into the world body (its geoms keep the link's name as
# their prefix), so the head camera body hangs off the world body, as Waldo's
# hangs off its mount.
BODY_NAME = "openarm_head_camera"
PEDESTAL_LINK = "openarm_body_link0"
CHEST_CAMERA = "chest"
# Waldo's names for the pack's meshes and their geoms.
MESH_PREFIX = "head_camera_"
# Waldo draws the whole mount in upstream openarm_mujoco's matte_black, the
# material the scene's arm links already carry.
MATERIAL = "matte_black"
# The model's entry rounds the rig to a tenth of a millimetre and seven
# decimals.
POSITION_TOLERANCE_M = 1e-4
ORIENTATION_TOLERANCE = 1e-5
# Waldo's wrapper geoms: the visuals in the scene's visual group with no
# contact, the hull with the pedestal's own collision settings in the
# collision group, so the viewer shows the meshes and the arms touch the hull.
_VISUAL_GROUP = 2
_COLLISION_GROUP = 3
_COLLISION_CONDIM = 3
_COLLISION_PRIORITY = 1
_COLLISION_SOLREF = (0.005, 1.0)
_COLLISION_FRICTION = (1.0, 0.01, 0.01)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_files() -> tuple[str, ...]:
    """The files the robot draws: the meshes, the hull and the rig."""
    return tuple(f"{name}.obj" for name in VISUAL_MESHES) + (COLLISION_FILE, RIG_FILE)


def pack_digest(inventory: dict[str, tuple[int, str]]) -> str:
    """The key a pack is published under: the SHA-256 of its sorted inventory
    of names, sizes and content hashes, the rule tools/openarm_head_camera
    writes packs by. The index is not in its own inventory."""
    files = [
        {"path": name, "size": size, "sha256": digest}
        for name, (size, digest) in sorted(inventory.items())
    ]
    document = json.dumps({"version": 1, "files": files}, separators=(",", ":")) + "\n"
    return sha256(document.encode())


def read_index(data: bytes, digest: Optional[str] = None) -> dict[str, tuple[int, str]]:
    """The inventory of the pack's index.json: name to (size, sha256). Refused
    unless it lists every file the robot draws and hashes to the digest, so a
    served index that is not the pinned pack's stages nothing."""
    digest = digest or PACK_DIGEST
    prefix = f"{PACK_PREFIX}/{digest}/"
    inventory: dict[str, tuple[int, str]] = {}
    for entry in json.loads(data)["files"]:
        key = str(entry["key"])
        name = key[len(prefix):]
        if not key.startswith(prefix) or not name or "/" in name:
            raise ValueError(f"pack index lists {key!r} outside {prefix}")
        inventory[name] = (int(entry["size"]), str(entry["sha256"]))
    missing = sorted(set(pack_files()) - inventory.keys())
    if missing:
        raise ValueError(f"pack index is missing {missing}")
    actual = pack_digest(inventory)
    if actual != digest:
        raise ValueError(f"pack index hashes to {actual}, not the pinned {digest}")
    return inventory


def verify(pack: Path, digest: Optional[str] = None) -> dict[str, tuple[int, str]]:
    """Checks a staged pack file by file against its index and the index
    against the digest. Every start reads the pack through this, so what the
    robot draws is the pinned derivation or nothing."""
    index = pack / INDEX_FILE
    if not index.is_file():
        raise FileNotFoundError(
            f"head camera pack not staged at {pack}: the node image stages it at "
            f"build (apptainer.def); a native run needs "
            f"`python3 {Path(__file__).name} fetch {pack}`"
        )
    inventory = read_index(index.read_bytes(), digest)
    for name, (size, expected) in inventory.items():
        data = (pack / name).read_bytes()
        if len(data) != size or sha256(data) != expected:
            raise ValueError(f"{pack / name} does not match the pack index")
    return inventory


def fetch(
    destination: Path,
    digest: Optional[str] = None,
    store_url: str = STORE_URL,
    open_url=urlopen,
) -> None:
    """Stages the pack at `destination` from the store: the index first,
    checked against the digest, then each file it lists, checked against
    the index. Files land in a scratch directory beside `destination` and
    move into place together, so a failed fetch leaves nothing behind. A
    pack already staged there is verified and kept."""
    digest = digest or PACK_DIGEST
    if destination.exists():
        verify(destination, digest)
        logger.info("head camera pack already staged at %s", destination)
        return
    base = f"{store_url}/{PACK_PREFIX}/{digest}/"

    def get(name: str) -> bytes:
        request = Request(base + name, headers={"User-Agent": USER_AGENT})
        with open_url(request, timeout=120) as response:
            return response.read()

    index_data = get(INDEX_FILE)
    inventory = read_index(index_data, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (scratch / INDEX_FILE).write_bytes(index_data)
        for name, (size, expected) in inventory.items():
            data = get(name)
            if len(data) != size or sha256(data) != expected:
                raise ValueError(f"{base}{name} does not match the pack index")
            (scratch / name).write_bytes(data)
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    logger.info("head camera pack staged at %s from %s", destination, base)


def body_position(rig: dict) -> tuple[float, float, float]:
    """The pack's mount origin in the pedestal link's frame, the `pos` of
    the wrapper's openarm_head_camera body."""
    body = rig["body"]
    if body["name"] != BODY_NAME:
        raise ValueError(f"the pack's body is {body['name']!r}, not {BODY_NAME!r}")
    x, y, z = (float(v) for v in str(body["pos"]).split())
    return (x, y, z)


def check_chest_camera(rig: dict, entry) -> None:
    """The chest camera the sensor renders for a model that draws the pack
    must be the pack's left eye: the same parent link, its position to a
    tenth of a millimetre, its orientation the same rotation. The model's
    entry is kept by hand from the rig the derivation prints, and this
    refuses a pack bump or an entry edit that leaves the stream looking out
    of the drawn lens."""
    chest = next((c for c in entry.cameras if c.name == CHEST_CAMERA), None)
    if chest is None:
        raise RuntimeError(f"the {entry.model} entry declares no {CHEST_CAMERA!r} camera")
    left = rig["cameras"]["left"]
    position = np.array(left["position_m"], dtype=np.float64)
    quat = np.array(left["quat_wxyz"], dtype=np.float64)
    if chest.parent_link != PEDESTAL_LINK:
        raise RuntimeError(
            f"the chest camera mounts on {chest.parent_link!r}, the head camera on {PEDESTAL_LINK!r}"
        )
    if np.abs(position - np.array(chest.pos)).max() > POSITION_TOLERANCE_M:
        raise RuntimeError(
            f"the chest camera at {chest.pos} is not at the pack's left lens front"
            f" {tuple(position.round(6))}"
        )
    alignment = abs(float(quat @ np.array(chest.quat_wxyz))) / (
        np.linalg.norm(quat) * np.linalg.norm(chest.quat_wxyz)
    )
    if 1.0 - alignment > ORIENTATION_TOLERANCE:
        raise RuntimeError(
            f"the chest camera orientation {chest.quat_wxyz} is not the pack's left eye's"
            f" {tuple(quat)}"
        )


@dataclass(frozen=True)
class Pack:
    """A staged pack, checked: the directory its meshes are read from and
    its mount origin in the pedestal link's frame."""

    directory: Path
    body_position: tuple[float, float, float]


def load(directory: Path, entries) -> Pack:
    """The pack staged at `directory`, verified against the pinned digest,
    with the chest camera of every entry in `entries` checked against its
    rig. The node reads it once at setup, so a pack that is not staged or
    not the pinned one, or a chest camera that has drifted off the pack's
    left eye, stops the node before any robot stands."""
    verify(directory)
    rig = json.loads((directory / RIG_FILE).read_text())
    for entry in entries:
        check_chest_camera(rig, entry)
    return Pack(directory=directory, body_position=body_position(rig))


def load_for(models, directory: Path) -> Optional[Pack]:
    """The pack the models of this engine that draw the head camera share, or
    None when none of them draws it."""
    drawing = [
        models.of(name).entry for name in models.names() if models.of(name).head_camera
    ]
    return load(directory, drawing) if drawing else None


def attach(spec, pack: Pack):
    """Adds the head camera to the scene's spec: the body at the pack's mount
    origin on the pedestal, its three meshes in the scene's matte black, and
    the convex hull as a collider in the collision group, Waldo's wrapper
    block geom for geom. Returns the body. Raises on a scene without the
    pedestal or its material, or one that has the head camera already."""
    parent = _pedestal(spec)
    if spec.body(BODY_NAME) is not None:
        raise RuntimeError(f"the scene already has a {BODY_NAME!r} body")
    if spec.material(MATERIAL) is None:
        raise RuntimeError(f"the scene defines no {MATERIAL!r} material for the head camera")

    body = parent.add_body()
    body.name = BODY_NAME
    body.pos = list(pack.body_position)
    for name in VISUAL_MESHES:
        geom = _mesh_geom(spec, body, name, pack.directory / f"{name}.obj", "model/obj")
        geom.material = MATERIAL
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = _VISUAL_GROUP
    hull = _mesh_geom(spec, body, "collision", pack.directory / COLLISION_FILE, "model/stl")
    hull.condim = _COLLISION_CONDIM
    hull.conaffinity = 1
    hull.priority = _COLLISION_PRIORITY
    hull.group = _COLLISION_GROUP
    hull.solref = list(_COLLISION_SOLREF)
    hull.friction = list(_COLLISION_FRICTION)
    return body


def _pedestal(spec):
    """The body the pedestal link compiles into: its own, or the world body
    the compiler folds a fixed base link into, which then carries the link's
    geoms by name."""
    body = spec.body(PEDESTAL_LINK)
    if body is not None:
        return body
    if any(geom.name.startswith(f"{PEDESTAL_LINK}_") for geom in spec.worldbody.geoms):
        return spec.worldbody
    raise RuntimeError(f"head camera link {PEDESTAL_LINK!r} is not in the scene")


def _mesh_geom(spec, body, name: str, path: Path, content_type: str):
    import mujoco  # pylint: disable=C0415

    mesh = spec.add_mesh()
    mesh.name = f"{MESH_PREFIX}{name}"
    # An absolute path: MuJoCo reads it as is, past the scene's meshdir.
    mesh.file = str(path.resolve())
    mesh.content_type = content_type
    geom = body.add_geom()
    geom.name = f"{MESH_PREFIX}{name}"
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh.name
    return geom


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="stage the pinned pack at a directory")
    fetch_command.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch(args.directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
