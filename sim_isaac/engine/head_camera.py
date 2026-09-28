#!/usr/bin/env python3
"""The OpenArm v2 head camera the Isaac robot carries: the ZED Mini on its
bracket under the head cover, as Enactic's OpenArm_2.0_w_Head_Camera CAD
assembly seats it on the pedestal.

Upstream's openarm_description, which the robot USD bundle is prepared
from, has no head camera. Waldo's tools/openarm_head_camera
(private-nodes-hub) reads it off the CAD into an immutable pack under the
public waldo-assets store: three OBJ meshes and one convex collision hull,
all in the frame of the CAD mount's origin, the stereo rig read off the
lens barrels, and provenance. A pack is keyed by the SHA-256 of its
inventory, so the digest pinned here names one exact set of files for good.

`fetch` stages that pack into the node image (apptainer.def runs it at
build), `load` reads the staged pack once at node setup, and `attach` puts
its meshes under the pedestal link of a robot on the stage, so the Isaac
robot carries the same head camera as Waldo's openarm_v2 wrapper. The chest
camera of a model that draws the pack, whose pose the camera sensor renders
from, is checked to sit at the pack's left lens front, the eye the real ZED's
rectified stream comes from.

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
USER_AGENT = "sim_isaac/head_camera.py"
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
# link, which is also the chest camera's parent.
BODY_NAME = "openarm_head_camera"
PEDESTAL_LINK = "openarm_body_link0"
CHEST_CAMERA = "chest"
# Waldo draws the whole mount in upstream openarm_mujoco's matte_black, the
# colour the bundle's arm links already carry.
COLOR = (0.247, 0.247, 0.247)
ROUGHNESS = 0.5
# The model's entry rounds the rig to a tenth of a millimetre and seven
# decimals.
POSITION_TOLERANCE_M = 1e-4
ORIENTATION_TOLERANCE = 1e-5


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


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pack OBJ as (positions, normals, triangles): `v` and `vn` records,
    one normal per position, and `f a//a b//b c//c` triangles naming both
    by the same index, which is how the derivation writes them."""
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[list[int]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or line.startswith("#") or fields[0] == "o":
            continue
        if fields[0] == "v":
            positions.append([float(v) for v in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            corners = []
            for corner in fields[1:]:
                position, _, normal = corner.partition("//")
                if normal != position:
                    raise ValueError(
                        f"{path}: face corner {corner!r} does not name one index for"
                        " its position and its normal"
                    )
                corners.append(int(position) - 1)
            if len(corners) != 3:
                raise ValueError(f"{path}: a {len(corners)}-sided face; the pack's meshes are triangles")
            triangles.append(corners)
        else:
            raise ValueError(f"{path}: unexpected OBJ record {fields[0]!r}")
    if not triangles or not positions or len(positions) != len(normals):
        raise ValueError(
            f"{path}: {len(positions)} positions, {len(normals)} normals and"
            f" {len(triangles)} triangles"
        )
    faces = np.array(triangles, dtype=np.int64)
    if faces.min() < 0 or faces.max() >= len(positions):
        raise ValueError(f"{path}: a face names a position the file lacks")
    return np.array(positions, dtype=np.float64), np.array(normals, dtype=np.float64), faces


_STL_FACET = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A binary STL as welded (points, triangles)."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: not a binary STL")
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    if count == 0 or len(data) != 84 + _STL_FACET.itemsize * count:
        raise ValueError(f"{path}: not a binary STL of {count} triangles")
    facets = np.frombuffer(data, dtype=_STL_FACET, count=count, offset=84)
    corners = facets["vertices"].reshape(-1, 3).astype(np.float64)
    points, inverse = np.unique(corners, axis=0, return_inverse=True)
    return points, np.asarray(inverse).ravel().reshape(-1, 3).astype(np.int64)


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


@dataclass(frozen=True, eq=False)
class Pack:
    """A staged pack, read and checked: its directory, its mount origin in
    the pedestal link's frame, each visual mesh by name as (positions,
    normals, triangles), and the collision hull as (points, triangles)."""

    directory: Path
    body_position: tuple[float, float, float]
    visuals: tuple[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]], ...]
    collision: tuple[np.ndarray, np.ndarray]


def load(directory: Path, entries) -> Pack:
    """The pack staged at `directory`, verified against the pinned digest,
    with the chest camera of every entry in `entries` checked against its
    rig and its meshes read. The node reads it once at setup, so a pack that
    is not staged or not the pinned one, a mesh it cannot read, or a chest
    camera that has drifted off the pack's left eye stops the node before
    any robot stands."""
    verify(directory)
    rig = json.loads((directory / RIG_FILE).read_text())
    for entry in entries:
        check_chest_camera(rig, entry)
    return Pack(
        directory=directory,
        body_position=body_position(rig),
        visuals=tuple((name, read_obj(directory / f"{name}.obj")) for name in VISUAL_MESHES),
        collision=read_stl(directory / COLLISION_FILE),
    )


def load_for(models, directory: Path) -> Optional[Pack]:
    """The pack the models of this engine that draw the head camera share, or
    None when none of them draws it."""
    drawing = [
        models.of(name).entry for name in models.names() if models.of(name).head_camera
    ]
    return load(directory, drawing) if drawing else None


def attach(stage, root: str, pack: Pack) -> str:
    """Puts the head camera under the pedestal link of the robot at `root`:
    the body at the pack's mount origin, its three meshes in matte black
    shaded by the pack's own normals, and the convex hull as the link's collider,
    a guide so it is never drawn. Returns the body's path. Raises on a link
    the robot lacks or a head camera already under it."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    root_prim = stage.GetPrimAtPath(root)
    link = None
    if root_prim and root_prim.IsValid():
        link = next((p for p in Usd.PrimRange(root_prim) if p.GetName() == PEDESTAL_LINK), None)
    if link is None:
        raise RuntimeError(f"head camera link {PEDESTAL_LINK!r} is not under {root}")
    body_path = link.GetPath().AppendChild(BODY_NAME)
    if stage.GetPrimAtPath(body_path):
        raise RuntimeError(f"{body_path} is already on the stage")

    body = UsdGeom.Xform.Define(stage, body_path)
    body.AddTranslateOp().Set(Gf.Vec3d(*pack.body_position))
    material = _material(stage, body_path.AppendChild("Looks").AppendChild("matte_black"))
    for name, (positions, normals, triangles) in pack.visuals:
        mesh = _mesh(stage, body_path.AppendChild(name), positions, triangles)
        # One normal per face corner, the layout the bundle's own visuals
        # use and the one Kit's renderer takes without complaint; the pack's
        # normal per position is spread over the corners that share it.
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals[triangles.ravel()].astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*COLOR)]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    points, triangles = pack.collision
    hull = _mesh(stage, body_path.AppendChild("collision"), points, triangles)
    hull.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull
    )
    return str(body_path)


def _mesh(stage, path, points: np.ndarray, triangles: np.ndarray):
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    return mesh


def _material(stage, path):
    """The preview surface the visuals library binds its links with."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(ROUGHNESS)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


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
