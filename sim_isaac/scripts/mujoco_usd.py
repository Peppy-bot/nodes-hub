"""Writing an Isaac Sim stage from a compiled MuJoCo model of Waldo's
catalogue: the poses, the materials, the meshes and the geoms of its bodies,
the release the model came from, what a bundle carries beside its stage (the
upstream notices, MODIFICATIONS.md and the manifest), the arguments a
builder takes, the release among them held to the one
sim_base_images/waldo_catalogue.py pins, and the reproducible archive a
bundle is published as. The builders of the robots' stages (build_so101.py,
build_botshare_base.py) share it.

CPU only: it needs the `mujoco`, `usd-core` and `numpy` packages.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import sys
import tarfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim_base_images"))
import waldo_catalogue  # noqa: E402  pylint: disable=C0413

MANIFEST_FILENAME = "bundle_manifest.json"
MODIFICATIONS_FILENAME = "MODIFICATIONS.md"
# Who modifies the upstream files a bundle's stage is made from.
MODIFIER = "Peppy Inc"
# The width MODIFICATIONS.md wraps its paragraphs at.
MODIFICATIONS_WIDTH = 79


@dataclass(frozen=True)
class Bundle:
    """What a robot's bundle carries beside its stage, as its builder gives
    it: the robot's id in Waldo's catalogue, the stage's file name, the
    builder script, the license of the upstream files the stage is made
    from, the prefix in the robot's tree those files stand at ("" for the
    tree's base, else the prefix of the overlay that lays them), how the
    stage changes them, and the upstream notices, each bundle file name by
    its path in the tree."""

    robot_id: str
    stage: str
    builder: Path
    license: str
    upstream: str
    changes: str
    notices: dict[str, str]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned_release(text: str, robot_id: str) -> Path:
    """The directory `--release` names, refused unless it holds the release
    waldo_catalogue.py pins, staged with `robot_id`'s model. A refusal gives
    the command that stages it."""
    release = Path(text)
    staging = (
        "stage the release waldo_catalogue.py pins in a new directory, a path that does not exist yet, with "
        f"`python3 sim_base_images/waldo_catalogue.py fetch <new directory> --robot {robot_id}` "
        "and give that directory as --release"
    )
    try:
        waldo_catalogue.verify(release)
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"{release} does not hold the release waldo_catalogue.py pins ({error}): {staging}"
        ) from None
    if not (release / robot_id).is_dir():
        raise argparse.ArgumentTypeError(
            f"the release staged at {release} holds no model of '{robot_id}': {staging}"
        )
    return release


def catalogue_entry(release: Path, robot_id: str) -> dict:
    """The entry of `robot_id` in the release's robots.json."""
    robots = json.loads((release / "robots.json").read_text())
    entries = [robot for robot in robots if robot["id"] == robot_id]
    if not entries:
        raise ValueError(
            f"the release staged at {release} lists no robot '{robot_id}' (it lists "
            f"{[robot['id'] for robot in robots]}): set the builder's ROBOT_ID to the id it lists for the robot"
        )
    if len(entries) > 1:
        raise ValueError(
            f"the release staged at {release} lists the robot '{robot_id}' {len(entries)} times: pin a release "
            "that lists it once, RELEASE_DIGEST in sim_base_images/waldo_catalogue.py"
        )
    return entries[0]


def release_source(release: Path, robot_id: str) -> dict:
    """The release of Waldo's catalogue a staged model of `robot_id` came
    from, and the tree of it."""
    index = json.loads((release / "index.json").read_text())
    return {"release": index["prefix"], "tree": catalogue_entry(release, robot_id)["tree"]}


def upstream_of(release: Path, robot_id: str, prefix: str) -> str:
    """Where the upstream files at `prefix` in the tree of `robot_id` come
    from: the base of its source for "", else the base of its overlay of
    that prefix."""
    source = catalogue_entry(release, robot_id)["source"]
    if not prefix:
        return source["base"]
    # A source with no overlay carries no "overlays".
    overlays = source.get("overlays", ())
    laid = [overlay["prefix"] for overlay in overlays]
    bases = [overlay["base"] for overlay in overlays if overlay["prefix"] == prefix]
    if not bases:
        raise ValueError(
            f"the robot '{robot_id}' of the release staged at {release} lays no overlay at '{prefix}' "
            f"(its overlays lay at {laid}): set `upstream` in the builder's BUNDLE to '' for the tree's base "
            "or to a prefix an overlay lays at"
        )
    if len(bases) > 1:
        raise ValueError(
            f"the robot '{robot_id}' of the release staged at {release} lays {len(bases)} overlays at "
            f"'{prefix}' (its overlays lay at {laid}): set `upstream` in the builder's BUNDLE to '' for the "
            "tree's base or to a prefix one overlay lays at"
        )
    return bases[0]


def bundle_date(text: str) -> datetime.date:
    """The date of a build, from its `--date` argument."""
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is no date: give the date of the build as YYYY-MM-DD") from None


def bundle_parser(doc: str, robot_id: str) -> argparse.ArgumentParser:
    """The arguments of the builder of `robot_id`'s bundle: the staged release
    it reads (pinned_release), the directory it writes the bundle into, the
    date of the build, and the archive to pack the bundle into."""
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--release",
        type=lambda text: pinned_release(text, robot_id),
        required=True,
        help=f"the release of Waldo's catalogue waldo_catalogue.py pins, staged with the model of {robot_id}",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--date",
        type=bundle_date,
        required=True,
        help="the date of the build, YYYY-MM-DD, which MODIFICATIONS.md and the manifest record",
    )
    parser.add_argument("--archive", type=Path, help="pack the output into this .tar.gz")
    return parser


def modifications(bundle: Bundle, release: Path, date: datetime.date) -> str:
    """The text of MODIFICATIONS.md: which upstream files the stage is a
    modified form of, under which license, how it changes them, and who
    made it, when, with what and from which release."""
    source = release_source(release, bundle.robot_id)
    paragraphs = (
        f"{bundle.stage} is a modified form of the files at "
        f"{upstream_of(release, bundle.robot_id, bundle.upstream)}, under {bundle.license}. "
        f"{bundle.changes}",
        f"{MODIFIER} made it on {date.isoformat()} with sim_isaac/scripts/{bundle.builder.name} and "
        f"sim_isaac/scripts/{Path(__file__).name} of Peppy-bot/nodes-hub, from the release "
        f"{source['release']} of Waldo's catalogue, its tree {source['tree']}.",
    )
    wrapped = (
        textwrap.fill(paragraph, MODIFICATIONS_WIDTH, break_long_words=False, break_on_hyphens=False)
        for paragraph in paragraphs
    )
    return "# Modifications\n\n" + "\n\n".join(wrapped) + "\n"


def write_bundle(
    bundle: Bundle, release: Path, output: Path, description_digest: str, date: datetime.date
) -> None:
    """Writes beside the stage in `output` the upstream notices `bundle`
    carries, as the release stages them, MODIFICATIONS.md, and the manifest
    of what built the bundle: the release and the tree the model came from
    and its license, the digest robot_description_py gives the model, the
    digest of each script that built it, who modified the upstream files
    and when, the versions of the tools, and the digest of every file."""
    tree = release / bundle.robot_id
    for name, path in bundle.notices.items():
        (output / name).write_bytes((tree / path).read_bytes())
    (output / MODIFICATIONS_FILENAME).write_text(modifications(bundle, release, date))
    manifest = {
        "version": 2,
        "stage": bundle.stage,
        "source": {**release_source(release, bundle.robot_id), "license": bundle.license},
        "description": {"model": bundle.robot_id, "digest": description_digest},
        "builder": {script.name: sha256(script.read_bytes()) for script in (bundle.builder, Path(__file__))},
        "modified": {"by": MODIFIER, "date": date.isoformat()},
        "tools": {
            "mujoco": mujoco.__version__,
            "usd": ".".join(str(v) for v in Usd.GetVersion()),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "files": {
            name: sha256((output / name).read_bytes())
            for name in (bundle.stage, *bundle.notices, MODIFICATIONS_FILENAME)
        },
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def finish_bundle(stage_path: Path, output: Path, archive: Path | None) -> None:
    """Reports the stage's size and packs `output` into `archive`, where one
    is given, reporting the archive's SHA-256."""
    print(f"{stage_path}: {stage_path.stat().st_size} bytes")
    if archive is not None:
        print(f"{archive}: sha256 {pack(output, archive)}")


def gf_quat(wxyz, kind=Gf.Quatf):
    w, x, y, z = (float(v) for v in wxyz)
    return kind(w, x, y, z)


def set_pose(prim, pos, quat_wxyz, scale=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in pos)))
    xform.AddOrientOp().Set(gf_quat(quat_wxyz))
    xform.AddScaleOp().Set(Gf.Vec3f(*(scale if scale is not None else (1.0, 1.0, 1.0))))


def material_for(stage, looks: str, model, geom_id: int, cache: dict) -> UsdShade.Material:
    """One preview-surface material per colour the model draws, under
    `looks`, named after the model's material, or `geom_colour` for a
    colour the geom carries itself."""
    mat_id = int(model.geom_matid[geom_id])
    rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
    name = model.material(mat_id).name if mat_id >= 0 else "geom_colour"
    key = (name, tuple(round(float(v), 4) for v in rgba))
    if key in cache:
        return cache[key]
    material = UsdShade.Material.Define(stage, f"{looks}/{name}")
    shader = UsdShade.Shader.Define(stage, f"{looks}/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(float(v) for v in rgba[:3]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    cache[key] = material
    return material


def define_mesh(stage, path: str, model, mesh_id: int) -> UsdGeom.Mesh:
    """The compiled mesh, in the frame MuJoCo placed its geom in."""
    vert_adr, vert_num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    face_adr, face_num = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    points = model.mesh_vert[vert_adr : vert_adr + vert_num]
    faces = model.mesh_face[face_adr : face_adr + face_num]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * face_num))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    low, high = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, low)), Gf.Vec3f(*map(float, high))])
    return mesh


def define_geom(
    stage, parent: str, name: str, model, geom_id: int, visual_group: int, looks: str, materials: dict
):
    """One geom of a link, drawn when its group is `visual_group` and
    collided with otherwise. Returns its prim."""
    visual = int(model.geom_group[geom_id]) == visual_group
    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    path = f"{parent}/{'visuals' if visual else 'collisions'}/{name}"
    pos, quat = model.geom_pos[geom_id], model.geom_quat[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        prim = define_mesh(stage, path, model, int(model.geom_dataid[geom_id])).GetPrim()
        set_pose(prim, pos, quat)
        approximation = UsdPhysics.Tokens.convexHull
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(2.0)
        cube.CreateExtentAttr([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)])
        prim = cube.GetPrim()
        set_pose(prim, pos, quat, scale=tuple(float(v) for v in size))
        approximation = None
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(float(size[0]))
        prim = sphere.GetPrim()
        set_pose(prim, pos, quat)
        approximation = None
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateRadiusAttr(float(size[0]))
        capsule.CreateHeightAttr(2.0 * float(size[1]))
        capsule.CreateAxisAttr(UsdGeom.Tokens.z)
        prim = capsule.GetPrim()
        set_pose(prim, pos, quat)
        approximation = None
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateRadiusAttr(float(size[0]))
        cylinder.CreateHeightAttr(2.0 * float(size[1]))
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        prim = cylinder.GetPrim()
        set_pose(prim, pos, quat)
        approximation = None
    else:
        raise ValueError(f"geom {geom_id} has a type this builder does not write: {kind}")
    if visual:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material_for(stage, looks, model, geom_id, materials))
        return prim
    UsdPhysics.CollisionAPI.Apply(prim)
    if approximation is not None:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    return prim


def pack(directory: Path, archive: Path) -> str:
    """Packs `directory` into `archive` reproducibly and returns the archive's
    SHA-256, the key it is published under."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(directory.iterdir()):
            info = tar.gettarinfo(str(path), arcname=path.name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as source:
                tar.addfile(info, source)
    with archive.open("wb") as target, gzip.GzipFile(
        fileobj=target, mode="wb", mtime=0, filename=""
    ) as compressed:
        compressed.write(buffer.getvalue())
    return sha256(archive.read_bytes())
