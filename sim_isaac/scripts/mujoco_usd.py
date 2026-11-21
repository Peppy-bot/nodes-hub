"""Writing an Isaac Sim stage from a compiled MuJoCo model of Waldo's
catalogue: the poses, the materials, the meshes and the geoms of its bodies,
the release the model came from, and the reproducible archive a bundle is
published as. The builders of the robots' stages
(build_so101.py, build_botshare_base.py) share it.

CPU only: it needs the `mujoco`, `usd-core` and `numpy` packages.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_source(release: Path, robot_id: str) -> dict:
    """The release of Waldo's catalogue a staged model of `robot_id` came
    from, and the tree of it."""
    index = json.loads((release / "index.json").read_text())
    robots = json.loads((release / "robots.json").read_text())
    (tree,) = [robot["tree"] for robot in robots if robot["id"] == robot_id]
    return {"release": index["prefix"], "tree": tree}


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
