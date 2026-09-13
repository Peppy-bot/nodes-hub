#!/usr/bin/env python3
"""Build the OpenArm v2 render meshes from pinned, material-bearing COLLADA files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import collada
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


SCRIPTS_DIR = Path(__file__).resolve().parent
VISUALS_FILENAME = "openarm_v2_visuals.usdc"
LINK_MESHES = {"openarm_body_link0": "body_link0"}
for side in ("left", "right"):
    for name in (
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "ee_base_link",
    ):
        LINK_MESHES[f"openarm_{side}_{name}"] = name
    LINK_MESHES[f"openarm_{side}_ee_link1"] = "finger_inner"
    LINK_MESHES[f"openarm_{side}_ee_link2"] = "finger_outer"


def load_sources(manifest: dict, directory: Path, *, download: bool) -> dict[str, Path]:
    """Verify every source, including files supplied for an offline build."""

    def load(name: str) -> tuple[str, Path]:
        source = manifest["meshes"][name]
        path = directory / f"{name}.dae"
        if download:
            url = (
                f"https://raw.githubusercontent.com/{manifest['repository']}/"
                f"{manifest['revision']}/{source['path']}"
            )
            with urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Visual source checksum mismatch: {path.name}")
        return name, path

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(load, manifest["meshes"]))


def _material(stage: Usd.Stage, path: Sdf.Path, color: tuple) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color[:3])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def convert_dae(stage: Usd.Stage, path: Path, root_path: Sdf.Path) -> None:
    """Keep scene transforms, face material assignments and split vertex normals."""
    document = collada.Collada(str(path))
    if document.scene is None or document.images or document.assetInfo.unitmeter != 1.0:
        raise ValueError(
            f"Expected a texture-free, unit-meter visual scene: {path.name}"
        )

    UsdGeom.Xform.Define(stage, root_path)
    materials = {}
    for geometry_index, geometry in enumerate(document.scene.objects("geometry")):
        geometry_path = root_path.AppendChild(f"geometry_{geometry_index}")
        transform = UsdGeom.Xform.Define(stage, geometry_path)
        # COLLADA uses column vectors; USD uses row vectors. These are link-local
        # meshes: their scene transforms, not up_axis metadata, define the frame.
        transform.AddTransformOp().Set(
            Gf.Matrix4d(geometry.matrix.astype(float).T.tolist())
        )
        for primitive_index, bound in enumerate(geometry.primitives()):
            if (
                not isinstance(bound, collada.triangleset.BoundTriangleSet)
                or bound.material is None
            ):
                raise ValueError(f"Expected material-bound triangles: {path.name}")
            effect = bound.material.effect
            color = effect.diffuse
            if (
                effect.shadingtype != "lambert"
                or not isinstance(color, tuple)
                or len(color) != 4
                or color[3] != 1.0
                or effect.transparency != 1.0
                or effect.transparent is not None
            ):
                raise ValueError(
                    f"Expected an opaque Lambert color: {path.name}/{effect.id}"
                )
            if color not in materials:
                materials[color] = _material(
                    stage,
                    root_path.AppendPath(f"Looks/material_{len(materials)}"),
                    color,
                )

            # Author the untransformed arrays beneath the scene Xform. USD then
            # transforms normals correctly, including nonuniform and mirrored scale.
            triangles = bound.original
            if not len(triangles) or triangles.normal is None:
                raise ValueError(
                    f"Expected nonempty triangles with normals: {path.name}"
                )
            mesh = UsdGeom.Mesh.Define(
                stage, geometry_path.AppendChild(f"mesh_{primitive_index}")
            )
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(triangles.vertex))
            mesh.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
            )
            mesh.CreateFaceVertexIndicesAttr(
                Vt.IntArray.FromNumpy(triangles.vertex_index.flatten())
            )
            normals = triangles.normal[triangles.normal_index].reshape(-1, 3)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(
                Vt.Vec3fArray.FromNumpy(
                    np.array(
                        [triangles.vertex.min(axis=0), triangles.vertex.max(axis=0)]
                    )
                )
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[color])
    if len(materials) < 2:
        raise ValueError(
            f"Expected detailed, multi-material visual geometry: {path.name}"
        )


def visual_attachments(stage: Usd.Stage) -> dict[str, Gf.Matrix4d]:
    """Resolve all link-local visual frames before changing any references."""
    attachments = {}
    for link in LINK_MESHES:
        root = stage.GetPrimAtPath(f"/openarm/{link}/visuals")
        visual = stage.GetPrimAtPath(f"/openarm/{link}/visuals/{link}_visual")
        if not root or not visual or not visual.IsA(UsdGeom.Xform):
            raise ValueError(f"Missing v2 visual attachment: {link}")
        if [
            child.GetName()
            for child in root.GetFilteredChildren(Usd.TraverseInstanceProxies())
        ] != [f"{link}_visual"]:
            raise ValueError(f"Expected one visual attachment for {link}")
        attachments[link] = UsdGeom.Xformable(visual).GetLocalTransformation()
    return attachments


def replace_visuals(
    stage: Usd.Stage, library: Path, attachments: dict[str, Gf.Matrix4d]
) -> None:
    """Replace visual references only; retain links, colliders, joints and drives."""
    for link, matrix in attachments.items():
        root = stage.GetPrimAtPath(f"/openarm/{link}/visuals")
        parent = UsdGeom.Xformable(root)
        parent_matrix = parent.GetLocalTransformation()
        reset_stack = parent.GetResetXformStack()
        root.SetInstanceable(False)
        root.GetReferences().SetReferences([])
        # A referenced prototype can contribute the parent's transform too.
        if parent.GetLocalTransformation() != parent_matrix:
            parent.MakeMatrixXform().Set(parent_matrix)
        parent.SetResetXformStack(reset_stack)
        for child in root.GetChildren():
            stage.RemovePrim(child.GetPath())
        visual = UsdGeom.Xform.Define(
            stage, root.GetPath().AppendChild(f"{link}_visual")
        )
        visual.AddTransformOp().Set(matrix)
        visual.GetPrim().GetReferences().AddReference(
            f"./{library.name}", Sdf.Path(f"/Visuals/{LINK_MESHES[link]}")
        )
        visual.GetPrim().SetInstanceable(True)


def build_visuals(stage_path: Path, sources: dict[str, Path], manifest: dict) -> None:
    stage = Usd.Stage.Open(str(stage_path))
    attachments = visual_attachments(stage)
    library_path = stage_path.parent / VISUALS_FILENAME
    library = Usd.Stage.CreateNew(str(library_path))
    UsdGeom.SetStageUpAxis(library, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(library, 1.0)
    library.SetDefaultPrim(UsdGeom.Scope.Define(library, "/Visuals").GetPrim())
    library.GetRootLayer().customLayerData = {
        "source": f"https://github.com/{manifest['repository']}",
        "revision": manifest["revision"],
        "license": "Apache-2.0",
        "conversion": "COLLADA visual geometry and materials converted to USD",
    }
    for name in sorted(set(LINK_MESHES.values())):
        convert_dae(library, sources[name], Sdf.Path(f"/Visuals/{name}"))
    library.GetRootLayer().Save()
    replace_visuals(stage, library_path, attachments)
    stage.GetRootLayer().Save()
    shutil.copyfile(
        SCRIPTS_DIR / "openarm_description.LICENSE.txt",
        stage_path.parent / "openarm_description.LICENSE.txt",
    )
    (stage_path.parent / "openarm_visual_sources.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Built {len(sources)} visual meshes for {len(attachments)} OpenArm v2 links")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory of checksum-verified DAEs for an offline build",
    )
    args = parser.parse_args()
    manifest = json.loads((SCRIPTS_DIR / "visual_sources.json").read_text())
    with tempfile.TemporaryDirectory(prefix="openarm-visuals-") as directory:
        sources = load_sources(
            manifest,
            args.source_dir or Path(directory),
            download=args.source_dir is None,
        )
        build_visuals(args.stage.resolve(), sources, manifest)


if __name__ == "__main__":
    main()
