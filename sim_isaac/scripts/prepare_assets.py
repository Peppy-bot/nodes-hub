#!/usr/bin/env python3
"""Prepare a verified, self-contained OpenArm Isaac asset archive for publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile

import collada
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

import build_visuals


SCRIPTS_DIR = Path(__file__).resolve().parent
STAGE_FILENAME = "openarm_bimanual_v2.usd"
# Isaac supplies this MDL module locally; it is not a robot asset download.
RUNTIME_ASSETS = {"OmniPBR.mdl"}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def nonvisual_specs(stage: Usd.Stage) -> str:
    """Compare all authored data except the 21 visual attachments."""
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(stage.GetRootLayer())
    for link in build_visuals.LINK_MESHES:
        body = layer.GetPrimAtPath(f"/openarm/{link}")
        del body.nameChildren["visuals"]
    return layer.ExportToString()


def validate_assets(directory: Path, original: Usd.Stage) -> dict:
    stage = Usd.Stage.Open(str(directory / STAGE_FILENAME))
    if nonvisual_specs(stage) != nonvisual_specs(original):
        raise ValueError("Conversion changed nonvisual robot data")
    if build_visuals.visual_attachments(stage) != build_visuals.visual_attachments(
        original
    ):
        raise ValueError("Conversion changed visual attachment frames")
    before = UsdGeom.XformCache()
    after = UsdGeom.XformCache()
    for link in build_visuals.LINK_MESHES:
        path = f"/openarm/{link}/visuals/{link}_visual"
        if before.GetLocalToWorldTransform(original.GetPrimAtPath(path)) != (
            after.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        ):
            raise ValueError(f"Conversion changed world attachment frames: {link}")

    for filename in ("openarm_bimanual.usd", STAGE_FILENAME):
        entry = directory / filename
        composed = Usd.Stage.Open(str(entry))
        if composed.GetCompositionErrors():
            raise ValueError(f"USD composition errors in {filename}")
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(str(entry))
        )
        if set(unresolved) - RUNTIME_ASSETS:
            raise ValueError(f"Unresolved robot assets: {unresolved}")
        for path in [layer.realPath for layer in layers] + assets:
            if not Path(path).is_file():
                raise ValueError(f"Missing robot dependency: {path}")
            if not Path(path).resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Robot dependency escapes the bundle: {path}")

    mesh_count = 0
    colors = set()
    for link in build_visuals.LINK_MESHES:
        visual = stage.GetPrimAtPath(f"/openarm/{link}/visuals/{link}_visual")
        if not visual.IsInstance():
            raise ValueError(f"Visual is not an instance: {link}")
        link_colors = set()
        for prim in Usd.PrimRange(visual, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            imageable = UsdGeom.Imageable(prim)
            if (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or imageable.ComputePurpose() not in ("default", "render")
                or imageable.ComputeVisibility() != "inherited"
            ):
                raise ValueError(f"Invalid render mesh: {prim.GetPath()}")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            shader, _, _ = (
                material.ComputeSurfaceSource() if material else (None, None, None)
            )
            if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
                raise ValueError(f"Missing preview material: {prim.GetPath()}")
            color = shader.GetInput("diffuseColor").Get()
            if color is None:
                raise ValueError(f"Missing source color: {prim.GetPath()}")
            link_colors.add(tuple(color))
            mesh_count += 1
        if len(link_colors) < 2:
            raise ValueError(f"Missing multi-material visual geometry: {link}")
        colors.update(link_colors)
    return {
        "visual_attachments": len(build_visuals.LINK_MESHES),
        "render_meshes": mesh_count,
        "source_colors": len(colors),
        "nonvisual_specs_unchanged": True,
        "attachment_frames_unchanged": True,
        "runtime_assets": sorted(RUNTIME_ASSETS),
    }


def write_archive(directory: Path, output: Path) -> None:
    """Fix order, ownership and timestamps so identical inputs give identical bytes."""
    with tempfile.TemporaryDirectory(
        prefix=".openarm-bundle-", dir=output.parent
    ) as temporary:
        archive_path = Path(temporary) / "bundle.tar.gz"
        with (
            archive_path.open("xb") as destination,
            gzip.GzipFile(
                filename="", fileobj=destination, mode="wb", mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                info = tarfile.TarInfo(path.relative_to(directory).as_posix())
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open("rb") as source:
                    archive.addfile(info, source)
        # Linking publishes the complete file atomically and refuses an existing name.
        output.hardlink_to(archive_path)


def prepare_assets(
    source_dir: Path, output: Path, mesh_source_dir: Path | None = None
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((SCRIPTS_DIR / "visual_sources.json").read_text())
    for filename, checksum in manifest["robot"]["files"].items():
        if sha256(source_dir / filename) != checksum:
            raise ValueError(f"Robot source checksum mismatch: {filename}")
    if (
        sha256(SCRIPTS_DIR / "openarm_description.LICENSE.txt")
        != manifest["license"]["sha256"]
    ):
        raise ValueError("Upstream license checksum mismatch")

    original = Usd.Stage.Open(str(source_dir / STAGE_FILENAME))
    with tempfile.TemporaryDirectory(prefix="openarm-isaac-assets-") as temporary:
        directory = Path(temporary) / "bundle"
        directory.mkdir()
        for filename in manifest["robot"]["files"]:
            target = directory / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / filename, target)
        meshes = Path(temporary) / "meshes"
        meshes.mkdir()
        sources = build_visuals.load_sources(
            manifest, mesh_source_dir or meshes, download=mesh_source_dir is None
        )
        build_visuals.build_visuals(directory / STAGE_FILENAME, sources, manifest)
        validation = validate_assets(directory, original)
        provenance = {
            "schema_version": 1,
            "source_manifest": "openarm_visual_sources.json",
            "toolchain": {
                "python": platform.python_version(),
                "usd": ".".join(map(str, Usd.GetVersion())),
                "pycollada": collada.__version__,
                "numpy": np.__version__,
                "scripts": {
                    name: sha256(SCRIPTS_DIR / name)
                    for name in ("build_visuals.py", "prepare_assets.py")
                },
            },
            "validation": validation,
            "files": {
                path.relative_to(directory).as_posix(): {
                    "sha256": sha256(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            },
        }
        (directory / "bundle_manifest.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        write_archive(directory, output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Robot assets from the pinned base image",
    )
    parser.add_argument(
        "--mesh-source-dir", type=Path, help="Verified DAEs for offline preparation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New .tar.gz archive; never overwritten",
    )
    args = parser.parse_args()
    provenance = prepare_assets(
        args.source_dir.resolve(), args.output, args.mesh_source_dir
    )
    print(json.dumps(provenance["validation"], indent=2))
    print(f"{sha256(args.output)}  {args.output}")


if __name__ == "__main__":
    main()
