"""Exercise material-bearing COLLADA conversion and instanced USD composition."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

import collada
import numpy as np
import pytest

pytest.importorskip("pxr", reason="USD Python wheels are unavailable on this platform")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import build_visuals as build  # noqa: E402
import prepare_assets as prepare  # noqa: E402


@pytest.fixture
def dae(tmp_path):
    document = collada.Collada()
    document.assetInfo.unitmeter = 1.0
    document.assetInfo.unitname = "meter"
    document.assetInfo.upaxis = "Y_UP"
    materials = []
    for name, color in (
        ("dark", (0.05, 0.05, 0.05, 1)),
        ("silver", (0.65, 0.65, 0.65, 1)),
    ):
        effect = collada.material.Effect(
            name + "_effect",
            [],
            "lambert",
            diffuse=color,
            transparency=1.0,
            transparent=None,
        )
        material = collada.material.Material(name, name, effect)
        document.effects.append(effect)
        document.materials.append(material)
        materials.append(collada.scene.MaterialNode(name, material, inputs=[]))
    vertices = collada.source.FloatSource(
        "vertices",
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        ("X", "Y", "Z"),
    )
    normals = collada.source.FloatSource(
        "normals",
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        ("X", "Y", "Z"),
    )
    geometry = collada.geometry.Geometry(document, "part", "part", [vertices, normals])
    inputs = collada.source.InputList()
    inputs.addInput(0, "VERTEX", "#vertices")
    inputs.addInput(1, "NORMAL", "#normals")
    geometry.primitives.append(
        geometry.createTriangleSet(np.array([0, 2, 1, 0, 2, 1]), inputs, "dark")
    )
    geometry.primitives.append(
        geometry.createTriangleSet(np.array([0, 1, 2, 2, 3, 0]), inputs, "silver")
    )
    document.geometries.append(geometry)
    child = collada.scene.Node(
        "child",
        children=[collada.scene.GeometryNode(geometry, materials)],
        transforms=[collada.scene.RotateTransform(1, 0, 0, 90)],
    )
    parent = collada.scene.Node(
        "parent",
        children=[child],
        transforms=[
            collada.scene.TranslateTransform(2, 3, 4),
            collada.scene.ScaleTransform(1, 2, 3),
        ],
    )
    scene = collada.scene.Scene("scene", [parent])
    document.scenes.append(scene)
    document.scene = scene
    path = tmp_path / "part.dae"
    document.write(str(path))
    return path


def _meshes(prim):
    return [
        p
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if p.IsA(UsdGeom.Mesh)
    ]


def _color(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert material
    shader, _, _ = material.ComputeSurfaceSource()
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    return tuple(shader.GetInput("diffuseColor").Get())


@pytest.fixture
def robot(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    UsdGeom.Xform.Define(stage, "/openarm")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/openarm"))
    for index, link in enumerate(build.LINK_MESHES):
        body = UsdGeom.Xform.Define(stage, f"/openarm/{link}")
        body.AddTranslateOp().Set((index, 0, 0.3))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(2.5)
        collider = UsdGeom.Cube.Define(stage, f"/openarm/{link}/collisions")
        collider.CreateSizeAttr(0.15)
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/openarm/joints/{link}")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateLowerLimitAttr(-45)
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular").CreateStiffnessAttr(100)
        prototype = UsdGeom.Xform.Define(stage, f"/prototypes/{link}")
        visual = UsdGeom.Xform.Define(
            stage, prototype.GetPath().AppendChild(link + "_visual")
        )
        visual.AddTranslateOp().Set((0.0205 if "ee_base_link" in link else 0, 0, 0))
        scale = (
            (0.001,) * 3
            if link == "openarm_body_link0"
            else (1, -1 if "left" in link else 1, 1)
        )
        visual.AddScaleOp().Set(scale)
        UsdGeom.Mesh.Define(stage, visual.GetPath().AppendChild("collision_mesh"))
        root = UsdGeom.Xform.Define(stage, f"/openarm/{link}/visuals")
        root.GetPrim().GetReferences().AddInternalReference(prototype.GetPath())
        root.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().Save()
    return stage


def test_converter_preserves_geometry_materials_split_normals_and_scene_transforms(dae):
    stage = Usd.Stage.CreateInMemory()
    build.convert_dae(stage, dae, Sdf.Path("/Visuals/part"))
    meshes = _meshes(stage.GetPrimAtPath("/Visuals/part"))
    assert len(meshes) == 2
    assert _color(meshes[0]) == pytest.approx((0.05, 0.05, 0.05))
    assert _color(meshes[1]) == pytest.approx((0.65, 0.65, 0.65))
    bound = next(collada.Collada(str(dae)).scene.objects("geometry"))
    for prim, triangles in zip(meshes, bound.primitives()):
        mesh = UsdGeom.Mesh(prim)
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3]
        assert list(mesh.GetFaceVertexIndicesAttr().Get()) == list(
            triangles.original.vertex_index.flatten()
        )
        assert mesh.GetSubdivisionSchemeAttr().Get() == "none"
        assert mesh.GetNormalsInterpolation() == "faceVarying"
        np.testing.assert_allclose(
            mesh.GetNormalsAttr().Get(),
            triangles.original.normal[triangles.original.normal_index].reshape(-1, 3),
        )
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        points = [
            transform.Transform(Gf.Vec3d(*point))
            for point in mesh.GetPointsAttr().Get()
        ]
        np.testing.assert_allclose(points, triangles.vertex, atol=1e-6)


@pytest.mark.parametrize("parent_transform", [False, True])
def test_replacement_preserves_physics_frames_and_composes_materials_on_instances(
    robot, dae, tmp_path, parent_transform
):
    if parent_transform:
        parent = UsdGeom.Xformable(robot.GetPrimAtPath("/prototypes/openarm_body_link0"))
        parent.AddTranslateOp().Set((5, 2, 3))
        parent.AddRotateXOp().Set(45)
        parent.AddScaleOp().Set((2, 3, 4))
    library_path = tmp_path / "visuals.usdc"
    library = Usd.Stage.CreateNew(str(library_path))
    for name in set(build.LINK_MESHES.values()):
        build.convert_dae(library, dae, Sdf.Path(f"/Visuals/{name}"))
    library.GetRootLayer().Save()
    physics = prepare.nonvisual_specs(robot)
    attachments = build.visual_attachments(robot)
    cache = UsdGeom.XformCache()
    world_frames = {
        link: cache.GetLocalToWorldTransform(
            robot.GetPrimAtPath(f"/openarm/{link}/visuals/{link}_visual")
        )
        for link in build.LINK_MESHES
    }
    build.replace_visuals(robot, library_path, attachments)
    robot.GetRootLayer().Save()
    assert prepare.nonvisual_specs(robot) == physics
    assert len(attachments) == 21
    assert build.visual_attachments(robot) == attachments
    for link in build.LINK_MESHES:
        path = f"/openarm/{link}/visuals/{link}_visual"
        prim = robot.GetPrimAtPath(path)
        assert prim.IsInstance()
        assert UsdGeom.XformCache().GetLocalToWorldTransform(prim) == world_frames[link]
        meshes = _meshes(prim)
        assert len(meshes) == 2
        assert not robot.GetPrimAtPath(path + "/collision_mesh")
        assert _color(meshes[0]) == pytest.approx((0.05,) * 3)
        assert _color(meshes[1]) == pytest.approx((0.65,) * 3)
        for mesh in meshes:
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            assert material.GetPath().HasPrefix(prim.GetPath())
            assert UsdGeom.Imageable(mesh).ComputePurpose() == "default"
            assert UsdGeom.Imageable(mesh).ComputeVisibility() == "inherited"
            assert not mesh.HasAPI(UsdPhysics.CollisionAPI)


def test_build_packages_render_library_provenance_and_license(robot, dae, tmp_path):
    attachments = build.visual_attachments(robot)
    physics = prepare.nonvisual_specs(robot)
    manifest = json.loads((_SCRIPTS / "visual_sources.json").read_text())
    manifest["revision"] = "a" * 40
    sources = {name: dae for name in set(build.LINK_MESHES.values())}
    build.build_visuals(Path(robot.GetRootLayer().realPath), sources, manifest)
    library = Usd.Stage.Open(str(tmp_path / build.VISUALS_FILENAME))
    assert library.GetRootLayer().customLayerData["revision"] == manifest["revision"]
    assert (tmp_path / "openarm_description.LICENSE.txt").read_bytes() == (
        _SCRIPTS / "openarm_description.LICENSE.txt"
    ).read_bytes()
    assert (
        json.loads((tmp_path / "openarm_visual_sources.json").read_text()) == manifest
    )
    assert build.visual_attachments(robot) == attachments
    assert prepare.nonvisual_specs(robot) == physics
    assert len(_meshes(robot.GetPrimAtPath("/openarm"))) == 42


def test_missing_attachment_fails_before_modifying_stage(robot, tmp_path, dae):
    robot.RemovePrim("/openarm/openarm_right_ee_link2/visuals")
    before = robot.GetRootLayer().ExportToString()
    with pytest.raises(
        ValueError, match="Missing v2 visual attachment: openarm_right_ee_link2"
    ):
        build.visual_attachments(robot)
    assert robot.GetRootLayer().ExportToString() == before
    assert not (tmp_path / build.VISUALS_FILENAME).exists()


def test_offline_sources_are_checksum_verified(dae):
    manifest = {
        "meshes": {"part": {"sha256": hashlib.sha256(dae.read_bytes()).hexdigest()}}
    }
    assert build.load_sources(manifest, dae.parent, download=False) == {"part": dae}
    manifest["meshes"]["part"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch: part.dae"):
        build.load_sources(manifest, dae.parent, download=False)


def test_downloads_use_pinned_urls_and_verify_every_file(tmp_path, monkeypatch):
    payloads = {"first": b"first visual", "second": b"second visual"}
    manifest = {
        "repository": "example/robot",
        "revision": "a" * 40,
        "meshes": {
            name: {
                "path": f"visual/{name}.dae",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in payloads.items()
        },
    }
    urls = []

    def download(url, timeout):
        assert timeout == 120
        urls.append(url)
        return io.BytesIO(payloads[Path(url).stem])

    monkeypatch.setattr(build, "urlopen", download)
    sources = build.load_sources(manifest, tmp_path, download=True)
    assert {name: path.read_bytes() for name, path in sources.items()} == payloads
    assert set(urls) == {
        f"https://raw.githubusercontent.com/example/robot/{'a' * 40}/visual/{name}.dae"
        for name in payloads
    }
    manifest["meshes"]["second"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch: second.dae"):
        build.load_sources(manifest, tmp_path, download=True)


@pytest.mark.parametrize(
    "kind, message",
    [
        ("scene", "visual scene"),
        ("unit", "visual scene"),
        ("material", "material-bound triangles"),
        ("single_color", "multi-material visual geometry"),
        ("transparent", "opaque Lambert color"),
    ],
)
def test_unsupported_visual_source_fails_instead_of_rendering_white(dae, kind, message):
    document = collada.Collada(str(dae))
    if kind == "scene":
        document.scene = None
    elif kind == "unit":
        document.assetInfo.unitmeter = 0.001
    elif kind == "material":
        document.scene.nodes[0].children[0].children[0].materials.clear()
    elif kind == "single_color":
        document.effects[1].diffuse = document.effects[0].diffuse
    elif kind == "transparent":
        document.effects[0].transparency = 0.5
    document.write(str(dae))
    with pytest.raises(ValueError, match=message):
        build.convert_dae(Usd.Stage.CreateInMemory(), dae, Sdf.Path("/Visuals/part"))


def test_maintenance_sources_pin_meshes_robot_stages_and_license():
    manifest = json.loads((_SCRIPTS / "visual_sources.json").read_text())
    assert len(manifest["revision"]) == 40
    assert set(manifest["meshes"]) == set(build.LINK_MESHES.values())
    assert all(
        len(source["sha256"]) == 64 and "/visual/" in source["path"]
        for source in manifest["meshes"].values()
    )
    assert "@sha256:" in manifest["robot"]["image"]
    assert len(manifest["robot"]["layer_sha256"]) == 64
    assert len(manifest["robot"]["files"]) == 5
    assert all(len(checksum) == 64 for checksum in manifest["robot"]["files"].values())
    assert prepare.sha256(_SCRIPTS / "openarm_description.LICENSE.txt") == manifest["license"]["sha256"]
    requirements = (_SCRIPTS / "requirements-visuals.txt").read_text()
    assert "pycollada==0.9.3" in requirements


@pytest.fixture
def bundle_sources(robot, dae, tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for filename in (
        "openarm_description.LICENSE.txt", "build_visuals.py", "prepare_assets.py"
    ):
        shutil.copyfile(_SCRIPTS / filename, scripts / filename)
    manifest = json.loads((_SCRIPTS / "visual_sources.json").read_text())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    robot.GetRootLayer().Export(str(source_dir / prepare.STAGE_FILENAME))
    v1 = Usd.Stage.CreateNew(str(source_dir / "openarm_bimanual.usd"))
    UsdGeom.Xform.Define(v1, "/openarm")
    for filename in manifest["robot"]["files"]:
        if not filename.startswith("configuration/"):
            continue
        path = source_dir / filename
        path.parent.mkdir(exist_ok=True)
        layer = Sdf.Layer.CreateNew(str(path))
        layer.Save()
        v1.GetRootLayer().subLayerPaths.append(filename)
    v1.GetRootLayer().Save()
    manifest["robot"]["files"] = {
        name: prepare.sha256(source_dir / name) for name in manifest["robot"]["files"]
    }
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    for name, source in manifest["meshes"].items():
        shutil.copyfile(dae, meshes / f"{name}.dae")
        source["sha256"] = prepare.sha256(dae)
    (scripts / "visual_sources.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(build, "SCRIPTS_DIR", scripts)
    return source_dir, meshes, manifest


def test_preparation_archives_complete_offline_bundle_with_verifiable_provenance(
    bundle_sources, tmp_path
):
    source_dir, meshes, sources = bundle_sources
    output = tmp_path / "bundle.tar.gz"
    provenance = prepare.prepare_assets(source_dir, output, meshes)
    assert provenance["validation"] == {
        "visual_attachments": 21,
        "render_meshes": 42,
        "source_colors": 2,
        "nonvisual_specs_unchanged": True,
        "attachment_frames_unchanged": True,
        "runtime_assets": ["OmniPBR.mdl"],
    }
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert set(names) == set(sources["robot"]["files"]) | {
            build.VISUALS_FILENAME, "openarm_visual_sources.json",
            "openarm_description.LICENSE.txt", "bundle_manifest.json",
        }
        for name, info in provenance["files"].items():
            payload = archive.extractfile(name).read()
            assert hashlib.sha256(payload).hexdigest() == info["sha256"]
            assert len(payload) == info["size"]
            member = archive.getmember(name)
            assert member.mtime == member.uid == member.gid == 0
            assert member.mode == 0o644
        assert json.load(archive.extractfile("bundle_manifest.json")) == provenance
        extracted = tmp_path / "extracted"
        archive.extractall(extracted, filter="data")
    original = Usd.Stage.Open(str(source_dir / prepare.STAGE_FILENAME))
    assert prepare.validate_assets(extracted, original) == provenance["validation"]
    # Source stages remain untouched; only the copied v2 stage is repaired.
    for name, checksum in sources["robot"]["files"].items():
        assert prepare.sha256(source_dir / name) == checksum
        if name != prepare.STAGE_FILENAME:
            assert prepare.sha256(extracted / name) == checksum
    second = tmp_path / "second.tar.gz"
    prepare.prepare_assets(source_dir, second, meshes)
    assert prepare.sha256(second) == prepare.sha256(output)


@pytest.mark.parametrize("kind", ["stage", "license", "mesh"])
def test_preparation_refuses_corrupt_sources_without_producing_a_bundle(
    bundle_sources, tmp_path, kind
):
    source_dir, meshes, _ = bundle_sources
    if kind == "stage":
        path = source_dir / prepare.STAGE_FILENAME
    elif kind == "license":
        path = prepare.SCRIPTS_DIR / "openarm_description.LICENSE.txt"
    else:
        path = meshes / "base_link.dae"
    path.write_bytes(b"incorrect source")
    output = tmp_path / "bundle.tar.gz"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare.prepare_assets(source_dir, output, meshes)
    assert not output.exists()


def test_preparation_never_overwrites_an_archive(bundle_sources, tmp_path):
    source_dir, meshes, _ = bundle_sources
    output = tmp_path / "bundle.tar.gz"
    output.write_bytes(b"published artifact")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare.prepare_assets(source_dir, output, meshes)
    assert output.read_bytes() == b"published artifact"


@pytest.mark.parametrize("kind", ["physics", "frame", "parent_frame", "metadata", "missing", "external"])
def test_validation_rejects_changed_physics_frames_or_incomplete_bundle(
    bundle_sources, tmp_path, kind
):
    source_dir, meshes, _ = bundle_sources
    output = tmp_path / "bundle.tar.gz"
    prepare.prepare_assets(source_dir, output, meshes)
    extracted = tmp_path / "extracted"
    with tarfile.open(output) as archive:
        archive.extractall(extracted, filter="data")
    stage = Usd.Stage.Open(str(extracted / prepare.STAGE_FILENAME))
    if kind == "physics":
        body = stage.GetPrimAtPath("/openarm/openarm_body_link0")
        UsdPhysics.MassAPI(body).GetMassAttr().Set(999)
        stage.GetRootLayer().Save()
        message = "nonvisual robot data"
    elif kind == "frame":
        visual = stage.GetPrimAtPath("/openarm/openarm_body_link0/visuals/openarm_body_link0_visual")
        UsdGeom.Xformable(visual).GetOrderedXformOps()[0].Set(Gf.Matrix4d(1))
        stage.GetRootLayer().Save()
        message = "attachment frames"
    elif kind == "parent_frame":
        parent = stage.GetPrimAtPath("/openarm/openarm_body_link0/visuals")
        UsdGeom.Xformable(parent).AddTranslateOp().Set((5, 0, 0))
        stage.GetRootLayer().Save()
        message = "world attachment frames"
    elif kind == "metadata":
        stage.SetTimeCodesPerSecond(999)
        stage.GetRootLayer().Save()
        message = "nonvisual robot data"
    else:
        v1 = Sdf.Layer.FindOrOpen(str(extracted / "openarm_bimanual.usd"))
        if kind == "external":
            outside = Sdf.Layer.CreateNew(str(tmp_path / "outside.usda"))
            outside.Save()
            v1.subLayerPaths.append("../outside.usda")
            message = "escapes the bundle"
        else:
            v1.subLayerPaths.append("missing.usda")
            message = "composition errors"
        v1.Save()
    original = Usd.Stage.Open(str(source_dir / prepare.STAGE_FILENAME))
    with pytest.raises(ValueError, match=message):
        prepare.validate_assets(extracted, original)


def test_failed_archive_write_cleans_up_and_can_be_retried(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "stage.usd"
    payload.write_bytes(b"fixture")
    output = tmp_path / "bundle.tar.gz"
    original_open = Path.open

    def fail_source_read(path, *args, **kwargs):
        if path == payload:
            raise OSError("injected source read failure")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_source_read)
        with pytest.raises(OSError, match="injected source read failure"):
            prepare.write_archive(source, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]
    prepare.write_archive(source, output)
    with tarfile.open(output) as archive:
        assert archive.extractfile("stage.usd").read() == b"fixture"


def test_archive_publication_never_replaces_a_concurrent_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "stage.usd").write_bytes(b"fixture")
    output = tmp_path / "bundle.tar.gz"
    original_link = Path.hardlink_to

    def concurrent_output(path, target):
        path.write_bytes(b"another publisher")
        return original_link(path, target)

    monkeypatch.setattr(Path, "hardlink_to", concurrent_output)
    with pytest.raises(FileExistsError):
        prepare.write_archive(source, output)
    assert output.read_bytes() == b"another publisher"
    assert set(tmp_path.iterdir()) == {source, output}
