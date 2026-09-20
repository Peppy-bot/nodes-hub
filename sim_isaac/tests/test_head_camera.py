"""The head camera pack: its digest rule and fetch, the pack's meshes on the
stage under the pedestal link, and the chest camera check against its rig.
A tiny pack in the store's layout stands in for the published one; the
published index is checked verbatim against the pinned digest."""

import copy
import dataclasses
import hashlib
import io
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sim_robot_core.models import shipped_entry

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))

import head_camera  # noqa: E402
from isaac_models import IsaacModels  # noqa: E402

# The model whose entry asks for the head camera, and whose chest camera
# renders from its left eye.
_OPENARM_V2 = shipped_entry("openarm_v2")
_LINK = "/openarm/openarm_body_link0"

# The rig of the pinned pack, as tools/openarm_head_camera prints it: the
# mount origin and the left lens front the checked-in chest camera sits at.
_RIG = {
    "body": {"name": "openarm_head_camera", "pos": "0.0315 0 0.743"},
    "cameras": {
        "left": {
            "position_m": [0.079202, 0.031497, 0.794103],
            "quat_wxyz": [0.6861027, 0.1710646, -0.1710646, -0.6861027],
        },
        "right": {
            "position_m": [0.079202, -0.031503, 0.794103],
            "quat_wxyz": [0.6861027, 0.1710646, -0.1710646, -0.6861027],
        },
    },
    "baseline_m": 0.063,
}

# The published pack's index.json, verbatim: the inventory the pinned digest
# is the hash of.
_PUBLISHED_FILES = [
    ("cameras.json", 1104, "87a949d097ef5e93be0bb2782d2d63127aa02205a0351dfc7c50011d7a10fa44", "application/json"),
    ("collision.stl", 64584, "ee6d9af579c6f03390514b1b1690174555a0169ca3d493ace452d3518a17aebd", "model/stl"),
    ("cover.obj", 2994386, "3de501c22d85bea3a1594499d8d996d27516b167a24d2907d8ee573f9d0caf94", "text/plain"),
    ("mount.obj", 397119, "f350f9713272e51038e30352645ef55c5ee57f65f687ae09ccefde98cb31a417", "text/plain"),
    ("provenance.json", 2947, "4f3eaca75b5a29fc8defe326825f417888a8a300dcb070543a6901c26c6fa38c", "application/json"),
    ("zed_mini.obj", 111177, "168092766817658e93ae77fa0194ef1882256774f118b4fa2c4d8f228a2221d8", "text/plain"),
]
_PUBLISHED_INDEX = {
    "files": [
        {
            "key": f"{head_camera.PACK_PREFIX}/{head_camera.PACK_DIGEST}/{name}",
            "size": size,
            "sha256": sha256,
            "content_type": content_type,
        }
        for name, size, sha256, content_type in _PUBLISHED_FILES
    ]
}

_TETRAHEDRON = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
_TETRAHEDRON_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_QUAD = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
_QUAD_NORMALS = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)]
_QUAD_TRIANGLES = [[0, 1, 2], [1, 3, 2]]


def _obj(name: str, offset: float) -> bytes:
    """A quad as two triangles with a normal per position, written like the
    derivation: each corner names its position and its normal by one index."""
    lines = [f"# OpenArm 2.0 head camera: {name}, metres, pedestal-frame axes", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in _QUAD + offset]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in _QUAD_NORMALS]
    lines += ["f " + " ".join(f"{i + 1}//{i + 1}" for i in face) for face in _QUAD_TRIANGLES]
    return ("\n".join(lines) + "\n").encode()


def _stl(points, faces) -> bytes:
    data = bytearray(80) + struct.pack("<I", len(faces))
    for face in faces:
        data += struct.pack("<3f", 0.0, 0.0, 0.0)
        for index in face:
            data += struct.pack("<3f", *points[index])
        data += struct.pack("<H", 0)
    return bytes(data)


def _pack_files(rig=_RIG) -> dict[str, bytes]:
    files = {f"{name}.obj": _obj(name, i * 0.01) for i, name in enumerate(head_camera.VISUAL_MESHES)}
    files["collision.stl"] = _stl(_TETRAHEDRON, _TETRAHEDRON_FACES)
    files["cameras.json"] = (json.dumps(rig, indent=2) + "\n").encode()
    files["provenance.json"] = b'{"derived_on": "2026-09-17"}\n'
    return files


def _digest(files: dict[str, bytes]) -> str:
    return head_camera.pack_digest({name: (len(data), _sha256(data)) for name, data in files.items()})


def _index(files: dict[str, bytes], digest: str) -> bytes:
    entries = [
        {
            "key": f"{head_camera.PACK_PREFIX}/{digest}/{name}",
            "size": len(data),
            "sha256": _sha256(data),
            "content_type": "application/octet-stream",
        }
        for name, data in sorted(files.items())
    ]
    return (json.dumps({"files": entries}, indent=2) + "\n").encode()


def _stage(directory: Path, files: dict[str, bytes]) -> str:
    digest = _digest(files)
    directory.mkdir(parents=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    (directory / "index.json").write_bytes(_index(files, digest))
    return digest


@pytest.fixture
def pack(tmp_path, monkeypatch):
    files = _pack_files()
    directory = tmp_path / "head_camera"
    digest = _stage(directory, files)
    monkeypatch.setattr(head_camera, "PACK_DIGEST", digest)
    served = dict(files, **{"index.json": (directory / "index.json").read_bytes()})
    return SimpleNamespace(directory=directory, files=files, digest=digest, served=served)


@pytest.fixture
def loaded(pack):
    """The staged pack as setup reads it."""
    return head_camera.load(pack.directory, [_OPENARM_V2])


def _store(pack, served=None, requests=None):
    """An opener serving the pack's files from the store's URL layout, to a
    request that names its agent (the store's Cloudflare front refuses
    urllib's default one)."""
    served = pack.served if served is None else served
    base = f"{head_camera.STORE_URL}/{head_camera.PACK_PREFIX}/{pack.digest}/"

    def open_url(request, timeout):
        assert request.get_header("User-agent") == head_camera.USER_AGENT
        url = request.full_url
        assert url.startswith(base) and timeout > 0
        name = url[len(base):]
        if requests is not None:
            requests.append(name)
        return io.BytesIO(served[name])

    return open_url


class TestPackIndex:
    def test_pinned_digest_is_the_hash_of_the_published_inventory(self):
        inventory = head_camera.read_index(json.dumps(_PUBLISHED_INDEX).encode())
        assert sorted(inventory) == [
            "cameras.json", "collision.stl", "cover.obj", "mount.obj", "provenance.json", "zed_mini.obj",
        ]
        assert inventory["cover.obj"] == (
            2994386, "3de501c22d85bea3a1594499d8d996d27516b167a24d2907d8ee573f9d0caf94",
        )

    def test_an_inventory_that_is_not_the_pinned_packs_is_refused(self, pack):
        swapped = dict(pack.files, **{"cover.obj": b"another derivation"})
        with pytest.raises(ValueError, match="hashes to"):
            head_camera.read_index(_index(swapped, pack.digest))

    def test_a_key_outside_the_pack_is_refused(self, pack):
        index = json.loads(pack.served["index.json"])
        name = index["files"][0]["key"].rsplit("/", 1)[1]
        index["files"][0]["key"] = f"{head_camera.PACK_PREFIX}/{pack.digest}/nested/{name}"
        with pytest.raises(ValueError, match="outside"):
            head_camera.read_index(json.dumps(index).encode())
        index = json.loads(pack.served["index.json"])
        index["files"][0]["key"] = f"sources/other/{name}"
        with pytest.raises(ValueError, match="outside"):
            head_camera.read_index(json.dumps(index).encode())

    def test_a_pack_missing_a_file_the_robot_draws_is_refused(self, pack):
        files = {name: data for name, data in pack.files.items() if name != "collision.stl"}
        with pytest.raises(ValueError, match=r"missing \['collision.stl'\]"):
            head_camera.read_index(_index(files, _digest(files)), _digest(files))


class TestFetch:
    def test_stages_every_file_verified_and_keeps_a_staged_pack(self, tmp_path, pack):
        requests = []
        destination = tmp_path / "node" / "assets" / "head_camera"
        head_camera.fetch(destination, open_url=_store(pack, requests=requests))

        assert requests[0] == "index.json"
        assert sorted(requests[1:]) == sorted(pack.files)
        assert {path.name: path.read_bytes() for path in destination.iterdir()} == pack.served
        assert [path.name for path in destination.parent.iterdir()] == ["head_camera"]

        head_camera.fetch(destination, open_url=_store(pack, requests=requests))
        assert len(requests) == 1 + len(pack.files)

    def test_a_file_off_its_index_leaves_nothing_behind(self, tmp_path, pack):
        destination = tmp_path / "node" / "assets" / "head_camera"
        served = dict(pack.served, **{"mount.obj": b"corrupt"})
        with pytest.raises(ValueError, match="mount.obj does not match"):
            head_camera.fetch(destination, open_url=_store(pack, served=served))
        assert not destination.exists()
        assert list(destination.parent.iterdir()) == []

    def test_an_index_off_the_digest_downloads_no_file(self, tmp_path, pack):
        destination = tmp_path / "node" / "head_camera"
        swapped = dict(pack.files, **{"cover.obj": b"another derivation"})
        served = dict(swapped, **{"index.json": _index(swapped, pack.digest)})
        requests = []
        with pytest.raises(ValueError, match="hashes to"):
            head_camera.fetch(destination, open_url=_store(pack, served=served, requests=requests))
        assert requests == ["index.json"]
        assert not destination.exists()

    def test_a_staged_pack_off_its_index_is_refused_not_replaced(self, pack):
        (pack.directory / "zed_mini.obj").write_bytes(b"edited by hand")
        with pytest.raises(ValueError, match="zed_mini.obj does not match"):
            head_camera.fetch(pack.directory, open_url=_store(pack, requests=[]))
        assert (pack.directory / "zed_mini.obj").read_bytes() == b"edited by hand"

    def test_the_command_line_stages_the_pinned_pack(self, tmp_path, pack, monkeypatch):
        monkeypatch.setattr(head_camera, "urlopen", _store(pack))
        destination = tmp_path / "head_camera"
        assert head_camera.main(["fetch", str(destination)]) == 0
        assert head_camera.verify(destination) == {
            name: (len(data), _sha256(data)) for name, data in pack.files.items()
        }


class TestMeshes:
    def test_obj_reads_positions_normals_and_triangles(self, pack):
        positions, normals, triangles = head_camera.read_obj(pack.directory / "mount.obj")
        assert positions == pytest.approx(_QUAD + 0.01)
        assert [tuple(n) for n in normals] == _QUAD_NORMALS
        assert triangles.tolist() == _QUAD_TRIANGLES

    @pytest.mark.parametrize(
        ("face", "message"),
        [
            ("f 1//2 2//3 3//1", "one index"),
            ("f 1//1 2//2 3//3 1//1", "4-sided"),
            ("f 1/1/1 2/2/2 3/3/3", "one index"),
            ("f 1//1 2//2 4//4", "lacks"),
        ],
    )
    def test_obj_off_the_derivations_layout_is_refused(self, tmp_path, face, message):
        path = tmp_path / "zed_mini.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\n" + face + "\n")
        with pytest.raises(ValueError, match=message):
            head_camera.read_obj(path)

    def test_stl_welds_shared_corners(self, pack):
        points, triangles = head_camera.read_stl(pack.directory / "collision.stl")
        assert sorted(map(tuple, points.tolist())) == sorted(_TETRAHEDRON)
        assert triangles.shape == (4, 3)
        for face, expected in zip(triangles, _TETRAHEDRON_FACES):
            assert [tuple(points[i]) for i in face] == [_TETRAHEDRON[i] for i in expected]

    def test_stl_of_another_length_is_refused(self, tmp_path):
        path = tmp_path / "collision.stl"
        path.write_bytes(_stl(_TETRAHEDRON, _TETRAHEDRON_FACES)[:-1])
        with pytest.raises(ValueError, match="binary STL"):
            head_camera.read_stl(path)


def _chest_camera_moved(**changes):
    """The OpenArm v2 entry with its chest camera changed."""
    cameras = tuple(
        dataclasses.replace(camera, **changes) if camera.name == head_camera.CHEST_CAMERA else camera
        for camera in _OPENARM_V2.cameras
    )
    return dataclasses.replace(_OPENARM_V2, cameras=cameras)


class TestChestCamera:
    def test_the_shipped_chest_camera_sits_at_the_packs_left_eye(self):
        head_camera.check_chest_camera(_RIG, _OPENARM_V2)

    def test_the_negated_quaternion_is_the_same_orientation(self):
        rig = copy.deepcopy(_RIG)
        rig["cameras"]["left"]["quat_wxyz"] = [-v for v in rig["cameras"]["left"]["quat_wxyz"]]
        head_camera.check_chest_camera(rig, _OPENARM_V2)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("position_m", [0.0794, 0.031497, 0.794103], "left lens front"),
            ("quat_wxyz", [0.6861027, -0.1710646, 0.1710646, -0.6861027], "orientation"),
        ],
    )
    def test_an_entry_off_the_packs_rig_is_refused(self, field, value, message):
        rig = copy.deepcopy(_RIG)
        rig["cameras"]["left"][field] = value
        with pytest.raises(RuntimeError, match=message):
            head_camera.check_chest_camera(rig, _OPENARM_V2)

    def test_a_chest_camera_mounted_off_the_pedestal_is_refused(self):
        entry = _chest_camera_moved(parent_link="openarm_left_ee_base_link")
        with pytest.raises(RuntimeError, match="the head camera on 'openarm_body_link0'"):
            head_camera.check_chest_camera(_RIG, entry)

    @pytest.mark.parametrize("model", ["openarm_v1", "so101"])
    def test_an_entry_without_a_chest_camera_is_refused(self, model):
        with pytest.raises(RuntimeError, match=f"the {model} entry declares no 'chest' camera"):
            head_camera.check_chest_camera(_RIG, shipped_entry(model))


class TestLoad:
    def test_reads_the_mount_origin_and_every_mesh(self, pack):
        loaded = head_camera.load(pack.directory, [_OPENARM_V2])
        assert loaded.directory == pack.directory
        assert loaded.body_position == (0.0315, 0.0, 0.743)
        assert [name for name, _ in loaded.visuals] == list(head_camera.VISUAL_MESHES)
        for i, (_, (positions, normals, triangles)) in enumerate(loaded.visuals):
            assert positions == pytest.approx(_QUAD + i * 0.01)
            assert [tuple(n) for n in normals] == _QUAD_NORMALS
            assert triangles.tolist() == _QUAD_TRIANGLES
        points, triangles = loaded.collision
        assert (len(points), len(triangles)) == (4, 4)

    def test_refuses_an_unstaged_pack(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch"):
            head_camera.load(tmp_path / "head_camera", [_OPENARM_V2])

    def test_refuses_a_pack_that_is_not_the_pinned_one(self, pack):
        (pack.directory / "cover.obj").write_bytes(b"v 0 0 0\n")
        with pytest.raises(ValueError, match="does not match the pack index"):
            head_camera.load(pack.directory, [_OPENARM_V2])

    def test_refuses_a_pack_the_chest_camera_is_off(self, tmp_path, monkeypatch):
        rig = copy.deepcopy(_RIG)
        rig["cameras"]["left"]["position_m"] = [0.0794, 0.031497, 0.794103]
        directory = tmp_path / "head_camera"
        monkeypatch.setattr(head_camera, "PACK_DIGEST", _stage(directory, _pack_files(rig)))
        with pytest.raises(RuntimeError, match="left lens front"):
            head_camera.load(directory, [_OPENARM_V2])

    def test_refuses_a_pack_framed_in_another_body(self, tmp_path, monkeypatch):
        rig = copy.deepcopy(_RIG)
        rig["body"]["name"] = "head"
        directory = tmp_path / "head_camera"
        monkeypatch.setattr(head_camera, "PACK_DIGEST", _stage(directory, _pack_files(rig)))
        with pytest.raises(ValueError, match="not 'openarm_head_camera'"):
            head_camera.load(directory, [_OPENARM_V2])


@pytest.fixture
def stage():
    pytest.importorskip("pxr", reason="USD Python wheels are unavailable on this platform")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for path in ("/openarm", "/openarm/openarm_mount", _LINK, f"{_LINK}/visuals", f"{_LINK}/collisions"):
        UsdGeom.Xform.Define(stage, path)
    return stage


class TestAttach:
    def test_puts_the_pack_under_the_pedestal_link(self, stage, loaded):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        body = head_camera.attach(stage, "/openarm", loaded)

        assert body == f"{_LINK}/openarm_head_camera"
        prim = stage.GetPrimAtPath(body)
        assert UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation() == Gf.Vec3d(0.0315, 0, 0.743)
        assert sorted(child.GetName() for child in prim.GetChildren()) == [
            "Looks", "collision", "cover", "mount", "zed_mini",
        ]
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"{body}/Looks/matte_black/Shader"))
        assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
        assert shader.GetInput("diffuseColor").Get() == Gf.Vec3f(0.247, 0.247, 0.247)
        assert shader.GetInput("roughness").Get() == 0.5
        for i, name in enumerate(head_camera.VISUAL_MESHES):
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{body}/{name}"))
            assert np.array(mesh.GetPointsAttr().Get()) == pytest.approx(_QUAD + i * 0.01)
            assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
            assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 1, 3, 2]
            # The pack's normal per position, spread over each face corner.
            assert [tuple(n) for n in mesh.GetNormalsAttr().Get()] == [
                _QUAD_NORMALS[i] for face in _QUAD_TRIANGLES for i in face
            ]
            assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
            assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
            assert UsdGeom.Imageable(mesh).ComputePurpose() == UsdGeom.Tokens.default_
            assert np.array(mesh.GetDisplayColorAttr().Get()) == pytest.approx(np.array([[0.247, 0.247, 0.247]]))
            material, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
            assert str(material.GetPath()) == f"{body}/Looks/matte_black"
            assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        hull = stage.GetPrimAtPath(f"{body}/collision")
        assert UsdGeom.Imageable(hull).ComputePurpose() == UsdGeom.Tokens.guide
        assert hull.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(hull).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
        assert len(UsdGeom.Mesh(hull).GetPointsAttr().Get()) == 4
        assert list(UsdGeom.Mesh(hull).GetFaceVertexCountsAttr().Get()) == [3] * 4
        assert not UsdShade.MaterialBindingAPI(hull).ComputeBoundMaterial()[0]

    def test_refuses_a_robot_without_the_pedestal_link(self, stage, loaded):
        stage.RemovePrim(_LINK)
        with pytest.raises(RuntimeError, match="openarm_body_link0"):
            head_camera.attach(stage, "/openarm", loaded)
        with pytest.raises(RuntimeError, match="openarm_body_link0"):
            head_camera.attach(stage, "/elsewhere", loaded)

    def test_refuses_a_second_attach(self, stage, loaded):
        head_camera.attach(stage, "/openarm", loaded)
        with pytest.raises(RuntimeError, match="already on the stage"):
            head_camera.attach(stage, "/openarm", loaded)


class TestLoadFor:
    def test_the_models_that_draw_the_head_camera_share_the_staged_pack(self, pack):
        loaded = head_camera.load_for(IsaacModels.read(), pack.directory)
        assert loaded.directory == pack.directory
        assert loaded.body_position == (0.0315, 0.0, 0.743)

    def test_an_engine_whose_models_draw_none_reads_no_pack(self, tmp_path):
        """Nothing is staged where it looks, and nothing needs to be."""
        (tmp_path / "so101.json5").write_text('{ stage: "so101/so101.usd", articulation_root: "." }')
        (tmp_path / "openarm_v1.json5").write_text(
            '{ stage: "openarm/openarm_bimanual.usd", articulation_root: "." }'
        )
        assert head_camera.load_for(IsaacModels.read(tmp_path), tmp_path / "head_camera") is None

    def test_an_engine_with_a_model_that_draws_it_needs_the_pack_staged(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch"):
            head_camera.load_for(IsaacModels.read(), tmp_path / "head_camera")

    def test_the_chest_camera_of_every_model_that_draws_it_is_checked(self, tmp_path, monkeypatch):
        rig = copy.deepcopy(_RIG)
        rig["cameras"]["left"]["position_m"] = [0.0794, 0.031497, 0.794103]
        directory = tmp_path / "head_camera"
        monkeypatch.setattr(head_camera, "PACK_DIGEST", _stage(directory, _pack_files(rig)))
        with pytest.raises(RuntimeError, match="left lens front"):
            head_camera.load_for(IsaacModels.read(), directory)

    def test_a_model_that_asks_for_the_head_camera_and_carries_no_chest_camera_is_refused(
        self, tmp_path, pack
    ):
        """The pack is the OpenArm v2's: an SO-101 has no chest camera to
        render from its left eye."""
        (tmp_path / "so101.json5").write_text(
            '{ stage: "so101/so101.usd", articulation_root: ".", head_camera: true }'
        )
        with pytest.raises(RuntimeError, match="the so101 entry declares no 'chest' camera"):
            head_camera.load_for(IsaacModels.read(tmp_path), pack.directory)
