"""The head camera pack: its digest rule and fetch, the pack's meshes on
the pedestal of a compiled scene the way Waldo's wrapper carries them, and
the chest camera check against its rig. A tiny pack in the store's layout
stands in for the published one; the published index is checked verbatim
against the pinned digest. The scenes are real (tiny) MJCF compiled by
MuJoCo."""

import copy
import dataclasses
import hashlib
import io
import json
import re
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from sim_robot_core.models import EngineModel, shipped_entry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the launcher imports)

_NODE_DIR = Path(__file__).resolve().parents[1]
_ENGINE_DIR = _NODE_DIR / "engine"
sys.path.insert(0, str(_ENGINE_DIR))

import head_camera  # noqa: E402
import mujoco_models  # noqa: E402
from _launcher import SimLauncher  # noqa: E402
from mujoco_models import MujocoModels  # noqa: E402

# The model that draws the pack, whose chest camera is checked against its rig.
_OPENARM_V2 = shipped_entry("openarm_v2")

# The rig of the pinned pack, as tools/openarm_head_camera prints it: the
# mount origin and the left lens front the OpenArm v2 entry's chest camera sits
# at.
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


def _obj(name: str, scale: float) -> bytes:
    """A centimetre-scale tetrahedron with a normal per position, written like
    the derivation: each corner names its position and its normal by one
    index. MuJoCo compiles only a closed mesh, so no flat test shape."""
    points = np.array(_TETRAHEDRON) * scale
    normals = points - points.mean(axis=0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    lines = [f"# OpenArm 2.0 head camera: {name}, metres, pedestal-frame axes", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    lines += ["f " + " ".join(f"{i + 1}//{i + 1}" for i in face) for face in _TETRAHEDRON_FACES]
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
    files = {f"{name}.obj": _obj(name, 0.01 * (i + 1)) for i, name in enumerate(head_camera.VISUAL_MESHES)}
    files["collision.stl"] = _stl([tuple(0.05 * v for v in p) for p in _TETRAHEDRON], _TETRAHEDRON_FACES)
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

    def test_an_entry_without_a_chest_camera_is_refused(self):
        with pytest.raises(RuntimeError, match="the openarm_v1 entry declares no 'chest' camera"):
            head_camera.check_chest_camera(_RIG, shipped_entry("openarm_v1"))


# The baked v2 scene's shape where the head camera lands: the pedestal link
# folded into the world body, its geoms keeping the link's name, and
# upstream's matte_black material the arms carry.
_SCENE = """<mujoco model="pedestal">
  <asset>
    <material name="matte_black" rgba="0.247 0.247 0.247 1"/>
  </asset>
  <worldbody>
    <geom name="openarm_body_link0_collision_column" type="box" size="0.05 0.05 0.3" pos="0 0 0.3"/>
    <body name="openarm_left_link1" pos="0 0.2 0.7">
      <joint name="openarm_left_joint1" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>"""


def _spec(xml=_SCENE):
    return mujoco.MjSpec.from_string(xml)


def _geom(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


class TestLoad:
    def test_reads_the_staged_pack_and_its_mount_origin(self, pack):
        loaded = head_camera.load(pack.directory, [_OPENARM_V2])
        assert loaded == head_camera.Pack(directory=pack.directory, body_position=(0.0315, 0.0, 0.743))

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


class TestAttach:
    def test_puts_waldos_head_camera_block_on_the_pedestal(self, loaded):
        spec = _spec()
        body = head_camera.attach(spec, loaded)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        assert body.name == "openarm_head_camera"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_head_camera")
        assert model.body_parentid[body_id] == 0
        assert model.body_weldid[body_id] == 0
        assert model.body_pos[body_id] == pytest.approx((0.0315, 0, 0.743))
        matte_black = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "matte_black")
        for i, name in enumerate(head_camera.VISUAL_MESHES):
            geom = _geom(model, f"head_camera_{name}")
            assert model.geom_bodyid[geom] == body_id
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            assert model.geom_matid[geom] == matte_black
            assert (model.geom_contype[geom], model.geom_conaffinity[geom], model.geom_group[geom]) == (0, 0, 2)
            mesh = model.geom_dataid[geom]
            assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh) == f"head_camera_{name}"
            # The pack's own normals, one per position, not recomputed ones.
            assert model.mesh_normalnum[mesh] == 4
            assert model.mesh_facenum[mesh] == 4
            # MuJoCo moves a mesh into its inertia frame and the geom with it;
            # in the world the vertices are the pack's, off the mount origin.
            first = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[first:first + model.mesh_vertnum[mesh]]
            world = data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T
            expected = np.array(_TETRAHEDRON) * 0.01 * (i + 1) + (0.0315, 0, 0.743)
            assert np.array(sorted(map(tuple, world.round(6)))) == pytest.approx(
                np.array(sorted(map(tuple, expected.round(6)))), abs=1e-6
            )
        hull = _geom(model, "head_camera_collision")
        assert model.geom_bodyid[hull] == body_id
        assert model.geom_type[hull] == mujoco.mjtGeom.mjGEOM_MESH
        assert (model.geom_contype[hull], model.geom_conaffinity[hull], model.geom_group[hull]) == (1, 1, 3)
        assert model.geom_condim[hull] == 3
        assert model.geom_priority[hull] == 1
        assert model.geom_solref[hull] == pytest.approx((0.005, 1))
        assert model.geom_friction[hull] == pytest.approx((1, 0.01, 0.01))
        assert model.geom_matid[hull] == -1

    def test_seats_on_the_pedestal_body_where_the_scene_keeps_one(self, loaded):
        spec = _spec(_SCENE.replace(
            '<geom name="openarm_body_link0_collision_column"',
            '<body name="openarm_body_link0" pos="0 0 0.01"><geom name="column"',
        ).replace('pos="0 0 0.3"/>', 'pos="0 0 0.3"/></body>'))
        head_camera.attach(spec, loaded)
        model = spec.compile()
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_head_camera")
        pedestal = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_body_link0")
        assert model.body_parentid[body] == pedestal

    def test_refuses_a_scene_without_the_pedestal(self, loaded):
        spec = _spec(_SCENE.replace("openarm_body_link0_collision_column", "column"))
        with pytest.raises(RuntimeError, match="openarm_body_link0"):
            head_camera.attach(spec, loaded)

    def test_refuses_a_scene_without_matte_black(self, loaded):
        spec = _spec(_SCENE.replace('name="matte_black"', 'name="glossy"'))
        with pytest.raises(RuntimeError, match="matte_black"):
            head_camera.attach(spec, loaded)
        assert spec.body("openarm_head_camera") is None

    def test_refuses_a_second_attach(self, loaded):
        spec = _spec()
        head_camera.attach(spec, loaded)
        with pytest.raises(RuntimeError, match="already has"):
            head_camera.attach(spec, loaded)


class TestLoadFor:
    def test_the_models_that_draw_the_head_camera_share_the_staged_pack(self, pack):
        loaded = head_camera.load_for(MujocoModels.read(), pack.directory)
        assert loaded == head_camera.Pack(directory=pack.directory, body_position=(0.0315, 0.0, 0.743))

    def test_an_engine_whose_models_draw_none_reads_no_pack(self, tmp_path):
        """Nothing is staged where it looks, and nothing needs to be."""
        (tmp_path / "so101.json5").write_text('{ scene: "so101/scene.xml" }')
        assert head_camera.load_for(MujocoModels.read(tmp_path), tmp_path / "head_camera") is None

    def test_the_chest_camera_of_every_model_that_draws_it_is_checked(self, tmp_path, monkeypatch):
        rig = copy.deepcopy(_RIG)
        rig["cameras"]["left"]["position_m"] = [0.0794, 0.031497, 0.794103]
        directory = tmp_path / "head_camera"
        monkeypatch.setattr(head_camera, "PACK_DIGEST", _stage(directory, _pack_files(rig)))
        with pytest.raises(RuntimeError, match="left lens front"):
            head_camera.load_for(MujocoModels.read(), directory)


class TestLauncher:
    @pytest.fixture(name="load")
    def load_fixture(self, tmp_path, monkeypatch):
        """Loads the pedestal scene the way a stand does, as a model that
        carries the OpenArm v2's chest camera."""
        monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
        (tmp_path / "openarm_bimanual_v2.xml").write_text(_SCENE)
        chest = tuple(c for c in _OPENARM_V2.cameras if c.name == head_camera.CHEST_CAMERA)
        entry = dataclasses.replace(_OPENARM_V2, cameras=chest)

        def load(head_camera_pack, renders: bool, draws: bool = True):
            engine = {
                "scene": "openarm_bimanual_v2.xml",
                "world_links": [head_camera.PEDESTAL_LINK],
                "head_camera": draws,
            }
            known = mujoco_models.parse(EngineModel(entry=entry, engine=engine))
            launcher = SimLauncher(
                None,
                threading.Event(),
                None,
                100,
                True,
                "0.0.0.0",
                8080,
                renders=renders,
                head_camera_pack=head_camera_pack,
            )
            return launcher._load_model(known)  # pylint: disable=W0212

        return load

    def test_loads_the_scene_with_the_head_camera_and_the_rendered_cameras(self, load, loaded):
        rendered = load(loaded, renders=True)
        assert _geom(rendered, "head_camera_cover") >= 0
        assert mujoco.mj_name2id(rendered, mujoco.mjtObj.mjOBJ_CAMERA, "chest") >= 0

    def test_the_head_camera_is_drawn_whether_or_not_the_cameras_render(self, load, loaded):
        drawn = load(loaded, renders=False)
        assert _geom(drawn, "head_camera_cover") >= 0
        assert drawn.ncam == 0

    def test_a_model_that_draws_no_head_camera_loads_the_scene_as_baked(self, load, loaded):
        bare = load(loaded, renders=False, draws=False)
        assert _geom(bare, "head_camera_cover") == -1
        assert bare.ngeom == 2

    def test_a_model_that_draws_it_is_refused_while_no_pack_is_staged(self, load):
        with pytest.raises(RuntimeError, match="draws the head camera, and no pack was staged"):
            load(None, renders=False)


def test_node_stages_the_head_camera_pack_where_launch_reads_it():
    # The baked scene is upstream's robot without its head camera; the node
    # stages Waldo's head camera pack at image build, after the %files copy
    # that brings head_camera.py in, at the directory launch.py reads at
    # setup. The staged pack is generated, so git ignores it.
    definition = (_NODE_DIR / "apptainer.def").read_text()
    module = "/opt/sim_mujoco/engine/head_camera.py"
    directory = "/opt/sim_mujoco/engine/assets/head_camera"
    fetch = re.search(
        rf"^\s*python3 {re.escape(module)} fetch \\\n\s*{re.escape(directory)}$",
        definition,
        re.MULTILINE,
    )
    assert fetch is not None
    assert definition.index("%files") < definition.index("%post") < fetch.start()
    assert re.search(r"^%post\n\s*set -e$", definition, re.MULTILINE)
    launch = (_ENGINE_DIR / "launch.py").read_text()
    assert '_HEAD_CAMERA_DIR = Path(__file__).parent / "assets" / "head_camera"' in launch
    assert "head_camera.load_for(models, _HEAD_CAMERA_DIR)" in launch
    assert "head_camera_pack=head_camera_pack" in launch
    ignored = (_NODE_DIR.parent / ".gitignore").read_text().splitlines()
    assert "sim_mujoco/engine/assets/head_camera/" in ignored
