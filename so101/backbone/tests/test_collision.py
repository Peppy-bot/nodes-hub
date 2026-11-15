"""The collision geometry of SO-101's URDF: every link with a body holds
collision meshes its model has, each a closed convex hull, the wrist camera
module holds the wrist camera, and the solvers load the meshes with no
self-collision pairs."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from so101_backbone.kinematics import ApproachSolver, Kinematics
from so101_backbone.model import ARM, DESCRIPTION, urdf_path

# The tool frame and the cameras' optical frames: points with no body of
# their own.
BODILESS = {ARM.grasp_link} | {f"{camera.name}_optical_frame" for camera in DESCRIPTION.cameras}

# How far a vertex of a hull may stand in front of one of its facets, in
# meters: the rounding of the STL's single-precision coordinates.
HULL_TOLERANCE_M = 1e-6

CAMERA_MODULE = "meshes/wrist_camera_module.stl"


def test_every_link_with_a_body_holds_collision_meshes_its_model_has():
    urdf = Path(urdf_path())
    for link in ET.parse(urdf).getroot().findall("link"):
        name = link.get("name")
        meshes = [mesh.get("filename") for mesh in link.findall("collision/geometry/mesh")]
        if name in BODILESS:
            assert meshes == [], name
            continue
        assert meshes, f"{name} holds no collision mesh"
        for mesh in meshes:
            assert (urdf.parent / mesh).is_file(), f"{name}'s {mesh} is not in the model"


def test_each_collision_mesh_is_a_closed_convex_hull():
    root = ET.parse(urdf_path()).getroot()
    meshes = {mesh.get("filename") for mesh in root.iter("mesh")}
    for mesh in sorted(meshes):
        corners = _facet_corners(Path(urdf_path()).parent / mesh)
        _assert_closed(mesh, corners)
        _assert_convex(mesh, corners)


def test_the_wrist_camera_module_holds_the_wrist_camera():
    (camera,) = DESCRIPTION.cameras
    link = next(
        link
        for link in ET.parse(urdf_path()).getroot().findall("link")
        if link.get("name") == camera.parent_link
    )
    (module,) = [
        collision
        for collision in link.findall("collision")
        if collision.find("geometry/mesh").get("filename") == CAMERA_MODULE
    ]
    # The module's mesh is in the link's frame, the frame of the camera's pose.
    origin = module.find("origin")
    assert [float(v) for v in origin.get("xyz").split() + origin.get("rpy").split()] == [0.0] * 6
    unit_normals, offsets = _facet_planes(
        _facet_corners(Path(urdf_path()).parent / CAMERA_MODULE)
    )
    heights = unit_normals @ np.array(camera.xyz) - offsets
    assert heights.max() < 0.0, f"the wrist camera stands {heights.max():.2e} m outside its module"


def test_the_solvers_load_the_collision_meshes_with_no_self_collision_pairs(capfd):
    Kinematics(urdf_path())
    ApproachSolver(urdf_path())
    captured = capfd.readouterr()
    assert "self collisions" not in captured.out + captured.err


def _facet_corners(path: Path) -> np.ndarray:
    """The corners of each facet of a binary STL, in its winding order:
    shape (facets, 3, 3)."""
    stl = path.read_bytes()
    count = int(np.frombuffer(stl, dtype="<u4", count=1, offset=80)[0])
    facet = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(stl, dtype=facet, count=count, offset=84)["corners"].astype(float)


def _facet_planes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The plane of each facet with an area, its unit normal the way its
    winding faces and its offset along that normal."""
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    with_area = lengths > 0.0
    unit_normals = normals[with_area] / lengths[with_area, None]
    return unit_normals, np.einsum("fk,fk->f", unit_normals, corners[with_area, 0])


def _assert_closed(mesh: str, corners: np.ndarray) -> None:
    """Each edge of a facet is an edge of exactly one other facet, which runs
    it the other way: the mesh is closed and every facet is wound alike."""
    _, index = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    a, b, c = index.reshape(-1, 3).T
    edges = {(int(p), int(q)) for p, q in zip(np.concatenate([a, b, c]), np.concatenate([b, c, a]))}
    assert len(edges) == 3 * len(corners), f"{mesh} runs an edge the same way twice"
    assert {(q, p) for p, q in edges} == edges, f"{mesh} has an edge of one facet"


def _assert_convex(mesh: str, corners: np.ndarray) -> None:
    """Every vertex stands on or behind the plane of every facet, which its
    winding faces outward."""
    vertices = np.unique(corners.reshape(-1, 3), axis=0)
    unit_normals, offsets = _facet_planes(corners)
    for start in range(0, len(offsets), 512):
        rows = slice(start, start + 512)
        height = (vertices @ unit_normals[rows].T - offsets[rows]).max()
        assert height <= HULL_TOLERANCE_M, f"{mesh} has a vertex {height:.2e} m in front of a facet"
