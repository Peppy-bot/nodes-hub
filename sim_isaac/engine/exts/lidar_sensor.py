from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from control_core_py.geometry import Placement, UnitQuaternion, xyzw_of_wxyz
from sim_robot_core.lidars import LidarConfig, LidarSweeper, Scan

from object_state import create_body_view, pose_of_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LidarMount:
    """Where a lidar's frame is fixed on its robot: the rigid body that
    carries the prim of its parent link, by its stage path, and the frame's
    placement in that body."""

    body_path: str
    frame_in_body: Placement


def mount_of(root, known, lidar: LidarConfig) -> LidarMount:
    """The mount of `lidar` on the robot of the model `known` under `root`:
    the rigid body nearest the prim of its parent link up the stage, that
    prim or one above it under `root`, and the frame's placement there, the
    link's placement in the body as the stage authors it turned and moved by
    the frame's pose in the link. A link no rigid body carries is refused,
    since no step reads where it stands."""
    from pxr import UsdGeom, UsdPhysics  # pylint: disable=E0401

    link = known.mount_prim(root, f"lidar '{lidar.name}'", lidar.parent_link)
    body = next(
        (prim for prim in _prim_and_above(link, root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        None,
    )
    if body is None:
        raise RuntimeError(
            f"lidar '{lidar.name}' hangs from '{lidar.parent_link}', and no rigid body carries "
            f"its prim {link.GetPath()}: give that prim or one above it under {root.GetPath()} "
            "PhysicsRigidBodyAPI"
        )
    matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(link, body)
    position, orientation = pose_of_matrix(matrix)
    link_in_body = Placement(position, UnitQuaternion.parse(orientation))
    frame_in_link = Placement(lidar.pos, UnitQuaternion.parse(xyzw_of_wxyz(lidar.quat_wxyz)))
    return LidarMount(str(body.GetPath()), link_in_body.compose(frame_in_link))


def _prim_and_above(prim, root):
    """`prim` and each prim above it, up to `root`."""
    while True:
        yield prim
        if prim.GetPath() == root.GetPath():
            return
        prim = prim.GetParent()


class PhysxLidarScene:
    """The PhysX scene a lidar's rays are cast in, and the pose PhysX gives
    the body that carries the lidar: a Fabric-backed stage does not update
    the pose USD authors as the body moves."""

    def __init__(self, body_path: str, view) -> None:
        import carb  # pylint: disable=E0401
        from omni.physx import get_physx_scene_query_interface  # pylint: disable=E0401

        self._body_path = body_path
        self._view = view
        self._float3 = carb.Float3
        self._query = get_physx_scene_query_interface()

    @staticmethod
    def of(body_path: str) -> Optional["PhysxLidarScene"]:
        """The scene of the body at `body_path`, None while PhysX simulates
        no body there."""
        view = create_body_view([body_path])
        if list(view.prim_paths) != [body_path]:
            return None
        return PhysxLidarScene(body_path, view)

    def body_placement(self) -> Placement:
        """Where the body stands in world now."""
        x, y, z, qx, qy, qz, qw = self._view.get_transforms().numpy().tolist()[0]
        return Placement((x, y, z), UnitQuaternion.parse((qx, qy, qz, qw)))

    def cast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_range_m: float,
    ) -> Optional[float]:
        """The distance along the unit `direction` from `origin` to the
        nearest collider within `max_range_m` (m), the body's own passed, or
        None where the ray meets none."""
        distances = []

        def take(hit) -> bool:
            if hit.rigid_body != self._body_path:
                distances.append(hit.distance)
            return True

        self._query.raycast_all(
            self._float3(*origin), self._float3(*direction), max_range_m, take
        )
        return min(distances, default=None)


class IsaacLidarSensor:
    """Sweeps one 2-D lidar of one robot's model on the engine clock and
    publishes its scans, each on the pair of the lidar's relay.

    The sweep is sim_robot_core.lidars' LidarSweeper: after each step it
    casts every ray due since the step before from the lidar's frame, at the
    pose the frame takes at the ray's own time, and a scan starts only while
    a pair views the lidar. The frame is the prim of the lidar's parent link,
    placed in the rigid body that carries that prim, and each ray is cast in
    PhysX's scene: it passes the colliders of that body, as Waldo's passes
    the geoms of the body that carries its lidar's site, and meets every
    other collider of the scene, the robot's other bodies included. A scan
    is stamped with the time of its ray 0.

    Setup is deferred like every other ext: the mount and the body's view
    can only be read once the stage plays. A teardown, which every edit of
    the stage and every pair that goes runs, lets go of the body's view and
    drops the scan in progress. The sweep stays, and with it the capture
    index, for as long as the robot stands.
    """

    def __init__(self, robot: str, root_prim: str, known, lidar: LidarConfig, io) -> None:
        # The robot this lidar is mounted on: every scan goes to that robot's
        # own lidar pair.
        self._robot = robot
        self._root = root_prim
        self._known = known
        self._lidar = lidar
        self._io = io
        self._sweeper = LidarSweeper(lidar, lambda: io.lidar_viewed(robot, lidar.name))
        self._mount: Optional[LidarMount] = None
        self._scene: Optional[PhysxLidarScene] = None
        self._ready = False
        self._dropping = False

    def setup(self) -> bool:
        """Read the lidar's mount off the live stage and take a view of the
        body that carries it. False while the stage or the robot is not
        loaded yet or PhysX simulates no body there; a parent link missing
        from a loaded robot, or one no rigid body carries, raises, because
        that is an error of the model's entry no retry can fix."""
        if self._ready:
            return True
        try:
            import omni.usd  # pylint: disable=E0401
        except ImportError as exc:
            logger.error(f"omni.usd unavailable: {exc}")
            return False
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        root = stage.GetPrimAtPath(self._root)
        if not root or not root.IsValid():
            return False
        mount = mount_of(root, self._known, self._lidar)
        scene = PhysxLidarScene.of(mount.body_path)
        if scene is None:
            return False
        self._mount = mount
        self._scene = scene
        self._ready = True
        sweep = self._lidar.sweep
        logger.info(
            f"IsaacLidarSensor ready: '{self._robot}' {self._lidar.name} on {mount.body_path}, "
            f"{sweep.rays} rays at {sweep.scan_rate_hz} Hz"
        )
        return True

    def teardown(self) -> None:
        """Lets go of the body's view, which an edit of the stage leaves
        reading nothing, and drops the scan in progress."""
        self._mount = None
        self._scene = None
        self._ready = False
        self._sweeper.interrupt()

    def step(self, now_s: float) -> None:
        """Runs on the main thread after physics advanced to `now_s`, the
        engine time of this step: casts every ray due since the step before
        and publishes each scan that completes."""
        frame = self._scene.body_placement().compose(self._mount.frame_in_body)
        for scan in self._sweeper.step(now_s, frame, self._scene.cast):
            self._publish(scan, now_s)

    def _publish(self, scan: Scan, now_s: float) -> None:
        published = self._io.publish_scan(
            self._robot,
            self._lidar,
            self._io.timestamp_at_s(scan.start_s, now_s),
            scan.capture_index,
            list(scan.ranges),
        )
        if not published and not self._dropping:
            self._dropping = True
            logger.warning(
                f"'{self._robot}' {self._lidar.name}: a scan was dropped before the wire because "
                "its pair is gone or the previous publish is still in flight, suppressing repeats"
            )
        elif published:
            self._dropping = False

    @property
    def is_ready(self) -> bool:
        return self._ready
