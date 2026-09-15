#!/usr/bin/env python3
"""Object state of the objects spawned through scene_control, read on
Isaac's main thread.

A snapshot is one capture of every spawned object, stamped on the timeline
the joint states use: the object_states stream publishes it on the state
tick and get_object_states answers it until the next capture.

Engine reads live in three small functions below. A dynamic object's pose
and velocities come from PhysX through a tensor rigid-body view: the USD
stage holds the pose it was authored with, which a Fabric-backed stage does
not update as the body moves. A static or visual-only object reports the
pose authored on the USD stage, which is where it is, and zero velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

# The stage root of runtime objects. The launcher spawns, moves and removes
# each scene_control object at object_prim_path(object_id), the path the
# reader reads it from.
RUNTIME_OBJECTS_PATH = "/World/RuntimeObjects"

_AT_REST = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ObjectRecord:
    """One spawned object in a snapshot: its world pose (orientation a unit
    quaternion x, y, z, w), its world-axis velocities at its origin, and the
    asset, physics, mass and scale it was spawned with."""

    object_id: str
    asset_id: str
    physics: str
    mass: float
    scale: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]

    def fields(self) -> dict:
        """The record as the generated object_state item types take it."""
        return {
            "object_id": self.object_id,
            "asset_id": self.asset_id,
            "physics": self.physics,
            "mass": self.mass,
            "scale": self.scale,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
        }


@dataclass(frozen=True)
class ObjectStateSnapshot:
    """Every spawned object at one capture, and the instant of that capture."""

    timestamp_s: float
    objects: tuple[ObjectRecord, ...]


def object_prim_path(object_id: str) -> str:
    return f"{RUNTIME_OBJECTS_PATH}/{object_id}"


class IsaacObjectReader:
    """Reads the world state of spawned objects, on Isaac's main thread only.

    The rigid-body view over the dynamic objects is kept between captures.
    It is created again when the set of dynamic objects changes, after any
    failed read, and after invalidate(): PhysX invalidates its tensor views
    when a prim leaves the stage.
    """

    def __init__(self) -> None:
        self._body_view = None
        self._body_paths: tuple[str, ...] = ()

    def invalidate(self) -> None:
        """Drop the rigid-body view; the next read creates it again."""
        self._body_view = None
        self._body_paths = ()

    def read(self, spawned: list[dict]) -> list[ObjectRecord]:
        """One record per registry entry, in the order given. Raises when an
        object has no state to read: a snapshot lists every spawned object
        or it is not one."""
        dynamic_paths = tuple(
            object_prim_path(obj["object_id"])
            for obj in spawned
            if obj["physics"] == "dynamic"
        )
        bodies = self._read_bodies(dynamic_paths) if dynamic_paths else {}

        records = []
        for obj in spawned:
            path = object_prim_path(obj["object_id"])
            if obj["physics"] == "dynamic":
                position, orientation, linear_velocity, angular_velocity = bodies[path]
            else:
                position, orientation = _authored_world_pose(path)
                linear_velocity = angular_velocity = _AT_REST
            records.append(
                ObjectRecord(
                    object_id=obj["object_id"],
                    asset_id=obj["asset_id"],
                    physics=obj["physics"],
                    mass=float(obj["mass"]),
                    scale=float(obj["scale"]),
                    position=position,
                    orientation=orientation,
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                )
            )
        return records

    def _read_bodies(self, paths: tuple[str, ...]) -> dict[str, tuple]:
        """Pose and velocities of each dynamic object, keyed by prim path."""
        try:
            # A spawn or move made since the last step is still buffered in
            # the physics backend; flushing it first makes a capture taken
            # right after an edit read the body as the edit left it.
            _flush_physics_changes()
            if self._body_view is None or self._body_paths != paths:
                self._body_view = _create_body_view(list(paths))
                self._body_paths = paths
            view = self._body_view
            # (N, 7): position, then the quaternion x, y, z, w.
            transforms = view.get_transforms().numpy().tolist()
            # (N, 6): linear then angular, world axes, at the centre of mass.
            velocities = view.get_velocities().numpy().tolist()
            # (N, 7): the centre of mass pose in the body frame.
            coms = view.get_coms().numpy().tolist()
            bodies = {}
            for path, transform, velocity, com in zip(
                view.prim_paths, transforms, velocities, coms
            ):
                orientation = tuple(transform[3:7])
                angular = tuple(velocity[3:6])
                bodies[path] = (
                    tuple(transform[0:3]),
                    orientation,
                    origin_velocity(velocity[0:3], angular, orientation, com[0:3]),
                    angular,
                )
            missing = [path for path in paths if path not in bodies]
            if missing:
                raise RuntimeError(
                    f"PhysX simulates no rigid body at {', '.join(missing)}"
                )
            return bodies
        except Exception:
            self.invalidate()
            raise


def origin_velocity(
    com_linear: list[float],
    angular: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    com_offset: list[float],
) -> tuple[float, float, float]:
    """The linear velocity at the body's origin from the one PhysX reports at
    its centre of mass: v_origin = v_com - w x (R * com_offset), where
    com_offset is the centre of mass in the body frame and R the body's
    orientation."""
    offset = _rotate(orientation, com_offset)
    wx, wy, wz = angular
    cross = (
        wy * offset[2] - wz * offset[1],
        wz * offset[0] - wx * offset[2],
        wx * offset[1] - wy * offset[0],
    )
    return tuple(float(v - c) for v, c in zip(com_linear, cross))


def _rotate(
    orientation: tuple[float, float, float, float], vector: list[float]
) -> tuple[float, float, float]:
    """Rotate vector by the unit quaternion (x, y, z, w)."""
    x, y, z, w = orientation
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


# --- engine reads, Isaac's main thread only ---


def _flush_physics_changes() -> None:
    import omni.physics.core  # pylint: disable=C0415

    omni.physics.core.get_physics_simulation_interface().flush_changes()


def _create_body_view(paths: list[str]):
    from isaacsim.core.simulation_manager import SimulationManager  # pylint: disable=C0415

    simulation_view = SimulationManager.get_physics_simulation_view()
    if simulation_view is None:
        raise RuntimeError("Isaac has no physics simulation view yet")
    return simulation_view.create_rigid_body_view(paths)


def _authored_world_pose(
    prim_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """World position and orientation (x, y, z, w) authored on the stage,
    with the spawn scale taken out of the rotation."""
    import omni.usd  # pylint: disable=C0415
    from pxr import Usd, UsdGeom  # pylint: disable=C0415

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"no prim at {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    rotation = matrix.RemoveScaleShear().ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(rotation.GetReal()),
        ),
    )
