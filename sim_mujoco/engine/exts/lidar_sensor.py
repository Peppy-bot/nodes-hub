"""The 2-D lidars of the robots standing in the scene, swept on the engine
clock by sim_robot_core.lidars' LidarSweeper, which holds when each ray of a
scan falls due, the frame's pose at each ray's time, the scans each step
completes and drops, their capture index and the scans no pair views. What
is MuJoCo's is here: where each lidar's frame stands, how a ray is cast, and
when the sweep is stepped and interrupted.

After each physics step every lidar is stepped at the instant of the
geometry that step computed: mj_step computes the scene's geometry before it
advances the clock, so after the step the data holds the geometry of the
engine time less one timestep, and every ray due since the step before is
cast against it.

A lidar's frame stands at its pose (LidarConfig.pos, quat_wxyz) on the body
its parent link is in the robot's MJCF, by the release's map of links to
bodies, as a camera's does. A ray is cast with mj_ray: it passes the geoms
of that body, as MuJoCo's rangefinder passes those of its site's body, and
meets every other geom the cameras draw (geom groups 0 to 2), the floor and
the robot's other bodies included.

A scan starts only while the lidar's robot holds the lidar's pair
(SimTopicIO.lidar_viewed). Composing the scene again interrupts each sweep,
which drops the scan in progress, and hands the sweep of each lidar of a
robot that stays to the scene composed next, so its capture index counts on
from where it was; a robot that stands again starts its count again.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from control_core_py.geometry import Placement, UnitQuaternion, Vector, xyzw_of_wxyz
from sim_robot_core.lidars import LidarConfig, LidarSweeper, Scan

# The geom groups a ray meets: the ones the cameras draw, MuJoCo's default
# visible groups 0 to 2, so a lidar and a depth camera see one scene.
_DRAWN_GEOM_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class _Lidar:
    """One lidar of one standing robot: the robot whose pair carries its
    scans, its config, the body that carries it, its pose in that body, and
    its sweep."""

    robot: str
    config: LidarConfig
    body: int
    mount: Placement
    sweeper: LidarSweeper


class MujocoLidarSensor:
    """Sweeps the lidars of the robots standing in the scene on the physics
    thread, and publishes each scan on the pair of its lidar's relay."""

    def __init__(self, model, data, robots, io, sweepers=None) -> None:
        """`robots` are the robots standing (world.Robot), each lidar of
        whose model is swept on its own. A lidar that was already sweeping
        takes its sweep from `sweepers`, what the sensor of the scene before
        this one handed on (`sweepers()`), and any other a new one."""
        self._model = model
        self._data = data
        self._io = io
        kept = sweepers or {}
        self._lidars = tuple(
            _Lidar(
                robot=robot.instance,
                config=lidar,
                body=_carrier(model, robot, lidar),
                mount=Placement(lidar.pos, UnitQuaternion.parse(xyzw_of_wxyz(lidar.quat_wxyz))),
                sweeper=kept.get((robot.instance, lidar.name))
                or _sweeper(io, robot.instance, lidar),
            )
            for robot in robots
            for lidar in robot.known.entry.lidars
        )

    def sweepers(self) -> dict:
        """Each lidar's sweep, by robot and lidar, interrupted, for the scene
        composed next."""
        for lidar in self._lidars:
            lidar.sweeper.interrupt()
        return {(lidar.robot, lidar.config.name): lidar.sweeper for lidar in self._lidars}

    def after_step(self, engine_time_s: float) -> None:
        """Steps every lidar's sweep after a physics step that advanced the
        engine clock to `engine_time_s`, at the instant of the geometry the
        step computed, and publishes each scan it completes."""
        geometry_s = engine_time_s - self._model.opt.timestep
        for lidar in self._lidars:
            scans = lidar.sweeper.step(
                geometry_s, self._frame(lidar), partial(self._cast, lidar.body)
            )
            for scan in scans:
                self._publish(lidar, scan, engine_time_s)

    def _publish(self, lidar: _Lidar, scan: Scan, engine_time_s: float) -> None:
        """One scan on the pair of its lidar's relay, stamped with the time
        of its ray 0 (SimTopicIO.timestamp_at_s) and naming the lidar's
        frame."""
        self._io.publish_scan(
            lidar.robot,
            lidar.config,
            self._io.timestamp_at_s(scan.start_s, engine_time_s),
            scan.capture_index,
            list(scan.ranges),
        )

    def _cast(
        self, body: int, origin: Vector, direction: Vector, _max_range_m: float
    ) -> Optional[float]:
        """How far along the ray from `origin` along the unit `direction`
        its first surface lies, past the geoms of `body`; None for none.
        mj_ray meets a surface at any distance, and the sweep marks one past
        the lidar's farthest range as no return."""
        import mujoco  # pylint: disable=C0415

        distance = mujoco.mj_ray(
            self._model,
            self._data,
            np.asarray(origin),
            np.asarray(direction),
            _DRAWN_GEOM_GROUPS,
            True,
            body,
            None,
        )
        return None if distance < 0.0 else distance

    def _frame(self, lidar: _Lidar) -> Placement:
        """Where `lidar`'s frame stands in world as the data's geometry
        stands its body: its pose in the body, carried by the body's pose in
        world."""
        orientation = xyzw_of_wxyz([float(value) for value in self._data.xquat[lidar.body]])
        carrier = Placement(
            tuple(float(value) for value in self._data.xpos[lidar.body]),
            UnitQuaternion.parse(orientation),
        )
        return carrier.compose(lidar.mount)


def _sweeper(io, robot: str, lidar: LidarConfig) -> LidarSweeper:
    """A new sweep of `robot`'s lidar, which starts a scan only while the
    robot holds the lidar's pair."""
    return LidarSweeper(lidar, lambda: io.lidar_viewed(robot, lidar.name))


def _carrier(model, robot, lidar: LidarConfig) -> int:
    """The body that carries `robot`'s lidar: the one its parent link is in
    the robot's MJCF, by the release's map of links to bodies, as a camera's.
    A parent the MJCF lacks is an error of the release, refused as the robot
    stands, naming the robot's bodies."""
    import mujoco  # pylint: disable=C0415

    name = robot.scene_name(robot.known.body_of(lidar.parent_link))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body >= 0:
        return body
    bodies = sorted(
        model.body(index).name[len(robot.prefix) :]
        for index in range(model.nbody)
        if model.body(index).name.startswith(robot.prefix)
    )
    raise RuntimeError(
        f"lidar '{lidar.name}' parent link '{lidar.parent_link}' is not a body of "
        f"{robot.known.scene_path().name}; bodies: {bodies}"
    )
