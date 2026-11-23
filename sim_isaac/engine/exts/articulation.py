from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from object_state import origin_velocity

logger = logging.getLogger(__name__)


class IsaacArticulation:
    """Read joint states from an articulated prim. Isaac registers a view
    under a name, so a stage holding several robots names each view after the
    robot it reads."""

    def __init__(self, prim_path: str, name: str) -> None:
        self._prim_path = prim_path
        self._name = name
        self._view = None
        self._num_dof: int = 0
        self._ready: bool = False

    def setup(self) -> bool:
        """Initialise the Articulation against the live USD stage."""
        if self._ready:
            return True
        try:
            from isaacsim.core.prims import Articulation  # pylint: disable=E0401

            self._view = Articulation(
                prim_paths_expr=self._prim_path,
                name=self._name,
            )
            self._view.initialize()
            self._num_dof = self._view.num_dof
            self._ready = True
        except Exception as exc:
            logger.error(
                f"Failed to initialise Articulation at '{self._prim_path}': {exc}"
            )
            self._view = None
            self._ready = False
            return False

        logger.info(
            f"Articulation ready: prim='{self._prim_path}'  dof={self._num_dof}"
        )
        return True

    def teardown(self) -> None:
        """Release the Articulation view and reset DOF count."""
        self._view = None
        self._num_dof = 0
        self._ready = False

    def get_dof_names(self) -> list[str]:
        """Return DOF names in the order Isaac Sim uses them."""
        if self._view is None:
            return []
        try:
            return list(self._view.dof_names)
        except Exception as exc:
            logger.warning(f"Could not read DOF names: {exc}")
            return []

    def get_joint_names(self) -> list[str]:
        """Cross-engine alias of get_dof_names; order matches get_joint_states()."""
        return self.get_dof_names()

    def get_joint_limits(self) -> Optional[tuple[list[float], list[float]]]:
        """Return (lower, upper) DOF limits in get_joint_states() order.
        PhysX's ±3.4e38 unbounded sentinels are squashed to ±1e6 (JSON-safe)."""
        if self._view is None:
            return None
        try:
            limits = self._view.get_dof_limits()[0]
            lower = [max(float(lo), -1e6) for lo in limits[:, 0]]
            upper = [min(float(hi), 1e6) for hi in limits[:, 1]]
            return lower, upper
        except Exception as exc:
            logger.warning(f"Could not read joint limits: {exc}")
            return None

    def get_joint_states(self) -> Optional[tuple[list[float], list[float]]]:
        """Read current joint positions and velocities."""
        if self._view is None:
            return None
        try:
            positions = self._view.get_joint_positions()[0].tolist()
            velocities = self._view.get_joint_velocities()[0].tolist()
            return positions, velocities
        except Exception as exc:
            logger.warning(f"Could not read joint states: {exc}")
            return None

    def get_root_state(self):
        """The world pose of the articulation's root body, PhysX's first, and
        its velocities at its origin, in world: (position, orientation (x, y,
        z, w), linear, angular). PhysX gives the root's linear velocity at its
        centre of mass, which is carried to the origin here. None when the
        view cannot read."""
        if self._view is None:
            return None
        try:
            positions, orientations = self._view.get_world_poses()
            velocities = self._view.get_velocities()
            com_positions, _ = self._view.get_body_coms(body_indices=np.array([0]))
            w, x, y, z = (float(value) for value in orientations[0])
            orientation = (x, y, z, w)
            angular = tuple(float(value) for value in velocities[0][3:6])
            linear = origin_velocity(
                [float(value) for value in velocities[0][0:3]],
                angular,
                orientation,
                [float(value) for value in com_positions[0][0]],
            )
            return tuple(float(value) for value in positions[0]), orientation, linear, angular
        except Exception as exc:
            logger.warning(f"Could not read the root of '{self._prim_path}': {exc}")
            return None

    def set_down_at(self, position, orientation) -> bool:
        """Stands the articulation's root at `position` turned to
        `orientation` (x, y, z, w), at rest: no velocity of its root and none
        of its joints. True once written."""
        if self._view is None:
            return False
        x, y, z, w = orientation
        try:
            self._view.set_world_poses(
                positions=np.array([position], dtype=np.float32),
                orientations=np.array([(w, x, y, z)], dtype=np.float32),
            )
            self._view.set_velocities(np.zeros((1, 6), dtype=np.float32))
            self._view.set_joint_velocities(np.zeros((1, self._num_dof), dtype=np.float32))
            return True
        except Exception as exc:
            logger.warning(f"Could not set down '{self._prim_path}': {exc}")
            return False

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom for the articulation."""
        return self._num_dof

    @property
    def is_ready(self) -> bool:
        """True when the articulation has been initialised."""
        return self._ready
