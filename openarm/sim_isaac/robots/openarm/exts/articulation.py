from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_ARTICULATION_NAME = "peppy_bridge_articulation"


class IsaacArticulation:
    """Read joint states and send position commands to an articulated prim."""

    def __init__(self, prim_path: str) -> None:
        self._prim_path = prim_path
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
                name=_ARTICULATION_NAME,
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
            f"Articulation ready — prim='{self._prim_path}'  dof={self._num_dof}"
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
        """Cross-engine alias of get_dof_names — order matches get_joint_states()."""
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

        if len(positions) != self._num_dof:
            logger.warning(
                f"Command length {len(positions)} does not match "
                f"robot DOF {self._num_dof} — dropped."
            )
            return False
        try:
            targets = np.array([positions], dtype=np.float32)
            self._view.set_joint_position_targets(targets)
            return True
        except Exception as exc:
            logger.warning(f"Could not apply joint command: {exc}")
            return False

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom for the articulation."""
        return self._num_dof

    @property
    def is_ready(self) -> bool:
        """True when the articulation has been initialised."""
        return self._ready
