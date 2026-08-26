from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class IsaacGripperSensor:
    """Reads finger joint positions and applied forces from an Isaac Sim
    articulation. finger_joints is a list of DOF names to monitor; names not
    found in the articulation are skipped with a warning at setup time."""

    def __init__(self, prim_path: str, finger_joints: list[str]) -> None:
        self._prim_path = prim_path
        self._finger_joints = finger_joints
        self._articulation = None
        self._finger_indices: list[int] = []
        self._resolved_names: list[str] = []
        self._ready: bool = False
        # One-shot guards, both read on the first force sample: the fallback
        # and the layout note fire every physics step otherwise.
        self._force_warning_logged: bool = False
        self._force_layout_logged: bool = False

    def setup(self) -> bool:
        """Initialise the Articulation and resolve finger joint indices."""
        if self._articulation is not None and self._ready:
            return True
        try:
            from isaacsim.core.prims import Articulation  # pylint: disable=E0401

            self._articulation = Articulation(prim_paths_expr=self._prim_path)
            self._articulation.initialize()

            dof_names = list(self._articulation.dof_names)
            self._finger_indices = []
            self._resolved_names = []
            for name in self._finger_joints:
                if name in dof_names:
                    self._finger_indices.append(dof_names.index(name))
                    self._resolved_names.append(name)
                else:
                    logger.warning(
                        f"IsaacGripperSensor: finger joint '{name}' not found"
                        f" in articulation at '{self._prim_path}'."
                        f" Available DOFs: {dof_names}"
                    )

            self._ready = True
            self._force_warning_logged = False
        except Exception as exc:
            logger.exception(f"Failed to setup IsaacGripperSensor at '{self._prim_path}'")
            self._articulation = None
            self._ready = False
            return False

        logger.info(
            f"IsaacGripperSensor ready — prim='{self._prim_path}'"
            f" fingers={self._resolved_names}"
        )
        return True

    def teardown(self) -> None:
        """Reset sensor state."""
        self._articulation = None
        self._finger_indices = []
        self._resolved_names = []
        self._ready = False
        self._force_warning_logged = False
        self._force_layout_logged = False

    def get_gripper_state(self) -> Optional[dict]:
        """Return finger joint names, positions, and applied forces."""
        if not self._ready or self._articulation is None:
            return None

        try:
            all_positions = self._articulation.get_joint_positions()[0]
            positions = [float(all_positions[i]) for i in self._finger_indices]

            try:
                # Rows are [fx, fy, fz, tx, ty, tz]. Depending on the Isaac
                # binding the array is per-dof (num_dof rows) or per-link with
                # the root wrench at row 0 (num_dof + 1 rows); detect which so
                # a finger never reads its parent joint's wrench.
                all_forces = self._articulation.get_measured_joint_forces()[0]
                import numpy as np  # pylint: disable=E0401

                offset = 1 if len(all_forces) == len(all_positions) + 1 else 0
                if not self._force_layout_logged:
                    logger.info(
                        f"measured joint forces: {len(all_forces)} rows for "
                        f"{len(all_positions)} dofs (row offset {offset})"
                    )
                    self._force_layout_logged = True
                applied_forces = [
                    float(np.linalg.norm(all_forces[i + offset][:3]))
                    for i in self._finger_indices
                ]
            except Exception as exc:
                if not self._force_warning_logged:
                    logger.warning(
                        "Could not read gripper applied forces — falling back"
                        f" to zeros (warning once): {exc}"
                    )
                    self._force_warning_logged = True
                applied_forces = [0.0] * len(self._finger_indices)

            return {
                "joint_names": self._resolved_names,
                "positions": positions,
                "applied_forces": applied_forces,
            }
        except Exception as exc:
            logger.warning(f"Could not read gripper state: {exc}")
            return None

    @property
    def is_ready(self) -> bool:
        """True when the Articulation has been initialised."""
        return self._ready
