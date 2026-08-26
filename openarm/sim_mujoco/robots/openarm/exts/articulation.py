from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_JOINT_TRANSMISSION_TYPE = 0  # mjTRN_JOINT


class MujocoArticulation:
    """Read joint states and send position commands to a MuJoCo model."""

    def __init__(self, model, data) -> None:
        self._model = model
        self._data = data
        self._qpos_indices: list[int] = []
        self._qvel_indices: list[int] = []
        self._ctrl_indices: list[int] = []
        self._joint_ids: list[int] = []
        self._joint_names: list[str] = []
        self._num_dof: int = 0
        self._ready: bool = False

    def setup(self) -> bool:
        """Resolve actuated joint indices + names from model metadata."""
        try:
            import mujoco  # pylint: disable=E0401,C0415

            actuator_indices = [
                i
                for i in range(self._model.nu)
                if self._model.actuator_trntype[i] == _JOINT_TRANSMISSION_TYPE
            ]
            actuator_joint_ids = [
                self._model.actuator_trnid[i, 0] for i in actuator_indices
            ]
            self._joint_ids = list(actuator_joint_ids)
            self._ctrl_indices = actuator_indices
            self._qpos_indices = [
                int(self._model.jnt_qposadr[jid]) for jid in actuator_joint_ids
            ]
            self._qvel_indices = [
                int(self._model.jnt_dofadr[jid]) for jid in actuator_joint_ids
            ]
            self._joint_names = [
                mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                for jid in actuator_joint_ids
            ]
            self._num_dof = len(actuator_indices)
            self._ready = True
        except Exception as exc:
            logger.exception(f"Failed to setup MujocoArticulation")
            return False

        logger.info(f"MujocoArticulation ready — dof={self._num_dof}")
        return True

    def get_joint_names(self) -> list[str]:
        """Return joint names in the same order as get_joint_states() output."""
        return list(self._joint_names)

    def get_joint_limits(self) -> Optional[tuple[list[float], list[float]]]:
        """Return (lower, upper) joint limits in get_joint_states() order.
        Unlimited joints report ±1e6 (JSON-safe stand-in for unbounded)."""
        if not self._ready:
            return None
        try:
            lower, upper = [], []
            for jid in self._joint_ids:
                if self._model.jnt_limited[jid]:
                    lo, hi = self._model.jnt_range[jid]
                    lower.append(float(lo))
                    upper.append(float(hi))
                else:
                    lower.append(-1e6)
                    upper.append(1e6)
            return lower, upper
        except Exception as exc:
            logger.warning(f"Could not read joint limits: {exc}")
            return None

    def teardown(self) -> None:
        """Reset articulation state."""
        self._ready = False
        self._num_dof = 0

    def get_joint_states(self) -> Optional[tuple[list[float], list[float]]]:
        """Read current joint positions and velocities."""
        if not self._ready:
            return None

        try:
            positions = [float(self._data.qpos[i]) for i in self._qpos_indices]
            velocities = [float(self._data.qvel[i]) for i in self._qvel_indices]
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
            for ctrl_idx, pos in zip(self._ctrl_indices, positions):
                self._data.ctrl[ctrl_idx] = pos
            return True
        except Exception as exc:
            logger.warning(f"Could not apply joint command: {exc}")
            return False

    @property
    def num_dof(self) -> int:
        """Number of actuated degrees of freedom."""
        return self._num_dof

    @property
    def is_ready(self) -> bool:
        """True when joint indices have been resolved."""
        return self._ready
