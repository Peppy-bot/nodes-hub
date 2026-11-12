from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MujocoArticulation:
    """Reads the state of every hinge and slide joint of a MuJoCo model: the
    joints the robots' limbs move, an actuator driving them or not, as a
    finger an equality couples to the one its actuator drives."""

    def __init__(self, model, data) -> None:
        self._model = model
        self._data = data
        self._qpos_indices: list[int] = []
        self._qvel_indices: list[int] = []
        self._joint_names: list[str] = []
        self._ready: bool = False

    def setup(self) -> bool:
        """Resolve each hinge and slide joint's state indices and name."""
        try:
            import mujoco  # pylint: disable=E0401,C0415

            moving = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
            joint_ids = [
                jid for jid in range(self._model.njnt) if int(self._model.jnt_type[jid]) in moving
            ]
            self._qpos_indices = [int(self._model.jnt_qposadr[jid]) for jid in joint_ids]
            self._qvel_indices = [int(self._model.jnt_dofadr[jid]) for jid in joint_ids]
            self._joint_names = [
                mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "" for jid in joint_ids
            ]
            self._ready = True
        except Exception:  # pylint: disable=W0718
            logger.exception("Failed to setup MujocoArticulation")
            return False

        logger.info("MujocoArticulation ready: %d joints", len(self._joint_names))
        return True

    def get_joint_names(self) -> list[str]:
        """Return joint names in the same order as get_joint_states() output."""
        return list(self._joint_names)

    def teardown(self) -> None:
        """Reset articulation state."""
        self._ready = False

    def get_joint_states(self) -> Optional[tuple[list[float], list[float]]]:
        """Read current joint positions and velocities."""
        if not self._ready:
            return None

        try:
            positions = [float(self._data.qpos[i]) for i in self._qpos_indices]
            velocities = [float(self._data.qvel[i]) for i in self._qvel_indices]
            return positions, velocities
        except Exception as exc:  # pylint: disable=W0718
            logger.warning(f"Could not read joint states: {exc}")
            return None
