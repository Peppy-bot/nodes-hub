from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_JOINT_TRANSMISSION_TYPE = 0  # mjTRN_JOINT


class MujocoGripperSensor:
    """Reads finger joint positions and measured actuator forces. finger_joints
    is a list of MJCF joint names; each resolves to a qpos index and an actuator
    index for data.actuator_force (-1 if no actuator drives the joint).
    """

    def __init__(self, model, data, finger_joints: list[str]) -> None:
        self._model = model
        self._data = data
        self._finger_joints = finger_joints
        # List of (resolved_name, qpos_adr, ctrl_adr or -1)
        self._resolved: list[tuple[str, int, int]] = []
        self._ready: bool = False
        # One-shot guard — the state-read fallback fires every physics step
        # when broken; warn once instead of spamming at step rate.
        self._read_warning_logged: bool = False

    def setup(self) -> bool:
        """Resolve joint and actuator indices from the model."""
        try:
            import mujoco  # pylint: disable=E0401

            resolved = []
            for name in self._finger_joints:
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    logger.warning(
                        f"MujocoGripperSensor: joint '{name}' not found — skipped."
                    )
                    continue

                qpos_adr = int(self._model.jnt_qposadr[jid])

                ctrl_adr = -1
                for act_i in range(self._model.nu):
                    if (
                        self._model.actuator_trntype[act_i] == _JOINT_TRANSMISSION_TYPE
                        and self._model.actuator_trnid[act_i, 0] == jid
                    ):
                        ctrl_adr = act_i
                        break

                resolved.append((name, qpos_adr, ctrl_adr))

            if self._finger_joints and not resolved:
                logger.error(
                    "MujocoGripperSensor: none of the configured finger joints "
                    f"resolved against the MJCF model: {self._finger_joints}"
                )
                return False

            self._resolved = resolved
            self._ready = True
            self._read_warning_logged = False
        except Exception as exc:
            logger.exception(f"Failed to setup MujocoGripperSensor")
            return False

        logger.info(
            f"MujocoGripperSensor ready — "
            f"fingers={[name for name, _, _ in self._resolved]}"
        )
        return True

    def teardown(self) -> None:
        """Reset sensor state."""
        self._resolved = []
        self._ready = False
        self._read_warning_logged = False

    def get_gripper_state(self) -> Optional[dict]:
        """Return finger joint names, positions, and measured actuator forces.
        The applied_forces value is data.actuator_force in Newtons (the force the
        actuator actually produced, not the ctrl command target)."""
        if not self._ready:
            return None

        try:
            joint_names = [name for name, _, _ in self._resolved]
            positions = [float(self._data.qpos[qpos]) for _, qpos, _ in self._resolved]
            applied_forces = [
                float(self._data.actuator_force[ctrl]) if ctrl >= 0 else 0.0
                for _, _, ctrl in self._resolved
            ]
            return {
                "joint_names": joint_names,
                "positions": positions,
                "applied_forces": applied_forces,
            }
        except Exception as exc:
            if not self._read_warning_logged:
                logger.warning(f"Could not read gripper state (warning once): {exc}")
                self._read_warning_logged = True
            return None

    @property
    def is_ready(self) -> bool:
        """True when joint indices have been resolved."""
        return self._ready
