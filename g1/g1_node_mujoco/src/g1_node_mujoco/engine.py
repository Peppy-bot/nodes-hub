"""MuJoCo G1 engine: loads the model, steps physics, exposes a telemetry snapshot.

This is the sim stand-in for the robot's onboard controller. Until a pretrained
`unitree_rl_gym` policy is wired in (the documented seam in `step`), it holds the
G1 upright with a model-agnostic joint PD (torque injected via `qfrc_applied`, so
it works regardless of the menagerie actuator types) and previews base velocity
kinematically. That makes the node runnable and its telemetry live without the
policy; swapping the PD hold for the policy is what makes it actually walk.

Thread model mirrors the openarm sim engine: `step()` runs on a physics thread;
`snapshot()` is read from the node's asyncio loop. A lock guards the exchange.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import mujoco
import numpy as np

# hg LowState.motor_state capacity; telemetry arrays are padded to this.
MOTOR_COUNT = 35


@dataclass
class Snapshot:
    fsm_id: int = 1
    tick: int = 0
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    gyroscope: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accelerometer: list[float] = field(default_factory=lambda: [0.0, 0.0, 9.81])
    rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    q: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    dq: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    tau_est: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)


def _load_model() -> mujoco.MjModel:
    """Load the G1 MJCF from robot_descriptions (downloads menagerie on first use)."""
    from robot_descriptions import g1_mj_description

    return mujoco.MjModel.from_xml_path(g1_mj_description.MJCF_PATH)


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> list[float]:
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


class G1MujocoEngine:
    """Owns the MuJoCo model/data and the standing controller."""

    def __init__(self, kp: float = 150.0, kd: float = 5.0) -> None:
        self._model = _load_model()
        self._data = mujoco.MjData(self._model)
        self._kp = kp
        self._kd = kd
        self._lock = threading.Lock()
        self._tick = 0
        self._fsm_id = 1
        self._cmd = np.zeros(3)  # vx, vy, vyaw (body frame)
        self._pd_enabled = True

        # A free base joint owns qpos[0:7] / qvel[0:6]; the actuated hinges follow.
        self._has_free_base = (
            self._model.njnt > 0 and self._model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )
        self._reset_home()
        self._joint_qadr, self._joint_dadr = self._actuated_joint_addrs()
        self._home_targets = np.array(
            [self._data.qpos[a] for a in self._joint_qadr]
        )

    def _reset_home(self) -> None:
        # Prefer a named "home" keyframe; otherwise the model's default qpos0.
        key = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, key)
        else:
            mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._base_pos = np.array(self._data.qpos[0:3]) if self._has_free_base else np.zeros(3)
        self._base_quat = (
            np.array(self._data.qpos[3:7]) if self._has_free_base else np.array([1.0, 0, 0, 0])
        )

    def _actuated_joint_addrs(self) -> tuple[list[int], list[int]]:
        # Hinge/slide joints (everything but the free base), in model order.
        qadr, dadr = [], []
        for j in range(self._model.njnt):
            jtype = self._model.jnt_type[j]
            if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qadr.append(self._model.jnt_qposadr[j])
                dadr.append(self._model.jnt_dofadr[j])
        return qadr, dadr

    # --- commands -------------------------------------------------------------
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        with self._lock:
            self._cmd = np.array([vx, vy, vyaw])

    def set_posture(self, fsm_id: int, pd_enabled: bool) -> None:
        with self._lock:
            self._fsm_id = fsm_id
            self._pd_enabled = pd_enabled

    # --- physics --------------------------------------------------------------
    def step(self) -> None:
        with self._lock:
            cmd = self._cmd.copy()
            pd_enabled = self._pd_enabled

        # Joint PD hold (model-agnostic: inject torque directly). This is the
        # seam a pretrained unitree_rl_gym policy replaces to actually walk.
        if pd_enabled:
            q = np.array([self._data.qpos[a] for a in self._joint_qadr])
            dq = np.array([self._data.qvel[a] for a in self._joint_dadr])
            tau = self._kp * (self._home_targets - q) - self._kd * dq
            for dadr, t in zip(self._joint_dadr, tau):
                self._data.qfrc_applied[dadr] = t

        # Kinematic base preview: glide the held base by the commanded velocity so
        # odometry and the viewer reflect the command until the policy drives it.
        if self._has_free_base:
            dt = self._model.opt.timestep
            yaw = _quat_to_rpy(*self._base_quat)[2] + cmd[2] * dt
            self._base_pos[0] += (cmd[0] * math.cos(yaw) - cmd[1] * math.sin(yaw)) * dt
            self._base_pos[1] += (cmd[0] * math.sin(yaw) + cmd[1] * math.cos(yaw)) * dt
            self._base_quat = np.array(
                [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            )
            self._data.qpos[0:3] = self._base_pos
            self._data.qpos[3:7] = self._base_quat
            self._data.qvel[0:3] = [cmd[0], cmd[1], 0.0]
            self._data.qvel[3:6] = [0.0, 0.0, cmd[2]]

        mujoco.mj_step(self._model, self._data)
        self._tick += 1

    # --- telemetry ------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        d = self._data
        n = min(len(self._joint_qadr), MOTOR_COUNT)
        q = [0.0] * MOTOR_COUNT
        dq = [0.0] * MOTOR_COUNT
        tau = [0.0] * MOTOR_COUNT
        for i in range(n):
            q[i] = float(d.qpos[self._joint_qadr[i]])
            dq[i] = float(d.qvel[self._joint_dadr[i]])
            tau[i] = float(d.qfrc_applied[self._joint_dadr[i]])

        if self._has_free_base:
            quat = [float(v) for v in d.qpos[3:7]]
            pos = [float(v) for v in d.qpos[0:3]]
            lin = [float(v) for v in d.qvel[0:3]]
            ang = [float(v) for v in d.qvel[3:6]]
        else:
            quat, pos, lin, ang = [1.0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

        return Snapshot(
            fsm_id=self._fsm_id,
            tick=self._tick,
            quaternion=quat,
            gyroscope=ang,
            accelerometer=[0.0, 0.0, 9.81],
            rpy=_quat_to_rpy(*quat),
            position=pos,
            orientation=quat,
            linear_velocity=lin,
            angular_velocity=ang,
            q=q,
            dq=dq,
            tau_est=tau,
        )

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data
