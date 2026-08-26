from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MujocoActuatorCtrl:
    """Resolves MJCF actuator names to ctrl indices and writes targets into
    data.ctrl[]. Used by ActuatorCtrlBridge to translate raw set_ctrl messages
    into in-process MuJoCo writes before mj_step.

    When the config entry carries per-joint MIT gains (kp/kd), setup() rewrites
    each position actuator's gain/bias so the servo torque law matches the real
    motors: tau = kp*(q_des - q) + kd*(dq_des - dq). The dq_des feedforward is
    folded into ctrl as q_des + (kd/kp)*dq_des, which is algebraically exact
    for MuJoCo's affine position actuator.
    """

    def __init__(self, model, data, params: dict | None = None) -> None:
        self._model = model
        self._data = data
        self._params = params or {}
        self._name_to_id: dict[str, int] = {}
        self._kd_over_kp: dict[str, float] = {}
        self._ready: bool = False

    def setup(self) -> bool:
        """Build the name → ctrl-id map. Indexes by both actuator name and the
        joint name the actuator drives — the joint-name alias is the cross-engine
        canonical key (matches Isaac dof_names), so components ship one payload
        for both engines.
        """
        try:
            import mujoco  # pylint: disable=E0401

            name_to_id: dict[str, int] = {}
            joint_aliases = 0
            for i in range(self._model.nu):
                actuator_name = (
                    mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    or ""
                )
                if actuator_name:
                    name_to_id[actuator_name] = i
                if int(self._model.actuator_trntype[i]) != int(
                    mujoco.mjtTrn.mjTRN_JOINT
                ):
                    continue
                trnid = int(self._model.actuator_trnid[i, 0])
                joint_name = (
                    mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
                    or ""
                )
                if joint_name and joint_name not in name_to_id:
                    name_to_id[joint_name] = i
                    joint_aliases += 1
            self._name_to_id = name_to_id
            self._default_forcerange = self._model.actuator_forcerange.copy()
            self._apply_gains()
            self._apply_gravity_compensation()
            self._ready = True
        except Exception as exc:
            logger.exception(f"Failed to setup MujocoActuatorCtrl")
            return False

        logger.info(
            f"MujocoActuatorCtrl ready — {self._model.nu} actuator(s) resolved"
            f" ({joint_aliases} joint-name aliases,"
            f" {len(self._kd_over_kp)} gain override(s))"
        )
        return True

    def _apply_gains(self) -> None:
        """Overwrite actuator gain/bias from config so the sim servo runs the
        real driver's MIT gains. Position actuator torque is
        gainprm[0]*ctrl + biasprm[1]*q + biasprm[2]*dq, so kp/kd map to
        (kp, -kp, -kd). Joints without configured gains keep the MJCF values.
        """
        import mujoco  # pylint: disable=E0401
        import numpy as np

        joint_names = self._params.get("joint_names") or []
        kps = self._params.get("kp") or []
        kds = self._params.get("kd") or []
        if not joint_names and not kps and not kds:
            return  # no gains configured
        if not (len(joint_names) == len(kps) == len(kds)):
            raise ValueError(
                f"gain config mismatch: {len(joint_names)} joint_names, "
                f"{len(kps)} kp, {len(kds)} kd"
            )
        # Per-dof inertia from the sim's own mass matrix (home config). The
        # real gearbox/motor adds damping the sim plant lacks; without it the
        # real driver's gains ring badly (~0.14 damping ratio). Raise the
        # servo damping to critical — the dq_des feedforward cancels it along
        # the trajectory, so tracking is unaffected while deviations damp.
        mujoco.mj_forward(self._model, self._data)
        full_m = np.zeros((self._model.nv, self._model.nv))
        # MuJoCo >= 3.10 binding: qM went CSR, so mj_fullM takes mjData.
        # Older bindings (the current base image ships 3.8.1 despite its
        # 3.10.0 tag — requirements.mujoco.txt was never bumped) take qM.
        try:
            mujoco.mj_fullM(self._model, self._data, full_m)
        except TypeError:
            mujoco.mj_fullM(self._model, full_m, self._data.qM)
        for name, kp, kd in zip(joint_names, kps, kds):
            ctrl_id = self._name_to_id.get(name)
            if ctrl_id is None:
                logger.warning(f"gain config: unknown joint '{name}' — skipped")
                continue
            jid = int(self._model.actuator_trnid[ctrl_id, 0])
            dof = int(self._model.jnt_dofadr[jid])
            inertia = float(full_m[dof, dof])
            kv = max(float(kd), 2.0 * (float(kp) * inertia) ** 0.5)
            self._model.actuator_gainprm[ctrl_id][0] = float(kp)
            self._model.actuator_biasprm[ctrl_id][1] = -float(kp)
            self._model.actuator_biasprm[ctrl_id][2] = -kv
            self._kd_over_kp[name] = kv / float(kp)

    def _apply_gravity_compensation(self) -> None:
        """Mirror the real driver's in-process gravity feedforward: MuJoCo's
        body_gravcomp applies an exact counter-gravity force per body, every
        step, inside the engine. Enabled for the whole robot subtree (the real
        arm and gripper drivers both feedforward). Coriolis is intentionally
        not compensated — negligible at teleop speeds."""
        if not self._params.get("gravity_compensation"):
            return
        import mujoco  # pylint: disable=E0401

        compensated = 0
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if name.startswith("openarm_"):
                self._model.body_gravcomp[i] = 1.0
                compensated += 1
        logger.info(f"gravity compensation enabled on {compensated} robot bodies")

    def teardown(self) -> None:
        self._ready = False
        self._name_to_id = {}
        self._kd_over_kp = {}

    def require_force_limited(self, names: list[str]) -> None:
        """Refuse fingers whose actuators cannot be force-capped: MuJoCo
        honors a runtime forcerange write only for actuators compiled with
        forcelimited set, so an uncheckable cap would silently do nothing."""
        missing = sorted(n for n in names if n not in self._name_to_id)
        if missing:
            raise RuntimeError(f"force-capped actuators not in the model: {missing}")
        unlimited = sorted(
            n
            for n in names
            if not self._model.actuator_forcelimited[self._name_to_id[n]]
        )
        if unlimited:
            raise RuntimeError(
                f'actuators lack forcelimited="true", so a force cap would '
                f"silently do nothing: {unlimited}"
            )

    def set_force_limit(self, names: list[str], limit: float) -> bool:
        """Cap the named actuators' force range at +/-limit (engine units),
        True once written. A non-positive limit restores each actuator's model
        default, so an unset wire effort leaves the MJCF behaviour
        untouched."""
        if not self._ready:
            return False
        for name in names:
            ctrl_id = self._name_to_id.get(name)
            if ctrl_id is None:
                logger.warning(f"unknown actuator '{name}' for force limit; dropped")
                continue
            if limit > 0:
                self._model.actuator_forcerange[ctrl_id] = [-limit, limit]
            else:
                self._model.actuator_forcerange[ctrl_id] = self._default_forcerange[ctrl_id]
        return True

    def write_targets(self, actuator_values: dict, velocity_values: dict | None = None) -> int:
        """Write each {name: value} pair into data.ctrl[ctrl_id[name]]. When a
        velocity setpoint is supplied for a gain-configured joint, it is folded
        into ctrl as q_des + (kd/kp)*dq_des. Returns the count of values
        applied. Unknown actuator names are logged and dropped so a single bad
        entry does not stop the rest of the batch."""
        if not self._ready:
            return 0
        if not isinstance(actuator_values, dict):
            logger.warning(
                f"actuator_values must be a dict, got {type(actuator_values).__name__}"
            )
            return 0
        velocities = velocity_values if isinstance(velocity_values, dict) else {}
        applied = 0
        for name, value in actuator_values.items():
            ctrl_id = self._name_to_id.get(name)
            if ctrl_id is None:
                logger.warning(f"unknown actuator '{name}' — dropped")
                continue
            try:
                ctrl = float(value)
                ratio = self._kd_over_kp.get(name)
                dq_des = velocities.get(name)
                if ratio is not None and dq_des is not None:
                    ctrl += ratio * float(dq_des)
                self._data.ctrl[ctrl_id] = ctrl
                applied += 1
            except Exception as exc:
                logger.warning(f"failed to write ctrl[{ctrl_id}] for '{name}': {exc}")
        return applied

    @property
    def is_ready(self) -> bool:
        return self._ready
