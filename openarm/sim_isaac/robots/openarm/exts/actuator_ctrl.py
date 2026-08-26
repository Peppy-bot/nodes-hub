from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_ARTICULATION_NAME = "peppy_actuator_ctrl"


class IsaacActuatorCtrl:
    """Resolves joint names to indices on a target articulation and writes
    position (and optional velocity) targets via the Articulation view. One
    instance per articulation (one gripper, one arm side, etc).

    When the config entry carries per-joint MIT gains (kp/kd) and torque caps,
    setup() applies them to the PhysX joint drives so the sim servo runs the
    real driver's torque law: tau = kp*(q_des - q) + kd*(dq_des - dq).
    """

    def __init__(self, prim_path: str, joint_names: list[str], params: dict | None = None) -> None:
        self._prim_path = prim_path
        self._joint_names = list(joint_names)
        self._params = params or {}
        self._view = None
        self._name_to_idx: dict[str, int] = {}
        self._ready: bool = False

    def setup(self) -> bool:
        """Initialise the Articulation, resolve joint name → index, and apply
        configured drive gains / torque caps."""
        if self._view is not None and self._ready:
            return True
        try:
            from isaacsim.core.prims import Articulation  # pylint: disable=E0401

            self._view = Articulation(
                prim_paths_expr=self._prim_path,
                name=_ARTICULATION_NAME,
            )
            self._view.initialize()
            dof_names = list(self._view.dof_names)
            self._name_to_idx = {
                n: i for i, n in enumerate(dof_names) if n in self._joint_names
            }
            missing = [n for n in self._joint_names if n not in self._name_to_idx]
            if missing:
                logger.warning(
                    f"IsaacActuatorCtrl: joints not found on '{self._prim_path}': {missing}"
                )
            # If joint_names was configured but NOTHING resolved, the bridge
            # would otherwise mark itself ready and silently drop every
            # command. Fail loud so misconfiguration surfaces at startup.
            if self._joint_names and not self._name_to_idx:
                logger.error(
                    f"IsaacActuatorCtrl: zero joints resolved on '{self._prim_path}'"
                    f" against dof_names={dof_names[:10]}{'…' if len(dof_names) > 10 else ''}"
                )
                self._view = None
                return False
            self._apply_gains()
            self._apply_gravity_compensation()
            # Post-gain ceilings, so a wire cap of 0 can restore them exactly.
            self._default_max_efforts = np.asarray(
                self._view.get_max_efforts()
            ).reshape(-1)
            self._ready = True
        except Exception as exc:
            logger.error(
                f"Failed to initialise IsaacActuatorCtrl at '{self._prim_path}': {exc}"
            )
            self._view = None
            return False

        logger.info(
            f"IsaacActuatorCtrl ready — prim='{self._prim_path}'"
            f" resolved={list(self._name_to_idx.keys())}"
        )
        return True

    def _apply_gains(self) -> None:
        """Apply per-joint drive stiffness/damping (MIT kp/kd) and torque caps
        from config. Joints without configured gains keep the USD drive values.
        """

        joint_names = self._params.get("joint_names") or self._joint_names
        kps = self._params.get("kp") or []
        kds = self._params.get("kd") or []
        efforts = self._params.get("max_efforts") or []
        if not kps and not kds:
            # No gains configured (the gainless finger controllers): the USD
            # drive values stay.
            return
        if not (len(joint_names) == len(kps) == len(kds)):
            raise ValueError(
                f"gain config mismatch: {len(joint_names)} joint_names, "
                f"{len(kps)} kp, {len(kds)} kd"
            )
        # Per-dof inertia from the articulation mass matrix (home config). The
        # real gearbox/motor adds damping the sim plant lacks; raise the drive
        # damping to critical. PhysX damping acts on (dq_target - dq) and we
        # stream dq_des as the velocity target, so tracking is unaffected
        # while deviations damp.
        diag_inertia = None
        try:
            mm = self._view.get_mass_matrices()
            diag_inertia = mm[0].diagonal()
        except Exception as exc:
            logger.warning(f"mass matrix unavailable ({exc}) — using configured kd")

        indices, kp_list, kd_list, effort_list = [], [], [], []
        for i, name in enumerate(joint_names):
            idx = self._name_to_idx.get(name)
            if idx is None:
                logger.warning(f"gain config: unknown joint '{name}' — skipped")
                continue
            kp = float(kps[i])
            kd = float(kds[i])
            if diag_inertia is not None:
                kd = max(kd, 2.0 * (kp * float(diag_inertia[idx])) ** 0.5)
            indices.append(idx)
            kp_list.append(kp)
            kd_list.append(kd)
            if i < len(efforts):
                effort_list.append(float(efforts[i]))
        if not indices:
            return
        joint_indices = np.array(indices)
        self._view.set_gains(
            kps=np.array([kp_list]),
            kds=np.array([kd_list]),
            joint_indices=joint_indices,
        )
        if len(effort_list) == len(indices):
            self._view.set_max_efforts(
                np.array([effort_list]), joint_indices=joint_indices
            )
        logger.info(
            f"IsaacActuatorCtrl: applied MIT gains to {len(indices)} joint(s)"
        )

    def _apply_gravity_compensation(self) -> None:
        """Mirror the real driver's in-process gravity feedforward by disabling
        gravity on every rigid-body link of the robot — PhysX then behaves as
        if an exact counter-gravity force were applied each step. Enabled for
        the whole robot subtree (the real arm and gripper drivers both
        feedforward). Coriolis is intentionally not compensated."""
        if not self._params.get("gravity_compensation"):
            return
        import omni.usd  # pylint: disable=E0401
        from pxr import PhysxSchema, Usd, UsdPhysics  # pylint: disable=E0401

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._prim_path)
        if not root.IsValid():
            logger.warning(
                f"gravity compensation: prim '{self._prim_path}' not found — skipped"
            )
            return
        compensated = 0
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateDisableGravityAttr().Set(True)
                compensated += 1
        logger.info(f"gravity compensation enabled on {compensated} robot links")

    def set_force_limit(self, names: list[str], limit: float) -> bool:
        """Cap the named joints' drive effort at `limit` (engine units),
        True once written. A non-positive limit restores each joint's
        setup-time ceiling, so an unset wire effort leaves the configured
        drive untouched."""
        if not self._ready or self._view is None:
            return False
        indices = [self._name_to_idx[n] for n in names if n in self._name_to_idx]
        if len(indices) != len(names):
            unknown = sorted(set(names) - set(self._name_to_idx))
            logger.warning(f"force limit for unknown joints {unknown} dropped")
        if not indices:
            return False
        if limit > 0:
            efforts = [float(limit)] * len(indices)
        else:
            efforts = [float(self._default_max_efforts[i]) for i in indices]
        self._view.set_max_efforts(
            np.array([efforts]), joint_indices=np.array(indices)
        )
        return True

    def teardown(self) -> None:
        self._view = None
        self._ready = False

    def write_targets(self, actuator_values: dict, velocity_values: dict | None = None) -> int:
        """Write each {name: value} pair into the articulation's joint position
        targets, plus velocity targets when supplied. Unknown names and
        non-numeric values are dropped per-item so a single bad entry does not
        poison the whole batch."""
        if not self._ready or self._view is None:
            return 0
        if not isinstance(actuator_values, dict):
            logger.warning(
                f"actuator_values must be a dict, got {type(actuator_values).__name__}"
            )
            return 0
        velocities = velocity_values if isinstance(velocity_values, dict) else {}
        try:

            indices: list[int] = []
            pos: list[float] = []
            vel: list[float] = []
            for name, value in actuator_values.items():
                idx = self._name_to_idx.get(name)
                if idx is None:
                    logger.warning(
                        f"unknown actuator '{name}' on '{self._prim_path}' — dropped"
                    )
                    continue
                try:
                    coerced = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        f"non-numeric actuator '{name}'={value!r} on "
                        f"'{self._prim_path}' — dropped"
                    )
                    continue
                indices.append(idx)
                pos.append(coerced)
                try:
                    vel.append(float(velocities.get(name, 0.0)))
                except (TypeError, ValueError):
                    vel.append(0.0)
            if not indices:
                return 0
            joint_indices = np.array(indices)
            self._view.set_joint_position_targets(
                np.array([pos], dtype=np.float32), joint_indices=joint_indices
            )
            self._view.set_joint_velocity_targets(
                np.array([vel], dtype=np.float32), joint_indices=joint_indices
            )
            return len(indices)
        except Exception as exc:
            logger.warning(f"Failed to write targets on '{self._prim_path}': {exc}")
            return 0

    @property
    def is_ready(self) -> bool:
        return self._ready
