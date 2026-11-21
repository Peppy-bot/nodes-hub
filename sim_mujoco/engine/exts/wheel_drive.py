"""The wheels of a robot's base, each run as its driver runs it in velocity
mode: a loop on the wheel's rate, its target the rate of the latest
setpoint, of proportional and integral action, the two actuators of its
MJCF on the wheel's joint. A `<velocity>` actuator, its `kv` the loop's
proportional gain, pushes against the rate's error; an `<intvelocity>`
actuator, its `kp` the loop's integral gain, has an activation that
integrates the target rate, the angle the target carries the wheel to, and
pushes against the wheel's angle from it. The joint's `actuatorfrcrange`
caps the two together at the motor's torque, and the integral is clamped at
what that cap holds, as a drive clamps its integral at its current limit: a
stalled wheel stores no more than the cap, and once let go runs on by no
more than the cap over the integral gain.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RateLoop:
    """One wheel's joint, where its angle sits, and the two actuators of its
    rate loop, with how far the integral actuator's activation may stand
    from the wheel's angle, below and above it (rad)."""

    joint: str
    qpos: int
    proportional: int
    integral: int
    integral_act: int
    windup: tuple[float, float]


def _part(model, actuator: int) -> str | None:
    """The part `actuator` plays in a wheel's rate loop: "proportional" for
    a velocity servo, whose force is its gain times its control less the
    joint's rate, "integral" for an integrated-velocity servo, whose force is
    its gain times its activation less the joint's angle, and None for any
    other."""
    import mujoco  # pylint: disable=E0401,C0415

    gain = float(model.actuator_gainprm[actuator][0])
    fixed_affine = (
        int(model.actuator_gaintype[actuator]) == int(mujoco.mjtGain.mjGAIN_FIXED)
        and int(model.actuator_biastype[actuator]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        and math.isfinite(gain)
        and gain > 0.0
    )
    if not fixed_affine:
        return None
    bias = [float(value) for value in model.actuator_biasprm[actuator][:3]]
    dynamics = int(model.actuator_dyntype[actuator])
    if dynamics == int(mujoco.mjtDyn.mjDYN_NONE) and bias == [0.0, 0.0, -gain]:
        return "proportional"
    if dynamics == int(mujoco.mjtDyn.mjDYN_INTEGRATOR) and bias == [0.0, -gain, 0.0]:
        return "integral"
    return None


class MujocoWheelDrive:
    """Drives the wheels of one robot's base, by their joints' names in the
    scene, left then right."""

    def __init__(self, model, data, wheels: Sequence[str]) -> None:
        self._model = model
        self._data = data
        self._wheels = list(wheels)
        self._loops: list[_RateLoop] = []

    def setup(self) -> None:
        """Resolves each wheel's rate loop: a hinge whose actuator force a
        range caps, driven through joint transmissions of gear 1 by one
        velocity servo and one integrated-velocity servo whose activation
        runs unclamped, as a wheel turns without end. Anything else raises
        RuntimeError naming the wheel and what its MJCF needs."""
        self._loops = [self._loop(wheel) for wheel in self._wheels]
        logger.info("MujocoWheelDrive ready: %s", ", ".join(self._wheels))

    def _loop(self, wheel: str) -> _RateLoop:
        import mujoco  # pylint: disable=E0401,C0415

        model = self._model
        where = f"wheel '{wheel}'"
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel)
        if joint < 0:
            raise RuntimeError(f"{where} is not a joint of the scene")
        if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise RuntimeError(f"{where} is not a hinge: a wheel turns on one")
        lowest, highest = (float(value) for value in model.jnt_actfrcrange[joint])
        if not model.jnt_actfrclimited[joint] or not (lowest < 0.0 < highest):
            raise RuntimeError(
                f"{where}: no range caps its actuator force; give the joint an actuatorfrcrange "
                "of the wheel motor's peak torque either way"
            )
        drivers = [
            actuator
            for actuator in range(model.nu)
            if int(model.actuator_trntype[actuator]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator, 0]) == joint
        ]
        parts = {_part(model, actuator): actuator for actuator in drivers}
        if len(drivers) != 2 or set(parts) != {"proportional", "integral"}:
            raise RuntimeError(
                f"{where}: {len(drivers)} actuators drive it; drive the wheel with one <velocity> "
                "actuator, its kv the loop's proportional gain, and one <intvelocity> actuator, "
                "its kp the loop's integral gain"
            )
        for actuator in drivers:
            gear = float(model.actuator_gear[actuator][0])
            if gear != 1.0:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                raise RuntimeError(
                    f"{where}: its actuator '{name}' has gear {gear:g}; a wheel's setpoints are its "
                    "rates, so give the actuator gear 1"
                )
        integral = parts["integral"]
        if model.actuator_actlimited[integral]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, integral)
            raise RuntimeError(
                f"{where}: its actuator '{name}' clamps its activation; a wheel turns without end, "
                "so give the <intvelocity> no actrange"
            )
        integral_gain = float(model.actuator_gainprm[integral][0])
        return _RateLoop(
            joint=wheel,
            qpos=int(model.jnt_qposadr[joint]),
            proportional=parts["proportional"],
            integral=integral,
            integral_act=int(model.actuator_actadr[integral]),
            windup=(lowest / integral_gain, highest / integral_gain),
        )

    def drive(self, rates: Sequence[float]) -> None:
        """Sets each wheel's target rate (rad/s), left then right, on both
        actuators of its loop."""
        for loop, rate in zip(self._loops, rates, strict=True):
            self._data.ctrl[loop.proportional] = rate
            self._data.ctrl[loop.integral] = rate

    def hold(self) -> None:
        """Holds each wheel where it stands: its target rate 0 and the
        integral of its rate's error 0, the integral actuator's activation
        set to the wheel's angle, so its loop holds the wheel there and
        pushes nothing."""
        for loop in self._loops:
            self._data.ctrl[loop.proportional] = 0.0
            self._data.ctrl[loop.integral] = 0.0
            self._data.act[loop.integral_act] = self._data.qpos[loop.qpos]

    def clamp_integrals(self) -> None:
        """Clamps the integral of each wheel's rate error at what its torque
        cap holds: the integral actuator's activation within the cap over the
        integral gain of the wheel's angle, either way."""
        for loop in self._loops:
            angle = float(self._data.qpos[loop.qpos])
            below, above = loop.windup
            act = float(self._data.act[loop.integral_act])
            self._data.act[loop.integral_act] = min(max(act, angle + below), angle + above)

    def teardown(self) -> None:
        self._loops = []
