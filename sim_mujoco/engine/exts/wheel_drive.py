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
more than the cap over the integral gain. The two actuators are held to
sim_robot_core.rate_loop's rule, which the Isaac node's stage builder holds
the same MJCF to.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sim_robot_core.rate_loop import wheel_rate_loop

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
        try:
            loop = wheel_rate_loop(model, wheel, joint)
        except ValueError as refusal:
            raise RuntimeError(str(refusal)) from refusal
        return _RateLoop(
            joint=wheel,
            qpos=int(model.jnt_qposadr[joint]),
            proportional=loop.proportional,
            integral=loop.integral,
            integral_act=int(model.actuator_actadr[loop.integral]),
            windup=(lowest / loop.integral_gain, highest / loop.integral_gain),
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
