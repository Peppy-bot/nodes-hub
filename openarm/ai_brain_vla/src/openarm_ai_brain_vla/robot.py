"""The only part of the brain that talks to the backbone. One call per
limb_motion move, each returning a plain outcome, and the goals in flight
remembered so a stop can cancel them whatever backend asked for them.

The backbone plans and governs every move, so nothing here checks
collisions or limits: a target the backbone cannot reach comes back as a
refused goal, which the caller reports in its own result. A move that
gives no result within `result_timeout_s` is cancelled and reported as
having given none, so the arm does not keep going unwatched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from peppygen import QoSProfile
from peppygen.consumed_actions.limb_motion import move_arm, move_gripper

from .ports import Quat, Vec3

GOAL_TIMEOUT_S = 5.0
RESULT_TIMEOUT_S = 120.0
CANCEL_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class MoveResult:
    success: bool
    message: str
    final_position: Optional[Vec3] = None
    final_orientation: Optional[Quat] = None


@dataclass(frozen=True)
class GripResult:
    success: bool
    message: str
    final_opening: Optional[float] = None


@dataclass(frozen=True)
class Ended:
    """How one limb_motion goal ended: success and message as the
    contract's result carries them, a refusal, a cancel and a lost result
    included, and the result's data when the backbone sent any."""

    success: bool
    message: str
    data: object = None


class Robot:
    def __init__(self, node_runner, *, result_timeout_s: float = RESULT_TIMEOUT_S) -> None:
        self._node_runner = node_runner
        self._result_timeout_s = result_timeout_s
        self._live: set = set()

    async def move_arm(
        self,
        arm: str,
        position: Vec3,
        orientation: Quat,
        *,
        duration_s: float = 0.0,
        plan_position_tolerance_m: float = 0.0,
        plan_orientation_tolerance_rad: float = 0.0,
    ) -> MoveResult:
        """Plans and runs one grasp-point move of `arm`, world frame, and
        waits for it to end. Zero tolerances and duration take the
        backbone's defaults."""
        request = move_arm.GoalRequest(
            arm_name=arm,
            position=list(position),
            orientation=list(orientation),
            duration_s=duration_s,
            plan_position_tolerance_m=plan_position_tolerance_m,
            plan_orientation_tolerance_rad=plan_orientation_tolerance_rad,
        )
        ended = await self._run(move_arm, request)
        data = ended.data
        return MoveResult(
            ended.success,
            ended.message,
            _vec3(data.final_position) if data is not None else None,
            _quat(data.final_orientation) if data is not None else None,
        )

    async def set_gripper(self, gripper: str, opening: float, *, max_effort: float = 0.0) -> GripResult:
        """Drives one gripper to an opening fraction, 0 closed to 1 open,
        and waits for it to end."""
        request = move_gripper.GoalRequest(gripper_name=gripper, opening=opening, max_effort=max_effort)
        ended = await self._run(move_gripper, request)
        data = ended.data
        return GripResult(ended.success, ended.message, float(data.final_opening) if data is not None else None)

    @property
    def moves_in_flight(self) -> int:
        """How many goals the backbone has accepted and not yet ended."""
        return len(self._live)

    async def stop(self) -> None:
        """Cancels every move in flight. The waiting calls return with the
        cancelled outcome once the backbone has brought the limb to rest."""
        for handle in list(self._live):
            await _cancel(handle)

    async def _run(self, module, request) -> Ended:
        """Fires one goal of `module` at the bound producer, keeps its
        handle while it runs, and reads how it ended."""
        name = module.TARGET_ACTION_NAME
        try:
            handle = await module.ActionHandle.fire_goal(
                self._node_runner, module.bound_producer(self._node_runner), request, GOAL_TIMEOUT_S, QoSProfile.Standard
            )
        except Exception as error:
            return Ended(False, f"{name} could not be sent: {error!r}")
        if not handle.accepted:
            return Ended(False, handle.reason or f"{name} was refused by the backbone")
        self._live.add(handle)
        try:
            result = await handle.get_result(self._result_timeout_s)
        except Exception as error:
            await _cancel(handle)
            return Ended(False, f"{name} gave no result: {error!r}")
        finally:
            self._live.discard(handle)
        data = result.data
        if result.status == module.ResultStatus.COMPLETED and data is not None:
            return Ended(data.success, data.message, data)
        if result.status == module.ResultStatus.CANCELLED:
            message = data.message if data is not None else ""
            return Ended(False, message or f"{name} was cancelled", data)
        return Ended(False, f"{name} ended as {result.status.name.lower()}")


async def _cancel(handle) -> None:
    try:
        await handle.cancel_goal(CANCEL_TIMEOUT_S)
    except Exception:
        pass


def _vec3(values) -> Optional[Vec3]:
    if values is None or len(values) != 3:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _quat(values) -> Optional[Quat]:
    if values is None or len(values) != 4:
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
