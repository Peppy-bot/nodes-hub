"""The only part of the brain that talks to the backbone. One call per
limb_motion move, each returning a plain outcome, and the goals in flight
remembered so a stop can cancel them whatever backend asked for them.

The backbone plans and governs every move, so nothing here checks
collisions or limits: a target the backbone cannot reach comes back as a
refused goal, which the caller reports in its own result.
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


class Robot:
    def __init__(self, node_runner) -> None:
        self._node_runner = node_runner
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
        try:
            handle = await move_arm.ActionHandle.fire_goal(
                self._node_runner,
                move_arm.bound_producer(self._node_runner),
                request,
                GOAL_TIMEOUT_S,
                QoSProfile.Standard,
            )
        except Exception as error:
            return MoveResult(False, f"move_arm could not be sent: {error!r}")
        if not handle.accepted:
            return MoveResult(False, handle.reason or "move_arm was refused by the backbone")
        self._live.add(handle)
        try:
            result = await handle.get_result(RESULT_TIMEOUT_S)
        except Exception as error:
            return MoveResult(False, f"move_arm gave no result: {error!r}")
        finally:
            self._live.discard(handle)
        if result.status == move_arm.ResultStatus.COMPLETED and result.data is not None:
            return MoveResult(
                result.data.success,
                result.data.message,
                _vec3(result.data.final_position),
                _quat(result.data.final_orientation),
            )
        if result.status == move_arm.ResultStatus.CANCELLED:
            message = result.data.message if result.data is not None else ""
            return MoveResult(False, message or "move_arm was cancelled", _vec3(result.data.final_position) if result.data else None)
        return MoveResult(False, f"move_arm ended as {result.status.name.lower()}")

    async def set_gripper(self, gripper: str, opening: float, *, max_effort: float = 0.0) -> GripResult:
        """Drives one gripper to an opening fraction, 0 closed to 1 open,
        and waits for it to end."""
        request = move_gripper.GoalRequest(gripper_name=gripper, opening=opening, max_effort=max_effort)
        try:
            handle = await move_gripper.ActionHandle.fire_goal(
                self._node_runner,
                move_gripper.bound_producer(self._node_runner),
                request,
                GOAL_TIMEOUT_S,
                QoSProfile.Standard,
            )
        except Exception as error:
            return GripResult(False, f"move_gripper could not be sent: {error!r}")
        if not handle.accepted:
            return GripResult(False, handle.reason or "move_gripper was refused by the backbone")
        self._live.add(handle)
        try:
            result = await handle.get_result(RESULT_TIMEOUT_S)
        except Exception as error:
            return GripResult(False, f"move_gripper gave no result: {error!r}")
        finally:
            self._live.discard(handle)
        if result.status == move_gripper.ResultStatus.COMPLETED and result.data is not None:
            return GripResult(result.data.success, result.data.message, result.data.final_opening)
        if result.status == move_gripper.ResultStatus.CANCELLED:
            message = result.data.message if result.data is not None else ""
            return GripResult(False, message or "move_gripper was cancelled", result.data.final_opening if result.data else None)
        return GripResult(False, f"move_gripper ended as {result.status.name.lower()}")

    async def stop(self) -> None:
        """Cancels every move in flight. The waiting calls return with the
        cancelled outcome once the backbone has brought the limb to rest."""
        for handle in list(self._live):
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
