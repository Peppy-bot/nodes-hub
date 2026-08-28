"""Action admission and execution over the coordinator.

Admission (the goal decide callback) validates the request and claims the
limb's single-flight busy slot; execution installs a trajectory plan and
waits for the coordinator to stream it out. Cancelling freezes the limb where
the trajectory stands. Every terminal path releases the claimed slot, and no
exception may ever kill an action server: an unforeseen failure rejects the
goal and abandons anything admission installed.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

from control_core_py import minimum_jerk
from control_core_py.runtime import Latch, log
from so101_description.limbs import ARM_LIMB, GRIPPER_LIMB
from so101_description.limits import JointLimits
from so101_description.transforms import (
    PoseError,
    matrix_from_pose,
    relative_rotation_rad,
)
from so101_description.units import NUM_JOINTS

from so101_backbone.coordinator import ArmPlan, Coordinator, GripperPlan
from so101_backbone.params import Config

# Slack past a gripper move's nominal travel time before the goal fails: the
# plan is anchored on live state, so a follower that stops reporting stalls
# it, and a stalled goal must not hold the busy slot forever.
GRIPPER_TRAVEL_GRACE_S = 2.0

# Backoff after a failed goal delivery, so a persistently broken action
# transport logs once and retries at a human pace instead of spinning the
# event loop.
_GOAL_RETRY_S = 1.0

# Ceiling on a goal's requested duration: a move claims its limb's
# single-flight slot for the whole run, so an absurd request must not pin
# the arm until a manual cancel.
MAX_REQUESTED_DURATION_S = 120.0


class Rejection(ValueError):
    """A goal that must not run, with the operator-facing reason."""


@dataclass(frozen=True)
class PoseGoal:
    """An admitted move_arm goal whose IK solve runs in the drive task: the
    arm slot is already claimed, so the anchor cannot move under the solve."""

    position: tuple[float, ...]
    orientation: tuple[float, ...]
    duration_s: float


class PendingSlot:
    """One server's admitted-but-not-yet-driven goal. Per server, so a failed
    delivery on one action can only ever walk back its own admission."""

    def __init__(self) -> None:
        self.plan: ArmPlan | GripperPlan | PoseGoal | None = None

    def take(self) -> ArmPlan | GripperPlan | PoseGoal | None:
        plan = self.plan
        self.plan = None
        return plan


async def serve(
    node_runner, module, token, decide, drive, layer: ActionLayer, pending: PendingSlot, label: str
) -> None:
    """Expose one action and run accepted goals until shutdown. Nothing that
    happens per goal may end this loop; a raising admission or accept leaves
    the busy slot released and the server alive."""
    action = await module.ActionHandle.expose(node_runner)
    # create_task holds only a weak reference; dropping a drive task mid-await
    # would evict the goal from the registry without completing it.
    drive_tasks: set[asyncio.Task] = set()
    delivery_failing = Latch()
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_goal = asyncio.ensure_future(action.handle_goal_next_request(decide))
            await asyncio.wait([cancelled, next_goal], return_when=asyncio.FIRST_COMPLETED)
            if not next_goal.done():
                next_goal.cancel()
                return
            try:
                ctx = next_goal.result()
            except asyncio.CancelledError:
                # Cancellation raised inside the accept path, not ours: ours
                # only happens on the shutdown branch above.
                log(f"{label} goal delivery cancelled mid-accept")
                layer.abandon(pending)
                continue
            except Exception as e:
                # The accept round-trip failed after decide may have claimed
                # the slot and installed a plan; both must be walked back or
                # teleop stays ignored forever.
                delivery_failing.trip(f"{label} goal delivery failing: {e!r}")
                layer.abandon(pending)
                await asyncio.sleep(_GOAL_RETRY_S)
                continue
            delivery_failing.clear()
            if ctx is None:
                log(f"{label} action handler closed")
                return
            task = asyncio.create_task(drive(ctx))
            drive_tasks.add(task)
            task.add_done_callback(drive_tasks.discard)
    finally:
        cancelled.cancel()
        # Every exit from the loop above can leave an admission claimed: a
        # decide that landed just before shutdown, or a closed handler
        # returning mid-round-trip. Unwalked, the limb stays claimed and
        # teleop is ignored for good. Idempotent, so the paths that walk it
        # back and carry on stay correct.
        layer.abandon(pending)
        # A goal in flight at shutdown waits on a plan the stopped control
        # loop will never finish; its finally releases the claimed slot.
        for task in drive_tasks:
            task.cancel()
        await asyncio.gather(*drive_tasks, return_exceptions=True)


class ActionLayer:
    def __init__(self, coordinator: Coordinator, config: Config, limits: JointLimits):
        self._coordinator = coordinator
        self._config = config
        self._limits = limits
        self._effort_unsupported = Latch()

    # -------------------------------------------------------------- admission

    def _admit_arm_move(self, target: tuple[float, ...], duration_s: float) -> ArmPlan:
        """Claim the arm's busy slot and install the trajectory, or raise
        Rejection. The claim happens last so a rejected goal never leaves the
        slot held."""
        if len(target) != NUM_JOINTS or not all(math.isfinite(p) for p in target):
            raise Rejection(f"target must be {NUM_JOINTS} finite joint positions")
        if not self._limits.contains(target):
            raise Rejection("target exceeds a joint position limit")
        _require_duration(duration_s)
        anchor = self._coordinator.arm_anchor()
        if anchor is None:
            raise Rejection("no fresh follower state to anchor the move from")
        if not self._coordinator.try_claim_arm():
            raise Rejection("another arm goal is in flight")
        try:
            profile = minimum_jerk.plan(
                anchor, target, duration_s, self._config.max_joint_velocity_rad_s
            )
            return self._coordinator.adopt_arm_plan(profile)
        except BaseException:
            # A failure between claim and adopt must not hold the slot.
            self._coordinator.release_arm()
            raise

    def _admit_gripper_move(self, opening: float, max_effort: float) -> GripperPlan:
        if not (math.isfinite(opening) and 0.0 <= opening <= 1.0):
            raise Rejection("opening must be finite and within 0..1")
        if not math.isfinite(max_effort) or max_effort < 0.0:
            raise Rejection("max_effort must be finite and non-negative")
        if max_effort > 0.0:
            # Same disposition as the streamed path: legal contract intent
            # this hardware cannot honor (the ceiling is fixed in firmware),
            # so the opening runs and the preference is dropped, not refused.
            self._effort_unsupported.trip(
                "move_gripper max_effort ignored: gripper ceiling is fixed in firmware"
            )
        else:
            self._effort_unsupported.clear()
        anchor = self._coordinator.gripper_anchor()
        if anchor is None:
            raise Rejection("no fresh gripper state to anchor the move from")
        if not self._coordinator.try_claim_gripper():
            raise Rejection("another gripper goal is in flight")
        try:
            timeout_s = (
                abs(opening - anchor) / self._config.max_gripper_rate_frac_s
                + GRIPPER_TRAVEL_GRACE_S
            )
            return self._coordinator.adopt_gripper_plan(opening, timeout_s)
        except BaseException:
            self._coordinator.release_gripper()
            raise

    def _admit_pose_move(self, position, orientation, duration_s: float) -> PoseGoal:
        """Validate and claim the arm; the verified IK solve runs in the
        drive task so a hard pose never stalls the event loop inside the
        goal decide callback, failing the accepted goal when no solution
        exists."""
        try:
            matrix_from_pose(position, orientation)
        except PoseError as e:
            raise Rejection(str(e)) from e
        _require_duration(duration_s)
        if self._coordinator.arm_anchor() is None:
            raise Rejection("no fresh follower state to anchor the move from")
        if not self._coordinator.try_claim_arm():
            raise Rejection("another arm goal is in flight")
        return PoseGoal(
            position=tuple(position), orientation=tuple(orientation), duration_s=duration_s
        )

    def _admit_posture(self, target: tuple[float, ...], duration_s: float) -> ArmPlan:
        return self._admit_arm_move(target, duration_s)

    # ----------------------------------------------------- decide constructors

    def decide_arm_joints(self, module, pending: PendingSlot):
        def decide(request):
            def admit():
                _require_arm_name(request.data.arm_name)
                return self._admit_arm_move(
                    tuple(request.data.joint_positions), request.data.duration_s
                )

            return self._decide(module, admit, pending)

        return decide

    def decide_posture(self, module, target: tuple[float, ...], pending: PendingSlot):
        def decide(request):
            return self._decide(
                module, lambda: self._admit_posture(target, request.data.duration_s), pending
            )

        return decide

    def decide_pose(self, module, pending: PendingSlot):
        def decide(request):
            def admit():
                _require_arm_name(request.data.arm_name)
                return self._admit_pose_move(
                    tuple(request.data.position),
                    tuple(request.data.orientation),
                    request.data.duration_s,
                )

            return self._decide(module, admit, pending)

        return decide

    def decide_gripper(self, module, pending: PendingSlot):
        def decide(request):
            def admit():
                _require_gripper_name(request.data.gripper_name)
                return self._admit_gripper_move(
                    request.data.opening, request.data.max_effort
                )

            return self._decide(module, admit, pending)

        return decide

    def _decide(self, module, admit, pending: PendingSlot):
        try:
            plan = admit()
        except Rejection as e:
            return module.GoalDecision.reject(str(e))
        except Exception as e:
            # No admission failure may escape into the generated server loop.
            log(f"goal admission failed unexpectedly: {e!r}")
            return module.GoalDecision.reject(f"admission failed: {e!r}")
        pending.plan = plan
        return module.GoalDecision.accept()

    def abandon(self, pending: PendingSlot) -> None:
        """Walk back an admission whose goal never materialized: abort any
        installed plan and release the limb it claimed."""
        plan = pending.take()
        if plan is None:
            return
        if isinstance(plan, ArmPlan):
            self._coordinator.abort_arm_plan(plan)
            self._coordinator.release_arm()
        elif isinstance(plan, PoseGoal):
            # Nothing installed yet: the solve had not run. Only the claim
            # is walked back.
            self._coordinator.release_arm()
        else:
            self._coordinator.abort_gripper_plan(plan)
            self._coordinator.release_gripper()

    # -------------------------------------------------------------- execution

    async def _finish_plan(
        self, ctx, plan, abort, results, started: float, timeout_s: float | None = None
    ) -> None:
        """Wait out the plan or a cancel (or the travel budget, when given)
        and complete the goal. results(action_time) builds (note, payload)
        and runs after any abort, so it reads the plan where it stopped."""
        cancel_task = asyncio.ensure_future(ctx.cancel_signal())
        done_task = asyncio.ensure_future(plan.done.wait())
        try:
            await asyncio.wait(
                [cancel_task, done_task], timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
            action_time = time.monotonic() - started
            if plan.done.is_set():
                note, payload = results(action_time)
                success, message = _terminal(plan, note)
                await ctx.complete(success, message, *payload)
            elif cancel_task.done():
                abort(plan)
                note, payload = results(action_time)
                await ctx.complete_cancelled(False, _join("cancelled", note), *payload)
            else:
                abort(plan)
                note, payload = results(action_time)
                await ctx.complete(
                    False,
                    _join("goal did not complete within its travel budget", note),
                    *payload,
                )
        except Exception as e:
            log(f"goal completion failed: {e!r}")
        finally:
            cancel_task.cancel()
            done_task.cancel()

    async def drive_arm(self, ctx, plan: ArmPlan | None) -> None:
        """Wait out the plan or a cancel, then complete with the measured
        landing point."""
        if plan is None:
            await _complete_guarded(
                ctx, False, "goal admission was walked back", [0.0] * NUM_JOINTS, 0.0
            )
            return

        def results(action_time):
            positions, note = self._final_arm_positions(plan)
            return note, (list(positions), action_time)

        try:
            await self._finish_plan(
                ctx, plan, self._coordinator.abort_arm_plan, results, time.monotonic()
            )
        finally:
            self._coordinator.release_arm()

    async def drive_posture(self, ctx, plan: ArmPlan | None) -> None:
        if plan is None:
            await _complete_guarded(ctx, False, "goal admission was walked back")
            return

        def results(_action_time):
            _, note = self._final_arm_positions(plan)
            return note, ()

        try:
            await self._finish_plan(
                ctx, plan, self._coordinator.abort_arm_plan, results, time.monotonic()
            )
        finally:
            self._coordinator.release_arm()

    def _ee_floor_s(self, kinematics, seed: tuple[float, ...], goal: PoseGoal) -> float:
        """The duration floor the EE speed caps impose on a pose move. The
        governor shapes only streamed input, so a move is sized against both
        caps up front: the quintic's 15/8 peak over the start-to-goal chord
        (the joint-space blend's true EE path can bow past the chord)."""
        position, orientation = kinematics.forward_kinematics(seed)
        travel_ratio = math.dist(position, goal.position) / self._config.max_ee_velocity_m_s
        turn_ratio = (
            relative_rotation_rad(orientation, goal.orientation)
            / self._config.max_ee_angular_velocity_rad_s
        )
        return minimum_jerk.QUINTIC_PEAK_VELOCITY * max(travel_ratio, turn_ratio)

    async def drive_pose(self, ctx, goal: PoseGoal | None, solver, kinematics) -> None:
        """Solve, install the trajectory, and wait it out. The solve runs
        off the event loop on a solver dedicated to point-to-point goals
        (the control tick's streaming solver is a separate instance)."""
        if goal is None:
            await _complete_guarded(
                ctx, False, "goal admission was walked back", *_no_pose(), 0.0
            )
            return
        started = time.monotonic()
        try:
            seed = self._coordinator.arm_anchor()
            if seed is None:
                await _complete_guarded(
                    ctx, False, "follower state went stale before the move started",
                    *_no_pose(), 0.0,
                )
                return
            try:
                solution = await asyncio.to_thread(
                    solver.inverse_kinematics, seed, goal.position, goal.orientation
                )
                if solution is None or not self._limits.contains(solution):
                    position, orientation = kinematics.forward_kinematics(seed)
                    await _complete_guarded(
                        ctx, False, "pose unreachable within tolerance and joint limits",
                        list(position), list(orientation), time.monotonic() - started,
                    )
                    return
                duration_s = max(goal.duration_s, self._ee_floor_s(kinematics, seed, goal))
                if duration_s > MAX_REQUESTED_DURATION_S:
                    # The requested duration was admitted against this ceiling,
                    # but the EE speed caps can raise it past one. Stretching
                    # anyway would hold the arm slot for longer than any goal
                    # is allowed to, so the move is refused instead.
                    position, orientation = kinematics.forward_kinematics(seed)
                    await _complete_guarded(
                        ctx, False,
                        f"the end-effector speed caps need {duration_s:.0f}s for this "
                        f"move, beyond the {MAX_REQUESTED_DURATION_S:.0f}s ceiling",
                        list(position), list(orientation), time.monotonic() - started,
                    )
                    return
                profile = minimum_jerk.plan(
                    seed, solution, duration_s, self._config.max_joint_velocity_rad_s
                )
            except Exception as e:
                # A solver surprise must still reach a terminal result; the
                # caller would otherwise wait out its whole timeout on an
                # accepted goal.
                log(f"pose solve failed: {e!r}")
                await _complete_guarded(
                    ctx, False, f"pose solve failed: {e!r}",
                    *_no_pose(), time.monotonic() - started,
                )
                return
            plan = self._coordinator.adopt_arm_plan(profile)

            def results(action_time):
                positions, note = self._final_arm_positions(plan)
                position, orientation = kinematics.forward_kinematics(positions)
                return (
                    _join(note, _orientation_note(orientation, goal.orientation)),
                    (list(position), list(orientation), action_time),
                )

            await self._finish_plan(
                ctx, plan, self._coordinator.abort_arm_plan, results, started
            )
        except Exception as e:
            log(f"pose goal completion failed: {e!r}")
        finally:
            self._coordinator.release_arm()

    async def drive_gripper(self, ctx, plan: GripperPlan | None) -> None:
        if plan is None:
            await _complete_guarded(ctx, False, "goal admission was walked back", 0.0, 0.0)
            return

        def results(action_time):
            opening, note = self._final_opening(plan)
            return note, (opening, action_time)

        try:
            await self._finish_plan(
                ctx,
                plan,
                self._coordinator.abort_gripper_plan,
                results,
                time.monotonic(),
                timeout_s=plan.timeout_s,
            )
        finally:
            self._coordinator.release_gripper()

    # ---------------------------------------------------------------- results

    def _final_arm_positions(self, plan: ArmPlan) -> tuple[tuple[float, ...], str]:
        """Prefer the measured landing point; fall back to where the
        trajectory stopped, saying so."""
        measured = self._coordinator.measured_joints.fresh(
            self._coordinator.follower_state_timeout_s
        )
        if measured is not None:
            return measured, ""
        return (
            plan.last_sample(time.monotonic()),
            "follower state stale; reporting the commanded position",
        )

    def _final_opening(self, plan: GripperPlan) -> tuple[float, str]:
        """Prefer the measured opening; fall back to where the ramp stood
        when the plan was cut; only a plan cut before any command streamed
        reports its target, saying so."""
        measured = self._coordinator.measured_gripper.fresh(
            self._coordinator.follower_state_timeout_s
        )
        if measured is not None:
            return measured, ""
        if plan.frozen is not None:
            return plan.frozen, "follower state stale; reporting the commanded position"
        return (
            plan.target,
            "follower state stale before any command streamed; reporting the goal target",
        )


def _require_duration(duration_s: float) -> None:
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise Rejection("duration_s must be finite and non-negative")
    if duration_s > MAX_REQUESTED_DURATION_S:
        raise Rejection(f"duration_s must not exceed {MAX_REQUESTED_DURATION_S:.0f}")


def _require_arm_name(arm_name: str) -> None:
    if arm_name != ARM_LIMB:
        raise Rejection(f'unknown arm_name {arm_name!r}; this robot has one arm, "{ARM_LIMB}"')


def _require_gripper_name(gripper_name: str) -> None:
    if gripper_name != GRIPPER_LIMB:
        raise Rejection(
            f'unknown gripper_name {gripper_name!r}; this robot has one gripper, "{GRIPPER_LIMB}"'
        )


def _terminal(plan: ArmPlan | GripperPlan, note: str) -> tuple[bool, str]:
    """success and message for a done plan: a coordinator-failed plan is a
    failed goal, whatever the wall clock says."""
    if plan.failed is not None:
        return False, _join(plan.failed, note)
    return True, note


# Beyond this the arm landed somewhere an operator would not call the pose
# they asked for, so the result says so rather than reporting a bare success.
_ORIENTATION_NOTE_FLOOR_RAD = math.radians(2.0)


def _orientation_note(reached, requested) -> str:
    """What to say about the orientation actually achieved. Five joints
    underactuate three rotational degrees of freedom, so a move that reached
    its position can still be turned well away from the requested pose; a
    result that reported only success would hide that."""
    error_rad = relative_rotation_rad(reached, requested)
    if error_rad < _ORIENTATION_NOTE_FLOOR_RAD:
        return ""
    return (
        f"orientation reached within {math.degrees(error_rad):.0f} degrees of the "
        "request: five joints underactuate three rotational degrees of freedom"
    )


def _no_pose() -> tuple[list[float], list[float]]:
    """The final-pose fields for a refusal that never had a pose to report:
    no anchor was resolved, or the solve produced nothing. A path that does
    know where the arm is reports that instead. Fresh lists per call, since
    they are handed to the generated completion API."""
    return [0.0] * 3, [0.0, 0.0, 0.0, 1.0]


async def _complete_guarded(ctx, *args) -> None:
    try:
        await ctx.complete(*args)
    except Exception as e:
        log(f"goal completion failed: {e!r}")


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)
