"""Action admission and execution over the coordinator, and the two
limb_motion services that read or end what it runs.

Admission (the goal decide callback) validates the request, its settle
fields included, and claims the limb's single-flight busy slot, or refuses
the goal with a failure kind; execution installs a plan and waits for the
coordinator to stream it out: an arm move completes once the measured
joints, or the measured grasp point, stand within the settle tolerance of
the target, and fails with `timeout` when its settle timeout runs out past
the planned end; a gripper move completes, past the planned end of its
travel, once the jaws stand still or its settle timeout runs out, and
reports the opening measured then, where an object may hold the jaws.
Cancelling freezes the limb where the trajectory stands, and so does the
stop service, whoever started the move: the goal then ends cancelled, with
the reason of the cancel or of the stop, and its result. A failed goal
gives no result. The plan check runs a move_arm goal's solve from the same
anchor and installs nothing. Every terminal path releases the claimed slot,
and no exception may ever kill an action server: an unforeseen failure
refuses or fails the goal and abandons anything admission installed.
"""

from __future__ import annotations

import asyncio
import functools
import math
import time
from dataclasses import dataclass

from control_core_py import minimum_jerk
from control_core_py.minimum_jerk import Profile
from control_core_py.runtime import Latch, log
from so101_description.limbs import ARM_LIMB, GRIPPER_LIMB
from so101_description.limits import JointLimits
from so101_description.transforms import relative_rotation_rad
from so101_description.units import NUM_JOINTS

from so101_backbone.coordinator import (
    FOLLOWER_SILENT,
    ArmPlan,
    Coordinator,
    GripperPlan,
    JointSettle,
    PoseSettle,
    PostureEnd,
)
from so101_backbone.failures import FailureKind
from so101_backbone.params import Config

# Backoff after a failed goal delivery, so a persistently broken action
# transport logs once and retries at a human pace instead of spinning the
# event loop.
_GOAL_RETRY_S = 1.0

# The contract's bound on every duration and timeout a goal carries: a
# number of seconds from 0 to 60. The posture moves share it.
MAX_REQUESTED_DURATION_S = 60.0

# A quaternion of a smaller norm has no direction to normalize.
QUATERNION_MIN_NORM = 1e-6

# The refusal of an arm goal, or a check of one, while another arm goal
# runs, and of a gripper goal while another gripper goal runs.
ARM_BUSY = "arm is already executing a motion"
GRIPPER_BUSY = "gripper is already executing a move"
# The refusals of a goal naming no limb of this robot.
UNKNOWN_ARM_NAME = f'unknown arm_name: this robot\'s arm is "{ARM_LIMB}"'
UNKNOWN_GRIPPER_NAME = f'unknown gripper_name: this robot\'s gripper is "{GRIPPER_LIMB}"'
# The refusal of an arm or gripper goal, or a check of one, while a leader
# streams commands for that limb: the leader has it until its stream stops.
ARM_LEADER_STREAMING = (
    "a leader is streaming arm commands; release the headset's grip, switch off"
    " the browser panel's stream, or stop the leader node, then retry"
)
GRIPPER_LEADER_STREAMING = (
    "a leader is streaming gripper commands; release the headset's grip, switch"
    " off the browser panel's stream, or stop the leader node, then retry"
)
# The refusal of an arm goal, or a check of one, while no follower state is
# fresh enough to anchor the move from.
NO_ANCHOR = "no fresh follower state to anchor the move from"
# The failure of a goal whose admission a failed delivery walked back.
WALKED_BACK = "goal admission was walked back"


class Refusal(ValueError):
    """A goal that must not run, or a check that gets an error reply: the
    failure kind and the operator-facing reason."""

    def __init__(self, kind: FailureKind, reason: str):
        super().__init__(reason)
        self.kind = kind


class PlanRefused(ValueError):
    """A pose move that has no plan from where the arm is, with the reason a
    move_arm goal fails with, as `unreachable`, and check_arm_move answers."""


@dataclass(frozen=True)
class PlanCheck:
    """The answer of check_arm_move: whether a plan reaches the pose, why
    not, and the time the move would take (s)."""

    reachable: bool
    reason: str
    duration_s: float


def stop_message(reason: str) -> str:
    """The reason of every goal a stop ends: the stop and its reason."""
    if not reason:
        return "stopped"
    return f"stopped: {reason}"


def cancel_reason(cause: str | None) -> str:
    """The reason of a goal that its cancel signal ended, in the words of
    peppy for `cause`, the cancel cause of the goal context."""
    if cause == "sender_gone":
        return "its sender is gone"
    return "cancelled by its caller"


@dataclass(frozen=True)
class PoseGoal:
    """A pose to plan: the fields move_arm and check_arm_move share."""

    position: tuple[float, ...]
    orientation: tuple[float, ...]
    duration_s: float
    # The caller's planning slack, judged against the goal pose alone, never
    # the path.
    position_tolerance_m: float
    orientation_tolerance_rad: float


@dataclass(frozen=True)
class PoseMove:
    """An admitted move_arm goal whose IK solve runs in the drive task: the
    arm slot is already claimed, so the anchor cannot move under the solve."""

    goal: PoseGoal
    # How the move ends past the planned end of its trajectory.
    settle: PoseSettle
    # The stop count at admission: a stop since ends the goal before it
    # moves, the solve included.
    stops_before: int


class PendingSlot:
    """One server's admitted-but-not-yet-driven goal. Per server, so a failed
    delivery on one action can only ever walk back its own admission."""

    def __init__(self) -> None:
        self.plan: ArmPlan | GripperPlan | PoseMove | None = None

    def take(self) -> ArmPlan | GripperPlan | PoseMove | None:
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

    def _admit_arm_move(
        self,
        target: tuple[float, ...],
        duration_s: float,
        settle_tolerances_rad: tuple[float, ...],
        settle_timeout_s: float,
    ) -> ArmPlan:
        """Claim the arm's busy slot and install a move_arm_joints goal's
        trajectory and settle, or raise Refusal."""
        assert len(target) == NUM_JOINTS
        if not all(math.isfinite(p) for p in target):
            raise Refusal(FailureKind.REJECTED, "non-finite joint target")
        _require_duration(duration_s, "duration_s")
        if not self._limits.contains(target):
            raise Refusal(FailureKind.REJECTED, "target out of joint limits")
        settle = _joint_settle(settle_tolerances_rad, settle_timeout_s)
        return self._admit_trajectory(target, duration_s, settle)

    def _admit_trajectory(
        self, target: tuple[float, ...], duration_s: float, settle: JointSettle | PostureEnd
    ) -> ArmPlan:
        """Claim the arm's busy slot and install the trajectory to `target`
        with `settle`, or raise Refusal. The claim happens last so a
        refused goal never leaves the slot held."""
        anchor = self._claim_free_arm()
        try:
            profile = minimum_jerk.plan(
                anchor, target, duration_s, self._config.max_joint_velocity_rad_s
            )
            return self._coordinator.adopt_arm_plan(profile, settle)
        except BaseException:
            # A failure between claim and adopt must not hold the slot.
            self._coordinator.release_arm()
            raise

    def _admit_gripper_move(
        self, opening: float, max_effort: float, settle_timeout_s: float
    ) -> GripperPlan:
        if not math.isfinite(opening):
            raise Refusal(FailureKind.REJECTED, "non-finite gripper opening")
        if not 0.0 <= opening <= 1.0:
            raise Refusal(FailureKind.REJECTED, f"opening {opening} outside [0, 1]")
        if not math.isfinite(max_effort) or max_effort < 0.0:
            raise Refusal(
                FailureKind.REJECTED, f"max_effort {max_effort} is not a non-negative finite value"
            )
        settle_timeout_s = _require_duration(settle_timeout_s, "settle_timeout_s")
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
            raise Refusal(
                FailureKind.RESOURCE_UNAVAILABLE, "no fresh gripper state to anchor the move from"
            )
        if self._coordinator.leader_streams_gripper():
            raise Refusal(FailureKind.LEADER_STREAMING, GRIPPER_LEADER_STREAMING)
        if not self._coordinator.try_claim_gripper():
            raise Refusal(FailureKind.PRECONDITION_FAILED, GRIPPER_BUSY)
        try:
            return self._coordinator.adopt_gripper_plan(opening, settle_timeout_s, anchor)
        except BaseException:
            self._coordinator.release_gripper()
            raise

    def _admit_pose_move(
        self,
        position,
        orientation,
        duration_s: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        settle_position_tolerance_m: float,
        settle_orientation_tolerance_rad: float,
        settle_timeout_s: float,
    ) -> PoseMove:
        """Validate and claim the arm; the verified IK solve runs in the
        drive task so a hard pose never stalls the event loop inside the
        goal decide callback, failing the accepted goal when no solution
        exists."""
        # Parsed before the claim: a refusal past this point would strand
        # the arm slot and teleop with it.
        move = _pose_move(
            _pose_goal(
                position, orientation, duration_s, position_tolerance_m, orientation_tolerance_rad
            ),
            settle_position_tolerance_m,
            settle_orientation_tolerance_rad,
            settle_timeout_s,
            self._coordinator.stop_count,
        )
        self._claim_free_arm()
        return move

    def _free_arm_anchor(self) -> tuple[float, ...]:
        """Where a move of the arm starts from. Raises Refusal, in the order
        every arm goal and check is refused: no fresh follower state, a
        leader streaming the arm, an arm goal in flight. Claims nothing."""
        anchor = self._coordinator.arm_anchor()
        if anchor is None:
            raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, NO_ANCHOR)
        if self._coordinator.leader_streams_arm():
            raise Refusal(FailureKind.LEADER_STREAMING, ARM_LEADER_STREAMING)
        if self._coordinator.arm_busy:
            raise Refusal(FailureKind.PRECONDITION_FAILED, ARM_BUSY)
        return anchor

    def _claim_free_arm(self) -> tuple[float, ...]:
        """Claim the arm's busy slot: the anchor its move starts from. Raises
        Refusal, and claims nothing, as _free_arm_anchor does."""
        anchor = self._free_arm_anchor()
        claimed = self._coordinator.try_claim_arm()
        assert claimed, "the arm was found free on this same event loop turn"
        return anchor

    def _admit_posture(self, target: tuple[float, ...], duration_s: float) -> ArmPlan:
        """A posture move runs to the planned end of its trajectory and
        reports the measured grasp point."""
        _require_duration(duration_s, "duration_s")
        return self._admit_trajectory(target, duration_s, PostureEnd())

    # ----------------------------------------------------- decide constructors

    def decide_arm_joints(self, module, pending: PendingSlot):
        def decide(request):
            def admit():
                d = request.data
                _require_arm_name(d.arm_name)
                return self._admit_arm_move(
                    tuple(d.joint_positions),
                    d.duration_s,
                    tuple(d.settle_tolerances_rad),
                    d.settle_timeout_s,
                )

            return self._decide(module, admit, pending)

        return decide

    def decide_posture(self, module, target: tuple[float, ...], pending: PendingSlot):
        def decide(request):
            return self._decide(
                module,
                lambda: self._admit_posture(target, request.data.duration_s),
                pending,
            )

        return decide

    def decide_pose(self, module, pending: PendingSlot):
        def decide(request):
            def admit():
                d = request.data
                _require_arm_name(d.arm_name)
                return self._admit_pose_move(
                    tuple(d.position),
                    tuple(d.orientation),
                    d.duration_s,
                    d.plan_position_tolerance_m,
                    d.plan_orientation_tolerance_rad,
                    d.settle_position_tolerance_m,
                    d.settle_orientation_tolerance_rad,
                    d.settle_timeout_s,
                )

            return self._decide(module, admit, pending)

        return decide

    def decide_gripper(self, module, pending: PendingSlot):
        def decide(request):
            def admit():
                d = request.data
                _require_gripper_name(d.gripper_name)
                return self._admit_gripper_move(d.opening, d.max_effort, d.settle_timeout_s)

            return self._decide(module, admit, pending)

        return decide

    def _decide(self, module, admit, pending: PendingSlot):
        try:
            plan = admit()
        except Refusal as e:
            return module.GoalDecision.refuse(e.kind, str(e))
        except Exception as e:
            # No admission failure may escape into the generated server loop.
            log(f"goal admission failed unexpectedly: {e!r}")
            return module.GoalDecision.refuse(
                FailureKind.ATTEMPT_FAILED, f"admission failed: {e!r}"
            )
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
        elif isinstance(plan, PoseMove):
            # Nothing installed yet: the solve had not run. Only the claim
            # is walked back.
            self._coordinator.release_arm()
        else:
            self._coordinator.abort_gripper_plan(plan)
            self._coordinator.release_gripper()

    # -------------------------------------------------------------- execution

    async def _finish_plan(self, ctx, plan, abort, results, started: float) -> None:
        """Wait out the plan, a cancel or a stop, and end the goal.
        results(action_time) builds the result fields and runs after any
        abort, so it reads the plan where it stopped. A stop has already cut
        the plan, and the goal ends cancelled with the stop's reason; a plan
        the coordinator failed fails the goal with its kind."""
        cancel_task = asyncio.ensure_future(ctx.cancel_signal())
        done_task = asyncio.ensure_future(plan.done.wait())
        try:
            await asyncio.wait([cancel_task, done_task], return_when=asyncio.FIRST_COMPLETED)
            action_time = time.monotonic() - started
            if plan.stopped is not None:
                await ctx.complete_cancelled(plan.stopped, *results(action_time))
            elif plan.failed is not None:
                await ctx.fail(plan.failed.kind, plan.failed.message)
            elif plan.done.is_set():
                await ctx.complete(*results(action_time))
            else:
                abort(plan)
                await ctx.complete_cancelled(
                    cancel_reason(ctx.cancel_cause()), *results(action_time)
                )
        except Exception as e:
            log(f"goal completion failed: {e!r}")
        finally:
            cancel_task.cancel()
            done_task.cancel()

    async def drive_arm(self, ctx, plan: ArmPlan | None) -> None:
        """Wait out the plan or a cancel, then end the goal with the measured
        landing point."""
        if plan is None:
            await _end_guarded(ctx.fail, FailureKind.ATTEMPT_FAILED, WALKED_BACK)
            return

        def results(action_time):
            return list(self._final_arm_positions(plan)), action_time

        try:
            await self._finish_plan(
                ctx, plan, self._coordinator.abort_arm_plan, results, time.monotonic()
            )
        finally:
            self._coordinator.release_arm()

    async def drive_posture(self, ctx, plan: ArmPlan | None, kinematics) -> None:
        """Wait until the plan ends, a cancel comes or a stop comes. Then end
        the goal; a completed or cancelled goal gives the grasp pose of the
        arm, from the joints that the follower measures at that time (see
        _measured_grasp_pose)."""
        if plan is None:
            await _end_guarded(ctx.fail, FailureKind.ATTEMPT_FAILED, WALKED_BACK)
            return

        def results(_action_time):
            return self._measured_grasp_pose(kinematics)

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

    async def _plan_pose(self, solver, kinematics, seed, goal: PoseGoal) -> Profile:
        """The trajectory of `goal` from `seed`: the joint solution reached in
        the requested duration, floored by the joint velocity caps and the
        end-effector speed caps. PlanRefused with the reason when no solution
        lies inside the joint limits, or the caps push the duration past the
        ceiling. The solve runs off the event loop."""
        solution = await asyncio.to_thread(
            functools.partial(
                solver.inverse_kinematics,
                seed,
                goal.position,
                goal.orientation,
                position_tolerance_m=goal.position_tolerance_m,
                orientation_tolerance_rad=goal.orientation_tolerance_rad,
            )
        )
        if solution is None or not self._limits.contains(solution):
            raise PlanRefused(_unreached(goal))
        duration_s = max(goal.duration_s, self._ee_floor_s(kinematics, seed, goal))
        profile = minimum_jerk.plan(
            seed, solution, duration_s, self._config.max_joint_velocity_rad_s
        )
        if profile.duration_s > MAX_REQUESTED_DURATION_S:
            # The requested duration was admitted against this ceiling, but
            # the caps can raise it past one. Stretching anyway would hold
            # the arm slot for longer than any goal is allowed to, so the
            # move has no plan instead.
            raise PlanRefused(
                f"the speed caps need {profile.duration_s:.0f}s for this "
                f"move, beyond the {MAX_REQUESTED_DURATION_S:.0f}s ceiling"
            )
        return profile

    async def check_arm_move(
        self,
        arm_name: str,
        position,
        orientation,
        duration_s: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        solver,
        kinematics,
    ) -> PlanCheck:
        """Whether a move_arm goal with these planning fields has a plan from
        where the arm is, without moving, the reason being the words the
        goal's `unreachable` failure gives. Raises Refusal with the kind and
        the message of the goal's refusal, in the same order as a goal: a
        field that move_arm does not take, no fresh follower state, a leader
        streaming the arm, an arm goal in flight; and with `attempt_failed`
        when the solve raises."""
        _require_arm_name(arm_name)
        goal = _pose_goal(
            position, orientation, duration_s, position_tolerance_m, orientation_tolerance_rad
        )
        seed = self._free_arm_anchor()
        try:
            profile = await self._plan_pose(solver, kinematics, seed, goal)
        except PlanRefused as e:
            return PlanCheck(reachable=False, reason=str(e), duration_s=0.0)
        except Exception as e:
            log(f"pose solve failed: {e!r}")
            raise Refusal(FailureKind.ATTEMPT_FAILED, f"pose solve failed: {e!r}") from e
        return PlanCheck(reachable=True, reason="", duration_s=profile.duration_s)

    def stop_moves(self, reason: str) -> list[str]:
        """End every planned move in flight for the stop service: the names
        of the limbs whose move was cut."""
        message = stop_message(reason)
        log(f"stop requested: {message}")
        return self._coordinator.stop_all(message)

    async def drive_pose(self, ctx, move: PoseMove | None, solver, kinematics) -> None:
        """Solve, install the trajectory with the move's settle, and wait it
        out. The solve runs off the event loop on a solver dedicated to
        point-to-point goals (the control tick's streaming solver is a
        separate instance). A stop that runs during the solve ends the goal
        cancelled with the stop's reason, and nothing is installed. A pose
        with no plan fails the goal with `unreachable`."""
        if move is None:
            await _end_guarded(ctx.fail, FailureKind.ATTEMPT_FAILED, WALKED_BACK)
            return
        goal = move.goal
        try:
            seed = self._coordinator.arm_anchor()
            if seed is None:
                await _end_guarded(ctx.fail, FOLLOWER_SILENT.kind, FOLLOWER_SILENT.message)
                return

            async def end_stopped() -> None:
                await _end_guarded(
                    ctx.complete_cancelled,
                    self._coordinator.last_stop,
                    *self._standing_pose(kinematics),
                    0.0,
                )

            if self._coordinator.stop_count != move.stops_before:
                await end_stopped()
                return
            started = time.monotonic()
            try:
                profile = await self._plan_pose(solver, kinematics, seed, goal)
            except PlanRefused as e:
                await _end_guarded(ctx.fail, FailureKind.UNREACHABLE, str(e))
                return
            except Exception as e:
                # A solver surprise must still reach a terminal outcome; the
                # caller would otherwise wait out its whole timeout on an
                # accepted goal.
                log(f"pose solve failed: {e!r}")
                await _end_guarded(
                    ctx.fail, FailureKind.ATTEMPT_FAILED, f"pose solve failed: {e!r}"
                )
                return
            if self._coordinator.stop_count != move.stops_before:
                await end_stopped()
                return
            plan = self._coordinator.adopt_arm_plan(profile, move.settle)

            def results(action_time):
                position, orientation = kinematics.forward_kinematics(
                    self._final_arm_positions(plan)
                )
                return list(position), list(orientation), action_time

            await self._finish_plan(
                ctx, plan, self._coordinator.abort_arm_plan, results, started
            )
        except Exception as e:
            log(f"pose goal completion failed: {e!r}")
        finally:
            self._coordinator.release_arm()

    async def drive_gripper(self, ctx, plan: GripperPlan | None) -> None:
        if plan is None:
            await _end_guarded(ctx.fail, FailureKind.ATTEMPT_FAILED, WALKED_BACK)
            return

        def results(action_time):
            return _settled(plan), self._final_opening(plan), action_time

        try:
            await self._finish_plan(
                ctx, plan, self._coordinator.abort_gripper_plan, results, time.monotonic()
            )
        finally:
            self._coordinator.release_gripper()

    # ---------------------------------------------------------------- results

    def _fresh_measured_joints(self) -> tuple[float, ...] | None:
        """The joints the follower measured, or None when its state is
        stale: the one freshness rule of every arm and posture result."""
        return self._coordinator.measured_joints.fresh(
            self._coordinator.follower_state_timeout_s
        )

    def _final_arm_positions(self, plan: ArmPlan) -> tuple[float, ...]:
        """Prefer the measured landing point; fall back to where the
        trajectory stopped."""
        measured = self._fresh_measured_joints()
        if measured is not None:
            return measured
        return plan.last_sample(time.monotonic())

    def _measured_grasp_pose(self, kinematics) -> tuple[list[str], list[float], list[float]]:
        """The pose fields of a posture result: (arm_names, positions,
        orientations). The pose is the forward kinematics of the measured
        joints. A posture result gives only a measured pose. Thus, when the
        follower state is stale, the three arrays are empty."""
        measured = self._fresh_measured_joints()
        if measured is None:
            return [], [], []
        position, orientation = kinematics.forward_kinematics(measured)
        return [ARM_LIMB], list(position), list(orientation)

    def _standing_pose(self, kinematics) -> tuple[list[float], list[float]]:
        """Where the grasp point stands with no plan installed, as a move
        that never ran reports it: the forward kinematics of the fresh
        measured joints, else of the joints commanded last, else no pose."""
        joints = self._fresh_measured_joints()
        if joints is None:
            joints = self._coordinator.commanded_arm
        if joints is None:
            return _no_pose()
        position, orientation = kinematics.forward_kinematics(joints)
        return list(position), list(orientation)

    def _final_opening(self, plan: GripperPlan) -> float:
        """The opening measured when the plan completed. Otherwise prefer the
        measured opening; fall back to where the ramp stood when the plan was
        cut."""
        if plan.completed_on is not None:
            return plan.completed_on
        measured = self._coordinator.measured_gripper.fresh(
            self._coordinator.follower_state_timeout_s
        )
        if measured is not None:
            return measured
        assert plan.frozen is not None, "a plan that did not complete was cut"
        return plan.frozen


def _pose_goal(
    position,
    orientation,
    duration_s: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> PoseGoal:
    """The pose goal these wire fields name, with its orientation normalized,
    or Refusal: a pose that is no pose, a tolerance that is no bar, or a
    duration out of its bound."""
    if not all(math.isfinite(v) for v in (*position, *orientation)):
        raise Refusal(FailureKind.REJECTED, "goal pose has non-finite values")
    norm = math.sqrt(sum(q * q for q in orientation))
    if norm <= QUATERNION_MIN_NORM:
        raise Refusal(FailureKind.REJECTED, "goal pose has an unnormalizable orientation")
    position_tolerance_m = _require_tolerance(position_tolerance_m, "plan_position_tolerance_m")
    orientation_tolerance_rad = _require_tolerance(
        orientation_tolerance_rad, "plan_orientation_tolerance_rad"
    )
    return PoseGoal(
        position=tuple(position),
        orientation=tuple(q / norm for q in orientation),
        duration_s=_require_duration(duration_s, "duration_s"),
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )


def _pose_move(
    goal: PoseGoal,
    settle_position_tolerance_m: float,
    settle_orientation_tolerance_rad: float,
    settle_timeout_s: float,
    stops_before: int,
) -> PoseMove:
    """The move_arm goal `goal` settles into, admitted at stop count
    `stops_before`, or Refusal: a settle tolerance that is no bar or below
    the plan tolerance of its axis, or a timeout out of its bound."""
    position_tolerance_m = _require_tolerance(
        settle_position_tolerance_m, "settle_position_tolerance_m"
    )
    orientation_tolerance_rad = _require_tolerance(
        settle_orientation_tolerance_rad, "settle_orientation_tolerance_rad"
    )
    _require_settle_covers_plan(position_tolerance_m, goal.position_tolerance_m, "position_tolerance_m")
    _require_settle_covers_plan(
        orientation_tolerance_rad, goal.orientation_tolerance_rad, "orientation_tolerance_rad"
    )
    settle = PoseSettle(
        position=goal.position,
        orientation=goal.orientation,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
        timeout_s=_require_duration(settle_timeout_s, "settle_timeout_s"),
    )
    return PoseMove(goal=goal, settle=settle, stops_before=stops_before)


def _joint_settle(tolerances_rad: tuple[float, ...], timeout_s: float) -> JointSettle:
    """The settle of a move_arm_joints goal, or Refusal: a positive finite
    tolerance per joint of the arm, and a timeout."""
    assert len(tolerances_rad) == NUM_JOINTS
    if not all(math.isfinite(v) and v > 0.0 for v in tolerances_rad):
        raise Refusal(
            FailureKind.REJECTED, "each settle_tolerances_rad entry must be a positive number"
        )
    return JointSettle(
        tolerances_rad=tuple(tolerances_rad),
        timeout_s=_require_duration(timeout_s, "settle_timeout_s"),
    )


def _require_duration(value: float, field: str) -> float:
    """`value`, the seconds the wire field `field` carries, or Refusal
    unless it is within the contract's bound."""
    if not (math.isfinite(value) and 0.0 <= value <= MAX_REQUESTED_DURATION_S):
        raise Refusal(
            FailureKind.REJECTED,
            f"{field} must be a number of seconds from 0 to {MAX_REQUESTED_DURATION_S:.0f}",
        )
    return value


def _require_tolerance(value: float, field: str) -> float:
    """`value`, the bar the wire field `field` carries, or Refusal unless
    it is positive and finite."""
    if not (math.isfinite(value) and value > 0.0):
        raise Refusal(FailureKind.REJECTED, f"{field} must be a positive number")
    return value


def _require_settle_covers_plan(settle_bar: float, plan_bar: float, axis: str) -> None:
    """Refuse a settle bar below the plan bar of its axis, naming the field
    to raise: the plan may sit that far from the pose."""
    if settle_bar < plan_bar:
        raise Refusal(
            FailureKind.REJECTED,
            f"settle_{axis} must be at least plan_{axis}: the plan may sit that far from the pose",
        )


def _unreached(goal: PoseGoal) -> str:
    """Why a solve found no plan, naming the bars it was judged against."""
    return (
        f"goal pose not planned within {goal.position_tolerance_m * 1000:.1f} mm / "
        f"{math.degrees(goal.orientation_tolerance_rad):.1f} deg: no solution reaches it "
        "from where the arm is now, inside the joint limits; the search is a local "
        "descent, so the same pose may solve from a different posture"
    )


def _require_arm_name(arm_name: str) -> None:
    if arm_name != ARM_LIMB:
        raise Refusal(FailureKind.REJECTED, UNKNOWN_ARM_NAME)


def _require_gripper_name(gripper_name: str) -> None:
    if gripper_name != GRIPPER_LIMB:
        raise Refusal(FailureKind.REJECTED, UNKNOWN_GRIPPER_NAME)


def _settled(plan: GripperPlan) -> bool:
    """Whether the jaws of `plan` stood still before its settle timeout: a
    plan that completed at the timeout, or that a cancel or a stop cut, did
    not settle."""
    return plan.completed_on is not None and plan.still_moving_after_s is None


def _no_pose() -> tuple[list[float], list[float]]:
    """The final-pose fields of a move that never ran with neither fresh
    measured joints nor a commanded setpoint to place the arm. Fresh lists
    per call, since they are handed to the generated completion API."""
    return [0.0] * 3, [0.0, 0.0, 0.0, 1.0]


async def _end_guarded(end, *args) -> None:
    """Ends the goal through `end`, the context's complete, fail or
    complete_cancelled, and logs an end the runtime refused."""
    try:
        await end(*args)
    except Exception as e:
        log(f"goal completion failed: {e!r}")
