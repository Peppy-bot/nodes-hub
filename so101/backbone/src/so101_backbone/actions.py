"""Action admission and execution over the coordinator, and the two
limb_motion services that read or end what it runs.

Admission (the goal decide callback) validates the request, its settle
fields included, and claims the limb's single-flight busy slot; execution
installs a plan and waits for the coordinator to stream it out: an arm move
ends once the measured joints, or the measured grasp point, stand within the
settle tolerance of the target, and fails when its settle timeout runs out
past the planned end; a gripper move ends, past the planned end of its
travel, once the jaws stand still or its settle timeout runs out, and
reports the opening measured then, where an object may hold the jaws.
Cancelling freezes the limb where the trajectory stands, and so does the
stop service, whoever started the move: the goal then ends as cancelled with
the stop's message. The plan
check runs a move_arm goal's solve from the same anchor and installs
nothing. Every terminal path releases the claimed slot, and no exception may
ever kill an action server: an unforeseen failure rejects the goal and
abandons anything admission installed.
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
    STALE_FOLLOWER_REFUSAL,
    ArmPlan,
    Coordinator,
    GripperPlan,
    JointSettle,
    PoseSettle,
    PostureEnd,
)
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
# The messages the posture moves complete with.
READY_DONE = "move to ready completed"
HOME_DONE = "move to home completed"
# The refusals of a goal naming no limb of this robot.
UNKNOWN_ARM_NAME = f'unknown arm_name: this robot\'s arm is "{ARM_LIMB}"'
UNKNOWN_GRIPPER_NAME = f'unknown gripper_name: this robot\'s gripper is "{GRIPPER_LIMB}"'
# The note a result adds when it reports the position commanded last, the
# follower having stopped reporting.
COMMANDED_JOINTS_NOTE = "the result comes from the joints commanded last"
COMMANDED_OPENING_NOTE = "the result comes from the opening commanded last"
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


class Rejection(ValueError):
    """A goal that must not run, with the operator-facing reason."""


class PlanRefused(ValueError):
    """A pose move that has no plan from where the arm is, with the reason a
    move_arm goal completes with and check_arm_move answers."""


def stop_message(reason: str) -> str:
    """The message every goal a stop ends carries: the stop and its reason."""
    if not reason:
        return "stopped"
    return f"stopped: {reason}"


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
        trajectory and settle, or raise Rejection."""
        assert len(target) == NUM_JOINTS
        if not all(math.isfinite(p) for p in target):
            raise Rejection("non-finite joint target")
        _require_duration(duration_s, "duration_s")
        if not self._limits.contains(target):
            raise Rejection("target out of joint limits")
        settle = _joint_settle(settle_tolerances_rad, settle_timeout_s)
        return self._admit_trajectory(target, duration_s, settle)

    def _admit_trajectory(
        self, target: tuple[float, ...], duration_s: float, settle: JointSettle | PostureEnd
    ) -> ArmPlan:
        """Claim the arm's busy slot and install the trajectory to `target`
        with `settle`, or raise Rejection. The claim happens last so a
        rejected goal never leaves the slot held."""
        anchor = self._coordinator.arm_anchor()
        if anchor is None:
            raise Rejection(NO_ANCHOR)
        if self._coordinator.leader_streams_arm():
            raise Rejection(ARM_LEADER_STREAMING)
        if not self._coordinator.try_claim_arm():
            raise Rejection(ARM_BUSY)
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
            raise Rejection("non-finite gripper opening")
        if not 0.0 <= opening <= 1.0:
            raise Rejection(f"opening {opening} outside [0, 1]")
        if not math.isfinite(max_effort) or max_effort < 0.0:
            raise Rejection(f"max_effort {max_effort} is not a non-negative finite value")
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
            raise Rejection("no fresh gripper state to anchor the move from")
        if self._coordinator.leader_streams_gripper():
            raise Rejection(GRIPPER_LEADER_STREAMING)
        if not self._coordinator.try_claim_gripper():
            raise Rejection(GRIPPER_BUSY)
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
        # Parsed before the claim: a rejection past this point would strand
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
        if self._coordinator.arm_anchor() is None:
            raise Rejection(NO_ANCHOR)
        if self._coordinator.leader_streams_arm():
            raise Rejection(ARM_LEADER_STREAMING)
        if not self._coordinator.try_claim_arm():
            raise Rejection(ARM_BUSY)
        return move

    def _admit_posture(
        self, target: tuple[float, ...], duration_s: float, completion: str
    ) -> ArmPlan:
        """A posture move runs to the planned end of its trajectory and
        reports the measured grasp point with `completion` as its message."""
        _require_duration(duration_s, "duration_s")
        return self._admit_trajectory(target, duration_s, PostureEnd(completion))

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

    def decide_posture(
        self, module, target: tuple[float, ...], completion: str, pending: PendingSlot
    ):
        def decide(request):
            return self._decide(
                module,
                lambda: self._admit_posture(target, request.data.duration_s, completion),
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
        elif isinstance(plan, PoseMove):
            # Nothing installed yet: the solve had not run. Only the claim
            # is walked back.
            self._coordinator.release_arm()
        else:
            self._coordinator.abort_gripper_plan(plan)
            self._coordinator.release_gripper()

    # -------------------------------------------------------------- execution

    async def _finish_plan(self, ctx, plan, abort, results, started: float) -> None:
        """Wait out the plan, a cancel or a stop and complete the goal.
        results(action_time) builds (note, payload) and runs after any abort,
        so it reads the plan where it stopped. A stop has already cut the
        plan, and the goal ends as cancelled with the stop's message."""
        cancel_task = asyncio.ensure_future(ctx.cancel_signal())
        done_task = asyncio.ensure_future(plan.done.wait())
        try:
            await asyncio.wait([cancel_task, done_task], return_when=asyncio.FIRST_COMPLETED)
            action_time = time.monotonic() - started
            if plan.stopped is not None:
                note, payload = results(action_time)
                await ctx.complete_cancelled(False, _join(plan.stopped, note), *payload)
            elif plan.done.is_set():
                note, payload = results(action_time)
                success, message = _terminal(plan, note)
                await ctx.complete(success, message, *payload)
            else:
                abort(plan)
                note, payload = results(action_time)
                await ctx.complete_cancelled(False, _join("goal cancelled", note), *payload)
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
                ctx.complete, False, "goal admission was walked back", [0.0] * NUM_JOINTS, 0.0
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

    async def drive_posture(self, ctx, plan: ArmPlan | None, kinematics) -> None:
        """Wait until the plan ends, a cancel comes or a stop comes. Then
        complete the goal with the grasp pose of the arm, whatever the value
        of success. The pose comes from the joints that the follower measures
        at that time. All terminal paths, also a walked-back admission, use
        the same rule (see _measured_grasp_pose)."""
        if plan is None:
            note, pose = self._measured_grasp_pose(kinematics)
            await _complete_guarded(
                ctx.complete, False, _join("goal admission was walked back", note), *pose
            )
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
            # move is refused instead.
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
    ) -> tuple[bool, str, float]:
        """Whether a move_arm goal with these planning fields has a plan from
        where the arm is, without moving: (success, message, duration_s), the
        message being the words the goal's refusal gives. Refused with the
        same messages, in the same order, as a goal: no fresh follower state,
        a leader streaming the arm, an arm goal in flight."""
        try:
            _require_arm_name(arm_name)
            goal = _pose_goal(
                position, orientation, duration_s, position_tolerance_m, orientation_tolerance_rad
            )
        except Rejection as e:
            return False, str(e), 0.0
        seed = self._coordinator.arm_anchor()
        if seed is None:
            return False, NO_ANCHOR, 0.0
        if self._coordinator.leader_streams_arm():
            return False, ARM_LEADER_STREAMING, 0.0
        if self._coordinator.arm_busy:
            return False, ARM_BUSY, 0.0
        try:
            profile = await self._plan_pose(solver, kinematics, seed, goal)
        except PlanRefused as e:
            return False, str(e), 0.0
        except Exception as e:
            log(f"pose solve failed: {e!r}")
            return False, f"pose solve failed: {e!r}", 0.0
        planned_s = profile.duration_s
        return True, f"a plan reaches the pose in {planned_s:.3f} s", planned_s

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
        as cancelled with the stop's message, and nothing is installed."""
        if move is None:
            await _complete_guarded(
                ctx.complete, False, "goal admission was walked back", *_no_pose(), 0.0
            )
            return
        goal = move.goal
        started = time.monotonic()
        try:
            seed = self._coordinator.arm_anchor()
            if seed is None:
                note, pose = self._standing_pose(kinematics)
                await _complete_guarded(
                    ctx.complete, False, _join(STALE_FOLLOWER_REFUSAL, note), *pose, 0.0
                )
                return

            async def end_stopped() -> None:
                note, pose = self._standing_pose(kinematics)
                await _complete_guarded(
                    ctx.complete_cancelled, False, _join(self._coordinator.last_stop, note),
                    *pose, 0.0,
                )

            if self._coordinator.stop_count != move.stops_before:
                await end_stopped()
                return
            try:
                profile = await self._plan_pose(solver, kinematics, seed, goal)
            except PlanRefused as e:
                position, orientation = kinematics.forward_kinematics(seed)
                await _complete_guarded(
                    ctx.complete, False, str(e), list(position), list(orientation), 0.0
                )
                return
            except Exception as e:
                # A solver surprise must still reach a terminal result; the
                # caller would otherwise wait out its whole timeout on an
                # accepted goal.
                log(f"pose solve failed: {e!r}")
                await _complete_guarded(
                    ctx.complete, False, f"pose solve failed: {e!r}",
                    *_no_pose(), time.monotonic() - started,
                )
                return
            if self._coordinator.stop_count != move.stops_before:
                await end_stopped()
                return
            plan = self._coordinator.adopt_arm_plan(profile, move.settle)

            def results(action_time):
                positions, note = self._final_arm_positions(plan)
                position, orientation = kinematics.forward_kinematics(positions)
                return note, (list(position), list(orientation), action_time)

            await self._finish_plan(
                ctx, plan, self._coordinator.abort_arm_plan, results, started
            )
        except Exception as e:
            log(f"pose goal completion failed: {e!r}")
        finally:
            self._coordinator.release_arm()

    async def drive_gripper(self, ctx, plan: GripperPlan | None) -> None:
        if plan is None:
            await _complete_guarded(ctx.complete, False, "goal admission was walked back", 0.0, 0.0)
            return

        def results(action_time):
            opening, note = self._final_opening(plan)
            return note, (opening, action_time)

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

    def _final_arm_positions(self, plan: ArmPlan) -> tuple[tuple[float, ...], str]:
        """Prefer the measured landing point; fall back to where the
        trajectory stopped, saying so."""
        measured = self._fresh_measured_joints()
        if measured is not None:
            return measured, ""
        return plan.last_sample(time.monotonic()), COMMANDED_JOINTS_NOTE

    def _measured_grasp_pose(
        self, kinematics
    ) -> tuple[str, tuple[list[str], list[float], list[float]]]:
        """The pose fields of a posture result and a note: (note, (arm_names,
        positions, orientations)). The pose is the forward kinematics of the
        measured joints. A posture result gives only a measured pose. Thus,
        when the follower state is stale, the three arrays are empty and the
        note gives the reason."""
        measured = self._fresh_measured_joints()
        if measured is None:
            return f"no arm poses: {ARM_LIMB} stopped reporting its joints", ([], [], [])
        position, orientation = kinematics.forward_kinematics(measured)
        return "", ([ARM_LIMB], list(position), list(orientation))

    def _standing_pose(self, kinematics) -> tuple[str, tuple[list[float], list[float]]]:
        """Where the grasp point stands with no plan installed, as a move
        that never ran reports it: the forward kinematics of the fresh
        measured joints, else of the joints commanded last with the note,
        else no pose."""
        measured = self._fresh_measured_joints()
        if measured is not None:
            position, orientation = kinematics.forward_kinematics(measured)
            return "", (list(position), list(orientation))
        commanded = self._coordinator.commanded_arm
        if commanded is None:
            return "", _no_pose()
        position, orientation = kinematics.forward_kinematics(commanded)
        return COMMANDED_JOINTS_NOTE, (list(position), list(orientation))

    def _final_opening(self, plan: GripperPlan) -> tuple[float, str]:
        """The opening measured when the plan completed. Otherwise prefer the
        measured opening; fall back to where the ramp stood when the plan was
        cut, saying so."""
        if plan.completed_on is not None:
            return plan.completed_on, ""
        measured = self._coordinator.measured_gripper.fresh(
            self._coordinator.follower_state_timeout_s
        )
        if measured is not None:
            return measured, ""
        assert plan.frozen is not None, "a plan that did not complete was cut"
        return plan.frozen, COMMANDED_OPENING_NOTE


def _pose_goal(
    position,
    orientation,
    duration_s: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> PoseGoal:
    """The pose goal these wire fields name, with its orientation normalized,
    or Rejection: a pose that is no pose, a tolerance that is no bar, or a
    duration out of its bound."""
    if not all(math.isfinite(v) for v in (*position, *orientation)):
        raise Rejection("goal pose has non-finite values")
    norm = math.sqrt(sum(q * q for q in orientation))
    if norm <= QUATERNION_MIN_NORM:
        raise Rejection("goal pose has an unnormalizable orientation")
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
    `stops_before`, or Rejection: a settle tolerance that is no bar or below
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
    """The settle of a move_arm_joints goal, or Rejection: a positive finite
    tolerance per joint of the arm, and a timeout."""
    assert len(tolerances_rad) == NUM_JOINTS
    if not all(math.isfinite(v) and v > 0.0 for v in tolerances_rad):
        raise Rejection("each settle_tolerances_rad entry must be a positive number")
    return JointSettle(
        tolerances_rad=tuple(tolerances_rad),
        timeout_s=_require_duration(timeout_s, "settle_timeout_s"),
    )


def _require_duration(value: float, field: str) -> float:
    """`value`, the seconds the wire field `field` carries, or Rejection
    unless it is within the contract's bound."""
    if not (math.isfinite(value) and 0.0 <= value <= MAX_REQUESTED_DURATION_S):
        raise Rejection(
            f"{field} must be a number of seconds from 0 to {MAX_REQUESTED_DURATION_S:.0f}"
        )
    return value


def _require_tolerance(value: float, field: str) -> float:
    """`value`, the bar the wire field `field` carries, or Rejection unless
    it is positive and finite."""
    if not (math.isfinite(value) and value > 0.0):
        raise Rejection(f"{field} must be a positive number")
    return value


def _require_settle_covers_plan(settle_bar: float, plan_bar: float, axis: str) -> None:
    """Refuse a settle bar below the plan bar of its axis, naming the field
    to raise: the plan may sit that far from the pose."""
    if settle_bar < plan_bar:
        raise Rejection(
            f"settle_{axis} must be at least plan_{axis}: the plan may sit that far from the pose"
        )


def _unreached(goal: PoseGoal) -> str:
    """Why a solve refused, naming the bars it was judged against."""
    return (
        f"goal pose not planned within {goal.position_tolerance_m * 1000:.1f} mm / "
        f"{math.degrees(goal.orientation_tolerance_rad):.1f} deg: no solution reaches it "
        "from where the arm is now, inside the joint limits; the search is a local "
        "descent, so the same pose may solve from a different posture"
    )


def _require_arm_name(arm_name: str) -> None:
    if arm_name != ARM_LIMB:
        raise Rejection(UNKNOWN_ARM_NAME)


def _require_gripper_name(gripper_name: str) -> None:
    if gripper_name != GRIPPER_LIMB:
        raise Rejection(UNKNOWN_GRIPPER_NAME)


def _terminal(plan: ArmPlan | GripperPlan, note: str) -> tuple[bool, str]:
    """success and message for a done plan: a coordinator-failed plan is a
    failed goal, whatever the wall clock says; a plan that did not fail
    completed, an arm plan as its settle ends it and a gripper plan past the
    planned end of its travel, with the message of that ending."""
    if plan.failed is not None:
        return False, _join(plan.failed, note)
    return True, _join(_completion(plan), note)


def _completion(plan: ArmPlan | GripperPlan) -> str:
    """The words a completed plan reports."""
    if isinstance(plan, ArmPlan):
        return plan.settle.completion
    if plan.still_moving_after_s is not None:
        return (
            "move complete: the jaws had not stood still by the settle timeout, "
            f"{plan.still_moving_after_s:.1f} s after the planned end"
        )
    return "move complete"


def _no_pose() -> tuple[list[float], list[float]]:
    """The final-pose fields of a result with no pose to report: a walked-back
    admission, a solve that raised, or a move that never ran with neither
    fresh measured joints nor a commanded setpoint to place the arm. Fresh
    lists per call, since they are handed to the generated completion API."""
    return [0.0] * 3, [0.0, 0.0, 0.0, 1.0]


async def _complete_guarded(complete, *args) -> None:
    """Ends the goal through `complete`, the context's complete or
    complete_cancelled, and logs a completion the runtime refused."""
    try:
        await complete(*args)
    except Exception as e:
        log(f"goal completion failed: {e!r}")


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)
