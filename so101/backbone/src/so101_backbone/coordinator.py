"""The per-tick motion authority: one loop resolves what the follower is told.

Precedence per limb per tick: an admitted action plan owns the limb (streamed
input is ignored while one runs); otherwise a fresh streamed leader target
passes through under the end-effector-speed governor (the joints-led
stream's only limiter; the pose-mode stream's IK output is additionally
rate-stepped per joint); otherwise silence, and the follower holds. The
arm and the gripper claim independently, so a long arm move never freezes
gripper teleop and vice versa.

Fresh follower state gates everything: with the measured stream stale no limb
gets a setpoint, an active plan fails, and the standing command resets. The
first tick after any silence re-engages from the measured position, governed
by the same caps, exactly like every other tick.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from control_core_py.minimum_jerk import Profile
from control_core_py.runtime import Latch
from so101_description.limits import JointLimits

from so101_backbone.governor import EEGovernor, rate_step
from so101_backbone.params import Config, UpstreamMode
from so101_backbone.reach import ReachBall

# The leader-stream deadman: samples that arrived longer ago than this are
# not followed and silence resets the standing command.
STALE_LEADER_TIMEOUT_S = 0.25
# The wire-age gate every inbound stream passes, leader commands and
# follower states alike: a sample stamped this far from now, past or future,
# is a replayed backlog or a clock fault, not something current. Distinct
# from the deadman above, which judges arrival recency rather than the
# producer's stamp, even though both currently sit at a quarter second.
STALE_WIRE_TIMEOUT_S = 0.25
# Follower liveness: measured state older than this silences the wire and
# fails active plans. Absolute rather than derived from this node's control
# period, because the follower publishes state on its own independent rate;
# 0.1 s is six periods of the family's validated 60 Hz state stream.
STALE_FOLLOWER_TIMEOUT_S = 0.1


class LatestValue:
    """Latest-wins holder with an arrival stamp, plus the producer's capture
    stamp and a clear().

    control_core_py's LatestSlot is the cross-thread form, lock-guarded for
    a device thread handing values to an event loop. Every holder here lives
    on the one event loop, so the lock would buy nothing, and the stamp this
    node must forward downstream rides with the value rather than in a
    holder beside it."""

    def __init__(self) -> None:
        self._value = None
        self._stamp = 0.0
        # The producer's capture timestamp (wire seconds), forwarded so the
        # downstream stamp names when the leader read the sample, not when
        # this node relayed it.
        self.wire_timestamp_s: float | None = None

    def set(self, value, wire_timestamp_s: float | None = None) -> None:
        self._value = value
        self._stamp = time.monotonic()
        self.wire_timestamp_s = wire_timestamp_s

    def fresh(self, timeout_s: float):
        if self._value is None or time.monotonic() - self._stamp > timeout_s:
            return None
        return self._value

    def clear(self) -> None:
        self._value = None


@dataclass
class ArmPlan:
    profile: Profile
    done: asyncio.Event = field(default_factory=asyncio.Event)
    started_monotonic: float | None = None
    aborted: bool = False
    # Set when the coordinator cut the plan short for a cause the goal must
    # report (follower staleness, a dead downstream wire).
    failed: str | None = None
    # The sample the stream stood at when the plan was cut; last_sample
    # reports it instead of re-evaluating the profile at a later clock.
    frozen: tuple[float, ...] | None = None

    def last_sample(self, now: float) -> tuple[float, ...]:
        if self.frozen is not None:
            return self.frozen
        if self.started_monotonic is None:
            return self.profile.start
        return self.profile.sample(now - self.started_monotonic)


@dataclass
class GripperPlan:
    target: float
    # Travel budget, sized at admission: a goal that stalls past it fails
    # instead of holding the busy slot forever. Required at construction,
    # because a plan that reached the wait with an unset budget would fail
    # the moment it started.
    timeout_s: float
    done: asyncio.Event = field(default_factory=asyncio.Event)
    aborted: bool = False
    failed: str | None = None
    # The opening the ramp stood at when the plan was cut, for reporting.
    frozen: float | None = None


class Coordinator:
    def __init__(self, config: Config, kinematics, limits: JointLimits, reach: ReachBall):
        self._config = config
        self._kinematics = kinematics
        self._limits = limits
        self._reach = reach
        self.leader_joints = LatestValue()
        self.leader_pose = LatestValue()
        self.leader_gripper = LatestValue()
        self.measured_joints = LatestValue()
        self.measured_gripper = LatestValue()

        period_s = 1.0 / config.control_rate_hz
        self._governor = EEGovernor(
            kinematics,
            config.max_ee_velocity_m_s,
            config.max_ee_angular_velocity_rad_s,
            period_s,
        )
        self._gripper_tick_cap = config.max_gripper_rate_frac_s * period_s
        # Per-tick joint steps for the pose-mode stream only: the IK output is
        # machine-generated, and near the workspace boundary the solver can
        # flip configurations tick to tick with joint swings the EE governor
        # cannot see. Stepping the anchor toward each solution also seeds the
        # next solve nearby, keeping the solver in one basin. The joints-led
        # stream is never shaped by this: it must match lerobot's.
        self._pose_joint_tick_caps = tuple(
            v * period_s for v in config.max_joint_velocity_rad_s
        )
        # The wire stamp to publish with this tick's setpoint: the leader's
        # capture time for streamed samples, None (meaning now) for plans.
        self.arm_tick_stamp_s: float | None = None
        self.gripper_tick_stamp_s: float | None = None

        self._commanded_arm: tuple[float, ...] | None = None
        self._commanded_gripper: float | None = None
        # The last command the follower actually received; anchors bind to
        # delivered state, never to a step nobody executed.
        self._delivered_arm: tuple[float, ...] | None = None
        self._delivered_gripper: float | None = None
        self._arm_plan: ArmPlan | None = None
        self._gripper_plan: GripperPlan | None = None
        # Captured per tick so completion and publish failure bind to the
        # plan the sample came from, not whatever is installed later.
        self._arm_tick_plan: ArmPlan | None = None
        self._arm_completing: ArmPlan | None = None
        self._gripper_tick_plan: GripperPlan | None = None
        self._gripper_completing: GripperPlan | None = None
        self._arm_busy = False
        self._gripper_busy = False
        self._ik_failing = Latch()
        self._out_of_reach = Latch()

    # -------------------------------------------------------------- busy slots

    def try_claim_arm(self) -> bool:
        """Single-flight admission for the arm: a second arm goal is rejected
        rather than queued, and streamed arm input is ignored while one runs.
        The gripper claims independently."""
        if self._arm_busy:
            return False
        self._arm_busy = True
        return True

    def try_claim_gripper(self) -> bool:
        if self._gripper_busy:
            return False
        self._gripper_busy = True
        return True

    # ------------------------------------------------------------------ anchors

    @property
    def follower_state_timeout_s(self) -> float:
        """The liveness window measured state must beat."""
        return STALE_FOLLOWER_TIMEOUT_S

    @property
    def _upstream_arm(self) -> LatestValue:
        """The holder the configured mode drives the arm from: it owns both
        the streamed target and the capture stamp that rides downstream with
        it."""
        if self._config.upstream_mode is UpstreamMode.JOINTS:
            return self.leader_joints
        return self.leader_pose

    def arm_anchor(self) -> tuple[float, ...] | None:
        """Where arm motion continues from: the last delivered command for
        continuity, else the measured position. None without fresh follower
        state, whatever stands: a limb nobody can see must not be driven."""
        measured = self.measured_joints.fresh(STALE_FOLLOWER_TIMEOUT_S)
        if measured is None:
            return None
        return self._delivered_arm if self._delivered_arm is not None else measured

    def gripper_anchor(self) -> float | None:
        measured = self.measured_gripper.fresh(STALE_FOLLOWER_TIMEOUT_S)
        if measured is None:
            return None
        return self._delivered_gripper if self._delivered_gripper is not None else measured

    # ------------------------------------------------------------------- plans

    def adopt_arm_plan(self, profile: Profile) -> ArmPlan:
        """Install an admitted action's trajectory. Streamed input already in
        flight must not re-target the arm when the plan ends, so the arm's
        leader holders are dropped here and again at release."""
        self._drop_arm_leader_input()
        self._arm_plan = ArmPlan(profile=profile)
        return self._arm_plan

    def adopt_gripper_plan(self, target: float, timeout_s: float) -> GripperPlan:
        self.leader_gripper.clear()
        self._gripper_plan = GripperPlan(target=target, timeout_s=timeout_s)
        return self._gripper_plan

    def abort_arm_plan(self, plan: ArmPlan) -> None:
        """Cut the trajectory: the stream goes silent at the sample where it
        stood, so the follower keeps holding its last received setpoint."""
        if self._arm_plan is plan:
            plan.frozen = plan.last_sample(time.monotonic())
            self._commanded_arm = plan.frozen
            self._arm_plan = None
        plan.aborted = True
        plan.done.set()

    def abort_gripper_plan(self, plan: GripperPlan) -> None:
        if self._gripper_plan is plan:
            plan.frozen = self._commanded_gripper
            self._gripper_plan = None
        plan.aborted = True
        plan.done.set()

    def arm_published(self, delivered: bool, command: tuple[float, ...] | None) -> None:
        """The outcome of publishing this tick's arm setpoint, and the value
        that went out. A plan completes only once its final sample is
        delivered, and an undelivered sample fails the plan it belonged to:
        wall clock alone never declares a goal done. The delivered value is
        passed rather than re-read, so an abort landing during the publish
        cannot make the next step anchor on a sample nobody received."""
        plan = self._arm_tick_plan
        completing = self._arm_completing
        self._arm_tick_plan = None
        self._arm_completing = None
        if not delivered:
            # Plan or stream alike: an undelivered command must not stand as
            # the next anchor; the governor must measure its step from what
            # the follower actually received.
            self._commanded_arm = None
            self._delivered_arm = None
            if plan is None:
                return
            if self._arm_plan is plan:
                self._arm_plan = None
            plan.frozen = plan.last_sample(time.monotonic())
            plan.failed = "downstream setpoint publish failed mid-goal"
            plan.done.set()
            return
        assert command is not None, "a delivered setpoint must name the value that went out"
        self._delivered_arm = command
        if plan is not None and completing is not None:
            completing.done.set()

    def gripper_published(self, delivered: bool, command: float | None) -> None:
        plan = self._gripper_tick_plan
        completing = self._gripper_completing
        self._gripper_tick_plan = None
        self._gripper_completing = None
        if not delivered:
            frozen = self._commanded_gripper
            self._commanded_gripper = None
            self._delivered_gripper = None
            if plan is None:
                return
            if self._gripper_plan is plan:
                self._gripper_plan = None
            plan.frozen = frozen
            plan.failed = "downstream setpoint publish failed mid-goal"
            plan.done.set()
            return
        assert command is not None, "a delivered setpoint must name the value that went out"
        self._delivered_gripper = command
        if plan is not None and completing is not None:
            completing.done.set()

    def release_arm(self) -> None:
        """Called by the action layer once an arm goal reaches a terminal
        state. The standing command resets too: the arm may have been
        displaced during the goal window, so the next stream must re-anchor
        on the measured position exactly like the silence path."""
        self._arm_busy = False
        self._commanded_arm = None
        self._delivered_arm = None
        self._drop_arm_leader_input()

    def release_gripper(self) -> None:
        self._gripper_busy = False
        self._commanded_gripper = None
        self._delivered_gripper = None
        self.leader_gripper.clear()

    def _drop_arm_leader_input(self) -> None:
        self.leader_joints.clear()
        self.leader_pose.clear()

    def _fail_arm_plan(self, plan: ArmPlan, reason: str) -> None:
        plan.frozen = plan.last_sample(time.monotonic())
        self._arm_plan = None
        self._commanded_arm = None
        self._delivered_arm = None
        plan.failed = reason
        plan.done.set()

    def _fail_gripper_plan(self, plan: GripperPlan, reason: str) -> None:
        self._gripper_plan = None
        plan.frozen = self._commanded_gripper
        self._commanded_gripper = None
        self._delivered_gripper = None
        plan.failed = reason
        plan.done.set()

    # -------------------------------------------------------------------- tick

    def arm_tick(self, now: float) -> tuple[float, ...] | None:
        """The joint setpoint to publish this tick, or None for silence. The
        caller reports each returned setpoint's fate through arm_published."""
        self._arm_tick_plan = None
        self._arm_completing = None
        self.arm_tick_stamp_s = None
        follower_live = (
            self.measured_joints.fresh(STALE_FOLLOWER_TIMEOUT_S) is not None
        )
        plan = self._arm_plan
        if plan is not None:
            if not follower_live:
                # A move in the dark can neither progress nor be verified.
                self._fail_arm_plan(plan, "follower state went stale mid-goal")
                return None
            if plan.started_monotonic is None:
                plan.started_monotonic = now
            t = now - plan.started_monotonic
            self._commanded_arm = plan.profile.sample(t)
            self._arm_tick_plan = plan
            if plan.profile.done(t):
                self._arm_plan = None
                # Streamed input that arrived mid-plan must not re-target the
                # arm on the very next tick; the leader re-engages fresh.
                # Completion waits for the final sample's delivery.
                self._drop_arm_leader_input()
                self._arm_completing = plan
            return self._commanded_arm

        if self._arm_busy:
            # A goal is in flight (or just ended, until release runs):
            # streamed input stays ignored so an action never fights a live
            # teleop stream. Release resets the anchors.
            return None
        anchor = self.arm_anchor()
        desired = self._streamed_arm_target(anchor)
        if desired is None or anchor is None:
            # Silence (leader or follower) resets the standing command; the
            # next input re-engages from measured under the governor.
            self._commanded_arm = None
            self._delivered_arm = None
            return None
        # Within the EE caps the leader's value goes downstream as-is.
        self._commanded_arm = self._governor.govern(anchor, desired)
        self.arm_tick_stamp_s = self._upstream_arm.wire_timestamp_s
        return self._commanded_arm

    def gripper_tick(self, now: float) -> float | None:
        self._gripper_tick_plan = None
        self._gripper_completing = None
        self.gripper_tick_stamp_s = None
        plan = self._gripper_plan
        if plan is not None:
            anchor = self.gripper_anchor()
            if anchor is None:
                self._fail_gripper_plan(plan, "follower state went stale mid-goal")
                return None
            # rate_step returns the target itself once within one tick's
            # cap, so completion is exact equality, never a snap past it.
            self._commanded_gripper = rate_step(anchor, plan.target, self._gripper_tick_cap)
            self._gripper_tick_plan = plan
            if self._commanded_gripper == plan.target:
                self._gripper_plan = None
                self.leader_gripper.clear()
                # Completion waits for the target sample's delivery.
                self._gripper_completing = plan
            return self._commanded_gripper

        if self._gripper_busy:
            # See arm_tick: goals own the wire until release resets anchors.
            return None
        desired = self.leader_gripper.fresh(STALE_LEADER_TIMEOUT_S)
        if desired is None:
            self._commanded_gripper = None
            self._delivered_gripper = None
            return None
        anchor = self.gripper_anchor()
        if anchor is None:
            self._commanded_gripper = None
            self._delivered_gripper = None
            return None
        # Within one tick's cap the leader's opening passes through exactly.
        self._commanded_gripper = rate_step(anchor, desired, self._gripper_tick_cap)
        self.gripper_tick_stamp_s = self.leader_gripper.wire_timestamp_s
        return self._commanded_gripper

    def _streamed_arm_target(self, seed: tuple[float, ...] | None) -> tuple[float, ...] | None:
        """The streamed target, or None for silence. The pose path is best
        effort: an out-of-reach pose tracks the workspace boundary through
        the bounded streaming solver, seeded on the anchor the caller
        already resolved, and its solution is limit-clamped and rate-stepped
        per joint from that anchor."""
        streamed = self._upstream_arm.fresh(STALE_LEADER_TIMEOUT_S)
        if self._config.upstream_mode is UpstreamMode.JOINTS:
            # Unclamped, like the reference teleop; the follower's servo
            # EPROM limits are the travel guard.
            return streamed
        if streamed is None or seed is None:
            return None
        position, orientation = streamed
        # Targets are clipped into the reachable ball before the solver sees
        # them, so it is never asked to chase an out-of-reach pose.
        clamped_position = self._reach.clamp(position)
        if clamped_position is position:
            self._out_of_reach.clear()
        else:
            self._out_of_reach.trip("pose target outside reach; tracking the workspace boundary")
        solution = self._kinematics.inverse_kinematics_streaming(
            seed, clamped_position, orientation
        )
        if solution is None:
            self._ik_failing.trip("streamed pose corrupted the solver output; holding")
            return None
        self._ik_failing.clear()
        clamped = self._limits.clamp(solution)
        return tuple(
            rate_step(anchor, target, cap)
            for anchor, target, cap in zip(seed, clamped, self._pose_joint_tick_caps, strict=True)
        )
