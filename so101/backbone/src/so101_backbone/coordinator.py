"""The per-tick motion authority: one loop resolves what the follower is told.

Precedence per limb per tick: an admitted action plan owns the limb (streamed
input is ignored while one runs), and a goal is admitted only while no leader
target arrived within the deadman, so a streaming leader keeps the limb;
otherwise a fresh streamed leader target passes through under the
end-effector-speed governor (the joints-led
stream's only limiter; the pose-mode stream's IK output is additionally
rate-stepped per joint); otherwise silence, and the follower holds. The
arm and the gripper claim independently, so a long arm move never freezes
gripper teleop and vice versa. An arm plan owns the arm past the planned end
of its trajectory: it keeps commanding its end sample until the measured
joints, or the measured grasp point, stand within the goal's settle
tolerance of the target, or the goal's settle timeout fails it. A gripper
plan keeps commanding its target past the planned end of its travel until
the jaws stand still, wherever they stand, or until the goal's settle
timeout, and completes either way.

Fresh follower state gates everything: with the measured stream stale no limb
gets a setpoint, an active plan fails, and the standing command resets. The
first tick after any silence re-engages from the measured position, governed
by the same caps, exactly like every other tick.

The stop service ends every plan installed, whoever admitted it: each plan
is cut where its stream stands, as a cancel cuts it, and carries the stop's
message for its goal to end with.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import ClassVar

from control_core_py.minimum_jerk import Profile
from control_core_py.runtime import Latch
from so101_description.limbs import ARM_LIMB, GRIPPER_LIMB
from so101_description.limits import JointLimits
from so101_description.transforms import relative_rotation_rad

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
# The failure of a move in flight once the follower's measured state is stale.
STALE_FOLLOWER_REFUSAL = "the follower stopped reporting"
# The jaws stand still once every opening the follower delivered over
# GRIPPER_STILL_WINDOW_S lies within GRIPPER_STILL_BAND of full jaw travel of
# the others: the band and window the limb_motion contract states.
GRIPPER_STILL_BAND = 0.01
GRIPPER_STILL_WINDOW_S = 0.25


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
        # When the value last arrived; None until one has.
        self._stamp: float | None = None
        # How many values have arrived: a reader that saw this count tells a
        # value delivered since from the one it already read.
        self.deliveries = 0
        # The producer's capture timestamp (wire seconds), forwarded so the
        # downstream stamp names when the leader read the sample, not when
        # this node relayed it.
        self.wire_timestamp_s: float | None = None

    def set(self, value, wire_timestamp_s: float | None = None) -> None:
        self._value = value
        self._stamp = time.monotonic()
        self.deliveries += 1
        self.wire_timestamp_s = wire_timestamp_s

    def fresh(self, timeout_s: float):
        if self._value is None or not self.set_within(timeout_s):
            return None
        return self._value

    def set_within(self, timeout_s: float) -> bool:
        """Whether a value arrived within `timeout_s`, cleared since or not."""
        return self._stamp is not None and time.monotonic() - self._stamp <= timeout_s

    def clear(self) -> None:
        self._value = None


@dataclass(frozen=True)
class JointSettle:
    """How a joint move ends past the planned end of its trajectory: each
    measured joint within its own tolerance (rad, one per joint in the
    arm's joint order) of its target, within timeout_s."""

    tolerances_rad: tuple[float, ...]
    timeout_s: float
    completion: ClassVar[str] = "trajectory complete"

    def shortfall(self, measured, target, _kinematics) -> str | None:
        """How far the joint farthest outside its tolerance still stands
        from `target`, or None once every joint is within its tolerance."""
        errors = [abs(m - t) for m, t in zip(measured, target, strict=True)]
        excesses = [e - bar for e, bar in zip(errors, self.tolerances_rad, strict=True)]
        worst = max(range(len(excesses)), key=excesses.__getitem__)
        if excesses[worst] <= 0.0:
            return None
        return f"still {errors[worst]:.3f} rad from the target at joint {worst + 1}"


@dataclass(frozen=True)
class PoseSettle:
    """How a pose move ends past the planned end of its trajectory: the
    measured grasp point within the tolerances of the commanded pose, within
    timeout_s. The position is judged as the Euclidean distance, the
    orientation as the geodesic angle."""

    position: tuple[float, ...]
    orientation: tuple[float, ...]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    timeout_s: float
    completion: ClassVar[str] = "cartesian move complete"

    def shortfall(self, measured, _target, kinematics) -> str | None:
        """How far the grasp point of the `measured` joints still stands from
        the commanded pose, or None once it is within both tolerances."""
        position, orientation = kinematics.forward_kinematics(measured)
        distance_m = math.dist(position, self.position)
        angle_rad = relative_rotation_rad(orientation, self.orientation)
        if (
            distance_m <= self.position_tolerance_m
            and angle_rad <= self.orientation_tolerance_rad
        ):
            return None
        return f"still {distance_m * 1000:.1f} mm / {angle_rad:.2f} rad from the target"


@dataclass(frozen=True)
class PostureEnd:
    """How a posture move ends: at the planned end of its trajectory,
    judging nothing, with `completion` as its message."""

    completion: str

    def shortfall(self, _measured, _target, _kinematics) -> None:
        return None


@dataclass
class ArmPlan:
    profile: Profile
    # How the plan ends past the planned end of its profile, judged on the
    # measured joints each tick, and the message it completes with.
    settle: JointSettle | PoseSettle | PostureEnd
    done: asyncio.Event = field(default_factory=asyncio.Event)
    started_monotonic: float | None = None
    aborted: bool = False
    # Set when the coordinator cut the plan short for a cause the goal must
    # report (follower staleness, a dead downstream wire).
    failed: str | None = None
    # The sample the stream stood at when the plan was cut; last_sample
    # reports it instead of re-evaluating the profile at a later clock.
    frozen: tuple[float, ...] | None = None
    # The stop service's message when it cut the plan: the goal ends as
    # cancelled with it.
    stopped: str | None = None

    def last_sample(self, now: float) -> tuple[float, ...]:
        if self.frozen is not None:
            return self.frozen
        if self.started_monotonic is None:
            return self.profile.start
        return self.profile.sample(now - self.started_monotonic)


@dataclass(frozen=True)
class StillRun:
    """A run of openings the follower delivered that all lie within
    GRIPPER_STILL_BAND of one another: the lowest and the highest of them,
    and when the first was measured."""

    low: float
    high: float
    since: float

    @classmethod
    def starting(cls, opening: float, now: float) -> StillRun:
        return cls(low=opening, high=opening, since=now)

    def extended(self, opening: float, now: float) -> StillRun:
        """This run with `opening` measured at `now` added, or a new run from
        it when it lies farther than the band from an opening of this one."""
        low, high = min(self.low, opening), max(self.high, opening)
        if high - low > GRIPPER_STILL_BAND:
            return StillRun.starting(opening, now)
        return StillRun(low=low, high=high, since=self.since)


@dataclass
class GripperPlan:
    target: float
    # How long past the planned end the plan waits for the jaws to stand
    # still before it completes wherever they stand.
    settle_timeout_s: float
    # The opening the ramp starts from: the gripper anchor at admission.
    chase_base: float
    done: asyncio.Event = field(default_factory=asyncio.Event)
    # The planned end: the first tick that found the commanded opening on
    # the target. None while the ramp travels.
    planned_end: float | None = None
    # The follower's delivery count the last tick read, which tells a fresh
    # opening from a cached one.
    read_delivery: int | None = None
    # The run of openings delivered since the planned end that lie within
    # the still band of one another; None before the first.
    still: StillRun | None = None
    aborted: bool = False
    failed: str | None = None
    # The opening the ramp stood at when the plan was cut, for reporting.
    frozen: float | None = None
    # The opening measured when the plan completed, which it reports.
    completed_on: float | None = None
    # How long past the planned end the settle timeout completed the plan,
    # before the jaws stood still; None when they stood still.
    still_moving_after_s: float | None = None
    # The stop service's message when it cut the plan: the goal ends as
    # cancelled with it.
    stopped: str | None = None

    def settles(self, measured: float, fresh: bool, since_end_s: float, now: float) -> bool:
        """Whether a tick `since_end_s` past the planned end, reading
        `measured`, completes the plan: once the jaws stand still, judged on
        `fresh` deliveries alone, the openings delivered over a still window
        lying within the still band of one another, or once the settle
        timeout runs out, which records how long past the planned end that
        was."""
        if fresh:
            self.still = (
                StillRun.starting(measured, now)
                if self.still is None
                else self.still.extended(measured, now)
            )
            if now - self.still.since >= GRIPPER_STILL_WINDOW_S:
                return True
        if since_end_s < self.settle_timeout_s:
            return False
        self.still_moving_after_s = since_end_s
        return True


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
        # How many stops have run, and the message of the last one: a goal
        # whose plan was not installed when a stop ran reads them to end as
        # stopped instead of installing its plan afterwards.
        self.stop_count = 0
        self.last_stop: str | None = None

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

    @property
    def arm_busy(self) -> bool:
        """Whether an arm goal holds the arm, from admission to release."""
        return self._arm_busy

    def leader_streams_arm(self) -> bool:
        """True while a leader's arm command arrived within the deadman. Reads
        the arrival time, which goal boundaries do not clear, so a leader
        keeps the arm across them."""
        return self._upstream_arm.set_within(STALE_LEADER_TIMEOUT_S)

    def leader_streams_gripper(self) -> bool:
        """Whether a leader's gripper command arrived within the deadman."""
        return self.leader_gripper.set_within(STALE_LEADER_TIMEOUT_S)

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

    @property
    def commanded_arm(self) -> tuple[float, ...] | None:
        """The joints commanded last, or None while the wire is silent."""
        return self._commanded_arm

    # ------------------------------------------------------------------- plans

    def adopt_arm_plan(
        self, profile: Profile, settle: JointSettle | PoseSettle | PostureEnd
    ) -> ArmPlan:
        """Install an admitted action's trajectory and its settle. Streamed
        input already in flight must not re-target the arm when the plan
        ends, so the arm's leader holders are dropped here and again at
        release."""
        self._drop_arm_leader_input()
        self._arm_plan = ArmPlan(profile=profile, settle=settle)
        return self._arm_plan

    def adopt_gripper_plan(
        self, target: float, settle_timeout_s: float, chase_base: float
    ) -> GripperPlan:
        self.leader_gripper.clear()
        self._gripper_plan = GripperPlan(
            target=target, settle_timeout_s=settle_timeout_s, chase_base=chase_base
        )
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
            plan.frozen = self._ramp_stands(plan)
            self._gripper_plan = None
        plan.aborted = True
        plan.done.set()

    def stop_all(self, message: str) -> list[str]:
        """Cut every installed plan for the stop service, whoever admitted
        it, each as a cancel cuts it and carrying `message` for its goal to
        end with. The names of the limbs whose plan was cut; the busy slots
        release with the goals, as after a cancel."""
        self.stop_count += 1
        self.last_stop = message
        stopped = []
        arm_plan = self._arm_plan
        if arm_plan is not None:
            arm_plan.stopped = message
            self.abort_arm_plan(arm_plan)
            stopped.append(ARM_LIMB)
        gripper_plan = self._gripper_plan
        if gripper_plan is not None:
            gripper_plan.stopped = message
            self.abort_gripper_plan(gripper_plan)
            stopped.append(GRIPPER_LIMB)
        return stopped

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
        """The outcome of publishing this tick's gripper setpoint, and the
        value that went out. An undelivered sample fails the plan it belonged
        to. A plan at its planned end completes on the delivery of that
        tick's target."""
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
        if completing is not None:
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

    def _ramp_stands(self, plan: GripperPlan) -> float:
        """Where the ramp of `plan` stands: the opening commanded last, or
        the chase base before its first tick."""
        return self._commanded_gripper if self._commanded_gripper is not None else plan.chase_base

    def _fail_gripper_plan(self, plan: GripperPlan, reason: str) -> None:
        self._gripper_plan = None
        plan.frozen = self._ramp_stands(plan)
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
        measured = self.measured_joints.fresh(STALE_FOLLOWER_TIMEOUT_S)
        plan = self._arm_plan
        if plan is not None:
            if measured is None:
                # A move in the dark can neither progress nor be verified.
                self._fail_arm_plan(plan, STALE_FOLLOWER_REFUSAL)
                return None
            if plan.started_monotonic is None:
                plan.started_monotonic = now
            t = now - plan.started_monotonic
            self._commanded_arm = plan.profile.sample(t)
            self._arm_tick_plan = plan
            if plan.profile.done(t):
                return self._end_arm_plan(plan, t, measured)
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

    def _end_arm_plan(self, plan: ArmPlan, t_s: float, measured) -> tuple[float, ...] | None:
        """A tick at or after the planned end of `plan`, its end sample
        commanded. The plan completes on the delivery of that sample when its
        settle is met on the `measured` joints; it fails
        once the settle has timed out, with the shortfall measured then, and
        that tick publishes nothing; otherwise it keeps commanding the end
        sample."""
        settle = plan.settle
        shortfall = settle.shortfall(measured, plan.profile.end, self._kinematics)
        if shortfall is None:
            self._arm_plan = None
            # Streamed input that arrived mid-plan must not re-target the
            # arm on the very next tick; the leader re-engages fresh.
            self._drop_arm_leader_input()
            self._arm_completing = plan
            return self._commanded_arm
        since_end_s = t_s - plan.profile.duration_s
        if since_end_s < settle.timeout_s:
            return self._commanded_arm
        self._arm_tick_plan = None
        self._fail_arm_plan(plan, f"{shortfall}, {since_end_s:.1f} s after the planned end")
        return None

    def gripper_tick(self, now: float) -> float | None:
        """The opening to publish this tick, or None for silence. The caller
        reports each returned opening's fate through gripper_published.

        A plan ramps the commanded opening to the target at the gripper rate
        cap, to its planned end, the first tick that finds the command on the
        target, and keeps commanding the target past it; it completes on the
        delivery of the tick that finds the jaws standing still, or of the
        tick its settle timeout runs out on."""
        self._gripper_tick_plan = None
        self._gripper_completing = None
        self.gripper_tick_stamp_s = None
        plan = self._gripper_plan
        if plan is not None:
            anchor = self.gripper_anchor()
            measured = self.measured_gripper.fresh(STALE_FOLLOWER_TIMEOUT_S)
            if anchor is None or measured is None:
                self._fail_gripper_plan(plan, STALE_FOLLOWER_REFUSAL)
                return None
            delivery = self.measured_gripper.deliveries
            fresh = delivery != plan.read_delivery
            plan.read_delivery = delivery
            # The planned end is the first tick that finds the opening
            # commanded last on the target: the ramp lands by exact equality.
            if plan.planned_end is None and anchor == plan.target:
                plan.planned_end = now
            if plan.planned_end is not None and plan.settles(
                measured, fresh, now - plan.planned_end, now
            ):
                self._gripper_plan = None
                # Streamed input that arrived mid-plan must not re-target the
                # gripper once the goal releases it.
                self.leader_gripper.clear()
                plan.completed_on = measured
                self._gripper_completing = plan
            # rate_step returns the target itself once within one tick's
            # cap, so the ramp lands by exact equality, never a snap past it.
            self._commanded_gripper = rate_step(anchor, plan.target, self._gripper_tick_cap)
            self._gripper_tick_plan = plan
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
