import math
import time

import pytest
from conftest import MEASURED, WIDE_LIMITS, WIDE_REACH, gripper_step, make_config
from control_core_py.minimum_jerk import plan

from so101_backbone.coordinator import (
    GRIPPER_STILL_BAND,
    GRIPPER_STILL_WINDOW_S,
    STALE_FOLLOWER_REFUSAL,
    STALE_FOLLOWER_TIMEOUT_S,
    Coordinator,
    PostureEnd,
    StillRun,
    JointSettle,
    PoseSettle,
)
from so101_backbone.params import UpstreamMode

FAR_TARGET = (1.0, 1.1, 1.2, 1.3, 1.4)


def make_coordinator(fake_kinematics, reach=WIDE_REACH, **config_overrides):
    return Coordinator(make_config(**config_overrides), fake_kinematics, WIDE_LIMITS, reach)


def adopt_gripper(c, target, settle_timeout_s=5.0):
    """Adopt a gripper plan to `target` on `c` which waits up to
    `settle_timeout_s` past the planned end of its ramp for the jaws to stand
    still. Under the transparent rate the tests run at the first sample lands
    the command on the target, and the next tick is the planned end."""
    return c.adopt_gripper_plan(target, settle_timeout_s, c.gripper_anchor())


def test_silence_publishes_nothing(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    assert c.arm_tick(time.monotonic()) is None
    assert c.gripper_tick(time.monotonic()) is None


def test_stream_passes_the_leader_target_through_unmodified(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    # Under the caps the governed step IS the leader's tuple, the same
    # object: no clamp, no walk, no arithmetic residue.
    assert c.arm_tick(time.monotonic()) is FAR_TARGET


def test_stream_without_measured_state_stays_silent(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is None


def test_leader_silence_goes_quiet_then_resumes_pass_through(fake_kinematics, monkeypatch):
    monkeypatch.setattr("so101_backbone.coordinator.STALE_LEADER_TIMEOUT_S", 0.01)
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is not None
    time.sleep(0.02)  # leader input ages past the deadman gate
    assert c.arm_tick(time.monotonic()) is None
    # Resume is the reference behavior: the fresh leader value goes through
    # (the governor alone may slow a large re-engage step).
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is FAR_TARGET


def test_plan_owns_the_arm_and_drops_mid_plan_stream_input(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 0.05, (100.0,) * 5), PostureEnd(""))

    c.leader_joints.set((9.0,) * 5)  # streamed input during the plan
    start = time.monotonic()
    while not adopted.done.is_set():
        c.measured_joints.set(MEASURED)  # the live follower keeps streaming
        sample = c.arm_tick(time.monotonic())
        if sample is not None:
            c.arm_published(True, sample)
        time.sleep(0.001)
        assert time.monotonic() - start < 2.0
    # The plan landed on its end target.
    assert c.arm_anchor() == FAR_TARGET
    # While the goal is still claimed, stream input stays ignored.
    c.leader_joints.set((9.0,) * 5)
    assert c.arm_tick(time.monotonic()) is None
    c.release_arm()
    # After release the drained mid-plan input is gone; silence until fresh.
    assert c.arm_tick(time.monotonic()) is None


def test_abort_freezes_where_the_trajectory_stands(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5), PostureEnd(""))
    # Run the trajectory well past its start before cutting it, so freezing
    # where it stands is distinguishable from reporting where it began.
    start = time.monotonic()
    sample = None
    while time.monotonic() - start < 0.05:
        # The live follower keeps streaming; without this the loop can outrun
        # STALE_FOLLOWER_TIMEOUT_S under a scheduling pause, arm_tick fails the
        # plan, and the test dies on a delivery assertion instead of the
        # timing it means to check.
        c.measured_joints.set(MEASURED)
        sample = c.arm_tick(time.monotonic())
        assert sample is not None
        c.arm_published(True, sample)
    assert sample is not None
    c.abort_arm_plan(adopted)
    assert adopted.done.is_set()
    assert adopted.aborted
    # The cut freezes on the sample the trajectory stood at: strictly between
    # where it began and where it was going, not either end.
    assert adopted.frozen is not None
    for stood, began, going in zip(adopted.frozen, MEASURED, FAR_TARGET, strict=True):
        assert began < stood < going


def test_busy_slots_are_single_flight_and_independent(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    assert c.try_claim_arm()
    assert not c.try_claim_arm()
    # The gripper claims independently of the arm.
    assert c.try_claim_gripper()
    assert not c.try_claim_gripper()
    c.release_arm()
    assert c.try_claim_arm()
    c.release_gripper()
    assert c.try_claim_gripper()


def test_gripper_plan_ramps_then_completes_once_the_jaws_stand_still(
    fake_kinematics, follower_never_stale
):
    c = make_coordinator(fake_kinematics, max_gripper_rate_frac_s=2.0)
    c.measured_gripper.set(0.0)
    assert c.try_claim_gripper()
    # The ramp from 0 to 1 at 2.0/s lands at 0.5 s; the tick after is the
    # planned end.
    adopted = adopt_gripper(c, 1.0)
    # A follower that tracks every command a tick late: 2.0/s at 100 Hz is
    # 0.02 per tick, so the ramp to 1.0 takes 50 ticks, and the plan runs
    # through the 49 before the planned end.
    last = 0.0
    for tick in range(49):
        value = gripper_step(c, last, tick * 0.01)
        assert 0.0 < value - last <= 0.02 + 1e-12
        last = value
        assert not adopted.done.is_set()
    # The command lands with the jaws at 0.98; the next tick finds it on the
    # target, the planned end, with the jaws at 1.0, and they stand still
    # there a window later.
    assert gripper_step(c, last, 0.5) == 1.0
    assert gripper_step(c, 1.0, 0.625) == 1.0
    assert not adopted.done.is_set()
    assert gripper_step(c, 1.0, 0.875) == 1.0
    assert adopted.done.is_set()
    assert adopted.failed is None
    assert adopted.completed_on == 1.0
    assert adopted.still_moving_after_s is None
    c.release_gripper()


def test_a_gripper_plan_owns_the_gripper_through_its_travel(fake_kinematics, follower_never_stale):
    # Transparent rate: the first sample out is the target itself, while the
    # jaws have not moved yet; the next tick is the planned end.
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    assert gripper_step(c, 1.0, 0.0) == 0.5
    assert not adopted.done.is_set(), "a delivered target sample does not end the plan"
    # The jaws lag: they close 0.05 every 50 ms. All the way the plan keeps
    # the gripper: it commands the target, ignores the leader stream, and
    # holds the busy slot against a second goal.
    for tick in range(1, 10):
        c.leader_gripper.set(1.0)
        assert gripper_step(c, 1.0 - 0.05 * tick, 0.05 * tick) == 0.5
        assert not adopted.done.is_set()
        assert not c.try_claim_gripper()
    # The jaws stand at 0.52 from 0.5 s on: a still window later the plan
    # completes on the delivery of that tick's target.
    assert gripper_step(c, 0.52, 0.5) == 0.5
    assert not adopted.done.is_set()
    assert gripper_step(c, 0.52, 0.75) == 0.5
    assert adopted.done.is_set()
    assert adopted.failed is None
    assert adopted.completed_on == 0.52
    # Until the action layer releases the slot the wire stays silent.
    assert gripper_step(c, 0.5, 1.0) is None
    assert not c.try_claim_gripper()
    c.release_gripper()


def test_a_delivery_farther_than_the_still_band_restarts_the_window(
    fake_kinematics, follower_never_stale
):
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    # The command lands on the first tick; the jaws stand at 0.6.
    assert gripper_step(c, 0.6, 0.0) == 0.5
    # The next tick is the planned end, where the run starts with the jaws
    # 0.015 closer, farther than the band from where they stood: a delivery
    # a window after the start does not complete the plan.
    moved = 0.6 - GRIPPER_STILL_BAND * 1.5
    assert gripper_step(c, moved, 0.125) == 0.5
    assert gripper_step(c, moved, 0.25) == 0.5
    assert not adopted.done.is_set()
    # A window after the planned end they still stand there: the plan completes.
    assert gripper_step(c, moved, 0.375) == 0.5
    assert adopted.done.is_set()
    assert adopted.completed_on == moved
    c.release_gripper()


def test_a_cached_opening_never_counts_as_still(fake_kinematics, follower_never_stale):
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    assert gripper_step(c, 0.6, 0.0) == 0.5
    # The planned end, where the run starts on the delivered 0.6.
    assert gripper_step(c, 0.6, 0.25) == 0.5
    # No delivery since: the cached 0.6 stands on the holder, fresh enough
    # to drive the gripper, and is no evidence that the jaws stopped,
    # however long past the window.
    for now in (0.5, 0.75, 1.0):
        assert c.gripper_tick(now) == 0.5
        c.gripper_published(True, 0.5)
        assert not adopted.done.is_set()
    # A fresh delivery within the band completes the plan.
    assert gripper_step(c, 0.6, 1.25) == 0.5
    assert adopted.done.is_set()
    c.release_gripper()


def test_the_settle_timeout_completes_a_plan_whose_jaws_keep_moving(
    fake_kinematics, follower_never_stale
):
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5, settle_timeout_s=1.0)
    for now, opening in ((0.0, 1.0), (0.5, 0.9), (0.75, 0.8), (1.0, 0.7), (1.25, 0.6)):
        assert gripper_step(c, opening, now) == 0.5
        assert not adopted.done.is_set()
    assert gripper_step(c, 0.55, 1.5) == 0.5
    assert adopted.done.is_set()
    assert adopted.failed is None
    assert adopted.completed_on == 0.55
    assert adopted.still_moving_after_s == 1.0
    c.release_gripper()


def test_a_gripper_an_item_holds_short_of_the_target_completes_where_the_jaws_stopped(
    fake_kinematics, follower_never_stale
):
    # An item stops the jaws at 0.8 on the way to 0.5: they stand still
    # there, the plan completes, and the opening the item holds them at is
    # what the result reads.
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    assert gripper_step(c, 1.0, 0.0) == 0.5
    assert gripper_step(c, 0.8, 0.25) == 0.5
    assert not adopted.done.is_set()
    assert gripper_step(c, 0.8, 0.5) == 0.5
    assert adopted.done.is_set()
    assert adopted.failed is None
    assert adopted.completed_on == 0.8
    c.release_gripper()


def test_gripper_plan_with_no_travel_completes_once_the_jaws_stand_still(
    fake_kinematics, follower_never_stale
):
    # The goal names the opening the gripper already holds.
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(0.5)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    # The first tick is the planned end and takes the reference; a delivery
    # a window later completes the plan once that tick's sample is
    # delivered.
    assert gripper_step(c, 0.5, 0.0) == 0.5
    c.measured_gripper.set(0.5)
    assert c.gripper_tick(GRIPPER_STILL_WINDOW_S) == 0.5
    assert not adopted.done.is_set(), "completion waits for delivery"
    c.gripper_published(True, 0.5)
    assert adopted.done.is_set()
    assert adopted.completed_on == 0.5
    c.release_gripper()


def test_a_run_holds_openings_within_the_band_of_one_another():
    run = StillRun.starting(0.8, 0.0)
    jittered = run.extended(0.8 - GRIPPER_STILL_BAND * 0.5, 0.1)
    assert (jittered.low, jittered.high, jittered.since) == (0.8 - GRIPPER_STILL_BAND * 0.5, 0.8, 0.0)
    # An opening within the band of the first but not of the lowest starts
    # a new run: the jaws moved more than the band in all.
    spread = jittered.extended(0.8 + GRIPPER_STILL_BAND * 0.6, 0.2)
    assert (spread.low, spread.high, spread.since) == (0.8 + GRIPPER_STILL_BAND * 0.6,) * 2 + (0.2,)


def test_follower_silence_fails_a_gripper_plan_in_flight(fake_kinematics, follower_never_stale):
    # A frozen last opening must not pass for a measured one.
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    assert gripper_step(c, 1.0, 0.0) == 0.5
    assert gripper_step(c, 0.6, 0.5) == 0.5
    c.measured_gripper.clear()  # no measured opening left to report
    assert c.gripper_tick(1.0) is None
    assert adopted.done.is_set()
    assert adopted.failed == STALE_FOLLOWER_REFUSAL
    assert adopted.completed_on is None
    c.release_gripper()


def test_publish_failure_fails_a_gripper_plan_in_flight(fake_kinematics, follower_never_stale):
    c = make_coordinator(fake_kinematics)
    c.measured_gripper.set(1.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 0.5)
    assert gripper_step(c, 1.0, 0.0) == 0.5
    assert gripper_step(c, 0.6, 0.5) == 0.5
    c.measured_gripper.set(0.6)
    assert c.gripper_tick(0.6) == 0.5
    c.gripper_published(False, None)
    assert adopted.done.is_set()
    assert "publish failed" in adopted.failed
    assert adopted.completed_on is None
    c.release_gripper()


def test_pose_mode_streams_through_ik(fake_kinematics):
    c = make_coordinator(fake_kinematics, upstream_mode=UpstreamMode.POSE)
    c.measured_joints.set(MEASURED)
    c.leader_pose.set(((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)))
    stepped = c.arm_tick(time.monotonic())
    assert stepped is not None
    assert fake_kinematics.solve_calls[0][0] == "stream", (
        "the bounded streaming solver serves teleop"
    )
    # A corrupted solver output holds instead of moving somewhere undefined.
    fake_kinematics.corrupted = True
    c.leader_pose.set(((5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0)))
    assert c.arm_tick(time.monotonic()) is None


def test_streamed_setpoints_carry_the_leader_capture_stamp_in_both_modes(fake_kinematics):
    # The downstream stamp must name when the leader captured the sample, not
    # when this node relayed it, or the wire hides this hop's latency from the
    # follower. Pose mode reads a different upstream slot than joints mode and
    # must forward the stamp from the one it actually read.
    joints = make_coordinator(fake_kinematics)
    joints.measured_joints.set(MEASURED)
    joints.leader_joints.set(FAR_TARGET, wire_timestamp_s=1234.5)
    assert joints.arm_tick(time.monotonic()) is not None
    assert joints.arm_tick_stamp_s == 1234.5

    pose = make_coordinator(fake_kinematics, upstream_mode=UpstreamMode.POSE)
    pose.measured_joints.set(MEASURED)
    pose.leader_pose.set(((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)), wire_timestamp_s=99.5)
    assert pose.arm_tick(time.monotonic()) is not None
    assert pose.arm_tick_stamp_s == 99.5


def test_pose_mode_ik_output_is_rate_stepped_per_joint(fake_kinematics):
    # Near the workspace boundary the streaming solver can flip
    # configurations tick to tick; the stream advances at most one tick's
    # per-joint velocity budget from the anchor toward each solution, and a
    # solution within one step lands exactly.
    c = make_coordinator(fake_kinematics, upstream_mode=UpstreamMode.POSE)
    c.measured_joints.set(MEASURED)
    c.leader_pose.set(((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)))
    # The scripted solution sits 0.1 above MEASURED per joint; 2.0 rad/s at
    # 100 Hz allows 0.02 per tick.
    stepped = c.arm_tick(time.monotonic())
    assert stepped == pytest.approx(tuple(m + 0.02 for m in MEASURED))
    c.arm_published(True, stepped)
    value = stepped
    for _ in range(10):
        if value == fake_kinematics.solution:
            break
        c.measured_joints.set(MEASURED)
        c.leader_pose.set(((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)))
        value = c.arm_tick(time.monotonic())
        c.arm_published(True, value)
    assert value == fake_kinematics.solution


def test_streamed_targets_are_not_limit_clamped(fake_kinematics):
    # The reference teleop forwards the leader verbatim; model limits guard
    # action targets and IK, never the stream. The servo EPROM limits are
    # the physical travel guard.
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set((3.0,) * 5)
    beyond = (9.0,) * 5  # far beyond the 3.1 rad model limit
    c.leader_joints.set(beyond)
    assert c.arm_tick(time.monotonic()) is beyond


def test_release_resets_anchors_to_measured(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 0.01, (100.0,) * 5), PostureEnd(""))
    start = time.monotonic()
    while not adopted.done.is_set():
        c.measured_joints.set(MEASURED)  # the live follower keeps streaming
        sample = c.arm_tick(time.monotonic())
        if sample is not None:
            c.arm_published(True, sample)
        assert time.monotonic() - start < 2.0
    c.release_arm()
    # The goal window may have displaced the arm; the next stream re-engages
    # from measured under the governor, and with transparent caps that is
    # the leader's value straight through.
    displaced = (0.5, 0.5, 0.5, 0.5, 0.5)
    c.measured_joints.set(displaced)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is FAR_TARGET


def test_follower_staleness_silences_the_stream_until_it_returns(fake_kinematics):
    # The liveness window is STALE_FOLLOWER_TIMEOUT_S, independent of the
    # control rate.
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is not None
    time.sleep(STALE_FOLLOWER_TIMEOUT_S * 1.5)  # the follower goes dark
    c.leader_joints.set(FAR_TARGET)
    # No fresh measured state: nothing may be commanded, however live the
    # leader looks.
    assert c.arm_tick(time.monotonic()) is None
    # The follower returns: the stream resumes as pass-through.
    c.measured_joints.set((0.5,) * 5)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is FAR_TARGET


def test_follower_staleness_fails_an_active_arm_plan(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5), PostureEnd(""))
    assert c.arm_tick(time.monotonic()) is not None
    time.sleep(STALE_FOLLOWER_TIMEOUT_S * 1.5)
    assert c.arm_tick(time.monotonic()) is None
    assert adopted.done.is_set()
    assert adopted.failed is not None
    c.release_arm()


def test_follower_staleness_fails_an_active_gripper_plan(fake_kinematics):
    # A binding rate cap keeps the ramp mid-flight across the sleep.
    c = make_coordinator(fake_kinematics, max_gripper_rate_frac_s=2.0)
    c.measured_gripper.set(0.0)
    assert c.try_claim_gripper()
    adopted = adopt_gripper(c, 1.0)
    assert c.gripper_tick(time.monotonic()) is not None
    time.sleep(STALE_FOLLOWER_TIMEOUT_S * 1.5)
    assert c.gripper_tick(time.monotonic()) is None
    assert adopted.done.is_set()
    assert adopted.failed is not None
    c.release_gripper()


def test_publish_failure_fails_the_plan_the_setpoint_came_from(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5), PostureEnd(""))
    assert c.arm_tick(time.monotonic()) is not None
    c.arm_published(False, None)
    assert adopted.done.is_set()
    assert "publish failed" in adopted.failed
    c.release_arm()


def test_publish_failure_with_no_plan_touches_nothing(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is not None
    # A failed streamed publish has no goal to fail; the plan installed
    # between the tick and the outcome must be untouched.
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5), PostureEnd(""))
    c.arm_published(False, None)
    assert not adopted.done.is_set()
    assert adopted.failed is None
    c.release_arm()


def test_final_sample_publish_failure_fails_the_goal(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 0.001, (1000.0,) * 5), PostureEnd(""))
    start = time.monotonic()
    target = None
    while target is None:
        target = c.arm_tick(time.monotonic())
        assert time.monotonic() - start < 2.0
    time.sleep(0.005)
    final = c.arm_tick(time.monotonic())  # past the duration: the endpoint sample
    assert final is not None
    assert not adopted.done.is_set(), "completion must wait for delivery"
    c.arm_published(False, None)
    assert adopted.done.is_set()
    assert "publish failed" in adopted.failed
    c.release_arm()


def test_gripper_streams_while_an_arm_plan_runs(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.measured_gripper.set(0.0)
    assert c.try_claim_arm()
    c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5), PostureEnd(""))
    c.arm_tick(time.monotonic())
    # The operator can still work the gripper during a long arm move.
    c.leader_gripper.set(1.0)
    assert c.gripper_tick(time.monotonic()) is not None


def test_streamed_publish_failure_resets_the_anchor(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is not None
    # The step never reached the follower: the governor must measure its
    # next step from delivered state (here: none), and the retry is still
    # the leader's value under transparent caps.
    c.arm_published(False, None)
    c.leader_joints.set(FAR_TARGET)
    assert c.arm_tick(time.monotonic()) is FAR_TARGET


def test_streamed_gripper_publish_failure_resets_the_anchor(fake_kinematics):
    c = make_coordinator(fake_kinematics, max_gripper_rate_frac_s=2.0)
    c.measured_gripper.set(0.0)
    c.leader_gripper.set(1.0)
    assert c.gripper_tick(time.monotonic()) is not None
    c.gripper_published(False, None)
    c.leader_gripper.set(1.0)
    # Undelivered: the rate cap re-steps from measured, not from the step
    # nobody received.
    retried = c.gripper_tick(time.monotonic())
    assert retried <= 0.02 + 1e-12


def test_admission_during_a_doomed_publish_anchors_on_delivered_state(fake_kinematics):
    from control_core_py.minimum_jerk import plan as make_profile

    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.leader_joints.set(FAR_TARGET)
    stepped = c.arm_tick(time.monotonic())
    assert stepped is not None
    # The publish is in flight when a goal is admitted: the anchor must be
    # what the follower last received (here: nothing delivered yet, so the
    # measured position), never the undelivered step.
    anchor = c.arm_anchor()
    assert anchor == MEASURED
    profile = make_profile(anchor, FAR_TARGET, 1.0, (100.0,) * 5)
    assert profile.start == MEASURED
    c.arm_published(False, None)


def test_a_stop_cuts_every_installed_plan_with_its_message_and_names_the_limbs(fake_kinematics):
    from control_core_py.minimum_jerk import plan as make_profile

    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    c.measured_gripper.set(0.0)
    assert c.stop_all("stopped: nothing yet") == []
    assert (c.stop_count, c.last_stop) == (1, "stopped: nothing yet")

    assert c.try_claim_arm() and c.try_claim_gripper()
    arm_plan = c.adopt_arm_plan(make_profile(MEASURED, FAR_TARGET, 1.0, (100.0,) * 5), PostureEnd(""))
    gripper_plan = adopt_gripper(c, 1.0)
    now = time.monotonic()
    sample = c.arm_tick(now)
    c.arm_published(True, sample)
    opening = c.gripper_tick(now)
    c.gripper_published(True, opening)

    assert c.stop_all("stopped: the operator asked") == ["arm", "gripper"]

    # Each plan is cut where its stream stood, as a cancel cuts it, and
    # carries the stop's message; the wire goes silent. The arm's cut is
    # sampled at the stop's instant, a hair past the tick's.
    assert arm_plan.stopped == "stopped: the operator asked"
    assert arm_plan.aborted and arm_plan.done.is_set()
    assert arm_plan.frozen == pytest.approx(sample, abs=1e-6)
    assert gripper_plan.stopped == "stopped: the operator asked"
    assert gripper_plan.aborted and gripper_plan.done.is_set()
    assert gripper_plan.frozen == opening
    assert c.arm_tick(now + 0.01) is None
    assert c.gripper_tick(now + 0.01) is None
    assert c.stop_count == 2
    # The goals release the slots, as after a cancel; the stop does not.
    assert c.arm_busy


def arm_settle(tolerances_rad=(0.1,) * 5, timeout_s=2.0) -> JointSettle:
    return JointSettle(tolerances_rad=tolerances_rad, timeout_s=timeout_s)


def test_an_arm_plan_keeps_commanding_its_end_sample_until_the_joints_settle(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 1.0, (100.0,) * 5), arm_settle())
    c.arm_published(True, c.arm_tick(0.0))
    # At the planned end the follower still measures the start: the end
    # sample goes out and the plan stays installed.
    c.measured_joints.set(MEASURED)
    assert c.arm_tick(1.0) == FAR_TARGET
    c.arm_published(True, FAR_TARGET)
    assert not adopted.done.is_set()
    # Every joint within 0.1 rad: the plan completes on the delivery of this
    # tick's end sample.
    c.measured_joints.set(tuple(t - 0.05 for t in FAR_TARGET))
    assert c.arm_tick(1.5) == FAR_TARGET
    assert not adopted.done.is_set(), "completion waits for delivery"
    c.arm_published(True, FAR_TARGET)
    assert adopted.done.is_set()
    assert adopted.failed is None
    c.release_arm()


def test_an_arm_plan_outside_its_tolerances_at_the_timeout_fails_naming_the_worst_joint(
    fake_kinematics,
):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    # Joint 3 ends 0.7 rad out against its 0.1 rad; joint 1 ends 0.15 rad
    # out, within its 0.2 rad.
    target = (0.15, 0.15, 0.9, 0.35, 0.45)
    settle = arm_settle(tolerances_rad=(0.2, 0.1, 0.1, 0.1, 0.1), timeout_s=2.0)
    adopted = c.adopt_arm_plan(plan(MEASURED, target, 1.0, (100.0,) * 5), settle)
    c.arm_published(True, c.arm_tick(0.0))
    for now in (1.0, 2.0, 2.9):
        c.measured_joints.set(MEASURED)
        assert c.arm_tick(now) == target
        c.arm_published(True, target)
        assert not adopted.done.is_set()
    c.measured_joints.set(MEASURED)
    assert c.arm_tick(3.0) is None
    assert adopted.done.is_set()
    assert adopted.failed == "still 0.700 rad from the target at joint 3, 2.0 s after the planned end"
    c.release_arm()


def test_a_joint_settle_names_the_joint_farthest_outside_its_own_tolerance():
    settle = arm_settle(tolerances_rad=(0.5, 0.01, 0.1, 0.1, 0.1))
    measured = (0.0,) * 5
    # Joint 1 is 0.4 rad out, inside its 0.5 rad. Joint 2 is 0.05 rad out,
    # 0.04 rad past its bar; joint 3 is 0.12 rad out, 0.02 rad past its bar.
    target = (0.4, 0.05, 0.12, 0.0, 0.0)
    assert settle.shortfall(measured, target, None) == (
        "still 0.050 rad from the target at joint 2"
    )
    assert settle.shortfall(target, target, None) is None


def test_a_pose_settle_judges_the_measured_grasp_point_against_the_commanded_pose(
    fake_kinematics,
):
    goal_position, _orientation = fake_kinematics.forward_kinematics(fake_kinematics.solution)
    settle = PoseSettle(
        position=goal_position,
        orientation=(0.0, 0.0, 0.0, 1.0),
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.2,
        timeout_s=2.0,
    )
    assert settle.shortfall(fake_kinematics.solution, None, fake_kinematics) is None
    assert settle.shortfall((0.0,) * 5, None, fake_kinematics) == (
        "still 16.3 mm / 0.00 rad from the target"
    )
    # The arm stands on the position, turned 44 degrees from the commanded
    # orientation.
    half = math.radians(44.0) / 2.0
    turned = PoseSettle(
        position=goal_position,
        orientation=(math.sin(half), 0.0, 0.0, math.cos(half)),
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.2,
        timeout_s=2.0,
    )
    assert turned.shortfall(fake_kinematics.solution, None, fake_kinematics) == (
        "still 0.0 mm / 0.77 rad from the target"
    )
