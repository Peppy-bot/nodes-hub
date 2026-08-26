import time

import pytest
from conftest import WIDE_LIMITS, WIDE_REACH, make_config
from control_core_py.minimum_jerk import plan

from so101_backbone.coordinator import STALE_FOLLOWER_TIMEOUT_S, Coordinator
from so101_backbone.params import UpstreamMode

MEASURED = (0.0, 0.1, 0.2, 0.3, 0.4)
FAR_TARGET = (1.0, 1.1, 1.2, 1.3, 1.4)


def make_coordinator(fake_kinematics, reach=WIDE_REACH, **config_overrides):
    return Coordinator(make_config(**config_overrides), fake_kinematics, WIDE_LIMITS, reach)


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
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 0.05, (100.0,) * 5))

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
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5))
    # Run the trajectory well past its start before cutting it, so freezing
    # where it stands is distinguishable from reporting where it began.
    start = time.monotonic()
    sample = None
    while time.monotonic() - start < 0.05:
        sample = c.arm_tick(time.monotonic())
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


def test_gripper_plan_ramps_and_completes(fake_kinematics):
    c = make_coordinator(fake_kinematics, max_gripper_rate_frac_s=2.0)
    c.measured_gripper.set(0.0)
    assert c.try_claim_gripper()
    adopted = c.adopt_gripper_plan(1.0, timeout_s=5.0)
    start = time.monotonic()
    last = 0.0
    while not adopted.done.is_set():
        c.measured_gripper.set(last)  # the live follower keeps streaming
        value = c.gripper_tick(time.monotonic())
        c.gripper_published(True, value)
        # 2.0/s at 100 Hz is 0.02 per tick.
        assert value - last <= 0.02 + 1e-12
        last = value
        assert time.monotonic() - start < 2.0
    assert last == 1.0
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
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 0.01, (100.0,) * 5))
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
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5))
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
    adopted = c.adopt_gripper_plan(1.0, timeout_s=5.0)
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
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5))
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
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5))
    c.arm_published(False, None)
    assert not adopted.done.is_set()
    assert adopted.failed is None
    c.release_arm()


def test_final_sample_publish_failure_fails_the_goal(fake_kinematics):
    c = make_coordinator(fake_kinematics)
    c.measured_joints.set(MEASURED)
    assert c.try_claim_arm()
    adopted = c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 0.001, (1000.0,) * 5))
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
    c.adopt_arm_plan(plan(MEASURED, FAR_TARGET, 10.0, (100.0,) * 5))
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
