import time

import pytest
from so101_description.units import GRIPPER_NAME, JOINT_NAMES

from so101_follower import __main__ as main_mod
from so101_follower.device import DeviceLoop


def make_loop(hardware, stale_s=0.25):
    loop = DeviceLoop(hardware, control_rate_hz=200)
    # Production cadences are constants; tests exercise the loop logic at
    # fast equivalents so misses and staleness resolve in milliseconds.
    loop._health_every_ticks = 4
    loop._stale_timeout_s = stale_s
    return loop


def wait_for(predicate, timeout_s=2.0):
    """Block until the predicate holds, or fail the test. Raising rather than
    returning false matches the harness helper of the same name, so a call
    site copied between the two files cannot silently stop checking."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition never held within {timeout_s}s")


def test_bringup_failure_raises_and_never_reports_ready(fake_hardware):
    fake_hardware.fail_connect = RuntimeError("arm calibration is missing")
    loop = make_loop(fake_hardware)
    with pytest.raises(RuntimeError, match="calibration"):
        loop.start()
    assert not loop.ready()


def test_failed_first_read_fails_bringup_and_releases_the_hardware(fake_hardware):
    fake_hardware.fail_position_reads = 1
    loop = make_loop(fake_hardware)
    with pytest.raises(IOError, match="scripted read failure"):
        loop.start()
    assert not loop.ready()
    # Connect may have enabled torque; a failed launch must release it.
    assert fake_hardware.disconnected


def test_disabled_torque_at_bringup_fails_the_launch(fake_hardware):
    fake_hardware.torque = (True, True, False, True, True, True)
    loop = make_loop(fake_hardware)
    with pytest.raises(RuntimeError, match="torque did not enable at bringup: elbow_flex"):
        loop.start()
    # The port (and whatever torque did enable) is released.
    assert fake_hardware.disconnected


def test_latched_servo_fault_at_bringup_fails_the_launch(fake_hardware):
    fake_hardware.faults = (0, 32, 0, 0, 0, 0)
    loop = make_loop(fake_hardware)
    faulted = r"faults latched at bringup: shoulder_lift \(overload\)"
    with pytest.raises(RuntimeError, match=faulted):
        loop.start()
    assert fake_hardware.disconnected


def test_nonfinite_first_read_fails_bringup(fake_hardware):
    fake_hardware.positions["shoulder_pan"] = float("nan")
    loop = make_loop(fake_hardware)
    with pytest.raises(ValueError, match="non-finite"):
        loop.start()
    # Connect may already have enabled torque, so a refused bringup must
    # still release the arm rather than leave it energised and undriven.
    assert fake_hardware.disconnected


def test_nonfinite_health_at_bringup_fails_the_launch(fake_hardware):
    # The bringup read runs the same finite guard as the loop: a corrupt
    # temperature must not seed the readings that later compare as nominal.
    fake_hardware.temps = (25.0, float("nan"), 25.0, 25.0, 25.0, 25.0)
    loop = make_loop(fake_hardware)
    with pytest.raises(ValueError, match="non-finite"):
        loop.start()
    assert fake_hardware.disconnected


def test_health_only_failure_counts_misses_and_keeps_readings(fake_hardware):
    loop = make_loop(fake_hardware)
    loop.start()
    try:
        wait_for(
            lambda: (h := loop.latest_health()) is not None and len(h.temps_c) == 6
        )
        stamp = loop.latest_health().readings_captured_monotonic
        # Only health reads fail; state reads keep succeeding.
        fake_hardware.fail_health_reads = 10_000
        good_temps = fake_hardware.temps
        # Move the hardware's readings somewhere new. Nothing can read them
        # while the tap holds, so a snapshot showing them would mean the node
        # fabricated a read it never made.
        fake_hardware.temps = tuple(t + 11.0 for t in good_temps)
        wait_for(lambda: loop.latest_health().consecutive_missed_reads >= 3)
        held = loop.latest_health()
        # The last good readings stand under their original capture stamp.
        assert held.temps_c == good_temps
        assert held.readings_captured_monotonic == stamp
    finally:
        loop.stop()


def test_nonfinite_health_read_counts_as_missed_not_poison(fake_hardware):
    loop = make_loop(fake_hardware)
    loop.start()
    try:
        wait_for(
            lambda: (h := loop.latest_health()) is not None and len(h.temps_c) == 6
        )
        good_loads = fake_hardware.loads
        fake_hardware.loads = (float("nan"), *good_loads[1:])
        # A corrupt reading is a missed read: the last good readings stand
        # and nothing non-finite ever enters a snapshot.
        wait_for(lambda: loop.latest_health().consecutive_missed_reads >= 1)
        held = loop.latest_health()
        assert held.load_fractions == good_loads
    finally:
        loop.stop()


def test_state_only_failure_never_flags_health_silent(fake_hardware):
    loop = make_loop(fake_hardware)
    loop.start()
    try:
        wait_for(lambda: loop.latest_health() is not None)
        fake_hardware.fail_position_reads = 10_000
        time.sleep(0.1)
        # State reads failing is state staleness, never health silence.
        assert loop.latest_health().consecutive_missed_reads == 0
    finally:
        loop.stop()


def test_fresh_targets_reach_the_bus(fake_hardware):
    loop = make_loop(fake_hardware)
    loop.start()
    try:
        loop.submit_arm_target((10.0, 0.0, -5.0, 0.0, 2.0))
        loop.submit_gripper_target(40.0)
        wait_for(lambda: any(GRIPPER_NAME in g for g in fake_hardware.written_goals))
        merged = fake_hardware.written_goals[-1]
        assert merged[JOINT_NAMES[0]] == 10.0
        assert merged[GRIPPER_NAME] == 40.0
    finally:
        loop.stop()
    assert fake_hardware.disconnected


def test_stale_targets_stop_goal_writes(fake_hardware):
    loop = make_loop(fake_hardware, stale_s=0.05)
    loop.start()
    try:
        loop.submit_arm_target((1.0, 2.0, 3.0, 4.0, 5.0))
        wait_for(lambda: len(fake_hardware.written_goals) > 0)
        time.sleep(0.15)  # let the target age past the stale timeout
        writes_after_stale = len(fake_hardware.written_goals)
        time.sleep(0.1)
        # Silence is the hold: no further goal writes once the target is stale.
        assert len(fake_hardware.written_goals) == writes_after_stale
    finally:
        loop.stop()


def test_failed_state_reads_never_fabricate_snapshots(fake_hardware):
    loop = make_loop(fake_hardware)
    loop.start()
    try:
        wait_for(lambda: loop.latest_state() is not None)
        fake_hardware.fail_position_reads = 10_000_000
        time.sleep(0.02)  # let any in-flight good read land
        before = loop.latest_state()
        time.sleep(0.1)
        # The last snapshot stands untouched: no new capture stamp appears
        # while every read fails, so its age is the evidence downstream.
        assert loop.latest_state().captured_monotonic == before.captured_monotonic
    finally:
        loop.stop()


async def test_bus_watchdog_terminates_after_the_exit_horizon(fake_hardware, monkeypatch):
    loop = make_loop(fake_hardware)
    loop.start()
    try:
        wait_for(lambda: loop.latest_state() is not None)
        monkeypatch.setattr(main_mod, "_BUS_DEAD_EXIT_S", 0.05)
        monkeypatch.setattr(main_mod, "_BUS_WATCH_PERIOD_S", 0.01)
        fake_hardware.fail_position_reads = 10_000_000
        terminated = []

        class Token:
            def is_cancelled(self):
                return bool(terminated)

            async def cancelled(self):
                import asyncio

                while not terminated:
                    await asyncio.sleep(0.01)

        import asyncio

        await asyncio.wait_for(
            main_mod._watch_bus(loop, Token(), terminate=terminated.append), 5.0
        )
        assert terminated == [1]
        # The watchdog released the hardware before dying.
        assert fake_hardware.disconnected
    finally:
        loop.stop()
