from so101_description.units import MOTOR_NAMES

from so101_follower.health import (
    LEVEL_CRITICAL,
    LEVEL_FAULT,
    LEVEL_NOMINAL,
    LEVEL_NOT_REPORTING,
    LEVEL_WARNING,
    TRIP_FRACTION_OF_STALL,
    AlertTracker,
    SustainedLoads,
    assess,
    describe_faults,
)

COOL = tuple(30.0 for _ in MOTOR_NAMES)
IDLE = tuple(0.1 for _ in MOTOR_NAMES)
DRIVEN = tuple(True for _ in MOTOR_NAMES)
NO_FAULTS = tuple(0 for _ in MOTOR_NAMES)


def test_nominal_report():
    report = assess(COOL, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=False)
    assert report.levels == (LEVEL_NOMINAL,) * 6
    assert report.winding_temp_c == COOL
    assert report.stall_fractions == IDLE


def test_temperature_thresholds():
    temps = (30.0, 56.0, 66.0, 30.0, 30.0, 30.0)
    report = assess(temps, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=False)
    assert report.levels[0] == LEVEL_NOMINAL
    assert report.levels[1] == LEVEL_WARNING
    assert report.levels[2] == LEVEL_CRITICAL


def test_sustained_load_thresholds_use_smoothed_value():
    spiky = (0.9,) + (0.1,) * 5
    calm = (0.1,) * 6
    report = assess(COOL, spiky, DRIVEN, NO_FAULTS, calm, bus_silent=False)
    # The instantaneous spike reports in stall_fractions but does not
    # trip a level; only the sustained estimate does.
    assert report.levels == (LEVEL_NOMINAL,) * 6
    assert report.stall_fractions == spiky

    sustained_hot = (0.85,) + (0.1,) * 5
    report = assess(COOL, spiky, DRIVEN, NO_FAULTS, sustained_hot, bus_silent=False)
    assert report.levels[0] == LEVEL_CRITICAL


def test_silent_bus_empties_readings_and_flags_every_motor():
    report = assess(COOL, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=True)
    assert report.levels == (LEVEL_NOT_REPORTING,) * 6
    assert report.winding_temp_c == ()
    assert report.stall_fractions == ()
    assert report.stall_fractions_sustained == ()


def test_ewma_converges_and_smooths():
    loads = SustainedLoads(tau_s=2.0)
    first = loads.update((1.0,) * 6, dt_s=0.25)
    assert first == (1.0,) * 6  # seeded, not ramped from zero
    smoothed = loads.update((0.0,) * 6, dt_s=0.25)
    assert 0.8 < smoothed[0] < 1.0
    for _ in range(100):
        smoothed = loads.update((0.0,) * 6, dt_s=0.25)
    assert smoothed[0] < 0.01


def _report(temps=COOL, torque=DRIVEN, faults=NO_FAULTS):
    return assess(temps, IDLE, torque, faults, IDLE, bus_silent=False)


def test_alert_transitions_and_clear():
    tracker = AlertTracker("so101_follower test")
    assert tracker.transitions(_report()) == []

    hot = _report(temps=(66.0,) + (30.0,) * 5)
    raised = tracker.transitions(hot)
    assert len(raised) == 1
    assert raised[0].source == "so101_follower test shoulder_pan"
    assert raised[0].severity == 2

    # Unchanged conditions re-emit via active(), not transitions().
    assert tracker.transitions(hot) == []
    assert [a.source for a in tracker.active()] == ["so101_follower test shoulder_pan"]

    cleared = tracker.transitions(_report())
    assert len(cleared) == 1
    assert cleared[0].severity == 0
    assert tracker.active() == []


def test_disabled_torque_faults_the_motor_over_any_reading():
    dropped = (False,) + (True,) * 5
    report = assess(COOL, IDLE, dropped, NO_FAULTS, IDLE, bus_silent=False)
    # Cool and idle readings cannot vouch for a motor that is not driving.
    assert report.levels[0] == LEVEL_FAULT
    assert report.levels[1:] == (LEVEL_NOMINAL,) * 5

    tracker = AlertTracker("so101_follower test")
    raised = tracker.transitions(report)
    assert len(raised) == 1
    assert raised[0].severity == 3
    assert "torque unexpectedly disabled" in raised[0].message


def test_servo_fault_bits_fault_the_motor_with_a_decoded_alert():
    # The overload latch from hardware: output cut, Torque_Enable still 1.
    overloaded = (0,) + (32,) + (0,) * 4
    report = assess(COOL, IDLE, DRIVEN, overloaded, IDLE, bus_silent=False)
    assert report.levels[1] == LEVEL_FAULT
    assert report.levels[0] == LEVEL_NOMINAL

    tracker = AlertTracker("so101_follower test")
    raised = tracker.transitions(report)
    assert len(raised) == 1
    assert raised[0].source == "so101_follower test shoulder_lift"
    assert raised[0].severity == 3
    assert "servo fault latched: overload" in raised[0].message

    # A changed cause at the same level re-alerts with the new decode.
    overheat_too = (0,) + (32 | 4,) + (0,) * 4
    changed = tracker.transitions(_report(faults=overheat_too))
    assert len(changed) == 1
    assert "overheating, overload" in changed[0].message


def test_describe_faults_decodes_known_bits_and_falls_back():
    assert describe_faults(32) == "overload"
    assert describe_faults(1 | 8) == "voltage, overcurrent"
    assert describe_faults(64) == "code 64"


def test_silent_bus_reports_no_fault_bits():
    report = assess(COOL, IDLE, DRIVEN, (32,) * 6, IDLE, bus_silent=True)
    assert report.levels == (LEVEL_NOT_REPORTING,) * 6
    assert report.fault_bits == (0,) * 6


def test_peak_fractions_are_relative_to_the_trip_point():
    loads = (0.8,) + (0.4,) * 5
    report = assess(COOL, loads, DRIVEN, NO_FAULTS, loads, bus_silent=False)
    peaks = report.peak_fractions()
    # At the servo's own overload trip (80% of stall) the wire reads 1.0.
    assert peaks[0] == 1.0
    assert peaks[1] == 0.4 / TRIP_FRACTION_OF_STALL
