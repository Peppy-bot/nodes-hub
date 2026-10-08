from so101_description.units import MOTOR_NAMES

from so101_follower.health import (
    LEVEL_CRITICAL,
    LEVEL_FAULT,
    LEVEL_NOMINAL,
    LEVEL_NOT_REPORTING,
    LEVEL_WARNING,
    ALERT_FLOOR_PERIOD_S,
    LOAD_WARN_FRACTION,
    LOAD_WARN_RELEASE_FRACTION,
    TEMP_WARN_C,
    TEMP_WARN_RELEASE_C,
    TRIP_FRACTION_OF_STALL,
    AlertTracker,
    Bands,
    SustainedLoads,
    assess,
    describe_faults,
)

COOL = tuple(30.0 for _ in MOTOR_NAMES)
IDLE = tuple(0.1 for _ in MOTOR_NAMES)
DRIVEN = tuple(True for _ in MOTOR_NAMES)
NO_FAULTS = tuple(0 for _ in MOTOR_NAMES)


def test_nominal_report():
    report = assess(COOL, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=False, bands=Bands())
    assert report.levels == (LEVEL_NOMINAL,) * 6
    assert report.winding_temp_c == COOL
    assert report.stall_fractions == IDLE


def test_temperature_thresholds():
    temps = (30.0, 56.0, 66.0, 30.0, 30.0, 30.0)
    report = assess(temps, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=False, bands=Bands())
    assert report.levels[0] == LEVEL_NOMINAL
    assert report.levels[1] == LEVEL_WARNING
    assert report.levels[2] == LEVEL_CRITICAL


def test_sustained_load_thresholds_use_smoothed_value():
    spiky = (0.9,) + (0.1,) * 5
    calm = (0.1,) * 6
    report = assess(COOL, spiky, DRIVEN, NO_FAULTS, calm, bus_silent=False, bands=Bands())
    # The instantaneous spike reports in stall_fractions but does not
    # trip a level; only the sustained estimate does.
    assert report.levels == (LEVEL_NOMINAL,) * 6
    assert report.stall_fractions == spiky

    sustained_hot = (0.85,) + (0.1,) * 5
    report = assess(COOL, spiky, DRIVEN, NO_FAULTS, sustained_hot, bus_silent=False, bands=Bands())
    assert report.levels[0] == LEVEL_CRITICAL


def test_silent_bus_empties_readings_and_flags_every_motor():
    report = assess(COOL, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=True, bands=Bands())
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


def _report(temps=COOL, torque=DRIVEN, faults=NO_FAULTS, bands=None):
    return assess(
        temps, IDLE, torque, faults, IDLE, bus_silent=False, bands=bands or Bands()
    )


def _tracker(clock=None):
    """A tracker on a fixed clock, with its opening set already sent."""
    tracker = AlertTracker("so101_follower test", monotonic=clock or (lambda: 0.0))
    opening = tracker.due(_report())
    assert opening is not None and opening.alerts == (), "the opening set is owed"
    tracker.mark_sent(opening)
    return tracker


def test_the_opening_set_goes_out_once_even_when_every_motor_is_quiet():
    # The topic retains one message, so a consumer that subscribes later
    # reads this one and can tell a quiet follower from one that never
    # started.
    clock = {"now": 0.0}
    tracker = AlertTracker("so101_follower test", monotonic=lambda: clock["now"])
    opening = tracker.due(_report())
    assert opening is not None
    assert opening.alerts == ()
    tracker.mark_sent(opening)
    assert tracker.due(_report()) is None


def test_the_alert_set_is_owed_on_change_and_clears_by_absence():
    tracker = _tracker()
    hot = _report(temps=(66.0,) + (30.0,) * 5)
    owed = tracker.due(hot)
    assert owed is not None
    assert [a.source for a in owed.alerts] == ["so101_follower test shoulder_pan"]
    assert owed.alerts[0].severity == 2
    # An unsent set stays owed; a sent one is owed again only on change.
    assert tracker.due(hot) == owed
    tracker.mark_sent(owed)
    assert tracker.due(hot) is None

    cleared = tracker.due(_report())
    assert cleared is not None
    assert cleared.alerts == ()
    tracker.mark_sent(cleared)
    assert tracker.due(_report()) is None


def test_the_unchanged_set_is_owed_again_on_the_cadence_floor():
    # The floor is what refreshes the measurement each message carries and
    # what lets a consumer age a quiet follower out.
    clock = {"now": 0.0}
    tracker = _tracker(lambda: clock["now"])
    hot = _report(temps=(66.0,) + (30.0,) * 5)
    owed = tracker.due(hot)
    assert owed is not None
    tracker.mark_sent(owed)

    clock["now"] += ALERT_FLOOR_PERIOD_S - 0.01
    assert tracker.due(hot) is None, "nothing is owed before the floor elapses"
    clock["now"] += 0.02
    again = tracker.due(hot)
    assert again is not None
    assert again.alerts == owed.alerts, "the same set goes out again"


def test_a_silent_bus_alerts_rather_than_raising():
    # A silent bus is the condition this channel exists to surface, and the
    # level it produces is outside the thresholds' own range.
    tracker = _tracker()
    silent = assess(
        COOL, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=True, bands=Bands()
    )
    assert silent.levels == (LEVEL_NOT_REPORTING,) * len(MOTOR_NAMES)
    owed = tracker.due(silent)
    assert owed is not None
    assert len(owed.alerts) == len(MOTOR_NAMES)
    assert all(a.severity == 3 for a in owed.alerts), (
        "silence is at least as bad as a fault, as it is for every producer"
    )
    assert "not reporting" in owed.alerts[0].message


def test_a_reading_parked_on_a_threshold_holds_one_level():
    # Without a release band each crossing would re-publish the whole set on
    # a reliable topic at the read rate, and a late consumer would read
    # whichever side of the threshold the newest message caught.
    bands = Bands()
    just_over = (TEMP_WARN_C,) + (30.0,) * 5
    just_under = (TEMP_WARN_C - 0.5,) + (30.0,) * 5
    engaged = assess(
        just_over, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=False, bands=bands
    )
    assert engaged.levels[0] == LEVEL_WARNING
    held = assess(
        just_under, IDLE, DRIVEN, NO_FAULTS, IDLE, bus_silent=False, bands=bands
    )
    assert held.levels[0] == LEVEL_WARNING, "a dip below the engage point holds"
    released = assess(
        (TEMP_WARN_RELEASE_C - 0.1,) + (30.0,) * 5,
        IDLE,
        DRIVEN,
        NO_FAULTS,
        IDLE,
        bus_silent=False,
        bands=bands,
    )
    assert released.levels[0] == LEVEL_NOMINAL, "below the release point it clears"


def test_a_load_parked_on_a_threshold_holds_one_level():
    bands = Bands()
    over = (LOAD_WARN_FRACTION,) + (0.1,) * 5
    under = (LOAD_WARN_FRACTION - 0.05,) + (0.1,) * 5
    assert (
        assess(COOL, over, DRIVEN, NO_FAULTS, over, bus_silent=False, bands=bands).levels[0]
        == LEVEL_WARNING
    )
    assert (
        assess(COOL, under, DRIVEN, NO_FAULTS, under, bus_silent=False, bands=bands).levels[0]
        == LEVEL_WARNING
    )
    clear = (LOAD_WARN_RELEASE_FRACTION - 0.05,) + (0.1,) * 5
    assert (
        assess(COOL, clear, DRIVEN, NO_FAULTS, clear, bus_silent=False, bands=bands).levels[0]
        == LEVEL_NOMINAL
    )


def test_disabled_torque_faults_the_motor_over_any_reading():
    dropped = (False,) + (True,) * 5
    report = assess(COOL, IDLE, dropped, NO_FAULTS, IDLE, bus_silent=False, bands=Bands())
    # Cool and idle readings cannot vouch for a motor that is not driving.
    assert report.levels[0] == LEVEL_FAULT
    assert report.levels[1:] == (LEVEL_NOMINAL,) * 5

    tracker = AlertTracker("so101_follower test")
    raised = tracker.due(report).alerts
    assert len(raised) == 1
    assert raised[0].severity == 3
    assert "torque unexpectedly disabled" in raised[0].message


def test_servo_fault_bits_fault_the_motor_with_a_decoded_alert():
    # The overload latch from hardware: output cut, Torque_Enable still 1.
    overloaded = (0,) + (32,) + (0,) * 4
    report = assess(COOL, IDLE, DRIVEN, overloaded, IDLE, bus_silent=False, bands=Bands())
    assert report.levels[1] == LEVEL_FAULT
    assert report.levels[0] == LEVEL_NOMINAL

    tracker = AlertTracker("so101_follower test")
    owed = tracker.due(report)
    raised = owed.alerts
    assert len(raised) == 1
    assert raised[0].source == "so101_follower test shoulder_lift"
    assert raised[0].severity == 3
    assert "servo fault latched: overload" in raised[0].message
    tracker.mark_sent(owed)

    # A changed cause at the same level owes the set again with the new decode.
    overheat_too = (0,) + (32 | 4,) + (0,) * 4
    changed = tracker.due(_report(faults=overheat_too)).alerts
    assert len(changed) == 1
    assert "overheating, overload" in changed[0].message


def test_describe_faults_decodes_known_bits_and_falls_back():
    assert describe_faults(32) == "overload"
    assert describe_faults(1 | 8) == "voltage, overcurrent"
    assert describe_faults(64) == "unknown bits 0x40"
    # The bug this guards: an unnamed bit arriving beside a named one used to
    # vanish, reporting a motor as merely overheating.
    assert describe_faults(4 | 64) == "overheating, unknown bits 0x40"
    # Bit 16 has no name in the table and must not be silently dropped.
    assert describe_faults(16) == "unknown bits 0x10"
    assert describe_faults(0) == "no fault"


def test_silent_bus_reports_no_fault_bits():
    report = assess(COOL, IDLE, DRIVEN, (32,) * 6, IDLE, bus_silent=True, bands=Bands())
    assert report.levels == (LEVEL_NOT_REPORTING,) * 6
    assert report.fault_bits == (0,) * 6


def test_peak_fractions_are_relative_to_the_trip_point():
    loads = (0.8,) + (0.4,) * 5
    report = assess(COOL, loads, DRIVEN, NO_FAULTS, loads, bus_silent=False, bands=Bands())
    peaks = report.peak_fractions()
    # At the servo's own overload trip (80% of stall) the wire reads 1.0.
    assert peaks[0] == 1.0
    assert peaks[1] == 0.4 / TRIP_FRACTION_OF_STALL
