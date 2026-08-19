import time

import numpy as np

from tests.helpers import IDENTITY
from xr_commander import panel
from xr_commander.alerts import CRITICAL, ActiveAlerts
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.frames import Pose
from xr_commander.motor_health import FAULT, HealthRow, MotorHealthReports
from xr_commander.panel import (
    HandSource,
    PanelLine,
    PanelState,
    alert_lines,
    hand_line,
    health_lines,
    recorder_row,
    render,
    snapshot,
)
from xr_commander.publish import LatestPose
from xr_commander.record import RecorderStatus
from xr_commander.task_page import UNNAMED_TASK

POSE = Pose(np.zeros(3), IDENTITY)


def quiet():
    """An alert map with nothing active."""
    return ActiveAlerts()


def raised(*alerts):
    """An alert map holding each (source, severity, message)."""
    active = ActiveAlerts()
    for source_name, severity, message in alerts:
        active.update("left_arm_inst", source_name, "motor_overload", severity, message)
    return active


def no_health():
    """A health store that has never received a report."""
    return MotorHealthReports()


def health_with(*sources):
    """A health store holding one fully sensed report per (source, levels)."""
    store = MotorHealthReports()
    for source_name, levels in sources:
        count = len(levels)
        store.update(
            source_name,
            levels,
            (0.10,) * count,
            (0.09,) * count,
            (0.03,) * count,
            (33.0,) * count,
            (31.0,) * count,
        )
    return store


def states(**overrides):
    conditions = {
        "arm_bound": True,
        "tracked": True,
        "squeezing": True,
        "follower_fresh": True,
        "engaged": True,
    }
    return {**conditions, **overrides}


def state_of(**overrides):
    return hand_line("LEFT", **states(**overrides)).state


def test_each_gate_the_operator_hits_gets_its_own_wording():
    # The panel exists to answer "why is nothing moving", so every refusal
    # must read differently.
    assert state_of(arm_bound=False) == "no arm bound"
    assert state_of(tracked=False) == "no controller"
    assert state_of(squeezing=False) == "grip released"
    assert state_of(follower_fresh=False) == "waiting for arm"
    assert state_of(engaged=False) == "re-anchoring"
    assert state_of() == "driving"


def test_the_earliest_unmet_gate_wins_over_every_later_one():
    # Pin each adjacent pair: reordering any two checks would report a later
    # problem than the one the operator has to fix first.
    assert state_of(arm_bound=False, tracked=False) == "no arm bound"
    assert state_of(tracked=False, squeezing=False) == "no controller"
    assert state_of(squeezing=False, follower_fresh=False) == "grip released"
    assert state_of(follower_fresh=False, engaged=False) == "waiting for arm"


def test_only_a_driving_hand_reads_as_driving():
    assert (
        hand_line("LEFT", **states()).colour
        != hand_line("LEFT", **states(engaged=False)).colour
    )


def test_the_label_is_carried_through():
    assert hand_line("RIGHT", **states()).label == "RIGHT"


class FakeSession:
    """A frame source holding one hand-set, stamped now."""

    def __init__(self, hands=None, age_s=0.0):
        self._hands = hands or {}
        self._age_s = age_s

    def latest(self):
        if self._hands is None:
            return None
        return XrFrame(
            received_monotonic_s=time.monotonic() - self._age_s, hands=self._hands
        )


def squeezing_hand():
    return HandSample(pose=POSE, squeezing=True, trigger=0.0)


def source(handedness, *, arm_bound=True, fresh=True, engaged=False):
    measured = LatestPose()
    if fresh:
        measured.set(POSE)
    clutch = HandClutch(1.0)
    if engaged:
        clutch.step(squeezing=True, hand=POSE, measured_ee=POSE)
    return HandSource(
        handedness=handedness,
        clutch=clutch,
        measured=measured,
        arm_paired=lambda: arm_bound,
    )


def test_each_row_reads_its_own_hand_not_a_neighbours():
    # Routing every row off one hand would still render two plausible rows.
    session = FakeSession({"left": squeezing_hand()})
    hands = [source("left"), source("right")]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert [row.label for row in rows] == ["LEFT", "RIGHT"]
    assert rows[0].state == "re-anchoring"  # squeezing, fresh, not yet engaged
    assert rows[1].state == "no controller"  # never reported a controller


def test_an_unbound_arm_says_so_instead_of_waiting_forever():
    # An optional pairing left unbound would otherwise read "waiting for arm"
    # for the whole session.
    session = FakeSession({"left": squeezing_hand()})
    rows = snapshot(
        session, [source("left", arm_bound=False)], 10.0, quiet(), no_health()
    ).hands
    assert rows[0].state == "no arm bound"


def test_a_pair_established_after_boot_reaches_the_panel():
    # The daemon may finish establishing the launcher's pairs after setup, so
    # a boot-time read would report "no arm bound" for the whole session.
    session = FakeSession({"left": squeezing_hand()})
    paired = []
    hand = HandSource(
        handedness="left",
        clutch=HandClutch(1.0),
        measured=LatestPose(),
        arm_paired=lambda: bool(paired),
    )
    assert (
        snapshot(session, [hand], 10.0, quiet(), no_health()).hands[0].state
        == "no arm bound"
    )
    paired.append(True)
    assert (
        snapshot(session, [hand], 10.0, quiet(), no_health()).hands[0].state
        != "no arm bound"
    )


def test_a_stale_follower_is_reported_per_hand():
    session = FakeSession({"left": squeezing_hand(), "right": squeezing_hand()})
    hands = [source("left", fresh=False), source("right", fresh=True)]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert rows[0].state == "waiting for arm"
    assert rows[1].state != "waiting for arm"


def test_the_link_reads_live_only_while_frames_are_fresh():
    fresh = FakeSession({"left": squeezing_hand()}, age_s=0.0)
    stale = FakeSession({"left": squeezing_hand()}, age_s=5.0)
    assert snapshot(fresh, [source("left")], 0.25, quiet(), no_health()).headset_live
    assert not snapshot(
        stale, [source("left")], 0.25, quiet(), no_health()
    ).headset_live


def test_a_stale_frame_leaves_every_hand_untracked():
    stale = FakeSession({"left": squeezing_hand()}, age_s=5.0)
    rows = snapshot(stale, [source("left")], 0.25, quiet(), no_health()).hands
    assert rows[0].state == "no controller"


def panel_state(*rows):
    return PanelState(headset_live=True, hands=tuple(rows))


def test_an_unwired_alert_slot_renders_differently_from_a_quiet_one():
    # "none" and "not wired" must differ in pixels, not just in state: the
    # panel is the only place the operator can see the difference.
    rows = (hand_line("LEFT", **states()),)
    quiet = render(PanelState(headset_live=True, hands=rows, alerts_bound=True))
    unwired = render(PanelState(headset_live=True, hands=rows, alerts_bound=False))
    assert not np.array_equal(quiet, unwired)


def test_a_rendered_frame_is_what_the_encoder_expects():
    frame = render(panel_state(hand_line("LEFT", **states())))
    assert frame.dtype == np.uint8
    assert frame.shape == (panel._HEIGHT, panel._WIDTH, 3)
    assert frame.flags["C_CONTIGUOUS"]


def test_the_link_state_changes_what_is_drawn():
    rows = (hand_line("LEFT", **states()),)
    live = render(PanelState(headset_live=True, hands=rows))
    down = render(PanelState(headset_live=False, hands=rows))
    assert not np.array_equal(live, down)


def test_each_row_is_drawn_in_its_own_band():
    # Rows are placed by index; collapsing the step would stack them.
    one = render(panel_state(hand_line("LEFT", **states())))
    two = render(
        panel_state(hand_line("LEFT", **states()), hand_line("RIGHT", **states()))
    )
    second_band = slice(
        panel._FIRST_ROW_BASELINE + panel._ROW_STEP - 30,
        panel._FIRST_ROW_BASELINE + panel._ROW_STEP + 6,
    )
    assert not np.array_equal(one[second_band], two[second_band])


def test_the_state_column_does_not_sit_on_the_label():
    assert panel._STATE_X > panel._MARGIN_X


def test_the_panel_name_is_not_one_the_frontend_reserves():
    # wrist_left/wrist_right anchor to the controllers; the panel must float.
    assert panel.TRACK_ID not in {"wrist_left", "wrist_right"}


def recording_status(frames=0):
    status = RecorderStatus()
    status.recording = True
    status.frames = frames
    return status


def test_no_recorder_means_no_row():
    assert recorder_row(None) is None
    session = FakeSession({"left": squeezing_hand()})
    assert (
        snapshot(session, [source("left")], 10.0, quiet(), no_health()).recorder is None
    )


def test_the_recorder_row_reads_the_status():
    idle = recorder_row(RecorderStatus())
    live = recorder_row(recording_status(frames=41))
    assert idle.state == "idle"
    assert "41" in live.state
    assert idle.colour != live.colour


def test_a_save_in_flight_beats_the_frozen_frame_counter():
    status = recording_status(frames=319)
    status.saving = True
    row = recorder_row(status)
    assert row.state == "saving..."
    assert row.colour != recorder_row(recording_status()).colour


def test_a_held_finish_shows_on_the_row():
    status = RecorderStatus()
    status.finish_held = True
    assert recorder_row(status).state == "finishing..."


def test_outcome_notes_show_and_recording_stays_visible():
    status = RecorderStatus()
    status.set_note("saved episode 3")
    assert recorder_row(status).state == "saved episode 3"
    live = recording_status(frames=9)
    live.set_note("finish refused")
    row = recorder_row(live)
    assert "recording 9" in row.state
    assert "finish refused" in row.state


def test_recording_does_not_wear_the_driving_colour():
    # Red-means-recording must stay distinguishable from green-means-driving.
    live = recorder_row(recording_status())
    assert live.colour != hand_line("LEFT", **states()).colour


def test_the_recorder_row_is_drawn_after_the_hands():
    rows = (hand_line("LEFT", **states()), hand_line("RIGHT", **states()))
    without = render(PanelState(headset_live=True, hands=rows))
    with_rec = render(
        PanelState(
            headset_live=True, hands=rows, recorder=recorder_row(recording_status())
        )
    )
    third_band = slice(
        panel._FIRST_ROW_BASELINE + 2 * panel._ROW_STEP - 30,
        panel._FIRST_ROW_BASELINE + 2 * panel._ROW_STEP + 6,
    )
    assert not np.array_equal(without[third_band], with_rec[third_band])


def test_a_frame_count_change_changes_the_state():
    # The redraw is equality-gated, so a stale count would freeze the row.
    session = FakeSession({"left": squeezing_hand()})
    hands = [source("left")]
    first = snapshot(
        session, hands, 10.0, quiet(), no_health(), recording_status(frames=1)
    )
    second = snapshot(
        session, hands, 10.0, quiet(), no_health(), recording_status(frames=2)
    )
    assert first != second


def alert(source_name="left arm j2", severity=1, message="holding 93% of rated"):
    return (source_name, severity, message)


def test_an_active_alert_reaches_the_panel():
    session = FakeSession({"left": squeezing_hand()})
    state = snapshot(session, [source("left")], 10.0, raised(alert()), no_health())
    assert len(state.alerts) == 1
    assert "LEFT ARM J2 WARNING" in state.alerts[0].text


def test_a_quiet_robot_lists_no_alerts():
    session = FakeSession({"left": squeezing_hand()})
    assert snapshot(session, [source("left")], 10.0, quiet(), no_health()).alerts == ()


def test_the_worst_alert_is_listed_first():
    state = snapshot(
        FakeSession({"left": squeezing_hand()}),
        [source("left")],
        10.0,
        raised(alert("left arm j2", 1, "warm"), alert("right arm j1", 3, "limp")),
        no_health(),
    )
    assert state.alerts[0].text.startswith("RIGHT ARM J1 FAULT")
    assert state.alerts[1].text.startswith("LEFT ARM J2 WARNING")


def test_severity_picks_the_alert_colour():
    warning, fault = alert_lines(
        raised(alert("a", 1, "x"), alert("b", 3, "y")).active()
    )
    assert warning.colour != fault.colour


def test_alerts_past_the_canvas_are_counted_not_dropped():
    # Silently showing three of six would read as a less broken robot than it
    # is; the last line has to say how many are hidden.
    lines = alert_lines(
        raised(*[alert(f"joint {i}", 1, "warm") for i in range(6)]).active()
    )
    assert len(lines) == panel._MAX_ALERT_LINES
    hidden = 6 - (panel._MAX_ALERT_LINES - 1)
    assert lines[-1].text == f"+{hidden} more"


def test_exactly_a_full_canvas_shows_every_alert():
    lines = alert_lines(
        raised(
            *[alert(f"joint {i}", 1, "warm") for i in range(panel._MAX_ALERT_LINES)]
        ).active()
    )
    assert len(lines) == panel._MAX_ALERT_LINES
    assert not any(line.text.endswith("more") for line in lines)


def test_the_alert_section_is_drawn_below_the_status_rows():
    rows = (hand_line("LEFT", **states()), hand_line("RIGHT", **states()))
    without = render(PanelState(headset_live=True, hands=rows))
    with_alert = render(
        PanelState(
            headset_live=True,
            hands=rows,
            alerts=(PanelLine("LEFT ARM J2 WARNING: warm", (60, 200, 250)),),
        )
    )
    status_area = slice(0, panel._RULE_Y)
    assert np.array_equal(without[status_area], with_alert[status_area])
    assert not np.array_equal(without[panel._RULE_Y :], with_alert[panel._RULE_Y :])


def test_a_long_alert_stays_inside_the_panel():
    # Shrunk rather than truncated: the joint and the number are at opposite
    # ends of the message, so a clipped line loses one of them.
    long_alert = PanelLine("RIGHT ARM J4 " + "CRITICAL: " * 8 + "99%", (70, 70, 235))
    frame = render(PanelState(headset_live=True, hands=(), alerts=(long_alert,)))
    lit = np.nonzero(np.any(frame != panel._BACKGROUND, axis=2).any(axis=0))[0]
    assert lit.max() < panel._WIDTH - 1


def test_a_snapshot_carries_health_rows():
    session = FakeSession({"left": squeezing_hand()})
    store = health_with(("left gripper", (1,)))
    state = snapshot(session, [source("left")], 10.0, quiet(), store)
    assert len(state.health) == 1
    assert state.health[0].text.startswith("LEFT GRIPPER WARNING")


def test_each_health_level_wears_its_own_colour():
    rows = [HealthRow(f"row {level}", level) for level in range(5)]
    colours = [line.colour for line in health_lines(rows)]
    assert len(set(colours)) == len(rows)


def test_a_health_heading_is_not_coloured_as_a_severity():
    heading, motor = health_lines(
        [HealthRow("LEFT ARM", None), HealthRow("j1 nominal", 0)]
    )
    assert heading.colour != motor.colour


def test_health_rows_past_the_canvas_are_counted_not_dropped():
    # The fixed four-source stack fits exactly; anything beyond it must be
    # counted rather than silently cut.
    rows = [HealthRow(f"j{i} nominal", 0) for i in range(panel._MAX_HEALTH_LINES + 7)]
    lines = health_lines(rows)
    assert len(lines) == panel._MAX_HEALTH_LINES
    assert lines[-1].text == "+8 more"


def test_exactly_a_full_health_canvas_shows_every_row():
    rows = [HealthRow(f"j{i} nominal", 0) for i in range(panel._MAX_HEALTH_LINES)]
    lines = health_lines(rows)
    assert len(lines) == panel._MAX_HEALTH_LINES
    assert not any(line.text.endswith("more") for line in lines)


def test_the_collapsed_health_line_wears_the_worst_hidden_level():
    # A fault buried past the canvas must still colour the panel; counting
    # it in nominal grey would read as a healthy overflow.
    nominal = [HealthRow(f"j{i} nominal", 0) for i in range(panel._MAX_HEALTH_LINES)]
    lines = health_lines(nominal + [HealthRow("j99 FAULT", FAULT)])
    (fault_line,) = health_lines([HealthRow("x", FAULT)])
    assert lines[-1].colour == fault_line.colour


def test_the_health_section_is_drawn_below_the_alerts():
    rows = (hand_line("LEFT", **states()),)
    without = render(PanelState(headset_live=True, hands=rows))
    with_health = render(
        PanelState(
            headset_live=True,
            hands=rows,
            health=health_lines([HealthRow("j1 nominal peak 3%", 0)]),
        )
    )
    above = slice(0, panel._MOTORS_RULE_Y)
    assert np.array_equal(without[above], with_health[above])
    assert not np.array_equal(
        without[panel._MOTORS_RULE_Y :], with_health[panel._MOTORS_RULE_Y :]
    )


def test_an_unwired_health_slot_renders_differently_from_a_quiet_one():
    # "no reports" and "not wired" must differ in pixels: with producers
    # wired, a silent section is a finding, not a launcher choice.
    rows = (hand_line("LEFT", **states()),)
    wired = render(PanelState(headset_live=True, hands=rows, health_bound=True))
    unwired = render(PanelState(headset_live=True, hands=rows, health_bound=False))
    assert not np.array_equal(wired, unwired)


def test_a_long_health_row_stays_inside_the_panel():
    long_row = PanelLine(
        "j7 NOT REPORTING peak 100% rated 112% sust 108% 101C/98C" + " x" * 40,
        (230, 120, 230),
    )
    frame = render(PanelState(headset_live=True, hands=(), health=(long_row,)))
    lit = np.nonzero(np.any(frame != panel._BACKGROUND, axis=2).any(axis=0))[0]
    assert lit.max() < panel._WIDTH - 1


def test_no_recorder_means_no_task_line():
    session = FakeSession({"left": squeezing_hand()})
    state = snapshot(session, [source("left")], 10.0, quiet(), no_health())
    assert state.task is None


def test_a_bound_recorder_with_no_label_still_draws_the_line():
    # An unnamed session records anyway, so the panel is where the operator
    # can notice it before the dataset has to be sorted out later.
    session = FakeSession({"left": squeezing_hand()})
    state = snapshot(
        session, [source("left")], 10.0, quiet(), no_health(), RecorderStatus(), None
    )
    assert state.task == panel.task_line(None)
    assert "NOT SET" in state.task


def test_the_unset_line_does_not_pass_the_placeholder_off_as_a_task():
    # Drawing the placeholder like any other label would read as a named
    # session at a glance.
    assert UNNAMED_TASK not in panel.task_line(None)


def test_a_relabelled_task_redraws_the_panel():
    # Equality-gated like the frame count: a stale label would show the last
    # take's task while the next one records.
    session = FakeSession({"left": squeezing_hand()})
    hands = [source("left")]
    first = snapshot(
        session, hands, 10.0, quiet(), no_health(), RecorderStatus(), "wipe the table"
    )
    second = snapshot(
        session, hands, 10.0, quiet(), no_health(), RecorderStatus(), "stack the blocks"
    )
    assert first != second
    assert not np.array_equal(render(first), render(second))


def test_naming_the_first_task_redraws_the_panel():
    session = FakeSession({"left": squeezing_hand()})
    hands = [source("left")]
    unset = snapshot(
        session, hands, 10.0, quiet(), no_health(), RecorderStatus(), None
    )
    named = snapshot(
        session, hands, 10.0, quiet(), no_health(), RecorderStatus(), "wipe the table"
    )
    assert unset != named
    assert not np.array_equal(render(unset), render(named))


def test_the_task_shares_the_link_line_instead_of_taking_a_status_row():
    # The row band is full at _MAX_STATUS_ROWS and every section below is laid
    # out from it, so a task must not push the rows down.
    rows = (hand_line("LEFT", **states()), hand_line("RIGHT", **states()))
    without = render(PanelState(headset_live=True, hands=rows))
    with_task = render(
        PanelState(headset_live=True, hands=rows, task=panel.task_line("wipe the table"))
    )
    row_band = slice(panel._FIRST_ROW_BASELINE - 30, panel._RULE_Y)
    assert np.array_equal(without[row_band], with_task[row_band])

    link_band = slice(panel._LINK_BASELINE - 30, panel._LINK_BASELINE + 6)
    assert not np.array_equal(without[link_band], with_task[link_band])


def test_a_task_too_long_for_its_column_stays_inside_the_canvas():
    long_task = panel.task_line("wipe the table " * 20)
    frame = render(PanelState(headset_live=True, hands=(), task=long_task))
    lit = np.nonzero(np.any(frame != panel._BACKGROUND, axis=2).any(axis=0))[0]
    assert lit.max() < panel._WIDTH - 1
