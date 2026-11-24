from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from peppygen.consumed_topics.alerts import alerts as alerts_topic

from tests.helpers import (
    DIRECTOR_LEASE,
    HOME_REACHED,
    FakeStops,
    ScriptedHeadset,
    source,
    squeezing_hand,
)
from xr_commander import panel
from xr_commander.alerts import ActiveAlerts
from xr_commander.clutch import HandClutch
from xr_commander.motor_health import HealthRow, Level, MotorHealthReports
from xr_commander.panel import (
    BaseSource,
    HandSource,
    PanelLine,
    PanelState,
    alert_lines,
    base_line,
    fit_text,
    hand_line,
    health_lines,
    recorder_row,
    render,
    snapshot,
)
from xr_commander.base import (
    AxisLimits,
    BaseLimits,
    BaseReport,
    BaseReports,
    FollowState,
    NotedStops,
)
from xr_commander.devices import Hand
from xr_commander.drive import GripStatus, RobotLine
from xr_commander.grip_world import Idle, Lingering, Pin, Slot
from xr_commander.publish import MeasuredPoses
from xr_commander.record import RecorderStatus
from xr_commander.task_page import UNNAMED_TASK

_PRODUCER = ("core", "left_arm_inst")


def quiet():
    """An alert store with one bound producer and nothing active."""
    return ActiveAlerts(bound_now=lambda: [_PRODUCER])


def raised(*alerts):
    """An alert store holding each (source, severity, message) as one set."""
    active = quiet()
    refused = active.replace(
        _PRODUCER,
        [
            alerts_topic.MessageActiveItem(
                source=source_name,
                kind="motor_condition",
                severity=severity,
                message=message,
            )
            for source_name, severity, message in alerts
        ],
    )
    assert refused == [], refused
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
            ("core", source_name),
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
        "gripper_bound": True,
        "tracked": True,
        "squeezing": True,
        "must_release": False,
        "follower_fresh": True,
        "engaged": True,
    }
    return {**conditions, **overrides}


def state_of(**overrides):
    return hand_line("LEFT", **states(**overrides)).state


def test_a_posture_note_takes_the_hands_row():
    note = "move to home refused: a leader is streaming arm commands"
    assert hand_line("RIGHT", **states(squeezing=False), note=note).state == note
    assert (
        hand_line("RIGHT", **states(tracked=False, squeezing=False), note=note).state
        == note
    )
    # The note shows while the grip is held too.
    assert hand_line("RIGHT", **states(), note=note).state == note
    assert (
        hand_line(
            "RIGHT", **states(arm_bound=False, gripper_bound=False), note=note
        ).state
        == "nothing bound"
    )


def test_the_posture_state_rides_the_right_hands_row():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    moving = "moving to home..."
    state = snapshot(
        session,
        [source("left"), source("right")],
        10.0,
        quiet(),
        no_health(),
        posture=lambda: moving,
    )
    rows = {row.label: row.state for row in state.hands}
    assert rows["RIGHT"] == moving
    assert rows["LEFT"] != moving
    state = snapshot(
        session,
        [source("left"), source("right")],
        10.0,
        quiet(),
        no_health(),
        posture=lambda: HOME_REACHED,
    )
    assert {row.label: row.state for row in state.hands}[
        "RIGHT"
    ] == HOME_REACHED


def test_each_gate_the_operator_hits_gets_its_own_wording():
    # The panel exists to answer "why is nothing moving", so every refusal
    # must read differently.
    assert state_of(arm_bound=False, gripper_bound=False) == "nothing bound"
    assert state_of(tracked=False) == "no controller"
    assert state_of(squeezing=False) == "grip released"
    assert state_of(must_release=True) == "release the grip, then squeeze again"
    assert state_of(follower_fresh=False) == "waiting for arm"
    assert state_of(engaged=False) == "re-anchoring"
    assert state_of() == "driving"


def test_a_gripper_only_hand_reports_its_grip_instead_of_a_missing_arm():
    # Either slot makes a hand wired, so a hand that only drives a gripper
    # gets a row; that row must describe the grip it actually does, not sit
    # forever on an arm it was never given.
    assert state_of(arm_bound=False) == "gripping"
    assert state_of(arm_bound=False, squeezing=False) == "grip released"
    # An absent arm has no pose to anchor, so the arm-only gates cannot
    # strand the row.
    assert state_of(arm_bound=False, follower_fresh=False) == "gripping"
    assert state_of(arm_bound=False, engaged=False) == "gripping"


def test_a_lone_wired_hand_is_named_for_the_slot_it_drives():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    for gripper_only, expected in ((False, "ARM"), (True, "GRIPPER")):
        hand = source("left", arm_bound=not gripper_only, gripper_bound=True)
        rows = snapshot(session, [hand], 10.0, quiet(), no_health()).hands
        assert [row.label for row in rows] == [expected]


def test_the_earliest_unmet_gate_wins_over_every_later_one():
    # Pin each adjacent pair: reordering any two checks would report a later
    # problem than the one the operator has to fix first.
    assert (
        state_of(arm_bound=False, gripper_bound=False, tracked=False) == "nothing bound"
    )
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


def test_each_row_reads_its_own_hand_not_a_neighbours():
    # Routing every row off one hand would still render two plausible rows.
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    hands = [source("left"), source("right")]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert [row.label for row in rows] == ["LEFT", "RIGHT"]
    assert rows[0].state == "re-anchoring"  # squeezing, fresh, not yet engaged
    assert rows[1].state == "no controller"  # never reported a controller


def test_an_unbound_arm_says_so_instead_of_waiting_forever():
    # An optional pairing left unbound would otherwise read "waiting for arm"
    # for the whole session.
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    rows = snapshot(
        session, [source("left", arm_bound=False)], 10.0, quiet(), no_health()
    ).hands
    assert rows[0].state == "nothing bound"


def test_a_pair_established_after_boot_reaches_the_panel():
    # The daemon may finish establishing the launcher's pairs after setup, so
    # a boot-time read would report "nothing bound" for the whole session.
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    paired = []
    hand = HandSource(
        handedness="left",
        clutch=HandClutch(1.0),
        grip=lambda: GripStatus(Idle(), frozenset(), ""),
        measured=MeasuredPoses(),
        arm_peer=lambda: paired[0] if paired else None,
        gripper_peer=lambda: None,
    )
    assert (
        snapshot(session, [hand], 10.0, quiet(), no_health()).hands[0].state
        == "nothing bound"
    )
    paired.append("backbone")
    assert (
        snapshot(session, [hand], 10.0, quiet(), no_health()).hands[0].state
        != "nothing bound"
    )


def test_a_long_row_state_stays_inside_the_panel():
    long_state = "move to home refused: a leader is streaming arm commands and more"
    rows = (panel.Row("RIGHT", long_state, (70, 70, 235)),)
    frame = render(PanelState(headset_live=True, hands=rows))
    lit = np.nonzero(np.any(frame != panel._BACKGROUND, axis=2).any(axis=0))[0]
    assert lit.max() <= panel._WIDTH - panel._MARGIN_X


def test_a_one_armed_robots_only_row_carries_the_posture_note():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    hands = [source("left"), source("right", arm_bound=False, gripper_bound=False)]
    rows = snapshot(
        session,
        hands,
        10.0,
        quiet(),
        no_health(),
        posture=lambda: "moving to home...",
    ).hands
    assert [(row.label, row.state) for row in rows] == [("ARM", "moving to home...")]


def test_only_wired_hands_get_rows_and_a_lone_one_reads_arm():
    # A single-arm robot pairs one hand; the other side's "nothing bound" row
    # is noise, and the survivor needs no handedness to disambiguate it.
    session = ScriptedHeadset(hands={"right": squeezing_hand()})
    hands = [source("left", arm_bound=False), source("right")]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert [row.label for row in rows] == ["ARM"]


def test_a_gripper_only_hand_keeps_its_row():
    # A side with a gripper but no pose_link still grips, so it still shows.
    session = ScriptedHeadset(hands={})
    hands = [source("left", arm_bound=False, gripper_bound=True), source("right")]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert [row.label for row in rows] == ["LEFT", "RIGHT"]


def test_nothing_wired_shows_every_hand():
    # A stack with no pairs at all must say why nothing drives, per hand.
    session = ScriptedHeadset(hands={})
    hands = [source("left", arm_bound=False), source("right", arm_bound=False)]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert [row.label for row in rows] == ["LEFT", "RIGHT"]
    assert all(row.state == "nothing bound" for row in rows)


def test_a_stale_follower_is_reported_per_hand():
    session = ScriptedHeadset(
        hands={"left": squeezing_hand(), "right": squeezing_hand()}
    )
    hands = [source("left", fresh=False), source("right", fresh=True)]
    rows = snapshot(session, hands, 10.0, quiet(), no_health()).hands
    assert rows[0].state == "waiting for arm"
    assert rows[1].state != "waiting for arm"


def test_the_link_reads_live_only_while_frames_are_fresh():
    fresh = ScriptedHeadset(hands={"left": squeezing_hand()}, age_s=0.0)
    stale = ScriptedHeadset(hands={"left": squeezing_hand()}, age_s=5.0)
    assert snapshot(fresh, [source("left")], 0.25, quiet(), no_health()).headset_live
    assert not snapshot(
        stale, [source("left")], 0.25, quiet(), no_health()
    ).headset_live


def test_a_stale_frame_leaves_every_hand_untracked():
    stale = ScriptedHeadset(hands={"left": squeezing_hand()}, age_s=5.0)
    rows = snapshot(stale, [source("left")], 0.25, quiet(), no_health()).hands
    assert rows[0].state == "no controller"


def panel_state(*rows):
    return PanelState(headset_live=True, hands=tuple(rows))


def test_an_unwired_alert_slot_draws_no_section_and_a_quiet_one_does():
    rows = (hand_line("LEFT", **states()),)
    blank = render(
        PanelState(
            headset_live=True, hands=rows, alerts_bound=False, health_bound=False
        )
    )
    quiet = render(
        PanelState(headset_live=True, hands=rows, alerts_bound=True, health_bound=False)
    )
    below = slice(panel._RULE_Y, None)
    assert not np.any(blank[below] != panel._BACKGROUND)
    assert not np.array_equal(quiet[below], blank[below])


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
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
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
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
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
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    state = snapshot(session, [source("left")], 10.0, raised(alert()), no_health())
    assert len(state.alerts) == 1
    assert "LEFT ARM J2 WARNING" in state.alerts[0].text


def test_a_quiet_robot_lists_no_alerts():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    assert snapshot(session, [source("left")], 10.0, quiet(), no_health()).alerts == ()


def test_the_worst_alert_is_listed_first():
    state = snapshot(
        ScriptedHeadset(hands={"left": squeezing_hand()}),
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
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
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
    lines = health_lines(nominal + [HealthRow("j99 FAULT", Level.FAULT)])
    (fault_line,) = health_lines([HealthRow("x", Level.FAULT)])
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


def test_an_unwired_health_slot_draws_no_section_and_a_silent_wired_one_does():
    # With producers wired, a silent section is a finding: "no reports".
    rows = (hand_line("LEFT", **states()),)
    wired = render(
        PanelState(headset_live=True, hands=rows, alerts_bound=False, health_bound=True)
    )
    unwired = render(
        PanelState(
            headset_live=True, hands=rows, alerts_bound=False, health_bound=False
        )
    )
    below = slice(panel._MOTORS_RULE_Y, None)
    assert not np.any(unwired[below] != panel._BACKGROUND)
    assert not np.array_equal(wired[below], unwired[below])


def test_a_long_health_row_stays_inside_the_panel():
    long_row = PanelLine(
        "j7 NOT REPORTING peak 100% rated 112% sust 108% 101C/98C" + " x" * 40,
        (230, 120, 230),
    )
    frame = render(PanelState(headset_live=True, hands=(), health=(long_row,)))
    lit = np.nonzero(np.any(frame != panel._BACKGROUND, axis=2).any(axis=0))[0]
    assert lit.max() < panel._WIDTH - 1


def test_no_recorder_means_no_task_line():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    state = snapshot(session, [source("left")], 10.0, quiet(), no_health())
    assert state.task is None


def test_a_bound_recorder_with_no_label_still_draws_the_line():
    # An unnamed session records anyway, so the panel is where the operator
    # can notice it before the dataset has to be sorted out later.
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
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
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
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
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    hands = [source("left")]
    unset = snapshot(session, hands, 10.0, quiet(), no_health(), RecorderStatus(), None)
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
        PanelState(
            headset_live=True, hands=rows, task=panel.task_line("wipe the table")
        )
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


def test_the_robot_row_comes_first_and_wears_the_colour_of_who_holds_the_robot():
    driving = panel.robot_row(RobotLine("alpha: driven from this headset", "driving"))
    leased = panel.robot_row(
        RobotLine(f"alpha: leased by director_inst ({DIRECTOR_LEASE})", "waiting")
    )
    assert (driving.label, driving.colour) == ("ROBOT", panel._DRIVING)
    assert leased.colour == panel._WAITING
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    state = snapshot(
        session,
        [source("left")],
        10.0,
        quiet(),
        no_health(),
        robot=RobotLine("alpha: free", "resting"),
    )
    assert state.robot == panel.Row("ROBOT", "alpha: free", panel._RESTING)
    rows = (hand_line("LEFT", **states()),)
    without = render(PanelState(headset_live=True, hands=rows))
    with_robot = render(PanelState(headset_live=True, hands=rows, robot=driving))
    first_band = slice(panel._FIRST_ROW_BASELINE - 30, panel._FIRST_ROW_BASELINE + 6)
    assert not np.array_equal(without[first_band], with_robot[first_band])


def test_a_refused_squeeze_flashes_its_cause_on_the_hand_that_squeezed():
    note = "squeeze refused: robot `alpha` is leased by `director_inst`"
    hand = HandSource(
        handedness="left",
        clutch=HandClutch(1.0),
        grip=lambda: GripStatus(Idle(), frozenset(), note),
        measured=MeasuredPoses(),
        arm_peer=lambda: "peer",
        gripper_peer=lambda: None,
    )
    rows = snapshot(
        ScriptedHeadset(hands={"left": squeezing_hand()}),
        [hand],
        10.0,
        quiet(),
        no_health(),
    ).hands
    assert rows[0].state == note


def test_a_hand_that_must_release_its_squeeze_says_so_on_its_row():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    hand = replace(
        source("left"),
        grip=lambda: GripStatus(Idle(), frozenset({Hand.LEFT}), ""),
    )
    rows = snapshot(session, [hand, source("right")], 10.0, quiet(), no_health())
    assert [row.state for row in rows.hands] == [
        "release the grip, then squeeze again",
        "no controller",
    ]


def test_the_four_status_rows_end_above_the_alerts_and_the_motors_inside_the_canvas():
    last_row = (
        panel._FIRST_ROW_BASELINE + (panel._MAX_STATUS_ROWS - 1) * panel._ROW_STEP
    )
    assert panel._MAX_STATUS_ROWS == 4
    assert last_row + panel._DESCENDER < panel._RULE_Y
    last_health = (
        panel._FIRST_HEALTH_BASELINE
        + (panel._MAX_HEALTH_LINES - 1) * panel._HEALTH_STEP
    )
    assert last_health + panel._DESCENDER <= panel._HEIGHT
    # Drawn with every row, the rule under the rows stays clear of text.
    state = PanelState(
        headset_live=True,
        robot=panel.robot_row(RobotLine("alpha: driven from this headset", "driving")),
        hands=(hand_line("LEFT", **states()), hand_line("RIGHT", **states())),
        recorder=recorder_row(recording_status()),
    )
    rule = render(state)[
        panel._RULE_Y : panel._RULE_Y + panel._RULE_HEIGHT,
        panel._MARGIN_X : panel._WIDTH - panel._MARGIN_X,
    ]
    assert np.all(rule == panel._RULE)


def test_a_long_message_is_truncated_rather_than_shrunk_to_a_smear():
    # Shrinking without a floor turns a long message into a one-pixel smear
    # carrying no information.
    font = cv2.FONT_HERSHEY_SIMPLEX
    long_text = "LEFT ARM J2 CRITICAL: " + "x" * 800
    body, scale = fit_text(long_text, font, 1.0, 2, 600)
    assert scale >= 0.5, "the glyphs stay legible"
    assert len(body) < len(long_text), "the message is truncated"
    assert body.startswith("LEFT ARM J2 CRITICAL:"), "the joint survives"
    assert cv2.getTextSize(body, font, scale, 2)[0][0] <= 600


def test_a_message_that_fits_is_left_exactly_as_it_is():
    font = cv2.FONT_HERSHEY_SIMPLEX
    body, scale = fit_text("J2 HOT", font, 1.0, 2, 600)
    assert (body, scale) == ("J2 HOT", 1.0)


def test_truncation_only_when_the_whole_text_cannot_fit_at_the_floor():
    # A single proportional estimate under-shrinks when glyph widths round
    # up, truncating messages that would fit whole a hair smaller. Sweep
    # the widths: any truncation must be forced, not an estimation artifact.
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "RIGHT ARM J4 CRITICAL: winding 91 C"
    floor = 0.9 * 0.55
    for available in range(60, 620, 4):
        body, _scale = fit_text(text, font, 0.9, 2, available)
        if body != text:
            width_at_floor = cv2.getTextSize(text, font, floor, 2)[0][0]
            assert width_at_floor > available, (
                f"truncated at {available}px though the whole text fits at the floor"
            )


# A differential base's limits along x, y and of the turn (m/s, rad/s).
LIMITS = BaseLimits(AxisLimits(-0.4, 0.8), AxisLimits(0.0, 0.0), AxisLimits(-1.5, 1.0))
# A base's report as the panel reads it: following, standing, arrived at the
# start of the instance's clock.
FOLLOWING = BaseReport(FollowState.FOLLOWING, "standing", LIMITS, 0)


def base_states(**overrides):
    """The base row's conditions, all met: the base drives."""
    conditions = {
        "dropped": None,
        "report": FOLLOWING,
        "tracked": True,
        "squeezing": True,
        "must_release": False,
        "engaged": True,
        "following": True,
    }
    return {**conditions, **overrides}


def base_state_of(**overrides) -> str:
    return base_line(**base_states(**overrides)).state


def holding(follow_state: FollowState, mode="standing") -> BaseReport:
    return BaseReport(follow_state, mode, LIMITS, 0)


DROPPED = "max_rate nan is not 0 or above"
BASE_NOTE = "squeeze refused: the base does not report"
# Each row of the base, by its place in the row's order: the conditions that
# give it, its text and its colour.
BASE_ROWS = {
    1: ({"note": BASE_NOTE}, BASE_NOTE, panel._WAITING),
    2: (
        {"dropped": DROPPED},
        f"drive_states unreadable ({DROPPED}): check its base node",
        panel._NOT_REPORTING_MAGENTA,
    ),
    3: (
        {"report": None},
        "the base does not report: check its base node",
        panel._NOT_REPORTING_MAGENTA,
    ),
    4: (
        {"report": holding(FollowState.BASE_SILENCE)},
        "the base's motion sensing is silent: it rests until it reports and 0.5 s"
        " pass with no setpoint",
        panel._WAITING,
    ),
    5: (
        {"report": holding(FollowState.ROBOT_STOP)},
        "a robot stop holds the base",
        panel._WAITING,
    ),
    6: (
        {"report": holding(FollowState.FOLLOWER_GOAL)},
        "a goal of the base's node holds it: cancel it or let it end",
        panel._WAITING,
    ),
    7: (
        {"report": holding(FollowState.BASE_MODE, mode="damped")},
        "base damped, no velocity: release its hold, such as its e-stop",
        panel._WAITING,
    ),
    8: ({"tracked": False, "squeezing": False}, "no controller", panel._RESTING),
    9: ({"squeezing": False}, "grip released", panel._RESTING),
    10: (
        {"must_release": True},
        "release the grip, then squeeze again",
        panel._WAITING,
    ),
    11: (
        {"engaged": False, "following": False},
        "taking the robot",
        panel._WAITING,
    ),
    12: ({"following": False}, "waiting for the base", panel._WAITING),
    13: ({}, "driving", panel._DRIVING),
}


def test_each_hold_of_the_base_says_why():
    texts = {
        follow_state: base_state_of(report=holding(follow_state, mode="damped"))
        for follow_state in FollowState
        if follow_state is not FollowState.FOLLOWING
    }
    assert texts == {
        FollowState.BASE_MODE: (
            "base damped, no velocity: release its hold, such as its e-stop"
        ),
        FollowState.BASE_SILENCE: (
            "the base's motion sensing is silent: it rests until it reports and"
            " 0.5 s pass with no setpoint"
        ),
        FollowState.ROBOT_STOP: "a robot stop holds the base",
        FollowState.FOLLOWER_GOAL: (
            "a goal of the base's node holds it: cancel it or let it end"
        ),
    }
    assert {
        base_line(**base_states(report=holding(state))).colour for state in texts
    } == {panel._WAITING}


@pytest.mark.parametrize("row", range(3, 14))
def test_every_fixed_base_text_fits_the_row_whole_in_the_longest_mode(row):
    width = panel._WIDTH - panel._MARGIN_X - panel._STATE_X
    conditions, _text, _colour = BASE_ROWS[row]
    if row == 7:
        conditions = {"report": holding(FollowState.BASE_MODE, mode="transition")}
    text = base_state_of(**conditions)
    fitted, _scale = fit_text(
        text, panel._FONT, panel._BODY_SCALE, panel._STROKE, width
    )
    assert fitted == text


@pytest.mark.parametrize("row", sorted(BASE_ROWS))
def test_each_base_row_gives_its_text_and_colour(row):
    conditions, text, colour = BASE_ROWS[row]
    assert base_line(**base_states(**conditions)) == panel.Row("BASE", text, colour)


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        (1, 2),
        (2, 3),
        (3, 8),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
    ],
)
def test_the_earliest_base_row_wins(earlier, later):
    conditions = {**BASE_ROWS[later][0], **BASE_ROWS[earlier][0]}
    assert base_state_of(**conditions) == BASE_ROWS[earlier][1]


IDLE_BASE = GripStatus(Idle(), frozenset(), "")


def base_source(
    *, peer="base peer", reports=None, now_ns=0, status=IDLE_BASE
) -> BaseSource:
    """The panel's view of a base paired as `peer`, reporting into
    `reports`, on an instance clock at `now_ns`, with the base grip at
    `status`."""
    reports = reports if reports is not None else BaseReports(lambda: now_ns)
    return BaseSource(
        grip=lambda: status,
        reports=reports,
        stops=NotedStops(FakeStops([]), reports),
        peer=lambda: peer,
    )


def reported_base(peer, follow_state="following", *, at_ns=0):
    """A base's reports holding one drive_states of `peer` that arrived at
    `at_ns` on the instance's clock, on a clock the test sets."""
    clock = {"now_ns": at_ns}
    reports = BaseReports(lambda: clock["now_ns"])
    reports.record(
        peer,
        SimpleNamespace(
            follow_state=follow_state,
            mode="standing",
            linear_limits=SimpleNamespace(
                min_velocity=[-0.4, 0.0], max_velocity=[0.8, 0.0]
            ),
            angular_limits=SimpleNamespace(min_rate=-1.5, max_rate=1.0),
        ),
    )
    return reports, clock


def test_a_base_reports_for_half_a_second_of_the_instances_clock_after_its_report():
    reports, clock = reported_base("base peer")
    session = ScriptedHeadset(hands={"left": squeezing_hand()})

    def row():
        base = base_source(reports=reports)
        return snapshot(session, [], 10.0, quiet(), no_health(), base=base).base

    assert row().state == "taking the robot"
    clock["now_ns"] = 500_000_000
    assert row().state == "taking the robot"
    clock["now_ns"] = 500_000_001
    assert row().state == "the base does not report: check its base node"


def test_the_base_row_asks_for_a_release_while_every_squeezing_hand_must_release():
    reports, _clock = reported_base("base peer")
    both = ScriptedHeadset(
        hands={"left": squeezing_hand(), "right": squeezing_hand()}
    )

    def row(session, must_release):
        status = GripStatus(Idle(), frozenset(must_release), "")
        base = base_source(reports=reports, status=status)
        return snapshot(session, [], 10.0, quiet(), no_health(), base=base).base

    release_first = "release the grip, then squeeze again"
    assert row(both, {Hand.LEFT, Hand.RIGHT}).state == release_first
    # The right hand's squeeze is a claim of its own on the base.
    assert row(both, {Hand.LEFT}).state == "taking the robot"
    left_only = ScriptedHeadset(hands={"left": squeezing_hand()})
    assert row(left_only, {Hand.LEFT}).state == release_first


def test_the_base_row_reads_the_pair_its_grip_drives_else_the_chosen_robots():
    reports, _clock = reported_base("driven peer", "robot_stop")
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    lingering = GripStatus(
        Lingering("alpha", frozenset({Pin(Slot.BASE, "driven peer")}), 1),
        frozenset(),
        "",
    )

    def row(status):
        base = base_source(peer="chosen peer", reports=reports, status=status)
        return snapshot(session, [], 10.0, quiet(), no_health(), base=base).base

    assert row(lingering).state == "a robot stop holds the base"
    assert row(IDLE_BASE).state == "the base does not report: check its base node"


def test_a_robot_that_pairs_a_base_alone_shows_the_base_and_no_hand():
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    hands = [source("left", arm_bound=False), source("right", arm_bound=False)]
    state = snapshot(session, hands, 10.0, quiet(), no_health(), base=base_source())
    assert state.hands == ()
    assert state.base == panel.Row(
        "BASE",
        "the base does not report: check its base node",
        panel._NOT_REPORTING_MAGENTA,
    )
    # A robot with no base gets no base row.
    unpaired = snapshot(
        session, hands, 10.0, quiet(), no_health(), base=base_source(peer=None)
    )
    assert unpaired.base is None
    assert [row.state for row in unpaired.hands] == ["nothing bound"] * 2


def test_the_base_row_is_drawn_after_the_hands_and_before_the_recorder():
    rows = (hand_line("LEFT", **states()), hand_line("RIGHT", **states()))
    without = render(PanelState(headset_live=True, hands=rows))
    with_base = render(
        PanelState(headset_live=True, hands=rows, base=base_line(**base_states()))
    )
    third_band = slice(
        panel._FIRST_ROW_BASELINE + 2 * panel._ROW_STEP - 30,
        panel._FIRST_ROW_BASELINE + 2 * panel._ROW_STEP + 6,
    )
    assert not np.array_equal(without[third_band], with_base[third_band])
    recorder = recorder_row(recording_status())
    both = render(
        PanelState(
            headset_live=True,
            hands=rows,
            base=base_line(**base_states()),
            recorder=recorder,
        )
    )
    fourth_band = slice(
        panel._FIRST_ROW_BASELINE + 3 * panel._ROW_STEP - 30,
        panel._FIRST_ROW_BASELINE + 3 * panel._ROW_STEP + 6,
    )
    assert not np.array_equal(with_base[fourth_band], both[fourth_band])


def test_a_base_row_past_the_four_status_rows_fails_loudly():
    # The robot's row, two hands and the recorder's fill the four rows.
    session = ScriptedHeadset(hands={"left": squeezing_hand()})
    full = {
        "recorder": recording_status(),
        "robot": RobotLine("alpha: free", "resting"),
    }
    hands = [source("left"), source("right")]
    snapshot(session, hands, 10.0, quiet(), no_health(), **full)
    with pytest.raises(AssertionError):
        snapshot(
            session, hands, 10.0, quiet(), no_health(), base=base_source(), **full
        )
