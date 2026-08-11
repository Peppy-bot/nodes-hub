import time

import numpy as np

from tests.helpers import IDENTITY
from xr_commander import panel
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.frames import Pose
from xr_commander.panel import (
    HandSource,
    PanelState,
    hand_line,
    recorder_row,
    render,
    snapshot,
)
from xr_commander.publish import LatestPose
from xr_commander.record import RecorderStatus

POSE = Pose(np.zeros(3), IDENTITY)


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
    rows = snapshot(session, hands, 10.0).hands
    assert [row.label for row in rows] == ["LEFT", "RIGHT"]
    assert rows[0].state == "re-anchoring"  # squeezing, fresh, not yet engaged
    assert rows[1].state == "no controller"  # never reported a controller


def test_an_unbound_arm_says_so_instead_of_waiting_forever():
    # An optional pairing left unbound would otherwise read "waiting for arm"
    # for the whole session.
    session = FakeSession({"left": squeezing_hand()})
    rows = snapshot(session, [source("left", arm_bound=False)], 10.0).hands
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
    assert snapshot(session, [hand], 10.0).hands[0].state == "no arm bound"
    paired.append(True)
    assert snapshot(session, [hand], 10.0).hands[0].state != "no arm bound"


def test_a_stale_follower_is_reported_per_hand():
    session = FakeSession({"left": squeezing_hand(), "right": squeezing_hand()})
    hands = [source("left", fresh=False), source("right", fresh=True)]
    rows = snapshot(session, hands, 10.0).hands
    assert rows[0].state == "waiting for arm"
    assert rows[1].state != "waiting for arm"


def test_the_link_reads_live_only_while_frames_are_fresh():
    fresh = FakeSession({"left": squeezing_hand()}, age_s=0.0)
    stale = FakeSession({"left": squeezing_hand()}, age_s=5.0)
    assert snapshot(fresh, [source("left")], 0.25).headset_live
    assert not snapshot(stale, [source("left")], 0.25).headset_live


def test_a_stale_frame_leaves_every_hand_untracked():
    stale = FakeSession({"left": squeezing_hand()}, age_s=5.0)
    rows = snapshot(stale, [source("left")], 0.25).hands
    assert rows[0].state == "no controller"


def panel_state(*rows):
    return PanelState(headset_live=True, hands=tuple(rows))


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
    assert snapshot(session, [source("left")], 10.0).recorder is None


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
    first = snapshot(session, hands, 10.0, recording_status(frames=1))
    second = snapshot(session, hands, 10.0, recording_status(frames=2))
    assert first != second
