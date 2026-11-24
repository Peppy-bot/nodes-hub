import math
from types import SimpleNamespace

import pytest

from tests.helpers import ScriptedHeadset, controller, squeezing_hand
from xr_commander.devices import (
    RESTING,
    Hand,
    RisingEdge,
    Thumbstick,
    XrFrame,
    fresh_sample,
    gripper_opening,
    hand_squeezing,
    parse_hands,
    past_rest_band,
)


def test_both_controllers_are_parsed_and_the_head_is_ignored():
    devices = [
        {
            "role": "head",
            "handedness": "none",
            "pose": {"position": {}, "orientation": {}},
        },
        controller("left", position=(1.0, 0.0, 0.0), squeezing=True, trigger=0.25),
        controller("right", position=(0.0, 2.0, 0.0)),
    ]
    hands = parse_hands(devices)
    assert set(hands) == {"left", "right"}
    assert hands["left"].squeezing is True
    assert hands["left"].trigger == pytest.approx(0.25)
    assert hands["right"].squeezing is False
    assert hands["right"].pose.position.position == (0.0, 2.0, 0.0)


def test_a_device_missing_its_grip_pose_or_gamepad_is_simply_untracked():
    without_pose = controller("left")
    del without_pose["gripPose"]
    without_gamepad = controller("right")
    del without_gamepad["gamepad"]
    assert parse_hands([without_pose, without_gamepad]) == {}


def test_a_gamepad_without_a_squeeze_button_is_untracked():
    short = controller("left")
    short["gamepad"]["buttons"] = short["gamepad"]["buttons"][:1]
    assert parse_hands([short]) == {}


@pytest.mark.parametrize(
    "part, value",
    [
        ("orientation", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}),
        # A squared norm 0.01 or more from 1.
        ("orientation", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0}),
        ("orientation", {"x": math.inf, "y": 0.0, "z": 0.0, "w": 1.0}),
        ("position", {"x": 0.0, "y": math.nan, "z": 0.0}),
    ],
)
def test_an_unusable_pose_drops_that_hand_rather_than_the_frame(part, value):
    broken = controller("left")
    broken["gripPose"][part] = value
    good = controller("right")
    hands = parse_hands([broken, good])
    assert set(hands) == {"right"}


def test_a_hostile_trigger_value_cannot_leave_the_opening_fraction_out_of_range():
    for value, expected in ((-3.0, 0.0), (7.5, 1.0), (math.nan, 0.0), (math.inf, 0.0)):
        hands = parse_hands([controller("left", trigger=value)])
        assert hands["left"].trigger == pytest.approx(expected)


def test_the_trigger_spans_exactly_the_configured_opening_range():
    # Released rests at the configured fraction, fully pulled is closed,
    # and the whole pull is live: no dead zone.
    assert gripper_opening(0.0, 0.67) == pytest.approx(0.67)
    assert gripper_opening(0.5, 0.67) == pytest.approx(0.335)
    assert gripper_opening(1.0, 0.67) == pytest.approx(0.0)
    assert gripper_opening(0.0, 1.0) == pytest.approx(1.0)


def test_the_face_buttons_parse_and_a_short_gamepad_reads_them_released():
    hands = parse_hands(
        [controller("left", primary=True), controller("right", secondary=True)]
    )
    assert hands["left"].primary_button
    assert not hands["left"].secondary_button
    assert hands["right"].secondary_button
    assert not hands["right"].primary_button


def test_a_non_boolean_pressed_value_reads_as_released():
    # bool("false") is True; a squeeze must never be conjured out of a
    # malformed payload, so only the JSON boolean counts as held.
    device = controller("left")
    device["gamepad"]["buttons"][1]["pressed"] = "false"
    hands = parse_hands([device])
    assert hands["left"].squeezing is False


def test_a_non_controller_or_unknown_handedness_is_skipped():
    devices = [
        {"role": "hand", "handedness": "left"},
        controller("none"),
        controller("LEFT"),
        controller(["left"]),
        controller(None),
    ]
    assert parse_hands(devices) == {}


@pytest.mark.parametrize(
    "devices",
    [
        ["not a device"],
        [{"role": "controller", "handedness": "left", "gripPose": "x", "gamepad": {}}],
        [
            {
                "role": "controller",
                "handedness": "left",
                "gripPose": {"position": 3, "orientation": 4},
                "gamepad": {"buttons": [{}, {}]},
            }
        ],
        [
            {
                "role": "controller",
                "handedness": "left",
                "gripPose": {
                    "position": {"x": 0, "y": 0, "z": 0},
                    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                },
                "gamepad": {"buttons": [1.0, 2.0]},
            }
        ],
    ],
)
def test_malformed_browser_json_drops_the_device_not_the_frame(devices):
    # The payload is unauthenticated browser JSON; any shape must parse to
    # an empty frame rather than raise into the server callback.
    assert parse_hands(devices) == {}


def test_a_rising_edge_is_a_press_against_an_observed_release():
    edge = RisingEdge()
    # Already down when first observed, then held across an untracked gap.
    assert [edge.step(held) for held in (True, None, True)] == [False] * 3
    # Released, then pressed: one press, held through a gap.
    assert [edge.step(held) for held in (False, True, None, True)] == [
        False,
        True,
        False,
        False,
    ]
    # Untracked before any reading: still no press on the first observed down.
    fresh = RisingEdge()
    assert [fresh.step(held) for held in (None, True, False, True)] == [
        False,
        False,
        False,
        True,
    ]


def test_no_frame_yet_reads_as_no_sample():
    no_frame = SimpleNamespace(latest=lambda: None)
    assert fresh_sample(no_frame, "left", 0.25) is None


def test_a_recent_frame_yields_its_hand():
    source = ScriptedHeadset(hands={"left": squeezing_hand()})
    assert fresh_sample(source, "left", 0.25) is not None


def test_a_hand_missing_from_a_recent_frame_reads_as_no_sample():
    source = ScriptedHeadset(hands={"left": squeezing_hand()})
    assert fresh_sample(source, "right", 0.25) is None


def test_a_stale_frame_is_discarded_rather_than_held():
    # A frozen headset is still reporting whatever button it last held. Treating
    # that as current would leave the robot engaged to an absent operator, so
    # age, not content, decides.
    source = ScriptedHeadset(hands={"left": squeezing_hand()}, age_s=1.0)
    assert fresh_sample(source, "left", 0.25) is None


def stick_of(axes) -> Thumbstick:
    """The right thumbstick of a squeezing controller whose gamepad carries
    `axes`."""
    (sample,) = parse_hands([controller("right", squeezing=True, axes=axes)]).values()
    return sample.thumbstick


def test_the_thumbstick_is_read_from_the_xr_standard_axes_forward_and_left_positive():
    # The thumbstick is at axes 2 and 3; the Gamepad API gives -1 forward and
    # left. The touchpad's axes 0 and 1 drive nothing.
    assert stick_of([0.9, 0.9, 0.0, -1.0]) == Thumbstick(forward=1.0, left=0.0)
    assert stick_of([0.0, 0.0, -1.0, 0.0]) == Thumbstick(forward=0.0, left=1.0)
    assert stick_of([0.0, 0.0, 1.0, 1.0]) == Thumbstick(forward=-1.0, left=-1.0)


def test_a_deflection_within_the_rest_band_rests_and_one_past_it_is_rescaled():
    assert past_rest_band(0.05) == 0.0
    assert past_rest_band(-0.05) == 0.0
    # Half a deflection past the band: 0.05 + 0.5 * 0.95.
    assert past_rest_band(0.525) == pytest.approx(0.5)
    assert past_rest_band(-0.525) == pytest.approx(-0.5)
    assert past_rest_band(1.0) == pytest.approx(1.0)
    assert stick_of([0.0, 0.0, 0.05, -0.04]) == RESTING


@pytest.mark.parametrize(
    "axes",
    [
        None,
        "forward",
        {"3": -1.0},
        [0.0, 0.0],
        [0.0, 0.0, True, False],
        [0.0, 0.0, "-1", None],
        [0.0, 0.0, math.nan, -math.inf],
    ],
)
def test_a_missing_or_malformed_axis_reads_as_rest_and_keeps_the_hand(axes):
    device = controller("right", squeezing=True)
    device["gamepad"]["axes"] = axes
    (parsed,) = parse_hands([device]).values()
    assert parsed.squeezing
    assert parsed.thumbstick == RESTING


def test_an_axis_past_full_deflection_reads_as_full_deflection():
    assert stick_of([0.0, 0.0, -7.0, 3.0]) == Thumbstick(forward=-1.0, left=1.0)


def test_a_hands_squeeze_is_read_from_its_sample_and_unknown_without_one():
    released = parse_hands([controller("left")])["left"]
    frame = XrFrame(0.0, {"left": released, "right": squeezing_hand()})
    assert hand_squeezing(frame, Hand.RIGHT) is True
    assert hand_squeezing(frame, Hand.LEFT) is False
    assert hand_squeezing(XrFrame(0.0, {}), Hand.LEFT) is None
    assert hand_squeezing(None, Hand.LEFT) is None
