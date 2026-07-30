import math

import numpy as np
import pytest

from xr_commander.devices import gripper_opening, parse_hands


def controller(
    handedness,
    *,
    position=(0.0, 0.0, 0.0),
    squeezing=False,
    trigger=0.0,
    primary=False,
):
    buttons = [
        {"pressed": trigger >= 1.0, "touched": True, "value": trigger},
        {
            "pressed": squeezing,
            "touched": True,
            "value": 1.0 if squeezing else 0.0,
        },
    ]
    if primary:
        buttons += [
            {"pressed": False, "touched": False, "value": 0.0},
            {"pressed": False, "touched": False, "value": 0.0},
            {"pressed": True, "touched": True, "value": 0.0},
        ]
    return {
        "role": "controller",
        "handedness": handedness,
        "gripPose": {
            "position": dict(zip("xyz", position, strict=True)),
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "gamepad": {
            "buttons": buttons,
            "axes": [0.0, 0.0, 0.0, 0.0],
        },
    }


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
    assert np.allclose(hands["right"].pose.position, [0.0, 2.0, 0.0])


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


def test_an_unusable_pose_drops_that_hand_rather_than_the_frame():
    broken = controller("left")
    broken["gripPose"]["orientation"] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}
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


def test_the_a_button_parses_and_a_short_gamepad_reads_it_released():
    hands = parse_hands([controller("left", primary=True), controller("right")])
    assert hands["left"].primary_button
    assert not hands["right"].primary_button


def test_a_non_controller_or_unknown_handedness_is_skipped():
    devices = [
        {"role": "hand", "handedness": "left"},
        controller("none"),
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
