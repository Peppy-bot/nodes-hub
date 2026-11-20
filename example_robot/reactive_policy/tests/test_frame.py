"""The frame the policy names for a subgoal's target, and the targets it
adopts and drops."""

import asyncio

from peppygen.consumed_topics.camera import video_stream
from peppygen.paired_topics.deliberation import subgoal

from reactive_policy.__main__ import Situation, frame_of, track_camera

POSITION = [0.0, 0.0, 0.5]
CAMERA_FRAME = "camera_color_optical_frame"
TTL_MS = 60_000


def target_in(frame):
    return subgoal.MessageTarget(frame=frame, timestamp=None, position=POSITION)


def situation_seeing(camera_frame):
    """A Situation whose camera's latest picture names `camera_frame`, None
    before any picture."""
    state = Situation(TTL_MS)
    state.camera_frame = camera_frame
    return state


def test_a_target_with_a_frame_lies_in_that_frame():
    assert frame_of(target_in("wrist_optical_frame")) == "wrist_optical_frame"


def test_a_target_with_no_frame_lies_in_the_robot_frame():
    assert frame_of(target_in(None)) == "the robot frame"


def test_a_target_with_no_frame_is_adopted():
    state = situation_seeing(CAMERA_FRAME)
    target = target_in(None)
    assert state.offer(1, target)
    assert state.authoritative_subgoal() == (1, target)


def test_a_target_in_the_frame_of_the_cameras_pictures_is_adopted():
    state = situation_seeing(CAMERA_FRAME)
    target = target_in(CAMERA_FRAME)
    assert state.offer(1, target)
    assert state.authoritative_subgoal() == (1, target)


def test_a_target_in_another_frame_is_dropped_and_the_first_drop_alone_logs_unknown_frame(
    capsys,
):
    state = situation_seeing(CAMERA_FRAME)
    in_force = target_in(None)
    state.offer(1, in_force)

    assert not state.offer(2, target_in("map"))
    assert not state.offer(3, target_in("odom"))

    assert state.authoritative_subgoal() == (1, in_force)
    assert capsys.readouterr().out.splitlines() == [
        (
            "[reactive_policy] dropped subgoal 2: unknown_frame: its target lies in "
            "'map', which this policy cannot place; give a target no frame, for the "
            "robot frame, or the frame of the camera's pictures, "
            "'camera_color_optical_frame'. Later drops are not logged."
        )
    ]


def test_before_any_picture_a_target_with_a_frame_is_dropped_naming_why(capsys):
    state = situation_seeing(None)

    assert not state.offer(1, target_in(CAMERA_FRAME))

    assert state.authoritative_subgoal() is None
    assert capsys.readouterr().out.splitlines() == [
        (
            "[reactive_policy] dropped subgoal 1: unknown_frame: its target lies in "
            "'camera_color_optical_frame', which this policy cannot place; give a "
            "target no frame, for the robot frame, or the frame of the camera's "
            "pictures, which no picture has named yet. Later drops are not logged."
        )
    ]


def test_the_camera_frame_is_the_frame_the_latest_picture_names(monkeypatch):
    def picture(frame, capture_index):
        header = video_stream.MessageHeader(
            frame=frame, timestamp=1.0, capture_index=capture_index
        )
        return None, video_stream.Message(
            header=header, encoding="rgb8", width=1, height=1, data=b"\0\0\0"
        )

    pictures = [picture(CAMERA_FRAME, 0), picture("wrist_optical_frame", 1), None]

    class Pictures:
        async def next(self):
            return pictures.pop(0)

    async def subscribe(_node_runner):
        return Pictures()

    monkeypatch.setattr(video_stream, "subscribe", subscribe)
    state = situation_seeing(None)

    asyncio.run(track_camera(None, state))

    assert (state.frames_seen, state.camera_frame) == (2, "wrist_optical_frame")
