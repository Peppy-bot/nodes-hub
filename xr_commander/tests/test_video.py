import asyncio
import time

import cv2
import numpy as np
import pytest
from teleop_xr.config import ViewConfig

from peppygen.consumed_topics.color_cameras import video_stream as color_video_stream

from tests.helpers import FakeToken, RecordingSink, boot, eventually, running_drain
from xr_commander.video import (
    CAMERA_VIEW,
    CameraTrack,
    assert_unique_track_ids,
    camera_views,
    decode_to_bgr,
    discover_tracks,
    drain_frames,
    no_signal_frame,
    shrink_to_width,
    watch_camera_silence,
)


def test_bgr8_passes_through_with_its_geometry_intact():
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3)
    decoded = decode_to_bgr("bgr8", 2, 3, frame.tobytes())
    assert decoded.shape == (3, 2, 3)
    assert np.array_equal(decoded, frame)
    # An owned, writable copy: a view would alias the recycled wire buffer.
    assert decoded.flags["OWNDATA"] and decoded.flags["WRITEABLE"]


def test_rgb8_is_reordered_because_the_encoder_expects_bgr():
    # teleop_xr's track runs a BGR-to-RGB conversion before encoding, so a frame
    # handed over in RGB would reach the headset with red and blue swapped.
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)  # pure red, RGB order
    decoded = decode_to_bgr("rgb8", 1, 1, rgb.tobytes())
    assert tuple(decoded[0, 0]) == (0, 0, 255)  # pure red, BGR order


def test_a_contiguous_array_is_produced_for_the_encoder():
    decoded = decode_to_bgr("rgb8", 4, 2, np.zeros((2, 4, 3), dtype=np.uint8).tobytes())
    assert decoded.flags["C_CONTIGUOUS"]


def test_yuyv_decodes_to_the_declared_geometry():
    raw = np.full((2, 4, 2), 128, dtype=np.uint8)
    decoded = decode_to_bgr("yuyv", 4, 2, raw.tobytes())
    assert decoded.shape == (2, 4, 3)


def test_mjpeg_round_trips_through_the_jpeg_decoder():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    # The declared geometry is deliberately wrong: a compressed frame carries
    # its own, and the header's is not used to reshape it.
    decoded = decode_to_bgr("mjpeg", 999, 999, buffer.tobytes())
    assert decoded.shape == (8, 8, 3)


def test_an_undecodable_mjpeg_payload_is_rejected():
    with pytest.raises(ValueError, match="failed to decode"):
        decode_to_bgr("mjpeg", 8, 8, b"not a jpeg")


def test_an_unknown_encoding_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unsupported encoding"):
        decode_to_bgr("z16", 2, 2, b"\x00" * 8)


def test_a_payload_that_contradicts_its_header_is_refused():
    # A silently wrong reshape would either throw somewhere far less
    # informative or reinterpret the wrong bytes as an image.
    with pytest.raises(ValueError, match="expected 12"):
        decode_to_bgr("rgb8", 2, 2, b"\x00" * 11)


def test_camera_views_names_every_track_with_the_shared_view_shape():
    views = camera_views(["wrist_left", "chest"])
    assert views == {"wrist_left": CAMERA_VIEW, "chest": CAMERA_VIEW}
    assert views["wrist_left"] is not CAMERA_VIEW  # callers own their copy


def test_the_view_shape_satisfies_the_installed_frontend_config():
    # The panel config is teleop_xr's schema; a shape drift must fail here,
    # not at node boot.
    ViewConfig(**CAMERA_VIEW)


def test_colliding_track_ids_are_refused():
    # A camera instance named after the status panel collides exactly here.
    with pytest.raises(ValueError, match="collide"):
        assert_unique_track_ids(["chest", "chest"])
    assert_unique_track_ids(["chest", "status"])


async def test_every_bound_producer_gets_a_track_named_by_its_instance():
    async with boot(color_cameras_instances=2) as h:
        bound = color_video_stream.bound_producers(h.node_runner)
        assert len(bound) == 2
        tracks = discover_tracks(h.node_runner, color_video_stream, RecordingSink)
        assert [t.instance_id for t in tracks] == [p.instance_id for p in bound]
        assert len(set(t.instance_id for t in tracks)) == 2
        assert all(isinstance(t.sink, RecordingSink) for t in tracks)


def bgr_message(width=1, height=1):
    data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    return color_video_stream.Message(
        header=color_video_stream.MessageHeader(timestamp=0.0, frame_id=0),
        encoding="bgr8",
        width=width,
        height=height,
        frame=data,
    )


async def test_an_undecodable_frame_is_dropped_and_logged_once(capsys):
    async with boot(color_cameras_instances=1) as h:
        camera = h.mocks.deps.color_cameras[0]
        cam_id = color_video_stream.bound_producers(h.node_runner)[0].instance_id
        sink = RecordingSink()
        health: dict[str, float] = {}
        async with running_drain(
            lambda token: drain_frames(
                h.node_runner,
                color_video_stream,
                {cam_id: sink},
                token,
                "test",
                0,
                health,
            )
        ):
            bad = color_video_stream.Message(
                header=color_video_stream.MessageHeader(timestamp=0.0, frame_id=0),
                encoding="z16",
                width=1,
                height=1,
                frame=b"xx",
            )
            await camera.video_stream.publish(bad)
            await camera.video_stream.publish(bad)
            await camera.video_stream.publish(bgr_message())
            # Per-producer order holds, so the landed good frame proves both
            # bad ones were already processed.
            await eventually(lambda: len(sink.frames) == 1, message="the good frame")
        assert capsys.readouterr().out.count("frame unusable") == 1  # latched
        assert cam_id in health  # stamped for the delivered frame only


def test_a_failed_camera_subscribe_is_loud_and_fatal_to_the_drain(capsys):
    # A refusing runner cannot be scripted over the real wire; the stub
    # stands in for the transport failing the subscribe.
    class ExplodingTopic:
        async def subscribe(self, _runner):
            raise RuntimeError("no transport")

    async def run():
        await asyncio.wait_for(
            drain_frames(None, ExplodingTopic(), {}, FakeToken(), "camera"),
            1.0,
        )

    asyncio.run(run())
    assert "subscribe failed" in capsys.readouterr().out


def test_no_signal_frame_is_drawn_and_sized():
    frame = no_signal_frame("wrist_left")
    assert frame.shape == (180, 320, 3)
    assert frame.any()  # the text actually rendered


def test_a_silent_camera_is_blanked_once_and_recovery_logged(capsys):
    sink = RecordingSink()
    track = CameraTrack(instance_id="cam", sink=sink)
    health = {"cam": time.monotonic()}
    token = FakeToken()

    async def run():
        task = asyncio.create_task(
            watch_camera_silence(
                [track], health, token, silent_after_s=0.03, poll_s=0.01
            )
        )
        await asyncio.sleep(0.09)
        assert len(sink.frames) == 1  # blanked once, not per poll
        health["cam"] = time.monotonic()  # the camera resumes
        await asyncio.sleep(0.03)
        token.cancel()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "went silent" in out
    assert "recovered" in out


def test_a_camera_that_never_produces_is_blanked_from_the_watcher_start():
    # A dead camera and no camera look identical as a blank panel; the timer
    # starts with the watcher so the panel says NO SIGNAL instead.
    sink = RecordingSink()
    track = CameraTrack(instance_id="cam", sink=sink)
    token = FakeToken()

    async def run():
        task = asyncio.create_task(
            watch_camera_silence([track], {}, token, silent_after_s=0.03, poll_s=0.01)
        )
        await asyncio.sleep(0.09)
        token.cancel()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(run())
    assert len(sink.frames) == 1  # blanked once, not per poll


def test_shrink_keeps_aspect_and_caps_width():
    tall = np.zeros((1200, 1920, 3), dtype=np.uint8)
    small = shrink_to_width(tall, 640)
    assert small.shape == (400, 640, 3)


def test_shrink_never_upscales_and_zero_disables():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert shrink_to_width(frame, 640) is frame
    assert shrink_to_width(frame, 0) is frame
