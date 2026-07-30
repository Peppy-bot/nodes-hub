from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from teleop_xr.config import ViewConfig

from tests.helpers import RecordingSink
from xr_commander.video import (
    CAMERA_VIEW,
    CameraTrack,
    assert_unique_track_ids,
    camera_views,
    decode_to_bgr,
    discover_tracks,
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
    tracks = [
        CameraTrack(instance_id="cam_a", track_id="wrist_left", sink=RecordingSink()),
        CameraTrack(instance_id="cam_b", track_id="chest", sink=RecordingSink()),
    ]
    views = camera_views(tracks)
    assert views == {"wrist_left": CAMERA_VIEW, "chest": CAMERA_VIEW}
    assert views["wrist_left"] is not CAMERA_VIEW  # callers own their copy


def test_the_view_shape_satisfies_the_installed_frontend_config():
    # The panel config is teleop_xr's schema; a shape drift must fail here,
    # not at node boot.
    ViewConfig(**CAMERA_VIEW)


def test_colliding_track_ids_are_refused():
    tracks = [
        CameraTrack(instance_id="cam_a", track_id="chest", sink=RecordingSink()),
        CameraTrack(instance_id="cam_b", track_id="chest", sink=RecordingSink()),
    ]
    with pytest.raises(ValueError, match="collide"):
        assert_unique_track_ids(tracks)
    assert_unique_track_ids(tracks[:1])


class FakeTopics:
    def __init__(self, producers):
        self._producers = producers

    def bound_producers(self, _runner):
        return self._producers


def discover(instance_ids, names=None):
    producers = [SimpleNamespace(instance_id=i) for i in instance_ids]
    return discover_tracks(None, FakeTopics(producers), RecordingSink, names or {})


def test_every_bound_producer_gets_a_track_named_through_the_mapping():
    tracks = discover(["cam_a", "cam_b"], names={"cam_a": "wrist_left"})
    assert [(t.instance_id, t.track_id) for t in tracks] == [
        ("cam_a", "wrist_left"),
        ("cam_b", "cam_b"),
    ]
