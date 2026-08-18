"""Wire-level boundary tests: the node's real `setup` booted through the
generated harness against an ephemeral, per-test Zenoh router.

The harness observes the emitted `camera/video_stream` topic and polls the
exposed `camera/*` services from a fresh caller identity — the same path a
real consumer node would take. No daemon, no launcher, no sleeps.
"""

import asyncio

from peppygen.fixtures import harness
from peppygen.fixtures.exposed_services.camera import set_exposure, video_stream_info

from uvc_camera_python_mock.__main__ import ASSETS_DIR, get_source_video_fps, setup

# With no parameter override, the harness hydrates the schema defaults from
# peppy.json5: video.resolution 1920x1080, video.topic_encoding "rgb8".
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
EXPECTED_ENCODING = "rgb8"
RGB8_BYTES_PER_PIXEL = 3

# The first frame needs the bundled AV1 asset opened, one frame decoded
# single-threaded and reformatted to 1080p — generous but bounded.
FIRST_FRAME_TIMEOUT = 30.0
SERVICE_TIMEOUT = 10.0


async def test_boot_emits_frames_shaped_by_parameter_defaults():
    """Booting emits real video_stream frames whose shape matches the schema
    defaults, and video_stream_info reports the same geometry plus the bundled
    asset's probed frame rate."""
    async with harness.start(setup) as h:
        frame = await asyncio.wait_for(
            h.emitted.camera_video_stream.next(), FIRST_FRAME_TIMEOUT
        )
        assert frame is not None, "fixture session closed before the first frame"
        assert frame.encoding == EXPECTED_ENCODING
        assert frame.width == EXPECTED_WIDTH
        assert frame.height == EXPECTED_HEIGHT
        assert (
            len(frame.frame) == EXPECTED_WIDTH * EXPECTED_HEIGHT * RGB8_BYTES_PER_PIXEL
        )
        assert frame.header.frame_id >= 0
        assert frame.header.timestamp > 0

        info = await video_stream_info.poll(h, SERVICE_TIMEOUT)
        assert info.width == EXPECTED_WIDTH
        assert info.height == EXPECTED_HEIGHT
        assert info.encoding == EXPECTED_ENCODING
        # The service reports the source asset's real frame rate (probed at
        # setup from robot.mp4), not the video.frame_rate parameter.
        assert info.frames_per_second == get_source_video_fps(ASSETS_DIR / "robot.mp4")


async def test_set_exposure_acks_and_echoes_requested_value():
    """The mock's control services acknowledge every request: success=True, a
    fixed "mock: <name> acknowledged" message, and the requested value echoed
    back as current_value (there is no hardware to adjust)."""
    async with harness.start(setup) as h:
        for requested in (125, 7):
            response = await set_exposure.poll(
                h,
                set_exposure.RequestData(mode="manual", value=requested),
                SERVICE_TIMEOUT,
            )
            assert response.success is True
            assert response.message == "mock: set_exposure acknowledged"
            assert response.current_value == requested


async def test_shutdown_is_clean_while_streaming():
    """Exiting the harness context while the decoder is mid-stream runs the
    node's shutdown hooks (decoder stop + consumer wake) without raising."""
    async with harness.start(setup) as h:
        frame = await asyncio.wait_for(
            h.emitted.camera_video_stream.next(), FIRST_FRAME_TIMEOUT
        )
        assert frame is not None
        assert h.setup_finished()
    # The context-manager exit ran shutdown: node convergence (including the
    # stop_decoder hook), then the fixture session and router. Reaching this
    # line without an exception is the assertion.
