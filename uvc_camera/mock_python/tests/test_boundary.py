"""Wire-level boundary tests: the node's real `setup` booted through the
generated harness against an ephemeral, per-test Zenoh router.

The harness observes the emitted `camera/video_stream` topic and polls the
exposed `camera/*` services from a fresh caller identity, the same path a
real consumer node would take, and plays the robot's sensor_mounts with the
generated mock, or leaves the sensor_mounts link vacant. No daemon, no
launcher.
"""

import asyncio
import re

import pytest
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_services.camera import set_exposure, video_stream_info
from peppygen.mock.deps.sensor_mounts import DescribeSensorsResponseData

from camera_core_py.sensor_mounts import SensorMountsError
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

# The name the camera runs under, and the frame the robot gives it.
CAMERA = "wrist_left"
CAMERA_FRAME = "wrist_left_color_optical_frame"

# How many pictures a test reads to see that each names the same frame.
PICTURES_READ = 3


def robot_sensors() -> DescribeSensorsResponseData:
    """A robot with a chest camera and a wrist camera on its left arm."""
    return DescribeSensorsResponseData(
        robot_frame="base_footprint",
        sensor_names=["chest", CAMERA],
        sensor_kinds=["rgbd_camera", "rgb_camera"],
        sensor_frames=["chest_color_optical_frame", CAMERA_FRAME],
        carried_by=["", "left_arm"],
    )


async def answer_sensors(h, answer: DescribeSensorsResponseData) -> None:
    """Answers the node's ask of describe_sensors at its start."""
    responder = await h.mocks.deps.sensor_mounts.describe_sensors.next_request(
        SERVICE_TIMEOUT
    )
    await responder.respond(answer)


async def mounted_camera():
    """The camera of the robot `robot_sensors` describes, with its ask
    answered."""
    h = await harness.start(setup, instance_id=CAMERA)
    await answer_sensors(h, robot_sensors())
    return h


async def next_picture(h):
    picture = await asyncio.wait_for(
        h.emitted.camera_video_stream.next(), FIRST_FRAME_TIMEOUT
    )
    assert picture is not None, "fixture session closed before the picture"
    return picture


async def test_boot_emits_frames_shaped_by_parameter_defaults():
    """Booting emits real video_stream frames whose shape matches the schema
    defaults, and video_stream_info reports the same geometry plus the bundled
    asset's probed frame rate."""
    h = await mounted_camera()
    try:
        picture = await next_picture(h)
        assert picture.encoding == EXPECTED_ENCODING
        assert picture.width == EXPECTED_WIDTH
        assert picture.height == EXPECTED_HEIGHT
        assert (
            len(picture.data) == EXPECTED_WIDTH * EXPECTED_HEIGHT * RGB8_BYTES_PER_PIXEL
        )
        assert picture.header.capture_index >= 0
        assert picture.header.timestamp > 0

        info = await video_stream_info.poll(h, SERVICE_TIMEOUT)
        assert info.width == EXPECTED_WIDTH
        assert info.height == EXPECTED_HEIGHT
        assert info.encoding == EXPECTED_ENCODING
        # The service reports the source asset's real frame rate (probed at
        # setup from robot.mp4), not the video.frame_rate parameter.
        assert info.frames_per_second == get_source_video_fps(ASSETS_DIR / "robot.mp4")
    finally:
        await h.shutdown()


async def test_each_picture_names_the_frame_sensor_mounts_gives_its_name():
    """The camera asks describe_sensors once at its start, and each picture
    names the frame the answer gives the name the camera runs under, with a
    capture index that counts the pictures."""
    h = await mounted_camera()
    try:
        pictures = [await next_picture(h) for _ in range(PICTURES_READ)]
        assert [p.header.frame for p in pictures] == [CAMERA_FRAME] * PICTURES_READ
        indices = [p.header.capture_index for p in pictures]
        assert indices == sorted(set(indices))
        assert h.mocks.deps.sensor_mounts.describe_sensors.captured_count() == 1
    finally:
        await h.shutdown()


async def test_with_its_sensor_mounts_link_vacant_each_picture_names_its_name():
    """With its sensor_mounts link vacant, the camera asks nothing, and each
    picture names the name the camera runs under."""
    h = await harness.start(setup, instance_id=CAMERA, sensor_mounts_vacant=True)
    try:
        assert h.mocks.deps.sensor_mounts is None
        pictures = [await next_picture(h) for _ in range(PICTURES_READ)]
        assert [p.header.frame for p in pictures] == [CAMERA] * PICTURES_READ
    finally:
        await h.shutdown()


async def test_an_answer_without_the_camera_fails_the_start():
    """An answer of describe_sensors that lists no camera of the name the
    node runs under fails its start, with the names the answer lists and
    what to run."""
    h = await harness.start(setup, instance_id="head")
    await answer_sensors(h, robot_sensors())
    await asyncio.wait_for(setup_end(h), SERVICE_TIMEOUT)
    refusal = (
        'sensor_mounts\' describe_sensors names no sensor "head" (its sensor_names: '
        "chest, wrist_left): run the camera under one of those names, or bind its "
        "sensor_mounts link to the robot that carries it, whose robot_identity "
        "implementer serves sensor_mounts"
    )
    with pytest.raises(SensorMountsError, match=re.escape(refusal)):
        await h.shutdown()


async def setup_end(h) -> None:
    """Returns once the node's setup has ended, for a test that reads its
    outcome: a shutdown before it would cancel the setup."""
    while not h.setup_finished():
        await asyncio.sleep(0.01)


async def test_set_exposure_acks_and_echoes_requested_value():
    """The mock's control services answer every request with the requested
    value echoed back as current_value (there is no hardware to adjust)."""
    h = await mounted_camera()
    try:
        for requested in (125, 7):
            response = await set_exposure.poll(
                h,
                set_exposure.RequestData(mode="manual", value=requested),
                SERVICE_TIMEOUT,
            )
            assert response.current_value == requested
    finally:
        await h.shutdown()


async def test_shutdown_is_clean_while_streaming():
    """Shutting the harness down while the decoder is mid-stream runs the
    node's shutdown hooks (decoder stop + consumer wake) without raising."""
    h = await mounted_camera()
    await next_picture(h)
    assert h.setup_finished()
    # Shutdown runs node convergence (including the stop_decoder hook), then
    # the fixture session and router. Reaching the end of the test without an
    # exception is the assertion.
    await h.shutdown()
