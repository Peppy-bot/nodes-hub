import asyncio

import pytest
from peppygen.consumed_services.sensor_streams import hold as sensor_streams_hold
from peppygen.consumed_services.sensor_streams import (
    release as sensor_streams_release,
)

from tests.helpers import FakeToken, boot, eventually
from xr_commander.devices import XrFrame
from xr_commander.sensor_hold import (
    HOLDER,
    Answer,
    ChosenRobotHold,
    SensorStreams,
    Timing,
    driven_robot,
    follow_chosen_robot,
)

HELD = Answer(success=True, message="held", lease_ms=9000)
RELEASED = Answer(success=True, message="released", lease_ms=9000)
REFUSED = Answer(success=False, message="no robot named alpha stands", lease_ms=0)


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeStreams:
    """The simulation's hold and release services as a pure test scripts
    them: every call recorded, each answered by the next scripted answer of
    its kind (an exception is raised), else by success."""

    def __init__(self, holds=(), releases=()):
        self.calls = []
        self._holds = list(holds)
        self._releases = list(releases)

    async def hold(self, robot):
        self.calls.append(("hold", robot))
        return self._next(self._holds, HELD)

    async def release(self, robot):
        self.calls.append(("release", robot))
        return self._next(self._releases, RELEASED)

    @staticmethod
    def _next(scripted, default):
        answer = scripted.pop(0) if scripted else default
        if isinstance(answer, Exception):
            raise answer
        return answer


def holding(streams, clock=None, timing=Timing()):
    return ChosenRobotHold(streams, timing, clock or FakeClock())


@pytest.mark.asyncio
async def test_the_first_choice_holds_every_sensor_of_the_robot():
    streams = FakeStreams()
    hold = holding(streams)

    await hold.follow("alpha")

    assert streams.calls == [("hold", "alpha")]


@pytest.mark.asyncio
async def test_a_changed_choice_releases_the_robot_held_before_it_holds_the_next():
    streams = FakeStreams()
    hold = holding(streams)

    await hold.follow("alpha")
    await hold.follow("bravo")

    assert streams.calls == [("hold", "alpha"), ("release", "alpha"), ("hold", "bravo")]


@pytest.mark.asyncio
async def test_the_hold_is_renewed_a_third_of_the_way_through_its_lease():
    streams = FakeStreams()
    clock = FakeClock()
    hold = holding(streams, clock)
    await hold.follow("alpha")

    clock.now += 2.875
    await hold.follow("alpha")
    assert streams.calls == [("hold", "alpha")]

    # A 9 s lease is renewed after 3 s, then 3 s after that renewal.
    clock.now += 0.125
    await hold.follow("alpha")
    clock.now += 3.0
    await hold.follow("alpha")
    assert streams.calls == [("hold", "alpha")] * 3


@pytest.mark.asyncio
async def test_a_short_lease_is_renewed_no_faster_than_the_choice_is_read():
    streams = FakeStreams(holds=[Answer(success=True, message="held", lease_ms=0)])
    clock = FakeClock()
    # Steps that are exact in binary, so the sum lands on the period.
    timing = Timing(follow_period_s=0.25)
    hold = holding(streams, clock, timing)
    await hold.follow("alpha")

    clock.now += 0.125
    await hold.follow("alpha")
    assert streams.calls == [("hold", "alpha")]
    clock.now += 0.125
    await hold.follow("alpha")
    assert streams.calls == [("hold", "alpha")] * 2


@pytest.mark.asyncio
async def test_no_robot_releases_the_robot_held_before_and_holds_nothing():
    streams = FakeStreams()
    clock = FakeClock()
    hold = holding(streams, clock)
    await hold.follow("alpha")

    await hold.follow(None)
    clock.now += 60.0
    await hold.follow(None)

    assert streams.calls == [("hold", "alpha"), ("release", "alpha")]


@pytest.mark.asyncio
async def test_no_robot_from_the_start_calls_nothing():
    streams = FakeStreams()
    hold = holding(streams)

    await hold.follow(None)
    await hold.stop()

    assert streams.calls == []


@pytest.mark.asyncio
async def test_a_refused_hold_is_logged_once_and_tried_again_after_the_retry_wait(
    capsys,
):
    streams = FakeStreams(holds=[REFUSED, REFUSED])
    clock = FakeClock()
    hold = holding(streams, clock, Timing(retry_s=1.0))
    await hold.follow("alpha")

    clock.now += 0.875
    await hold.follow("alpha")
    assert streams.calls == [("hold", "alpha")]

    clock.now += 0.125
    await hold.follow("alpha")
    assert streams.calls == [("hold", "alpha")] * 2
    log = capsys.readouterr().out
    assert log.count("sensor hold of alpha refused: no robot named alpha stands") == 1

    # The third try is accepted, and the log says that the robot is held.
    clock.now += 1.0
    await hold.follow("alpha")
    assert "holding the sensors of alpha" in capsys.readouterr().out
    assert streams.calls == [("hold", "alpha")] * 3


@pytest.mark.asyncio
async def test_a_failed_hold_call_is_tried_again_after_the_retry_wait(capsys):
    streams = FakeStreams(holds=[TimeoutError("no answer")])
    clock = FakeClock()
    hold = holding(streams, clock, Timing(retry_s=1.0))
    await hold.follow("alpha")
    assert "sensor hold of alpha failed: TimeoutError('no answer')" in (
        capsys.readouterr().out
    )

    clock.now += 1.0
    await hold.follow("alpha")

    assert streams.calls == [("hold", "alpha")] * 2
    assert "holding the sensors of alpha" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_refused_robot_is_still_released_when_the_choice_changes():
    # A hold whose answer was lost can have landed, so the robot is released
    # all the same.
    streams = FakeStreams(holds=[TimeoutError("no answer")])
    hold = holding(streams)

    await hold.follow("alpha")
    await hold.follow("bravo")

    assert streams.calls == [("hold", "alpha"), ("release", "alpha"), ("hold", "bravo")]


@pytest.mark.asyncio
async def test_a_failed_release_does_not_stop_the_next_hold(capsys):
    streams = FakeStreams(releases=[ConnectionError("gone")])
    hold = holding(streams)

    await hold.follow("alpha")
    await hold.follow("bravo")

    assert streams.calls == [("hold", "alpha"), ("release", "alpha"), ("hold", "bravo")]
    assert "sensor release of alpha failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_stop_releases_the_hold_and_no_hold_follows_it():
    streams = FakeStreams()
    clock = FakeClock()
    hold = holding(streams, clock)
    await hold.follow("alpha")

    await hold.stop()
    clock.now += 60.0
    await hold.follow("alpha")

    assert streams.calls == [("hold", "alpha"), ("release", "alpha")]


class BlockedHold(FakeStreams):
    """A simulation whose first hold answers only once the test lets it."""

    def __init__(self):
        super().__init__()
        self.hold_started = asyncio.Event()
        self.answer_hold = asyncio.Event()

    async def hold(self, robot):
        self.calls.append(("hold", robot))
        self.hold_started.set()
        await self.answer_hold.wait()
        return HELD


@pytest.mark.asyncio
async def test_stop_waits_for_a_hold_in_flight_so_no_hold_lands_after_the_release():
    streams = BlockedHold()
    hold = holding(streams)
    following = asyncio.create_task(hold.follow("alpha"))
    await streams.hold_started.wait()

    stopping = asyncio.create_task(hold.stop())
    # The stop waits on the call in flight: nothing is released yet.
    await asyncio.sleep(0)
    assert streams.calls == [("hold", "alpha")]

    streams.answer_hold.set()
    await asyncio.wait_for(asyncio.gather(following, stopping), 5.0)
    assert streams.calls == [("hold", "alpha"), ("release", "alpha")]


def test_the_headset_drives_the_chosen_robot_only_while_a_page_is_in_session():
    clock = FakeClock()
    frames = [None]
    driven = driven_robot(lambda: "alpha", lambda: frames[0], 5.0, clock)
    # No page sent a frame: nobody wears the headset.
    assert driven() is None
    frames[0] = XrFrame(received_monotonic_s=clock.now, hands={})
    assert driven() == "alpha"
    # A gap up to the timeout keeps the session.
    clock.now += 5.0
    assert driven() == "alpha"
    clock.now += 0.001
    assert driven() is None


def test_a_session_without_a_robot_paired_drives_none():
    clock = FakeClock()
    frame = XrFrame(received_monotonic_s=clock.now, hands={})
    assert driven_robot(lambda: None, lambda: frame, 5.0, clock)() is None


@pytest.mark.asyncio
async def test_the_end_of_the_session_releases_the_robot_held():
    streams = FakeStreams()
    clock = FakeClock()
    frames = [XrFrame(received_monotonic_s=clock.now, hands={})]
    hold = holding(streams, clock)
    driven = driven_robot(lambda: "alpha", lambda: frames[0], 5.0, clock)
    await hold.follow(driven())
    clock.now += 6.0
    await hold.follow(driven())
    assert streams.calls == [("hold", "alpha"), ("release", "alpha")]


@pytest.mark.asyncio
async def test_the_loop_follows_the_choice_until_the_token_cancels():
    streams = FakeStreams()
    hold = holding(streams, timing=Timing(follow_period_s=0.001))
    chosen = ["alpha"]
    token = FakeToken()
    task = asyncio.create_task(follow_chosen_robot(hold, lambda: chosen[0], token))
    try:
        await eventually(lambda: ("hold", "alpha") in streams.calls, "alpha held")
        chosen[0] = "bravo"
        await eventually(lambda: ("hold", "bravo") in streams.calls, "bravo held")
        assert streams.calls[:3] == [
            ("hold", "alpha"),
            ("release", "alpha"),
            ("hold", "bravo"),
        ]
    finally:
        token.cancel()
        await asyncio.wait_for(task, 5.0)


def sensor_state(module, success=True, message="held", lease_ms=9000):
    """A hold or release answer with no sensor listed."""
    return module.ResponseData(
        success=success,
        message=message,
        lease_ms=lease_ms,
        sensors=[],
        rgb_cameras=[],
        rgbd_cameras=[],
    )


def wired_streams(h):
    target = sensor_streams_hold.bound_producer(h.node_runner)
    assert target is not None
    return SensorStreams(
        h.node_runner,
        target,
        hold_module=sensor_streams_hold,
        release_module=sensor_streams_release,
        timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_the_hold_and_its_release_reach_the_simulation_over_the_wire():
    async with boot() as h:
        mock = h.mocks.deps.sensor_streams
        mock.hold.enqueue_response(sensor_state(sensor_streams_hold))
        mock.release.enqueue_response(sensor_state(sensor_streams_release))
        hold = holding(wired_streams(h))

        await hold.follow("alpha")
        await hold.stop()

        (held,) = mock.hold.captured()
        assert (held.robot, held.holder, held.sensors) == ("alpha", HOLDER, [])
        (released,) = mock.release.captured()
        assert (released.robot, released.holder) == ("alpha", HOLDER)


@pytest.mark.asyncio
async def test_a_refusal_over_the_wire_is_tried_again():
    async with boot() as h:
        mock = h.mocks.deps.sensor_streams
        mock.hold.enqueue_response(
            sensor_state(
                sensor_streams_hold, success=False, message="no robot", lease_ms=0
            )
        )
        mock.hold.enqueue_response(sensor_state(sensor_streams_hold))
        clock = FakeClock()
        hold = holding(wired_streams(h), clock, Timing(retry_s=1.0))

        await hold.follow("alpha")
        clock.now += 1.0
        await hold.follow("alpha")

        assert [request.robot for request in mock.hold.captured()] == ["alpha"] * 2


@pytest.mark.asyncio
async def test_a_vacant_slot_binds_no_simulation_to_hold_sensors_of():
    async with boot(sensor_streams_vacant=True) as h:
        assert sensor_streams_hold.bound_producer(h.node_runner) is None
