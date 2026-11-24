"""Tests for the clock half of SimTopicIO: recording the engine clock,
stamping from it (object-state captures included), the guarded publish
chain, and the guarded object-state stream. peppylib and peppygen exist
only inside the node's image, so the runtime fakes are installed before the
import; everything under test is pure python.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from sim_robot_core.object_state import ObjectRecord, ObjectStateSnapshot, RobotRecord

import runtime_fakes  # installs the runtime sim_topics imports

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))

import sim_topics  # noqa: E402  (needs the runtime fakes)


class _FakeClockPublisher:
    """Stands in for peppylib's ClockPublisher: records every publish, can
    hold publishes open behind a gate, and can fail specific instants."""

    def __init__(self) -> None:
        self.published = []
        self.gate = None
        self.failing = set()

    @property
    def domain(self):
        return "harness@cn-a"

    async def publish(self, time_ns: int) -> None:
        self.published.append(time_ns)
        if self.gate is not None:
            await self.gate.wait()
        if time_ns in self.failing:
            raise RuntimeError("the domain's tick was not sent")


@pytest.fixture(name="loop")
def loop_fixture():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class _Robots:
    """The scene's robots, which a pair's copy names. The clock half of the
    bridge never reads one."""

    def sole_name(self):
        return None


@pytest.fixture(name="io")
def io_fixture(loop):
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=_Robots())
    io._clock_publisher = _FakeClockPublisher()
    return io


async def _drain(loop_turns: int = 10) -> None:
    """Let scheduled callbacks and done-callbacks run: pure scheduling, no
    wall-clock dependence."""
    for _ in range(loop_turns):
        await asyncio.sleep(0)


def test_a_consumer_stamps_from_its_bound_clock(loop, monkeypatch):
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=_Robots())
    monkeypatch.setattr(sim_topics.clock, "now_ns", lambda: 5_000_000_000)
    assert io.timestamp_s() == 5.0
    # And the publisher paths are inert: nothing declared, nothing recorded.
    io.record_engine_time(1.0)
    io.publish_clock_tick()
    assert io.timestamp_s() == 5.0


def test_a_publisher_stamps_from_its_recorded_engine_clock(io):
    io.record_engine_time(1.25)
    assert io.timestamp_s() == 1.25
    io.record_engine_time(1.5)
    assert io.timestamp_s() == 1.5


def test_a_publisher_stamps_an_earlier_engine_instant_as_itself(io):
    # A scan's ray 0, due 0.15 s before the step its last ray completes.
    io.record_engine_time(1.25)
    assert io.timestamp_at_s(1.1, 1.25) == 1.1


def test_a_consumer_stamps_an_earlier_engine_instant_that_long_before_its_bound_clocks_now(
    loop, monkeypatch
):
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=_Robots())
    monkeypatch.setattr(sim_topics.clock, "now_ns", lambda: 5_000_000_000)
    assert io.timestamp_at_s(1.1, 1.25) == pytest.approx(4.85)


def test_a_publisher_must_record_before_stamping_or_publishing(io):
    with pytest.raises(RuntimeError, match="record an engine step"):
        io.timestamp_s()
    with pytest.raises(RuntimeError, match="record an engine step"):
        io.publish_clock_tick()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.001])
def test_a_diverged_engine_clock_is_rejected_at_the_boundary(io, bad):
    with pytest.raises(ValueError, match="non-publishable instant"):
        io.record_engine_time(bad)


@pytest.mark.parametrize("engine_time_s", [0.0, 5e-10])
def test_an_engine_clock_below_one_nanosecond_still_carries_an_instant(io, loop, engine_time_s):
    """Zero is the clock topic's not-ready sentinel, so an engine sitting at or
    below one nanosecond is floored rather than refused: the stamp and the tick
    carry the same instant, and neither is zero."""
    io.record_engine_time(engine_time_s)
    assert io.timestamp_s() > 0.0
    io.publish_clock_tick()
    loop.run_until_complete(_drain())
    assert io._clock_publisher.published == [1]

def test_publish_lands_the_recorded_instant_in_nanoseconds(io, loop):
    io.record_engine_time(2.5)
    io.publish_clock_tick()
    loop.run_until_complete(_drain())
    assert io._clock_publisher.published == [2_500_000_000]


@pytest.mark.parametrize("engine_time_s, stamp_ns", [(2.01, 2_010_000_000), (4.1, 4_100_000_000)])
def test_the_tick_carries_the_instant_of_the_stamps_of_its_step(io, loop, engine_time_s, stamp_ns):
    """As a float, 4.1 s lies just under 4 100 000 000 ns: the wire carries a
    stamp of it at the nearest nanosecond, and the tick carries that same
    instant, so a consumer whose clock reads the tick has reached the stamp."""
    io.record_engine_time(engine_time_s)
    io.publish_clock_tick()
    loop.run_until_complete(_drain())
    stamp = runtime_fakes.convert_time(io.timestamp_s())
    assert stamp.sec * runtime_fakes.NANOS_PER_SEC + stamp.nsec == stamp_ns
    assert io._clock_publisher.published == [stamp_ns]


def test_a_tick_arriving_mid_flight_is_latched_never_reordered(io, loop):
    publisher = io._clock_publisher

    async def scenario():
        publisher.gate = asyncio.Event()
        io._publish_clock_tick_on_loop(100)
        await _drain()
        # Two more while the first is out: only the newest survives.
        io._publish_clock_tick_on_loop(200)
        io._publish_clock_tick_on_loop(300)
        assert publisher.published == [100]
        publisher.gate.set()
        await _drain()
        assert publisher.published == [100, 300]

    loop.run_until_complete(scenario())


def test_publish_failures_are_latched_to_one_line_each_way(io, loop, caplog):
    publisher = io._clock_publisher
    publisher.failing = {1_000_000_000, 2_000_000_000}

    async def scenario():
        for instant in [1_000_000_000, 2_000_000_000, 3_000_000_000]:
            io._publish_clock_tick_on_loop(instant)
            await _drain()

    with caplog.at_level("INFO"):
        loop.run_until_complete(scenario())
    assert publisher.published == [1_000_000_000, 2_000_000_000, 3_000_000_000]
    down = [r for r in caplog.records if "is not being published" in r.message]
    recovered = [r for r in caplog.records if "is being published again" in r.message]
    assert len(down) == 1, "two consecutive failures log one line"
    assert len(recovered) == 1, "recovery logs one line"


def test_a_capture_is_stamped_on_the_state_timeline_or_not_at_all(io, loop, monkeypatch):
    # A publisher that has not recorded a step has no instant to give.
    assert io.capture_timestamp_s() is None
    io.record_engine_time(3.25)
    assert io.capture_timestamp_s() == 3.25

    follower = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=_Robots())
    monkeypatch.setattr(sim_topics.clock, "now_ns", lambda: 7_000_000_000)
    assert follower.capture_timestamp_s() == 7.0


def test_the_placements_stand_at_the_start_of_the_timeline_until_the_first_step(
    io, loop, monkeypatch
):
    # A publisher that has not recorded a step stamps the robots' placements
    # at the instant a timeline at zero carries, and at the engine clock once
    # it steps.
    assert io.placement_timestamp_s(None) == sim_topics._MIN_ENGINE_TIME_S
    io.record_engine_time(3.25)
    assert io.placement_timestamp_s(None) == 3.25
    # The world's records hold at the step that recorded them.
    assert io.placement_timestamp_s(2.5) == 2.5

    follower = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=_Robots())
    monkeypatch.setattr(sim_topics.clock, "now_ns", lambda: 7_000_000_000)
    assert follower.placement_timestamp_s(None) == 7.0


class _GatedPublisher:
    """A topic publisher whose publishes stay in flight until the gate opens."""

    def __init__(self) -> None:
        self.payloads = []
        self.gate = None

    async def publish(self, payload) -> None:
        self.payloads.append(payload)
        await self.gate.wait()


def test_object_snapshots_publish_whole_in_world_and_drop_while_one_is_in_flight(io, loop, monkeypatch):
    topic = sim_topics.object_states
    monkeypatch.setattr(topic, "MessageObjectsItem", lambda **fields: fields, raising=False)
    monkeypatch.setattr(topic, "MessageRobotsItem", lambda **fields: fields, raising=False)
    monkeypatch.setattr(
        topic, "build_message", lambda frame, timestamp, objects, robots: (frame, timestamp, objects, robots), raising=False
    )
    publisher = _GatedPublisher()
    io._object_states_pub = publisher

    robot = RobotRecord.fixed("alpha", "openarm_v2", (0.0, -1.5, 0.0), (0.0, 0.0, 0.0, 1.0))

    def snapshot(timestamp_s, *object_ids, robots=()):
        return ObjectStateSnapshot(timestamp_s=timestamp_s, objects=tuple(
            ObjectRecord(
                object_id=object_id, name=object_id, asset_id="props/blocks/red_block", physics="dynamic", mass=0.1,
                scale=1.0,
                position=(0.5, 0.0, 0.8), orientation=(0.0, 0.0, 0.0, 1.0),
                linear_velocity=(0.0, 0.0, -0.2), angular_velocity=(0.0, 0.0, 0.0),
            )
            for object_id in object_ids
        ), robots=robots)

    first, second, third = snapshot(1.5, "obj_a", "obj_b", robots=(robot,)), snapshot(1.6, "obj_a"), snapshot(1.7)

    async def scenario():
        publisher.gate = asyncio.Event()
        assert io.publish_object_states(first) is True
        await _drain()
        assert io.publish_object_states(second) is False, "the first is still in flight"
        publisher.gate.set()
        await _drain()
        assert io.publish_object_states(third) is True
        await _drain()

    loop.run_until_complete(scenario())
    assert publisher.payloads == [
        ("world", 1.5, [record.fields() for record in first.objects], [robot.fields()]),
        ("world", 1.7, [], []),
    ]
