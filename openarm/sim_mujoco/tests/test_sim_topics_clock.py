"""Tests for the sim-time half of SimTopicIO: recording the engine clock,
stamping from it, and the guarded fan-out chain. peppylib and peppygen exist
only inside the node's image, so minimal fakes are installed before the
import; everything under test is pure python.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
sys.path.insert(0, str(_ROBOT_DIR))

_PAIRED_SLOTS = [
    "chest",
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
    "wrist_left",
    "wrist_right",
]
_PAIRED_TOPICS = [
    "depth_stream",
    "stream_info",
    "video_stream",
    "joint_setpoints",
    "joint_states",
    "gripper_setpoints",
    "gripper_states",
]


def _install_runtime_fakes() -> None:
    """The smallest module tree that satisfies sim_topics' imports. The clock
    tests construct their own publisher and never call these."""
    peppylib = types.ModuleType("peppylib")
    peppylib.NodeRunner = object
    peppylib.TopicPublisher = object
    peppylib_clock = types.ModuleType("peppylib.clock")
    peppylib_clock.SimTimePublisher = type("SimTimePublisher", (), {})
    peppylib.clock = peppylib_clock

    peppygen = types.ModuleType("peppygen")
    peppygen_clock = types.ModuleType("peppygen.clock")
    peppygen_clock.now_ns = lambda: 0
    peppygen.clock = peppygen_clock
    paired = types.ModuleType("peppygen.paired_topics")
    modules = {
        "peppylib": peppylib,
        "peppylib.clock": peppylib_clock,
        "peppygen": peppygen,
        "peppygen.clock": peppygen_clock,
        "peppygen.paired_topics": paired,
    }
    for slot in _PAIRED_SLOTS:
        slot_module = types.ModuleType(f"peppygen.paired_topics.{slot}")
        for topic in _PAIRED_TOPICS:
            topic_module = types.ModuleType(f"peppygen.paired_topics.{slot}.{topic}")
            topic_module.LINK_ID = slot
            setattr(slot_module, topic, topic_module)
        setattr(paired, slot, slot_module)
        modules[f"peppygen.paired_topics.{slot}"] = slot_module
    sys.modules.update(modules)


_install_runtime_fakes()

import sim_topics  # noqa: E402  (needs the fakes above)


class _FakeFanOut:
    """Stands in for peppylib's SimTimePublisher: records every publish, can
    hold publishes open behind a gate, and can fail specific instants."""

    def __init__(self) -> None:
        self.published = []
        self.gate = None
        self.failing = set()

    @property
    def participants(self):
        return ["cn-a", "cn-b"]

    async def publish(self, time_ns: int) -> None:
        self.published.append(time_ns)
        if self.gate is not None:
            await self.gate.wait()
        if time_ns in self.failing:
            raise RuntimeError("sim time did not reach every machine; unreached: cn-b")


@pytest.fixture(name="loop")
def loop_fixture():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(name="io")
def io_fixture(loop):
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop)
    io._sim_clock = _FakeFanOut()
    return io


async def _drain(loop_turns: int = 10) -> None:
    """Let scheduled callbacks and done-callbacks run: pure scheduling, no
    wall-clock dependence."""
    for _ in range(loop_turns):
        await asyncio.sleep(0)


def test_a_follower_stamps_from_the_daemon_clock(loop, monkeypatch):
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop)
    monkeypatch.setattr(sim_topics.clock, "now_ns", lambda: 5_000_000_000)
    assert io._timestamp_s() == 5.0
    # And the source paths are inert: nothing declared, nothing recorded.
    io.record_engine_time(1.0)
    io.publish_sim_time()
    assert io._timestamp_s() == 5.0


def test_a_source_stamps_from_its_recorded_engine_clock(io):
    io.record_engine_time(1.25)
    assert io._timestamp_s() == 1.25
    io.record_engine_time(1.5)
    assert io._timestamp_s() == 1.5


def test_a_source_must_record_before_stamping_or_publishing(io):
    with pytest.raises(RuntimeError, match="record an engine step"):
        io._timestamp_s()
    with pytest.raises(RuntimeError, match="record an engine step"):
        io.publish_sim_time()


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
    assert io._timestamp_s() > 0.0
    io.publish_sim_time()
    loop.run_until_complete(_drain())
    assert io._sim_clock.published == [1]

def test_publish_lands_the_recorded_instant_in_nanoseconds(io, loop):
    io.record_engine_time(2.5)
    io.publish_sim_time()
    loop.run_until_complete(_drain())
    assert io._sim_clock.published == [2_500_000_000]


def test_a_tick_arriving_mid_flight_is_latched_never_reordered(io, loop):
    fan_out = io._sim_clock

    async def scenario():
        fan_out.gate = asyncio.Event()
        io._publish_sim_time_on_loop(100)
        await _drain()
        # Two more while the first is out: only the newest survives.
        io._publish_sim_time_on_loop(200)
        io._publish_sim_time_on_loop(300)
        assert fan_out.published == [100]
        fan_out.gate.set()
        await _drain()
        assert fan_out.published == [100, 300]

    loop.run_until_complete(scenario())


def test_fan_out_failures_are_latched_to_one_line_each_way(io, loop, caplog):
    fan_out = io._sim_clock
    fan_out.failing = {1_000_000_000, 2_000_000_000}

    async def scenario():
        for instant in [1_000_000_000, 2_000_000_000, 3_000_000_000]:
            io._publish_sim_time_on_loop(instant)
            await _drain()

    with caplog.at_level("INFO"):
        loop.run_until_complete(scenario())
    assert fan_out.published == [1_000_000_000, 2_000_000_000, 3_000_000_000]
    down = [r for r in caplog.records if "not reaching the whole fleet" in r.message]
    recovered = [r for r in caplog.records if "reaching the whole fleet again" in r.message]
    assert len(down) == 1, "two consecutive failures log one line"
    assert len(recovered) == 1, "recovery logs one line"
