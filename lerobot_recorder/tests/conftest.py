"""episode.py imports the generated peppy bindings at module import; unit
tests run without a peppy runtime, so a minimal stand-in mirroring the
generated API is always installed, shadowing any installed peppygen (the
real package needs a daemon and must not leak into unit tests)."""

import os
import sys
import tempfile
import time
import types
from dataclasses import dataclass

# pyarrow's bundled jemalloc corrupts parquet reads after in-process av/torch
# use (the integration tests' dataset reload); the system allocator is stable.
# Must be set before pyarrow first loads, hence here, and unconditionally so
# ambient shell state cannot reintroduce the corruption.
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
# The dataset reloads in the integration tests cache under HF_HOME; pointing
# it at a per-run temp dir keeps the suite from growing ~/.cache/huggingface.
# Must be set before huggingface_hub first loads, hence here.
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="lerobot_recorder_tests_hf_")


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_fake_peppygen() -> None:
    peppygen = _module("peppygen")
    clock = _module("peppygen.clock")
    clock.now_ns = time.time_ns
    peppygen.clock = clock

    consumed_services = _module("peppygen.consumed_services")
    services_rgbd = _module("peppygen.consumed_services.rgbd_cameras")
    depth_info = _module("peppygen.consumed_services.rgbd_cameras.depth_stream_info")

    async def _no_runtime(*_args, **_kwargs):
        raise RuntimeError("no peppy runtime under test")

    depth_info.poll = _no_runtime
    services_rgbd.depth_stream_info = depth_info
    consumed_services.rgbd_cameras = services_rgbd

    consumed_topics = _module("peppygen.consumed_topics")
    topics_rgbd = _module("peppygen.consumed_topics.rgbd_cameras")
    video_stream = _module("peppygen.consumed_topics.rgbd_cameras.video_stream")
    video_stream.bound_producers = lambda _runner: []
    topics_rgbd.video_stream = video_stream
    consumed_topics.rgbd_cameras = topics_rgbd

    topics_color = _module("peppygen.consumed_topics.color_cameras")
    color_video_stream = _module("peppygen.consumed_topics.color_cameras.video_stream")
    color_video_stream.bound_producers = lambda _runner: []
    topics_color.video_stream = color_video_stream
    consumed_topics.color_cameras = topics_color

    exposed_actions = _module("peppygen.exposed_actions")
    record_episode = _module("peppygen.exposed_actions.record_episode")

    @dataclass
    class GoalDecision:
        accepted: bool
        reason: str | None

        @staticmethod
        def accept():
            return GoalDecision(True, None)

        @staticmethod
        def reject(reason=None):
            return GoalDecision(False, reason)

    record_episode.GoalDecision = GoalDecision
    exposed_actions.record_episode = record_episode

    exposed_services = _module("peppygen.exposed_services")
    finish_session = _module("peppygen.exposed_services.finish_session")

    @dataclass
    class FinishResponse:
        session: str
        error: str | None

    finish_session.Response = FinishResponse
    finish_session.handle_next_request = _no_runtime
    exposed_services.finish_session = finish_session

    resume_session = _module("peppygen.exposed_services.resume_session")

    @dataclass
    class ResumeResponse:
        session: str
        episodes: int
        error: str | None

    resume_session.Response = ResumeResponse
    resume_session.handle_next_request = _no_runtime
    exposed_services.resume_session = resume_session

    # The observed pairing slots, under the module paths and attribute names
    # the generator emits, so the wiring under test resolves the same modules
    # it would at runtime. Each slot's sources are settable per test.
    paired_topics = _module("peppygen.paired_topics")
    for link_id, topic in (
        ("observed_joints", "joint_states"),
        ("commanded_joints", "joint_setpoints"),
        ("observed_grippers", "gripper_states"),
        ("commanded_grippers", "gripper_setpoints"),
    ):
        slot = _module(f"peppygen.paired_topics.{link_id}")
        module = _module(f"peppygen.paired_topics.{link_id}.{topic}")
        module.LINK_ID = link_id
        module.TOPIC_NAME = topic
        module.sources = lambda _runner: []
        module.subscribe = _no_runtime
        setattr(slot, topic, module)
        setattr(paired_topics, link_id, slot)

    depth_stream = _module("peppygen.consumed_topics.rgbd_cameras.depth_stream")
    depth_stream.bound_producers = lambda _runner: []
    depth_stream.subscribe = _no_runtime
    topics_rgbd.depth_stream = depth_stream
    video_stream.subscribe = _no_runtime
    color_video_stream.subscribe = _no_runtime

    clock.init = _no_runtime

    parameters = _module("peppygen.parameters")

    @dataclass
    class Parameters:
        robot_type: str = "bot"
        fps: int = 30
        storage_root: str = "/tmp/unused"
        s3_uri: str = ""
        image_writer_threads: int = 1
        max_staleness_s: float = 0.5
        min_remaining_disk_bytes: int = 1

    parameters.Parameters = Parameters
    peppygen.parameters = parameters

    class NodeBuilder:
        def run(self, _setup):
            raise RuntimeError("no peppy runtime under test")

    class NodeRunner:
        pass

    peppygen.NodeBuilder = NodeBuilder
    peppygen.NodeRunner = NodeRunner


_install_fake_peppygen()
