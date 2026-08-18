"""How messages get from the observed slots into the cache: which parser each
slot's drain uses, and which cache slot a message lands in. The wiring resolves
the real generated modules; tests that need a specific membership monkeypatch
the slot accessors on those modules."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import lerobot_recorder.__main__ as main_mod
from lerobot_recorder import recording
from lerobot_recorder.__main__ import (
    _check_same_links,
    _link_drains,
    _recorded_links,
    _resolve_existing_session,
    _route_link,
    _write_session_json,
)
from lerobot_recorder.plan import LinkKind, SourceEntry
from lerobot_recorder.recording import Cache, GripperSample, JointSample
from tests.test_recording import CORE, NOW_NS, action_entry, joint_entry, make_plan

BACKBONE = (CORE, "backbone_inst", "left_arm_link")
OTHER_ARM = (CORE, "backbone_inst", "right_arm_link")


def make_params(**overrides):
    """The generated Parameters dataclass carries no field defaults, so the
    schema's defaulted fields are spelled out once here (same values the
    manifest defaults to, except the test-sized disk floor)."""
    from peppygen.parameters import Parameters

    values = dict(
        robot_type="bot",
        fps=30,
        storage_root="/tmp/unused",
        s3_uri="",
        image_writer_threads=1,
        streaming_encoding=True,
        max_staleness_s=0.5,
        min_remaining_disk_bytes=1,
    )
    values.update(overrides)
    return Parameters(**values)


def joint_message(position: float):
    return SimpleNamespace(
        timestamp=NOW_NS / 1e9, positions=[position], velocities=[], efforts=[]
    )


def cache_with(*keys) -> Cache:
    return Cache(links=dict.fromkeys(keys), color=[], rgbd_video=[], rgbd_depth=[])


def observed(key):
    core_node, instance_id, link_id = key
    return SimpleNamespace(
        producer=SimpleNamespace(core_node=core_node, instance_id=instance_id),
        source_link_id=link_id,
    )


def test_two_sources_on_one_instance_cache_separately():
    """Every openarm pairing is led by the one backbone instance, so routing
    on the instance alone would collapse both arms into one cache slot and
    record whichever setpoint arrived last for both."""
    cache = cache_with(BACKBONE, OTHER_ARM)
    route = _route_link(cache, LinkKind.JOINT, recording.action_sample)

    route(observed(BACKBONE), joint_message(0.25))
    route(observed(OTHER_ARM), joint_message(-0.75))

    assert cache.links[BACKBONE].positions == (0.25,)
    assert cache.links[OTHER_ARM].positions == (-0.75,)


def test_a_member_that_joined_after_discovery_is_dropped():
    """The recorded shape is the plan captured at startup, so a member the
    daemon adds later has no column to write into."""
    cache = cache_with(BACKBONE)
    route = _route_link(cache, LinkKind.JOINT, recording.action_sample)

    route(observed(OTHER_ARM), joint_message(1.0))

    assert cache.links == {BACKBONE: None}


def test_an_unusable_message_leaves_the_previous_sample():
    cache = cache_with(BACKBONE)
    route = _route_link(cache, LinkKind.JOINT, recording.action_sample)
    route(observed(BACKBONE), joint_message(0.5))

    route(observed(BACKBONE), SimpleNamespace(timestamp=0.0, positions=[9.0], velocities=[], efforts=[]))

    assert cache.links[BACKBONE].positions == (0.5,)


def test_each_slot_drains_through_the_parser_for_its_role():
    """A measured stream parsed as a setpoint (or the reverse) would read
    fields the other message does not carry, so every frame would be lost to a
    per-message route error with the drains still looking healthy. The routes
    exercised here are the ones the wiring built, not ones this test paired
    up itself."""
    cache = cache_with(BACKBONE, OTHER_ARM)
    drained = []

    def fake_drain(topic_module, route, _token, label):
        drained.append((label, topic_module.TOPIC_NAME, route))

    original, main_mod._drain = main_mod._drain, fake_drain
    try:
        _link_drains(token=None, cache=cache)
    finally:
        main_mod._drain = original

    assert [(label, topic) for label, topic, _ in drained] == [
        ("observed_joints", "joint_states"),
        ("commanded_joints", "joint_setpoints"),
        ("observed_grippers", "gripper_states"),
        ("commanded_grippers", "gripper_setpoints"),
    ]

    # A gripper's measured stream carries an effort and its setpoint stream
    # does not, so the parser each slot was wired with is visible in what the
    # sample records.
    routes = {label: route for label, _, route in drained}
    message = SimpleNamespace(timestamp=NOW_NS / 1e9, opening=0.4, effort=0.9)

    routes["observed_grippers"](observed(BACKBONE), message)
    assert cache.links[BACKBONE] == GripperSample(opening=0.4, effort=0.9, timestamp_ns=NOW_NS)

    routes["commanded_grippers"](observed(OTHER_ARM), message)
    assert cache.links[OTHER_ARM] == GripperSample(opening=0.4, effort=0.0, timestamp_ns=NOW_NS)


class NeverCancelledToken:
    node_runner = None

    def is_cancelled(self) -> bool:
        return False

    async def cancelled(self) -> None:
        await asyncio.Event().wait()


def test_a_closed_subscription_ends_the_drain():
    """next() returning None means the node is shutting down; anything but
    returning would spin on a subscription that will never produce again."""
    cache = cache_with(BACKBONE)
    replies = [(observed(BACKBONE), joint_message(0.5)), None]

    class Topic:
        LINK_ID = "observed_joints"

        @staticmethod
        async def subscribe(_node_runner):
            return SimpleNamespace(next=lambda: _pop(replies))

    async def _pop(remaining):
        return remaining.pop(0)

    route = _route_link(cache, LinkKind.JOINT, recording.state_sample)

    async def run():
        await asyncio.wait_for(
            main_mod._drain(Topic, route, NeverCancelledToken(), "observed_joints"), timeout=5.0
        )

    asyncio.run(run())
    assert cache.links[BACKBONE].positions == (0.5,)


def test_a_service_handler_error_does_not_end_the_loop(monkeypatch):
    """One raising handler must not leave the service unanswered for the
    node's whole life."""
    calls = {"n": 0}

    class Svc:
        @staticmethod
        async def handle_next_request(_node_runner, _handler):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("handler blew up")

    node_runner = SimpleNamespace(
        cancellation_token=lambda: SimpleNamespace(is_cancelled=lambda: calls["n"] >= 3)
    )
    monkeypatch.setattr(main_mod, "SERVICE_RETRY_S", 0)
    asyncio.run(main_mod._serve(node_runner, Svc, handler=None, label="svc"))
    assert calls["n"] == 3


def test_cache_covers_state_and_action_sources_alike():
    """A cache slot per planned source, none pre-filled; the action half
    matters because a state-only cache would drop every setpoint."""
    plan = make_plan(state=(joint_entry(),), action=(action_entry(),))
    cache = Cache.for_plan(plan)
    assert set(cache.links) == {joint_entry().key, action_entry().key}
    assert all(sample is None for sample in cache.links.values())


def test_joint_setpoints_and_states_share_the_wire_shape():
    """Both roles of a joint pairing carry the same vectors, so one parser
    reading the other's message must still produce the same sample."""
    message = joint_message(0.125)
    assert recording.state_sample(LinkKind.JOINT, message) == JointSample(
        positions=(0.125,), velocities=(), efforts=(), timestamp_ns=NOW_NS
    )
    assert recording.action_sample(LinkKind.JOINT, message) == recording.state_sample(
        LinkKind.JOINT, message
    )


def session_with(tmp_path, links: dict):
    session_dir = tmp_path / "2026-08-06_10-00-00"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(json.dumps({"robot_type": "bot", **links}))
    return session_dir


def leader_entry(link_id: str, feature_key: str) -> SourceEntry:
    return SourceEntry(key=(CORE, "backbone", link_id), kind=LinkKind.JOINT, feature_key=feature_key)


def two_limb_plan(left_feeds: str, right_feeds: str):
    """Two limbs led from one instance, differing only by link, the backbone
    shape a leader swap permutes."""
    return make_plan(
        state=(joint_entry(0), joint_entry(1)),
        action=(leader_entry("left_link", left_feeds), leader_entry("right_link", right_feeds)),
    )


def test_resuming_with_the_leaders_swapped_is_refused(tmp_path):
    """A swap keeps the recorded key set and every dataset feature identical
    and only permutes which source feeds which action column, so the check
    must compare the mapping, not the sources."""
    session_dir = session_with(tmp_path, _recorded_links(two_limb_plan("arm0", "arm1")))

    with pytest.raises(ValueError, match="would cross the limbs"):
        _check_same_links(session_dir, two_limb_plan("arm1", "arm0"))


def test_sources_differing_only_by_core_node_stay_distinct():
    """Provenance keyed on instance/link alone would collapse these two limbs
    into one entry and hide a swap between them."""
    plan = make_plan(
        state=(joint_entry(0), joint_entry(1)),
        action=tuple(
            SourceEntry(key=(core, "arm", "link"), kind=LinkKind.JOINT, feature_key=feature)
            for core, feature in (("cnA", "arm0"), ("cnB", "arm1"))
        ),
    )
    assert _recorded_links(plan)["action_links"] == {
        "cnA/arm/link": "arm0",
        "cnB/arm/link": "arm1",
    }


def test_resuming_the_session_this_launch_wrote_is_allowed(tmp_path):
    """Round trip through the writer, so a session.json written without its
    links cannot silently disable the check for every later resume."""
    plan = two_limb_plan("arm0", "arm1")
    session_dir = tmp_path / "2026-08-06_10-00-00"
    session_dir.mkdir()
    _write_session_json(session_dir, make_params(), plan)
    _check_same_links(session_dir, plan)


def test_a_session_without_a_session_json_is_refused(tmp_path):
    """Every session this node opens records its links, so a directory
    without them is not a session this node wrote."""
    bare = tmp_path / "2026-08-06_10-00-00"
    bare.mkdir()
    with pytest.raises(ValueError, match="session.json unreadable"):
        _check_same_links(bare, make_plan())


def test_a_session_json_without_links_is_refused(tmp_path):
    with pytest.raises(ValueError, match="records no state_links"):
        _check_same_links(session_with(tmp_path, {}), make_plan())


def test_setup_discovers_the_seeded_membership_inline(tmp_path, monkeypatch):
    """The boot config seeds the observer slots before the process starts, so
    setup reads the robot's shape directly and the session exists the moment
    setup returns; nothing is deferred behind Running anymore."""
    import peppygen.clock
    from peppygen.consumed_topics.color_cameras import (
        video_stream as color_cameras_video_stream,
    )
    from peppygen.consumed_topics.rgbd_cameras import (
        video_stream as rgbd_cameras_video_stream,
    )
    from peppygen.paired_topics.commanded_grippers import gripper_setpoints
    from peppygen.paired_topics.commanded_joints import joint_setpoints
    from peppygen.paired_topics.observed_grippers import gripper_states
    from peppygen.paired_topics.observed_joints import joint_states

    monkeypatch.setattr(
        joint_states, "sources", lambda _r: [observed((CORE, "arm_inst", "link"))]
    )
    monkeypatch.setattr(
        joint_setpoints, "sources", lambda _r: [observed((CORE, "lead_inst", "arm_link"))]
    )
    # The real modules read the remaining slots off the runner this test has
    # none of; this launch binds no grippers and no cameras.
    for module in (gripper_states, gripper_setpoints):
        monkeypatch.setattr(module, "sources", lambda _r: [])
    for module in (color_cameras_video_stream, rgbd_cameras_video_stream):
        monkeypatch.setattr(module, "bound_producers", lambda _r: [])

    async def clock_ready(_runner):
        return None

    monkeypatch.setattr(peppygen.clock, "init", clock_ready)
    runner = SimpleNamespace(
        cancellation_token=lambda: NeverCancelledToken(), on_shutdown=lambda _hook: None
    )
    params = make_params(storage_root=str(tmp_path))

    def session_created():
        return any((child / "session.json").is_file() for child in tmp_path.iterdir())

    async def run():
        tasks = await asyncio.wait_for(main_mod.setup(params, runner), timeout=5.0)
        assert session_created(), "the session opens during setup, from the seeded shape"
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_setup_refuses_unusable_parameters(tmp_path):
    runner = SimpleNamespace(cancellation_token=lambda: NeverCancelledToken())
    for bad in (
        make_params(storage_root=str(tmp_path), fps=0),
        make_params(storage_root=str(tmp_path), image_writer_threads=0),
    ):
        with pytest.raises(ValueError, match="must be positive"):
            asyncio.run(main_mod.setup(bad, runner))


def test_resolving_a_session_validates_the_operator_supplied_name(tmp_path):
    """The resume request's name is the node's only untrusted input; resolving
    it without validation would join a traversal shape onto the storage root."""
    from lerobot_recorder.storage import StorageTarget

    target = StorageTarget(root=tmp_path, s3=None)
    with pytest.raises(ValueError, match="is not a session name"):
        _resolve_existing_session(target, "../evil", make_plan())

    plan = make_plan()
    session_dir = session_with(tmp_path, _recorded_links(plan))
    assert _resolve_existing_session(target, session_dir.name, plan) == session_dir
