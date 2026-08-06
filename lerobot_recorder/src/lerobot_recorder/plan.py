"""What this launch records: the limbs and cameras discovered from the bound
sources at startup.

The observer slots are `zero_or_more`, so the robot's shape comes from the
launcher rather than this node: each pairing kind contributes as many limbs as
the launcher bound. A limb is one measured source paired with the commanded
source at the same position in its slot, which is what makes the launcher's
binding order load-bearing.

Pairing wire messages carry no joint names, so dataset dimension names derive
from each limb's name plus joint index (left_arm_inst_j0,
left_grip_inst_opening); a limb is named after the follower instance observed
for it. Joint counts and which optional vectors a source delivers are
discovered from its first message and locked into the schema at the first
episode. Each camera producer becomes one dataset image key derived from its
instance id, in binding order. The plan's `action` entries name the sources
recorded into LeRobot's action feature (peppy actions are unrelated).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum

ProducerKey = tuple[str, str]
# An observed source's identity: its instance's wire address plus the
# producer-side link of the observed pairing. The link is what keeps two
# sources apart when one instance leads several pairings, which is exactly
# the backbone's shape.
SourceKey = tuple[str, str, str]

# How long the observer slots' membership must hold still before it counts as
# fully delivered: delivery is per slot, so a set read between two slots'
# deliveries is a partial robot.
SOURCE_SETTLE_S = 0.5
# How long to wait for any membership at all. A launch that bound nothing is
# indistinguishable from one whose delivery has not landed, so discovery gets
# to report the empty set rather than waiting forever.
SOURCE_DELIVERY_TIMEOUT_S = 15.0
# How often the membership is re-read while waiting for it to settle.
SOURCE_POLL_S = 0.05


def producer_key(producer) -> ProducerKey:
    return (producer.core_node, producer.instance_id)


def source_key(source) -> SourceKey:
    return (source.producer.core_node, source.producer.instance_id, source.source_link_id)


class LinkKind(Enum):
    JOINT = "joint"
    GRIPPER = "gripper"


@dataclass(frozen=True)
class SourceEntry:
    """One observed source's contribution to a recorded vector."""

    key: SourceKey
    kind: LinkKind
    feature_key: str

    @property
    def label(self) -> str:
        """How this source is named in operator-facing messages."""
        _core_node, instance_id, link_id = self.key
        return f"{instance_id}/{link_id}"


def sanitize_key(instance_id: str) -> str:
    """Dataset feature keys allow [a-z0-9_] only."""
    key = re.sub(r"[^a-z0-9_]", "_", instance_id.lower())
    return key or "source"


def assign_camera_names(
    used_names: set[str], instance_ids: list[str], *, with_depth: bool
) -> list[str]:
    """Unique dataset image keys for one camera slot, in binding order. An
    rgbd camera also reserves its derived '<name>_depth' key, so a camera
    named x_depth can never collide with rgbd camera x."""
    names = []
    for instance_id in instance_ids:
        name = sanitize_key(instance_id)
        while name in used_names or (with_depth and f"{name}_depth" in used_names):
            name = f"{name}_cam"
        used_names.add(name)
        if with_depth:
            used_names.add(f"{name}_depth")
        names.append(name)
    return names


@dataclass(frozen=True)
class CameraEntry:
    key: ProducerKey
    name: str


@dataclass(frozen=True)
class RecordingPlan:
    state: tuple[SourceEntry, ...]
    action: tuple[SourceEntry, ...]
    color_cameras: tuple[CameraEntry, ...]
    rgbd_cameras: tuple[CameraEntry, ...]
    color_index: dict[ProducerKey, int]
    rgbd_index: dict[ProducerKey, int]
    # Commanded source -> the measured source of the same limb. Until a leader
    # has commanded anything the action falls back to it, so the pairing the
    # launcher's binding order established has to outlive discovery.
    action_fallback: dict[SourceKey, SourceKey]


def limb_slots():
    """Per pairing kind, the generated (measured, commanded) topic modules.
    Imported on call so the pure planning logic stays importable without the
    peppy runtime (unit tests)."""
    from peppygen.paired_topics.commanded_grippers import (
        gripper_setpoints as commanded_grippers,
    )
    from peppygen.paired_topics.commanded_joints import (
        joint_setpoints as commanded_joints,
    )
    from peppygen.paired_topics.observed_grippers import (
        gripper_states as observed_grippers,
    )
    from peppygen.paired_topics.observed_joints import joint_states as observed_joints

    return (
        (LinkKind.JOINT, observed_joints, commanded_joints),
        (LinkKind.GRIPPER, observed_grippers, commanded_grippers),
    )


# Per limb_slots entry, the (measured, commanded) member lists read together.
LimbSources = tuple[tuple[tuple, tuple], ...]


def read_limb_sources(node_runner) -> LimbSources:
    """One read of every observer slot's member list, in limb_slots order.
    The live sets are the daemon's to replace whole at any moment, so
    everything downstream works from a snapshot instead of re-reading."""
    return tuple(
        (tuple(observed.sources(node_runner)), tuple(commanded.sources(node_runner)))
        for _kind, observed, commanded in limb_slots()
    )


async def wait_for_sources(
    node_runner, *, settle_s: float = SOURCE_SETTLE_S, timeout_s: float = SOURCE_DELIVERY_TIMEOUT_S
) -> LimbSources:
    """Wait for the daemon to finish delivering the observer slots' member
    sets, and return the snapshot discovery should consume: a membership
    change between settling and discovery could otherwise pair sources from
    two different plan revisions.

    Delivery is asynchronous and per slot (each carries that slot's complete
    membership), and the framework serves it independently of node setup, so
    reading the sets the instant setup starts sees an empty robot. An empty
    set is also a legal steady state, and nothing distinguishes "not delivered
    yet" from "bound to nothing", so this settles on membership holding still
    rather than on any particular count, and gives up at `timeout_s` to let
    discovery report what was actually bound.

    Waiting here happens before the node has a shutdown hook, so a stop while
    it runs ends the start (checked before every exit, timeout included)
    rather than letting it build a session inside the teardown window."""
    token = node_runner.cancellation_token()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    previous = None
    settled_at = None
    while True:
        if token.is_cancelled():
            raise asyncio.CancelledError
        snapshot = read_limb_sources(node_runner)
        keys = tuple(
            tuple(tuple(source_key(s) for s in sources) for sources in pair) for pair in snapshot
        )
        if keys != previous:
            previous, settled_at = keys, loop.time()
        elif any(sources for pair in keys for sources in pair) and (
            loop.time() - settled_at >= settle_s
        ):
            total = sum(len(sources) for pair in keys for sources in pair)
            print(f"[recorder] observing {total} source(s)", flush=True)
            return snapshot
        if loop.time() >= deadline:
            print(
                f"[recorder] gave up waiting for observed sources after {timeout_s:g}s",
                flush=True,
            )
            return snapshot
        await asyncio.sleep(SOURCE_POLL_S)


def limb_name(used_names: set[str], source) -> str:
    """Dataset name for one limb, from the follower instance observed for it.
    An instance that follows several pairings of one kind takes the observed
    link into the name, so the two limbs stay distinct."""
    _core_node, instance_id, link_id = source_key(source)
    name = sanitize_key(instance_id)
    if name in used_names:
        name = sanitize_key(f"{instance_id}_{link_id}")
    if name in used_names:
        raise ValueError(f"two limbs both name themselves {name!r}")
    used_names.add(name)
    return name


def discover(node_runner, limb_sources: LimbSources) -> RecordingPlan:
    from peppygen.consumed_topics.color_cameras import (
        video_stream as color_cameras_video_stream,
    )
    from peppygen.consumed_topics.rgbd_cameras import (
        video_stream as rgbd_cameras_video_stream,
    )

    color_producers = color_cameras_video_stream.bound_producers(node_runner)
    rgbd_producers = rgbd_cameras_video_stream.bound_producers(node_runner)

    limb_names: set[str] = set()
    state: list[SourceEntry] = []
    action: list[SourceEntry] = []
    action_fallback: dict[SourceKey, SourceKey] = {}
    for (kind, _observed, _commanded), (measured_sources, commanded_sources) in zip(
        limb_slots(), limb_sources, strict=True
    ):
        if len(measured_sources) != len(commanded_sources):
            raise ValueError(
                f"{kind.value} limbs are bound {len(measured_sources)} measured to "
                f"{len(commanded_sources)} commanded; each measured source needs the "
                f"commanded source of its own pairing, bound in the same order"
            )
        for measured, command in zip(measured_sources, commanded_sources, strict=True):
            name = limb_name(limb_names, measured)
            measured_entry = SourceEntry(key=source_key(measured), kind=kind, feature_key=name)
            command_entry = SourceEntry(key=source_key(command), kind=kind, feature_key=name)
            state.append(measured_entry)
            action.append(command_entry)
            action_fallback[command_entry.key] = measured_entry.key
    recorded = [*state, *action]
    if len({entry.key for entry in recorded}) != len(recorded):
        raise ValueError(
            "one source is bound to two limbs; each limb needs its own pairing, "
            "or both would share a cache slot and record the same values twice"
        )
    if not state:
        raise ValueError(
            "no limbs are bound; a dataset with no observation.state and no action "
            "records nothing a policy can be trained on"
        )

    used_names: set[str] = set()

    def camera_entries(producers, *, with_depth: bool) -> tuple[CameraEntry, ...]:
        ids = [p.instance_id for p in producers]
        names = assign_camera_names(used_names, ids, with_depth=with_depth)
        return tuple(
            CameraEntry(key=producer_key(p), name=name)
            for p, name in zip(producers, names, strict=True)
        )

    color = camera_entries(color_producers, with_depth=False)
    rgbd = camera_entries(rgbd_producers, with_depth=True)

    return RecordingPlan(
        state=tuple(state),
        action=tuple(action),
        color_cameras=color,
        rgbd_cameras=rgbd,
        color_index={e.key: i for i, e in enumerate(color)},
        rgbd_index={e.key: i for i, e in enumerate(rgbd)},
        action_fallback=action_fallback,
    )
