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

import re
from dataclasses import dataclass
from enum import Enum

ProducerKey = tuple[str, str]
# An observed source's identity: its instance's wire address plus the
# producer-side link of the observed pairing. The link is what keeps two
# sources apart when one instance leads several pairings, which is exactly
# the backbone's shape.
SourceKey = tuple[str, str, str]


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


@dataclass(frozen=True)
class BoundSources:
    """One read of everything the launcher bound, the only input `discover`
    consumes. The boot config seeds every slot with the launch's membership,
    so this answers the robot's shape from setup onward; the live sets are
    materialized into tuples here (the daemon replaces them whole when the
    plan changes), so everything downstream works from this one snapshot
    instead of re-reading."""

    # Per pairing kind, the (kind, measured, commanded) member lists read
    # together.
    limbs: tuple[tuple[LinkKind, tuple, tuple], ...]
    color_producers: tuple
    rgbd_producers: tuple


def snapshot_sources(node_runner, slots, color_module, rgbd_module) -> BoundSources:
    """One read of every slot's member list into an immutable snapshot; the
    slot and camera modules arrive as arguments so the read is testable
    against scripted slots."""
    return BoundSources(
        limbs=tuple(
            (kind, tuple(observed.sources(node_runner)), tuple(commanded.sources(node_runner)))
            for kind, observed, commanded in slots
        ),
        color_producers=tuple(color_module.bound_producers(node_runner)),
        rgbd_producers=tuple(rgbd_module.bound_producers(node_runner)),
    )


def read_bound_sources(node_runner) -> BoundSources:
    """`snapshot_sources` over the real generated modules."""
    from peppygen.consumed_topics.color_cameras import (
        video_stream as color_cameras_video_stream,
    )
    from peppygen.consumed_topics.rgbd_cameras import (
        video_stream as rgbd_cameras_video_stream,
    )

    return snapshot_sources(
        node_runner, limb_slots(), color_cameras_video_stream, rgbd_cameras_video_stream
    )


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


def discover(bound: BoundSources) -> RecordingPlan:
    limb_names: set[str] = set()
    state: list[SourceEntry] = []
    action: list[SourceEntry] = []
    action_fallback: dict[SourceKey, SourceKey] = {}
    for kind, measured_sources, commanded_sources in bound.limbs:
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
            "records nothing a policy can be trained on. If the launcher does bind "
            "pairings, the daemon predates observation seeding (peppy too old)"
        )

    used_names: set[str] = set()

    def camera_entries(producers, *, with_depth: bool) -> tuple[CameraEntry, ...]:
        ids = [p.instance_id for p in producers]
        names = assign_camera_names(used_names, ids, with_depth=with_depth)
        return tuple(
            CameraEntry(key=producer_key(p), name=name)
            for p, name in zip(producers, names, strict=True)
        )

    color = camera_entries(bound.color_producers, with_depth=False)
    rgbd = camera_entries(bound.rgbd_producers, with_depth=True)

    return RecordingPlan(
        state=tuple(state),
        action=tuple(action),
        color_cameras=color,
        rgbd_cameras=rgbd,
        color_index={e.key: i for i, e in enumerate(color)},
        rgbd_index={e.key: i for i, e in enumerate(rgbd)},
        action_fallback=action_fallback,
    )
