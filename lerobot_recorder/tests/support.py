"""Builders the suites share: source keys and entries on one core node, a
recording plan over them, and the node's launch parameters."""

from lerobot_recorder.plan import (
    CameraEntry,
    LinkKind,
    PairingEnd,
    RecordingPlan,
    SourceEntry,
    SourceKey,
)

NOW_NS = 1_000_000_000_000
STALENESS_S = 0.25


CORE = "cn"


def key(instance_id: str, link_id: str = "link", peer: PairingEnd | None = None) -> SourceKey:
    """A source on `CORE`, named directly unless a peer names its pair."""
    return SourceKey(source=PairingEnd(CORE, instance_id, link_id), peer=peer)


def joint_entry(i=0) -> SourceEntry:
    return SourceEntry(key=key(f"arm{i}"), kind=LinkKind.JOINT, feature_key=f"arm{i}")


def gripper_entry(i=0) -> SourceEntry:
    return SourceEntry(key=key(f"grip{i}"), kind=LinkKind.GRIPPER, feature_key=f"grip{i}")


# The commanded source of a limb sits on its own instance but records under
# the limb's name, exactly as discovery pairs them.
def action_entry(i=0) -> SourceEntry:
    return SourceEntry(key=key(f"commanded_arm{i}"), kind=LinkKind.JOINT, feature_key=f"arm{i}")


ARM0 = joint_entry().key


def make_plan(state=None, action=(), color=0, rgbd=0) -> RecordingPlan:
    state = (joint_entry(),) if state is None else state
    color_entries = tuple(
        CameraEntry(key=("core", f"cam{i}"), name=f"cam{i}") for i in range(color)
    )
    rgbd_entries = tuple(
        CameraEntry(key=("core", f"rgbd{i}"), name=f"rgbd{i}") for i in range(rgbd)
    )
    # Discovery pairs a commanded source with the measured source of its own
    # limb; here the shared feature key stands in for that binding order.
    measured = {(s.feature_key, s.kind): s.key for s in state}
    return RecordingPlan(
        state=tuple(state),
        action=tuple(action),
        color_cameras=color_entries,
        rgbd_cameras=rgbd_entries,
        color_index={e.key: i for i, e in enumerate(color_entries)},
        rgbd_index={e.key: i for i, e in enumerate(rgbd_entries)},
        action_fallback={
            a.key: measured[(a.feature_key, a.kind)]
            for a in action
            if (a.feature_key, a.kind) in measured
        },
    )


def make_params(**overrides):
    """The generated Parameters dataclass carries no field defaults, so the
    schema's defaulted fields are spelled out once here (same values the
    manifest defaults to, except the test-sized disk floor). Suites with
    different pacing override the fields they care about."""
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
