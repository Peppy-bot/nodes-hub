import struct
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_recorder.plan import CameraEntry, LinkKind, RecordingPlan, SourceEntry
from lerobot_recorder.recording import (
    GRIPPER_LAYOUT,
    MAX_EPISODE_S,
    Cache,
    CameraFrame,
    GripperSample,
    JointSample,
    LinkLayout,
    NotReady,
    SampleGap,
    SourceSchema,
    action_sample,
    capture_schema,
    decode_color,
    decode_depth,
    efforts_dim_names,
    sample,
    schema_mismatch,
    stamp_to_ns,
    state_dim_names,
    state_sample,
    validate_depth_units,
    velocities_dim_names,
)

NOW_NS = 1_000_000_000_000
STALENESS_S = 0.25


CORE = "cn"


def joint_entry(i=0) -> SourceEntry:
    return SourceEntry(key=(CORE, f"arm{i}", "link"), kind=LinkKind.JOINT, feature_key=f"arm{i}")


def gripper_entry(i=0) -> SourceEntry:
    return SourceEntry(
        key=(CORE, f"grip{i}", "link"), kind=LinkKind.GRIPPER, feature_key=f"grip{i}"
    )


# The commanded source of a limb sits on its own instance but records under
# the limb's name, exactly as discovery pairs them.
def action_entry(i=0) -> SourceEntry:
    return SourceEntry(
        key=(CORE, f"commanded_arm{i}", "link"), kind=LinkKind.JOINT, feature_key=f"arm{i}"
    )


ARM0 = joint_entry().key
GRIP0 = gripper_entry().key
COMMANDED_ARM0 = action_entry().key


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


def joints(n, stamp_ns=NOW_NS, velocities=True, efforts=False) -> JointSample:
    return JointSample(
        positions=(0.1,) * n,
        velocities=(0.0,) * n if velocities else (),
        efforts=(0.2,) * n if efforts else (),
        stamp_ns=stamp_ns,
    )


def grip(opening=0.5, effort=0.1, stamp_ns=NOW_NS) -> GripperSample:
    return GripperSample(opening=opening, effort=effort, stamp_ns=stamp_ns)


def rgb_frame(w=4, h=2, stamp_ns=NOW_NS) -> CameraFrame:
    return CameraFrame(
        encoding="rgb8", width=w, height=h, data=bytes(w * h * 3), stamp_ns=stamp_ns
    )


def capture(cache, plan, depth_units=()):
    units = None if depth_units is None else list(depth_units)
    return capture_schema(cache, plan, units, NOW_NS, STALENESS_S)


def test_stamp_to_ns_rejects_pre_epoch_and_non_finite():
    assert stamp_to_ns(0.0) is None
    assert stamp_to_ns(-1.0) is None
    # int() raises on these, which would escape as a per-message route error.
    assert stamp_to_ns(float("nan")) is None
    assert stamp_to_ns(float("inf")) is None
    assert stamp_to_ns(float("-inf")) is None
    assert stamp_to_ns(1.5) == 1_500_000_000


def joint_message(positions, velocities=(), efforts=(), stamp=1.5):
    return SimpleNamespace(
        stamp=stamp, positions=list(positions), velocities=list(velocities), efforts=list(efforts)
    )


def gripper_message(opening=0.5, effort=0.1, stamp=1.5):
    return SimpleNamespace(stamp=stamp, opening=opening, effort=effort)


def test_state_sample_parses_joint_message():
    parsed = state_sample(LinkKind.JOINT, joint_message([1.0, 2.0], velocities=[0.1, 0.2]))
    assert parsed == JointSample(
        positions=(1.0, 2.0), velocities=(0.1, 0.2), efforts=(), stamp_ns=1_500_000_000
    )
    assert state_sample(LinkKind.JOINT, joint_message([1.0], stamp=0.0)) is None
    assert state_sample(LinkKind.JOINT, joint_message([float("nan")])) is None
    assert state_sample(LinkKind.JOINT, joint_message([1.0], efforts=[float("inf")])) is None


def test_state_sample_parses_gripper_message():
    parsed = state_sample(LinkKind.GRIPPER, gripper_message(opening=0.7, effort=0.3))
    assert parsed == GripperSample(opening=0.7, effort=0.3, stamp_ns=1_500_000_000)
    assert state_sample(LinkKind.GRIPPER, gripper_message(opening=float("nan"))) is None
    assert state_sample(LinkKind.GRIPPER, gripper_message(stamp=-1.0)) is None


def test_schema_locks_layouts_and_derives_names():
    plan = make_plan(state=(joint_entry(), gripper_entry()))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7, efforts=True)
    cache.links[GRIP0] = grip()
    schema = capture(cache, plan)
    assert schema.layouts == (
        LinkLayout(dims=7, has_velocities=True, has_efforts=True),
        GRIPPER_LAYOUT,
    )
    names = state_dim_names(plan, schema)
    assert names[:2] == ("arm0_j0", "arm0_j1")
    assert names[-1] == "grip0_opening"
    assert len(names) == 8
    assert velocities_dim_names(plan, schema) == names[:7]
    # The gripper's effort dimension carries force, so it is not the
    # opening its state dimension carries.
    assert efforts_dim_names(plan, schema) == (*names[:7], "grip0_effort")


def test_schema_then_sample_roundtrip():
    plan = make_plan(state=(joint_entry(), gripper_entry()), color=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7, efforts=True)
    cache.links[GRIP0] = grip(opening=0.5, effort=0.3)
    cache.color = [rgb_frame()]
    schema = capture(cache, plan)

    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.state.shape == (8,)
    assert row.state.dtype == np.float32
    assert row.state[7] == pytest.approx(0.5)
    # Velocity covers the sources that deliver it; effort adds grip effort.
    assert row.velocities.shape == (7,)
    assert row.efforts.shape == (8,)
    assert row.efforts[7] == pytest.approx(0.3)
    assert row.images["cam0"].shape == (2, 4, 3)


def test_vectors_absent_when_wire_omits_them():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(2, velocities=False)
    schema = capture(cache, plan)
    assert schema.layouts == (LinkLayout(dims=2, has_velocities=False, has_efforts=False),)
    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.velocities is None
    assert row.efforts is None


def test_gripper_effort_recorded_even_without_arm_efforts():
    plan = make_plan(state=(joint_entry(), gripper_entry()))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    cache.links[GRIP0] = grip(effort=0.3)
    schema = capture(cache, plan)
    assert efforts_dim_names(plan, schema) == ("grip0_effort",)
    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.efforts.shape == (1,)
    assert row.efforts[0] == pytest.approx(0.3)


def test_schema_requires_every_link():
    plan = make_plan(state=(joint_entry(), gripper_entry()))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    with pytest.raises(NotReady, match="grip0"):
        capture(cache, plan)


def test_schema_requires_fresh_links():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7, stamp_ns=NOW_NS - 10_000_000_000)
    with pytest.raises(NotReady, match="stale"):
        capture(cache, plan)


def test_layout_rejects_empty_joints():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = JointSample(positions=(), velocities=(), efforts=(), stamp_ns=NOW_NS)
    with pytest.raises(NotReady, match="no joints"):
        capture(cache, plan)


def test_layout_rejects_mismatched_vector_lengths():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = JointSample(
        positions=(0.1,) * 7, velocities=(0.0,) * 3, efforts=(), stamp_ns=NOW_NS
    )
    with pytest.raises(NotReady, match="3 velocities for 7 joints"):
        capture(cache, plan)


def test_schema_requires_fresh_camera():
    plan = make_plan(state=(joint_entry(),), color=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    cache.color = [rgb_frame(stamp_ns=NOW_NS - 10_000_000_000)]
    with pytest.raises(NotReady, match="no fresh frame"):
        capture(cache, plan)


def test_schema_mismatch_flags_layout_drift():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    locked = capture(cache, plan)
    cache.links[ARM0] = joints(8)
    assert "changed shape" in schema_mismatch(locked, capture(cache, plan))
    cache.links[ARM0] = joints(7, velocities=False)
    assert "changed shape" in schema_mismatch(locked, capture(cache, plan))
    cache.links[ARM0] = joints(7)
    assert schema_mismatch(locked, capture(cache, plan)) is None


def test_sample_tolerates_future_stamps():
    """Producer/recorder clock skew within the gate must not read as stale:
    a future stamp ages as zero rather than negative."""
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    schema = capture(cache, plan)
    cache.links[ARM0] = joints(7, stamp_ns=NOW_NS + 10_000_000_000)
    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.state.shape == (7,)


def test_action_holds_last_setpoint_regardless_of_age():
    """A setpoint is a latest-wins command: the last one stays the action
    until replaced, so age never gates it the way state staleness does."""
    plan = make_plan(state=(joint_entry(),), action=(action_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    cache.links[COMMANDED_ARM0] = joints(7, stamp_ns=NOW_NS - 600_000_000_000)
    schema = capture(cache, plan)
    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.action.shape == (7,)


def test_action_sample_reads_gripper_setpoints_without_effort():
    # Gripper setpoints carry no effort field; the sample must not touch one.
    message = SimpleNamespace(stamp=NOW_NS / 1e9, opening=0.4)
    s = action_sample(LinkKind.GRIPPER, message)
    assert s.opening == 0.4
    assert s.effort == 0.0


def test_sample_flags_stale_link_by_producer_stamp():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    schema = capture(cache, plan)
    cache.links[ARM0] = joints(7, stamp_ns=NOW_NS - 300_000_000)
    with pytest.raises(SampleGap, match="stale"):
        sample(cache, schema, plan, NOW_NS, STALENESS_S)


def test_sample_flags_shape_change():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    schema = capture(cache, plan)
    cache.links[ARM0] = joints(8)
    with pytest.raises(SampleGap, match="carries 8 joints"):
        sample(cache, schema, plan, NOW_NS, STALENESS_S)


def test_sample_flags_dropped_velocities():
    plan = make_plan(state=(joint_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    schema = capture(cache, plan)
    cache.links[ARM0] = joints(7, velocities=False)
    with pytest.raises(SampleGap, match="dropped velocities"):
        sample(cache, schema, plan, NOW_NS, STALENESS_S)


def test_sample_flags_resolution_change():
    plan = make_plan(state=(joint_entry(),), color=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    cache.color = [rgb_frame(w=4, h=2)]
    schema = capture(cache, plan)
    cache.color = [rgb_frame(w=8, h=2)]
    with pytest.raises(SampleGap, match="resolution"):
        sample(cache, schema, plan, NOW_NS, STALENESS_S)


def test_sample_flags_mjpeg_payload_header_mismatch():
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="JPEG")
    plan = make_plan(state=(joint_entry(),), color=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    # Header claims 4x4; the embedded JPEG is 2x2.
    cache.color = [
        CameraFrame(encoding="mjpeg", width=4, height=4, data=buffer.getvalue(), stamp_ns=NOW_NS)
    ]
    schema = capture(cache, plan)
    with pytest.raises(SampleGap, match="does not match header"):
        sample(cache, schema, plan, NOW_NS, STALENESS_S)


def test_sample_flags_truncated_payload():
    plan = make_plan(state=(joint_entry(),), color=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    cache.color = [rgb_frame()]
    schema = capture(cache, plan)
    cache.color = [
        CameraFrame(encoding="rgb8", width=4, height=2, data=bytes(5), stamp_ns=NOW_NS)
    ]
    with pytest.raises(SampleGap, match="cam0"):
        sample(cache, schema, plan, NOW_NS, STALENESS_S)


def make_schema(color=()) -> SourceSchema:
    return SourceSchema(
        layouts=(),
        action_layouts=(),
        color_geometry=tuple(color),
        rgbd_geometry=(),
        depth_geometry=(),
        depth_units=(),
    )


def test_schema_mismatch_reports_camera_drift():
    base = make_schema(color=((4, 2),))
    assert schema_mismatch(base, base) is None
    assert "resolution" in schema_mismatch(base, make_schema(color=((8, 2),)))


def test_decode_bgr8_swaps_channels():
    data = bytes([10, 20, 30] * 2)
    frame = CameraFrame(encoding="bgr8", width=2, height=1, data=data, stamp_ns=NOW_NS)
    rgb = decode_color(frame)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_decode_depth_scales_to_meters():
    z16 = np.array([[1000, 2000]], dtype="<u2").tobytes()
    frame = CameraFrame(encoding="z16", width=2, height=1, data=z16, stamp_ns=NOW_NS)
    meters = decode_depth(frame, 0.001)
    assert meters.shape == (1, 2, 1)
    assert meters.dtype == np.float32
    assert meters[0, 0, 0] == pytest.approx(1.0)
    assert meters[0, 1, 0] == pytest.approx(2.0)


def test_yuyv_decodes_grey():
    # Limited-range mid grey: Y=128, U=V=128 -> 1.164384*(128-16) = 130.4.
    frame = CameraFrame(
        encoding="yuyv", width=2, height=1, data=bytes([128, 128, 128, 128]), stamp_ns=NOW_NS
    )
    rgb = decode_color(frame)
    assert rgb.shape == (1, 2, 3)
    assert np.allclose(rgb, 130, atol=1)


def test_yuyv_range_endpoints():
    # Limited-range black (Y=16) is RGB 0 and reference white (Y=235) is RGB
    # 255; footroom and headroom codes clamp instead of wrapping.
    def decode(y):
        frame = CameraFrame(
            encoding="yuyv", width=2, height=1, data=bytes([y, 128, y, 128]), stamp_ns=NOW_NS
        )
        return decode_color(frame)

    assert np.array_equal(decode(16), np.zeros((1, 2, 3), dtype=np.uint8))
    assert np.array_equal(decode(235), np.full((1, 2, 3), 255, dtype=np.uint8))
    assert np.array_equal(decode(0), np.zeros((1, 2, 3), dtype=np.uint8))
    assert np.array_equal(decode(255), np.full((1, 2, 3), 255, dtype=np.uint8))


def test_max_episode_cap_stays_within_float32_timestamp_tolerance():
    """The dataset library stores frame timestamps as float32 k/fps and reads
    them back against a 1e-4 s tolerance; rounding first exceeds it past
    2048 s. Every frame up to the cap must stay loadable."""

    def rounding_error(t: float) -> float:
        return abs(struct.unpack("f", struct.pack("f", t))[0] - t)

    for fps in (15, 30, 60):
        assert max(rounding_error(k / fps) for k in range(MAX_EPISODE_S * fps)) < 1e-4
    # 2048.03 s, the first failing 30 fps timestamp the cap stays under.
    assert rounding_error(61441 / 30) > 1e-4


def gripper_action_entry(i=0) -> SourceEntry:
    return SourceEntry(
        key=(CORE, f"commanded_grip{i}", "link"), kind=LinkKind.GRIPPER, feature_key=f"grip{i}"
    )


def test_action_falls_back_to_measured_until_commanded():
    """No command yet records as hold-in-place: schema capture and sampling
    take the paired state link's values, then follow the real setpoint the
    moment one arrives."""
    plan = make_plan(
        state=(joint_entry(), gripper_entry()),
        action=(action_entry(), gripper_action_entry()),
    )
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    cache.links[GRIP0] = grip(opening=0.4, effort=0.3)

    schema = capture(cache, plan)
    assert [layout.dims for layout in schema.action_layouts] == [7, 1]

    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.action[:7].tolist() == pytest.approx([0.1] * 7)
    assert row.action[7] == pytest.approx(0.4)

    commanded = JointSample(
        positions=(0.9,) * 7, velocities=(), efforts=(), stamp_ns=NOW_NS
    )
    cache.links[COMMANDED_ARM0] = commanded
    cache.links[gripper_action_entry().key] = grip(opening=0.8, effort=0.0)
    row = sample(cache, schema, plan, NOW_NS, STALENESS_S)
    assert row.action[:7].tolist() == pytest.approx([0.9] * 7)
    assert row.action[7] == pytest.approx(0.8)


def test_action_layout_normalizes_optional_vectors():
    """A layout captured from a real setpoint stream carrying velocities must
    equal one captured through the fallback, or a mid-session first command
    would falsely read as a shape change."""
    plan = make_plan(state=(joint_entry(),), action=(action_entry(),))
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    fallback_schema = capture(cache, plan)
    cache.links[COMMANDED_ARM0] = joints(7, velocities=True, efforts=True)
    real_schema = capture(cache, plan)
    assert fallback_schema.action_layouts == real_schema.action_layouts


def test_action_without_command_or_paired_state_still_refuses():
    plan = make_plan(
        state=(joint_entry(),),
        action=(
            action_entry(),
            SourceEntry(
                key=(CORE, "commanded_arm9", "link"), kind=LinkKind.JOINT, feature_key="arm9"
            ),
        ),
    )
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(7)
    with pytest.raises(NotReady, match="commanded_arm9/link has not produced"):
        capture(cache, plan)


def test_unusable_depth_units_are_rejected_where_they_enter():
    """Every depth sample is scaled by its unit, so an unusable one would
    encode a whole dataset of meaningless metres."""
    plan = make_plan(state=(joint_entry(),), rgbd=1)
    for unit in (0.0, -0.001, float("nan"), float("inf")):
        with pytest.raises(NotReady, match="depth_unit"):
            validate_depth_units(plan, [unit])
    validate_depth_units(plan, [0.001])


def test_a_probe_before_creation_needs_no_depth_units():
    """A pre-creation liveness probe has no polled units; fabricating some
    would refuse the goal blaming a camera for a number it never sent."""
    plan = make_plan(state=(joint_entry(),), rgbd=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(2)
    cache.rgbd_video = [rgb_frame()]
    cache.rgbd_depth = [rgb_frame()]
    assert capture(cache, plan, depth_units=None).depth_units == ()
    assert capture(cache, plan, depth_units=(0.001,)).depth_units == (0.001,)


def test_schema_rejects_degenerate_camera_geometry():
    plan = make_plan(state=(joint_entry(),), color=1)
    cache = Cache.for_plan(plan)
    cache.links[ARM0] = joints(2)
    cache.color = [CameraFrame(encoding="rgb8", width=0, height=0, data=b"", stamp_ns=NOW_NS)]
    with pytest.raises(NotReady, match="0x0"):
        capture(cache, plan)
