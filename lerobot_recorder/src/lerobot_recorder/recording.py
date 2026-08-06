"""Latest-value cache keyed by observed link, the source schema captured at
the first episode, and the pure zero-order-hold sampler.

Every cached sample carries its producer stamp; producers stamp from the
daemon-resolved peppy clock and staleness is measured against the recorder's
reading of that same clock, so freshness holds across hosts and under sim
time only when both sides honor that convention. The sampler never waits for
a fresher value, so no future data leaks into a frame.
Link vector layouts (joint counts, which optional vectors a source delivers)
come from each source's first message and are locked into the schema; a wire
message that no longer fits ends the episode.
Throughout this module "action" is LeRobot's action feature (the commanded
setpoints recorded per frame), not a peppy action.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np

from .plan import LinkKind, RecordingPlan, SourceEntry, SourceKey

# Episodes auto-stop at this length. The library stores frame timestamps as
# float32 k/fps and reads them back against a fixed 1e-4 s tolerance; float32
# rounding first exceeds that at 2048.03 s (30 fps), so 2000 keeps every
# frame loadable. It also bounds the open-episode cost: staged frame images
# only become video at save, and the tabular buffer lives in RAM until then.
MAX_EPISODE_S = 2000
# Free space is re-checked mid-episode on this cadence against the
# min_remaining_disk_bytes floor.
DISK_CHECK_PERIOD_S = 1.0
# An episode whose sampling falls this far behind the fps grid ends: its
# frames would claim k/fps spacing they no longer have.
MAX_SCHEDULE_LAG_S = 1.0


@dataclass(frozen=True)
class JointSample:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    efforts: tuple[float, ...]
    stamp_ns: int


@dataclass(frozen=True)
class GripperSample:
    opening: float
    effort: float
    stamp_ns: int


LinkSample = JointSample | GripperSample


@dataclass(frozen=True)
class CameraFrame:
    encoding: str
    width: int
    height: int
    data: bytes
    stamp_ns: int


@dataclass
class Cache:
    """One latest-value slot per observed link and per bound camera stream,
    written by the drain tasks and read by the sampler; all access stays on
    the event loop."""

    links: dict[SourceKey, LinkSample | None]
    color: list[CameraFrame | None]
    rgbd_video: list[CameraFrame | None]
    rgbd_depth: list[CameraFrame | None]

    @staticmethod
    def for_plan(plan: RecordingPlan) -> Cache:
        return Cache(
            links={entry.key: None for entry in (*plan.state, *plan.action)},
            color=[None] * len(plan.color_cameras),
            rgbd_video=[None] * len(plan.rgbd_cameras),
            rgbd_depth=[None] * len(plan.rgbd_cameras),
        )


def stamp_to_ns(stamp_s: float) -> int | None:
    """Producer stamp (epoch seconds) as integer nanoseconds; None for a
    pre-epoch or non-finite stamp a consumer must not anchor staleness on."""
    if not math.isfinite(stamp_s) or stamp_s <= 0:
        return None
    return int(stamp_s * 1_000_000_000)


def state_sample(kind: LinkKind, message) -> LinkSample | None:
    """Cacheable sample from one measured-stream message; None when it cannot
    anchor a frame (pre-epoch stamp or non-finite values)."""
    if kind is LinkKind.JOINT:
        return _joint_sample(message)
    return _gripper_sample(message.stamp, message.opening, message.effort)


def _joint_sample(message) -> JointSample | None:
    stamp_ns = stamp_to_ns(message.stamp)
    if stamp_ns is None:
        return None
    positions = tuple(message.positions)
    velocities = tuple(message.velocities)
    efforts = tuple(message.efforts)
    values = (*positions, *velocities, *efforts)
    if not all(math.isfinite(v) for v in values):
        return None
    return JointSample(
        positions=positions, velocities=velocities, efforts=efforts, stamp_ns=stamp_ns
    )


def action_sample(kind: LinkKind, message) -> LinkSample | None:
    """Cacheable sample from one setpoint message. Joint setpoints share the
    state wire shape; gripper setpoints carry no effort (the cap is not the
    action), so the sample's effort stays 0 and is never recorded."""
    if kind is LinkKind.JOINT:
        return _joint_sample(message)
    return _gripper_sample(message.stamp, message.opening, 0.0)


def _gripper_sample(stamp: float, opening: float, effort: float) -> GripperSample | None:
    stamp_ns = stamp_to_ns(stamp)
    if stamp_ns is None or not (math.isfinite(opening) and math.isfinite(effort)):
        return None
    return GripperSample(opening=opening, effort=effort, stamp_ns=stamp_ns)


def _age_s(stamp_ns: int, now_ns: int) -> float:
    return max(0, now_ns - stamp_ns) / 1e9


@dataclass(frozen=True)
class LinkLayout:
    """One link's locked vector shape: joint count and which optional vectors
    its first message delivered (empty on the wire means unsensed)."""

    dims: int
    has_velocities: bool
    has_efforts: bool


GRIPPER_LAYOUT = LinkLayout(dims=1, has_velocities=False, has_efforts=True)


@dataclass(frozen=True)
class SourceSchema:
    """Source shapes locked at the first episode; a mid-session change is a
    gap. layouts aligns 1:1 with the plan's state entries."""

    layouts: tuple[LinkLayout, ...]
    # Aligns 1:1 with the plan's action entries; only dims matters (the action
    # records commanded positions alone).
    action_layouts: tuple[LinkLayout, ...]
    color_geometry: tuple[tuple[int, int], ...]
    rgbd_geometry: tuple[tuple[int, int], ...]
    depth_geometry: tuple[tuple[int, int], ...]
    # meters per z16 LSB, one per rgbd producer (from depth_stream_info).
    depth_units: tuple[float, ...]


def state_dim_names(plan: RecordingPlan, schema: SourceSchema) -> tuple[str, ...]:
    return tuple(
        name
        for entry, layout in zip(plan.state, schema.layouts, strict=True)
        for name in _entry_dim_names(entry, layout)
    )


def velocities_dim_names(plan: RecordingPlan, schema: SourceSchema) -> tuple[str, ...]:
    return tuple(
        name
        for entry, layout in zip(plan.state, schema.layouts, strict=True)
        if layout.has_velocities
        for name in _entry_dim_names(entry, layout)
    )


def efforts_dim_names(plan: RecordingPlan, schema: SourceSchema) -> tuple[str, ...]:
    return tuple(
        name
        for entry, layout in zip(plan.state, schema.layouts, strict=True)
        if layout.has_efforts
        # A gripper's effort dimension carries force, not the opening its
        # state dimension carries, so it cannot share that name.
        for name in _entry_dim_names(entry, layout, gripper_quantity="effort")
    )


def action_dim_names(plan: RecordingPlan, schema: SourceSchema) -> tuple[str, ...]:
    return tuple(
        name
        for entry, layout in zip(plan.action, schema.action_layouts, strict=True)
        for name in _entry_dim_names(entry, layout)
    )


def _entry_dim_names(
    entry: SourceEntry, layout: LinkLayout, *, gripper_quantity: str = "opening"
) -> tuple[str, ...]:
    """Dimension names for one link. A joint link names its joints, which are
    the same in every feature; a gripper link's single dimension names the
    quantity that feature carries."""
    if entry.kind is LinkKind.GRIPPER:
        return (f"{entry.feature_key}_{gripper_quantity}",)
    return tuple(f"{entry.feature_key}_j{i}" for i in range(layout.dims))


class NotReady(Exception):
    """A precondition for capturing the schema (or starting) is unmet."""


class SampleGap(Exception):
    """A source went missing, stale, or changed shape mid-episode: the episode
    ends with a save and this reason in the result."""


def _fresh_link_sample(
    cache: Cache, entry: SourceEntry, now_ns: int, staleness_s: float, gap: type[Exception]
) -> LinkSample:
    sample = cache.links[entry.key]
    if sample is None:
        raise gap(f"link {entry.label} has not produced yet")
    if _age_s(sample.stamp_ns, now_ns) > staleness_s:
        raise gap(f"link {entry.label} stale")
    return sample


def _hold_in_place(sample: LinkSample) -> LinkSample:
    """The action recorded before any command exists: the measured pose as a
    setpoint. The gripper cap is not the action, so effort stays 0 exactly as
    it does for a real gripper setpoint."""
    if isinstance(sample, JointSample):
        return JointSample(
            positions=sample.positions, velocities=(), efforts=(), stamp_ns=sample.stamp_ns
        )
    return GripperSample(opening=sample.opening, effort=0.0, stamp_ns=sample.stamp_ns)


def _held_link_sample(
    cache: Cache, entry: SourceEntry, fallback_keys: dict[SourceKey, SourceKey], gap: type[Exception]
) -> LinkSample:
    """A setpoint is a latest-wins command: the last one remains the action
    until replaced, so age never gates it. Before the first command, the
    paired state link stands in (see _hold_in_place); with neither, the
    source is genuinely absent."""
    sample = cache.links[entry.key]
    if sample is not None:
        return sample
    fallback_key = fallback_keys.get(entry.key)
    fallback = cache.links.get(fallback_key) if fallback_key is not None else None
    if fallback is None:
        raise gap(f"link {entry.label} has not produced yet")
    return _hold_in_place(fallback)


def _dims_only(layout: LinkLayout) -> LinkLayout:
    return LinkLayout(dims=layout.dims, has_velocities=False, has_efforts=False)


def _link_layout(entry: SourceEntry, sample: LinkSample) -> LinkLayout:
    if isinstance(sample, GripperSample):
        return GRIPPER_LAYOUT
    dims = len(sample.positions)
    if dims == 0:
        raise NotReady(f"link {entry.label} reports no joints")
    for what, values in (("velocities", sample.velocities), ("efforts", sample.efforts)):
        if len(values) not in (0, dims):
            raise NotReady(
                f"link {entry.label} reports {len(values)} {what} for {dims} joints"
            )
    return LinkLayout(
        dims=dims,
        has_velocities=len(sample.velocities) == dims,
        has_efforts=len(sample.efforts) == dims,
    )


def validate_depth_units(plan: RecordingPlan, depth_units: list[float]) -> None:
    """Every depth sample is scaled by its unit, so a camera reporting 0, NaN
    or inf would silently encode a whole dataset of meaningless metres.
    Checked where units enter from the wire, not inside every probe."""
    for entry, unit in zip(plan.rgbd_cameras, depth_units, strict=True):
        if not math.isfinite(unit) or unit <= 0:
            raise NotReady(f"rgbd camera {entry.name} reports depth_unit {unit}")


def capture_schema(
    cache: Cache,
    plan: RecordingPlan,
    depth_units: list[float] | None,
    now_ns: int,
    staleness_s: float,
) -> SourceSchema:
    """None depth_units marks a probe from before the units were polled; the
    result then carries no units and exists only for liveness and mismatch
    checks, which do not read them."""
    layouts = tuple(
        _link_layout(entry, _fresh_link_sample(cache, entry, now_ns, staleness_s, NotReady))
        for entry in plan.state
    )
    # Only dims matters for an action layout (positions alone are recorded);
    # normalizing keeps a fallback-captured layout identical to one captured
    # from a real setpoint stream carrying optional vectors.
    action_layouts = tuple(
        _dims_only(_link_layout(entry, _held_link_sample(cache, entry, plan.action_fallback, NotReady)))
        for entry in plan.action
    )

    def fresh_geometry(frames, entries, what):
        geometry = []
        for entry, frame in zip(entries, frames, strict=True):
            if frame is None or _age_s(frame.stamp_ns, now_ns) > staleness_s:
                raise NotReady(f"{what} {entry.name} has no fresh frame")
            if frame.width <= 0 or frame.height <= 0:
                raise NotReady(f"{what} {entry.name} reports {frame.width}x{frame.height}")
            geometry.append((frame.width, frame.height))
        return tuple(geometry)

    return SourceSchema(
        layouts=layouts,
        action_layouts=action_layouts,
        color_geometry=fresh_geometry(cache.color, plan.color_cameras, "camera"),
        rgbd_geometry=fresh_geometry(cache.rgbd_video, plan.rgbd_cameras, "rgbd camera"),
        depth_geometry=fresh_geometry(cache.rgbd_depth, plan.rgbd_cameras, "rgbd depth"),
        depth_units=() if depth_units is None else tuple(depth_units),
    )


def schema_mismatch(locked: SourceSchema, probe: SourceSchema) -> str | None:
    """Refusal reason when live sources no longer fit the dataset schema
    locked at creation."""
    if probe.layouts != locked.layouts or probe.action_layouts != locked.action_layouts:
        return "a source changed shape since the dataset was created"
    locked_geometry = (locked.color_geometry, locked.rgbd_geometry, locked.depth_geometry)
    probe_geometry = (probe.color_geometry, probe.rgbd_geometry, probe.depth_geometry)
    if probe_geometry != locked_geometry:
        return "a camera changed resolution since the dataset was created"
    return None


@dataclass
class FrameRow:
    state: np.ndarray
    # The commanded positions, concatenated in the state's source order.
    action: np.ndarray
    # None when no source delivers the vector (feature absent from the dataset).
    velocities: np.ndarray | None
    efforts: np.ndarray | None
    images: dict[str, np.ndarray] = field(default_factory=dict)


def sample(
    cache: Cache,
    schema: SourceSchema,
    plan: RecordingPlan,
    now_ns: int,
    staleness_s: float,
) -> FrameRow:
    state: list[float] = []
    velocities: list[float] = []
    efforts: list[float] = []
    for entry, layout in zip(plan.state, schema.layouts, strict=True):
        s = _fresh_link_sample(cache, entry, now_ns, staleness_s, SampleGap)
        if isinstance(s, JointSample):
            if len(s.positions) != layout.dims:
                raise SampleGap(
                    f"link {entry.label} carries {len(s.positions)} joints, "
                    f"schema locked {layout.dims}"
                )
            state.extend(s.positions)
            if layout.has_velocities:
                if len(s.velocities) != layout.dims:
                    raise SampleGap(f"link {entry.label} dropped velocities")
                velocities.extend(s.velocities)
            if layout.has_efforts:
                if len(s.efforts) != layout.dims:
                    raise SampleGap(f"link {entry.label} dropped efforts")
                efforts.extend(s.efforts)
        else:
            state.append(s.opening)
            efforts.append(s.effort)

    action: list[float] = []
    for entry, layout in zip(plan.action, schema.action_layouts, strict=True):
        s = _held_link_sample(cache, entry, plan.action_fallback, SampleGap)
        if isinstance(s, JointSample):
            if len(s.positions) != layout.dims:
                raise SampleGap(
                    f"link {entry.label} carries {len(s.positions)} joints, "
                    f"schema locked {layout.dims}"
                )
            action.extend(s.positions)
        else:
            action.append(s.opening)

    row = FrameRow(
        state=np.asarray(state, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        velocities=np.asarray(velocities, dtype=np.float32) if velocities else None,
        efforts=np.asarray(efforts, dtype=np.float32) if efforts else None,
    )

    def take(frame: CameraFrame | None, geometry, entry, decode):
        if frame is None:
            raise SampleGap(f"camera {entry.name} stopped producing")
        if _age_s(frame.stamp_ns, now_ns) > staleness_s:
            raise SampleGap(f"camera {entry.name} silent")
        if (frame.width, frame.height) != geometry:
            raise SampleGap(f"camera {entry.name} changed resolution")
        try:
            image = decode(frame)
        except Exception as e:
            raise SampleGap(f"camera {entry.name}: {e}") from e
        # Self-describing payloads (mjpeg) can disagree with the header.
        width, height = geometry
        if image.shape[:2] != (height, width):
            raise SampleGap(
                f"camera {entry.name} payload {image.shape[1]}x{image.shape[0]} "
                f"does not match header {width}x{height}"
            )
        return image

    for i, entry in enumerate(plan.color_cameras):
        row.images[entry.name] = take(
            cache.color[i], schema.color_geometry[i], entry, decode_color
        )
    for i, entry in enumerate(plan.rgbd_cameras):
        row.images[entry.name] = take(
            cache.rgbd_video[i], schema.rgbd_geometry[i], entry, decode_color
        )
        row.images[f"{entry.name}_depth"] = take(
            cache.rgbd_depth[i],
            schema.depth_geometry[i],
            entry,
            lambda frame, unit=schema.depth_units[i]: decode_depth(frame, unit),
        )
    return row


def decode_color(frame: CameraFrame) -> np.ndarray:
    """Camera payload as HxWx3 RGB uint8."""
    h, w = frame.height, frame.width
    if frame.encoding == "rgb8":
        return np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3)
    if frame.encoding == "bgr8":
        return np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3)[:, :, ::-1]
    if frame.encoding == "yuyv":
        return _yuyv_to_rgb(np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 2))
    if frame.encoding == "mjpeg":
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(frame.data)).convert("RGB"))
    raise ValueError(f"unsupported color encoding {frame.encoding!r}")


def decode_depth(frame: CameraFrame, depth_unit_m: float) -> np.ndarray:
    """z16 payload as HxWx1 float32 meters (what lerobot's depth encoder
    quantizes)."""
    if frame.encoding != "z16":
        raise ValueError(f"unsupported depth encoding {frame.encoding!r}")
    z16 = np.frombuffer(frame.data, dtype="<u2").reshape(frame.height, frame.width)
    return (z16.astype(np.float32) * depth_unit_m)[:, :, np.newaxis]


def _yuyv_to_rgb(yuyv: np.ndarray) -> np.ndarray:
    """Limited-range BT.601 YUYV 4:2:2 to RGB, vectorized per pixel pair.
    UVC sources are limited range per V4L2's default quantization; the same
    convention uvc_camera and the ZED node's OpenCV conversion use."""
    y = 1.164384 * (yuyv[:, :, 0].astype(np.float32) - 16.0)
    u = np.repeat(yuyv[:, 0::2, 1].astype(np.float32), 2, axis=1) - 128.0
    v = np.repeat(yuyv[:, 1::2, 1].astype(np.float32), 2, axis=1) - 128.0
    r = y + 1.596027 * v
    g = y - 0.391762 * u - 0.812968 * v
    b = y + 2.017232 * u
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
