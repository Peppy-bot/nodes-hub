# lerobot_recorder

A robot-agnostic teleoperation recorder. It samples any robot's joint state,
commands, and cameras onto a fixed-fps grid and writes a
[LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset per
session, optionally mirrored to S3 or Cloudflare R2. Nothing in the node is
robot-specific: a launcher binds producers to the node's dependency slots, and
the order of those bindings defines the dataset's vector layout.

## How it works

The node implements the `recorder` contract (a `recorder_status` topic plus
`start_episode` / `stop_episode` services) and depends on five cardinality
slots:

| Slot | Contract | Cardinality | Becomes |
|---|---|---|---|
| `state_sources` | `joint_states` | one or more | `observation.state` (+ `observation.velocity`) |
| `action_sources` | `joint_commands` | zero or more | `action` |
| `color_cameras` | `uvc_camera` | zero or more | `observation.images.<key>` (color) |
| `rgbd_cameras` | `rgbd_camera` | zero or more | `observation.images.<key>` + `<key>_depth` |
| `depth_cameras` | `depth_camera` | zero or more | `observation.images.<key>` (depth) |

At startup the node enumerates the producers bound to each slot
(`bound_producers()`) and builds an ordered recording plan. Each drain task
holds one merged subscription that fans in every producer on its slot and
routes messages to a per-producer latest-value cache slot, keyed by the
producer's `(core_node, instance_id)` identity. No sink or disk I/O ever runs on
a drain path: the messaging layer blocks a subscription callback when its buffer
fills, so a stalled drain would freeze every subscription in the node.

The episode manager owns the recorder state machine and an fps pacer in one
select loop, so start/stop and sampling cannot race. On each tick it takes a
zero-order-hold snapshot of the cache (reading only values that have already
arrived, so nothing from the future leaks into a frame) and hands it to the
LeRobot writer, which runs on its own blocking thread. A separate storage task
mirrors immutable files to object storage as they roll over.

## Dataset layout

`observation.state` is the concatenation of every bound state source's
positions, in binding order. If **every** state source also reports velocities,
an `observation.velocity` feature is included. `action` is the concatenation of
every bound action source's commanded positions; with no action sources it
mirrors `observation.state` (a follower holds position, so the measured state is
the effective command). Each camera producer becomes one
`observation.images.<key>` video feature; an RGB-D camera additionally yields a
`<key>_depth` feature when `record_depth` is set.

Because the launcher's binding order is the vector layout, wire producers in the
order a policy should see them (for a bimanual arm, e.g. left arm, left gripper,
right arm, right gripper).

### Dimension names

Per-joint dimension names are derived from each producer's instance id plus the
joint index (e.g. a source bound as `left_arm` yields `left_arm_j0`,
`left_arm_j1`, ...). The generic `joint_states` / `joint_commands` contracts do
not carry names on the wire: the peppy Rust code generator does not yet support
string-array message fields. Choose meaningful instance ids in the launcher to
get readable dataset feature names.

## Output

Each run creates `<output_root>/<utc_timestamp>/` containing:

- `<dataset_name>/` — the LeRobot v3 dataset (valid and loadable after every
  completed episode; an episode in flight when the process dies is lost).
- `session.json` — provenance: all launch parameters, the recorder's version and
  git revision, and the producers observed on each stream.

Sessions never resume: one dataset per run. Combine sessions offline with
LeRobot tooling.

## Depth

RGB-D and depth-only cameras are recorded as single-channel `gray12le`
HEVC-lossless video with `is_depth_map` set, matching LeRobot 0.6. Incoming
`z16` codes are scaled to millimetres by `depth_unit_m` (metres per LSB;
RealSense default `0.001`) and log-quantized so the Python loader dequantizes
back to metres. Depth needs `libx265`; set `record_depth: false` to skip it.

## Storage

`storage_backend`:

- `local` (default) — the dataset stays on disk only.
- `s3` / `r2` — the dataset is also mirrored to a bucket. Immutable chunk files
  upload as they roll over; metadata and any still-open chunks are synced at
  session end, so multi-GB files are never re-uploaded while still being
  appended. Set `storage_bucket`, `storage_endpoint` (required for R2:
  `https://<account>.r2.cloudflarestorage.com`), `storage_region` (`auto` for
  R2), and `storage_prefix`. Credentials come from the standard
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables.

## Episode control

Call the `start_episode` service (with a task string, or empty for the node's
`default_task`) to begin, and `stop_episode` (`save: true` to keep,
`false` to discard) to end. `recorder_status` streams the state, current
episode, frame count, task, a human-readable message, and free disk space.

`start_episode` is refused until every bound source has produced at least one
message (so joint counts and camera geometry are known) and every camera has a
fresh frame, and under a 1 GiB free-disk floor. An episode auto-stops (and
saves) if a state source goes stale, a camera falls silent past
`camera_timeout_s`, the video encoder cannot keep up, or `max_episode_s` is
reached.

## Parameters

Required: `robot_type`, `fps`, `output_root`, `dataset_name`, `default_task`,
`launcher_id`. Notable optional: `video_codec` (`libx264` default / `libsvtav1`),
`camera_keys` (`"instance_id=key,..."` overrides; unlisted cameras use a
sanitized instance id), `record_depth`, `depth_unit_m`, the `storage_*` set,
`state_staleness_s`, `camera_start_fresh_s`, `camera_timeout_s`, `max_episode_s`,
`status_rate_hz`. See `peppy.json5` for defaults and full descriptions.

## Requirements

`ffmpeg` and `ffprobe` on PATH (the container installs them), with `libx264`
(or `libsvtav1`) for color and `libx265` for depth. Producers implementing the
`joint_states` (and optionally `joint_commands`) and camera contracts, streaming
while recording.

## Development

The `lerobot_dataset` dependency is a path to the sibling `public-peppy-libs`
worktree until that branch is pushed, after which it becomes a git pin.
Regenerate bindings with `peppy node sync` (the generic contracts must be
resolvable, e.g. `peppy repo add` the contracts checkout that provides
`joint_states`, `joint_commands`, and `recorder`).
