# lerobot_recorder

Records a robot's pairing traffic plus any bound cameras into a
[LeRobot v3](https://github.com/huggingface/lerobot) dataset. The node is a
read-only observer: it attaches to each pairing's measured back-channel
(`observation.state`, plus velocities/efforts where sensed) and to the same
pairing's leader setpoint stream (LeRobot's `action` feature: commanded
positions and gripper openings) without claiming either endpoint, samples
every source onto a fixed fps grid, and writes episodes driven by the
`record_episode` peppy action. The two "action"s are unrelated: LeRobot's is
a per-frame dataset feature, peppy's is a goal/feedback/result RPC. A
setpoint is a latest-wins command, so the action feature holds the last
commanded value between messages rather than aging out the way state
staleness does.

### What gets recorded

Nothing here names a robot. The observer slots are `zero_or_more`, so a
launcher binds as many `joint_link` and `gripper_link` pairings as its robot
has and the node records exactly those: one arm or two, grippers or none.

A limb is one measured source paired with the commanded source bound at the
same position in the matching slot, so the launcher's binding order is what
says which leader drives which follower. Bind a slot's measured and commanded
sides in the same order; a count that does not line up is refused at startup,
because nothing here can guess which leader a follower was missing. Pairing
the two sides by position (rather than by instance) is what lets one leader
instance drive every limb, which is the shape the openarm backbone has.

A limb is named after the follower instance observed for it, and dimension
names are that name plus a joint index (`left_arm_inst_j0`); a gripper's
single dimension names the quantity its feature carries
(`left_grip_inst_opening` in state, `left_grip_inst_effort` in efforts). An
instance that follows several pairings of one kind takes the observed link
into its name to stay distinct. Joint counts and which optional vectors
(velocities, efforts) a source delivers are discovered from its first message.
See `peppy.json5` for the parameter reference.

Requires peppy v0.23.0 or newer: the observed membership is read from the
boot config the daemon stamps at spawn, and an older daemon does not stamp
it (startup refuses with an error naming the skew).

### Sessions

A session is one LeRobot dataset holding every episode recorded since the
session opened, exactly the unit lerobot's own recording flow produces: its
`lerobot-record` script records episodes into a single dataset and calls
`finalize()` once at the end of the run. The `finish_session` service is
that end-of-run step without a restart: it finalizes the current dataset
(parquet footers), completes the mirror, and opens a fresh session in
place. A finished session directory is immediately loadable and replayable;
the live session is not (its open files lack footers until finalize, a
lerobot format property). Recording continues into the new session.
Training across several session datasets is standard lerobot practice
(`MultiLeRobotDataset`, or merge them with
`lerobot-edit-dataset --operation.type merge`). `finish_session` refuses
while an episode is recording and when the session has no episodes;
stopping the node finishes the current session the same way.

### Resuming a session

`resume_session` swaps the recorder onto a past session by its directory
name (the one `finish_session` returned), so new episodes append to that
dataset: lerobot's own record, finalize, resume lifecycle. Finished and
unfinished sessions alike are resumable as long as they hold at least one
saved episode; the library appends in fresh chunk files, so finished files
are never rewritten and the session can be finished again afterwards. The
current session must be empty; it is abandoned in place, never deleted.
Resume is refused until every live source is fresh, because the session's
recorded features are checked against the live sources before anything is
constructed; a session recorded with different cameras, rates, or joints
is refused with the difference named. Each session also records which
source fed which dimension (`session.json`), and a launch that binds the
same sources to different limbs is refused: features alone cannot catch
that swap, since limbs keep their follower-derived names either way. After
an unclean stop a session
remains resumable, but episodes since its last chunk rotation may be
unreadable (their parquet footer was never written).

### The action feature

The dataset's `action` feature holds the leader's commanded positions. A
commanded link that has not produced yet falls back to the state link
observing the same limb: before the first command, the recorded action is
hold-in-place (the measured pose), so recording works from a cold start in
Actions mode or with only one side driven. The moment a real setpoint
arrives, the action follows it.

## Storage

`storage_root` is an absolute host directory, bind-mounted same-path into
the container by the manifest's `mount_paths`, so sessions persist on the
host by contract rather than by any default runtime bind. Each run creates
`<storage_root>/<utc_ts>/`; a node restart with the same root finds the
previous runs' sessions, which is what makes them resumable by name. The
node refuses to start when the root does not exist, because a missing root
means the mount is not in effect and datasets would die with the container.

With `s3_uri` set (`s3://bucket[/prefix]`), the same root doubles as the
staging area: the dataset mirrors to the bucket after every saved episode
plus once at shutdown, and the root keeps the local copy. Still-growing
files (the open parquet, the current video chunk) are unreadable in the
mirror until the dataset is finalized and mirrored, which `finish_session`
does on demand and shutdown does once more for the session left open.

The shutdown finalize-and-mirror runs inside the daemon's cooperative
shutdown window (`shutdown_grace_secs`, 5 s by default), so a large
unmirrored backlog can be cut off mid-pass; the node logs
`final mirror pass starting` and a `mirror complete` line, and a missing
completion line means the remote copy is behind the local one. Prefer
`finish_session` before stopping the node (it runs the same finalize and
mirror with no time bound), raise `shutdown_grace_secs` for stacks that
stop mid-session, and recover an interrupted pass by re-syncing the kept
local copy: `aws s3 sync <session dir> s3://bucket/prefix/<session>`.

### S3 / R2 credentials

Credentials come from the standard AWS environment variables, never from
launch parameters. The node refuses to start with `s3_uri` set and no
credentials in the environment.

| Variable | Required | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | yes | |
| `AWS_SECRET_ACCESS_KEY` | yes | |
| `AWS_ENDPOINT_URL` | non-AWS stores | bare host, no bucket or path: `https://<account_id>.r2.cloudflarestorage.com` |
| `AWS_ENDPOINT_URL_S3` | no | S3-scoped variant; wins over `AWS_ENDPOINT_URL` when both are set |
| `AWS_DEFAULT_REGION` | recommended | R2 uses `auto` |
| `AWS_SESSION_TOKEN` | no | only for temporary STS credentials |

Example R2 environment:

```sh
export AWS_ACCESS_KEY_ID=<r2-access-key-id>
export AWS_SECRET_ACCESS_KEY=<r2-secret>
export AWS_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
export AWS_DEFAULT_REGION=auto
```

Any S3-compatible store works the same way through `AWS_ENDPOINT_URL`
(`boto3 >= 1.34` reads it natively).

At startup a mirroring target logs the fully resolved destination
(endpoint, bucket, prefix), so a wrong or missing endpoint environment is
visible at launch instead of at the first upload. The `s3_uri` netloc is
the bucket, never a host: the URI names a location inside whatever store
the environment points the client at, which is the standard `s3://`
convention (aws-cli, spark, smart_open).

## Episodes

One `record_episode` goal is one episode. A goal is refused until every
observed source and camera has produced a fresh sample (`max_staleness_s`),
the disk floor holds (`min_remaining_disk_bytes`), and the dataset schema
still matches the live sources. A source going stale or silent mid-episode
ends the episode with a save and the reason in the goal result; episodes
also auto-stop at the length where LeRobot's float32 timestamps would stop
loading. Feedback carries one message per recorded frame with
`disk_free_bytes` riding along about once per second.
