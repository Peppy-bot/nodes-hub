# lerobot_recorder smoke test

An end-to-end check with synthetic sources: two `mock_joint_source` instances
feed the recorder's `state_sources` and `action_sources` slots, and the
recorder writes a state-only LeRobot v3 dataset that Python `lerobot` loads.
This exercises the whole node path (cardinality discovery, per-producer drain
routing, runtime schema, fps sampling, dataset write, provenance, services)
without a robot or cameras. Camera encoding is covered separately by the
`lerobot_dataset` crate's compliance harness (color and depth, loader-verified).

## Prerequisites

- The generic contracts must be resolvable (`joint_states`, `joint_commands`,
  `recorder`). Register the contracts checkout:
  `peppy repo add <contracts-hub checkout with these contracts>`
- Register this nodes checkout: `peppy repo add <this nodes-hub checkout>`
- Both node containers built: `peppy node add ./mock_joint_source -sb` and
  `peppy node add ./lerobot_recorder -sb`.

## Run

```sh
peppy stack launch ./lerobot_recorder/smoke/mock_smoke.json5
```

The recorder discovers two 7-joint state sources and two action sources, so the
dataset will have `observation.state` and `action` of width 14 (named
`left_arm_j0..6`, `right_arm_j0..6`), plus `observation.velocity` (the mocks
report velocities).

Drive one episode through the recorder's services (the recorder instance id is
`recorder`; adjust the core node as needed):

```sh
# start, wait a few seconds, stop-and-save
peppy service call recorder start_episode '{ "task": "smoke" }'
sleep 3
peppy service call recorder stop_episode '{ "save": true }'
```

(If your peppy build calls services differently, use whatever
`recorder_status` / `start_episode` / `stop_episode` invocation your CLI
provides; the contract is in `recorder/recorder.json5`.)

Then tear the stack down.

## Verify

The dataset lands at `/tmp/lerobot_smoke/<utc_timestamp>/smoke_dataset`. Load it
with the same compliance environment the crate uses:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("smoke/dataset", root="/tmp/lerobot_smoke/<ts>/smoke_dataset")
print(ds.meta.total_episodes, len(ds))
item = ds[0]
print(item["observation.state"].shape, item["action"].shape)  # both (14,)
assert "observation.velocity" in ds.meta.features
```

`session.json` in the session dir records the parameters, recorder version, and
the observed producers.

## Cameras

To smoke cameras too, bind camera producers (implementing `uvc_camera`,
`rgbd_camera`, or `depth_camera`) to the recorder's `color_cameras` /
`rgbd_cameras` / `depth_cameras` slots in the launcher. Depth needs `libx265`
in the recorder container (already installed).
