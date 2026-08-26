# OpenArm Isaac Sim 6.0 Integration

Experimental Isaac Sim 6.0.1 integration for OpenArm bimanual robot using Peppy.

This branch is intended for reproducibility testing and community feedback. It provides headless Isaac Sim startup, WebRTC streaming, OpenArm runtime control, custom USD loading and reusable manipulation scenarios.

## Features

- Isaac Sim 6.0.1
- OpenArm v2 bimanual robot
- Peppy runtime integration
- Headless WebRTC streaming
- Runtime TCP commander on port `5556`
- Whole-robot root repositioning
- Runtime 7-DOF arm targets
- Runtime USD spawn / move / remove
- Runtime task scenes:
  - `tabletop`
  - `shelf_reach`
- OpenArm USD assets tracked with Git LFS

## Requirements

Recommended host setup:

- Ubuntu 24.04 LTS
- NVIDIA GPU with a compatible proprietary driver
- Peppy
- Git LFS
- 32 GB RAM recommended

Isaac Sim base image:

```text
nvcr.io/nvidia/isaac-sim:6.0.1
```

Systems with less RAM may require additional swap during image build or startup.

## Build

From the repository root:

```bash
cd openarm_sim_isaac

peppy node sync .

peppy node add . \
  -sb \
  --force \
  --idle-timeout 18000
```

## WebRTC Streaming

Host IP is intentionally not hard-coded.

Set it before starting node:

```bash
export PEPPY_ISAAC_PUBLIC_IP=<YOUR_HOST_IP>
```

Optional port overrides:

```bash
export PEPPY_ISAAC_SIGNAL_PORT=49100
export PEPPY_ISAAC_STREAM_PORT=47998
```

## Run

```bash
peppy node run \
  -i isaac601_openarm \
  --idle-timeout 1800 \
  --max-timeout 7200 \
  openarm_sim_isaac:v1 \
  hardware_version=v2 \
  state_rate_hz=50 \
  headless=true
```

## Runtime Commander

From the repository root:

```bash
python3 commander.py --help
```

List joints:

```bash
python3 commander.py joints
```

Move the complete OpenArm root:

```bash
python3 commander.py robot 1.5 0.0 0.0
```

Command one arm with seven joint targets in radians:

```bash
python3 commander.py arm left 0 -0.10 0 -0.15 0 0.10 0
python3 commander.py arm right 0 0.10 0 0.15 0 -0.10 0
```

Release runtime arm override:

```bash
python3 commander.py release left
python3 commander.py release right
```

## Runtime Scenes

### Tabletop

```bash
python3 commander.py scene tabletop
```

Creates an amber table, red cube end blue tray relative to the current OpenArm pose.

Remove it:

```bash
python3 commander.py remove Tabletop
```

### Shelf Reach

```bash
python3 commander.py scene shelf_reach
```

Creates a multi-level reachability shelf with coloured cubes positioned in front of the current OpenArm pose.

Remove it:

```bash
python3 commander.py remove ShelfReach
```

## Runtime USD Loading

Spawn arbitrary USD asset:

```bash
python3 commander.py spawn \
  MyObject \
  /absolute/path/to/object.usd \
  1.0 0.0 0.8 \
  --scale 1.0
```

Move it:

```bash
python3 commander.py move MyObject 1.2 0.2 0.8
```

Remove it:

```bash
python3 commander.py remove MyObject
```

The USD path must be accessible from the running Isaac Sim container.

## Isaac Sim Environments

The launcher can reference built-in Isaac Sim environments, including:

```text
/Isaac/Environments/Simple_Room/simple_room.usd
/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd
/Isaac/Environments/Office/office.usd
```

The base environment is configured in:

```text
openarm_sim_isaac/robots/openarm/_launcher.py
```

Runtime task scenes are loaded on top of the base environment.

## Git LFS

The OpenArm USD assets are tracked with Git LFS:

```bash
git lfs install
git lfs track 'openarm_sim_isaac/robot_assets/**/*.usd'
git add .gitattributes
git add openarm_sim_isaac/robot_assets
git lfs ls-files
```

Expected `.gitattributes` rule:

```text
openarm_sim_isaac/robot_assets/**/*.usd filter=lfs diff=lfs merge=lfs -text
```

`git lfs ls-files` may be empty until the matching USD files are staged or committed.

## Troubleshooting

Check that the runtime commander is listening:

```bash
ss -lntp | grep 5556
```

Inspect a Peppy run log:

```bash
grep -E \
'Runtime commander|Scene loaded|Runtime command failed|ERROR|Traceback' \
~/.peppy/logs/run/<RUN_ID>.log \
| tail -n 100
```

## Known Limitations

- Runtime task primitives may still need explicit collision, rigid-body annd mass configuration for contact-rich manipulation.
- Converted OBJ assets may not preserve source materials or textures automatically.
- Runtime USD paths must be visible inside the Isaac Sim container.
- WebRTC requires the correct host IP to be supplied through `PEPPY_ISAAC_PUBLIC_IP`.
- Isaac Sim can require substantial RAM, swap, disk space and GPU memory.

## Planned Scenarios

Potential additions include:

- gear assembly
- sorting
- stacking
- bin picking
- shelf replenishment
- bimanual handover
- peg insertion

## Feedback

This branch is intended for testing and feedback.

Useful reports include:

- installation or image-build failures
- WebRTC connection problems
- GPU or memory issues
- OpenArm articulation or joint-order issues
- runtime commander failures
- scene placement or reachability problems
- additional manipulation-scene ideas
- Isaac Sim 6 compatibility issues

When reporting an issue, please include:

```bash
peppy --version
nvidia-smi
git rev-parse --short HEAD
```

and the relevant Peppy run log.

## Status

Experimental / feedback branch.
