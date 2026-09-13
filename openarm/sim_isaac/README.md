# OpenArm Isaac Sim 6.1 Integration

Experimental Isaac Sim 6.1.0 integration for OpenArm bimanual robot using Peppy.

This branch is intended for reproducibility testing and community feedback. It provides headless Isaac Sim startup, WebRTC streaming, OpenArm runtime control, custom USD loading and reusable manipulation scenarios.

## Features

- Isaac Sim 6.1.0
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
- Detailed OpenArm v2 visual meshes with their embedded materials

## Requirements

Recommended host setup:

- Ubuntu 24.04 LTS
- NVIDIA GPU with the proprietary driver, 595.58.03 or newer
- Peppy
- 32 GB RAM recommended

Isaac Sim base image:

```text
nvcr.io/nvidia/isaac-sim:6.1.0
```

The node builds on `peppybot/openarm-isaac-sim`, which
`openarm/robot_initializer/scripts/build_base_images.sh` produces from that
image with the robot assets and the NGX core library baked in.

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

## Runtime and Performance

Both headless and windowed launches use the packaged
`robots/openarm/config/openarm.sim.kit` experience with physics, USD/RTX rendering
and viewport controls. Headless mode enables WebRTC; `cameras_enabled=true`
enables Replicator for robot camera capture. Extensions resolve from the Isaac Sim
installation, with settings persistence and extension-registry lookup disabled.
Runtime scenes and props use `isaacsim.storage.native`'s default Isaac 6.1 asset
root; `PEPPY_ROBOT_ASSETS_DIR` selects the robot USD directory.

The node targets 60 Hz using wall-monotonic absolute deadlines. Each due iteration
runs one Isaac update, one bridge step, queued runtime and scene commands, then
force expiry and arm targets. All that work counts toward the frame period;
waiting is interruptible by shutdown and long stalls resynchronize the schedule
without unbounded catch-up. Kit's main limiter and global sync-to-present are
disabled so the Python loop owns pacing in both headless and windowed modes.
The streamer can re-enable the main limiter at startup or on connection and
reconnection. Before each due update, the loop checks that setting and clears it
only if enabled, preventing an extra app-only wait that excludes bridge work.
`state_rate_hz` only limits state and clock publications, not physics or bridge
stepping.

The node renders with RTX Real-Time 2.0 (`RealTimePathTracing`), DLSS
(`anti_aliasing=3`) and an initial viewport render resolution of 1280x720. That
renderer denoises only through DLSS Ray Reconstruction, which runs on the NGX
core library shipped with the NVIDIA driver, and Peppy's `--nv` GPU binding does
not carry the host's copy into the container. The base image therefore carries
the core itself: `robot_initializer/scripts/Dockerfile.isaac` takes
`libnvidia-ngx.so.1` from the driver Isaac Sim 6.1 was tested with, 595.58.03,
pinned by version and checksum. The core reads the running driver through NVML
and the DLSS snippets check that version against their own minimum, so the host
needs a driver at least that new, not that exact version. Kit falls back to TAA
without a word when the core is missing and streams raw path-tracing noise, so
after the warmup the launcher reads the effective `/rtx/rendermode` and
`/rtx/post/aa/op` and refuses to run on anything but the requested profile.
WebRTC captures
the app window, not just the viewport, and allows dynamic resizing; the encoded
stream resolution can therefore differ from 1280x720. WebRTC targets 60 fps. DLSS
frame generation stays explicitly disabled with `/rtx-transient/dlssg/enabled=false`
so displayed frames represent real rendered output, not generated intermediate
frames.

Fixed timeline stepping and synchronous rendering keep camera reads aligned with
engine updates. The focused experience limits extension overhead, but 60 Hz is a
target, not a guarantee: scene loading, camera capture and moving-view rendering
can exceed the frame budget and reduce state cadence and the
simulation-time/wall-time ratio. Slow-loop logs measure work only, excluding
deliberate pacing waits.

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

## Robot visual assets

The Isaac base image downloads one complete, prepared bundle from the
`isaac-sim-assets` R2 bucket. Its immutable versioned key and SHA-256 are pinned in
`openarm/robot_initializer/scripts/isaac_assets.env`. The Docker build copies
that file before downloading, so a pin change invalidates the asset layer's
cache. It verifies the archive checksum before extraction and never falls back
to a mutable asset directory. Node image builds and robot startup use only the
baked files, with no GitHub access, asset conversion or conversion dependencies.

The bundle contains:

- `openarm_bimanual_v2.usd`, with its repaired visual references.
- `openarm_v2_visuals.usdc`, referenced through relative paths by all 21 v2 links.
- `openarm_bimanual.usd` and its three `configuration/` layers for v1.
- `openarm_visual_sources.json`, with the upstream revision, input checksums and
  original robot image digest.
- `openarm_description.LICENSE.txt`, the upstream Apache-2.0 license.
- `bundle_manifest.json`, with output checksums, tool versions, converter script
  hashes and validation results.

Each material-bearing mesh region keeps its source color, triangle topology,
normals and scene transform. The body scale, mirrored left-arm frames and gripper
offsets are retained. Link poses, joints, drives, masses and collision geometry
are unchanged. Robot stage dependencies resolve within the bundle; the
`OmniPBR.mdl` shader module is supplied locally by Isaac Sim.

### Asset maintenance

`scripts/build_visuals.py` and `scripts/prepare_assets.py` are CPU-only maintenance
tools, not image-build steps. Preparation verifies every source stage, COLLADA
mesh and the license, converts only the v2 visuals, and checks USD composition,
material bindings, unchanged nonvisual specs and all attachment transforms.
Archives have sorted entries and fixed timestamps, ownership and permissions.
Preparation refuses to overwrite an existing output.

From the repository root, extract the source stages from the digest pinned in
`scripts/visual_sources.json`. Creating the source container does not run Isaac
or require a GPU:

```bash
source_image=$(python3 -c 'import json; print(json.load(open("openarm/sim_isaac/scripts/visual_sources.json"))["robot"]["image"])')
source_container=$(docker create --platform linux/amd64 "$source_image")
mkdir -p /tmp/openarm-isaac-source
docker cp "$source_container:/opt/robot_assets/openarm/isaac/." /tmp/openarm-isaac-source/
docker rm "$source_container"

uv run --no-project --python 3.11 --with usd-core==26.5 \
  --with-requirements openarm/sim_isaac/scripts/requirements-visuals.txt \
  python openarm/sim_isaac/scripts/prepare_assets.py \
  --source-dir /tmp/openarm-isaac-source \
  --output /tmp/openarm-isaac-assets.tar.gz
```

The tool downloads checksum-pinned DAEs from the immutable upstream revision.
`--mesh-source-dir <directory>` instead accepts an offline directory containing
`<mesh-name>.dae` files and verifies the same checksums. Only the five pinned
source stages are copied; backup files are not bundled.

Publish the complete archive, not the visual library alone. Assign a bundle
version and retain the archive's checksum in its object key. Conditional creation
prevents overwriting an existing object:

```bash
bundle=/tmp/openarm-isaac-assets.tar.gz
version=2
checksum=$(sha256sum "$bundle" | cut -d ' ' -f 1)
key="openarm/${version}/${checksum}.tar.gz"
AWS_ACCESS_KEY_ID="$WALDO_R2_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$WALDO_R2_SECRET_ACCESS_KEY" \
aws --endpoint-url "$WALDO_R2_JURISDICTION_ENDPOINT" s3api put-object \
  --bucket isaac-sim-assets --key "$key" --body "$bundle" \
  --if-none-match '*' --content-type application/gzip \
  --cache-control 'public, max-age=31536000, immutable'
```

Download the published object and verify its SHA-256 before setting
`ISAAC_ASSETS_KEY` and `ISAAC_ASSETS_SHA256` in `isaac_assets.env`. Bump
`ISAAC_IMAGE_REV` in `build_base_images.sh`, then use an authenticated Docker
builder to publish the base image:

```bash
RCLONE_S3_ACCESS_KEY_ID="$WALDO_R2_ACCESS_KEY_ID" \
RCLONE_S3_SECRET_ACCESS_KEY="$WALDO_R2_SECRET_ACCESS_KEY" \
bash openarm/robot_initializer/scripts/build_base_images.sh --isaac-only
```

The publisher stamps the node's `From:` tag. Commit the pin and tag together,
then restage the node with `peppy node add openarm/sim_isaac -sb --force`.

Run the GPU-free regression suites from the repository root:

```bash
uv run --project openarm/sim_isaac/tests --locked --group dev \
  pytest openarm/sim_isaac/tests
```

On Linux ARM64, PyPI has no `usd-core` distribution. The USD-specific test module
is skipped there; the installer, camera, startup and timing suites still run.
A native OpenUSD installation from conda-forge supports asset preparation and
the USD tests on ARM64 without Isaac Sim or a GPU.

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
