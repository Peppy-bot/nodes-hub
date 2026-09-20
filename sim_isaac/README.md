# sim_isaac

The Isaac Sim 6.1.0 simulation node. It opens an empty stage and stands every
robot that attaches to it, of any model it has an entry for, side by side: an
OpenArm v1, an OpenArm v2 and an SO-101 share one stage, one physics scene and
the same four pairing slots. It runs headless with WebRTC streaming, renders
each robot's own cameras, and serves runtime scene and object editing.

## Features

- Isaac Sim 6.1.0
- Any number of robots on one stage, each of its own model: `openarm_v1`,
  `openarm_v2`, `so101`
- One entry per model under `engine/models/`, beside the entry
  `sim_robot_core` shares with every engine
- Peppy runtime integration over four pairing slots that name no robot
- Headless WebRTC streaming
- Runtime TCP commander on port `5556`
- Whole-robot root repositioning
- Runtime USD spawn / move / remove
- Detailed OpenArm v2 visual meshes with their embedded materials
- The OpenArm v2 head camera, the ZED Mini on its bracket under the head cover,
  from Enactic's CAD

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

The node builds on `peppybot/sim-isaac`, which
`sim_base_images/build_base_images.sh` produces from that image with the robot
assets, the NGX core library and `sim_robot_core` baked in. The robot assets
sit one directory per robot under `/opt/robot_assets` (`openarm/`, `so101/`),
which is the default of `PEPPY_ROBOT_ASSETS_DIR`.

Systems with less RAM may require additional swap during image build or startup.

## Layout

```text
sim_isaac/
  peppy.json5          the node: its contracts, its four slots, its parameters
  apptainer.def        the node image, on peppybot/sim-isaac
  engine/              the engine, copied to /opt/sim_isaac/engine
    launch.py          the entry point
    models/            one entry per model this engine stands
    isaac_models.py    parses them, strictly, at node setup
    config/sim_isaac.kit
  scripts/             CPU-only maintenance tools for the OpenArm bundle
  tests/               GPU-free suites, their own uv project
```

## The four slots

The engine declares one slot per kind of pair, each `zero_or_more`:

| Slot | Pairing | Role | A pair names |
|---|---|---|---|
| `arms` | `joint_link` | follower | an arm of a robot |
| `grippers` | `gripper_link` | follower | a gripper of a robot |
| `rgb_cameras` | `sim_rgb_camera_link` | camera | a colour camera of a robot |
| `rgbd_cameras` | `sim_rgbd_camera_link` | camera | a colour and depth camera of a robot |

Nothing in a slot's name says which robot, limb or camera a pair is, so every
pair is read for it through `sim_robot_core.pairs`:

- a pair's robot is the copy it carries, which is the name the robot attached
  under. A pair with no copy belongs to the only robot on the stage;
- a limb pair's limb is the link the pair comes from on the robot's side, so a
  backbone names its downstream links after its limbs: `left_arm`, `right_arm`,
  `left_gripper`, `right_gripper` on an OpenArm, `arm` and `gripper` on an
  SO-101. Those are the names the robot answers to in `limb_motion` and
  `limb_state`, the names `attach` answers with, and the names the engine logs;
- a camera pair's camera is its relay's name in the copy (`wrist_left`,
  `front`): the relay's instance id without the copy's prefix.

A robot's lease is renewed only while its pairs are its model's: every limb of
its model, and nothing its model lacks. A camera its model carries may go
unpaired, since a camera nobody views is not rendered. When the lease runs
out the stay ends, saying either that none of the robot's limbs were paired or
naming both lists: what its pairs name and what its model has. A robot is
ready once it stands and holds a pair for every limb of its model, so an
SO-101 is ready with its one arm and its one gripper. Each robot on the stage
is held to its own model.

## Models

What a robot of a model is made of is `sim_robot_core`'s entry of that model,
shared with the MuJoCo node: its limbs under the names the robot answers to,
the joints each limb moves, where each gripper's closed pose sits on its
finger joints' range, its cameras with the links they hang from, and the
posture it starts in. `sim_robot_core` is installed in the base image
(`sim_base_images/requirements.isaac.txt`), pinned to one commit.

What Isaac Sim alone knows about a model is `engine/models/<model>.json5`,
parsed strictly at node setup by `engine/isaac_models.py`:

| Key | What it says |
|---|---|
| `stage` | The model's USD, relative to `PEPPY_ROBOT_ASSETS_DIR` |
| `articulation_root` | Where the stage keeps its articulation root, relative to the robot's prim; `"."` for a stage whose default prim carries it |
| `link_prims` | URDF link name to the prim that is that link, where they differ |
| `arm_gains` | `kp`, `kd` and `max_efforts`, one per joint of an arm, applied to every arm |
| `gripper_gains` | `kp`, `kd` and `max_efforts`, one per finger of a gripper, applied to every gripper |
| `gravity_compensation` | Whether gravity is taken off the robot's own links |
| `head_camera` | Whether the robot draws the OpenArm v2 head camera pack |

An unknown key, a gain list of the wrong length or a stage outside the baked
assets stops the node at setup. A model with no entry is refused at `attach`
naming the models the engine stands, and none falls back to another's entry.

Everything downstream of `attach` takes the robot's own model:

- the stage referenced under `/World/<robot>`;
- the posture the robot starts in, authored on each joint's PhysX state and on
  its drive's target as the robot joins, so the arm stands still until its
  first setpoint. A model with no start posture starts where its stage puts
  it;
- gravity compensation, authored on the rigid-body links under the robot's own
  prim;
- one actuator controller per limb, with the model's gains and effort
  ceilings. Gains are what the articulation view takes: per radian on a
  revolute joint, per metre on a prismatic one. A drive's damping is raised to
  critical where `kd` falls short of it;
- the gripper opening, mapped onto each finger's own span through
  `sim_robot_core.models.finger_span`: an OpenArm's fingers close at their
  joints' zero, the SO-101's jaw at its joint's lower limit;
- the camera rig, which is the cameras of the robot's own model, each defined
  under the prim its parent link is on that robot. With `cameras_enabled` a rig
  is mounted on each robot holding a camera pair, and a model with no camera
  renders none;
- the head camera pack, which is the business of the models whose entry says
  `head_camera: true`. The node reads no pack when no model draws it.

### What a model's USD carries

The engine references the stage under a prim of the robot's own name and reads
it through PhysX articulation views, so the USD:

- is authored z-up in metres and names a default prim, which becomes the
  robot's prim. The robot's base sits at that prim's origin;
- carries `PhysicsArticulationRootAPI` where `articulation_root` says, and a
  base fixed to the world;
- has one prim per link, named after the URDF link or mapped by `link_prims`,
  with `PhysicsRigidBodyAPI`. A camera is defined under the prim of its parent
  link, so that name is unique under the robot's prim;
- has one joint prim per joint of the shared entry, named after it, since PhysX
  names a dof after its joint prim. Each is a `PhysicsRevoluteJoint` or a
  `PhysicsPrismaticJoint` with finite limits, which a gripper's span is read
  from, and a `PhysicsDriveAPI`, which the gains and the start posture are
  written to;
- keeps links and joints out of instanceable prims; visual and collision
  geometry below a link may be instanceable;
- carries no physics scene of its own: the stage's one scene runs the CPU
  pipeline the articulation views read and write on;
- uses collision approximations PhysX simulates on a dynamic body on the CPU,
  such as convex hulls.

### Adding a model

1. Ship the model's entry in `sim_robot_core` (`models/<model>.json5`) and move
   the pin in `sim_base_images/requirements.isaac.txt` and
   `tests/pyproject.toml` to the commit that carries it.
2. Bake the model's USD under `/opt/robot_assets/<robot>/` in
   `sim_base_images/Dockerfile.isaac`.
3. Write `engine/models/<model>.json5`.
4. Cover it in `tests/test_isaac_models.py`.

`peppy.json5` stays as it is: a model changes no slot.

## Build

From the repository root:

```bash
cd sim_isaac

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
  -i simulation_inst \
  --idle-timeout 1800 \
  --max-timeout 7200 \
  sim_isaac:v1 \
  cameras_enabled=false \
  state_rate_hz=50 \
  headless=true
```

A robot names its model on its own initializer when it joins.

## Runtime and Performance

Both headless and windowed launches use the packaged
`engine/config/sim_isaac.kit` experience with physics, USD/RTX rendering
and viewport controls. Headless mode enables WebRTC; a robot that pairs a
camera has its model's rig rendered for it through Replicator. Extensions
resolve from the Isaac Sim installation, with settings persistence and
extension-registry lookup disabled. Runtime scenes and props use
`isaacsim.storage.native`'s default Isaac 6.1 asset root;
`PEPPY_ROBOT_ASSETS_DIR` selects the directory the robots' USDs are baked
under.

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
the core itself: `sim_base_images/Dockerfile.isaac` takes
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

The TCP commander on port `5556` takes one JSON command per line. A robot
is moved by the name it joined under, to a position in metres facing a `yaw`
in radians about +Z:

```json
{"command": "move_robot_root", "robot": "alpha", "position": [1.5, 0.0, 0.0], "yaw": 1.57}
```

A robot's limbs are driven through its own backbone over the limb pairings;
the scene commander's `move_robot` reaches the same command.

## Runtime USD Loading

Spawn a USD asset, move it, remove it, over the TCP commander:

```json
{"command": "spawn_usd", "name": "MyObject", "path": "/absolute/path/to/object.usd", "position": [1.0, 0.0, 0.8], "yaw": 0.5, "scale": 1.0}
{"command": "move_object", "name": "MyObject", "position": [1.2, 0.2, 0.8]}
{"command": "remove", "name": "MyObject"}
```

The USD path must be accessible from the running Isaac Sim container. `yaw`
turns the object about +Z in radians; a spawn that names none stands as
authored.

Objects spawned through scene_manipulation (`obj_...` ids) belong to it: the
commander refuses to `spawn`, `spawn-isaac` or `remove` one of those names, and
the node logs the refusal as a failed runtime command. Remove or replace them
through scene_manipulation, which keeps their object state; `move` stays allowed.

## Isaac Sim Environments

The stage opens empty. scene_manipulation's `load_scene` references a built-in
Isaac Sim environment under `/World/RuntimeScene` by its asset id, replacing
the one loaded before it; `get_assets_list` lists them, among them:

```text
scene/simple_room
scene/warehouse_multiple_shelves
scene/office
```

The scene ids and the Isaac Sim paths behind them are in
`engine/_launcher.py`. Robots stand beside the runtime scene, not inside it,
so loading one leaves them where they are; `load_scene` and `clear_scene`
remove every object scene_manipulation spawned.

## Robot asset bundles

The Isaac base image downloads one complete, prepared bundle per robot from
the `isaac-sim-assets` R2 bucket and bakes each under
`/opt/robot_assets/<robot>/`: the OpenArm bundle under `openarm/`, the
SO-101's under `so101/`. Their immutable versioned keys and SHA-256s are
pinned in `sim_base_images/isaac_assets.env`. The Docker build copies that
file before downloading, so a pin change invalidates the asset layer's cache.
It checks every pin, downloads and verifies every archive, and only then
extracts them, never falling back to a mutable asset directory. Robot startup
uses only the baked files and the head camera pack below, with no GitHub
access, asset conversion or conversion dependencies.

### OpenArm visual assets

The OpenArm bundle contains:

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

### Head camera

The bundle is upstream's robot, which has no head camera. The ZED Mini on its
bracket under the head cover, as Enactic's `OpenArm_2.0_w_Head_Camera` CAD
assembly seats it on the pedestal, comes from the pack Waldo's
`tools/openarm_head_camera` (private-nodes-hub) derives from that CAD and
publishes to the public `waldo-assets` store: three OBJ meshes and one convex
collision hull in the frame of the CAD mount's origin, the stereo rig read off
the lens barrels, and provenance. A pack is immutable, keyed by the SHA-256 of
its inventory, and `engine/head_camera.py` pins that digest. The
`apptainer.def` stages the pack at image build into
`engine/assets/head_camera`, the index checked against the
digest and every file against the index, and node setup checks it again and
reads its meshes at every start, for the models whose entry says
`head_camera: true`. Each robot of such a model gets the meshes
under its own `openarm_body_link0`, the chest camera's parent, in the same
matte black as the arm links, with the hull as that link's collider. The
chest camera of each of those models' shared entry must sit at the pack's left
lens front, the eye the real ZED's rectified stream comes from, and the node
refuses to start otherwise. An OpenArm v1 and an SO-101 have no head camera.

A native run stages the pack by hand:

```bash
python3 sim_isaac/engine/head_camera.py fetch \
  sim_isaac/engine/assets/head_camera
```

A new derivation is a new pack: pin its digest in `head_camera.py`, move the
chest camera of `sim_robot_core`'s `openarm_v2` entry to the rig it prints,
and restage the node with `peppy node add sim_isaac -sb --force`.

### Asset maintenance

`scripts/build_visuals.py` and `scripts/prepare_assets.py` are CPU-only maintenance
tools, not image-build steps. Preparation verifies every source stage, COLLADA
mesh and the license, converts only the v2 visuals, and checks USD composition,
material bindings, unchanged nonvisual specs and all attachment transforms.
Archives have sorted entries and fixed timestamps, ownership and permissions.
Preparation refuses to overwrite an existing output.

From the repository root, extract the source stages from the image digest
pinned in `scripts/visual_sources.json`, at the directory that file names in
it. Creating the source container does not run Isaac or require a GPU:

```bash
source_image=$(python3 -c 'import json; print(json.load(open("sim_isaac/scripts/visual_sources.json"))["robot"]["image"])')
source_directory=$(python3 -c 'import json; print(json.load(open("sim_isaac/scripts/visual_sources.json"))["robot"]["directory"])')
source_container=$(docker create --platform linux/amd64 "$source_image")
mkdir -p /tmp/openarm-isaac-source
docker cp "$source_container:$source_directory/." /tmp/openarm-isaac-source/
docker rm "$source_container"

uv run --no-project --python 3.11 --with usd-core==26.5 \
  --with-requirements sim_isaac/scripts/requirements-visuals.txt \
  python sim_isaac/scripts/prepare_assets.py \
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
bash sim_base_images/build_base_images.sh --isaac-only
```

The publisher stamps the node's `From:` tag. Commit the pin and tag together,
then restage the node with `peppy node add sim_isaac -sb --force`.

### The SO-101 stage

`so101_description` carries a kinematics-only URDF, so the SO-101's stage is
built from its upstream geometry: MuJoCo Menagerie's `robotstudio_so101`
(Apache-2.0) at the commit `sim_base_images/so101_model.lock.json` pins, the
same files the MuJoCo node bakes and Waldo's catalogue names, so the three
engines stand one robot. `scripts/build_so101.py` compiles the upstream MJCF
with MuJoCo and writes the stage from the compiled model: one link prim per
URDF link, with its mass, visual meshes and collision shapes, one revolute
joint per hinge with a position drive, the base fixed to the world by the
articulation's root joint, and the `gripper_frame_link` tool frame. Where
upstream and the description differ the description wins: each joint's
limits are the URDF's, and `tests/test_so101_stage.py` holds the stage to it,
tool frame included. CPU only, no Isaac Sim. From the repository root, with
the upstream staged by `sim_base_images/so101_model.py fetch <directory>`:

```bash
uv run --no-project --python 3.11 --with mujoco==3.10.0 --with usd-core==26.5 --with numpy \
  python sim_isaac/scripts/build_so101.py \
  --model /tmp/so101-upstream/so101 \
  --urdf <so101_description>/urdf/so101_kinematics.urdf \
  --lock sim_base_images/so101_model.lock.json \
  --output /tmp/so101-isaac-assets \
  --archive /tmp/so101-isaac-assets.tar.gz
SO101_KINEMATICS_URDF=<so101_description>/urdf/so101_kinematics.urdf \
SO101_UPSTREAM_MODEL=/tmp/so101-upstream/so101 \
  uv run --project sim_isaac/tests --locked --group dev pytest sim_isaac/tests/test_so101_stage.py
```

The same stage packs to the same bytes, so the archive is published under
its own SHA-256, as the OpenArm bundle is, at `so101/<version>/<sha256>.tar.gz`,
and pinned as `SO101_ASSETS_KEY` and `SO101_ASSETS_SHA256` in
`isaac_assets.env`; the image build then follows the OpenArm recipe above.
Inside the image the suite reads the baked stage from
`PEPPY_ROBOT_ASSETS_DIR` instead of building it.

## Tests

Run the GPU-free regression suites from the repository root:

```bash
uv run --project sim_isaac/tests --locked --group dev \
  pytest sim_isaac/tests
```

On Linux ARM64, PyPI has no `usd-core` distribution. The USD-specific tests
are skipped there; the installer, camera, startup and timing suites still run.
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

- Runtime task primitives may still need explicit collision, rigid-body and mass configuration for contact-rich manipulation.
- Converted OBJ assets may not preserve source materials or textures automatically.
- Runtime USD paths must be visible inside the Isaac Sim container.
- WebRTC requires the correct host IP to be supplied through `PEPPY_ISAAC_PUBLIC_IP`.
- Isaac Sim can require substantial RAM, swap, disk space and GPU memory.

## Feedback

Useful reports include:

- installation or image-build failures
- WebRTC connection problems
- GPU or memory issues
- articulation or joint-order issues of any model
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
