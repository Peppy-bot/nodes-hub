# openarm Nodes

[Peppy](https://github.com/Peppy-bot/peppy) nodes for the OpenArm bimanual robot (v1.0 and v2.0). The full stack lets you drive two 7-DOF arms and two grippers from your browser, against the real robot, Isaac Sim, or MuJoCo. The nodes and the UI stay the same; only the launcher changes.

| Component | What it does |
|---|---|
| [`openarm_robot_initializer`](./openarm_robot_initializer) | aggregates per-limb readiness into `is_ready` |
| [`openarm_arm`](./openarm_arm) | drives one arm side (7 joints) |
| [`openarm_gripper`](./openarm_gripper) | drives one gripper side (v1.0 prismatic or v2.0 pinch, by `hardware_version`) |
| [`openarm_arm_sim`](./openarm_arm_sim) | relays one arm side between the backbone and a sim engine |
| [`openarm_gripper_sim`](./openarm_gripper_sim) | relays one gripper side between the backbone and a sim engine |
| [`openarm_sim_mujoco`](./openarm_sim_mujoco) | MuJoCo engine: the physics behind the relays |
| [`openarm_sim_isaac`](./openarm_sim_isaac) | Isaac Sim engine: the physics behind the relays |
| [`openarm_backbone`](./openarm_backbone) | routes goals to the correct side |
| [`openarm_commander`](./openarm_commander) | browser control panel |
| [`openarm_ker`](./openarm_ker) | streams joint setpoints from a physical leader arm |
| [`openarm_isaac_webviewer`](./openarm_isaac_webviewer) | serves the Isaac Sim WebRTC browser viewer |
| [`openarm_scene_commander`](./openarm_scene_commander) | browser scene/object/physics control for Isaac Sim |

Sim support splits into engine-agnostic relays plus one node per engine: `openarm_arm_sim` and `openarm_gripper_sim` face the backbone exactly like the real nodes and lead the matching limb slot on the engine node (`openarm_sim_mujoco` or `openarm_sim_isaac`), which owns the physics and models v1.0 or v2.0 hardware via its `hardware_version` parameter. The launcher decides which nodes fill each slot, so the backbone and the UI never know which engine is underneath.

This guide takes you from a fresh machine to a moving arm. MuJoCo is the quickest way to see everything working. The Isaac browser stack (WebRTC viewer plus Scene Commander) is covered in its own section below.

## 1. Prerequisites

- Ubuntu 22.04 or 24.04
- [Peppy](https://peppy.bot) 0.16 or newer, installed with `curl -fsSL https://peppy.bot/install.sh | sh`
- Docker, running
- For Isaac only: an NVIDIA GPU with the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) configured

Clone this repo together with [contracts-hub](https://github.com/Peppy-bot/contracts-hub) (the contracts) and [launchers-hub](https://github.com/Peppy-bot/launchers-hub) (the stack launchers) so the paths below line up:

```text
ws/
├── contracts-hub/
├── launchers-hub/
└── openarm-nodes/
```

## 2. Start the daemon and register the repos

The daemon builds, runs, and connects every node. Registering the repos is what lets it resolve nodes and contracts by name. The launcher depends on this, so don't skip it on a fresh machine.

```sh
peppy service serve &

peppy repo add /path/to/ws/contracts-hub
peppy repo add /path/to/ws/openarm-nodes
peppy repo add /path/to/ws/nodes-hub
peppy repo add /path/to/ws/launchers-hub
peppy repo refresh
```

`peppy repo refresh` walks the registered repos and ends with a summary like `Repository refresh complete. N node(s), M contract(s) found.` You can double-check what got registered with `peppy repo list`.

To run a development branch instead of main, register the repositories by git URL with `--ref` (and `--top` to override an existing registration) so peppy does not shadow them with main:

```sh
peppy repo add \
  https://github.com/Peppy-bot/openarm-nodes.git \
  --ref feature/isaacsim-6.0.1-clean \
  --top

peppy repo add \
  https://github.com/Peppy-bot/contracts-hub.git \
  --ref modern-jaguar \
  --top

peppy repo add \
  https://github.com/Peppy-bot/launchers-hub.git \
  --ref feature/openarm-isaac-browser-stack \
  --top

peppy repo refresh
```

## 3. Build the nodes

Each `peppy node add <path> -sb` registers the node in the stack, generates its API code from the manifest and contracts, and builds its container. The first sim engine build also pulls its base image (about 1 GB for MuJoCo and 7.5 GB for Isaac), so it gets a much larger idle timeout than the rest; without it the daemon kills the build mid-download.

MuJoCo stack:

```sh
peppy node add /path/to/ws/openarm-nodes/openarm_sim_mujoco -sb --idle-timeout 18000
peppy node add /path/to/ws/openarm-nodes/openarm_arm_sim -sb --idle-timeout 1800
peppy node add /path/to/ws/openarm-nodes/openarm_gripper_sim -sb --idle-timeout 1800
peppy node add /path/to/ws/openarm-nodes/openarm_robot_initializer -sb --idle-timeout 1800
peppy node add /path/to/ws/openarm-nodes/openarm_backbone -sb --idle-timeout 1800
peppy node add /path/to/ws/openarm-nodes/openarm_commander -sb --idle-timeout 1800
```

For Isaac, swap the engine node; the relays, initializer, backbone, and commander are engine-agnostic and don't need rebuilding:

```sh
peppy node add /path/to/ws/openarm-nodes/openarm_sim_isaac -sb --idle-timeout 18000
```

The `sim_cameras` launcher option, on either engine, additionally needs the camera relay nodes, which live in the separate [nodes-hub](https://github.com/Peppy-bot/nodes-hub) repo because nothing in them is OpenArm-specific:

```sh
peppy node add /path/to/ws/nodes-hub/sim_rgb_camera -sb --idle-timeout 1800
peppy node add /path/to/ws/nodes-hub/sim_rgbd_camera -sb --idle-timeout 1800
```

Real robot:

```sh
peppy node add /path/to/ws/openarm-nodes/openarm_robot_initializer -sb --idle-timeout 1800
peppy node add /path/to/ws/openarm-nodes/openarm_arm -sb --idle-timeout 1800
peppy node add /path/to/ws/openarm-nodes/openarm_gripper -sb --idle-timeout 1800
```

Both hardware generations run the same nodes; the launcher's `hardware_version` argument selects which one each arm and gripper drives.

Upgrading a v2.0 rig from `openarm_gripper_v2`: stop and remove any gripper instance running that node before launching. Both nodes drive the same motor id on the same bus, but the old one holds a different instance lock, so nothing would stop the two from commanding one gripper at once.

After changing a node's code, rebuild it by re-running the same command with `--force` added.

Now verify everything built:

```sh
peppy stack list
```

Every node you added should show `Stage: Ready`. If one is stuck at an earlier stage, jump to Troubleshooting.

## 4. Launch the stack

The `--with=` selection names the engine, so pick the one matching the nodes built above:

```sh
# MuJoCo
peppy stack launch openarm_v2 --with=mujoco
# Isaac
peppy stack launch openarm_v2 --with=isaac_sim
```

The launcher starts the instances in dependency order (sim first, then arms and grippers, then backbone, then the UI) and wires them together. Once it prints `Launch complete`:

- open **http://localhost:8765** for the control panel, one slider per joint
- MuJoCo: open **http://localhost:8080** for the browser viewer
- Isaac: open **http://<ISAAC_HOST_IP>:8210** for the WebRTC browser viewer (see the [Isaac browser stack](#isaac-browser-stack--webrtc-viewer-and-scene-commander) section below)

Move a slider, press **Send**, and watch the arm follow in the viewer. The launchers themselves are documented in [launchers-hub/openarm](https://github.com/Peppy-bot/launchers-hub/tree/main/openarm). Check the stack's health any time:

```sh
peppy stack list
```

Every instance should be `running` and `healthy`. To stop everything, Ctrl-C the launch terminal, or stop instances individually:

```sh
peppy node stop commander_inst
```

---

# Isaac browser stack — WebRTC viewer and Scene Commander

The Isaac stack streams to a plain browser over WebRTC and adds two browser tools on top of the sim: the **Isaac viewer** (live video at `:8210`) and the **Scene Commander** (scene/object/physics control at `:8766`). This section covers the WebRTC configuration and what each tool does.

## Configure Isaac WebRTC

Find the LAN IP address of the machine running Isaac Sim:

```bash
hostname -I
```

For example:

```text
192.168.1.120
```

Configure the address Isaac should advertise to the browser (do this before launching the stack, then restart peppy so the daemon environment picks it up):

```bash
systemctl --user set-environment \
  APPTAINERENV_PEPPY_ISAAC_PUBLIC_IP=192.168.1.120 \
  APPTAINERENV_PEPPY_ISAAC_SIGNAL_PORT=49100 \
  APPTAINERENV_PEPPY_ISAAC_STREAM_PORT=47998

systemctl --user restart peppy.service
```

Replace `192.168.1.120` with the actual IP of the Isaac machine. Verify:

```bash
systemctl --user show-environment \
| grep APPTAINERENV_PEPPY_ISAAC
```

Expected:

```text
APPTAINERENV_PEPPY_ISAAC_PUBLIC_IP=192.168.1.120
APPTAINERENV_PEPPY_ISAAC_SIGNAL_PORT=49100
APPTAINERENV_PEPPY_ISAAC_STREAM_PORT=47998
```

Do not hard-code the IP address in the repository. The address may change when using DHCP or Wi-Fi.

## Open the Isaac browser viewer

Open:

```text
http://<ISAAC_HOST_IP>:8210
```

For example:

```text
http://192.168.1.120:8210
```

The browser viewer displays the live Isaac Sim stream. WebRTC uses:

```text
8210/TCP     Web viewer
49100/TCP    WebRTC signalling
47998/UDP    WebRTC media stream
```

If the viewer opens but the video does not connect, first verify that the configured public IP is reachable from the browser machine.

## Open Scene Commander

Open:

```text
http://<ISAAC_HOST_IP>:8766
```

When working directly on the Isaac machine:

```text
http://127.0.0.1:8766
```

Scene Commander is used to construct and modify the Isaac environment while the simulator is running.

### Scene controls

The **Scene** section allows you to:

- Select an Isaac environment
- Set the scene scale
- Load the selected scene
- Clear the runtime scene

Scene assets are provided by the Isaac-side asset catalogue, so the exact catalogue can evolve without changing the web application.

Examples include warehouse and grid environments.

Runtime scenes are created under:

```text
/World/RuntimeScene
```

## Search and spawn objects

Scene Commander provides an asset search field.

Examples of searchable assets include:

```text
table
can
cracker
drill
block
bottle
```

For a selected object, configure:

```text
X position
Y position
Z position
Scale
Physics
Mass
```

Then press:

```text
Spawn Object
```

Isaac assigns each spawned object a unique ID similar to:

```text
obj_626b16769f97
```

Runtime objects are created under:

```text
/World/RuntimeObjects/<object_id>
```

## Object physics modes

Scene Commander supports three physics modes.

### None

```text
Physics = none
```

Use this for assets that should only be placed visually and should not participate in rigid-body physics.

This is useful for:

- initial layout
- visual props
- complex assets during placement testing

### Static

```text
Physics = static
```

Use this for fixed objects such as:

- tables
- shelves
- walls
- fixtures
- work surfaces

Static objects participate in collision but do not move dynamically.

### Dynamic

```text
Physics = dynamic
```

Use this for objects that should:

- fall under gravity
- collide
- be pushed
- be grasped
- be lifted
- receive external forces

Mass is specified in kilograms.

Example:

```text
Physics = dynamic
Mass = 0.5 kg
```

## Runtime Object controls

Every spawned object appears in the **Runtime Objects** section.

The object entry shows information such as:

```text
object_id
asset_id
physics mode
mass
position
```

From this section you can:

```text
Move
Remove
Apply Force        dynamic objects only
```

### Move an object

Change its:

```text
X
Y
Z
```

coordinates and press:

```text
Move
```

This changes the object's runtime position without restarting Isaac Sim.

### Remove an object

Press:

```text
Remove
```

to delete that runtime object.

## Apply force to a dynamic object

Objects spawned with:

```text
Physics = dynamic
```

provide additional force controls.

The Runtime Object card contains:

```text
Force magnitude (N)
Duration (s)

+X    -X
+Y    -Y
+Z    -Z
```

For example:

```text
Mass       = 0.5 kg
Force      = 1.0 N
Duration   = 0.5 s
Direction  = +Y
```

Press:

```text
+Y
```

to apply:

```text
[0.0, 1.0, 0.0] N
```

in the Isaac world frame.

The force is automatically disabled after the selected duration.

For lightweight objects, start with small forces.

For example:

```text
0.1 kg object:
0.2 – 0.5 N

0.5 kg object:
0.5 – 2 N
```

Avoid starting with very large forces such as `20 N` on lightweight objects.

Force control is currently world-frame XYZ force control.

Torque and mouse-based viewport force manipulators are not currently exposed.

## Move the OpenArm robot root

Scene Commander also provides robot-root positioning.

Set:

```text
X
Y
Z
```

and use:

```text
Move Robot
```

to reposition the OpenArm base in the Isaac world.

This is useful when configuring new workcells or positioning the robot relative to tables, shelves and manipulation objects.

## Open OpenArm Commander

Open:

```text
http://<ISAAC_HOST_IP>:8765
```

or locally:

```text
http://127.0.0.1:8765
```

OpenArm Commander controls the robot itself.

## Typical workflow

A normal interactive session is:

```text
1. Launch the OpenArm Isaac stack

2. Open Isaac Viewer
   http://<IP>:8210

3. Open Scene Commander
   http://<IP>:8766

4. Load an environment

5. Search for a table or fixture

6. Spawn it as Static

7. Search for manipulation objects

8. Spawn them as Dynamic

9. Set appropriate mass

10. Use Runtime Objects to adjust positions

11. Apply small external forces if required

12. Open OpenArm Commander
    http://<IP>:8765

13. Move the arms and grippers

14. Observe everything live through the Isaac Viewer
```

## Ports summary

| Service | Port | Purpose |
|---|---:|---|
| Isaac Viewer | 8210/TCP | Browser interface |
| OpenArm Commander | 8765/TCP | Arm and gripper control |
| Scene Commander | 8766/TCP | Scene/object/physics control |
| WebRTC Signalling | 49100/TCP | Isaac WebRTC signalling |
| WebRTC Stream | 47998/UDP | Isaac WebRTC media |

## Current architecture

```text
                       Browser
                          |
         +----------------+----------------+
         |                |                |
         v                v                v
   Isaac Viewer     Scene Commander   OpenArm Commander
      :8210              :8766             :8765
         |                |                |
         |                v                v
         |        openarm_sim_isaac   openarm_backbone
         |                ^                |
         |                |                v
         |                |         simulation relays
         |                |                |
         +----------------+----------------+
                          |
                          v
                      Isaac Sim
```

---

## Troubleshooting

**`repo-node 'X:v1' not found in nodes.json5` when launching**
The repo that provides X was never registered with the daemon. Run the `peppy repo add` lines from step 2 followed by `peppy repo refresh`, then launch again.

**The sim engine build dies partway through**
The base image download outlived the daemon's idle timeout. Re-run the add with `--idle-timeout 18000`. The Isaac image is large and the first build genuinely takes a while; later builds reuse the cached image and finish quickly.

**A node won't reach `Stage: Ready`**
Rebuild it and read the build log peppy prints on failure:
```sh
peppy node add /path/to/ws/openarm-nodes/<node> -sb --force --idle-timeout 1800
```

**The stack launches but the arms don't respond**
The sim keeps loading after `Launch complete`, and Isaac can take a minute. Watch its log until the world is up:

```sh
peppy node info openarm_sim_mujoco:v1   # or openarm_sim_isaac:v1
```

**A move finishes with "reached (target clamped to joint limits)"**
Not an error. The requested angle was beyond that joint's physical range, so the arm went as far as the model allows and reported success there.

**The Isaac stream is a black screen**
Stop the stack, clear the shader cache with `rm -rf ~/.cache/isaac-sim`, and launch again.

**Port 8765 or 8080 is already in use**
An older instance is still running. Find it with `peppy stack list` and stop it with `peppy node stop <instance_id>`.

## Adding an item to this repository

This repository publishes what `peppy_repository.json5` says it publishes, and nothing else. An item
that is not listed there is invisible to peppy, so after adding, moving, or renaming a node, run:

```sh
peppy repo index .
```

Commit the updated `peppy_repository.json5` alongside your change. CI runs `peppy repo index --check`
on every pull request and fails if the index has drifted from the repository, naming the file and the
identity involved.

Generation refuses, naming both files, if your change claims a `name:tag` another one already
publishes. Rename yours: within one repository, a `name:tag` is claimed by exactly one file.
