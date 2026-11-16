# rerun_viewer

`rerun_viewer` draws, live in one 3D view in a browser, what a robot stack already has on
the wire. It serves the [Rerun](https://rerun.io) web viewer (Rerun 0.41.0) and draws:

- the color and depth images of one camera;
- the depth samples of that camera as points in the robot frame;
- the motion planner's voxel map: the occupied cubes that its plans keep their clearance
  from, as the planner publishes them;
- each arm's grasp point with its axes and a trail of its past positions;
- each link of the robot as its collision shapes, where the joints put it;
- each joint's position and each gripper's opening, as plots;
- the planning scene of `motion_planning:v1`: each object as its shapes, each attached
  object at the grasp point of the arm that carries it, a see-through shell of the
  scene's clearance around each shape, and the pairs allowed to touch;
- a log of each change of the planning scene between two reads ("added brain/lime_1",
  "attached brain/lime_1 to openarm_right_tcp").

It only listens. It never moves the robot and never changes the planning scene.

It draws only the contracts that its `peppy.json5` lists: `limb_state:v1`,
`robot_identity:v1`, `rgbd_camera:v1`, `camera_geometry:v1`, `camera_mounts:v1` and
`motion_planning:v1`.

## The cubes are the planner's map

The cubes under `world/planner/map` are the motion planner's voxel map, as the planner
publishes it on `occupancy_map` of `motion_planning:v1`, by default once a second
(`occupancy_map_rate_hz` of motion_planner). They are the cubes that its plans, Cartesian
paths and checks keep their clearance from: the robot and the scene's objects are cut out
of them, and they hold the cubes that the camera cannot see now. Each map is drawn at the
time the planner read it. The text entity `planner/map` says how many cubes the map holds
or, when it holds no depth frame, why (before the first frame, and after
`clear_occupancy_map`). The points under `world/camera/points` are the camera's depth now.
The text entity `notes/cubes` says this too.

## What it needs

| Link | Contract | Cardinality | Without it |
|---|---|---|---|
| `limb_state` | `limb_state:v1` | one | the node does not start |
| `identity` | `robot_identity:v1` | zero_or_one | no robot links |
| `camera` | `rgbd_camera:v1` | zero_or_one | no images, no points |
| `geometry` | `camera_geometry:v1` | zero_or_one | no points |
| `camera_mounts` | `camera_mounts:v1` | zero_or_one | no points |
| `motion_planning` | `motion_planning:v1` | zero_or_one | no scene, no voxel map; each joint is named by its index |

The viewer asks the camera's geometry and pose when a depth frame first needs them, and
asks again after a failed ask, so a camera that answers late is drawn once it answers. It
asks the geometry again when the frames change their size or their alignment. For a
camera that an arm carries, it asks the pose again for each cloud: the points take the
pose of that ask, which can be some tens of milliseconds after the frame, so the points
of a camera on a moving arm stand that much off. It reads the planning scene at
`scene_poll_hz` and draws it each time it differs from the read before, at its time on
the node's clock.

With the `motion_planning` link bound, the joints are plotted once the planner has named
them; without it, each joint is named by its index (`joint_0`, `joint_1`, ...).

An attached object is drawn at the grasp point of the arm whose `grasp_link`
(`describe_arms`) carries it. An object that another link carries is logged and not drawn.

## The robot's links

The viewer asks `get_description` of `robot_identity` once, again until it answers, and
reads the URDF, the SRDF and the collision meshes (binary STL) with `robot_description`.
It draws each link that has collision shapes, under `world/robot/<link>`: a box, a
cylinder, a sphere, a mesh, and a capsule as its cylinder and two spheres. These are the
shapes that the motion planner reads from the same description, not the finer visual
meshes that RViz shows, which `robot_identity` does not give. The planner checks convex
pieces that it fits around these shapes, a little larger than them, and keeps its
clearance from those pieces; the viewer does not draw the pieces.

Each link takes the color of its part: mid grey for the body and the base of each arm,
dark and light grey in turn for the segments of each arm from the first, and orange for
the fingers. Each mesh, of a link or of the planning scene, goes to the viewer with a
normal for each vertex, so that the viewer's light shades it.

For each `limb_states` snapshot, the viewer places each link by forward kinematics over
the URDF: each arm's joints, in the order of the arm's chain in the SRDF, at the
snapshot's positions, and each finger joint in proportion to its gripper's opening,
between the gripper's `closed` and `open` states. A joint that moves and that no arm and
no gripper drives stands at 0. A snapshot that lacks an arm or a gripper of the
description, or gives an arm another number of joints, is logged once, and the links stay
where the last snapshot put them. A description that does not read is logged, and the
viewer draws no links.

The shapes go to the viewer once, for the whole recording; each snapshot sends only the
pose of each link. Before the first snapshot, the shapes stand at the robot frame's origin.

## Attach it to a running stack

No launcher deploys the viewer. Start it beside a running stack and link it to the
instances of one robot copy (`peppy stack list` names them). For the copy `alpha` of a
simulated OpenArm:

```bash
peppy node add ./rerun_viewer -sb
peppy node run rerun_viewer:v1 --instance-id alpha_viewer --clock simulation \
  --link limb_state@alpha_backbone_inst --link identity@alpha_init_inst \
  --link camera@alpha_chest --link geometry@alpha_chest \
  --link camera_mounts@alpha_backbone_inst --link motion_planning@alpha_planner_inst \
  bind_ip=<address>
```

`--clock simulation` puts the viewer on the simulation's clock, as the other instances of
a simulated stack are. On a physical robot, leave `--clock` out. To leave a link out, give
`--vacant-link <link>=<why>` in place of its `--link`.

Then open the page from the browser, with the address the viewer serves on:

```
http://<address>:9090/?url=rerun%2Bhttp%3A%2F%2F<address>%3A9876%2Fproxy
```

The node logs this address at its start. The page reads its data from the second port
(`grpc_port`); the `url` part of the address tells it where.

To replay a saved recording, run the Rerun viewer of the same version on any machine:

```bash
uvx --from rerun-sdk==0.41.0 rerun recording.rrd
```

## Arguments

| Argument | Default | What it does |
|---|---|---|
| `bind_ip` | `127.0.0.1` | the address the viewer serves on: this machine only by default; give the Tailscale address to open it from another machine |
| `web_port` | 9090 | the port of the web page |
| `grpc_port` | 9876 | the port the page reads the data from |
| `memory_limit` | `1GB` | how much data the viewer keeps for a page that opens late, as `1GB` or `25%`; past it, the oldest data goes first |
| `recording_path` | empty | a file to save the whole run to (`.rrd`); empty saves none |
| `image_rate_hz` | 2.0 | the most color images, and the most depth images, a second it draws |
| `cloud_rate_hz` | 2.0 | the most depth frames a second it turns into points |
| `cloud_stride` | 4 | the points come from every n-th pixel across and down |
| `scene_poll_hz` | 2.0 | how many times a second it reads the planning scene |
| `trail_length` | 100 | how many past grasp-point positions each trail holds |
| `camera_name` | `chest` | the camera of `camera_mounts` that the `camera` link is |

The color images go to the page as JPEG. The rates bound the work of the node: a color
frame of the simulated OpenArm chest camera is 1280x720 (2.8 MB raw), and the camera sends
up to 15 a second.

The cost at the defaults, measured on a machine of 30 cores against the simulated OpenArm
`alpha`, with limb_states at 20 Hz and the planner's map at 1 Hz, the viewer running for
16 minutes:

| Chest camera | CPU | Resident memory |
|---|---|---|
| idle, 1 frame a second | 7.2 % of one core | 1016 MB |
| held, 15 frames a second | 12.4 % of one core | 1131 MB |

The memory is mostly the history that the data server keeps for a page that opens late. It
grows to `memory_limit` within minutes, since the raw depth image alone is about 3.7 MB a
second at 2 images a second, and stays there: past the limit, the oldest data goes first.

The data server answers the page served from `bind_ip` and `web_port` only
(`http://<bind_ip>:<web_port>`). With `bind_ip=0.0.0.0` it answers the page under any
address on `web_port`.

## The code

Each part of the node knows only what its comment says:

- `src/wire/`: peppy. The subscriptions and the service calls, and the parse of each
  message into a value of the model, once, at the edge.
- `src/model/`: pure Rust. The values that the viewer draws and their arithmetic: depth to
  points, a scene to shapes and shells, the change log, the planner's voxel map, and the
  robot's body (its description and STL meshes to shapes, joints to link poses).
- `src/draw/`: the canvas only. How each value is drawn, and the paths of the entities.
- `src/canvas.rs`: Rerun. The `Canvas` trait that the drawers use, and `RerunCanvas`, the
  only code that calls Rerun. The tests draw on `FakeCanvas` (`src/testing.rs`).
- `src/config.rs`: the node's parameters, parsed once at start.
- `src/node.rs`: the start: one follower for each link that is bound.

## Build and test

```bash
peppy node add ./rerun_viewer -sb   # build the image
cd rerun_viewer && peppy node sync && cargo test --locked
```

The unit tests check the parse of each message, the arithmetic of the model with numbers
that a reader can check by hand, and the canvas calls of each drawer. The tests in
`tests/` run the node over peppy's test harness: mock providers on a real router, a fake
canvas, and waits that end when the expected entity is drawn.
