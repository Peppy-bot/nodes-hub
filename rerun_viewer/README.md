# rerun_viewer

`rerun_viewer` draws, live in one 3D view in a browser, what a robot stack already has on
the wire. It serves the [Rerun](https://rerun.io) web viewer (Rerun 0.41.0) and draws:

- the color image of one camera, and its depth image when `depth_image` is set;
- the depth samples of that camera as points in the robot frame, where the robot's tree
  of frames places the frame each depth frame names, at the time it was taken;
- the motion planner's voxel map: the occupied cubes that its plans keep their clearance
  from, as the planner publishes them;
- each arm's grasp point with its axes and a trail of its past positions;
- each link of the robot as its collision shapes, where the joints put it;
- each joint's position and each gripper's opening, as plots: one plot for each arm with a
  line for each joint, and one plot with a line for each gripper;
- the planning scene of `motion_planning:v1`: each object as its shapes, with its id as a
  label ("(carried)" after the id of an object that an arm carries), each attached
  object at the grasp point of the arm that carries it, a see-through shell of the
  scene's clearance around each shape, and the pairs allowed to touch;
- a log of each change of the planning scene between two reads ("added brain/lime_1",
  "attached brain/lime_1 to openarm_right_tcp").

It only listens. It never moves the robot and never changes the planning scene.

It draws only the contracts that its `peppy.json5` lists: `limb_state:v1`,
`robot_identity:v1`, `rgbd_camera:v1`, `camera_geometry:v1`, `sensor_mounts:v1`,
`transforms:v1` and `motion_planning:v1`.

## Start it beside a running stack

No launcher deploys the viewer. Start it beside a running stack and link it to the
instances of one robot copy (`peppy stack list` names them). For the copy `alpha` of a
simulated OpenArm:

```bash
peppy node add ./rerun_viewer -sb
peppy node run rerun_viewer:v1 --instance-id alpha_viewer --clock simulation \
  --link limb_state@alpha_backbone_inst --link identity@alpha_init_inst \
  --link camera@alpha_chest --link geometry@alpha_chest \
  --link sensor_mounts@alpha_init_inst \
  --link transforms@alpha_init_inst --link transforms@alpha_backbone_inst \
  --link motion_planning@alpha_planner_inst \
  bind_ip=<address> cache_window_s=<seconds> lookup_timeout_s=<seconds>
```

`<address>` is the address of this machine that the browser reaches, such as its
Tailscale address. Without `bind_ip`, the viewer serves this machine only. The
`transforms` link takes every publisher of the robot's tree: the initializer, which
publishes the fixed joints, and the backbone, which publishes the joints it measures.
Give `cache_window_s` and `lookup_timeout_s` the values that launchers-hub
`<robot>/fragments/motion_planner.json5` gives the planner: the viewer places depth
frames as the planner does.
`--clock simulation` puts the viewer on the simulation's clock, as the other instances of
a simulated stack are. On a physical robot, leave `--clock` out. To leave a link out, give
`--vacant-link <link>=<why>` in place of its `--link`.

Then open the page from the browser, with the address the viewer serves on:

```text
http://<address>:9090/?url=rerun%2Bhttp%3A%2F%2F<address>%3A9876%2Fproxy
```

The node logs this address at its start. The page reads its data from the second port
(`grpc_port`); the `url` part of the address tells it where.

To replay a saved recording (`recording_path`), run the Rerun viewer of the same version
on any machine:

```bash
uvx --from rerun-sdk==0.41.0 rerun recording.rrd
```

## The page

The viewer lays the page out (a Rerun blueprint, kept by the data server for each page):

- left, the 3D view `Robot`: the robot frame, at the entity `world`, with the links,
  the arms' grasp points and trails, the camera's points, the planner's cubes and the
  planning scene;
- right, from the top: the camera's image (and the depth image in a second tab with
  `depth_image`); the plots `Joints`, one line for each joint of each arm, and
  `Grippers`, one line for each gripper; and the texts `Scene changes`, `Planner map`,
  `Allowed pairs` and `Notes`.

The side panels and the time panel start collapsed; each opens with its button at the
top right. The page shows the newest data while its time cursor follows the stream.

In the 3D view, each kind of thing has its own look:

| Look | What it is |
| --- | --- |
| grey shapes, with orange fingers | the robot's links |
| green shape, with its id | an object of the planning scene |
| magenta shape, with its id and "(carried)" | an object that an arm carries, at that arm's grasp point |
| red wireframe | the scene's clearance around a shape of the planning scene |
| see-through blue cubes | the planner's voxel map |
| light blue points | the camera's depth samples |
| three axes and a yellow line | an arm's grasp point and the trail of its past positions |

To hide a kind, open the left panel and click the eye of its entity under the view
`Robot`: `world/planner/map` for the cubes, `world/camera/points` for the points,
`world/robot` for the links, `world/scene` for the objects that stand in the scene and
their shells.

## Arguments

| Argument | Default | What it does |
| --- | --- | --- |
| `bind_ip` | `127.0.0.1` | the address the viewer serves on: this machine only by default; give the Tailscale address to open it from another machine |
| `web_port` | 9090 | the port of the web page |
| `grpc_port` | 9876 | the port the page reads the data from |
| `memory_limit` | `8MB` | how much data the viewer keeps for a page that opens late, newest first, as `8MB` or `1%`; past it, the oldest data goes first, the robot's link shapes (about 2.5 MB for an OpenArm) last |
| `page_stall_timeout_s` | 15.0 | how long a page may take no data, while data waits for it, before its connection is closed; 0 keeps every page |
| `recording_path` | empty | a file to save the run to (`.rrd`), as the data server takes it; empty saves none |
| `arms_rate_hz` | 10.0 | the most snapshots of the arms a second it draws |
| `image_rate_hz` | 2.0 | the most color images, and the most depth images, a second it draws |
| `image_max_width` | 640 | the widest color image it draws, in pixels; a wider raw frame is shrunk by a whole factor, a JPEG frame goes as it came |
| `depth_image` | false | whether it draws the depth image; the points show the depth without it |
| `cloud_rate_hz` | 2.0 | the most depth frames a second it turns into points |
| `cloud_stride` | 4 | the points come from every n-th pixel across and down |
| `scene_poll_hz` | 2.0 | how many times a second it reads the planning scene |
| `trail_length` | 100 | how many past grasp-point positions each trail holds |
| `cache_window_s` | none | how long each moving transform's past stamps are kept, counted back from its newest; finite and above 0 |
| `lookup_timeout_s` | none | how long a depth frame waits for the tree to place the frame it names at its time; finite and 0 or above, 0 for no wait |

The color images go to the page as JPEG. The rates bound the work of the node and the
data on the link: a color frame of the simulated OpenArm chest camera is 1280x720 (2.8 MB
raw), drawn at 640x360 (about 34 kB as JPEG), and the camera sends up to 15 a second. A
raw depth image of that camera is about 1.8 MB, which is why the viewer draws it only
with `depth_image`.

The data server answers the page served from `bind_ip` and `web_port` only
(`http://<bind_ip>:<web_port>`). With `bind_ip=0.0.0.0` it answers the page under any
address on `web_port`.

## What it needs

| Link | Contract | Cardinality | Without it |
| --- | --- | --- | --- |
| `limb_state` | `limb_state:v1` | one | the node does not start |
| `identity` | `robot_identity:v1` | zero_or_one | no robot links |
| `camera` | `rgbd_camera:v1` | zero_or_one | no images, no points |
| `geometry` | `camera_geometry:v1` | zero_or_one | no points |
| `sensor_mounts` | `sensor_mounts:v1` | zero_or_one | no points; the arms, the scene and the voxel map are drawn in the frame each names |
| `transforms` | `transforms:v1` | zero_or_more | no points |
| `motion_planning` | `motion_planning:v1` | zero_or_one | no scene, no voxel map; each joint is named by its index |

The viewer asks the camera's geometry when a depth frame first needs it, and again after a
failed ask. It asks the robot frame, which `sensor_mounts`' `describe_sensors` names, in the
background when a depth frame, a snapshot of the arms, a voxel map or a read of the
planning scene first needs it, once for all of them, and again when one needs it after a
failed ask. Until `sensor_mounts` answers, the viewer drops each of them, a depth frame
drawing its image and no points, so a camera or a `sensor_mounts` that answers late is
drawn from the next value after its answer. With the `sensor_mounts` link bound, a snapshot of
`limb_states`, a map of `occupancy_map` or a read of `get_planning_scene` that names
another frame than the robot frame is not drawn, with both frames in the log; without the
link, the viewer draws each in the frame it names, which its contract gives as the robot
frame. It asks the geometry
again when the frames change their size, their alignment or the frame they name. A depth
frame names the camera's frame in its header, and `get_depth_to_color_extrinsics` places
the depth stream in that frame; a frame that names another draws no points, with both
frames in the log. It keeps the robot's tree from the transforms of every publisher of the
`transforms` link, and places the frame each depth frame names in the robot frame at the
time the depth frame was taken, so the points of a camera on a moving arm stand where the
arm held it. A depth frame waits up to `lookup_timeout_s` of the node's clock, the
simulation's under `--clock simulation`, for the transforms of its time; a newer depth
frame takes the place of one that waits, and one that the tree does not place draws no
points, with the lookup's failure kind (`unknown_frame` for a frame no transform of the
tree names, `no_chain`, `extrapolation` or `broken_tree`) and its cause in the log. The
moving transforms are dropped when the node's clock goes back. A simulation clock that
closes ends the node with an error, for its restart policy to start it again. It reads the
planning scene at `scene_poll_hz` and draws it at each read, at its time on the node's
clock, and logs the changes from the read before.

The arms are drawn at most `arms_rate_hz` times a second: the grasp points, the links and
the plots. Each snapshot still adds its point to the trails. With the `motion_planning`
link bound, the joints are plotted once the planner has named them; without it, each
joint is named by its index (`joint_0`, `joint_1`, ...).

An attached object is drawn at the grasp point of the arm whose `grasp_link`
(`describe_arms`) carries it. An object that another link carries is logged and not drawn.

## Live: the stack as it is now

The viewer shows the stack as it is now, as RViz does, not a history:

- The node keeps only the newest draw of each entity until Rerun's data server takes it
  (`src/live.rs`). When the server takes the draws slower than they come, a newer draw of
  an entity replaces the one that waits, and the node logs once that it keeps up by
  dropping draws. The lines of `scene/changes` are events: each waits in order, at most
  256.
- The node sends only what changes the view. A draw that is the same as the last one sent
  for its entity is left out: a pose within 0.1 mm, a value of a plot within 1e-4, the
  same points, the same image. Each entity is sent again once its data is 2 s newer, so a
  page that opens late has everything within 2 s. A stack at rest sends about 18 messages
  and 80 kB a second, and one with an arm in motion about 90 messages and 190 kB a
  second (measured on the simulated OpenArm): a link of 1.5 MB a second carries it with
  room, and the page stays within half a second of the stack.
- The data server keeps `memory_limit` of data for a page that opens late, 8 MB by
  default: the robot's link shapes and some seconds of the rest. It sends that page the
  newest data first.
- The viewer draws the planning scene at each read (`scene_poll_hz`), so a page that opens
  late shows the scene as it is within one read.

Rerun's data server sends each message to each page and waits for the slowest: a page
that stops reading, as a browser tab left in the background does after some time, would
hold the data back from every page. So each page's connection passes through a relay of
the node (`src/relay.rs`), which closes the connection of a page that takes no data for
`page_stall_timeout_s` (15 s by default) while data waits for it. The other pages go on.
The page that was closed shows what it last read: load it again to follow the stack.
With the stack at rest, a page that stops reading is closed after about 2 minutes, since
the connection's buffers take that long to fill. `page_stall_timeout_s=0` keeps every
page, however long it stalls.

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

## The cost

The cost at the defaults, measured on a machine of 30 cores against the simulated OpenArm,
with limb_states at 20 Hz and the planner's map at 1 Hz, the viewer running for 10
minutes:

| Chest camera | CPU | Resident memory |
| --- | --- | --- |
| idle, 1 frame a second | 3.7 % of one core | 78 MB |
| held, 15 frames a second | 8.2 % of one core | 82 MB |

The memory stays there: the data server keeps at most `memory_limit` of data, and the node
keeps the newest draw of each entity and the last one it sent.

## The code

Each part of the node knows only what its comment says:

- `src/wire/`: peppy. The subscriptions and the service calls, and the parse of each
  message into a value of the model, once, at the edge.
- `src/model/`: pure Rust. The values that the viewer draws and their arithmetic: depth to
  points, a scene to shapes and shells, the change log, the planner's voxel map, and the
  robot's body (its description and STL meshes to shapes, joints to link poses).
- `src/draw/`: the canvas only. How each value is drawn, and the paths of the entities.
- `src/canvas.rs`: Rerun. The `Canvas` trait that the drawers use, the page's layout, and
  `RerunCanvas`, the only code that calls Rerun. The tests draw on `FakeCanvas`
  (`src/testing.rs`).
- `src/drawn.rs`: a call to a canvas as a value (`Drawn`), and when two draws are the same
  for the eye.
- `src/live.rs`: the canvas the node draws on. It keeps the newest draw of each entity and
  hands on what changes the view to `RerunCanvas`, on a thread of its own, so only that
  thread waits on Rerun.
- `src/relay.rs`: the relay in front of Rerun's data server, which closes the connection
  of a page that stopped reading.
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
