# SO-101 Nodes

Peppy nodes for the SO-101 arm (TheRobotStudio / HuggingFace lerobot
ecosystem): six Feetech STS3215 servos on one half-duplex serial bus, five
revolute joints plus a gripper. All nodes are Python and reuse
[lerobot](https://github.com/huggingface/lerobot) as a library: the Feetech
bus and calibration handling (`SOFollower` / `SOLeader`) in the hardware
nodes, and the placo-based `RobotKinematics` behind `so101_description`'s
kinematics module.

## Nodes

| Node | Role |
|---|---|
| `so101_follower` | The real arm. Pure follower: `joint_link` + `gripper_link` follower slots on one node (one process must own the serial port), `component_ready`, `motor_health`, `alert`. No motion logic. |
| `so101_leader` | The passive leader arm as a teleop device. Open-loop `joint_link` + `gripper_link` leader; staleness is the deadman (the hardware has no engage button). |
| [`robot_initializer`](../robot_initializer) | The node every robot uses, run with `model: "so101"`: answers who the robot is on `get_identity` and whether it is ready on `is_ready`, from the follower's readiness on hardware and from the simulation standing it in a simulation, which it joins as the `so101` model. |
| `so101_backbone` | The motion authority in between. Follower role toward whatever leads (joint, pose, or gripper streams), leader role toward the follower; exposes the `limb_motion` and `postures` move actions, the `limb_motion` services (`stop` ends every planned move in flight, whoever started it, each goal then ending cancelled with the reason `stopped: <reason>`; `check_arm_move` runs a `move_arm` goal's solve from the same anchor and moves nothing, and gives an error reply with the kind of the refusal to a check that `move_arm` would refuse), the `limb_state` readout and the `workspace` services (`describe_workspace` and `check_positions`: where the arm reaches, from the design alone; see [Workspace](#workspace)). Every pose is in the URDF's `base_link` frame, the robot frame the contracts define: `+x` the way the robot faces, `+z` up. A joints-led stream is clamped into the joint limits and rate-stepped per joint at the joint velocity limits, under the end-effector-speed governor; a pose-led stream is reach-clipped and solved first, then treated the same. Joint moves, postures and Cartesian moves are timed within the per-joint velocity and acceleration caps. A joint path (`execute_joint_path`) names the arm, `arm`, once in its `arms`, rounds its corners as `limb_motion` states, begins at the joints the follower measures as the move starts, clamped into the joint limits, where the arm holds while the path is planned, and is timed within the per-joint velocity and acceleration caps, its speed integrated over a grid of at least 1000 cells along the path, slowed evenly to a longer `duration_s`. A goal fails with `precondition_failed`, naming the joint, while a measured joint stands farther than the `start_tolerance_rad` parameter from its `expected_start_joint_positions`, when it names them, or, once the path is planned, while a measured joint, clamped into the joint limits, stands farther than `start_tolerance_rad` from where the arm holds, the arm having moved while the path was planned; and it fails with `rejected`, giving the time, when those caps need longer than 60 s for the path. A goal refused for its expected start moves nothing; a path that fails once the arm holds, before it plays, leaves the arm held at its measured joints, moved inside its joint limits where it stood past one, and runs none of the path; once the path plays, a failure holds the arm where it stands. Each arm move completes once the measured arm stands within the settle tolerances its goal names, at or after the planned end, and fails with `timeout` at its settle timeout; a posture once each joint stands within 0.1 rad of the posture, with a settle timeout of 2 s, so an arm that an object stops short of the posture fails it. `move_to_ready` claims the arm and sends the gripper no command. `move_to_home` claims the gripper and the arm, and closes the gripper before the arm moves: the close ends as a gripper move ends, with a settle timeout of 2 s, so jaws that an item stops close on it and the arm goes home with it; the travel of the close has 8 s at the gripper rate cap, and a goal whose close would travel longer is refused with `rejected`; a gripper with no fresh state gets no close, and the goal moves the arm alone; the arm move is planned from where the arm stands once the close completed, and `duration_s` is its time alone; a cancel or a `stop` during the close ends the goal cancelled, a close that fails ends the goal with its failure, and in both cases the arm does not move. A gripper move ramps at the gripper rate cap and, from the planned end of that travel on, completes once the measured gripper stands still, on its target or short of it where an object holds the jaws, or at its settle timeout wherever the jaws stand. Everything gates on fresh follower state: a goal is refused with `resource_unavailable` without it, and a move fails with `resource_unavailable` when it goes stale. A goal on a limb a leader is streaming is refused with `leader_streaming` until the stream has been silent for the deadman timeout, and a goal on a limb that runs another goal with `precondition_failed`; an admitted goal owns its limb until it ends. |

```text
so101_leader ──joint+gripper──▶ so101_backbone ──joint+gripper──▶ so101_follower
xr_commander ──pose+gripper──▶ (same backbone, upstream_mode="pose")
lerobot_recorder observes the follower pairings and the backbone's leader slots
```

The backbone's two downstream links carry its limbs' names, `arm` and
`gripper`, the names it answers to in `limb_motion` and `limb_state` and the
ones its `get_limb_names` service reports. In a simulation there is no
follower node: the engine plays the follower role, the same backbone leads
`simulation_inst/arms` and `simulation_inst/grippers` through those links, and
the link a pair comes from is how the engine knows which limb it drives.
[`sim_mujoco`](../sim_mujoco), [`sim_isaac`](../sim_isaac) and Waldo all stand
the `so101` model, with its `wrist` camera. The backbone does not wait on
`robot_ready`: it holds still while its limbs report nothing, which is what a
robot not yet admitted looks like.

There is no gravity or friction compensation anywhere in this family, by
design rather than omission: the STS3215 has no torque or current control
mode, the follower tracks positions with its in-servo PID, and the leader is
fully passive (light feel comes from its gearing). The follower and the
backbone both reject any `joint_setpoints` carrying a non-empty `efforts`
vector.

## Terminology

"Leader" and "follower" are overloaded between lerobot and peppy, so this
family uses them precisely:

- Unqualified, they name **pairing roles**: on any `joint_link` /
  `gripper_link` / `pose_link` pairing, the leader role emits setpoints and
  the follower role executes them and reports state. The backbone plays the
  follower role upstream and the leader role downstream while being neither
  robot.
- **"SO-101 leader arm"** and **"follower arm"** are lerobot's hardware
  product names. The two hardware nodes are named after those products, and
  each happens to play the matching pairing role, which is the only reason
  the names align.
- The node leading the backbone, whichever it is, is the **commander**
  (openarm vocabulary): `so101_leader`, the stack's headset (`xr_commander`),
  or a future policy runner. The leader arm fills the copy's
  `commander_inst`; the headset is the stack's `headset_inst`.

## Calibration (once per arm, on the host)

```sh
pipx install "lerobot[feetech]"   # or any venv
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/so101_leader --teleop.id=leader
```

Calibration JSONs land in lerobot's calibration directory; point the nodes'
`calibration_dir` at it (each node's manifest bind-mounts it) and `robot_id` /
`teleop_id` at the file stems. The nodes call `connect(calibrate=False)` and
refuse to start uncalibrated: lerobot's interactive calibration flow must
never block a headless container.

## URDF

Nothing to fetch: `so101_description` embeds a geometry-stripped variant of
SO-ARM100's `so101_new_calib.urdf` (Apache-2.0), and the backbone builds its
FK/IK and joint limits from that model, so the postures and limits are
always validated against the same bytes the constants were verified on. The
end-effector frame is `gripper_frame_link`, a fixed TCP near the jaw
region. On this single-moving-jaw gripper the pad midpoint shifts with the
opening, so poses name that fixed frame rather than the contracts' exact
grasp point; measure any offset that matters during hardware bring-up.

## Inverse kinematics, and what it refuses

The backbone solves with lerobot's placo-based `RobotKinematics`.
`RobotKinematics.inverse_kinematics` is **one linearised QP step**, sized for
streaming small deltas rather than for reaching a pose from anywhere.
`so101_description.kinematics` wraps it: a point-to-point solve iterates that
step up to `IK_MAX_ITERATIONS` and accepts only when forward kinematics puts
the grasp point where the solve asks. A `move_arm` walks the straight line to the
pose, as the OpenArm does: one verified solve per sample of the line, the grasp
point within 1 mm of its sample and the last sample within 0.01 rad of a named
orientation, and plays the configurations as a joint path, so a `move_arm` never moves the arm to a pose it did
not actually reach. A `move_arm` goal no line reaches fails with `unreachable`, and
`check_arm_move` answers `reachable` false with the same reason. A line, a joint move
or a posture move the joint limits cannot run within 60 s fails with `rejected`.

That verification makes refusals honest, not rare. The search stays a local
descent from the arm's current posture, so it finds solutions on that branch
and no others, and a line must solve at each of its samples. Measured on 300
goals near an arbitrary in-limits posture, each a pose reachable by construction
(the forward kinematics of a joint vector within 0.6 rad of the posture):

| Goal | Plans a line |
|---|---|
| The pose, its orientation exactly reachable | 271 of 300 |
| The pose, its orientation off by 0.05 rad | 52 of 300 |
| Its position alone, `orientation` left out | 218 of 300 |

Read a refusal as *this branch did not walk the line*, not as *the arm cannot
reach it*. The same target frequently solves from a different starting posture.
A caller that hits a refusal can move the arm somewhere else and ask again, or,
where the launch runs a motion planner, plan a path with `plan_joint_path` and run
it with `execute_joint_path`. A goal refused for its orientation names the slack it
takes in the orientation constraint's `tolerance_rad` or `free_axis`.

### The path is a straight line

`move_arm` walks the line to the pose and plays its configurations as a joint path.
On those goals the grasp point stays under 1 mm off the line. The orientation is
free along the way and lands within 0.01 rad of a named orientation. A joint step past
0.35 rad is a jump to another branch and refuses the line.

The joint limits and the goal's duration time the move. Teleop follows the commanded
stream tick by tick and plans no line.

Two further limits worth knowing:

- **Orientation is hard to meet.** Five joints underactuate three rotational
  degrees of freedom, so orientation is a soft, low-weight objective of the QP,
  and a goal that names an orientation the arm cannot end on within 0.01 rad has no
  plan. A goal that leaves `orientation` out is a position alone: the solves aim at
  the turn the line starts with, and on those goals it ends a median 0.17 rad and at
  most 0.96 rad from it.
- **The reach ball bounds outward extent only.** Streamed pose targets are
  clamped into a ball fitted to the sampled reachable set
  (`backbone/src/so101_backbone/reach.py`) so the streaming solver is never
  asked for a far-out-of-reach pose. Points near the base sit inside that ball
  and are not clamped.

This is accepted for now. Teleop through `xr_commander` drives the streaming
path, which is best effort and unverified by design, and is unaffected. The
cost falls on scripted `move_arm` goals.

## Workspace

The backbone implements `workspace:v1`. `describe_workspace` and
`check_positions` tell where the robot can work, from its design alone, in
the robot frame (the URDF's `base_link` frame). The answers know nothing of
the room: a surface is a flat, level plane at the height that the request
gives, with nothing on it.

- **Reach.** The arm `arm` reaches a point when the `ApproachSolver` of
  `so101_description` finds joints inside the joint limits that put the grasp
  point within 1 cm of the point, with the approach axis of the gripper
  within 0.05 rad of straight down or straight forward. Forward kinematics
  verifies the joints. The backbone tries down first, then forward, for each
  point. The roll of the gripper about its approach axis is free. For a point
  that the reach sphere of the solver rules out at 1 cm, the solver gives no
  joints at once and does no search (see `ApproachSolver` in
  `so101_description.kinematics`).
- **Short by.** For a point that the arm does not reach, `short_by` is the
  least distance: the distance from the point to the nearest grasp point that
  the search of the solver finds, in any orientation. The search of the solver
  is local, so this distance can be longer than the true least distance. Thus
  this distance does not decide whether the arm reaches a point. `short_by` is
  this distance also when the grasp point gets within 1 cm of the point, but
  in no grasp direction. The documentation of `ApproachSolver` and
  `ApproachSolver.least_distance` gives how far from the true least distance
  this distance can be, and the measured rates at which the solver reaches
  points.
- **Grasp point.** The grasp point is the origin of `gripper_frame_link`, the
  fixed frame that `limb_state` reports. On this gripper the midpoint between
  the pads moves with the opening, so this frame is an approximation of the
  grasp point of the contracts (see [URDF](#urdf)).
- **View.** The robot has no perception camera: its one camera, `wrist`, is
  on the arm. Thus `perception_camera` is `""`, the view of each point is
  `no_camera`, and the message of each point says that the view is not
  checked. A point is workable when the arm reaches it.

The grid, the limits, the parsing of a request and its refusals, the reach
memo, and the composition of each answer and its messages come from
`workspace_core_py`, the Python bindings of `workspace_core`. Thus these
answers read the same as the answers of the OpenArm and of a simulation. The
backbone gives an error reply with `rejected` to a height that is not a
finite number or that is more than 1000 m from the base point, an empty
position list, a list that does not divide into points of 3 values, a value
that is not a finite number, and a coordinate more than 1000 m from the base
point.
Both services read nothing of the arm and move nothing, so they answer from
bringup, also when no follower state comes in.

The solver work for one surface (663 grid targets) lasts many ticks of the
control loop (the documentation of `ApproachSolver` gives its measured time).
The process of the backbone runs the control loop, and only one of its threads
runs Python at a time. Thus the backbone does the solver work in one worker
process, with a solver of its own. It starts this process with the `spawn`
method at the first request. On a loaded AMD EPYC Genoa, the start of the
worker with a first point took 0.22 to 0.27 s of wall time. A request sends
its points to the worker in chunks of one surface (663 points), the next chunk
when the worker has measured the chunk before it. The worker measures one
chunk at a time, in the order that the requests send them. Thus a request
waits for at most one chunk of each request in front of it.

The backbone keeps the reach of the last 32 surface heights (to the
millimetre) in its own process. A height that it keeps gets its answer
without the worker: on that host, in a median time of about 0.1 ms. When the
backbone keeps 32 heights, a new height replaces the height that it kept
first.

When the node stops, the worker stops after the chunk that it measures. The
worker also stops when the process of the backbone stops without a shutdown.
If the worker stops while it has a chunk, the request of that chunk gets an
error reply with `attempt_failed`. If the worker stops while it has no chunk,
the next chunk starts a new worker. But if that chunk goes to the process
pool of the worker before the pool has seen that the worker stopped, the
request of that chunk gets the same error reply, and the chunk after it
starts a new worker. After the node stops, each request gets an error reply
with `resource_unavailable`.

## Serial devices

The follower and leader adapters are identical USB serial bridges. Install
`launchers-hub/so101/rules/60-so101.rules` to pin stable
`/dev/so101_follower` / `/dev/so101_leader` symlinks before first launch.

## Shared libraries

Four libs in the
[public-peppy-libs](https://github.com/Peppy-bot/public-peppy-libs)
repository, consumed as uv git dependencies like the Rust nodes consume
`control_core`:

- `control_core_py`: generic Python node plumbing (asyncio stream helpers,
  parameter validators, the hardware device-thread skeleton). Nothing
  SO-101-specific.
- `so101_description`: the robot's identity (joint and motor names, wire
  units, limb names, the named postures, STS3215-shaped setpoint parsing,
  the lerobot device boundary, and the embedded kinematics URDF with its
  TCP frame, parsed limits, and placo FK/IK behind a `kinematics` extra,
  with the `ApproachSolver` of the workspace answers),
  `openarm_description`'s sibling.
- `workspace_core_py`: the Python bindings of `workspace_core`, for the
  backbone only: the parsing of a `workspace:v1` request, the grid, the
  reach memo, and the answers of a robot without a perception camera. It is
  a PyO3 extension that uv builds from its source, so the image of the
  backbone installs a Rust toolchain for the build and removes it after.
- `path_timing_py`: the Python bindings of `control_core`'s path timing, for
  the backbone only: the joint trajectory of `execute_joint_path`, its
  corners rounded and its timing under the joint velocity and acceleration
  caps, as the OpenArm backbone plays it in Rust. It is a PyO3 extension,
  built as `workspace_core_py` is.

All three nodes follow `main`, like the Rust nodes' `control_core`.

## Testing

Each node carries pure-logic tests (parsing, health policy, governor and
coordinator arbitration, action lifecycles, and the workspace answers of the
backbone with its worker process); the follower, leader, and backbone add
peppygen harness tests that boot the node in-process against mocked pairing
peers and fake hardware.

```sh
cd <node> && peppy node sync
PEPPY_ZENOHD_PATH=~/.peppy/bin/zenohd uv run pytest
```

The harness spins an ephemeral zenoh router per test, so no daemon or stack
is needed; `PEPPY_ZENOHD_PATH` names the router binary shipped beside the
`peppy` CLI.
