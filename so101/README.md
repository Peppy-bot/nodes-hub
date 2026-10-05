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
| `so101_backbone` | The motion authority in between. Follower role toward whatever leads (joint, pose, or gripper streams), leader role toward the follower; exposes the `limb_motion` and `postures` move actions, the `limb_motion` services (`stop` ends every planned move in flight, whoever started it, each goal then ending as cancelled with `stopped: <reason>`; `check_arm_move` runs a `move_arm` goal's solve from the same anchor and moves nothing), the `limb_state` readout and the `workspace` services (`describe_workspace` and `check_positions`: where the arm reaches, from the design alone; see [Workspace](#workspace)). Every pose is in the URDF's `base_link` frame, the robot frame the contracts define: `+x` the way the robot faces, `+z` up. A joints-led stream passes through under the end-effector-speed governor, its only limiter; a pose-led stream is reach-clipped, solved, and rate-stepped per joint before that same governor. Move actions run minimum-jerk plans sized by the per-joint velocity caps, and Cartesian moves additionally by the EE speed caps. A gripper move ramps at the gripper rate cap and ends once the measured gripper stands still, on its target or short of it where an object holds the jaws. Everything gates on fresh follower state. |

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
  (openarm vocabulary): `so101_leader`, `xr_commander`, or a future policy
  runner. Every commander option fills the launcher's `commander_inst`.

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
the end effector within `IK_POSITION_TOLERANCE_M` of the target, so a
`move_arm` never moves the arm to a pose it did not actually reach.

That verification makes refusals honest, not rare. The search stays a local
descent from the arm's current posture, so it finds solutions on that branch
and no others. Measured against poses that are reachable by construction (a
joint vector drawn inside the model's limits, and its forward kinematics taken
as the target, so a solution provably exists):

| Seed posture | Refused |
|---|---|
| `ready` (the calibration middle) | 14% (n=400) |
| An arbitrary in-limits posture, which is what `move_arm` seeds from | 39% (n=300) |

Refusals concentrate near the base rather than at the rim: 25% within 0.23 m
of the base against 7% beyond 0.45 m. The gap between the seed posture and the
target posture shows no such gradient, so distance from the seed is not what
predicts a refusal.

Read a refusal as *this branch did not reach the pose*, not as *the arm cannot
reach it*. The same target frequently solves from a different starting posture,
which is why the seed posture dominates the table above. A caller that hits a
refusal can move the arm somewhere else and ask again.

### The path between the endpoints is not planned

`move_arm` plans endpoints only. It runs one IK solve for the goal pose and
then blends from the current joints to the solution with a minimum-jerk
quintic **in joint space**. Nothing constrains where the end effector goes in
between, and nothing verifies it. The arm does not walk the straight line
between start and goal; it walks whatever curve the joint blend traces.

Measured over 200 solved pose moves, the end effector's greatest departure
from the straight line between the endpoints:

| | Departure |
|---|---|
| Median | 91 mm |
| p90 | 240 mm |
| Worst seen | 472 mm |

On an arm whose whole reach is about 0.54 m, that is not a small bow. 6% of
moves dip more than 20 mm below *both* endpoints, so a move between two points
above the table can pass below the lower of them.

This also makes `max_ee_velocity_m_s` a nominal cap rather than a guarantee.
`_ee_floor_s` sizes the move's minimum duration from the straight-line chord,
but the path actually flown is longer than the chord, so the real end-effector
speed exceeds the cap by that ratio: median 1.20x, p90 1.51x, worst seen 2.88x.
The same applies to `max_ee_angular_velocity_rad_s`, sized from the relative
rotation between the endpoints.

Practically: keep the volume clear around more than the straight line, and
treat the speed cap as nominal. Teleop is unaffected, since the streaming path
follows the commanded stream tick by tick and never plans between two poses.

Two further limits worth knowing:

- **Orientation is not verified.** Five joints underactuate three rotational
  degrees of freedom, so orientation is a soft, low-weight objective of the QP
  and only position is gated. A successful `move_arm` reports the orientation
  it actually reached; check it if it matters.
- **The reach ball bounds outward extent only.** Streamed pose targets are
  clamped into a ball fitted to the sampled reachable set
  (`backbone/src/so101_backbone/reach.py`) so the streaming solver is never
  asked for a far-out-of-reach pose. Points near the base sit inside that ball
  and are not clamped, which is the region where refusals cluster.

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
  `no_camera`, and the messages say that the view is not checked. A point is
  workable when the arm reaches it.

The grid, the limits, the parsing of a request and its refusals, the reach
memo, and the composition of each answer and its messages come from
`workspace_core_py`, the Python bindings of `workspace_core`. Thus these
answers read the same as the answers of the OpenArm and of a simulation. The
backbone refuses a height that is not a finite number or that is more than
1000 m from the base point, an empty position list, a list that does not
divide into points of 3 values, a value that is not a finite number, and a
coordinate more than 1000 m from the base point.
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
If the worker stops while it has a chunk, the backbone refuses the request
of that chunk. If the worker stops while it has no chunk, the next chunk
starts a new worker. But if that chunk goes to the process pool of the
worker before the pool has seen that the worker stopped, the backbone
refuses the request of that chunk, and the chunk after it starts a new
worker.

## Serial devices

The follower and leader adapters are identical USB serial bridges. Install
`launchers-hub/so101/rules/60-so101.rules` to pin stable
`/dev/so101_follower` / `/dev/so101_leader` symlinks before first launch.

## Shared libraries

Three libs in the
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

The follower and the leader follow `main`, like the Rust nodes'
`control_core`. The backbone pins its three libs at one commit (`rev`), so
they always come from one state of that repository.

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
