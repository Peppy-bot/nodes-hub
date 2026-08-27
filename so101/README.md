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
| `so101_backbone` | The motion authority in between. Follower role toward whatever leads (joint, pose, or gripper streams), leader role toward the follower; exposes the `limb_motion` and `postures` move actions and the `limb_state` readout. A joints-led stream passes through under the end-effector-speed governor, its only limiter; a pose-led stream is reach-clipped, solved, and rate-stepped per joint before that same governor. Move actions run minimum-jerk plans sized by the per-joint velocity caps, and Cartesian moves additionally by the EE speed caps. Everything gates on fresh follower state. |

```text
so101_leader ──joint+gripper──▶ so101_backbone ──joint+gripper──▶ so101_follower
xr_commander ──pose+gripper──▶ (same backbone, upstream_mode="pose")
lerobot_recorder observes the follower pairings and the backbone's leader slots
```

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

## Serial devices

The follower and leader adapters are identical USB serial bridges. Install
`launchers-hub/so101/rules/60-so101.rules` to pin stable
`/dev/so101_follower` / `/dev/so101_leader` symlinks before first launch.

## Shared libraries

Two libs in
[public-peppy-libs](https://github.com/Peppy-bot/public-peppy-libs),
consumed as uv git dependencies exactly like the Rust nodes consume
`control_core`:

- `control_core_py`: generic Python node plumbing (asyncio stream helpers,
  parameter validators, the hardware device-thread skeleton). Nothing
  SO-101-specific.
- `so101_description`: the robot's identity (joint and motor names, wire
  units, limb names, the named postures, STS3215-shaped setpoint parsing,
  the lerobot device boundary, and the embedded kinematics URDF with its
  TCP frame, parsed limits, and placo FK/IK behind a `kinematics` extra),
  `openarm_description`'s sibling. A future arm adds its own description
  and reuses `control_core_py` unchanged.

The pyprojects pin `main`, like the Rust nodes' `control_core`.

## Testing

Each node carries pure-logic tests (parsing, health policy, governor and
coordinator arbitration, action lifecycles); the follower, leader, and
backbone add peppygen harness tests that boot the node in-process against
mocked pairing peers and fake hardware.

```sh
cd <node> && peppy node sync
PEPPY_ZENOHD_PATH=~/.peppy/bin/zenohd uv run pytest
```

The harness spins an ephemeral zenoh router per test, so no daemon or stack
is needed; `PEPPY_ZENOHD_PATH` names the router binary shipped beside the
`peppy` CLI.
