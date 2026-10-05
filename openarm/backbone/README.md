# openarm_backbone

The bimanual motion authority. One node sits between whatever leads (the
commander panel or a leader arm rig) and the four followers (two arms, two
grippers), and everything that reaches a follower has passed through one
governed pipeline against one self-collision model. The same binary drives
hardware, MuJoCo, and Isaac, because the launcher decides what pairs into
each slot.

```text
        leader_left_arm . leader_right_arm             [joints mode]
        leader_left_arm_pose . leader_right_arm_pose   [pose mode; pose_states back up]
        leader_left_gripper . leader_right_gripper     [either mode]
 (streams in)  joint_ or pose_setpoints  |  gripper_setpoints  [pairing slots, leading node]
                           v             v
 collision_ctrl --> +--------------------------------+ --> collision_status, limb_states
 (governor_control) |          coordinator           |     (readout topics)
 move_arm[_joints]  |  planners --> GOVERNOR --> pub | --> get_limb_names (service)
 move_gripper ----> |  (per arm)    (16 DOF)         |     (answered from bringup)
                    +--------------------------------+
                           v             v
                  joint_setpoints        |    gripper_setpoints  [pairing slots, follower side]
 (streams out) left_arm . right_arm . left_gripper . right_gripper
                        ^    joint_states / gripper_states (measured, relayed back up)
```

Sixteen degrees of freedom are governed as one configuration: seven joints per
arm plus each gripper's opening fraction (0 closed to 1 fully open).
A gripper is an ordinary governed DOF, so every guarantee the arms get covers the
fingers identically.

## The tick

`coordinator::run` owns the loop. Each tick, in order:

1. **Consume the streams of busy sides** - a discrete move wipes its side's
   streamed command every tick, so a setpoint still in flight when the move
   was fired cannot re-target the arm (or snap the grippers) when the move ends.
2. **Apply controls** - the commander's live `governor_control` stream retunes
   the enable toggle, the band, and the EE speed cap; invalid values are
   rejected, keeping the last good ones.
3. **Admit the followers** (`liveness`) - a follower that stopped delivering
   measured state freezes its limb at the held setpoint and its slot goes
   silent; the first delivery back re-anchors the setpoint on the measured
   pose, so a restarted follower is never handed the drift it accumulated
   while nobody could see the arm.
4. **Advance the arms** - each planner turns its inputs (streamed command or an
   in-flight move) into a rate-limited candidate, and hands out the side's
   end-effector Jacobian when the target came from the operator's stream.
5. **Service gripper moves** - `move_gripper` goals chase through the same
   governed configuration as everything else, then hold their target until
   the measured gripper stands still.
6. **Govern** - one `Governor::govern` call over the whole 16-DOF step.
7. **Publish** - governed setpoints to the follower slots, measured states
   relayed up the leader slots, and the proximity readout at ~20 Hz.

## The governed pipeline

`Governor::govern(prev, cand, measured, hands, dt)` runs six stages, each
contractive with respect to the last:

```text
 parse -> limit(speed) -> sense -> limit(tripwire) -> project -> clip
           always on      [------- collision avoidance, toggleable ------]
```

1. **Parse.** A non-finite endpoint is a fault hold, not a step.
2. **Limit (speed).** `DofSpeed` (per-DOF rate bound) and `EeSpeed` (the
   operator's hand-speed cap, applied per side to stream-driven ticks via the
   Jacobian the planner handed out). These are motion shaping, not collision
   guards: they run in both modes, so the collision toggle gates collision
   avoidance and nothing else.
3. **Sense.** One immutable snapshot of everything the limiters and the
   projection decide on. The clip stage still probes the model live - its job
   is to check configurations no snapshot can anticipate - and every query in
   the governor goes through one placement-explicit door
   (`governor/model.rs`): each call takes the whole configuration, so a read
   at a stale finger placement is unrepresentable, not merely avoided.
4. **Limit (tripwire).** `MeasuredTripwire`, defense in depth against tracking
   error: latched with hysteresis on the *measured* clearance, it holds
   closing motion per side until the real gap recovers. It alone feeds the
   collision readout, so a speed cap can never read as a collision event.
5. **Project.** The closing-velocity barrier (a Faverjon-Tournassoud velocity
   damper): remove just enough of the gap-closing component that the clearance
   loses no more than `allowed_closing(d) * dt`, leaving tangential and
   separating motion at full speed. Directional, so it cannot be a per-DOF
   limiter; it runs after them so its guarantee holds on the published step.
6. **Clip.** The exact floor scan. Surface distance is not monotone along a
   joint-space segment, so this walks the realized segment and retracts to the
   furthest point that stays at or above the step floor. A separating side can
   earn an exemption from a pushing partner's clip, and the exemption's result
   is re-scanned on the true prev-to-published line, so the point that goes
   out is always one proved on the path the arms actually travel.

Limiters are pure functions of the step, constructed per tick with exactly the
data they read, and combine by keeping the most restrictive fraction per DOF,
so their order can never change the governed step, only which name is recorded
on a tie.

Inside an actual overlap the floor relaxes to a bounded rate of loss
(`RECOVERY_LOSS_M_PER_S`), because an escape routinely sweeps deeper before it
separates; a strict floor would trap the operator inside the collision.

## What the toggle means

Disabling the governor stands down stages 3-6: sensing, the tripwire, the
barrier, and the scan. The speed caps keep working. Disabling with a closing
command live *will* collide (that is what disabling means); the governed path
back out exists once re-enabled, and the sim campaign exercises exactly that.

A wedge lesson worth knowing: parked against the stop with the grippers wide, a
bundled "go somewhere safe" command can be net-closing for the binding pair
(the chase's first step sweeps the open fingers toward the torso) and the
governor will rightly refuse it. Single-purpose commands escape: close the
grippers first, then reposition. The regression suite pins this from a captured
field pose.

## Actions

| Action | Goal | Refused when |
|---|---|---|
| `move_arm_joints` | `arm_name`, 7 joint positions (rad), `duration_s` (the limb_motion contract) | non-finite, negative duration, out of joint limits, side busy |
| `move_arm` | `arm_name`, robot-frame pose (position m + quaternion `[x, y, z, w]`), `duration_s` (the limb_motion contract) | non-finite, degenerate quaternion, negative duration, side busy |
| `move_gripper` | `gripper_name`, opening fraction in [0, 1], `max_effort` (the limb_motion contract) | non-finite, out of range, negative effort cap, side busy |
| `move_to_ready` | `duration_s` (the postures contract; both arms to the Ready posture) | non-finite, negative duration, over 600 s, either arm busy |
| `move_to_home` | `duration_s` (the postures contract; both arms to the Home rest) | non-finite, negative duration, over 600 s, either arm busy |

`arm_name`/`gripper_name`: `left_arm`/`right_arm` and `left_gripper`/`right_gripper`,
an unknown name refused in the result. Every pose here, commanded or
reported, is in the robot frame the contracts define, which for this robot
is the root link of the URDF: the bottom of the pedestal's base plate on the
column axis, `+x` to the robot's front, `+y` to its left, `+z` up. It is the
gripper's **grasp point** (midway between the pads on the jaw closing axis,
`+z` out of the gripper), not the wrist flange they mount on:
`openarm_description` carries the offset per generation and `arm_model` builds
the model with it, so the pose solved for, the pose reported, and the point the
EE speed cap applies to are one frame. One move per side at a time (a
single-flight busy slot whose release rides a drop guard, so no terminal can
leak it). `move_arm` plans the quietest tier that works: a held-elbow line, a
steered-elbow line, or the guarded servo (a damped resolved-rate law that can
cross singular surfaces a discrete IK walk cannot). Planning runs after
admission (peppy's goal decision is pre-context, so reachability cannot be a
refusal): an accepted goal whose pose no tier reaches, servo rollout
included, completes unsuccessfully at once. Completion of an arm move is
graded on the commanded motion with a 2x-nominal timeout; results report the
measured state and the caller judges how close it landed (the governor may
have held it short, and that is not a failure of the move machinery).
When the arm's follower stops reporting, the move fails with
`the follower stopped reporting; the result comes from the joints commanded last`.
Its result then reports the held setpoint, because the last measurement is
stale.

`move_gripper` ends on the measured gripper instead. The commanded opening
ramps to the target under the same 2x-nominal timeout, then the move stays in
flight, still commanding the target and still holding the side's busy slot,
until the measured opening stands still: within 0.002 of a reference opening
for 0.25 s, judged on delivered openings only. `final_opening` is the opening
measured then and `action_time` runs to that moment.

| End of the move | `success` | `message` |
|---|---|---|
| Still, within 0.01 of the target | true | `move complete` |
| Still, farther from the target (an object or the effort cap holds the jaws) | true | `move complete: the gripper stopped at <opening>, short of the target <target>` |
| Still moving 3 s after the commanded opening landed | false | `the gripper still moved <seconds> s after the commanded move ended` |
| The follower stops reporting while the gripper settles | false | `the follower stopped reporting` |
| The commanded ramp overruns its timeout (a collision-governed clamp) | false | `overran 2x its <seconds>s nominal travel, short of the target (a collision-governed clamp ends here)` |
| Cancelled, in the ramp or while settling | false, completed as cancelled | `goal cancelled` |

A gripper that stops short is a success on purpose: the move was neither
refused, failed nor cancelled, and the caller judges the grasp from
`final_opening`. After the move, the backbone sends the gripper nothing
until the next gripper move or a leader's gripper stream, so the gripper
keeps the move's target opening and `max_effort` (relayed unchanged) and
goes on squeezing what it holds.

`move_to_ready` and `move_to_home` give each arm's planner a joint move to
the posture and complete when both moves end. `success` says that both
moves ran their time out, not that the arms arrived: the governor can hold
an arm short, and the backbone does not see an arm that stops against an
object. So the message of a success is `the move to ready ran its time` or
`the move to home ran its time`. Whatever the terminal, the result gives `arm_names` (`left_arm`,
`right_arm`, the order of limb_state) and, in that order, `positions` (3 per
arm, m) and `orientations` (4 per arm, `[x, y, z, w]`): the grasp pose of
each arm in the robot frame, from the joints it measured when its move
ended, as limb_state gives it for those joints. After a cancel or a stop,
that is where each arm was when the move ended. When an arm has no such
pose, the three arrays are empty and the message ends with
`; no arm poses: <arm> has not measured its joints` (a goal during the seed
wait), `; no arm poses: <arm> stopped reporting its joints` (its follower
stopped reporting, so its last measurement is stale) or
`; no arm poses: <arm> did not report the end of its move` (its planner is
unavailable or dropped the move). A posture move sends no
gripper command, so each gripper goes on as its last command drives it. On
v2 with both grippers fully open, the jaws at Home sit closer to the torso
than the validated stop distance (`d_stop_m` 5 mm), so the governor holds
both arms short of Home: close the grippers first.

### Services of limb_motion

`stop` ends every planned move in flight, whoever started it: the arm moves
through their planners, the gripper moves through their terminal, the
posture moves through both arms, and the goals admitted but not started
yet. Each goal ends as cancelled with `stopped: <reason>` (`stopped` for an
empty reason); a posture goal ends as cancelled with its first arm's
message. Each limb holds the setpoint it was last governed to, as after a
cancel, and the answer's `stopped` names the limbs whose move was in flight,
empty with `nothing was moving` when none was. The stop does not latch: a
goal admitted after it runs. A leader's streamed setpoints are not stopped.

`check_arm_move` takes the fields of a `move_arm` goal and answers whether
a plan reaches the pose from the arm's held setpoint, with the time the
move would take, without moving: the same validation, the same planner and
the same refusal words as the goal (`goal pose not planned within ...`).
It is refused while the arm executes a move, with the message a second
goal gets, and it checks no collision with the robot or the room.

Both are answered by the coordinator on the tick after the request, so
they wait behind the readiness gate with the moves; a stop during the seed
wait stops nothing and a check then is refused as a goal is.

### camera_mounts

`get_camera_poses` answers where the generation's design carries each
camera, in the robot frame, as the pose of the camera's colour optical frame
(`+x` to the right of the image, `+y` down it, `+z` along the view): the
numbers `openarm_description` lists, which are the simulation's, so they are
exact in a simulation and nominal on hardware. A camera fixed to the base
(v2's `chest`) has one pose; a camera an arm carries (v2's `wrist_left` and
`wrist_right`, which hang off the link the grasp point hangs off) is composed
with that arm's grasp pose from the last `limb_states` snapshot, and the
answer carries that snapshot's stamp. Served from bringup, refused with
`the robot has not measured its joints yet` until a snapshot exists; a v1
robot answers success with no camera. A description that mounts a camera on
a link a moving joint carries, other than the grasp point's, stops bringup.

### workspace

`describe_workspace` and `check_positions` answer where the robot can work
from its design alone, in the robot frame, knowing nothing of the room: a
surface is a flat, level plane at the height asked, with nothing in the way.
A point is reachable when an arm brings its grasp point within 1 cm of it
in one of the 16 grasp orientations `workspace_core` lists (the gripper
pointing straight down or straight forward, each at 8 rolls), solved by the
arm's closed-form IK seeded at Ready: a grasp within 1 cm of an edge of
the arm's reach, outside or inside it, is solved moved into that reach by
less than 1 cm, in the same orientation.
The arm whose base stands nearer the point is tried first and named. A
point out of reach carries how far the closest arm stops its grasp point
short of it in any orientation, its joint limits aside: at most 0.01 when
the grasp point gets within 1 cm of it, but in no grasp orientation, which
the message words as `no arm reaches it with its gripper pointing down or
forward`. A point is in view when it lies inside the perception camera's
field of view and its depth, by the depth model the camera gives (`z`, the
depth along the optical axis, or `range`, the straight-line distance), lies
inside the camera's depth range. The perception camera is derived
from the description, never configured: the one camera that gives depth
and that no arm carries (v2's `chest`; v1 has none, and is judged on reach
alone). Its pose is the design's, the one `camera_mounts` reports; its
field of view, depth model and depth range are what the camera linked as
`perception_geometry` (`camera_geometry:v1`) answers at each request,
within 2 s for each of its two answers.

| `perception_geometry` | The answers |
|---|---|
| vacant | no view check (`no_camera`), and the message says `The view is not checked: no camera geometry is linked for the chest camera.` |
| linked, answering | the view is checked through the camera's own intrinsics and depth stream |
| linked, colour intrinsics refused, not answered in time, or not a pinhole model | refused: `cannot read the colour intrinsics of the chest camera, the perception camera: <reason>` |
| linked, depth intrinsics not answered in time, not a depth range, or of a depth model `camera_geometry:v1` does not name | refused: `cannot read the depth intrinsics of the chest camera, the perception camera: <reason>` |
| linked, depth intrinsics refused (a camera that gives no depth) | refused: `the camera linked as the chest camera, the perception camera, gives no depth: <reason>` |

The grid, the limits, the verdicts, the largest workable rectangle and the
messages of the verdicts, of the count of workable points and of a view not
checked (for a robot without a perception camera too) are
`workspace_core`'s, so these answers and the simulation's read alike; the
refusals of a request and of the camera's geometry, in the table above and
below, are the backbone's own.

`describe_workspace` measures the surface at its height to the millimetre,
each grid point's reach at a target 4 cm above it, and keeps the reach
grids of the last 32 heights it measured (the one stored first goes past
that), so a repeated height answers from the stored grid. A height that is
not a finite number, an empty position list, one that does not split into
points of 3 values, a value that is not a finite number, and a coordinate
more than 1000 m from the robot's base point are refused. Both services
read nothing of the arms and move nothing, so they answer from bringup,
ahead of the readiness gate. The robot's own body is not checked for hiding
a point from the camera.

## Module map

| Module | Owns | Why it lives here |
|---|---|---|
| `main.rs` | bringup: params, models, channels, task supervision | first task exit is fatal; the daemon restarts a clean process |
| `startup.rs` | the robot_initializer gate | nothing streams before the robot is ready |
| `streams.rs` | every subscription + parse-at-the-boundary types (`GripperCommand`, `ArmState`, `GripperState`) | one receive policy (`subscribe_pair` + `accept`); a malformed message is dropped with a reason, never driven |
| `upstream.rs` | `UpstreamMode` (which upstream slot kind is followed) + `Upstream` (the parsed joint or pose command) | one command authority per arm is unrepresentable, not checked per tick |
| `publish.rs` | every publisher (`Publishers`), one stamp/build/publish/log path | peppy vocabulary: a publisher on a slot; "wire" means the transport encoding only |
| `coordinator.rs` | the tick, gripper move execution, the upstream relay | IO orchestration; its seam with the safety core is exactly one `govern` call, which is why the governor is not folded into it |
| `liveness.rs` | follower admission (`Live` / `Reanchor` / `Stale`) | delivery-cadence policy for the coordinator, independent of any message type |
| `planner.rs` | per-arm mode machine (Follow / joint move / Cartesian move) -> one rate-limited candidate + the stream-tick Jacobian | knows one arm only; never the other arm, never the collision model |
| `chase.rs` | `rate_limited`, the one per-tick rate clamp every chase shares | the arm chase, the gripper chase and the servo re-clamp cannot round differently |
| `trajectory.rs` | quintic joint trajectories, Cartesian line planning, tier selection | plan-time; validates what the planner then executes |
| `servo.rs` | the guarded servo law + its plan-time rollout | identical law offline and online, so acceptance is proof |
| `governor/mod.rs` | `GovState`, the pipeline, the runtime controls, disposition and readout | the only mutable resource is the collision model |
| `governor/sense.rs` | the pre-projection model read (`Sensed`) | limiters and the projection decide on one snapshot |
| `governor/model.rs` | `ConfiguredModel`, the only door to the collision model | every query takes the whole configuration; stale-placement reads are unrepresentable |
| `governor/limiters/` | the `Limiter` trait, one module per limiter, and `allowance.rs` (the `Allowance`/`Limits` currency they speak) | everything expressible as a per-DOF fraction lives together |
| `governor/barrier.rs` | the projection and the floor scan | the two stages that are not per-DOF fractions |
| `torso.rs` | the torso clip regions the URDF does not carry | geometry facts, versioned with the node |
| `actions/` | goal admission (validate + claim), nothing else | execution belongs to the planner/coordinator that owns the state |
| `workspace.rs` | the workspace judgement from the design: reach per grasp orientation, the perception camera, the per-height reach memo, the parsing of a request and of the camera's geometry | pure: no messaging, so every answer is unit-tested without a node |
| `workspace_service.rs` | the two workspace services and the read of the linked camera's geometry | the thin edge between the generated handlers and `workspace.rs` |
| `serving.rs` | the loop of each service answered for the life of the node (`get_limb_names`, `get_camera_poses`, the workspace services): ended by the node's cancel, an error logged and waited out for 1 s | one loop, so no service can hot-spin on a broken transport or outlive the node |
| `types.rs`, `arm_pair.rs` | `ARM_DOF`, `JointVec`, `Side`, the motion-timeout rule, `ArmPair` | shared primitives |

## Parameters and links

See `peppy.json5` for the full commented list. The operational governor
parameters (`d_stop_m`, `d_safe_m`, `collision_governor_enabled`,
`max_ee_velocity_m_s`) are required launcher arguments with no node defaults,
and the commander's `governor_control` stream retunes them live.
`follower_state_rate_hz` is required too: the rate the followers deliver
measured state at, of which four silent periods freeze the limb, and under
three quarters of which draws a warning (100 for the real arms and MuJoCo;
the Isaac fragment declares its 60 fps frame rate, since that simulation
reports once per rendered frame).
`upstream_mode` is likewise required: `"joints"` follows the joint_link
leader slots, `"pose"` the pose_link ones, and only the named kind is
subscribed. Link the leader into the slots that kind names, or nothing it
streams is read and the arms never move.
`max_ee_angular_velocity_rad_s`, required like the linear cap, caps the
rotation of streamed poses and the servo reference; unlike the linear cap it
is launch-time only, not retuned by the live speed control. All ten pairing slots (six toward the
leading node, four toward the followers) are optional and established by the
launcher: each one either names a peer, from this instance's `links` or the
peer's own, or is declared `{ vacant: "<why>" }`, and a slot left unmentioned
by both ends fails launch validation. Publishing on a vacant slot is a legal
no-op, so partial deployments and monitors boot cleanly. The
`perception_geometry` dependency slot is optional too: bind it to the
camera_geometry provider of the perception camera (`chest` on v2) for the
workspace answers to check the view, or leave it vacant for reach alone.

## Build, run, test

```sh
# Build into the node stack (never plain cargo for deployment):
peppy node add /path/to/nodes-hub/openarm/backbone -sb

# Launch the whole stack (sim shown; the backbone and commander pair
# mutually, so cold starts go through a launcher):
peppy stack launch openarm_simulation --with mujoco

# Unit tests run directly; both hardware generations' models are exercised:
cargo test
```

## Performance

Two hand-run reports ship with the suite:
`governor_tick_timing_report` (what a tick costs) and `control_rate_sweep`
(whether a faster loop is affordable), both
`cargo test --release <name> -- --ignored --nocapture`. Numbers below are the
**Jetson** (aarch64). One distance or gradient query ~170 us there; the speed 
limiters ~0.1 us; a disabled-governor tick 0.1 us, because the model is never 
touched.

Cost is probes per tick, and probes per tick are set by how far the step
travels and whether the clip stage engages. The steady-state regimes (parked,
holding, approaching) settle at a handful of probes and 0.8-1.8 ms, but they
are not the budget that matters: none of them clips, so none reaches the
machinery that makes a tick expensive. A jog does. It commands a fresh pose
every tick, so its segment needs full probe density, and a clipped tick walks
that segment three times (the strict scan, the exempted scan, and the re-scan
that proves the published line).

**A tick's cost is proportional to the distance it travels, so it falls as the
control rate rises.** A segment is `speed * dt` long and is probed once per
`MAX_PROBE_ARC_RAD`, giving `v_max * dt / MAX_PROBE_ARC_RAD` probes per scan:
at 100 Hz the 16.75 rad/s J1/J2 limit is 0.168 rad per tick and ~17 probes,
while at 500 Hz the same jog is 0.034 rad and ~3, floored by
`SEGMENT_SAMPLES_MIN` at 4. The work per second is about the same either way
(~1700 vs ~2000 probes); the rate only decides whether it arrives in one
indivisible lump that must fit inside one period, or spread thin. Sweeping
the rate against a jog at the J1/J2 limit (Jetson, 20 s per rate):

| rate    | budget | p50     | p95     | p99     | max      | over budget  |
| ------- | ------ | ------- | ------- | ------- | -------- | ------------ |
| 100 Hz  | 10 ms  | 0.34 ms | 1.46 ms | 3.00 ms | 10.30 ms | 1 / 2000     |
| 250 Hz  | 4 ms   | 0.31 ms | 2.17 ms | 2.19 ms | 2.98 ms  | 0 / 5000     |
| 500 Hz  | 2 ms   | 0.31 ms | 0.97 ms | 1.02 ms | 2.53 ms  | 7 / 10000    |
| 1000 Hz | 1 ms   | 0.31 ms | 0.97 ms | 0.98 ms | 2.40 ms  | 178 / 20000  |

Per-tick cost against jog speed at the shipped rate, stated in rad/s because
that is the rate-independent quantity (Jetson, 100 Hz, 4000-tick walk):

| jog         | per tick  | p50     | p95     | p99     | max      |
| ----------- | --------- | ------- | ------- | ------- | -------- |
| 2.0 rad/s   | 0.020 rad | 0.50 ms | 1.27 ms | 1.78 ms | 3.08 ms  |
| 5.0 rad/s   | 0.050 rad | 0.40 ms | 1.01 ms | 1.34 ms | 3.77 ms  |
| 16.75 rad/s | 0.168 rad | 0.38 ms | 1.05 ms | 2.82 ms | 10.29 ms |

The last row is the J1/J2 velocity limit, the fastest any joint is chased at
(J3/J4 are clamped lower by `DofSpeed` before the scan sees them). A live 
MuJoCo test of aggressive teleop logs no control-loop overruns at all; the 
same test against a build whose clip stage refined every breach by bisection 
logged them at ~37/min, worst 15 ms late. Overruns degrade safely when they 
happen - `dt` is the nominal period rather than the measured one, so a late 
tick cannot inflate the next step, the Pacer re-anchors instead of bursting to 
catch up, and the follower runs its own control loop off the last setpoint it 
got - but they cost smoothness, so the loop is sized to avoid them rather than 
to survive them.

Part of the remaining tail is not the code's. Timing an *identical* 12-probe
tick 20000 times spreads p50 1.5 ms to max 6.8 ms on a loaded machine: this is
a general-purpose kernel, not an RTOS, so preemption, migration and frequency
scaling set a jitter floor that no algorithmic work removes. Treat a measured
maximum as compute plus scheduling, and compare implementations on p99.

The test suite pins behavior, not just code paths: bit-exact passthrough where
the followers require it, the floor holding under a 28k-tick random walk (with
the scan's between-probe residue as a named, tested bound), escapes from
penetration never trapped, the captured wedge pose refused but never a trap,
and the follower-restart re-anchor. The sim campaign drives the live stack
through collisions, band and cap retunes, toggle cycles at the wall, follower
kills, and the whole action surface.
