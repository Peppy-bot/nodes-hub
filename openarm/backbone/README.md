# openarm_backbone

The bimanual motion authority. One node sits between whatever leads (the
commander panel or a leader arm rig) and the four followers (two arms, two
grippers), and everything that reaches a follower has passed through one
governed pipeline against one self-collision model. The node is
engine-agnostic: the same binary drives hardware, MuJoCo, and Isaac, because
the launcher decides what pairs into each slot.

```text
        leader_left_arm . leader_right_arm             [joints mode]
        leader_left_arm_pose . leader_right_arm_pose   [pose mode; pose_states back up]
        leader_left_gripper . leader_right_gripper     [either mode]
 (streams in)  joint_ or pose_setpoints  |  gripper_setpoints  [pairing slots, leading node]
                           v             v
 collision_ctrl --> +--------------------------------+ --> collision_status, limb_states
 (governor_control) |          coordinator           |     (readout topics)
 move_arm[_joints]  |  planners --> GOVERNOR --> pub |
 move_gripper ----> |  (per arm)    (16 DOF)         |
                    +--------------------------------+
                           v             v
                  joint_setpoints        |    gripper_setpoints  [pairing slots, follower side]
 (streams out) left_arm_link . right_arm_link . left_gripper_link . right_gripper_link
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
   governed configuration as everything else.
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
| `move_arm` | `arm_name`, world pose (position m + quaternion `[x, y, z, w]`), `duration_s` (the limb_motion contract) | non-finite, degenerate quaternion, negative duration, side busy |
| `move_gripper` | `gripper_name`, opening fraction in [0, 1], `max_effort` (the limb_motion contract) | non-finite, out of range, negative effort cap, side busy |
| `move_to_ready` | `duration_s` (the postures contract; both arms to the Ready posture) | non-finite, negative duration, over 600 s, either arm busy |
| `move_to_home` | `duration_s` (the postures contract; both arms to the Home rest) | non-finite, negative duration, over 600 s, either arm busy |

`arm_name`/`gripper_name`: `left_arm`/`right_arm` and `left_gripper`/`right_gripper`,
an unknown name refused in the result. Every world pose here, commanded or
reported, is the gripper's **grasp point** (midway between the pads on the jaw
closing axis, `+z` out of the gripper), not the wrist flange they mount on:
`openarm_description` carries the offset per generation and `arm_model` builds
the model with it, so the pose solved for, the pose reported, and the point the
EE speed cap applies to are one frame. One move per side at a time (a
single-flight busy slot whose release rides a drop guard, so no terminal can
leak it). `move_arm` plans the quietest tier that works: a held-elbow line, a
steered-elbow line, or the guarded servo (a damped resolved-rate law that can
cross singular surfaces a discrete IK walk cannot). Planning runs after
admission (peppy's goal decision is pre-context, so reachability cannot be a
refusal): an accepted goal whose pose no tier reaches, servo rollout
included, completes unsuccessfully at once. Completion is graded on the
commanded motion with a
2x-nominal timeout; results report the measured state and the caller judges
how close it landed (the governor may have held it short, and that is not a
failure of the move machinery).

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
| `types.rs`, `arm_pair.rs` | `ARM_DOF`, `JointVec`, `Side`, the motion-timeout rule, `ArmPair` | shared primitives |

## Parameters and links

See `peppy.json5` for the full commented list. The operational governor
parameters (`d_stop_m`, `d_safe_m`, `collision_governor_enabled`,
`max_ee_velocity_m_s`) are required launcher arguments with no node defaults,
and the commander's `governor_control` stream retunes them live.
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
no-op, so partial deployments and monitors boot cleanly.

## Build, run, test

```sh
# Build into the node stack (never plain cargo for deployment):
peppy node add /path/to/openarm-nodes/openarm_backbone -sb

# Launch the whole stack (sim shown; the backbone and commander pair
# mutually, so cold starts go through a launcher):
peppy stack launch /path/to/launchers-hub/openarm/openarm_v2_teleop_mujoco.json5

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
