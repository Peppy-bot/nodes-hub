# openarm_backbone

The bimanual motion authority. One node sits between whatever leads (the
commander panel or a leader arm rig) and the four followers (two arms, two
grippers), and everything that reaches a follower has passed through one
governed pipeline against one self-collision model, built by `self_collision_model`
from the robot's description, which leaves out the pairs whose collisions the
SRDF disables. The same binary drives
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
 execute_joint_path |  (per arm)    (16 DOF)         |     (answered from bringup)
 move_gripper ----> |                                |
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

1. **Consume the streams of busy arms** - a discrete move wipes its arm's
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
4. **Serve the requests** (`admit_and_serve`) - the stop, the plan checks,
   the grasp measurements and the starts of `execute_joint_path` paths queued
   since the last tick, after the followers are admitted: a plan check, or a
   path that starts, on the tick an arm's follower recovers plans from the
   arm's re-anchored setpoint, where the arm stands, as a `move_arm` goal its
   planner takes that tick does. Each path that starts is handed to its
   judgment, and the shares of each path whose judgment came resume.
5. **Advance the arms** - each planner turns its inputs (streamed command or an
   in-flight move) into a rate-limited candidate, and hands out the arm's
   end-effector Jacobian when the target came from the operator's stream. An
   arm on an `execute_joint_path` path holds its command until the path
   plays, then proposes the next point of its path, at one pace with the
   other arm of a path of both.
6. **Service gripper moves** - `move_gripper` goals chase through the same
   governed configuration as everything else, then hold their target until
   the jaws stand still, or the goal's settle timeout passes, after the
   planned end of the commanded travel, then report where the jaws stand.
7. **Govern** - one `Governor::govern_along` call over the whole 16-DOF step.
   The arms of an `execute_joint_path` path are scaled along their step as a
   whole, and their planners play the path's clock on by the share of the
   step they got.
8. **Publish** - governed setpoints to the follower slots, measured states
   relayed up the leader slots, and the proximity readout at ~20 Hz.

The action handlers admit goals before the loop runs them, and refuse each
goal with a failure kind of the limb_motion and postures contracts. A goal on
a limb whose follower delivers no fresh state, before its first state or after
it stopped reporting, is refused with `resource_unavailable` and `no fresh
follower state to anchor the move from` (`no fresh gripper state to anchor the
move from` for a gripper), and `check_arm_move` gives an error reply of that
kind; until the coordinator has seeded on every follower's first state, every
goal is refused that way. A goal on a limb that received a leader command
within the last 250 ms is refused with `leader_streaming`, so whoever is
streaming the limb keeps it until the stream stops. A posture is refused
while a limb that it moves is being streamed, either arm, and for
`move_to_home` either gripper too, and `check_arm_move` gives an error reply
of the same kind. A goal on a limb that executes a move is refused with
`precondition_failed`, and a goal value the backbone does not take with
`rejected`. Releasing the headset's grip streams the measured pose for a
quarter second, so a posture button pressed within about half a second of
letting go is refused; a later press goes through.

A refusal of an `execute_joint_path` goal of both arms leads with the arm's
name (`right_arm: no fresh follower state to anchor the move from`,
`right_arm: arm is already executing a motion`).

## The governed pipeline

`Governor::govern_along(prev, cand, measured, hands, paths, dt)` runs six
stages, each contractive with respect to the last:

```text
 parse -> limit(speed) -> sense -> limit(tripwire) -> project -> clip
           always on      [------- collision avoidance, toggleable ------]
```

1. **Parse.** A non-finite endpoint is a fault hold, not a step.
2. **Limit (speed).** `DofSpeed` (per-DOF rate bound) and `EeSpeed` (the
   operator's hand-speed cap, applied per arm to stream-driven ticks via the
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
   closing motion per arm until the real gap recovers. It alone feeds the
   collision readout, so a speed cap can never read as a collision event.
5. **Project.** The closing-velocity barrier (a Faverjon-Tournassoud velocity
   damper): remove just enough of the gap-closing component that the clearance
   loses no more than `allowed_closing(d) * dt`, leaving tangential and
   separating motion at full speed. Directional, so it cannot be a per-DOF
   limiter; it runs after them so its guarantee holds on the published step.
6. **Clip.** The exact floor scan. Surface distance is not monotone along a
   joint-space segment, so this walks the realized segment and retracts to the
   furthest point that stays at or above the step floor. A separating arm can
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

| Action | Goal | Refused when (`rejected`, `leader_streaming`, `precondition_failed` for a busy arm) |
|---|---|---|
| `move_arm_joints` | `arm_name`, 7 joint positions (rad), `duration_s`, `settle_tolerances_rad` (one per joint), `settle_timeout_s`, `contact_gap_m` (the limb_motion contract) | non-finite, a duration or timeout outside 0 to 60 s, a joint more than 0.001 rad outside its limits (one closer runs at the limit), a tolerance that is not positive or not one per joint, a negative contact gap, a leader streaming the arm, the arm busy |
| `execute_joint_path` | `arms` (one arm or both, each once), each with its `arm_name`, one or more `waypoints` of 7 joint positions (rad), `expected_start_joint_positions` (empty, or 7 joint positions) and `settle_tolerances_rad` (one per joint); `duration_s`, `settle_timeout_s`, `contact_gap_m` (the limb_motion contract) | an unknown or repeated arm name, no waypoint, arms with different numbers of waypoints, an array that does not hold 7 values, a non-finite value, a waypoint joint more than 0.001 rad outside its arm's joint limits (one closer runs at the limit), a tolerance that is not positive, a duration or timeout outside 0 to 60 s, a negative contact gap, a leader streaming a named arm, a named arm busy |
| `move_arm` | `arm_name`, robot-frame position (m) and optional quaternion `[x, y, z, w]`, `duration_s`, the settle tolerances (position m, orientation rad), `settle_timeout_s`, `contact_gap_m` (the limb_motion contract) | non-finite, degenerate quaternion, a duration or timeout outside 0 to 60 s, a tolerance that is not positive, a negative contact gap, a leader streaming the arm, the arm busy |
| `move_gripper` | `gripper_name`, opening fraction in [0, 1], `max_effort`, `settle_timeout_s` (the limb_motion contract) | non-finite, out of range, negative effort cap, a timeout outside 0 to 60 s, a leader streaming the gripper, the gripper busy |
| `move_to_ready` | `duration_s` (the postures contract; both arms to the Ready posture, and no command to the grippers) | non-finite, a duration outside 0 to 60 s, a leader streaming either arm, either arm busy |
| `move_to_home` | `duration_s`, the time of the arm move alone (the postures contract; both grippers closed, then both arms to the Home rest) | non-finite, a duration outside 0 to 60 s, a leader streaming either arm or either gripper, either arm or either gripper busy |

A refusal for the joint limits names the joint, its position and its limits.
A target or a waypoint up to 0.001 rad outside a limit runs at that limit: a
measured pose sent back as a target lies that close, such as the elbow of a
straight arm, which settles a hair on either side of its lower limit.

`arm_name`/`gripper_name`: `left_arm`/`right_arm` and `left_gripper`/`right_gripper`,
the names the robot's description gives its arms and grippers, an unknown
name refused with `rejected` naming the robot's arms or grippers. Every pose here, commanded or
reported, is in the robot frame the contracts define, which for this robot
is the root link of the URDF: the bottom of the pedestal's base plate on the
column axis, `+x` to the robot's front, `+y` to its left, `+z` up. It is the
gripper's **grasp point** (midway between the pads on the jaw closing axis,
`+z` out of the gripper), not the wrist flange they mount on: the description
names it as each arm's grasp link, and each arm's model mounts its tool there,
so the pose solved for, the pose reported, and the point the EE speed cap
applies to are one frame. One move per arm at a time (a
single-flight move slot whose release rides a drop guard, held by the move,
or by the coordinator for a goal it ends before the move starts, so every
terminal frees it). `move_arm` plans a straight line (`srs_model::plan_line`): a
walk along it in MoveIt's steps and within its Cartesian precision of 1 mm and 0.01 rad,
each step one descent of the KDL search from the state before. The walk plays as a joint
trajectory, timed from rest to rest within the joint velocity and acceleration
limits (`control_core::path_timing`, the time-optimal timing of a path through
waypoints), as every other planned motion is. It lands on the goal pose. A goal that
leaves its orientation out is a position alone: the line keeps the turn the hand
starts with, or the goal has no line. The settle tolerances judge arrival, and no plan
reads them. Planning runs after admission (peppy's goal decision is pre-context, so
reachability cannot be a refusal): an accepted goal that no line reaches fails at once
with `unreachable` and `goal pose not planned: ...`, a message
that names `plan_joint_path` and `execute_joint_path`, which reach what a line
cannot; a line, a joint move or a posture move the limits cannot run within 60 s
fails with `rejected`.
From the planned end of its
trajectory on, an arm move keeps commanding the target and judges the
measured arm against it: a joint move completes once every joint stands
within its `settle_tolerances_rad` entry of its target, a Cartesian move once
the measured grasp point stands within `settle_position_tolerance_m` and
`settle_orientation_tolerance_rad` of the commanded pose. A move whose arm
still stands outside its tolerance `settle_timeout_s` after the planned end
fails with `timeout` and the remaining error, `still 0.031 rad from the target
at joint 4, 2.0 s after the planned end` or `still 8.0 mm / 0.02 rad from the
target, 2.0 s after the planned end`. A failed goal gives no result. A goal
that completes or ends cancelled reports the measured state at that moment; a
cancel of its caller ends it for the reason `cancelled by its caller`.

A `move_arm_joints`, `execute_joint_path` or `move_arm` goal with
`contact_gap_m` above 0 also ends on contact. On each tick of the move the
arm's planner compares two grasp points: the one of the joints the arm
measures, and the one of the joints the arm was commanded to last
(`Planner::stands_on_contact`, one check for the three moves). An arm that an
object holds on its way falls behind its command. Once the two grasp points
stand farther apart than the gap, the move completes with `ended_on_contact`
true, and its result gives where the arm stands. The ends of a move come in
the order of the limb_motion contract (`settle::verdict`): a cancel or the
stop, a settle, which completes the move with `ended_on_contact` false, a
contact, and the settle timeout. A gap of 0 is no contact check.

What the arm is commanded to after a move depends on how the move ended
(`Planner::end_move`):

| End of the move | The arm's command from then on |
|---|---|
| Settled | the target of the move |
| Ended on contact, failed with `timeout`, cancelled, or ended by `stop`, once the planner had started the move | it moves to the joints the arm measures on that tick, clamped into its joint limits, and holds them: an arm that an object holds short of its command stops pushing on it |
| Ended with no fresh measurement of the arm (its follower stopped reporting) | the command it was last governed to |
| Refused, failed before the planner started the move, or ended while a joint path waited for its judgment | the command as it was |

The command moves to the measured joints as every command moves
(`Planner::hold_where_it_stands`): a step of the joint velocity limits each
tick, each step judged by the governor. The first step goes out on the tick
the move ends, and for a share of a joint path, whose end is judged once its
step is governed, on the tick after. The arm stands still where the command
goes, so these steps are published with no velocity (`published_velocity`).
Where the collision guard slows or holds those steps, the command stops
where the guard leaves it once the steps the way takes at the velocity
limits have run out, so a command the guard holds at the stop distance stays
there, and no step toward that pose comes later. A move or a leader's stream
that takes the command ends the hold; a joint path that waits for its
judgment leaves it standing, and the hold goes on if the path never runs.

A posture move completes once each arm stands
within 0.1 rad of the posture in every joint, fails with `timeout` 2 s after
the planned end otherwise, and reports where each arm stands. When the arm's follower stops reporting, the
move fails with `resource_unavailable` and `the follower stopped reporting`,
and so does a goal that reaches the loop before every follower reported
(`the follower has not reported its first state yet`). A goal cancelled while
the arm has no fresh measurement reports the held setpoint, and its reason
ends with `; the result comes from the joints commanded last`.

`move_gripper` reports where the jaws stand. The commanded opening chases the
target at the opening rate `max_gripper_rate_frac_s` through the governor, and
the planned end is the tick that travel comes to rest: the commanded opening
stands on the target, or the collision guard holds it. A hold the guard lets
go of ends that rest, and the travel goes on to the next. The jaws lag the
command: the gripper motor's own position loop drives them at a speed in
proportion to the distance still to go (measured on an openarm_v2, a time
constant of about 0.18 s, so they stand still within about 1 s of any move).
From the planned end on, the move keeps commanding the target and holding the
arm's move slot until the measured opening has stood still: every opening
the follower delivered over 0.25 s lies within 0.01 of the travel of the
others, the band and window the limb_motion contract states. Jaws an object
or the effort cap stops short of the target stand still there too. At the latest `settle_timeout_s` after the planned end the move
completes wherever the jaws stand: a gripper move never fails on its settle,
since a grasp is expected to stop short. `final_opening` is the opening
measured when the move ended and `action_time` runs to that moment.

| End of the move | The goal |
|---|---|
| The jaws stood still at or after the planned end, on the target or stopped short of it by an object, the effort cap or the collision guard | completes, `settled` true |
| `settle_timeout_s` after the planned end, before the jaws stood still | completes, `settled` false |
| The follower stops reporting during the move | fails with `resource_unavailable` and `the follower stopped reporting`, no result |
| Cancelled before it completes | ends cancelled for the reason of the cancel, `settled` false |

The caller judges a grasp from `final_opening`. After the move, the backbone
sends the gripper nothing until the next gripper move or a leader's gripper
stream. Thus the gripper
keeps the move's target opening and `max_effort` (relayed unchanged), and
goes on squeezing what it holds.

`execute_joint_path` moves the arms its `arms` names along one path: from
the joints each named arm measures when the path starts to play, within
0.001 rad (below), clamped into its joint limits, whether or not the goal
gives an expected start, through each of their waypoints in turn, the arms
passing their k-th waypoints together, each inner corner rounded on an arc
within 0.1 of the corner as MoveIt's time-optimal trajectory generation
blends it, stopping where the path turns back and where every named arm's
waypoint lies within 0.001 rad of its waypoint before in every joint
(`control_core::path_timing::JointTrajectory`). `move_arm_joints`,
`move_arm` and the postures start from the held setpoint. The path runs
through both arms' joints, an arm it does not name held at its setpoint, so
a path of one arm and a path of both are one trajectory: where one arm
repeats its waypoint and the other does not, the path runs on, the other arm
moving. Each goal waits until the coordinator takes it. The coordinator
takes the goal on every named arm at once, or fails it before any arm moves
(`planner::start_path`), with `precondition_failed` when a named arm runs
another move, or stands farther from its `expected_start_joint_positions`,
when it gives one, than the `start_tolerance_rad` parameter in some joint,
the message naming the arm and the joint, counted from 1 (`left_arm stands
0.200 rad from expected_start_joint_positions at joint 2, past the 0.010 rad
start tolerance; plan the path from where the arm stands`). A stop ends a
goal the coordinator has not taken, as it ends the goals queued for an arm.

From then on, the named arms' commands hold where they stand, their slots
claimed, while the path is judged beside the coordination loop
(`path_checks.rs`) from where each named arm measures when the coordinator
takes the goal, clamped into its joint limits, against the guard as the
governor holds it and the other limbs where they stand then. A long path
takes the judgment many control ticks, and the loop ticks on meanwhile. The
path fails, running none of it:

- with `rejected` when the joint velocity and acceleration caps cannot run
  the path within 60 s, the message giving the time they take (`the joint
  velocity and acceleration caps take 80.0 s to run this path, past the 60 s
  a move may take; split it into paths of fewer waypoints`);
- with `unreachable` when the path checker (`governor::PathChecker`), which
  walks the whole path from its start on a copy of the governor's collision
  model, the named arms where the path puts them and the other limbs where
  they stand, each probe against the floor the guard holds a step at from
  where the step starts, finds that the floor scan would hold an arm short,
  or that the path closes the gap along it from a probe at or inside the
  stop distance, where the guard's closing barrier holds every closing step
  however short (`the robot starts with two of its links 5.0 mm apart,
  inside the collision guard's stop distance, where the guard holds every
  step that closes the gap, and the path closes the gap between
  openarm_right_ee_base_link and openarm_left_ee_link1 where they stand 5.0
  mm apart, so the guard would hold the arm short`). An arm the guard holds
  at the stop distance can measure inside it, standing off its command
  toward the other body; from such a start the floor is the clearance
  there, so the path may open the gap and fails where it closes it. Inside
  a contact the check allows none of the loss the guard's recovery rate
  allows each tick, so it refuses a path that starts in a contact and sweeps
  deeper on its way out, an escape the guard runs, in words that say so
  (`the robot starts 1.2 mm deep in contact with itself, where the collision
  guard lets a step lose clearance at up to 200 mm/s while the arm escapes
  and the check lets it lose none, ...`). The check reads the clearance at
  its probes, at most 0.01 rad apart along the path and at each rest: inside
  the stop distance the floor is the clearance itself, so a probe vouches
  for no stretch after it, and a stretch that closes the gap strictly
  between two probes, both of which open it, the later no closer than the
  earlier, passes the check, and the guard holds it there.

A path that fails before it plays leaves each named arm's command where it
held, and runs none of the path; once the path plays, a failure holds each
arm where it stands.

A cancel, the stop or a follower gone stale ends the goal while it waits
for the judgment, as it ends a path that runs, and the loop stops the
judgment at its next measured probe, which frees the checker for the next
path: on the stop's own tick, and on the tick after a cancel or a stale
follower ends the goal. A goal the stop ends on the tick the coordinator
takes it is never judged. Once a judgment lets the path run, each named
arm's run start is the joints it measures on the first tick with every
named arm measured, clamped into its joint limits. An arm that then
measures farther than `start_tolerance_rad` from its
`expected_start_joint_positions`, when the goal gives one, fails the goal
with `precondition_failed` in the words of the coordinator's first check,
and one whose run start stands that far from where it measured when the
goal was taken fails it with `precondition_failed` too (`left_arm stands
0.020 rad at joint 2 from where it stood when the path was taken, past the
0.010 rad start tolerance; send the path again`). A run start farther
than 0.001 rad from the start the path was judged from, in some joint, has
the path judged again from the run starts, the commands still holding, so
the path that plays is the path a judgment passed, with its `unreachable`
and `rejected` refusals; an arm still that far once the judgments span 0.2 s
fails the goal with `precondition_failed` (`left_arm still moved 0.008 rad at
joint 1 while its path was judged N times; let the arm settle and send the path
again`). All of these come before the path plays. Otherwise each named
arm's command moves onto the path's start, within 0.001 rad of where the
arm stands, as a recovered follower's moves onto its measured pose, and
each named arm's planner follows its arm's share of the judged path from
there. Thus an arm still settling while the path is judged runs the path on
from where it stands, and the path that runs starts within
`start_tolerance_rad` of where the goal was taken, and of the expected
start, in every joint. Each tick a share proposes the path's next point,
and the governor scales the step of the path's arms as one: every stage
takes one share of the step across the path's joints, and the floor scan's
separating hold never parts the arms of a path of both. The path's clock
(`control_core::timing::MotionClock`) plays on by that share; on a path of
both arms each arm's share plays its own clone of the path's trajectory,
and both clocks play at one pace (`planner::commit_both`). Thus the arms
stay on the rounded path while the guard slows them, and pass each waypoint
together. A share is judged on each tick from the first at or after its
planned end: once its clock has played to the end and every joint stands
within its `settle_tolerances_rad` entry of the arm's last waypoint, it
completes, and `settle_timeout_s` after the planned end it fails with
`timeout`: naming the joint farthest outside its tolerance of the last
waypoint and by how much (`still 0.120 rad from the target at joint 3, 2.0
s after the planned end`), or, with every joint within its tolerance while
the guard held the path's clock back short of its end, `still short of the
end of its path, 2.0 s after the planned end`. A share that fails ends the
others, on the same tick when a follower goes stale; on a path of both arms
its failure leads with its arm's name (`right_arm: the follower stopped
reporting`), and the goal fails with the failure of the share that failed
first. The goal completes once every named arm settles, and gives each
arm's joints as its share measured them when it ended, in the order of
`arms`, and the time since the path started. A goal with `contact_gap_m`
above 0 ends on contact for every named arm once one arm stands past the
gap: both shares are judged on one reading of the contact of both arms
(`planner::commit_both`), so on that tick that arm's share completes with
`ended_on_contact` true, the other share with `ended_on_contact` false, and
the goal completes. A contact of one arm comes before the settle timeout of
the other. Each arm of a path that played and then ended on contact, failed,
was cancelled or was stopped holds where it stands, as above; a goal that
ends cancelled says `ended_on_contact` false for every arm.

`move_to_ready` and `move_to_home` give each arm's planner a joint move to
the posture. The goal completes once both arms settled at the posture: each
measured joint within 0.1 rad of it, by 2 s after the planned end. An arm
that the governor holds short, or that stops against an object the backbone
does not see, such as a bench in front of the robot, fails its move with
`timeout`. The goal fails with the first failure of an arm's move, its
message led by the arm's name (`left_arm: still 0.470 rad from the target at
joint 6, 2.0 s after the planned end`, `left_arm: the follower stopped
reporting`), and
with `attempt_failed` when an arm's planner does not take or answer its
move.

`move_to_ready` claims both arms and sends no gripper command, so each
gripper goes on as its last command drives it. `move_to_home` claims both
arms and both grippers, and closes both grippers before the arms move. It
refuses for an arm ahead of a gripper, so both postures refuse a robot whose
arms cannot move in the same words. A gripper whose follower reports no
state does not refuse it:

- Each close is a gripper move to the closed opening. It keeps the effort
  cap of the last setpoint that its gripper got, from a `move_gripper` or
  from a leader, so jaws that hold an item keep the effort of their grasp.
  A gripper that got no setpoint since the backbone started closes under
  the effort ceiling of the follower.
- A close ends as a `move_gripper` ends: once the jaws stand still, and at
  the latest 2 s after the planned end of the commanded travel, where they
  stand. Jaws that an item stops close on it, and the arms go home with it.
- The commanded travel of a close has 8 s, so a close takes at most 10 s.
  The operator can set an opening rate (`max_gripper_rate_frac_s`) at which
  the travel takes longer: the close then fails with `timeout`
  (`left_gripper: the commanded travel of the jaws did not end within 8 s:
  the opening rate (max_gripper_rate_frac_s) is too low for the move, or the
  collision guard slows the jaws`).
- A gripper whose follower reports no state when the coordinator takes its
  close gets no setpoint. The goal leaves that gripper as it stands and goes
  on, so a robot with a silent gripper still parks its arms. A follower
  that stops reporting during the close fails it.
- The arms keep their claims through the close, and each planner gets its
  move once both closes ended. `duration_s` is the time of the arm move
  alone.
- A cancel or a `stop` during the close ends the goal cancelled, and a
  close that fails ends the goal with its failure, its message led by the
  name of the gripper (`left_gripper: the follower stopped reporting`). In
  both cases the arms do not move.

At Home, every pair of the robot's bodies stands farther apart than the
governor's band reaches (`d_safe_m`), on v1 and v2, with the grippers at any
opening. So the arms arrive at Home with whatever the grippers hold, an item
that holds both jaws fully open included.

A goal that completes or ends cancelled gives the pose of each arm:

- `arm_names`: `left_arm` and `right_arm`, in the order of limb_state.
- `positions`: 3 values per arm, in metres, in the order of `arm_names`.
- `orientations`: 4 values per arm, `[x, y, z, w]`, in the same order.

Each pose is the grasp point of the arm in the robot frame. The
coordinator measures both arms when the posture goal completes, after the
moves of both arms ended. limb_state gives the same pose for the same
joints. After a cancel or a stop, the pose is where the arm stood on that
tick, and the arm's command moves there and holds it.

When an arm has no fresh measurement when the goal ends, the three arrays
are empty, and the log says why: the follower of the arm stopped reporting,
so its last measurement is stale, or the coordinator did not answer.

### Services of limb_motion

`stop` ends every planned move in flight, whoever started it: the arm moves
through their planners, the gripper moves through their terminal, the
posture moves through the limbs whose shares are in flight (both grippers
during the close of `move_to_home`, both arms otherwise), and the goals
admitted but not started yet. A goal that holds a limb and that the
coordinator has not taken when the stop comes ends when the coordinator
takes it: the stop marks the move slot of the limb. This ends a goal on its
way from its action to the coordinator, and a `move_to_home` goal between
the end of its close and the moves of its arms. Each goal ends cancelled for
the reason `stopped: <reason>` (`stopped` for an empty reason), a posture
goal too. Each arm whose move the stop ends holds where it stands, as after
a cancel: its command moves to the joints it measures on the stop's tick. An
arm with no fresh measurement, and an arm whose joint path had not started
to play, keeps the command it was last governed to, and so does each gripper.
The answer's `stopped` names the limbs whose move was in flight, empty when
none was. The stop does not latch: a goal admitted after it runs.
A leader's streamed setpoints are not stopped.
A coordinator that does not answer cannot carry the stop out: the stop then
gives an error reply with `attempt_failed`.

`check_arm_move` takes the pose and the duration of a
`move_arm` goal and answers whether a plan reaches the pose from the arm's held
setpoint, without moving: `reachable` and the time the move would take, or
`reachable` false and the `reason` that the goal's failure gives (`goal pose not
planned: ...`), with the same validation and the same planner as the goal. A
request the goal would refuse gets an error reply of the same kind and words:
`rejected` for a value the goal does not take, `resource_unavailable`,
`leader_streaming`, and `precondition_failed` while the arm executes a move.
It checks no collision with the robot or the room.

Both are answered by the coordinator on the tick after the request, so
they wait behind the readiness gate with the moves; a stop during the seed
wait stops nothing and a check then gets an error reply with
`resource_unavailable`, as a goal fails then.

### camera_mounts

`get_camera_poses` answers where the robot's design carries each
camera, in the robot frame, as the pose of the camera's colour optical frame
(`+x` to the right of the image, `+y` down it, `+z` along the view): each
camera's optical frame in the description's URDF, which the simulation
reads too, so they are exact in a simulation and nominal on hardware. A camera fixed to the base
(v2's `chest`) has one pose; a camera an arm carries (v2's `wrist_left` and
`wrist_right`, which hang off the link the grasp point hangs off) is composed
with that arm's grasp pose from the last `limb_states` snapshot, and the
answer carries that snapshot's stamp. Served from bringup, an error reply
with `precondition_failed` and `the robot has not measured its joints yet`
until a snapshot exists; a v1 robot answers with no camera. A description that mounts a camera on
a link a moving joint carries, other than the grasp point's, stops bringup.

### workspace

`describe_workspace` and `check_positions` answer where the robot can work
from its design alone, in the robot frame, knowing nothing of the room: a
surface is a flat, level plane at the height asked, the top of a block under
it that stands from the front of the robot's body to the far end of the
measured grid, and reaches as far to each side of the robot as it reaches
ahead (`workspace_core::surface_block`), with nothing else in the way; a
point `check_positions` names stands on no surface.
A point is reachable, by `workspace_core`'s reach rule
(`workspace_core::reach`), when an arm brings its grasp point within 1 cm
of it in one of the 16 grasp orientations `workspace_core` lists (the
gripper pointing straight down or straight forward, each at 8 rolls),
solved by the arm's closed-form IK seeded at Ready, in a pose of which no
collision piece of the arm overlaps the block of the surface: a grasp
within 1 cm of an edge of the arm's reach, outside or inside it, is solved
moved into that reach by less than 1 cm, in the same orientation.
The arm whose base stands nearer the point is tried first and named. A
point out of reach carries how far the closest arm stops its grasp point
short of it in any orientation, its joint limits aside: at most 0.01 when
the grasp point gets within 1 cm of it, but in no grasp orientation, or on
a surface only in poses that overlap its block, which the point's message
words as `no arm reaches it with its gripper pointing down or forward`. A
point is in view when it lies inside the perception camera's
field of view, a pixel of its colour image showing it through the lens the
camera gives, and its depth, by the depth model the camera gives (`z`, the
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
| vacant | no view check: each point's `view` is `no_camera`, and `view` is absent |
| linked, answering | the view is checked through the camera's own intrinsics and depth stream |
| linked, colour intrinsics an error reply, not answered in time, not a pinhole model, or of a lens `camera_geometry:v1` does not name | error reply, `precondition_failed`: `cannot read the colour intrinsics of the chest camera, the perception camera: <reason>` |
| linked, depth intrinsics an error reply other than `rejected`, not answered in time, not a depth range, or of a depth model `camera_geometry:v1` does not name | error reply, `precondition_failed`: `cannot read the depth intrinsics of the chest camera, the perception camera: <reason>` |
| linked, depth intrinsics an error reply with `rejected` (a camera that gives no depth) | error reply, `precondition_failed`: `the camera linked as the chest camera, the perception camera, gives no depth: <reason>` |

The `<reason>` of a camera's error reply is its message. The grid, the
limits, the verdicts, the largest workable rectangle and the message of each
point are `workspace_core`'s, so these answers and the simulation's read
alike. The parsing of a request and its refusals (below), the reach memo and
the composition of each answer are `workspace_core::design`'s. The error
replies of the camera's geometry, in the table above, are the backbone's
own.

`describe_workspace` measures the surface at its height to the millimetre,
each grid point's reach at a target 4 cm above it, and keeps the reach grids
of the last 32 heights it measured (the one stored first goes past that), so
a repeated height answers from the stored grid. A height that is not a
finite number or that is more than 1000 m from the robot's base point, an
empty position list, one that does not split into points of 3 values, a
value that is not a finite number, and a coordinate more than 1000 m from
the robot's base point get an error reply with `rejected`. Both services read nothing of the arms
and move nothing, so they answer from bringup, ahead of the readiness gate.
The robot's own body is not checked for hiding a point from the camera.

## Module map

| Module | Owns | Why it lives here |
|---|---|---|
| `main.rs` | tracing, then the node runtime running `setup`, and the exit with the fault that ended the node | all node logic lives in the library, so the integration tests boot the same entry point |
| `node.rs` | bringup: params in `setup`, then, once the robot's initializer names the model, models, channels, task supervision | first task exit is fatal; the daemon restarts a clean process |
| `identity.rs` | the model the robot's initializer names, held to this backbone's arms, then the digest of the description it serves, held to the one this backbone was built with | the backbone runs only a model it drives, from the description it was built with |
| `startup.rs` | the robot_initializer gate | nothing streams before the robot is ready |
| `streams.rs` | every subscription + parse-at-the-boundary types (`GripperCommand`, `ArmState`, `GripperState`) | one receive policy (`subscribe_pair` + `accept`); a malformed message is dropped with a reason, never driven |
| `upstream.rs` | `UpstreamMode` (which upstream slot kind is followed) + `Upstream` (the parsed joint or pose command) | one command authority per arm is unrepresentable, not checked per tick |
| `publish.rs` | every publisher (`Publishers`), one stamp/build/publish/log path | peppy vocabulary: a publisher on a slot; "wire" means the transport encoding only |
| `coordinator.rs` | the tick, each gripper's target and chase, the upstream relay | IO orchestration; its seam with the safety core is exactly one `govern_along` call, which is why the governor is not folded into it |
| `gripper_move.rs` | gripper move execution: a `move_gripper` goal, or the close of a gripper for a `move_to_home` goal, admitted into a free arm, the effort cap it relays, the terminal it gets once the jaws stand still, its settle timeout passes or its travel timeout runs out, and the end of its goal or the report to its posture goal | the gripper analog of `planner.rs` + `settle.rs`; the coordinator drives it each tick |
| `liveness.rs` | follower admission (`Live` / `Reanchor` / `Stale`) | delivery-cadence policy for the coordinator, independent of any message type |
| `move_slot.rs` | `MoveSlot`, each limb's single-flight move slot (claim, release, the `SlotRelease` a move holds, and the mark a stop leaves on a slot whose goal the coordinator has not taken), and the refusals of a busy limb | one owner for the slot the action handlers claim, and release when a goal never reaches the coordinator, and the planners and the coordinator release when they end a goal or its move |
| `planner.rs` | per-arm mode machine (Follow / joint move / Cartesian move / share of an `execute_joint_path` path) -> one rate-limited candidate + the stream-tick Jacobian | knows one arm only, but for the path its share follows; never the collision model |
| `planner/path.rs` | the shares of an `execute_joint_path` path: its start on the arms it names (`start_path`), its judgment (`judge_path`, `resume_judged_path`), the pace a committed step plays its clock at, one pace for the two clocks of a path of both arms (`commit_both`) | reaches the collision model only through the `PathGuard` `judge_path` is handed |
| `planner/stream_smoothing.rs` | the smoothing of a streamed pose's steps: one Butterworth per joint at MoveIt Servo's default cutoff for the control period | a smoother can overshoot its input, so the planner holds its output to the velocity limit |
| `path_checks.rs` | the judgments of `execute_joint_path` paths on a blocking thread, and the resumption of their shares on the tick each judgment comes | the loop never waits on a judgment |
| `settle.rs` | `Settle`, `PoseTolerance` (the slack a Cartesian move settles within), `ContactGap` (the gap a goal names for a contact), the verdict an arm move gets on each tick, and how far a joint target or a grasp point still stands from its target | one rule for the joint move, the joint path and the Cartesian move |
| `governor/mod.rs` | `GovState`, the pipeline, the runtime controls, disposition and readout | the only mutable resource is the collision model |
| `governor/sense.rs` | the pre-projection model read (`Sensed`) | limiters and the projection decide on one snapshot |
| `governor/model.rs` | `ConfiguredModel`, the only door to the collision model | every query takes the whole configuration; stale-placement reads are unrepresentable |
| `governor/limiters/` | the `Limiter` trait, one module per limiter, and `allowance.rs` (the `Allowance`/`Limits` currency they speak) | everything expressible as a per-DOF fraction lives together |
| `governor/path_check.rs` | `PathChecker`, a joint path walked against the step floor before the move starts on a copy of the governor's collision model, and `GuardAt`, the guard it is walked against | the floor scan's resolution and floor, over a whole path |
| `governor/barrier.rs` | the projection and the floor scan | the two stages that are not per-DOF fractions |
| `actions/` | the actions and services a caller reaches: move goal admission (validate + claim), the posture and `execute_joint_path` goals run to their one result, and the stop and plan-check services that ask the coordinator; each action's goal fields are parsed off the wire in its module, the times and tolerances every move goal names in `actions/mod.rs` | execution belongs to the planner/coordinator that owns the state |
| `failure.rs` | the failure kinds of the moves and of `check_arm_move`, each given to every generated member, and the failure of a follower that stopped reporting | one failure crosses from a posture share to its posture goal |
| `workspace.rs` | the OpenArm's part of the workspace answer from the design: its arms as `workspace_core`'s reach rule asks them, the perception camera, the per-height reach memo it keeps, the parsing of the camera's geometry; `workspace_core::design` parses the request and composes the answer | pure: no messaging, so every answer is unit-tested without a node |
| `camera_mounts.rs` | the camera_mounts contract: each camera resolved once at bringup into the frame that carries it, then composed with the grasp poses measured last | a mount the backbone cannot place stops bringup |
| `workspace_service.rs` | the two workspace services and the read of the linked camera's geometry | the thin edge between the generated handlers and `workspace.rs` |
| `types.rs` | the limb tables, the world-pose wire codec, every arm's joints in one vector | shared primitives; `ArmId` and `PerArm` come from `robot_description`, `ARM_DOF` and `JointVec` from `srs_model` |
| `robot.rs` | the robot this node is built for (`ARMS` arms of `ARM_DOF` joints, each arm's and gripper's name, `LIMBS`), a request's arm or gripper by name, the postures of a named state | a model of another shape ends the node, naming the models of this shape |
| `arm_order.rs` | `ArmOrder`, the arms a request names in its order, on `control_core::requested_arms` | one reading of a request's `arms` for `execute_joint_path` |

## Parameters and links

The backbone reads the robot's model from the robot's initializer on its
`robot_identity` link: it asks once its setup has returned, since the
initializer may start after it, and asks again each second until the
initializer answers. A model whose arms are not `left_arm` and `right_arm` of
seven joints, each with its gripper, ends the node with an error that names the
served model and the models of this shape, as does an initializer that serves
another build of the model's description.

See `peppy.json5` for the full commented list. The operational governor
parameters (`d_stop_m`, `d_safe_m`, `collision_governor_enabled`,
`max_stream_ee_velocity_m_s`) are required launcher arguments with no node defaults,
and the commander's `governor_control` stream retunes them live.
`follower_state_rate_hz` is required too: the rate the followers deliver
measured state at, of which four silent periods freeze the limb, and under
three quarters of which draws a warning. A launcher declares what its followers
deliver: the real arms deliver at their `state_rate_hz`, and Isaac once per rendered
frame.
`upstream_mode` is likewise required: `"joints"` follows the joint_link
leader slots, `"pose"` the pose_link ones, and only the named kind is
subscribed. Link the leader into the slots that kind names, or nothing it
streams is read and the arms never move.
`max_stream_ee_velocity_m_s` caps the hand on every stream and on nothing else:
`move_arm`, `move_arm_joints`, the postures and `execute_joint_path` run under
the joint limits and the goal's duration.
`start_tolerance_rad`, required too, is how far (rad, each joint) an arm's
measured joints may stand from the start of an `execute_joint_path` path:
from the goal's `expected_start_joint_positions` when the coordinator takes
the goal and again at the path's run start, where the arm measures once the
path is judged; and at the run start from where it measured when the
coordinator took the goal, both clamped into its joint limits. Set it to the
most an arm may stand off the start its caller planned from, or move while
its path is judged. All ten pairing slots (six toward the leading node, four
toward the followers) are optional and established by the launcher: each one
either names a peer, from this instance's `links` or the peer's own, or is
declared `{ vacant: "<why>" }`, and a slot left unmentioned by both ends
fails launch validation. Publishing on a vacant slot is a legal no-op, so
partial deployments and monitors boot cleanly. The `perception_geometry`
dependency slot is optional too: bind it to the camera_geometry provider of
the perception camera (`chest` on v2) for the workspace answers to check the
view, or leave it vacant for reach alone.

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
`MAX_PROBE_ARC_RAD`, giving `v_max * dt / MAX_PROBE_ARC_RAD` probes per scan,
floored by `SEGMENT_SAMPLES_MIN`. The work per second is about the same at any
rate; the rate only decides whether it arrives in one indivisible lump that must
fit inside one period, or spread thin. Sweeping the rate against a jog at the
J1/J2 limit (Jetson, 20 s per rate):

| rate    | budget | p50     | p95     | p99     | max      | over budget  |
| ------- | ------ | ------- | ------- | ------- | -------- | ------------ |
| 100 Hz  | 10 ms  | 0.34 ms | 1.46 ms | 3.00 ms | 10.30 ms | 1 / 2000     |
| 250 Hz  | 4 ms   | 0.31 ms | 2.17 ms | 2.19 ms | 2.98 ms  | 0 / 5000     |
| 500 Hz  | 2 ms   | 0.31 ms | 0.97 ms | 1.02 ms | 2.53 ms  | 7 / 10000    |
| 1000 Hz | 1 ms   | 0.31 ms | 0.97 ms | 0.98 ms | 2.40 ms  | 178 / 20000  |

Per-tick cost against jog speed, stated in rad/s because that is the
rate-independent quantity (measured on a Jetson at 100 Hz over a 4000-tick walk):

| jog         | per tick  | p50     | p95     | p99     | max      |
| ----------- | --------- | ------- | ------- | ------- | -------- |
| 2.0 rad/s   | 0.020 rad | 0.50 ms | 1.27 ms | 1.78 ms | 3.08 ms  |
| 5.0 rad/s   | 0.050 rad | 0.40 ms | 1.01 ms | 1.34 ms | 3.77 ms  |
| 16.75 rad/s | 0.168 rad | 0.38 ms | 1.05 ms | 2.82 ms | 10.29 ms |

The last row is the J1/J2 velocity limit (J3/J4 are clamped lower by `DofSpeed`
before the scan sees them). A live 
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
