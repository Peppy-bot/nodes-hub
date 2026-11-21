# botshare_backbone

The robot's backbone of a differential mobile base alone: a base on two driven
wheels, left and right. One node sits between whatever drives the base (its
leader on drive_link, or a client of base_motion) and the wheels, and every
velocity that reaches the wheels has passed through base_state's limiter at
describe_base's limits. The node opens no device: its `base` link pairs with
whatever follows joint_link for the wheels, Waldo's `bases` slot in
simulation, and the launcher decides what pairs there.

The node asks the robot's initializer for the robot's model on
robot_identity's `get_identity`, and for the digest of the description it
serves on `get_description`, as openarm_backbone does (`identity.rs`): the
model's URDF in robot_description gives the base (`model.rs`), and the node
holds the served digest to the one of the bundled description it was built
with. It answers `describe_base` from then, and runs its goals, its streams
and the robot stop once the initializer reports the robot ready on
robot_ready's `is_ready` (`startup.rs`). The URDF's root link is the robot
frame, base_footprint. The wheels are the URDF's two continuous joints, each
on a link that fixed joints alone join to its root link, each turning about
+y, so a positive turn rolls the base forward; the wheel separation is the
distance between the two joints' origins, on one axle whose midpoint stands at
the root link's origin, and a wheel's radius is the radius of its link's one
collision cylinder about the joint's axis. The outline is the rectangle on the
floor that holds every collision element of every link. A model whose URDF
gives no such base, or whose root link is not base_footprint, ends the node
naming why, and so does another description of the model than the bundled one.
The node senses no state of the base's drives, so its mode is standing or
moving whenever the wheels report, both modes that take velocity.

## The robot's tree

The node holds the robot's tree (control_core's `RobotTree`, `cache_window_s`
long): the messages of the publishers bound to its `other_transforms` link,
the robot's initializer with the model's fixed joints, and the transforms it
publishes itself, odom to base_footprint and its wheel joints, each taken into
the tree as it publishes them (`follow.rs` follows the others, and
`control.rs` takes its own). Each goal and each setpoint is placed by the
tree, waiting up to `lookup_timeout_s` on the node's clock for it
(control_core's `Placer`): a move_base target in odom and in the robot frame,
where its frame stands at the latest time or, at a timestamp, where it stood
then, through the root of its frame's chain; a drive_base velocity and each
drive_link setpoint in the robot frame, by the place of their frame there at
their own time, so a velocity in the robot frame stands in it at any time.

```text
   drive (drive_link, follower, one leader)               base_motion: drive_base, move_base
   drive_setpoints in, drive_states out                   robot_stop: stop
                 |                                                  |
                 v                                                  v
          +--------------------------------------------------------+ --> base_states (base_state)
          |  control loop: follower, then control_core's           | --> describe_base
          |  limiter -> differential mix ->                        | --> transforms (transforms)
          |  wheel setpoint, and the odometry                      |
          +--------------------------------------------------------+
                 |                                        ^
                 v   joint_setpoints [left, right]        |   joint_states [left, right]
   base (joint_link, leader): what follows joint_link for the wheels, Waldo's `bases` slot in simulation
```

## The step

`control::run` owns the loop. It takes every event in arrival order: a wheel
measurement, a leader's setpoint, an accepted goal and a robot stop. After
each, and at each poll of one step of wall time, it reads the node's clock
(`peppygen::clock`, the simulation's clock in a simulation) and runs every step
that is due. Steps fall on whole multiples of `1 / smoothing_frequency_hz` of
that clock, so a slow or paused simulation steps on its own time, and a loop
that catches up several steps runs each at its own time. Each step, in order:

1. **The wheels**: the step judges them at its time. They are silent before
   their first measurement and while their newest measurement, by its stamp,
   is older than base_state's 0.5 s.
2. **The setpoint**: while the wheels report, a goal in flight hands the
   limiter the velocity it wants this step, and a goal that has its ending
   hands it zero. While they are silent, a goal in flight fails at once with
   `resource_unavailable`, and the limiter takes a zero setpoint already past
   drive_link's 0.5 s, so the leader drives no base whose wheels are silent.
3. **The limiter**: control_core's `Limiter` moves the executed
   velocity toward the setpoint by at most an acceleration limit over the
   step, the axes scaled together so the base keeps the curvature it was asked
   for, and brings it to rest at the deceleration limits once the newest
   setpoint is more than 0.5 s old. The leader's setpoint counts from when it
   arrived. Once the base rests with no live setpoint, the limiter gives
   nothing, and the step sends nothing.
4. **The wheel setpoint**: control_core's differential mix turns the
   velocity into the two wheel rates, and its `WheelSetpoint::next` gives
   each wheel's setpoint on the measured angles, its anchor bound one step of
   travel at the fastest wheel rate the base's limits allow: a base held by a
   wall winds up no more than that travel, and a base on a slope comes to
   rest. During a silence the setpoints keep coming, each continuing from the
   last, and bring the base to rest at its deceleration limits. The first
   setpoint of an odometry epoch, and the first after a silence, take the
   measured angles. Each epoch starts with the wheels held where they stand:
   at its first measurement the node sends a setpoint of rate 0 at the
   measured angles, before any goal, so a follower that still runs the last
   setpoint of a node that ended stops, and the limiter starts over at rest,
   so a goal or the leader's next setpoint moves the base from rest at the
   acceleration limits, and the base comes to rest from no earlier goal.
5. **The reports**: see below.
6. **The goal's end**: a goal that completes or is cancelled ends once the
   base rests: nothing is commanded and base_state's rest holds. A goal past
   its settle fails at once with `timeout`, and the limiter brings the base to
   rest.

A clock that reads earlier than the last step the loop ran, as a restarted
simulation's clock does, starts the base's timing over: a goal in flight fails
with `resource_unavailable`, its message `the base's clock went back`, the
steps fall on the grid of the new time, the limiter brings the base to rest
from its last command, the leader's newest setpoint and the robot stop, when
they came after the new time, take it as their time, so the stream and the
hold last on from it, and an odometry epoch whose last measurement is stamped
after the new time ends. A goal admitted at a later time than the clock reads
when it reaches the loop fails the same way, and so does a goal placed by an
odometry that started over before the loop takes it, its message `the base's
odometry started over`.

## The follower

The node follows its one leader of drive_link by the arrival of each
setpoint on the node's clock (`follower.rs`). The leader streams from each
setpoint that arrives until more than 0.5 s pass with no setpoint. The base
follows the leader's setpoints while no hold keeps the base from it; a
setpoint that arrives during a hold is dropped, and counts for every 0.5 s
rule. Three holds keep the base from the leader, and `drive_states` names the
first that holds in its `follow_state`:

- `base_mode`: the wheels' silence, during which the base comes to rest. When
  the leader streamed as the silence started, the hold lasts on after it ends
  until more than 0.5 s pass with no setpoint.
- `robot_stop`: a robot stop, from the stop until more than 0.5 s pass, from
  the stop on, with no setpoint.
- `follower_goal`: a base_motion goal while it runs, from its admission to
  its end.

`following` otherwise. A setpoint is dropped before any of this, and counts
for no rule, when a value is not finite, when the robot's tree cannot place
it, and when, placed in the robot frame, its linear_velocity's z or its
angular_velocity's x or y lies farther than 1e-9 from 0, as drive_link counts
such a setpoint as no setpoint (`streams.rs`).

## The robot stop

robot_stop's `stop` is the robot stop. It cancels the goal in flight with the
reason `stopped: <reason>` (`stopped` for an empty reason), holds the base
against the leader, and brings the base to rest. The robot stop answers
`["base"]` when it changed how a goal ends or the leader streamed, and an
empty list otherwise; a goal that a cancel ended before keeps that cancel's
reason. A goal admitted but not yet in the loop when the robot stop arrives
ends cancelled for it as it arrives. A robot stop that the loop does not take
and answer within `stop_answer_timeout_s`, or that finds the loop gone, gives
an error reply with `attempt_failed`.

## The odometry and the reports

The odometry is control_core's `Odometry` over the measured wheel angles,
integrated over the time between the wheels' own stamps, a silence included: a
measurement after a silence takes in the motion the wheels measured across it.
The odometry runs in epochs. The first measurement starts one, and so does a
measurement of another wheel follower than the epoch's, one stamped before the
epoch's last, or one with a wheel farther from its last angle than a wheel
turns in the time between them at the fastest wheel rate the base's limits
allow, with the anchor bound and one limiter step of travel of lag on top
(`BaseConfig::wheel_reach_rad`): the odometry starts at the origin,
`odometry_start_timestamp` takes the wall clock's time then, raised 1 ns past
the last epoch's when the wall clock reads no later, so each epoch's value is
new, and a goal in flight fails with `resource_unavailable`, its message `the
base's odometry started over`. A measurement stamped less than a microsecond
after the last one is dropped.

`base_states`, and `drive_states` to the leader with its `follow_state`,
report each measurement as an odometry record, stamped with it: the pose of
base_footprint in odom and its twist in base_footprint, once the node's clock
reads its stamp; a measurement a newer one replaces before then is not
reported. A report's `mode` is `standing` while the base rests by the rule
below as of the report, and `moving` otherwise; its `parts_within_footprint`
is true, since the base carries no limb. While the wheels report and no newer
measurement waits for the clock, a step that would leave 0.1 s without a
report publishes the newest again, its stamp included. While they are silent,
the node publishes neither.

The base rests while the wheels report and each axis of every `base_states`
whose stamp lies within 0.2 s of the newest lies within its rest bound,
`x.rest_velocity`, 0 along y, and `angular.rest_rate` (`rest.rs`).

The node publishes the transforms it measures on `transforms`, each set in a
message of its own (`transforms.rs`). Before each `base_states` goes the
transform odom to base_footprint at the pose the state reports, under the
state's stamp and started at its `odometry_start_timestamp`, so it starts over
with each epoch. The robot's tree takes it first, and the goals are placed by
that odometry from then: a goal sent on a `base_states` is placed by the
odometry it reports, and a goal placed by an odometry before a start-over
fails as the loop takes it. At each measurement the odometry takes go the
transforms of the two wheel joints, each wheel's link at its joint's origin
turned about its axle by its measured angle, under the measurement's stamp and
started when the node started, so an epoch that starts over leaves their
start. The model's fixed joints are the robot initializer's: the node
publishes nothing on `static_transforms`.

## Goals

The handlers check each goal in base_motion's order (`goal.rs`, `actions.rs`):
they refuse a value the base does not take with `rejected`; place the goal by
the robot's tree, refusing one the tree cannot place with the kind of the
lookup, `unknown_frame`, `no_chain`, `extrapolation` or `broken_tree`; refuse
with `rejected` a target that, placed in odom, lies off the floor plane or
tilts, a velocity that, placed in the robot frame, moves along z or y or turns
about x or y, each past 1e-9 of 0, and a velocity outside describe_base's
limits, each message naming the value as placed and what to send; then admit
it, claiming the base's move slot (control_core's `MoveSlot`):

- `resource_unavailable` while the wheels are silent, and before
  `base_states` reports their odometry;
- `leader_streaming` while the leader streams, followed or held;
- `precondition_failed` while another goal holds the slot: while it runs, and
  while the base still comes to rest from it.

base_motion's refusal while the mode is neither standing nor moving never
comes from this node, whose mode is one of the two while the wheels report.

`drive_base` takes a velocity whose linear_velocity, placed in the robot
frame, lies within the x range and within 1e-9 of 0 along y, a y the base
takes as 0, as it takes z, and whose angular_velocity's z lies within the
`angular` range, a hold_duration_s above 0 and at most 60 s, and a settle
timeout from 0 to 60 s. The base ramps to the velocity, holds it for the hold
duration from the step at which the command reached it, and comes to rest. The
drive completes once the base rests, and fails with `timeout` when it does not
by the settle timeout after the command reached 0, its message giving the
base's velocity then.

`move_base` takes a target, which it places in odom at acceptance and steers
to there, settle tolerances above 0, and a duration and a settle timeout from
0 to 60 s (`move_control.rs`). The plan times, from where the robot frame
stands at acceptance, the turn toward the target, the drive ahead and the turn
to the target's heading, each turn the shorter way, as the fastest moves
describe_base's limits allow at the speed of the way each leg goes, and scales
each leg's speed to take the requested duration. The node takes a base that
drives ahead and turns both ways: x's max_velocity above 0, and angular's
min_rate below 0 and max_rate above 0. A move whose plan takes longer than
60 s, as the fastest move to a far target does on a slow base, is refused, the
refusal giving its time and saying to split it into moves to nearer targets.
Each step steers on the odometry: a turn as Nav2's Spin turns, a drive as
Nav2's DriveOnHeading drives, each braking at the speed from which the
limiter's steps stop the base on the end of its leg. The drive turns toward
the target again once the target's bearing leaves the orientation tolerance,
or the node's `rotate_to_heading_min_angle_rad`, as Nav2's regulated pure
pursuit's rotate_to_heading_min_angle, when the tolerance is wider. A turn
ends once what remains of it falls below one limiter step at the angular
deadband. A drive past its target backs up to it on a base that drives
backward, and turns toward it again on a base that cannot. While the base
stands outside the tolerances, the axis of the leg it is on commands at least
just above its deadband. The move completes once, at or after its planned end,
the base rests within both tolerances, and fails with `timeout` at the settle
timeout after the planned end, its message giving how far the base stands from
the target.

Each result gives `final_pose`, the robot frame's pose in odom at the time of
the newest wheel measurement when the goal ended. A goal that a robot stop or
its caller's cancel ends comes to rest and ends cancelled with its result, the
reason the robot stop's or the cancel's. When the base has not come to rest by
the goal's settle timeout after the first step that found the robot stop or
the cancel, as on a slope or with a measurement noisier than the rest
velocity, the goal ends cancelled then, where the base stands. A robot stop
and a cancel each turn a goal that runs, or a drive that completed and still
comes to rest, into a cancelled one, and leave a goal already cancelled as it
was. The goal's move slot is freed before the outcome goes out, so a client
that sends its next goal on the result is admitted.

`describe_base` answers the base's name, `base`, the name of its wheels'
pairing slot, its odometry source, `measured`, its linear_limits along x and
y, y 0 to 0, and its angular_limits, each with its rest bound, and the four
corners of its footprint, counterclockwise, in base_footprint at every time.

## Module map

| Module | Owns |
|---|---|
| `main.rs` | tracing, then the node runtime running `setup` |
| `node.rs` | bringup: the parameters, the robot's tree, the channels once the model is named and the robot ready, task supervision; the first task to exit takes the node down |
| `identity.rs` | the robot's model and the digest of its description, asked of the robot's initializer |
| `startup.rs` | the wait for the robot's initializer to report the robot ready |
| `follow.rs` | the other publishers of the robot's tree, taken into it |
| `config.rs` | every parameter parsed once, refused by name, and with the model's base, what the node runs on |
| `model.rs` | the base a model's URDF gives: its wheels and its outline |
| `control.rs` | the control loop: events, steps, the odometry epochs, the goal in flight, the wheel setpoints, the reports and the transforms |
| `transforms.rs` | the messages of transforms: odom to base_footprint, and the wheel joints at their angles |
| `follower.rs` | whether the base follows its leader of drive_link, and the holds that keep the base from it |
| `rest.rs` | base_state's rest over the last 0.2 s of base_states |
| `move_control.rs` | move_base's plan and its steering, pure |
| `goal.rs` | the base_motion goals parsed off the wire, and their placed values held to the floor plane and the base's limits |
| `status.rs` | what a goal is admitted on |
| `actions.rs` | the drive_base and move_base handlers |
| `streams.rs` | the wheel and setpoint subscriptions |
| `services.rs` | describe_base and the robot stop |
| `failure.rs` | the failure kinds of the goals, given to each generated member with those of a placement |
| `clock.rs` | the node's clock, wall or simulated, in integer nanoseconds, and the wire times |

## Parameters and links

See `peppy.json5` for the commented list and each range the node enforces.
Every parameter but `cache_window_s` is a required launcher argument: the
limits of x and of the turn (`angular`), each with its rest bound, the
limiter's rate, the odometry window, the rotate-to-heading angle of
move_base's drive, how long a robot stop waits for the control loop, and how
long a placement waits for the robot's tree. `cache_window_s` keeps tf2's 10 s
by default. The `base` slot is required and names the wheels' follower; the
`drive` slot holds one leader or none. `robot_init`, `robot_identity` and
`other_transforms` bind to the robot's initializer.

## Build, run, test

```sh
# Build into the node stack:
peppy node add /path/to/nodes-hub/botshare/backbone -sb

# A base over MCP in Waldo:
peppy stack launch simulation_mcp --join botshare_base_sim:india

# Unit and harness tests; the harness tests run the node on a simulated clock
# the test ticks:
cargo test
```
