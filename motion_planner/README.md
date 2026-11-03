# motion_planner

Plans joint paths for one arm of a robot around what its perception camera sees. The
node keeps a voxel map of the camera's depth frames in the robot frame, plans an arm's
path on it with OMPL, keeping a clearance from the map and from the robot's own body,
and checks paths from other sources the same way. It implements `motion_planning:v1`
and moves nothing: a caller runs a plan with limb_motion's `execute_joint_path`,
passing the plan's `start_joint_positions` as `expected_start_joint_positions`.

```text
 backbone: limb_states ---------> joint history ---+--> cut-out of the robot
 backbone: camera_mounts, workspace  (2 s)         |
 camera:   depth_stream, geometry --> depth points +--> voxel map (robot frame)
                                                          |
 plan_joint_path --> start --> goal samples (IK) --> OMPL --> rounded path check
 check_joint_path --> start, waypoints ----------------------> rounded path check
                                                          |
                          <-- waypoints and grasp poses, or clear, map_stamp
```

## The robot

The node plans for the model the robot's identity names: it asks
`robot_identity.get_identity` of the instance its `identity` slot binds, each second until
it answers, since the robot's initializer may start after it, and then starts and serves.
`openarm_v2` is the model it knows: another leaves it serving nothing, its log naming the
model and the ones it knows. The model gives:

- each arm by its limb name (`left_arm`, `right_arm`), its gripper's (`left_gripper`,
  `right_gripper`), and its joints in limb_state's order, with the limits
  limb_motion enforces: the URDF's, the elbow's lower limit raised to its
  singularity floor;
- the grasp point's pose, at the end of the arm's chain from its base link to the grasp
  point, where limb_state reports it;
- the links: every link the arm's first joint moves, its gripper's included; the
  gripper's links, which a held object may touch: every link the arm's last joint moves
  that has collision geometry (`openarm_left_ee_base_link`, `openarm_left_ee_link1` and
  `openarm_left_ee_link2`); the grasp link (`openarm_left_tcp`) and its frame's approach
  (`+z`) and closing (`+y`) axes;
- the body around the arms (the torso, the shoulder mounts, the head camera), and
  OpenArm's self-collision model of `self_collision_model`, which leaves out the pairs
  the joint limits keep apart: each upper arm (`link3`) against the torso and against its
  own arm's base.

At home (`openarm_description::home`) with the gripper closed, the robot's bodies stand
0.077 m apart at the least, a hand from the torso. With a gripper open past about 0.45, a
finger stands within the clearance of the torso there, so a plan to home is `unreachable`,
the message naming the finger and the torso, until the gripper closes.

`describe_arms` answers all of it.

## The map

- `workspace.describe_workspace` names the perception camera, and the node reads that
  camera's pose from `camera_mounts.get_camera_poses`: the camera's depth optical frame
  stands where that pose puts its color frame, moved by the camera's own
  `get_depth_to_color_extrinsics`. The depth frames, the lens and the depth model come
  from the camera the launcher binds to the `camera` and `geometry` slots.
- A depth frame clears the space along each ray up to its point and marks the voxel
  at the point occupied, as MoveIt's occupancy map does. A point past `max_range_m`
  clears its ray up to that range and marks nothing.
- The robot is cut out of each frame: both arms at the joint positions, and their
  fingers at the grippers' openings, they stood at when the camera took it, read from a
  history of `limb_states` that spans 2 s and interpolated at the frame's stamp, and the
  body fixed around them. Each collision piece is grown to `padding_scale` about the
  mean of its vertices (a box, a cylinder or a sphere about its origin), and holds every
  point within `padding_offset_m` of the scaled piece. MoveIt's robot filter takes the
  same two numbers and grows a body by sharp planes (`collision_geometry`'s README, The
  robot filter). A point inside the grown pieces is the robot seeing itself: its voxel
  takes the lowest log-odds and reads free, as MoveIt sets its model cells, and its ray
  clears the space before it.
- A frame waits for the joint positions at its stamp. Two frames wait at most: the
  oldest that has arrived since the map last took one, and the newest after it; a frame
  between them is never taken. The map takes the newer of the two whose stamp a
  `limb_states` message has reached, so frames come while `limb_states` lags the camera
  by more than a frame interval. While a frame waits, a refused plan or check names it
  and how far the newest `limb_states` message lags it. A frame stamped before the
  oldest joint positions the history holds is skipped.
- A `limb_states` message stamped before the newest, as after a clock that steps back,
  restarts the history and clears the map at its stamp: the map waits on its next
  frame, a frame stamped at or before the restart marks nothing, and the frames that
  wait for joint positions on the clock before it are dropped.
- The deprojection, the history and the cut-out are `motion_planning_core`'s
  (`DepthCamera`, `JointHistory`, `RobotMask`, `insert_cut_out`), so a replay of
  recorded frames and joint positions through them gives the map of each frame the
  node takes. Which frames the map takes (the rate, the wait for the joint positions,
  the cut-off of a clear) and where the fingers hang at each frame's openings are the
  node's.
- The map takes in at most `max_update_rate_hz` frames a second, one at a time; the
  frames that arrive while it takes one in wait as the bullet above says.
- A frame's `align_mode` is the one the camera's `get_depth_intrinsics` and
  `get_depth_to_color_extrinsics` describe. A frame with another is skipped and the
  node asks the camera again; so does a pair of answers that name different ones. A
  frame encoded other than `z16` is skipped.
- Space the camera has not seen holds nothing. A table's underside, or what stands
  behind an object, is free to a plan.
- `clear_occupancy_map` empties the map. The frames after it fill it again: a frame
  the camera took at or before the newest joint positions measured at the clear marks
  nothing, however late it arrives.
- The four camera slots bind as one: with `camera` and `geometry` bound, the map reads
  the camera, and `camera_mounts` and `workspace` are bound too. With `camera` and
  `geometry` vacant the map stays empty, whether `camera_mounts` and `workspace` are
  bound or not, as on a robot without a camera rig: plans and checks keep the clearance
  from the robot's own body, and answer no `map_stamp`. Any other binding leaves the map
  waiting on a camera it cannot read, and every plan and check is refused, the message
  naming the slots to bind or leave vacant.
- While the map holds no frame, a refused plan or check names what the map waits on:
  the camera's description and the call that failed or the answer that gives no lens,
  depth model or pose, a frame skipped and why, or a fault in the robot's description,
  which ends the asking: no perception camera named, none placed by that name, one
  placed at no pose, or one an arm carries.

## A plan

- `arms` holds one arm. Its path starts at `start_joint_positions`, or, when that is
  empty, at the arm's last measured joint positions. The other arm stands where it was
  last measured, and the grippers at their measured openings: the planned arm's
  fingers move with its hand.
- The goal is joint constraints alone, or position and orientation constraints alone.
  A position or orientation constraint names any link of the arm: its grasp link,
  the link of one of its joints or a link fixed below it, or a finger where it hangs
  at the gripper's opening; the position and the orientation constraints name one
  link, and a goal that names two is refused with `rejected`, naming both. This
  departs from MoveIt, whose `ConstraintSamplerManager::selectDefaultSampler`
  (moveit_core, constraint_samplers, MoveIt 2.15.2) keeps one IK sampler, for the link
  whose constraints leave the least sampling volume, and accepts a goal sample only
  when the other links' constraints hold by chance. Goal configurations are drawn
  with the arm's inverse kinematics, up to `goal_samples` of them within the plan's
  time.
- Every state of the path keeps `min_clearance_m` from the map and from the robot's
  own body, measured by the self-collision model; so does the path
  `execute_joint_path` runs from the plan's start, its corners rounded as it rounds
  them. The planner checks states at the plan's clearance, `min_clearance_m` plus 1 mm
  from the map and plus 2 mm between the robot's own links, and walks each motion by
  conservative advancement: a step moves no point of the arm farther than the spare
  above `min_clearance_m`, and at most 1.5 cm, so every state between two checked ones
  keeps `min_clearance_m`. The messages of `unreachable` and `no_path` name the plan's
  clearance in meters; `unreachable`'s also names, of the goal configuration that falls
  short least, the pair of links, or the link and the map, it falls short against and
  how far apart they stand. Every state of a planned path past its start keeps the plan's
  clearance.
- A start closer than the plan's clearance is valid when, until that clearance is
  reached, neither the shortfall from the map nor the one from the robot's own body
  grows at a checked state, each on its own, to the rounding of the arithmetic
  (1e-9 m); between two checked states each comes at most half its distance nearer.
  The start sets each one's floor: 0 from a start below `min_clearance_m`, and from one
  at or above it `min_clearance_m` or the start's distance less 0.2 mm (map) or 0.4 mm
  (own body), whichever is lower. Between two checked states each dips by at most half
  its room, its farthest checked distance above that floor, so a start nearer than
  `min_clearance_m` stays above half its distance, a start just above it comes at most
  0.1 mm (map) or 0.2 mm (own body) below it, and a motion ends where its step falls
  under 0.1 mm. The rule walks the runs on the straight line from the start; any other
  run is walked from its first state by its spare, and fails at a first state short of
  the plan's clearance. A start outside a joint limit by no more than
  `start_tolerance_rad` is moved onto the limit.
- The backbone runs a path from where the arm stands, which its expected-start check
  allows up to its own `start_tolerance_rad` from the plan's start in each joint. Run
  from there, the path comes nearer anything near its start by at most the sum over the
  joints of that tolerance times the joint's reach, the bound `motion_planning_core`
  walks by, as MoveIt's execution starts within its `allowed_start_tolerance` and
  checks nothing more.
- A plan runs on what its admission took: the map, the arms' measured joint
  positions and the grippers' openings, and the self-collision model at them. A frame,
  a clear or a message of limb_states after acceptance changes nothing of it.
- `objective` is `any`: the node plans with RRTConnect, which answers the first path it
  finds within `max_iterations`, and simplifies it in the time left. A goal of another
  objective is refused, the message naming `any`.
- A plan's time follows MoveIt's allowed planning time, as `motion_planning_core`
  implements it, with no share kept for any part: the goal sampling and the attempts
  take `allowed_planning_time_s`, which a request gives above 0 and at most 60, from when
  the node takes the goal, and the simplifier what they leave. The goal sampling draws
  up to `goal_samples` configurations, ending early at the deadline; the attempts plan
  in all the time left, each under the whole of it; the simplifier takes at most 1000
  iterations of what they leave, and its first pass runs to its end past the deadline, as
  MoveIt's does. The check of the rounded path runs to its end after the deadline, so a
  plan answers after its deadline by the time the simplifier's first pass, when the
  attempts leave it none of the time, and that check take. With more than 1 attempt,
  the hybrid of the attempts' paths and its checks also run after the deadline.
- The planner makes `planning_attempts` attempts, each from its own seed: four at once,
  MoveIt's default `max_planning_threads`, and each further four once the four before
  them have ended, while the time lasts; a single attempt plans on the plan's own
  thread. Their paths join into their hybrid by OMPL's path hybridization, as MoveIt's
  `ModelBasedPlanningContext::solve` joins the attempts of a request whose
  `num_planning_attempts` is above 1 (`ompl_bridge`'s README, Attempts). The plan
  keeps one copy of the self-collision model for each attempt that runs at once, at most
  `ompl_bridge::PLANNING_THREADS` (4), and each check of the robot's own body runs on a
  copy no other check holds.
- A plan is seeded from the map's stamp, so the same request from the same start on
  the same map gives the same path when no part of the plan ends at its deadline, as a
  plan that finds its path in its time.
- A cancel stops the plan at any part of it, and the goal ends cancelled, whatever the
  plan answers after it.

The answer is the arm's `start_joint_positions`, where its path starts, and its
`waypoints` after the start, the last meeting the goal, each with the grasp point's
`position` and `orientation` there; and `map_stamp`, the stamp of the newest depth
frame in the map.

## A check

`check_joint_path` checks one arm's path from its start (empty: the arm's last
measured joint positions) through its waypoints with the checks of a plan's answer
(`motion_planning_core::check_arm_path`): the start inside its limits by
`start_tolerance_rad` and out of contact, each waypoint inside the joint limits and,
after the start, at the plan's clearance, each run between them walked as a plan walks
it, the first from the start by the start rule, and each arc of the rounded path. A plan
ends with this check of its answer, so a planned path passes it. `clear` is true when the whole path passes. `clear_waypoints`
counts the waypoints, from the first, up to the first fault: a fault on the run from a
waypoint, or on the corner rounded at it, leaves the waypoints up to that one, and the
path to them, run on its own, passes. A start in contact or outside a joint limit by
more than `start_tolerance_rad`, or a fault on the run from the start, leaves 0.
`reason` names the fault and how near the path comes. A walk of a run or of an arc
starts only within 10 s of the request, and a check with a walk left at 10 s fails with
`timeout`, the message saying to send a path of fewer waypoints: under the 15 s the
robot's MCP endpoint waits on `check_joint_path`.

A plan's refusals come at the goal's admission, before the node takes the goal; its
failures come after.

## Departures from MoveIt

Each is a choice a caller sees; the launchers' values are OpenArm v2's.

- Clearance: MoveIt checks contact alone by default (`CollisionEnv` padding 0.0), and
  pads links against the world only (`pad_self_collisions` false). Here every state keeps `min_clearance_m` (0.04 m)
  from the map and from the robot's own body, and a checked state 1 mm more from the
  map and 2 mm more between the robot's own links.
- Motion check: MoveIt's OMPL checks the states of a motion 0.01 of the state space's
  extent apart (`longest_valid_segment_fraction`, moveit2 2.15.2
  `ModelBasedPlanningContext`) and nothing between them. Here every state between two
  checked ones keeps `min_clearance_m`, the steps at most 1.5 cm of any point.
- The start: MoveIt refuses a start in contact (`START_STATE_IN_COLLISION`,
  `CheckStartStateCollision`) and plans from any other. Here a start in contact fails
  `start_in_collision`, and a start nearer than the plan's clearance plans by the start
  rule above: between checked states it comes at most half its distance nearer.
- A start outside a joint limit: MoveIt refuses it (`START_STATE_INVALID`). Here a start
  within `start_tolerance_rad` (0.05 rad) of the limit moves onto it.
- The answer: MoveIt checks the simplified path alone and answers `INVALID_MOTION_PLAN`
  when it fails. Here the path the robot runs, its corners rounded, is checked, with
  waypoints added within a quarter of a corner's runs where a corner falls short, and
  the planned path is answered when only it passes.
- The goal: MoveIt samples goals while it plans (`ConstrainedGoalSampler`, OMPL's
  `GoalLazySamples`), and for a goal on several links samples the one whose constraints
  leave the least volume, accepting a sample that meets the others by chance. Here the
  sampler draws up to `goal_samples` (64) configurations before OMPL plans, and a goal
  names one link.
- The time: the check of the rounded path runs to its end after the deadline.
- Self pairs: MoveIt's OpenArm v2 SRDF (enactic/openarm_ros2,
  `openarm_v2.0/openarm_bimanual.srdf`) disables adjacent links alone, and checks each
  upper arm against the torso and against its own arm's base. Here those pairs are left
  out: within the joint limits they stay 12.9 mm and 18.5 mm apart
  (`openarm_description`'s `never_touching_pairs`).
- Attempts: as MoveIt's, but for the departures `ompl_bridge`'s README lists under
  Attempts.
- Repeatability: a plan is seeded from the map's stamp, so a request repeats on the same
  map when no part of the plan ends at its deadline; MoveIt's samplers draw from the
  process's seed generator.

| Refusal or failure | When |
|---|---|
| `rejected` | The request names more or fewer than one arm, an arm, joint or link the arm does not have, a name twice in one list of constraints, or a number outside its range; or the goal mixes joint and pose constraints, or its position and orientation constraints name more than one link. The message names what to send. |
| `precondition_failed` | The camera is linked and the map has taken in no depth frame since the node started or the map was cleared, the message naming what the map waits on; or no joint positions have been measured; or a plan's start lies outside a joint limit by more than `start_tolerance_rad`, the message naming the joint. |
| `start_in_collision` | A plan's start touches the map or the robot's own body; the message names the pair. |
| `unreachable` | No configuration meets the goal within `goal_samples` draws and the time allowed, or every one that does comes nearer than the plan's clearance, which the message names with the pair of links, or the link and the map, that the one falling short least falls short against. |
| `no_path` | The planner finds no path within the time and the iterations allowed, or the start rule leaves no motion from the start; a planned waypoint falls short of the plan's clearance, or a corner's rounding does however near it the added waypoints lie; the planned path fails its check, simplified and as planned; or the time allowed ends before the plan answers. The message names the plan's clearance where it falls short. |
| `timeout` | A check has a walk of a run or an arc left at 10 s. |
| `attempt_failed` | The planner fails on its own: OMPL reports an error, or the plan's task ends without an answer. |

## Parameters and links

See `peppy.json5` for the commented list. Every parameter is required, and the
launcher sets each one. `min_clearance_m` stands at or above the backbone's
`d_safe_m`, so the backbone's guard does not slow the arm on a planned path. A
parameter outside its range stops the node with a message that names it.

The node reads `limb_state`, `camera_mounts` and `workspace` from the backbone, and
`camera` (`rgbd_camera:v1`) and `geometry` (`camera_geometry:v1`) from the perception
camera. The four camera slots are optional, and bind as the plan section says.

## Build, run, test

```sh
# Build into the node stack:
peppy node add /path/to/nodes-hub/motion_planner -sb

# The unit tests and the harness tests; the build needs the packages apptainer.def
# installs (Boost serialization, Eigen, pkg-config, libclang):
cargo test
```
