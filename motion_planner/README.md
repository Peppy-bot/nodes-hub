# motion_planner

Plans joint paths for one or both arms of a robot around what its perception camera sees
and the objects of its planning scene. The node keeps a voxel map of the camera's depth
frames in the robot frame and a planning scene of collision objects, objects attached to
an arm's grasp link and pairs allowed to touch, as MoveIt's planning scene holds them; it
plans the arms' path with OMPL, keeping a clearance from the map, the objects, the
robot's own body and between the arms, computes straight lines of a link by inverse
kinematics, and checks paths from other sources the same way.
It implements `motion_planning:v1` and moves nothing: a caller runs a plan with
limb_motion's `execute_joint_path`, passing every arm of the answer and each arm's
`start_joint_positions` as its `expected_start_joint_positions`.

```text
 backbone: limb_states ---------> joint history ---+--> cut-out of the robot
 backbone: camera_mounts, workspace  (2 s)         |    and of the objects
 camera:   depth_stream, geometry --> depth points +--> voxel map (robot frame)
 apply_collision_objects, apply_attached_collision_object,  |
 set_allowed_collision --> planning scene ------------------+
                                                          |
 plan_joint_path --> start --> OMPL, goal samples (IK) -----> rounded path check
 plan_cartesian_path --> start --> a step at a time (IK) ---> rounded path check
 check_joint_path --> start, waypoints ----------------------> rounded path check
                                                          |
                          <-- waypoints and grasp poses, or clear, map_stamp
```

## The robot

The node plans for the model the robot's identity names: it asks
`robot_identity.get_identity` of the instance its `identity` slot binds until it answers,
each ask waiting up to 5 s and the next a second after, since the robot's initializer may
start after it, and then starts and serves. `openarm_v2` is the model it knows: another
ends the node with an error that names the model and the ones it knows. The model gives:

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

At home (`openarm_description::home`) with the grippers closed, the robot's bodies stand
0.077 m apart at the least, a hand from the torso. At 0.4 of a gripper's opening a finger
stands 0.050 m from the torso there, and at 0.5 or more within the self clearance (0.039 m
at 0.5), so a plan to home is `unreachable`, the message naming the finger and the torso,
until the gripper closes.

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
  robot filter). So is each shape of the planning scene's objects, by the same two
  numbers and the same rule: within `padding_offset_m` of the shape scaled about its
  center, a mesh's the center of its bounds; an object an arm holds stands where its
  grasp link stood at the frame's stamp. A point inside the grown pieces or shapes is
  the robot or an object seen: its voxel takes the lowest log-odds and reads free, as
  MoveIt sets its model cells, though another point of the same frame ends in it, and
  its ray clears the space before it.
- Adding, moving or attaching an object clears every voxel it intersects where it then
  stands: a call of `apply_collision_objects` clears where it leaves each object it adds
  or moves, and nothing for one it then removes.
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
  (`DepthCamera`, `JointHistory`, `RobotMask`, `insert_cut_out`, and the planning
  scene's `PlanningScene::riders` and `PlanningScene::standing_shapes`), so a replay of
  recorded frames, joint positions and scenes through them gives the map of each frame
  the node takes. Which frames the map takes (the rate, the wait for the joint
  positions, the cut-off of a clear) and where the fingers hang at each frame's
  openings are the node's.
- The map takes in at most `max_update_rate_hz` frames a second, one at a time; the
  frames that arrive while it takes one in wait as the bullet above says.
- A frame's `align_mode` is the one the camera's `get_depth_intrinsics` and
  `get_depth_to_color_extrinsics` describe. A frame with another is skipped and the
  node asks the camera again; so does a pair of answers that name different ones. A
  frame encoded other than `z16` is skipped.
- Space the camera has not seen holds nothing. A table's underside, or what stands
  behind an object, is free to a plan.
- `clear_occupancy_map` empties the map; the planning scene stays. The frames after it
  fill it again: a frame
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

- `arms` holds one arm or each of them, each once, as the backbone refuses them. Each arm's path starts at its `start_joint_positions`, or, when
  that is empty, at the arm's last measured joint positions. An arm `arms` does not name
  stands where it was last measured, and the grippers at their measured openings: a
  planned arm's fingers move with its hand.
- Each constraint goes to the named arm whose joint or link it names; one that names
  none of theirs is refused with `rejected`, listing each named arm's. An arm's goal is
  joint constraints alone, or position and orientation constraints alone, and the two
  arms may take different kinds. A position or orientation constraint names any link of
  its arm: its grasp link, the link of one of its joints or a link fixed below it, or a
  finger where it hangs at the gripper's opening; an arm's position and orientation
  constraints name one link, and an arm whose constraints name two is refused with
  `rejected`, naming both. An arm named with no constraint stays where it starts: the
  plan moves the others, it stands among the parts of the robot they keep clear of, and
  the answer gives its start as each of its waypoints, as MoveIt never moves an arm
  outside its planning group. A plan of both arms plans them as one group, the left
  arm's joints then the right's (`motion_planning_core::Arms<7, 14>`), checked against
  every pair of the self-collision model. This departs from MoveIt, whose
  `ConstraintSamplerManager::selectDefaultSampler` (moveit_core, constraint_samplers,
  MoveIt 2.15.2) keeps one IK sampler, for the link whose constraints leave the least
  sampling volume, and accepts a goal sample only when the other links' constraints hold
  by chance. Goals are sampled as MoveIt's `ConstrainedGoalSampler` samples them: each
  sample sets out from the last goal found, from the start for the first, through
  motion_planning_core's port of MoveIt's KDL inverse kinematics, and each attempt keeps
  up to `goal_samples` goals within its time. A sample of both arms samples each arm in
  turn, as MoveIt's `UnionConstraintSampler` samples a group, each arm's configurations
  checked with the left arm at its sampled joints and the right arm at the last goal's,
  from the start for the first: the whole state from a start that keeps
  `min_clearance_m`, as MoveIt checks it; from a start nearer than that, which the plan
  starts from as the start rule says, the left arm's turn its pairs without the right
  arm, and the right arm's turn its pairs with the left, so an arm that starts nearer
  than the clearance holds back no goal.
- Every state of the path keeps `min_clearance_m` from the map, from the planning
  scene's objects, and between each pair of the robot's own links that the self-collision
  model checks; so does the path `execute_joint_path` runs from the plan's start, its
  corners rounded as it rounds them. An object attached to a planned arm's grasp link
  moves with its hand as the fingers do. It keeps the clearance from the map, the other
  objects, the parts of the robot the plan does not move, an arm it does not move and the
  body, and from the other planned arm and the objects it carries. Between an attached
  object and the links of the arm that carries it, outside its touch links, the clearance
  is 0: they keep out of contact. Objects one arm carries are never checked against each
  other, as MoveIt checks no two bodies attached to one link. An object attached to the
  grasp link of an arm the plan does not move stands where that arm stands, and the
  planned arms keep the clearance from it but for that arm's grasp link and touch links.
  The link that carries an object and its touch links are never checked against it, and
  nor is a pair that `set_allowed_collision` allows. A mesh is checked by its triangles,
  and a link wholly inside a closed mesh collides with it. The planner checks states at
  the plan's clearance, `min_clearance_m` plus 1 mm from the map and the objects, plus 2
  mm between the robot's own links, and 2 mm between an attached object and the links of
  the arm that carries it, and walks each motion by conservative advancement: a step
  moves no point of the arm farther than the spare above that pair's clearance, and at
  most 1.5 cm, so every state between two checked ones keeps `min_clearance_m`, or stays
  out of contact where the clearance is 0. The messages of `unreachable` and `no_path`
  name the plan's clearances in meters; `unreachable`'s also names, of the goal
  configuration that falls short least, the pair it falls short against, two links, a
  link or a carried object and the map or an object, or a carried object and a link, and
  how far apart they stand. Every state of a planned path past its start keeps the plan's
  clearance.
- A start closer than the plan's clearance is valid when, until that clearance is
  reached, none of its shortfalls, from the map and the objects, within the robot's own
  body, and between an attached object and the arm that carries it, grows at a checked
  state, each on its own, to the rounding of the arithmetic (1e-9 m); between two
  checked states each comes at most half its distance nearer. The start sets each one's
  floor: 0 from a start below the clearance asked of it, `min_clearance_m`, or 0 for an
  attached object against its arm, and from one at or above it that clearance or the
  start's distance less 0.2 mm (map and objects) or 0.4 mm (own body, and an attached
  object against its arm), whichever is lower. Between two checked states each dips by
  at most half its room, its farthest checked distance above that floor, so a start
  nearer than the clearance asked stays above half its distance, a start just above it
  comes at most 0.1 mm (map and objects) or 0.2 mm (own body) below it, an attached
  object against its arm, whose clearance is 0, never below 0, and a motion ends where
  its step falls under 0.1 mm. The rule walks the runs on the straight line from the
  start; any other run is walked from its first state by its spare, and fails at a first
  state short of the plan's clearance. A start outside a joint limit by no more than
  `start_tolerance_rad` is moved onto the limit.
- The backbone runs a path from where the arm stands, which its expected-start check
  allows up to its own `start_tolerance_rad` from the plan's start in each joint. Run
  from there, the path comes nearer anything near its start by at most the sum over the
  joints of that tolerance times the joint's reach, the bound `motion_planning_core`
  walks by, as MoveIt's execution starts within its `allowed_start_tolerance` and
  checks nothing more.
- A plan runs on what its admission took: the planning scene, the map, the arms'
  measured joint positions and the grippers' openings, and the self-collision model at
  them. A frame, a clear, a change of the scene or a message of limb_states after
  acceptance changes nothing of it.
- `objective` is `any`: the node plans with RRTConnect, which answers the first path it
  finds within `max_iterations`, and simplifies it in the time left. A goal of another
  objective is refused, the message naming `any`.
- A plan's time follows MoveIt's allowed planning time, as `motion_planning_core`
  implements it, with no share kept for any part: the attempts take
  `allowed_planning_time_s`, which a request gives above 0 and at most 60, from when the
  node takes the goal, and the simplifier what they leave. Each attempt plans under the
  whole of it, and samples up to `goal_samples` goals as its planner asks for them,
  ending at the deadline; the simplifier takes at most 1000 iterations of what they
  leave, and its first pass runs to its end past the deadline, as MoveIt's does. The
  check of the rounded path runs to its end after the deadline, so a plan answers after
  its deadline by the time the simplifier's first pass, when the attempts leave it none
  of the time, and that check take. With more than 1 attempt, the hybrid of the attempts'
  paths and its checks also run after the deadline.
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
  the same map and planning scene gives the same path when no part of the plan ends at
  its deadline, as a plan that finds its path in its time.
- A cancel stops the plan at any part of it, and the goal ends cancelled, whatever the
  plan answers after it.

The answer is the arm's `start_joint_positions`, where its path starts, and its
`waypoints` after the start, the last meeting the goal, each with the grasp point's
`position` and `orientation` there; and `map_stamp`, the stamp of the newest depth
frame in the map.

`path_orientation_constraints` lists orientations held along the whole path, such as a
carried cup kept upright: each names a link of a named arm, with `orientation`, `tolerance_rad`
and `free_axis` as a goal's. Every state of the path from the plan's start, its rounded corners
included, holds each, and their order carries no meaning (`motion_planning_core`'s "Path
orientations"). Run from within `limb_motion`'s start tolerance of that start, a link turns from
where the path holds it by at most the sum of its joints' differences from the start. A goal that names a
position of a link and no orientation takes the orientation of that link's constraint. An arm named
with no goal holds its constraints when its start lies within `tolerance_rad`. A start more than
0.99 of `tolerance_rad` from an orientation fails with `no_path`, and so does a goal farther from it
than 0.9 of `tolerance_rad` or the start's own angle from it, whichever is more.

## A Cartesian path

`plan_cartesian_path` follows a link of one arm along straight lines through `waypoints`,
as `motion_planning_core::cartesian_path` computes them after MoveIt's
`compute_cartesian_path`, at MoveIt's precision, 1 mm and 0.01 rad, in steps of 0.01 m and
0.035 rad.

- `arm_name` names the arm. Its path starts at `start_joint_positions`, or, when that is
  empty, at its last measured joint positions, and the other arm stands where it was last
  measured.
- `link_name` is its grasp link or a link fixed to it. Any other link, an inner link or a
  finger, is refused with `rejected`, the message naming the grasp link. So are no `waypoints`
  and a pose that is no pose (`pose_from_parts`).
- With `avoid_collisions` true the path keeps clear of the snapshot its admission took, as a
  plan does: refused with `precondition_failed` before the camera's depth frame, a start in
  contact fails with `start_in_collision`, and `map_stamp` is the frame's stamp. With it false,
  nothing is checked, no frame is needed, and the answer has no `map_stamp`.
- It is refused with `precondition_failed` before the arms are measured and for a start
  outside a joint limit by more than `start_tolerance_rad`.
- A cancel ends it, at any part of it, and the goal ends cancelled.

The result gives `start_joint_positions`; `waypoints`, each with the grasp point's `position`
and `orientation` there, as much of the path as is valid; and `fraction`, the share of the
lines covered. A path whose lines stand still has its last state as its one waypoint, and one
whose first step fails has none.

## A check

`check_joint_path` checks the path of one arm or both from their starts (empty: the
arm's last measured joint positions) through their waypoints, as many for each arm, on
the planning scene and the map as they stand, with the checks of a plan's answer
(`motion_planning_core::check_path`). An arm whose every waypoint is its start, inside
its joint limits, stands still where it starts, as a plan holds an arm it does not move,
and the check moves the others; when every arm stands still, it moves them all. The
checks: the start inside its limits by
`start_tolerance_rad` and out of contact, each waypoint inside the joint limits and,
after the start, at the plan's clearance, each run between them walked as a plan walks
it, the first from the start by the start rule, and each arc of the rounded path. A plan
ends with this check of its answer, so a planned path passes it. `clear` is true when the whole path passes. `clear_waypoints`
counts the waypoints, from the first, up to the first fault: a fault on the run from a
waypoint, or on the corner rounded at it, leaves the waypoints up to that one, and the
path to them, every arm run together through them with the last of them as its end,
passes. A start in contact or outside a joint limit by
more than `start_tolerance_rad`, or a fault on the run from the start, leaves 0.
`reason` names the fault and how near the path comes. A walk of a run or of an arc
starts only within 10 s of the request, and a check with a walk left at 10 s fails with
`timeout`, the message saying to send a path of fewer waypoints: under the 15 s the
robot's MCP endpoint waits on `check_joint_path`.

A plan's refusals come at the goal's admission, before the node takes the goal; its
failures come after.

## The planning scene

The scene holds, as MoveIt's planning scene does besides its octomap, the collision
objects other nodes add, the objects attached to an arm's grasp link that move with it,
and the pairs of names allowed to touch. `motion_planning_core`'s `PlanningScene`
decides every rule of it (`motion_planning_core`'s README, The planning scene); the node
serves its four members on the scene as it stands:

- `apply_collision_objects` applies its objects in order, all or none, and an empty list
  changes nothing. "add" places an object of boxes, spheres, cylinders and closed
  meshes, posed in the robot frame, in place of one of the same id, whose allowed pairs
  it keeps; "move" reads the id and the pose; "remove" reads the id and drops every pair
  that names the object.
- `apply_attached_collision_object` attaches an object to an arm's grasp link where both
  stand, the link where the arms' last measured joint positions put it, with touch links
  it may touch besides that link; a detach, which reads the id alone, leaves the object
  in the scene where it stands. Its allowed pairs stay through both.
- `set_allowed_collision` lets an object, attached or not, touch links of the robot,
  other objects, or `"occupancy_map"`, the voxel map, or stops it. To set a held item
  down on a table, a caller pairs the item with the table, and the table with the
  gripper's links too where they come nearer to it than `clearance_m`: the table is an
  object, which every link keeps that clearance from.
- `get_planning_scene` answers the objects no link carries and the attached objects,
  posed in their grasp links' frames, each list in the order of the ids; each allowed
  pair once, its names in order, the pairs in the order of their names; `clearance_m`,
  which is `min_clearance_m`; and `voxel_size_m`.

| Refusal | When |
|---|---|
| `rejected` | An object's id is empty, or names a link or `"occupancy_map"`; an add has no shape, or a shape that cannot be built, an open mesh among them; an add or a move has a pose whose position is not finite or whose orientation is no unit quaternion; "move" or "remove" names an unknown object; an operation names an attached object; an operation is none of the member's; an attach names an object the scene does not hold, a link that is no arm's grasp link, a touch link that is no link of the robot, an object's id among them, or an attached object; a detach names an object the scene does not hold or one no link carries; `set_allowed_collision`'s `name` is no object of the scene, or `other_names` holds a name that is no link, no object and not `"occupancy_map"`, or the object itself. The message names what to send, and the call changes nothing. |
| `precondition_failed` | An attach before the arms' joint positions are measured. |

Where the scene departs from MoveIt, each a choice a caller sees, against moveit2 2.15.2
(`PlanningScene`, `moveit_core/planning_scene/src/planning_scene.cpp`, unless named) and
geometric_shapes 2.3.4:

- The link: an object attaches to an arm's grasp link alone, which a call names as
  describe_arms names it. `processAttachedCollisionObjectMsg` attaches one to any link of
  the robot (lines 1542-1547).
- The link's contact: an attached object is never checked against the link that carries
  it. MoveIt's FCL check exempts a link only when the object's touch links name it
  (`collision_detection_fcl/src/collision_common.cpp`, lines 131-156), as
  `MoveGroupInterface::attachObject` names it when it is given no touch links
  (`move_group_interface.cpp`, line 955).
- Ids: an empty id is refused, where MoveIt's `processCollisionObjectRemove` removes every
  object for a remove of an empty id (lines 1939-1941). An add whose id names a link of
  the robot, or an attached object, is refused.
  `processCollisionObjectAdd` adds a world object of any id but its octomap's, an
  attached object's among them, beside the attached object (lines 1893-1935).
- A detach reads the object's id alone. MoveIt refuses one whose `link_name` names a
  link other than the one that carries the object (lines 1720-1727).
- Pairs: a pair names an object of the scene. MoveIt's allowed collision matrix holds
  any two names, two links among them.
- Touch links name links of the robot, and an object's id among them is refused. MoveIt
  exempts an attached body that another names among its touch links
  (`collision_detection_fcl/src/collision_common.cpp`, lines 167-173).
- Meshes: a mesh is a closed solid, and an open one is refused; MoveIt takes the
  triangles it is given (geometric_shapes `createMeshFromVertices`,
  `src/mesh_operations.cpp`, line 112). A link wholly inside a closed mesh collides with
  it (`motion_planning_core`'s README, A valid state).
- The depth cut-out: an object's shape grows as a link's piece does, within
  `padding_offset_m` of the scaled shape, and a mesh as itself. MoveIt's shape mask
  grows the same two numbers by sharp planes, a box's half sides and a cylinder's radius
  and half length each by the padding, and cuts out a mesh's convex hull
  (geometric_shapes `bodies::Box`, `bodies::Cylinder` and `bodies::ConvexMesh`,
  `src/bodies.cpp`; the mask's `ShapeMask::addShape`,
  `point_containment_filter/src/shape_mask.cpp`).
- The map at an add: adding, moving or attaching an object clears every voxel it
  intersects. MoveIt's `PlanningSceneMonitor::excludeWorldObjectFromOctree` adds the
  object to the shape mask of the scans to come, and its octomap keeps each voxel until
  a scan updates it (`planning_scene_monitor/src/planning_scene_monitor.cpp`, lines
  1038-1059).

## Departures from MoveIt

Each is a choice a caller sees; the launchers' values are OpenArm v2's.

- Path orientations: MoveIt projects onto one orientation path constraint, and with two or more
  plans in pose space through inverse kinematics; on a group of two arms its Jacobian is refused
  (`RobotState::getJacobian`, moveit2 2.15.2 robot_state.cpp:1516). Here every constraint is
  projected, on one arm and on two. MoveIt judges a rotation vector's components against each
  tolerance, about the axes of the constraint's frame, at the states along each motion and at the
  waypoints of its answer. Here `tolerance_rad` bounds the angle itself, or with `free_axis` the
  angle of the held axis, at every state of the rounded path, and the plan corrects its states to
  0.9 of it, which leaves the last tenth for what lies between them.
- A Cartesian path: MoveIt checks each state for collisions, not its start and not the motion
  between two, and each interval's midpoint for the precision. Here the rounded path the
  move runs is checked for both, a start in contact fails with `start_in_collision`, and a
  start nearer than the clearance departs by the start rule in one run. MoveIt judges each
  state alone, so a first state still nearer than its padding ends its path there. Any link
  but the grasp link and those fixed to it is refused with `rejected`, where MoveIt's inverse
  kinematics fails and its fraction is 0. At a clearance of 0.04 m a line among obstacles ends
  before MoveIt's at the same padding: over the 27 drawn cases where it does, MoveIt's last state
  stood 0 to 39 mm from the obstacle, 21 mm at the median, and no state here after the start comes
  nearer than the clearance.
- Clearance: MoveIt checks contact alone by default (`CollisionEnv` padding 0.0), and
  pads links against the world only (`pad_self_collisions` false). Here every state keeps
  `min_clearance_m` (0.04 m) from the map, the objects, an object an arm the plan does
  not move holds among them, which MoveIt checks as that arm's attached body unpadded,
  and between each pair of links the self-collision model checks, the two arms' and an
  object one arm carries against the other arm and its objects included, and a checked
  state 1 mm more from the map and the objects and 2 mm more between the robot's own
  links. Between an attached object and the links of the arm that carries it, outside its
  touch links, the clearance is 0: they keep out of contact, 2 mm apart at a checked
  state, where MoveIt checks an attached object against the robot unpadded. MoveIt pads
  no object an arm holds, since it pads only links with collision geometry and the grasp
  point has none, which it pads by 0 (moveit2 2.15.2 collision_env.cpp:127-133 and
  :166-176, collision_env_fcl.cpp:195-196): here a held object keeps `min_clearance_m`
  from the map and the objects. MoveIt pads a link's mesh along its vertex normals
  (geometric_shapes 2.3.4 shapes.cpp:402-404), so a face beside a right-angled edge moves
  out 2.8 cm at a 4 cm padding; here the clearance holds all round. MoveIt checks the
  robot's own links unpadded, so a goal nearer than the clearance between them fails
  `unreachable` here where MoveIt reaches it; the parity tally of both arms at a
  clearance above 0 draws goals that keep it.
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
  leave the least volume, accepting a sample that meets the others by chance. Here each
  attempt samples up to `goal_samples` (10) goals as its planner asks, and an arm's goal
  names one of its links. For a link the last joint does not move, and for a finger of the arm, a part
  its hand carries that hangs on a joint of its own (`Frame::Part`, `Frame::from` its
  rider), MoveIt builds no inverse kinematics and keeps the states it samples that meet
  the goal by chance; here the goal sampling descends toward what the goal names and takes
  the descents' approximate ends, the goal's constraints deciding (motion_planning_core's
  README, "The goal").
- The time: the check of the rounded path runs to its end after the deadline.
- Self pairs: MoveIt's OpenArm v2 SRDF (enactic/openarm_ros2,
  `openarm_v2.0/openarm_bimanual.srdf`) disables adjacent links alone, and checks each
  upper arm against the torso and against its own arm's base. Here those pairs are left
  out: within the joint limits they stay 12.9 mm and 18.5 mm apart
  (`openarm_description`'s `never_touching_pairs`).
- Attempts: as MoveIt's, but for the departures `ompl_bridge`'s README lists under
  Attempts.
- Repeatability: a plan is seeded from the map's stamp, so a request repeats on the same
  map and planning scene when no part of the plan ends at its deadline; MoveIt's
  samplers draw from the process's seed generator.

| Refusal or failure | When |
|---|---|
| `rejected` | The request names no arm, an arm twice, an arm, joint or link the named arms do not have, a name twice in one list of constraints, or a number outside its range; an arm's constraints mix joint and pose constraints, or its position and orientation constraints name more than one of its links; or a check's arms hold different numbers of waypoints. The message names what to send. |
| `precondition_failed` | The camera is linked and the map has taken in no depth frame since the node started or the map was cleared, the message naming what the map waits on; or no joint positions have been measured; or a plan's start lies outside a joint limit by more than `start_tolerance_rad`, the message naming the joint. |
| `start_in_collision` | A plan's start touches the map, an object of the planning scene or the robot's own body; the message names the pair. |
| `unreachable` | No configuration meets the goal before the goal sampling ends (once the samples since the last goal found reach 1000 less the goals found so far, as MoveIt's `max_goal_sampling_attempts` bounds the samples its sampling function makes before it returns a goal, counting from the goals OMPL has taken) or within the time allowed, the message naming in a plan of both arms the first arm that met its goal in no sample; or every configuration, or state of both arms, that the goal sampling tried comes nearer than the plan's clearance, which the message names with the pair that the one falling short least falls short against: two links, a link or a carried object and the map or an object, a carried object and a link, or two objects the two arms carry. The message says so when the deadline ended the sampling, and that more time may find a goal. |
| `no_path` | The planner finds no path within the time and the iterations allowed, or the start rule leaves no motion from the start; a planned waypoint falls short of the plan's clearance, or a corner's rounding does however near it the added waypoints lie; the planned path fails its check, simplified and as planned; the time allowed ends before the plan answers; or a start or a goal breaks a path orientation, or no configuration that meets the goal keeps them. The message names the plan's clearance where it falls short, and the link where a path orientation breaks. |
| `timeout` | A check has a walk of a run or an arc left at 10 s. |
| `attempt_failed` | The planner fails on its own: OMPL reports an error, or the plan's task ends without an answer. |

## Parameters and links

See `peppy.json5` for the commented list. Every parameter is required, and the
launcher sets each one. `min_clearance_m` stands at or above the backbone's
`d_safe_m`, so the backbone's guard does not slow the arm on a planned path. A
parameter outside its range stops the node with a message that names it.

The node reads `identity` (`robot_identity:v1`) from the robot's initializer,
`limb_state`, `camera_mounts` and `workspace` from the backbone, and `camera`
(`rgbd_camera:v1`) and `geometry` (`camera_geometry:v1`) from the perception camera. The four camera slots are optional, and bind as the plan section says.

## Build, run, test

```sh
# Build into the node stack:
peppy node add /path/to/nodes-hub/motion_planner -sb

# The unit tests and the harness tests; the build needs the packages apptainer.def
# installs (Boost serialization, Eigen, pkg-config, libclang):
cargo test
```
