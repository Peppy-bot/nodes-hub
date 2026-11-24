# base_navigator

Drives a robot's mobile base to a pose around what its depth cameras see, turns it in place, and
drives it straight ahead or back clear of the same, as Nav2 1.5.2's NavigateToPose, Spin,
DriveOnHeading and BackUp drive a base, on navigation_core's ports of Nav2. It implements
`base_navigation` and leads the base over `drive_link`.

```text
   base_navigation: navigate_to_pose, spin, drive_on_heading, back_up, clear_grid
                 |
                 v
   depth_stream (each camera) --> frame intake, placed by the robot's tree
   transforms, static_transforms --> the robot's tree        |
   drive_states (drive_link) --> odometry, mode, follow_state |
   describe_base (base_state) --> limits, footprint           v
                                            grid of the floor in odom: one occupancy
          +---------------------------------------------------------------+
          |  control loop, a step every controller period on the node's   |
          |  clock: the shared endings, the goal's step, the stop, the    |
          |  release burst                     planner: a task of its own |
          +---------------------------------------------------------------+
                 |
                 v   drive_setpoints
   base (drive_link, leader): the robot's backbone
```

## What it reads

- `base_state`'s `describe_base`, read each time `drive_states` reports a new
  `odometry_start_timestamp`, the first included: the base's name, its odometry source, the
  limits along x and of the turn, and its footprint, padded by `footprint_padding_m` and inflated
  by `inflation_radius_m` and `cost_scaling_factor`. Each read logs the stop's horizons.
- `drive_link`'s `drive_states` on the `base` pair: where the odometry places the robot frame in
  odom, the velocity the base executes, its mode, `parts_within_footprint` and `follow_state`.
  Each arrival is timed on the node's clock; more than 0.5 s with none is a silence.
- `sensor_mounts`' `describe_sensors`, asked until it answers: each camera of the `cameras` slot
  must be a sensor of `rgbd_camera` fixed to the base, its `carried_by` empty, under the name its
  copy's fragment wrote (its instance id outside a copy). Another camera is refused, naming why.
- Each camera's `get_depth_intrinsics`, `get_depth_to_color_extrinsics` (the `geometry` slot,
  paired by name) and `depth_stream_info`, read by camera_core's depth camera reader, asked again
  until they answer. A camera's range is its `max_depth_m`.
- `transforms` and `static_transforms` of every publisher bound to the `transforms` slot, taken
  into the robot's tree (control_core's `RobotTree`, `cache_window_s` long).

## The grid of the floor

The grid is navigation_core's live obstacles, kept in odom: a square of `grid.side_m`, from
`grid.min_z_m` up to `max_obstacle_height_m`, in voxels of `grid.voxel_size_m`. Each depth frame
of a camera is held to the camera's alignment and frame, then placed by the tree at its header's
timestamp: the camera's frame in the robot frame, and the robot frame in odom, each lookup
waiting up to `lookup_timeout_s`. A frame the tree cannot place is dropped and logged with the
lookup's failure kind. The grid drops each return within the base's outline, clears the cells
under the padded footprint, and, from the returns within `obstacle_max_range_m` of the camera
along the floor, marks a voxel between the two obstacle heights and a column whose rays read the
floor deeper than `max_step_down_m`: every camera sees the floor.

The grid answers one occupancy query, the cost grid of what the sensors have seen, inflated for
the padded footprint, built once per change: the planner, the collision test, the sweeps and the
slowing read it. The stop (navigation_core's port of Nav2's collision monitor) slows every command
over the larger of `time_before_collision_s` and the sum of `source_timeout_s` and the time the
base takes to rest from its fastest velocity, checking the padded footprint at each projected
step against each camera's newest points, as Nav2's collision monitor checks a source's points,
and against the cost grid along its outline, a corner off the grid counting as occupied, as Nav2's
controller costs a footprint. A camera is silent while no frame of it went into the grid within
`source_timeout_s`; a frame the tree cannot place goes into none.

The grid empties when the odometry starts over and is built again around odom's origin for the
footprint read then; clear_grid empties it. While a goal runs, and while the release burst after
it streams, the grid rolls with the base, as Nav2's local costmap rolls: it centers on the base
when the goal is accepted and with each depth frame, on where the base stood at the frame's time,
keeping what the square still holds. A goal's end lies inside the grid centered on the base when
it is accepted.

## A goal

A goal is checked in base_navigation's order (`admission.rs`): it waits up to `source_timeout_s`
for describe_base's read of the newest odometry start; its values; a target in `map`, refused with
`not_localized` while no map is loaded; a target in another frame, placed in odom by the tree at
its timestamp, refused with the lookup's kind; drive_states' silence, base_silence and each silent
camera; base_mode, parts_within_footprint, robot_stop and follower_goal.

A new goal replaces the goal in flight, which ends cancelled at once: by its caller's cancel when
one came, its result where the newest drive_states placed the robot frame, and otherwise with its
result where the robot frame stood at the newest drive_states when the new goal was accepted.
Each step ends the goal in flight by the first of its endings (`control.rs`'s `Cause`), in this
order: the caller's cancel, its result where the newest drive_states placed the robot frame; a
follow_state other than following, robot_stop and follower_goal cancelling it with the result
where drive_states first reported it, base_mode and base_silence failing it; the odometry
starting over; more than 0.5 s with no drive_states; a silent camera; a limb outside the
footprint; the passing of the goal's `timeout_s`; the goal's completion at its end, its result
where the newest drive_states placed the robot frame; then every other failure. After each
ending, completed, cancelled or failed, while the navigator's stream is live and the base follows
it, the navigator streams its release burst, zero velocity for 0.25 s and at least 3 setpoints,
then nothing; a drive_states that reports a hold ends it, and so does the next goal's first
setpoint.

### A navigation

A navigation runs in tries (`drive.rs`). A try plans with Smac 2D on its own task (`planning.rs`),
so a plan still running delays no controller step, or keeps the path it has while that path is
shorter than `near_goal_distance_m` with no pose in collision or off the grid, and follows it
(`following.rs`): the path handler, the progress checker, regulated pure pursuit and the goal
checker, in Nav2's controller server's order, planning again at `replan_rate_hz`. A plan the planner
refuses leaves the navigation no path to keep, as Nav2's ComputePathToPose leaves an empty path. The
try after the clearing recovery reads that rule on the emptied grid, where no pose is in collision,
and follows the path it keeps from the step after the clearing, on the emptied grid. A plan that
finds no path, and a follow that fails, empty the grid and run once more on the readings that arrive
after the clearing; while a follow's retry waits, its replanning runs on, as Nav2's
ComputePathToPose runs on while FollowPath retries: a path a plan finds becomes the retry's, and
after a refusal the retry has none and fails, as Nav2's controller server refuses an empty path.
When a try fails, the recoveries run (`recovery.rs`): a clearing, a turn of
`recoveries.turn_angle_rad`, a wait of `recoveries.wait_s` streaming zero velocity, and a back-up of
`recoveries.back_up_distance_m`, each once; after the back-up the navigation fails, `blocked` or
`no_path`. The navigation completes at the step the goal checker finds the goal reached.

### The moves

A spin turns as Nav2's Spin does, and a drive_on_heading and a back_up drive as Nav2's
DriveOnHeading does (`moves.rs`, `recovery.rs`, on navigation_core's ports of the two laws). A turn
commands the rate from which describe_base's angular deceleration brings the base to rest at the
end of the turn, between the slowest turn rate and the base's limit. A straight drive commands its
speed held within one step's change by describe_base's acceleration and deceleration along x of
its last command, slowed to the speed from which that deceleration brings the base to rest at its
distance, raised to the slowest speed, and at most the base's limit. Each step sweeps the padded
footprint ahead up to `simulate_ahead_time_s`. A sweep that meets an occupied cell fails the move
with `blocked`, and `timeout_s` passing first with `timeout`. The move completes at the step the
odometry measures the whole turn or distance.

## Clocks

Every time the node reads is its instance's bound clock, the simulation clock in a simulation
(control_core's `NodeClock`): the steps of the control loop, the arrivals of drive_states and of
each camera's frames, the waits of the tree's lookups and of the admission, and every timeout.
The navigator stamps each setpoint it streams on `drive_link` with its step's time on that
clock, the clock of transforms; a setpoint in the robot frame stands in it at any time, so the
follower places it as it places one with no timestamp.

## Parameters

Each parameter is described in `peppy.json5`. `tests/parameters.json` holds the values the unit
tests and the harness tests run; the launcher's fragment holds Nav2's defaults, each cited.

## Tests

Each module's unit tests run on simulated clocks or on times they set: the admission's order of
checks, the shared endings and the release burst, the tries and recoveries of a navigation with
plans run in the test, the gated planner, the turns, the straight drives and the moves, the
follow, the grid's marks, clearing and centering, and describe_base's parse and wait.

The harness tests (`tests/navigator.rs`) boot the node in-process through `setup_with` on a
simulated clock the test ticks, against a mock base of drive_link that commands each setpoint
through base_state's limiter and reports drive_states with the transform odom to base_footprint,
the robot's sensors and the cameras' fixed frames, describe_base, and depth cameras that render
boxes, tables, drops and the base's own deck, from a body that can rest tilted. The rig steps the
node through its wire alone: each step publishes the step's drive_states and frames and ticks the
clock; while a goal runs, it waits for the goal's feedback of that step or the end of its
feedback, and it keeps each setpoint by its stamp. The wire orders neither the node's taking of a
step's inputs nor a plan's end against the tick, so each test asserts what holds whichever step
these land on. They cover the refusals in base_navigation's order, the drives around what the
cameras see, the marks of a tilted base beside a drop, the recoveries, a plan held while the
controller and the stop step, the replacement, the cancels and the holds, the release burst, the
order of endings within a step, the silences, the odometry's start-over, the rolling grid, the
moves, and a clock that lags the messages or leaps.
