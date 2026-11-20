# room_mapper

The node that maps the rooms a robot's mobile base drives in with the base's 2-D lidar,
saves each map under a name, loads a saved map, and keeps the base localized in the loaded
map. It implements `room_map:v1` on `mapping_core` of public-peppy-libs, which ports
slam_toolbox 2.10.0's mapping and localization and Nav2 1.5.2's AMCL global localization,
and it publishes `map` to `odom` on `transforms:v1`. It is GPL-3.0-or-later, as
`mapping_core` is, and it is the only node that links `mapping_core`.

Its links:

| Link | Contract | What the node does with it |
|---|---|---|
| `room_map` | `room_map:v1` (implements) | serves its eight services; publishes `map_poses` and `map_grids` |
| `transforms` | `transforms:v1` (implements) | publishes `map` to `odom`; nothing on `static_transforms` |
| `lidar` | `lidar_2d:v1` | reads `scan_stream` |
| `base` | `base_state:v1` | reads `base_states`; calls `describe_base` at each odometry start |
| `sensor_mounts` | `sensor_mounts:v1` | checks once per lidar frame that the lidar is fixed to the base |
| `transform_lookup` | `transform_lookup:v1` | looks the lidar's frame up in `base_footprint` |

## Configuration

`peppy.json5` gives each parameter with its meaning. None has a default; the launcher's
fragment gives each value with its derivation. `mapping`, `localization` and `amcl` take
slam_toolbox's and AMCL's parameters by their own names; `matching_limits` and
`search_limits` are `room_map`'s limits; `footprint_padding_m` pads `describe_base`'s
footprint for the rays the node leaves out; `source_timeout_s` is `room_map`'s source
timeout; `maps_directory` is bound into the container at its own path. A parameter the
node does not take stops it at its start, the error naming the parameter.

## The lidar and the odometry

The node maps with one 2-D lidar fixed to the base, as slam_toolbox and AMCL read a
laser's pose once. At the first scan of a frame, `describe_sensors` must name that frame a
`lidar_2d` sensor whose `carried_by` is empty, or the node stops naming the lidar; then
`lookup_transform` places the frame in `base_footprint`, asked again at each scan until it
answers. Until it answers, the node takes no scan, logs each failed lookup with its failure
kind, and the members that need scans answer `resource_unavailable` naming that failure.
A scan or a `base_states` counts as arrived no earlier than its timestamp, as a
simulation's clock reaches the node on a topic of its own, behind the messages stamped on it.

The node keeps `base_states` for the current odometry start and the one before it, back
10 s from the newest, tf2's default cache time. It takes a scan once two measurements of
one start enclose its whole sweep, its earliest ray to its latest, a clockwise sweep's last
ray before its timestamp, and places each ray where the base stood at the ray's time,
interpolated as `transforms` interpolates. A ray that ends within `describe_base`'s
footprint grown by `footprint_padding_m` is read as unmeasured. Scans wait in arrival order
for their odometry; a scan that no start will enclose is dropped and logged.

When the odometry starts over, the estimate goes on in the frame it began in: the base
counts as still between the newest scan before the new start and the first scan after it,
and each later scan's odometry is carried into that frame. `map` to `odom` is published in
the odometry of the scan it places, stamped at that scan.

## The room

The loaded map is the map being built while mapping runs, and otherwise the map that
`load_map` or `finish_mapping` loaded last, until `remove_map` removes it. A new map's map
frame is the odometry frame where mapping started. Maps are saved by name in
`maps_directory` through `mapping_core`'s `MapStore`, each save and removal whole.

- A call takes the scans that arrive at the node after it, as slam_toolbox and AMCL take
  the next scan they receive.
- Mapping runs slam_toolbox's synchronous mapping on each scan after `start_mapping`; the
  base is localized while it runs.
- `set_initial_pose` waits for the first scan after the call in the odometry it was called
  in, carries the target from its timestamp, or from the call, by that odometry, and starts
  slam_toolbox's localization at the match within the search. A second call waits for the
  first to answer.
- `reinitialize_global_localization` runs AMCL's global localization on the node's one
  drand48 stream, seeded once with `amcl.random_seed`, as AMCL's node seeds its stream; the
  search hands the pose to the localization once its deviations lie within the search
  limits, and ends when no scan arrives within the source timeout.
- Each scan an estimate takes answers `map_poses`: an update's own pose, or the newest
  update carried by the odometry. `map` to `odom` starts over at the first answer after a
  `load_map`, a `start_mapping` or a `remove_map`, and at each answer whose scan's odometry
  start differs from the one before.
- `map_grids` carries the loaded map's grid when a map is loaded or finished, and while
  mapping runs at most once per `mapping.map_update_interval` once its scans changed; none
  of a new map before its first scan.

The members refuse in `room_map`'s order: `start_mapping`, `set_initial_pose` and
`reinitialize_global_localization` first wait for the read of `describe_base` at the
newest odometry start, then refuse `rejected`, `precondition_failed` and
`resource_unavailable` in turn, each message naming its cause and what to do.

## License

GPL-3.0-or-later, `COPYING`. `NOTICE` names what the image ships beside the binary:
`COPYING`, `NOTICE` and `mapping_core`'s `NOTICE` as `MAPPING_CORE_NOTICE`, the BSD notices
of the code `mapping_core` translates.

## Build

```sh
peppy node sync
peppy node build
```

## Tests

`cargo test` runs the unit tests and the harness tests: `tests/room_map.rs` (a map built
while a mock base drives a drawn room, saved, listed and loaded; `start_mapping`'s
refusals; commanded odometry; no scan before the lidar's place is known; a lidar that a
part carries) and `tests/localization.rs` (`set_initial_pose`'s match and refusals, a
second call, no scan within the source timeout, `get_map_state`'s flags, the wait for
`describe_base` after the odometry starts over, and `map` to `odom` across that start
over), on the rig of `tests/common`.

`tests/recorded_drives.rs` runs on a developer's machine alone (`cargo test -- --ignored`):
it replays a lidar recording of Waldo's from ros_reference's recording cache through the
node's topics: the map of a loop through two rooms places the walls of the lap's last scan
within a cell of their true place, and after a restart, `load_map` and a `set_initial_pose`
0.3 m and 0.3 rad off the true pose match within a cell.
