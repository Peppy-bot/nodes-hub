# openarm_sim_attachment

One OpenArm's seat in a simulation that hosts several.

Toward the robot the node offers the four limb pairings a simulation offers
a robot it hosts alone: `left_arm` and `right_arm` (`joint_link`),
`left_gripper` and `right_gripper` (`gripper_link`), all in the follower
role, so `openarm_sim_arm` and `openarm_sim_gripper` lead them exactly as
they lead an engine's own slots and everything above them (initializer,
backbone, commander, recorder) is wired as it is on hardware.

Toward the simulation it holds one seat of the `simulation_robot:v1`
contract:

- it fires the `attach` goal naming the model of its `hardware_version`
  (`openarm_v1` or `openarm_v2`) and, unless `placement.auto` is on, where
  the robot stands;
- it sends every limb's latest setpoint as one `command` at
  `command_rate_hz`, whether or not anything changed, because that call is
  also the robot's heartbeat: a robot whose commands stop for the
  simulation's lease leaves the scene;
- it publishes the state that comes back as the goal's feedback on the limb
  it measures, keeping the simulation's capture timestamp.

The simulation tells robots apart by the seat that commands it, so nothing
on the wire carries a robot id and no robot learns of another. The node's
life is the robot's stay: it fails to start when the simulation refuses the
robot, gives the seat back when it shuts down, and stops when the stay ends
so the runtime restarts it and the robot rejoins.

## Parameters

| Parameter | What it says |
|---|---|
| `hardware_version` | `v1` or `v2`: which OpenArm the simulation stands for this robot |
| `command_rate_hz` | how often the limbs' setpoints go out (default 100) |
| `placement.auto` | let the simulation park the robot on a free spot (default) |
| `placement.x`, `.y`, `.z`, `.yaw` | where the robot stands when `auto` is off, in metres and radians |

## Running it

The [OpenArm fleets](https://github.com/Peppy-bot/launchers-hub/tree/main/openarm)
deploy one per simulated robot; a launch or a `peppy stack join` of
`openarm_v1_sim` or `openarm_v2_sim` brings its seat with it. Waldo is the
simulation that seats robots today.

```sh
peppy node add /path/to/ws/nodes-hub/openarm/sim_attachment -sb
peppy stack launch openarm_sim_fleet
peppy stack join openarm_v1_sim -i bravo
peppy stack remove bravo
```
