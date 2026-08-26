# openarm_robot_initializer

The robot's readiness gate. It polls the per-limb `is_ready` of the four
`openarm_hardware_ready` producers the launcher binds (two arms, two grippers)
and exposes the robot-level `is_ready` service the backbone gates on, true only
once every limb reports ready. A component that dies or becomes unreachable
flips the robot back to not-ready on the next poll.

The node is engine-agnostic: the hardware drivers and the sim bridge nodes
implement the same per-limb contract, so this one initializer serves every
backend. Loading the simulated world is the sim engine nodes' business
(`openarm_sim_mujoco` / `openarm_sim_isaac`).

## Build

```sh
peppy node add /path/to/ws/openarm-nodes/openarm_robot_initializer -sb
```

Rebuild after code changes by re-running with `--force`. When the build
finishes, `peppy stack list` shows the node at `Stage: Ready`.

## Run

Every declared slot must be bound when an instance starts, so the node starts
through a launcher, which links its four slots to the concrete arm and gripper
instances. The launchers in
[launchers-hub/openarm](https://github.com/Peppy-bot/launchers-hub/tree/main/openarm)
do exactly that; the [top-level README](../README.md) walks through the whole
sequence:

```sh
peppy stack launch /path/to/ws/launchers-hub/openarm/openarm_v2_teleop_mujoco.json5
```

Watch it come up with:

```sh
peppy node info openarm_robot_initializer:v1
```

## Troubleshooting

**`is_ready` never becomes true**
One of the four limbs is not reporting ready, and an unreachable component
counts as not ready. This node reports only the aggregate; find the holdout in
the limb instances' own logs (`peppy node info <node>:v1` per limb node).
