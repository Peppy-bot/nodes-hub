# openarm_initializer

The node that brings one OpenArm into being and says when it is ready.

## The seat

A robot whose launcher binds a simulation to this node's `simulation` slot
takes its seat there before it does anything else. It attaches as the model
its `hardware_version` names, at the placement it was given or on a free spot
the engine picks, and the engine answers with the limbs it gave the robot.
From then on the four limb pairings above this node carry the robot: the
relays' setpoints go out as one command at `command_rate_hz`, and every
measured state the engine feeds back is published on the limb it measures.

That command is also the robot's heartbeat. A robot whose commands stop for
the engine's lease leaves the scene, and shutting the node down gives the seat
back at once, so the body leaves with the robot.

The engine tells its robots apart by the seat that commands them, so nothing
on a limb carries a robot id and one engine hosts as many OpenArms as there
are seats. Toward the robot the seat plays the follower role of every limb's
pairing, which is the surface a simulation offers a robot it hosts alone, so
the relays bind to it as they bind to such a simulation.

A robot that drives its own hardware leaves the `simulation` slot vacant and
takes no seat.

## The readiness gate

Seated or not, the node then polls the per-limb `is_ready` of the four
`component_ready` producers the launcher binds (two arms, two grippers) and
exposes the robot-level `is_ready` service the backbone gates on, true only
once every limb reports ready. A component that dies or becomes unreachable
flips the robot back to not-ready on the next poll.

The hardware drivers and the simulation relays implement the same per-limb
contract, so one initializer serves the real robot and every simulation.

Readiness comes after the seat: a robot the simulation has not stood serves
no `is_ready` at all, so the backbone never gates on a readiness the scene
cannot back. The relays report themselves ready once the simulation's first
state has reached them, which is state this node publishes, so the order runs
one way: seat, then limbs, then the robot.

## Build

```sh
peppy node add /path/to/ws/nodes-hub/openarm/initializer -sb
```

Rebuild after code changes by re-running with `--force`. When the build
finishes, `peppy stack list` shows the node at `Stage: Ready`.

## Run

Every declared slot must be bound when an instance starts, so the node starts
through a launcher, which links its four readiness slots to the concrete arm
and gripper instances and, for a simulated robot, its `simulation` slot to the
simulation and its four limb slots to the relays. The OpenArm fragments in
[launchers-hub](https://github.com/Peppy-bot/launchers-hub/tree/main/openarm/fragments)
do exactly that; the [top-level README](../README.md) walks through the whole
sequence:

```sh
peppy stack launch openarm_simulation
```

Watch it come up with:

```sh
peppy node info openarm_initializer:v1
```

## Troubleshooting

**`is_ready` never becomes true**
One of the four limbs is not reporting ready, and an unreachable component
counts as not ready. This node reports only the aggregate; find the holdout in
the limb instances' own logs (`peppy node info <node>:v1` per limb node).

**the node stops right after it starts, saying the simulation refused it**
The engine would not stand this robot, and its reason is in the message: a
model its catalogue does not carry, or a placement another robot occupies.
`peppy stack list` reports the instance failed. Fix the robot's
`hardware_version` or its `placement`, or make room in the world, then
`peppy stack join` the copy again.

**the robot left the scene and its node is failed**
A seat ends when the engine takes the robot out or its commands stop for the
engine's lease, and the node stops with it rather than serving a readiness
the scene no longer backs. Nothing brings the copy back on its own:
`peppy stack join` it again.
