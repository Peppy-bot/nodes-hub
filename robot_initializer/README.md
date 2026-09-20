# robot_initializer

The node that brings one robot into being, says who it is and says when it is
ready. It serves a robot of any model: the launcher names the model in the
`model` parameter, `openarm_v2` for an OpenArm or `so101` for an SO-101, and
nothing else in the node is about one robot rather than another.

## Joining a simulation

A robot whose launcher binds a simulation to this node's `simulation` slot
joins it before it does anything else. It attaches as the copy it runs as,
naming its `model` verbatim, the id the simulation's catalogue knows the robot
by, and the placement it was given, or none for a free spot of the engine's
choosing, and the engine answers with the limbs it gave the robot. The node
keeps no list of models: a simulation refuses a model it does not stand,
naming the ones it does. The goal stays open for as long as the robot is in
the scene,
and shutting the node down takes the robot out, within peppy's shutdown
grace. A `placement` names where the robot's base stands; the default lets
the engine choose a free spot, which is what MuJoCo, whose scene places its
one robot, takes.

The robot's limbs reach the engine through the backbone's pairings, one pair
per limb on the engine's slots, and the engine tells one robot's limbs from
another's by the copy each pair belongs to, which is the name this node
attached under. Isaac Sim and Waldo stand as many robots as their machines
can run; MuJoCo stands one.

A robot that drives its own hardware leaves the `simulation` slot vacant; its
`model` is identity only, the one `get_identity` answers.

## The readiness gate

The node exposes the robot-level `is_ready` service the backbone gates on,
true only once everything that answers for the robot reports ready. On a
real robot the launcher binds on `limbs` whatever drivers answer for it: one
per limb on an OpenArm, the single `so101_follower` on an SO-101. On a
simulated one the simulation bound on `simulation` answers for the whole
robot: standing, and holding a pair for every limb its model has. One initializer serves the real robot and every simulation. An
answer that dies or becomes unreachable flips the robot back to not-ready on
the next poll, and a launcher that binds neither slot is refused at start.

Readiness comes after joining: a robot the simulation refused never serves
`is_ready` at all, so the backbone never gates on a readiness the scene
cannot back.

## Who the robot is

The node exposes `get_identity`, which answers the name the robot stands
under, the model it is and the core node hosting it. The name is the copy the
launch put the robot in, or the node's own instance id for a robot launched
outside one, and the model is its `model` parameter as the launcher wrote it.
It is the name and the model the node attaches under, so the
entry a simulation lists for this robot carries the same `robot`. The answer
is fixed when the node starts and is served from then on, before the robot
joins a scene and whether or not it is ready.

## Build

```sh
peppy node add /path/to/ws/nodes-hub/robot_initializer -sb
```

Rebuild after code changes by re-running with `--force`. When the build
finishes, `peppy stack list` shows the node at `Stage: Ready`.

## Run

Every declared slot must be bound when an instance starts, so the node starts
through a launcher, which names the drivers of its limbs on a real robot and
the simulation it joins on a simulated one. The OpenArm fragments in
[launchers-hub](https://github.com/Peppy-bot/launchers-hub/tree/main/openarm/fragments)
do exactly that for an OpenArm; the [OpenArm README](../openarm/README.md)
walks through the whole sequence:

```sh
peppy stack launch openarm_simulation
```

Watch it come up with:

```sh
peppy node info robot_initializer:v1
```

## Troubleshooting

**`is_ready` never becomes true**
Something answering for the robot is not reporting ready, and an unreachable
answer counts as not ready. On a real robot that is one of the limb drivers;
in a simulation it is the simulation, which reports the robot ready only
once it stands and every one of its limb pairs is held. This node reports
only the aggregate; find the holdout in the drivers' own logs (`peppy node
info <node>:v1` per limb node) or the simulation's.

**the node stops right after it starts, saying the simulation refused it**
The engine refused to stand this robot, and its reason is in the message: a
model its catalogue does not carry, a placement another robot occupies, or
a name that already stands. `peppy stack list` reports the instance failed.
Fix the robot's `model` or its `placement`, or make room in the world, then
`peppy stack join` the copy again.

**the node stops right after it starts, saying `model` names the robot's model**
The launcher left `model` empty. Give it the id the simulation's catalogue
names the robot by, `openarm_v2` or `so101` for example.

**the node stops soon after it starts, saying the simulation did not stand it**
The engine admitted the robot and then could not stand it (its files could
not be fetched, or its model could not be built into the scene); the
engine's reason is in the message, and `peppy stack list` reports the
instance finished. Fix what the reason names, then `peppy stack join` the
copy again.

**the robot left the scene and its node stopped**
A stay ends when the engine takes the robot out or its limbs hold no pair
for the engine's lease, and the node stops with it: a robot that is not in
the scene has no readiness to serve. `peppy stack list` reports the instance
finished, with the engine's own account of why in the node's run log.
Nothing brings the copy back on its own: `peppy stack join` it again.
