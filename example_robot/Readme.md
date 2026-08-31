# Example robot

Reference nodes for a fictional robot, written to be read rather than deployed.
Each is a complete, independent node; together they show how the pieces of a
real stack fit.

| Node | What it does |
|---|---|
| `my_python_robot_arm` / `my_rust_robot_arm` | Drives three joints, publishes `joint_states`, exposes a `move_arm` action. The same node in two languages. |
| `my_python_robot_backbone` / `my_rust_robot_backbone` | Coordinates two arms, one per slot (`left_robot_arm`, `right_robot_arm`), behind its own `move_arm` action. |
| `my_python_robot_brain` / `my_rust_robot_brain` | Consumes a camera by contract and commands the backbone. |
| `reactive_policy`, `deliberative_planner`, `episode_recorder` | A manipulation stack split across two machines. See below. |

## Split-compute manipulation

Three of the nodes above are meant to run together, deliberately split across
machines: a fast reactive policy on the robot, a slow deliberative planner
wherever the accelerator is, and a recorder beside the planner.

They are mocks in the shape of the real components. The planner runs no
inference and the policy servos on nothing, so what the three demonstrate is the
wiring, the placement, and the lifecycle across a machine boundary, not
inference quality. That is also what makes them safe to drive from peppy's
multi-daemon E2E: no model weights, and no behavior that depends on how fast the
host is.

| Node | Runs on | Role |
|---|---|---|
| `reactive_policy:v1` | the robot | Closes a 200 Hz servo loop against local hardware, and escalates what it cannot resolve. |
| `deliberative_planner:v1` | the accelerator | Plans over tens of seconds at ~1 Hz and pushes subgoals back down. |
| `episode_recorder:v1` | the accelerator | Records what the robot actually did, for later training. |

Between them, each of the three cross-machine mechanisms appears exactly once.

- **Producer link.** The planner's `scene` slot is bound to the same camera
  instance the policy reads locally. Both name it identically; neither records
  which machine it runs on.
- **Pairing.** `deliberation` (in
  [`pairings-hub`](https://github.com/Peppy-bot/pairings-hub), roles `planner`
  and `executor`) connects the policy and the planner. It is a pairing rather
  than two producer links because the relationship is bidirectional and
  exclusive: each side holds a pinned view of exactly one counterpart.
- **Observation.** The recorder taps the executor side of that pairing without
  joining it. It claims no endpoint and emits nothing, so however it behaves it
  cannot perturb control.

The policy echoes the subgoal it adopted back to the planner in
`situation.active_subgoal_id`, so adoption is reported by the side that adopts
rather than inferred by the side that sent. Delivery is not adoption, and an
adopted subgoal can lapse on the executor's own staleness bound without the
planner sending anything. That field is also what lets the recorder, party to
neither side of the pair, tell a newly adopted subgoal from a redelivery.

### Degradation

The uplink is an enhancement path, not a dependency. A subgoal is authoritative
for `subgoal_ttl_ms` after the policy adopts it; past that the policy falls back
to its own local behavior. That bound is the node's own, because an unreachable
daemon cannot send a dissolution notice, so freshness is never something the
runtime can promise on the policy's behalf.

The servo path is local by construction: the launcher places the camera, the
arm, and the policy on one core node, so no placement choice can put a network
inside the 200 Hz loop.

### Running it

These three are the nodes the Peppy federation guide walks through, driven by
the `split_compute_manipulation` launcher:

```bash
peppy stack launch --place robot_onboard@self \
                   --place cloud_inference@cn-atlas-h100 \
                   split_compute_manipulation
```

To run the whole topology on one machine, with no second machine and no uplink:

```bash
peppy stack launch --local split_compute_manipulation
```
