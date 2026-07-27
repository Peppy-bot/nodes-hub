# Split-compute manipulation

The three mock nodes behind peppy's `Federation` guide and its multi-daemon
E2E: a fast reactive policy that belongs on the robot, a slow deliberative
planner that belongs on a datacenter accelerator, and a recorder that watches
what the robot actually did.

They are mocks in the shape of the real components, in the style of the
`example_robot` family. The point they exist to demonstrate is the WIRING, the
PLACEMENT, and the LIFECYCLE across a daemon boundary, not inference quality, so
the planner emits a canned subgoal on a timer and the policy reports which
subgoal it is currently servoing on. Deterministic, no model weights in CI, and
no dependence on host speed.

Between them the three exercise each cross-daemon relationship kind exactly
once:

- a **producer link**: the planner consumes the same camera the policy reads
  locally, named identically on both sides;
- a **pairing**: `task_delegation`, bidirectional because escalation is a
  conversation rather than a feed;
- an **observation**: the recorder taps the executor side of that pairing
  without joining it, so it cannot perturb control.
