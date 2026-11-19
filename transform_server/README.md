# transform_server

The node that holds a robot's tree of frames and answers lookups in it for the
nodes that hold no tree, as tf2_ros's buffer server answers `LookupTransform`. It
implements `transform_lookup:v1`. Its `transforms` link binds every publisher of
the robot's tree: the robot's initializer, which publishes the transform of each
fixed joint of the model on `static_transforms`, and the robot's backbone, which
publishes the transforms of the joints it measures on `transforms`. The node keeps
both in a `RobotTree` of `control_core::robot_tree`, which holds a `FrameTree`
of the crate `frame_tree` in public-peppy-libs, and control_core's followers take
each message into it.

## Configuration

| Parameter | Meaning |
|---|---|
| `cache_window_s` | How long the node keeps each moving transform's stamps, counted back from the newest stamp of its child frame: finite and above 0, at least 1 ns. A lookup at a time older than that finds no value. Any other value stops the node at its start, naming the parameter. |

The launcher binds the node to the simulation clock when the robot's tree is
stamped on it, and leaves it on the wall clock otherwise.

## A lookup

`lookup_transform` gives where the source frame stands in the target frame, at
the request's `time` or, with none, at the latest time the chain between them
holds a value at, and the time the answer holds at. An answer from static
transforms alone holds at every time and gives no time. A request whose lookup
fails waits up to its `timeout_s` on the node's clock, the simulation clock in a
simulation, counted from when the node takes the request, and the node looks again
at each message the tree takes and a last time when the wait runs out; a `timeout_s`
of 0 looks once. A wait ends too when the node's clock reads before the request, as
after a simulation starts over. The node answers requests one at a time, in the order
they arrive, so a request can wait behind the waits of the requests before it. A node
that stops answers the request that waits at once, with the lookup as the tree
stands.

A request is checked first, in this order: an empty `target_frame` or
`source_frame`, a `time` before the Unix epoch, or a `timeout_s` that is not a
number from 0 to 1 is `rejected`, naming the value and what to send. A lookup that
fails gives the failure kind of its first failure, in the order the `transforms`
contract gives them, and a message naming the cause:

| Failure kind | Cause |
|---|---|
| `unknown_frame` | No transform the node received names a frame of the request. |
| `no_chain` | The two frames are in trees that share no frame. |
| `extrapolation` | A moving transform of the chain has no value at the time: the message names the transform, the time and the stamps the node holds of it. |
| `broken_tree` | Two publishers send the transform of a frame of the chain, or the chain runs through a cycle: the message names the frame and both publishers, or each transform of the cycle. |

The node's clock is read at each message and at each lookup. When it reads a
time before the one before it, as when a simulation starts over, the node drops
the stamps of every moving transform and keeps the static ones. When the ticks
of its simulation clock end, the node exits with an error for its restart policy
to start it again.

## Build

```sh
peppy node add /path/to/ws/nodes-hub/transform_server -sb
```

Rebuild after code changes by re-running with `--force`.

## Tests

```sh
cargo test
```

- `src/lib.rs`: a `cache_window_s` the node refuses, naming it.
- `src/clock.rs`: the end of a simulation clock's ticks, which raises the node's fault
  and stops it.
- `src/lookup.rs`: the checks of a request, the failure kind of each failure of a
  lookup, the time of an answer, and on a simulation clock ticked by hand, a wait that
  fails once the clock passes its deadline and one a node that stops ends at once.
- `tests/lookup.rs`: lookups over the wire, on a simulation clock, from a tree of two
  publishers, the initializer and the backbone: at a time between two stamps, at the
  latest time, over static transforms alone and the other way round; each failure kind
  and a request refused with `rejected`, a wait above 1 s among them; a transform from
  both publishers; and a lookup sent with the message that lets it succeed.

The other waits of a lookup, and the clock going back, are tested in
`control_core::robot_tree`.
