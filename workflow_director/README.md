# workflow_director

The node that checks workflows and runs them on the robots of the stack. It
implements `workflow_director:v1`. A workflow (`workflow/v1`) names roles and
the contracts of their skills; a run (`run/v1`) binds each role to a robot of
the fleet listing. The documents and their language are in the crate
`workflow_documents` of public-peppy-libs.

## Configuration

| Parameter | Meaning |
|---|---|
| `allowed_contracts` | The contracts that runs may command and observe, `name:tag` items separated by commas, such as `"postures:v1, limb_motion:v1"`. With none, no run starts. |
| `poll_s` | The poll interval of each step without a `poll_s` of its own. |
| `cancel_timeout_s` | How long, in wall time, a cancelled goal has to end before its robot counts as in an unknown state. |
| `journal_root` | The absolute host directory of the journals, bound into the container at the same path. |
| `journal_max_runs`, `journal_max_mib` | The journals of ended runs kept; the oldest go first past either bound. |

The launcher gives the node a daemon target `fleet` with the members `list`,
`contract`, `changes` and `lease`, and binds it to the simulation clock when
its robots read that clock.

## The fleet

The node follows the fleet with one `fleet.changes` goal and holds its leases
with one `fleet.lease` goal for its whole life. The end of that goal, whose
leases end with it, ends the node with an error, and its restart policy
`on_failure` starts it again with a new lease goal. Each change of the listing
reads the pinned document of each new contract pin, and the model of each new
robot from its `robot_identity:v1`. `robots` gives that snapshot, with the
contracts of each robot filtered to the allowed ones.

## Checking

`check` checks a workflow against the contracts of the listing: without roles,
each alias against every pin of its contract; with roles, each role against
the pins of its robot. It names each error at its field or at
`<role>.<step id>`.

## A run

A `run` goal that is well formed is accepted and gets the id `r<N>`. The run
reads a fresh fleet listing, refuses a halted stack, binds its roles (the
named roles, then each `"auto"` role to the first fitting robot in name
order, among the robots the caller offered to the director when it offered
any), and checks the workflow with those robots. For a bundle it compares
the stack with the lock of the bundle. Then it takes the lease of each robot
under the name `run r<N>`, lists the locations of each skill instance the
workflow gives a location to (`locations:v1`, whatever the allowed
contracts) and refuses a run that names one the skill does not know. Then it
checks the assumptions, waits for the confirmations of an attended run, runs
the initialization and the lanes, and at its end releases each lease before
the event `run_ends`. The lock of the run names each contract it reads: those
of its calls, observations and monitor, `locations:v1` of each skill it gives
a location, `alert:v1` of each physical robot and `robot_identity:v1` of each
robot whose role names a model.

The calls go through the codec API of peppy, so a workflow can name contracts
this node was not built with. Before each call of a member with the mark
`moves_robot`, the node asks `fleet.check_move`. A halt or the end of one of
its leases stops a run at once: every goal is cancelled and nothing more is
sent.

An incident of an attended run suspends its lane: the lane runs its safe
state, the lease of its robot opens for a take-over, and `resume` closes it,
checks the lane and runs the failed step again. An incident of an unattended
run fails its lane. An alert of severity 2 or 3 of a bound physical robot
refuses the run when it stands at the start, is an incident of the robot's
lane during the run, and refuses a resume of that lane while it stands.

`run`, `resume` and `abort` give each event of the run as feedback, and the
status of the run every 5 s; `run_events` gives every event of every run.

## The work cell

A run made from roles takes its work cell block as `work_cell` (JSON5), and a
run document gives its own `work_cell` and `watch`; a request that gives a
run document with roles or with `work_cell` is refused with `rejected`. A run
of a bundle whose work cell block gives a value of another shape than its
parameter in the bundle, or a value of a watch parameter, or whose own watch
parameter has the name of a parameter of the bundle, is refused. After the
locations, the run reads each watch parameter of the run and of its bundle
through the robots, as it reads the check of an assumption, and a value
that does not read refuses the run. The event `uncovered` and the run
result name each value of the work cell block and each watch value that no
test covered: missing, outside the envelope of the bundle, or with no
envelope. The run result gives the work cell block and each watch value.

`locations` teaches each locations:v1 instance of a robot the locations
given, or lists what each knows; a teach is refused on a robot with a lease,
naming its holder. A teach holds the robot while it runs: it takes the lease
of the robot under the name `locations`, and no run of the director binds
the robot until the teach ends.

## Journals

Each run has a journal, `<journal_root>/workflow_director/journals/r<N>.jsonl`:
the request, the inputs of its plan, the locations each skill knows, each
call with how it ended, each topic read with how old its message was, each
read that took the answer of another reader's read, each answer of the
daemon to an open or a close of a lease with when it came, each command and
each event, and its end. A run that the node did not
end when it stopped ends `interrupted` when it starts again. `journal` gives
the journal of a run, and `replay` drives the engine again from a journal and
compares its decisions with those of the run.

## Tests

```bash
cargo test
```

The engine tests run whole runs against a scripted host on a paused clock.
