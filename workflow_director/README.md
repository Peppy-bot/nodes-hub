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
any, and simulated in an unattended run; see the sign-off records for the
physical robots of an unattended run), and checks the workflow with those
robots. For a bundle it compares
the stack with the lock of the bundle. Then it takes the lease of each robot
under the name `run r<N>`, lists the locations of each skill instance the
workflow gives a location to (`locations:v1`, whatever the allowed
contracts) and refuses a run that names one the skill does not know. Then it
checks the assumptions, refuses an unattended run that does not confirm each
assumption without a check (naming the first one: no person confirms it),
waits for the confirmations of an attended run, runs
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

`locations` teaches each locations:v1 instance of a robot the locations
given, or lists what each knows; a teach is refused on a robot with a lease,
naming its holder. A teach holds the robot while it runs: it takes the lease
of the robot under the name `locations`, and no run of the director binds
the robot until the teach ends.

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

## Unattended runs on physical robots

An attended run may bind physical robots, and goes on with values that no
test covered. An unattended run that binds a physical robot needs, before
its initialization: `versions: "refuse"` (a run document with
`versions: "warn"` is refused), checked before the lock; then a bundle
whose validation gives an envelope, with each value of its work cell block
inside it, and a sign-off record that matches the run; then, after the
locations, each watch value inside the envelope. Each refusal names its
cause. A run whose caller confirms assumptions binds no physical robot.

## Sign-off records

A sign-off record names what a trial of a bundle tested: the bundle digest,
the core node of the stack, the binding with the name, the model and the
core node of each bound robot, the versions of the director, of the nodes of
the bound robots and of the node of the anomaly monitor, the incidents
policy and the peppy version. Each is a file,
`<journal_root>/workflow_director/sign_offs/<trial>.json`, which the director
reads at its start; a record that does not read stops the start, naming its
file. A run matches a record when each of these is the same, or when only
its incidents policy differs and is stricter: the same monitor, and each
severity of its `continue_on_severity` also in that of the record. A node
with no commit matches no record. The refusal of an unattended run that no
record matches names how each record of its bundle differs.

In an unattended run, `"auto"` binds physical robots only as one whole
record of the bundle binds them: the first record, in the order of the robot
names of its roles in role name order, whose binding agrees with the named
roles and whose robots are all present and free by the rule of `"auto"`.
With no such record, `"auto"` binds only simulated robots. `forget_robot`
removes each record that binds a robot, after a change of its body, and
gives how many it removed.

## Trials

`trial` runs a trial (`trial/v1`) of a bundle on the physical robots of the
stack, and gets the id `t<N>`. The director refuses it (`trial_refused`)
before its first run, naming each cause, unless the stack is not halted, the
bundle opens with its digests and has an envelope, each value of its work
cell is inside the envelope, each assumption of the workflow has a check,
each bound robot is physical and held by no run, teach or other trial of
the director, the workflow holds for the robots, and the stack matches the
lock of the bundle with no version policy. A trial holds its robots from the
check of their holders to its end, so a trial in its other checks refuses
neither another trial nor a run. The trial then takes the lease
of each robot under the name `trial t<N>` (a refused take refuses it, naming
the robot and the cause), holds them to its end, and removes each sign-off
record of its bundle and binding.

Each run has four steps. While the operator prepares the work cell, the
leases stand open for a take-over, and `prepares` gives the number of the
run and the measures so far. `next_run` closes the leases and starts the
run, whose `started_by` is the trial: the run uses the leases of the trial,
takes and releases none, and has the checks of an unattended run on physical
robots but the sign-off record. It evaluates every check, a false one
included, gives `checks_to_judge`, and waits for `judge_checks`: `starts`,
`not_ready` (the run does not count) or `false_check` (the run counts as not
completed, and each wrong check as a false check). The run then runs
unattended, and `suspend` refuses it, naming the trial. Once it ends,
`judge_ending` counts it: completed when it completed, a false success when
it completed and its task is not done, and each emergency stop and other
unsafe event as an unsafe event. A run that the director refuses on a watch
value outside the envelope does not count (its refusal names that cause,
`watch_outside_the_envelope`): the work cell is not ready. Any
other run that ends before it starts counts as not completed. A run that
does not start opens the leases again, and `next_run` fails with
`attempt_failed`. A cancel of `next_run`, or the loss of its sender, aborts
its run. Once the trial ends, `judge_ending` takes no judgement.

The trial ends after its last run that counts, or once the runs left cannot
meet the thresholds of the bundle (plain rates and counts). It then releases
its leases, keeps a sign-off record when its runs meet the thresholds, and
gives `trial_ends` with its report: the trial document, each run with its
ending, its journal and the judgements of the operator, the four measures,
the core node, whether it signs off, and its sign-off record. Its goal
completes only for a sign-off; it ends cancelled when its caller cancels it
or is gone, and fails with `trial_not_signed_off`, `trial_halted` (a halt)
or `trial_interrupted` (a lease that ended with no halt). A halt refuses
every new trial, and a trial that ends early signs off nothing.

Each trial has its record, `<journal_root>/workflow_director/trials/t<N>.json`,
and a trial that the director did not end when it stopped ends interrupted
when it starts again. Two trials that start together get ids of their own.
`runs` lists the trials, and `status` gives the report of a trial `t<N>`:
while the trial runs, each of its runs without its journal, which `journal`
gives; once the trial ended, with the journal of each run.

## Journals

Each run has a journal, `<journal_root>/workflow_director/journals/r<N>.jsonl`:
the request (of a bundle, the bundle document and the workflow alone: the
bundle document names each other file by its digest), the inputs of its plan, the locations each skill knows, each
call with how it ended, each topic read with how old its message was, each
read that took the answer of another reader's read, each answer of the
daemon to an open or a close of a lease with when it came, each command and
each event, and its end. A run that the node did not
end when it stopped ends `interrupted` when it starts again. `journal` gives
the journal of a run, and `replay` drives the engine again from a journal and
compares its decisions with those of the run. The journal of a run of a
trial stays with the report of its trial: the bounds of the journals neither
count nor remove it.

## Tests

```bash
cargo test
```

The engine tests run whole runs against a scripted host on a paused clock.
The tests of `tests/` run the director over the wire, against a scripted
fleet and a scripted arm: `serve.rs` its runs, and `validation.rs` the work
cell, the gates of an unattended run on a physical robot and the trials.
They need `PEPPY_ZENOHD_PATH` to name a `zenohd` binary.
