"""The headset's side of the fleet: the listing its daemon gives through the
daemon target `fleet`, the two long goals that keep it, and the calls of a
lease holder.

From its start the headset keeps two goals of `robot_leases:v1` running:
`changes`, whose feedback is the fleet listing and then the listing after each
change, and `lease`, during which the headset holds leases and whose feedback
gives the end of each of them. When either goal ends, the headset lets go of
every robot and starts both again as soon as the daemon answers. While the
daemon still ends the last lease goal of the headset, it refuses a new one,
and the headset asks again after `Timing.retry_s`. Any other refusal of a
start leaves the headset unable to drive until the next launch.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Protocol

from xr_commander.bus import (
    REAL_CLOCK,
    Answered,
    CancellationToken,
    Clock,
    Failed,
    Latch,
    Outcome,
    cause,
    log,
    outcome_of,
)

# The name of the headset's leases, which every refusal of another client
# gives.
LEASE_NAME = "driven from the headset"

# What a squeeze, a press or a release meets while the daemon does not
# answer a call.
DAEMON_SILENT = "the daemon did not answer"


@dataclass(frozen=True)
class Timing:
    """How long a call of the lease handshake may take, to the daemon or to
    the stop of a backbone, and how long the headset waits before it asks
    the daemon again."""

    call_timeout_s: float = 2.0
    retry_s: float = 1.0


@dataclass(frozen=True)
class Holder:
    """A client of the fleet as the listing names it."""

    instance: str
    core_node: str


@dataclass(frozen=True)
class Lease:
    """The lease of one robot as the listing gives it."""

    holder: Holder
    name: str
    open: bool


@dataclass(frozen=True)
class FleetView:
    """One fleet listing, as the headset reads it."""

    revision: int
    robots: frozenset[str]
    leases: Mapping[str, Lease]
    # Who halted the stack, None while it is not halted.
    halted_by: str | None

    @classmethod
    def parse(cls, listing, revision: int) -> FleetView:
        """The view of the listing of a `changes` feedback."""
        leases = {
            lease.robot: Lease(
                holder=Holder(lease.holder, lease.holder_core_node),
                name=lease.name,
                open=lease.open,
            )
            for lease in listing.leases
        }
        return cls(
            revision=revision,
            robots=frozenset(robot.name for robot in listing.robots),
            leases=MappingProxyType(leases),
            halted_by=listing.halted_by if listing.halted else None,
        )


@dataclass(frozen=True)
class Free:
    """No client holds the lease of the robot."""


@dataclass(frozen=True)
class DrivenHere:
    """The headset drives the robot: under its own lease, or under the open
    lease of `open_of`."""

    open_of: str | None = None


@dataclass(frozen=True)
class Leased:
    """Another client holds the lease of the robot, under the name `name`."""

    holder: str
    name: str


@dataclass(frozen=True)
class OpenLease:
    """`holder` opened its lease of the robot for a take-over."""

    holder: str


@dataclass(frozen=True)
class NotListed:
    """The robot is not in the fleet listing, so the headset does not drive
    it."""


RobotState = Free | DrivenHere | Leased | OpenLease | NotListed


def robot_state(
    view: FleetView, robot: str, me: Holder, *, driven_under_open: bool
) -> RobotState:
    """`robot` as `view` gives it to the headset `me`; `driven_under_open`
    says whether the headset drives it under an open lease."""
    if robot not in view.robots:
        return NotListed()
    lease = view.leases.get(robot)
    if lease is None:
        return Free()
    if lease.holder == me:
        return DrivenHere()
    if lease.open:
        if driven_under_open:
            return DrivenHere(open_of=lease.holder.instance)
        return OpenLease(lease.holder.instance)
    return Leased(lease.holder.instance, lease.name)


def state_text(state: RobotState) -> str:
    """How the page and the status panel say `state`."""
    match state:
        case Free():
            return "free"
        case DrivenHere(open_of=None):
            return "driven from this headset"
        case DrivenHere(open_of=holder):
            return f"driven from this headset under the open lease of {holder}"
        case Leased(holder=holder, name=name):
            return f"leased by {holder} ({name})"
        case OpenLease(holder=holder):
            return f"open for a take-over by {holder}"
        case NotListed():
            return "not in the fleet listing"


def halted_text(halted_by: str) -> str:
    return f"the stack is halted by {halted_by} until peppy fleet clear-halt"


@dataclass(frozen=True)
class Starting:
    """The goals are starting, or starting again, and no listing came yet."""


@dataclass(frozen=True)
class Following:
    """Both goals run and the listing is current."""


@dataclass(frozen=True)
class Refused:
    """The daemon refused a goal: nothing drives until the next launch."""

    cause: str


@dataclass(frozen=True)
class DaemonLost:
    """The daemon does not answer."""


GoalsStatus = Starting | Following | Refused | DaemonLost


def status_text(status: GoalsStatus) -> str | None:
    """Why the headset cannot ask the fleet now, None while it can."""
    match status:
        case Following():
            return None
        case Starting():
            return "waiting for the fleet listing"
        case Refused(cause=refusal):
            return (
                f"the daemon refused the fleet goals until the next launch: {refusal}"
            )
        case DaemonLost():
            return "the daemon is lost"


class Fleet:
    """The latest listing and the state of the goals that keep it, on the
    node's loop; `follow_fleet` is the one writer."""

    def __init__(self, me: Holder) -> None:
        self.me = me
        self.view: FleetView | None = None
        self.status: GoalsStatus = Starting()
        # Counts the starts of the goals: a release that the daemon did not
        # answer is asked again only while the goals of the start that it
        # was first asked under run.
        self.generation = 0

    def unavailable(self) -> str | None:
        """Why a squeeze or a press cannot ask the fleet now, None while it
        can: the status is `Following` only once a listing came."""
        return status_text(self.status)


class FleetListener(Protocol):
    """What the headset does with the feedback of the two goals."""

    def fleet_changed(self, view: FleetView) -> None: ...
    def own_lease_ended(self, robot: str, why: str) -> None: ...
    def let_go_of_every_robot(self, why: str) -> None: ...


class FleetCalls:
    """The calls of a lease holder to its daemon, each bounded by the call
    timeout: an `Answered` when the daemon agrees, a `Failed` with the cause
    of its refusal, an `Unanswered` when it is silent."""

    def __init__(
        self,
        node_runner,
        *,
        take_module: ModuleType,
        release_module: ModuleType,
        check_move_module: ModuleType,
        timing: Timing,
    ) -> None:
        self._node_runner = node_runner
        self._daemon = take_module.bound_producer(node_runner)
        self._take = take_module
        self._release = release_module
        self._check_move = check_move_module
        self._timeout_s = timing.call_timeout_s

    async def take(self, robot: str) -> Outcome:
        """Take the lease of `robot` under `LEASE_NAME`."""
        request = self._take.Request(robot=robot, name=LEASE_NAME)
        call = self._take.poll(
            self._node_runner, self._daemon, request, self._timeout_s
        )
        return await outcome_of(call, self._take.Failure, f"take of {robot}")

    async def release(self, robot: str) -> Outcome:
        """Release the lease of `robot`; a refusal is logged."""
        request = self._release.Request(robot=robot)
        call = self._release.poll(
            self._node_runner, self._daemon, request, self._timeout_s
        )
        released = await outcome_of(call, self._release.Failure, f"release of {robot}")
        if isinstance(released, Failed):
            log(f"release of {robot} refused: {released.cause}")
        return released

    async def check_move(
        self, *, robot: str | None = None, instance: str | None = None
    ) -> Outcome:
        """Ask whether the headset may move `robot`, or the robot of the
        instance `instance` that it calls. A yes comes back as an `Answered`
        with the revision of the fleet that it reads, and the refusal in an
        answer as a `Failed`."""
        request = self._check_move.Request(instance=instance, robot=robot)
        call = self._check_move.poll(
            self._node_runner, self._daemon, request, self._timeout_s
        )
        what = f"move check of {robot or instance}"
        checked = await outcome_of(call, self._check_move.Failure, what)
        match checked:
            case Answered(response=answer) if answer.data.refusal is not None:
                return Failed(cause(answer.data.refusal.message))
            case Answered(response=answer):
                return Answered(answer.data.revision)
        return checked


async def follow_fleet(
    node_runner,
    fleet: Fleet,
    listener: FleetListener,
    *,
    changes_module: ModuleType,
    lease_module: ModuleType,
    timing: Timing,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Keep the two goals running until shutdown, or until the daemon
    refuses one."""
    daemon = changes_module.bound_producer(node_runner)
    # Log the first start that the daemon does not answer, and the first that
    # it refuses while it ends the last lease goal, until one is accepted.
    unanswered = Latch()
    not_yet = Latch()
    # The goals of an earlier start whose cancel got no answer: the daemon
    # refuses a new lease goal while the last one runs.
    lingering: tuple = ()
    while not token.is_cancelled():
        lingering = await _cancel_quietly(lingering, timing)
        started = await _start_goals(
            node_runner,
            daemon,
            changes_module,
            lease_module,
            timing,
            unanswered,
            not_yet,
        )
        if isinstance(started, Refused):
            fleet.status = started
            log(status_text(started))
            return
        if isinstance(started, NotYet):
            fleet.status = Starting()
            await clock.sleep(timing.retry_s)
            continue
        if started is None:
            fleet.status = DaemonLost()
            await clock.sleep(timing.retry_s)
            continue
        unanswered.clear()
        not_yet.clear()
        changes_goal, lease_goal = started
        fleet.generation += 1
        fleet.status = Starting()
        ended, error = await _read_until_an_end(
            changes_goal, lease_goal, fleet, listener
        )
        log(f"the {ended} goal of the fleet ended: {error!r}")
        fleet.view = None
        fleet.status = (
            DaemonLost() if isinstance(error, ConnectionError) else Starting()
        )
        listener.let_go_of_every_robot(
            status_text(fleet.status)
            if isinstance(fleet.status, DaemonLost)
            else f"the {ended} goal of the fleet ended"
        )
        lingering = await _cancel_quietly((changes_goal, lease_goal), timing)


@dataclass(frozen=True)
class NotYet:
    """The daemon refused the lease goal while it still ends the last lease
    goal of this process."""


async def _start_goals(
    node_runner,
    daemon,
    changes_module,
    lease_module,
    timing,
    unanswered: Latch,
    not_yet: Latch,
):
    """Both goals, accepted; a `Refused` when the daemon refused one; a
    `NotYet` while the last lease goal ends; None when it did not answer."""
    started = []
    for module in (changes_module, lease_module):
        try:
            handle = await module.ActionHandle.fire_goal(
                node_runner, daemon, timing.call_timeout_s
            )
        except Exception as e:
            unanswered.trip(
                f"the {module.TARGET_ACTION_NAME} goal of the fleet unanswered: {e!r}"
            )
            await _cancel_quietly(started, timing)
            return None
        if not handle.accepted:
            await _cancel_quietly(started, timing)
            refusal = handle.refusal
            if (
                module is lease_module
                and refusal.kind == module.FailureKind.PRECONDITION_FAILED
            ):
                not_yet.trip(f"the fleet goals start again later: {refusal.message}")
                return NotYet()
            return Refused(cause(refusal.message))
        started.append(handle)
    return tuple(started)


async def _read_until_an_end(
    changes_goal, lease_goal, fleet: Fleet, listener: FleetListener
):
    """Hand each feedback of the two goals on until one of them ends: the
    name of that goal and the error its read raised."""

    async def read_changes() -> None:
        while True:
            feedback = await changes_goal.on_next_feedback_message()
            fleet.view = FleetView.parse(feedback.listing, feedback.revision)
            fleet.status = Following()
            listener.fleet_changed(fleet.view)

    async def read_lease() -> None:
        while True:
            event = await lease_goal.on_next_feedback_message()
            if event.event == "lease_end":
                listener.own_lease_ended(event.robot, event.cause)

    readers = {
        asyncio.create_task(read_changes()): "changes",
        asyncio.create_task(read_lease()): "lease",
    }
    try:
        done, _ = await asyncio.wait(readers, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    ended = next(iter(done))
    return readers[ended], ended.exception()


async def _cancel_quietly(handles, timing: Timing) -> tuple:
    """End goals that may still run, a goal already ended answering that it
    has nothing to cancel: the goals whose cancel got no answer."""
    unanswered = []
    for handle in handles:
        try:
            await handle.cancel_goal(timing.call_timeout_s)
        except Exception as e:
            log(f"a goal of the fleet could not be cancelled: {e!r}")
            unanswered.append(handle)
    return tuple(unanswered)
