"""The daemon's book as the stream traces keep it, by
`book.rs`'s rules: who halted the stack, each robot's lease and the book's
revision, which each change moves on; how the daemon serves a take and a
release of the headset; the listing of the book, and what a listing tells of
a lease and of a confirm's settle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tests.helpers import DIRECTOR, DIRECTOR_LEASE, FIRST_REVISION, HEADSET, fleet_view
from xr_commander.fleet import LEASE_NAME, FleetView, Holder, Lease

# The headset's lease of a robot, once the daemon served its take.
HEADSET_LEASE = Lease(HEADSET, LEASE_NAME, open=False)
# The director's lease of a robot, opened for a take-over.
DIRECTOR_OPEN = Lease(DIRECTOR, DIRECTOR_LEASE, open=True)


def leased_by(view: FleetView | None, robot: str, holder: Holder) -> bool:
    """Whether the listing `view` names `holder` as the holder of `robot`'s
    lease."""
    lease = None if view is None else view.leases.get(robot)
    return lease is not None and lease.holder == holder


@dataclass(frozen=True)
class Settle:
    """The earliest revision that a confirm of the hold of `epoch` answered:
    the first listing that recent settles it."""

    epoch: int
    revision: int

    @classmethod
    def answered(cls, pending: Settle | None, epoch: int, revision: int) -> Settle:
        """The settle once a confirm of the hold of `epoch` answered at
        `revision`, beside the `pending` one: the earliest revision answered
        for that hold."""
        if pending is None or pending.epoch != epoch:
            return cls(epoch, revision)
        return cls(epoch, min(pending.revision, revision))

    def listed(
        self, view: FleetView | None, robot: str
    ) -> tuple[Settle | None, int | None]:
        """The settle once `view` is listed, and the hold epoch it dooms, None
        for none: a listing as recent settles it, and dooms its hold unless it
        names the headset as `robot`'s lease holder; any other listing keeps
        it."""
        if view is None or view.revision < self.revision:
            return self, None
        return None, (None if leased_by(view, robot, HEADSET) else self.epoch)


@dataclass(frozen=True)
class Daemon:
    """The daemon's book: who halted the stack, the lease of each robot, and
    its revision."""

    halted_by: str | None
    leases: Mapping[str, Lease]
    revision: int

    @classmethod
    def start(cls) -> Daemon:
        """The book of the first listing: no halt and no lease."""
        return cls(None, MappingProxyType({}), FIRST_REVISION)

    def with_lease(self, robot: str, lease: Lease | None) -> Daemon:
        """The book with `lease` as `robot`'s, none for None; a change moves
        the revision on."""
        if self.leases.get(robot) == lease:
            return self
        others = {name: kept for name, kept in self.leases.items() if name != robot}
        leases = others if lease is None else {**others, robot: lease}
        return Daemon(self.halted_by, MappingProxyType(leases), self.revision + 1)

    def without_lease_of(self, robot: str, holder: Holder) -> Daemon:
        """The book with `holder`'s lease of `robot` ended."""
        lease = self.leases.get(robot)
        if lease is None or lease.holder != holder:
            return self
        return self.with_lease(robot, None)

    def without_leases_of(self, holder: Holder) -> Daemon:
        """The book with every lease of `holder` ended, as the end of its
        lease goal ends them."""
        book = self
        for robot in sorted(self.leases):
            book = book.without_lease_of(robot, holder)
        return book

    def director_opens(self, robot: str) -> tuple[Daemon, ...]:
        """The book after each change of the director's take-over lease of the
        free `robot`: its take, then its open."""
        assert self.halted_by is None and robot not in self.leases, (
            "the director takes and opens the lease of a free robot of a stack"
            " that is not halted"
        )
        taken = self.with_lease(robot, Lease(DIRECTOR, DIRECTOR_LEASE, open=False))
        return taken, taken.with_lease(robot, DIRECTOR_OPEN)

    def halt(self, by: str) -> tuple[Daemon, ...]:
        """The book after each change of a halt by `by`: its start, with
        every lease kept, then an end of each lease, each with the book
        empty."""
        assert self.halted_by is None, "a halt starts while the stack runs"
        start = Daemon(by, self.leases, self.revision + 1)
        ends = tuple(
            Daemon(by, MappingProxyType({}), start.revision + count)
            for count in range(1, len(self.leases) + 1)
        )
        return start, *ends

    def cleared(self) -> Daemon:
        """The book once the halt is cleared."""
        assert self.halted_by is not None, "a clear ends a halt"
        return Daemon(None, self.leases, self.revision + 1)

    def take(self, robot: str, *, goals_down: bool) -> Daemon | None:
        """The book once the daemon serves a take of `robot` by the headset,
        None for a take it refuses: while the headset's goals are down, the
        stack is halted, or another client holds the lease."""
        lease = self.leases.get(robot)
        others = lease is not None and lease.holder != HEADSET
        if goals_down or self.halted_by is not None or others:
            return None
        return self.with_lease(robot, HEADSET_LEASE)

    def release(self, robot: str) -> Daemon | None:
        """The book once the daemon serves a release of `robot` by the
        headset, None for a release of a lease the headset does not hold."""
        lease = self.leases.get(robot)
        if lease is None or lease.holder != HEADSET:
            return None
        return self.with_lease(robot, None)

    def view(self, robots: Iterable[str]) -> FleetView:
        """The listing of the book, with `robots` listed."""
        return fleet_view(
            robots,
            leases=self.leases,
            halted_by=self.halted_by,
            revision=self.revision,
        )
