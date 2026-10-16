"""The robots paired into the headset, and the one it drives.

One headset serves a stack: the backbone of every robot the headset leads
pairs into its slots as the robot joins, each pair carrying the copy its
robot runs in. The hands drive one robot at a time, the one chosen on the
headset's page, and are silent toward every other pair. With no choice, or
the chosen robot not paired, the hands stay on the robot they drive while
it is paired, and otherwise go to the robot that first paired.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from types import ModuleType


def robot_of_peer(member) -> str:
    """A pair's robot: the copy its backbone runs in, or the backbone's
    instance id for one launched outside any copy."""
    return member.copy if member.copy is not None else member.info.producer.instance_id


def robot_of_producer(member) -> str:
    """A bound producer's robot, by the same rule as a pair's."""
    return member.copy.name if member.copy is not None else member.producer.instance_id


class PairedRobots:
    """The live view of the slots, by robot, and the choice among them.

    Read per tick on the node's loop and per request on the page's server
    thread, under one lock. The slots follow the stack's joins and removals,
    so a pair that arrives or goes is seen on the next read. The robots are
    listed in name order; a robot that joins takes nothing from the one
    driven, the hands fall back to the first that paired, and the chosen
    robot returning takes the hands back.
    """

    def __init__(
        self,
        node_runner,
        arms: Mapping[str, ModuleType],
        grippers: Mapping[str, ModuleType],
    ) -> None:
        """`arms` and `grippers` are the pose and gripper slots' modules by
        the hand that drives them."""
        self._node_runner = node_runner
        self._arms = dict(arms)
        self._slots = (*arms.values(), *grippers.values())
        self._lock = threading.Lock()
        self._choice: str | None = None
        # The robot the hands drove at the last read.
        self._driven: str | None = None
        # Every robot ever paired, in the order it first paired.
        self._first_paired: list[str] = []

    def names(self) -> list[str]:
        """Every paired robot once, in name order."""
        with self._lock:
            return self._names_locked()

    def _names_locked(self) -> list[str]:
        paired = {
            robot_of_peer(member)
            for module in self._slots
            for member in module.peers(self._node_runner)
        }
        self._first_paired.extend(
            name for name in sorted(paired) if name not in self._first_paired
        )
        return sorted(paired)

    def standing(self) -> tuple[list[str], str | None]:
        """The paired robots and the one the hands drive, read together: the
        robot chosen while it is paired, else the one they drove while it is
        paired, else the first that paired; None with nothing paired."""
        with self._lock:
            names = self._names_locked()
            standing = (self._choice, self._driven, *self._first_paired)
            self._driven = next((name for name in standing if name in names), None)
            return names, self._driven

    def chosen(self) -> str | None:
        """The robot the hands drive, as `standing` reads it."""
        return self.standing()[1]

    def arm_sides(self, robot: str) -> frozenset[str]:
        """The hands `robot` pairs an arm with."""
        return frozenset(
            side
            for side, module in self._arms.items()
            if any(
                robot_of_peer(member) == robot
                for member in module.peers(self._node_runner)
            )
        )

    def peer_for(self, module: ModuleType):
        """The chosen robot's pair on `module`'s slot, or None while that
        robot holds no pair there (a one-armed robot on the other side)."""
        chosen = self.chosen()
        if chosen is None:
            return None
        for member in module.peers(self._node_runner):
            if robot_of_peer(member) == chosen:
                return member.info
        return None

    def producer_for(self, module: ModuleType):
        """The producer on `module`'s contract slot that serves the chosen
        robot, or None while none does."""
        return self.producer_of(module, self.chosen())

    def producer_of(self, module: ModuleType, robot: str | None):
        """The producer on `module`'s contract slot that serves `robot`, or
        None while none does: the robot's own, else one outside any copy
        that is no robot itself (a recorder the launcher deploys beside the
        robots serves each of them)."""
        if robot is None:
            return None
        members = module.bound_members(self._node_runner)
        own = (m for m in members if robot_of_producer(m) == robot)
        # A robot that ever paired is a robot, its pairs restarting or not.
        robots = self._first_paired
        shared = (
            m for m in members if m.copy is None and robot_of_producer(m) not in robots
        )
        member = next(own, None) or next(shared, None)
        return member.producer if member is not None else None

    def choose(self, name: str) -> bool:
        """Drive `name` from the next tick on; refused, changing nothing,
        while no pair carries it."""
        with self._lock:
            if name not in self._names_locked():
                return False
            self._choice = name
            return True
