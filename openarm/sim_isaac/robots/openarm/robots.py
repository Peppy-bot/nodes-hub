#!/usr/bin/env python3
"""The robots standing in the scene.

A robot is reserved when its attach goal is admitted and stands once the
engine has put it on the stage, so a limb commanded while the scene is still
recompiling keeps its latest setpoint until the robot stands. Each robot's
limbs reach it through
its own pairs, and the copy the engine reads on a pair is the name the robot
attached under, which is how one robot's limbs are told from another's.

A robot stays while its limbs are connected: the pairs dissolve when its
nodes stop, and a robot holding none of them past its lease leaves the scene.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Caller:
    """Who attached a robot: the instance peppy delivered the message from,
    on the core node it runs on. That pair is unique across the mesh, so two
    stacks whose instances happen to share a name never reach each other's
    robot."""

    core_node: str
    instance_id: str

    def __str__(self) -> str:
        return f"{self.instance_id}@{self.core_node}"


@dataclass(frozen=True)
class Limbs:
    """A model's limbs in the order the engine's own limb slots carry them,
    and the joints each arm carries."""

    arm_names: tuple[str, ...]
    arm_joints: tuple[int, ...]
    gripper_names: tuple[str, ...]


@dataclass
class Robot:
    """One robot in the scene. `limbs` arrives when the engine has stood it,
    and until then the robot takes no command."""

    name: str
    model: str
    caller: Caller
    limbs: Limbs | None = None
    # When this robot last held a limb pair, set as it is reserved so a robot
    # that never pairs gives its name back once its lease lapses.
    last_paired_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def standing(self) -> bool:
        return self.limbs is not None


class Registry:
    """The robots in the scene, by the name each stands under."""

    def __init__(self) -> None:
        self._robots: dict[str, Robot] = {}
        self._lock = threading.Lock()

    def admit(self, name: str, model: str, caller: Caller, now_s: float) -> bool:
        """Admits this caller's robot under `name`, and says whether it was
        already standing. Raises when a robot of another caller stands under
        the name, when this caller stands another robot, or when this caller
        re-attaches its robot as another model.

        A caller re-attaching the robot it already stands is re-registering
        the same live copy: the robot stays exactly as it is and the new goal
        hosts it, which is what an initializer that died and came back does.

        The lease runs from here: a robot that attaches and never pairs a
        limb gives its name back like any other."""
        if not name:
            raise ValueError("a robot stands under the name of the copy it runs as")
        with self._lock:
            held = self._robots.get(name)
            if held is not None:
                if held.caller != caller:
                    raise ValueError(
                        f"a robot of {held.caller} already stands as '{name}'"
                    )
                if held.model != model:
                    raise ValueError(
                        f"{caller} stands '{name}' as {held.model}; attach as that "
                        "model, or remove the copy first"
                    )
                if not held.standing():
                    raise ValueError(
                        f"{caller} is already joining '{name}'; it stands in a moment"
                    )
                return True
            for other in self._robots.values():
                if other.caller == caller:
                    raise ValueError(
                        f"{caller} already stands a robot in this scene, as '{other.name}'"
                    )
            self._robots[name] = Robot(
                name=name, model=model, caller=caller, last_paired_s=now_s
            )
            return False

    def stand(self, name: str, limbs: Limbs, now_s: float) -> None:
        """Hands the reserved robot the limbs the engine gave it, and gives it
        its lease back: standing it takes as long as the scene's recompile
        does."""
        robot = self.of_name(name)
        if robot is None:
            return
        with robot.lock:
            robot.limbs = limbs
            robot.last_paired_s = now_s

    def release(self, name: str) -> Robot | None:
        """Gives a name back. Returns the robot that stood under it."""
        with self._lock:
            return self._robots.pop(name, None)

    def of_name(self, name: str) -> Robot | None:
        with self._lock:
            return self._robots.get(name)

    def of_caller(self, caller: Caller) -> Robot | None:
        with self._lock:
            return next(
                (robot for robot in self._robots.values() if robot.caller == caller),
                None,
            )

    def robots(self) -> list[Robot]:
        with self._lock:
            return list(self._robots.values())

    def standing(self) -> dict[str, Robot]:
        """The robots the engine has stood, by name, for the thread that runs
        the physics."""
        with self._lock:
            return {
                name: robot for name, robot in self._robots.items() if robot.standing()
            }

    def sole_name(self) -> str | None:
        """The name of the only robot in the scene, when there is exactly
        one. A limb pair carries no copy for a robot launched outside one,
        and outside a copy there is one robot for it to belong to."""
        with self._lock:
            if len(self._robots) != 1:
                return None
            return next(iter(self._robots))

    def note_paired(self, names: set[str], now_s: float) -> None:
        """Records that each named robot is holding a limb pair now, which is
        what keeps it in the scene."""
        for name in names:
            robot = self.of_name(name)
            if robot is None:
                continue
            with robot.lock:
                robot.last_paired_s = now_s

    def renew(self, now_s: float) -> None:
        """Gives every robot in the scene its lease back. The scene takes no
        commands while it is being changed, so the robots already in it are
        held to their leases from the moment it takes them again."""
        for robot in self.robots():
            with robot.lock:
                robot.last_paired_s = now_s
