#!/usr/bin/env python3
"""The seats robots hold in the scene.

A seat is reserved when an attach goal is admitted and holds its limbs once
the engine has stood the robot, so a command that arrives while the scene is
still recompiling is refused as still joining. A seated robot's setpoints
live in its seat until the physics thread picks them up, and every command
renews the seat's lease: a robot silent past the lease leaves the scene.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import NamedTuple


class Answer(NamedTuple):
    """How a command was answered: whether the robot took it, whether the
    engine is still standing that robot, and what to say."""

    taken: bool
    joining: bool
    message: str


@dataclass(frozen=True)
class Caller:
    """Who holds a seat: the instance peppy delivered the message from, on
    the core node it runs on. That pair is unique across the mesh, so two
    stacks whose instances happen to share a name never reach each other's
    robot."""

    core_node: str
    instance_id: str

    @property
    def instance(self) -> str:
        """The name the robot stands under, which is the caller's instance
        id: a stack mints one per robot, and the scene reads better for it."""
        return self.instance_id

    def __str__(self) -> str:
        return f"{self.instance_id}@{self.core_node}"


@dataclass(frozen=True)
class Limbs:
    """A model's limbs in the order every arms / grippers array of the
    contract follows, and the joints each arm carries."""

    arm_names: tuple[str, ...]
    arm_joints: tuple[int, ...]
    gripper_names: tuple[str, ...]


@dataclass(frozen=True)
class ArmSetpoint:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]


@dataclass(frozen=True)
class GripperSetpoint:
    opening: float
    max_effort: float


@dataclass
class Setpoints:
    """The latest setpoint of each of a robot's limbs, indexed as its limbs
    are. A limb no command has named yet is None, and the engine leaves it
    where its model stands."""

    arms: list[ArmSetpoint | None]
    grippers: list[GripperSetpoint | None]


@dataclass
class Seat:
    """One robot's seat. `limbs` arrives when the engine has stood the robot,
    and until then the seat takes no commands."""

    caller: Caller
    model: str
    limbs: Limbs | None = None
    setpoints: Setpoints | None = None
    # When this seat last heard from its robot, set as the seat is reserved so
    # a robot that never stands still gives its name back.
    last_command_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def standing(self) -> bool:
        return self.limbs is not None

    def take(self) -> Setpoints | None:
        """The setpoints as they stand, for the physics thread."""
        with self.lock:
            if self.setpoints is None:
                return None
            return Setpoints(list(self.setpoints.arms), list(self.setpoints.grippers))


class Registry:
    """The robots that have attached, by the caller holding each seat."""

    def __init__(self) -> None:
        self._seats: dict[Caller, Seat] = {}
        self._lock = threading.Lock()

    def reserve(self, caller: Caller, model: str, now_s: float) -> Seat:
        """Reserves this caller's seat. Raises when the caller already holds
        one, or when another caller's robot stands under the same name.

        The lease runs from here, not from standing: a robot that attaches and
        then goes quiet gives its name back like any other, rather than holding
        it for as long as the engine runs."""
        with self._lock:
            if caller in self._seats:
                raise ValueError(f"{caller} already holds a seat in this scene")
            for held in self._seats:
                if held.instance == caller.instance:
                    raise ValueError(
                        f"a robot of {held} already stands as '{caller.instance}'"
                    )
            seat = Seat(caller=caller, model=model, last_command_s=now_s)
            self._seats[caller] = seat
            return seat

    def stand(self, caller: Caller, limbs: Limbs, now_s: float) -> None:
        """Hands the reserved seat the limbs the engine gave the robot, and
        gives it its lease back: standing it takes as long as the scene's
        recompile does."""
        seat = self.seat_of(caller)
        if seat is None:
            return
        with seat.lock:
            seat.limbs = limbs
            seat.setpoints = Setpoints(
                arms=[None] * len(limbs.arm_names),
                grippers=[None] * len(limbs.gripper_names),
            )
            seat.last_command_s = now_s

    def release(self, caller: Caller) -> Seat | None:
        """Gives a seat back. Returns it when the caller held one."""
        with self._lock:
            return self._seats.pop(caller, None)

    def seat_of(self, caller: Caller) -> Seat | None:
        with self._lock:
            return self._seats.get(caller)

    def seats(self) -> list[Seat]:
        with self._lock:
            return list(self._seats.values())

    def standing(self) -> dict[str, Seat]:
        """The seats whose robots are in the scene, by the name each stands
        under, for the thread that runs the physics."""
        with self._lock:
            return {
                seat.caller.instance: seat for seat in self._seats.values() if seat.standing()
            }

    def command(self, caller: Caller, arms, grippers, now_s: float) -> Answer:
        """Writes one command into a robot's limbs and renews its lease. A
        command carrying a limb the robot cannot take changes nothing."""
        seat = self.seat_of(caller)
        if seat is None:
            return Answer(False, False, f"{caller} holds no seat in this scene")
        with seat.lock:
            if seat.limbs is None:
                # The heartbeat counts while the robot is still joining:
                # standing it takes as long as the scene's rebuild, and a robot
                # commanding faithfully through that must not lose its seat for
                # it. What the command carries cannot be judged until the robot
                # has limbs to judge it against.
                seat.last_command_s = now_s
                return Answer(
                    False, True, f"'{caller.instance}' is still joining the scene"
                )
            try:
                arm_setpoints = _arm_setpoints(seat.limbs, arms)
                gripper_setpoints = _gripper_setpoints(seat.limbs, grippers)
            except ValueError as error:
                return Answer(False, False, str(error))
            for index, setpoint in enumerate(arm_setpoints):
                if setpoint is not None:
                    seat.setpoints.arms[index] = setpoint
            for index, setpoint in enumerate(gripper_setpoints):
                if setpoint is not None:
                    seat.setpoints.grippers[index] = setpoint
            seat.last_command_s = now_s
            return Answer(True, False, "")

    def renew(self, now_s: float) -> None:
        """Gives every robot in the scene its lease back. The scene takes no
        commands while it is being changed, so the robots already in it are
        held to their leases from the moment it takes them again."""
        for seat in self.seats():
            with seat.lock:
                seat.last_command_s = now_s

    def lapsed(self, now_s: float, lease_s: float) -> list[Seat]:
        """The seats whose robots have not commanded within the lease, standing
        or still joining."""
        return [
            seat for seat in self.seats() if now_s - seat.last_command_s > lease_s
        ]


def _arm_setpoints(limbs: Limbs, arms) -> list[ArmSetpoint | None]:
    """One setpoint per arm of the model, or None for an arm this command
    leaves alone. Raises naming the arm the command cannot drive."""
    if len(arms) != len(limbs.arm_names):
        raise ValueError(
            f"this robot has {len(limbs.arm_names)} arms, the command carries {len(arms)}"
        )
    setpoints: list[ArmSetpoint | None] = []
    for name, joints, arm in zip(limbs.arm_names, limbs.arm_joints, arms):
        positions = tuple(float(value) for value in arm.positions)
        velocities = tuple(float(value) for value in arm.velocities)
        if not positions:
            setpoints.append(None)
            continue
        if len(positions) != joints:
            raise ValueError(
                f"arm '{name}' has {joints} joints, the command carries {len(positions)} positions"
            )
        if velocities and len(velocities) != joints:
            raise ValueError(
                f"arm '{name}' has {joints} joints, the command carries "
                f"{len(velocities)} velocities"
            )
        if not all(math.isfinite(value) for value in (*positions, *velocities)):
            raise ValueError(f"arm '{name}' was commanded a value that is not a number")
        setpoints.append(ArmSetpoint(positions=positions, velocities=velocities))
    return setpoints


def _gripper_setpoints(limbs: Limbs, grippers) -> list[GripperSetpoint | None]:
    """One setpoint per gripper of the model, or None for a gripper this
    command leaves alone. Raises naming the gripper it cannot drive."""
    if len(grippers) != len(limbs.gripper_names):
        raise ValueError(
            f"this robot has {len(limbs.gripper_names)} grippers, "
            f"the command carries {len(grippers)}"
        )
    setpoints: list[GripperSetpoint | None] = []
    for name, gripper in zip(limbs.gripper_names, grippers):
        if not gripper.commanded:
            setpoints.append(None)
            continue
        opening = float(gripper.opening)
        max_effort = float(gripper.max_effort)
        if not math.isfinite(opening) or not math.isfinite(max_effort):
            raise ValueError(f"gripper '{name}' was commanded a value that is not a number")
        if max_effort < 0.0:
            raise ValueError(f"gripper '{name}' was commanded a negative force limit")
        setpoints.append(GripperSetpoint(opening=opening, max_effort=max_effort))
    return setpoints
