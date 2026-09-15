"""The seats robots hold: who may take one, what a command may carry, and
when a robot's lease runs out."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robots" / "openarm"))

from seats import Caller, Limbs, Registry  # noqa: E402  pylint: disable=C0413

LIMBS = Limbs(arm_names=("left", "right"), arm_joints=(7, 7), gripper_names=("left", "right"))


class Arm:
    """One arm of a command, as the contract's array carries it."""

    def __init__(self, positions=(), velocities=()) -> None:
        self.positions = list(positions)
        self.velocities = list(velocities)


class Gripper:
    def __init__(self, commanded=True, opening=0.0, max_effort=0.0) -> None:
        self.commanded = commanded
        self.opening = opening
        self.max_effort = max_effort


def caller(instance: str = "alpha", core_node: str = "sim16") -> Caller:
    return Caller(core_node=core_node, instance_id=instance)


def standing(instance: str = "alpha") -> tuple[Registry, Caller]:
    """A registry holding one robot that is in the scene."""
    seats = Registry()
    held = caller(instance)
    seats.reserve(held, "openarm_v2", 0.0)
    seats.stand(held, LIMBS, now_s=0.0)
    return seats, held


def arms(*positions):
    return [Arm(values) for values in positions]


def grippers(*openings):
    return [Gripper(opening=opening) if opening is not None else Gripper(commanded=False)
            for opening in openings]


class TestTakingASeat:
    def test_a_caller_holds_one_seat(self) -> None:
        seats, held = standing()
        with pytest.raises(ValueError, match="already holds a seat"):
            seats.reserve(held, "openarm_v2", 0.0)

    def test_two_callers_cannot_stand_under_one_name(self) -> None:
        seats, _ = standing()
        with pytest.raises(ValueError, match="already stands as 'alpha'"):
            seats.reserve(caller("alpha", core_node="other"), "openarm_v1", 0.0)

    def test_the_same_name_on_another_node_is_another_robot(self) -> None:
        seats, _ = standing()
        seats.reserve(caller("bravo", core_node="other"), "openarm_v1", 0.0)
        assert len(seats.seats()) == 2

    def test_a_seat_given_back_frees_its_name(self) -> None:
        seats, held = standing()
        assert seats.release(held) is not None
        seats.reserve(caller("alpha", core_node="other"), "openarm_v1", 0.0)

    def test_a_reserved_seat_is_not_standing_yet(self) -> None:
        seats = Registry()
        seats.reserve(caller(), "openarm_v2", 0.0)
        assert seats.standing() == {}

    def test_a_standing_seat_is_listed_under_its_name(self) -> None:
        seats, _ = standing()
        assert list(seats.standing()) == ["alpha"]


class TestCommands:
    def test_a_robot_still_joining_takes_no_command(self) -> None:
        seats = Registry()
        held = caller()
        seats.reserve(held, "openarm_v2", 0.0)
        taken, message = seats.command(held, arms([0.0] * 7, [0.0] * 7), grippers(0.0, 0.0), 1.0)
        assert not taken
        assert "still joining" in message

    def test_a_caller_with_no_seat_is_told_so(self) -> None:
        seats = Registry()
        taken, message = seats.command(caller(), [], [], 1.0)
        assert not taken
        assert "holds no seat" in message

    def test_a_command_of_every_limb_is_taken(self) -> None:
        seats, held = standing()
        assert seats.command(
            held, arms([0.1] * 7, [0.2] * 7), grippers(0.5, 0.5), 1.0
        ) == (True, "")
        setpoints = seats.seat_of(held).take()
        assert setpoints.arms[0].positions == (0.1,) * 7
        assert setpoints.grippers[1].opening == 0.5

    def test_an_arm_of_the_wrong_length_is_refused_by_name(self) -> None:
        seats, held = standing()
        taken, message = seats.command(held, arms([0.0] * 7, [0.0] * 6), grippers(0.0, 0.0), 1.0)
        assert not taken
        assert "arm 'right' has 7 joints, the command carries 6 positions" == message

    def test_too_few_arms_is_refused(self) -> None:
        seats, held = standing()
        taken, message = seats.command(held, arms([0.0] * 7), grippers(0.0, 0.0), 1.0)
        assert not taken
        assert "this robot has 2 arms, the command carries 1" == message

    def test_too_few_grippers_is_refused(self) -> None:
        seats, held = standing()
        taken, message = seats.command(held, arms([0.0] * 7, [0.0] * 7), grippers(0.0), 1.0)
        assert not taken
        assert "this robot has 2 grippers, the command carries 1" == message

    def test_velocities_of_the_wrong_length_are_refused(self) -> None:
        seats, held = standing()
        taken, message = seats.command(
            held, [Arm([0.0] * 7, [0.0] * 3), Arm([0.0] * 7)], grippers(0.0, 0.0), 1.0
        )
        assert not taken
        assert "3 velocities" in message

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_position_that_is_not_a_number_is_refused(self, bad: float) -> None:
        seats, held = standing()
        taken, message = seats.command(
            held, [Arm([bad] + [0.0] * 6), Arm([0.0] * 7)], grippers(0.0, 0.0), 1.0
        )
        assert not taken
        assert "arm 'left' was commanded a value that is not a number" == message

    def test_an_opening_that_is_not_a_number_is_refused(self) -> None:
        seats, held = standing()
        taken, message = seats.command(
            held,
            arms([0.0] * 7, [0.0] * 7),
            [Gripper(opening=float("nan")), Gripper(opening=0.0)],
            1.0,
        )
        assert not taken
        assert "gripper 'left' was commanded a value that is not a number" == message

    def test_a_negative_force_limit_is_refused(self) -> None:
        seats, held = standing()
        taken, message = seats.command(
            held,
            arms([0.0] * 7, [0.0] * 7),
            [Gripper(opening=0.5, max_effort=-1.0), Gripper(opening=0.0)],
            1.0,
        )
        assert not taken
        assert "gripper 'left' was commanded a negative force limit" == message

    def test_a_refused_command_changes_no_limb(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.1] * 7, [0.2] * 7), grippers(0.5, 0.5), 1.0)
        seats.command(held, arms([0.9] * 7, [0.0] * 3), grippers(0.9, 0.9), 2.0)
        setpoints = seats.seat_of(held).take()
        assert setpoints.arms[0].positions == (0.1,) * 7
        assert setpoints.grippers[0].opening == 0.5

    def test_an_arm_with_no_positions_keeps_its_setpoint(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.1] * 7, [0.2] * 7), grippers(0.5, 0.5), 1.0)
        seats.command(held, arms([], [0.3] * 7), grippers(0.5, 0.5), 2.0)
        setpoints = seats.seat_of(held).take()
        assert setpoints.arms[0].positions == (0.1,) * 7
        assert setpoints.arms[1].positions == (0.3,) * 7

    def test_an_uncommanded_gripper_keeps_its_setpoint(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.1] * 7, [0.2] * 7), grippers(0.5, 0.5), 1.0)
        seats.command(held, arms([0.1] * 7, [0.2] * 7), grippers(None, 0.8), 2.0)
        setpoints = seats.seat_of(held).take()
        assert setpoints.grippers[0].opening == 0.5
        assert setpoints.grippers[1].opening == 0.8


class TestLeases:
    def test_a_robot_that_commands_keeps_its_seat(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.0] * 7, [0.0] * 7), grippers(0.0, 0.0), now_s=10.0)
        assert seats.lapsed(now_s=11.0, lease_s=2.0) == []

    def test_a_robot_silent_past_the_lease_has_lapsed(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.0] * 7, [0.0] * 7), grippers(0.0, 0.0), now_s=10.0)
        lapsed = seats.lapsed(now_s=12.5, lease_s=2.0)
        assert [seat.caller for seat in lapsed] == [held]

    def test_the_lease_runs_from_the_moment_the_robot_stood(self) -> None:
        seats, _ = standing()
        assert seats.lapsed(now_s=1.0, lease_s=2.0) == []
        assert len(seats.lapsed(now_s=3.0, lease_s=2.0)) == 1

    def test_a_scene_that_changed_gives_every_lease_back(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.0] * 7, [0.0] * 7), grippers(0.0, 0.0), now_s=10.0)
        seats.renew(now_s=20.0)
        assert seats.lapsed(now_s=21.0, lease_s=2.0) == []

    def test_a_robot_that_attaches_and_goes_quiet_gives_its_name_back(self) -> None:
        """The lease runs from the reservation, so a robot that never stands
        cannot hold its name for as long as the engine runs."""
        seats = Registry()
        seats.reserve(caller(), "openarm_v2", 0.0)

        assert seats.lapsed(now_s=1.0, lease_s=2.0) == []
        assert len(seats.lapsed(now_s=100.0, lease_s=2.0)) == 1

    def test_a_robot_still_joining_keeps_its_seat_while_it_commands(self) -> None:
        """Standing a robot takes as long as the scene's rebuild; one that
        heartbeats through that keeps its seat."""
        seats = Registry()
        held = caller()
        seats.reserve(held, "openarm_v2", 0.0)

        for now in (1.0, 2.0, 3.0, 4.0):
            answered, message = seats.command(
                held, arms([0.0] * 7, [0.0] * 7), grippers(0.0, 0.0), now_s=now
            )
            assert not answered and "still joining" in message

        assert seats.lapsed(now_s=5.0, lease_s=2.0) == []
        assert len(seats.lapsed(now_s=7.0, lease_s=2.0)) == 1

    def test_a_scene_that_changed_gives_a_joining_robot_its_lease_back(self) -> None:
        seats = Registry()
        seats.reserve(caller(), "openarm_v2", 0.0)

        seats.renew(now_s=20.0)

        assert seats.lapsed(now_s=21.0, lease_s=2.0) == []

    def test_a_refused_command_does_not_renew_the_lease(self) -> None:
        seats, held = standing()
        seats.command(held, arms([0.0] * 7, [0.0] * 7), grippers(0.0, 0.0), now_s=10.0)
        seats.command(held, arms([0.0] * 3), grippers(0.0, 0.0), now_s=12.0)
        assert len(seats.lapsed(now_s=12.5, lease_s=2.0)) == 1
