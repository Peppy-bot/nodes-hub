"""FK/IK over placo's kinematics solver, radians at this boundary.

placo speaks 4x4 matrices; everything above this module speaks joint_link
radians and wire poses. IK is verified by FK before it is trusted: placo
returns its best effort even far from the target, and an unreached pose must
fail the caller instead of moving the arm somewhere else.

What that verification costs, measured on poses reachable by construction (a
random in-limits joint vector, its FK taken as the target): 14% of targets are
refused when seeded from the ready posture and 39% when seeded from an
arbitrary one, concentrated near the base (25% within 0.23 m, 7% beyond
0.45 m). The cause is structural rather than a tuning miss. A solve is a
single linearised QP step, so iterating it is a local descent from the seed
and reaches only what the seed's branch reaches; a refusal means this branch
did not arrive, not that the pose is unreachable. Callers that need the wider
workspace re-seed and ask again.

ApproachSolver is the other solver here, for the workspace answer: where the
grasp point can go, with the gripper's approach axis along a direction. It
has its own model and QP, and its own seeds, which it takes from a table of
postures across the whole joint space. Its ReachSphere holds every point the
grasp point can go to, by construction from the joint chain of the URDF.
"""

from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import placo
from so101.transforms import (
    matrix_from_pose,
    pose_from_matrix,
    relative_rotation_rad,
    slerp,
)

from so101_backbone.limits import JointLimits
from so101_backbone.limits import from_description as limits_from_description
from so101_backbone.model import ARM, DESCRIPTION, END_EFFECTOR_FRAME

# A solve runs one QP step, sized for streaming small deltas; a
# point-to-point solve iterates it until FK verifies the position.
IK_MAX_ITERATIONS = 100
# The streamed path is best effort: a few steps bound the per-tick cost, and
# an out-of-reach pose tracks the workspace boundary instead of freezing.
IK_STREAM_ITERATIONS = 3
# The samples a straight line's walk solves: on the longest line of the arm's
# reach, 0.87 m, a step under MoveIt's 1 cm.
LINE_SAMPLES = 100
# The most a joint may turn from one configuration of a walk to the next: past
# it the solution has jumped to another branch. 218 walks of drawn position goals
# step at most 0.32 rad.
MAX_LINE_STEP_RAD = 0.35
# How far the grasp point may stand off a walk's line: MoveIt's
# CartesianPrecision::translational default.
LINE_POSITION_BAR_M = 0.001
# How far the end of a walk may stand off an orientation it names: MoveIt's
# CartesianPrecision::rotational default.
LINE_ORIENTATION_BAR_RAD = 0.01
# The largest angle between two rotations: an orientation bar this wide takes
# any orientation.
ANY_ORIENTATION_RAD = math.pi

# Position and orientation are both soft objectives of one QP, minimising
# `position_weight * |dp|^2 + orientation_weight * |dtheta|^2`. The terms are
# metres against radians, so the weight ratio is not the trade: at 0.01 a 20
# degree miss outweighs a 7 mm miss by more than twenty to one, and the
# solver spends position buying orientation. Five joints
# underactuate three rotational degrees of freedom, so on this arm that
# purchase is frequently impossible and the position is spent for nothing.
#
# Weighted this low the orientation objective still resolves the wrist onto
# the pose it was given, reaching a reachable orientation within a third of a
# degree, while an unreachable one stops taking the position with it. Zero is
# not the answer; it abandons reachable orientations entirely.
ORIENTATION_WEIGHT = 1e-4
# The position objective's weight, the unit the orientation weight is
# measured against.
POSITION_WEIGHT = 1.0

# The approach solver (ApproachSolver). Its numbers are measured on the
# SO-101 model; the class docstring gives the measurements.
#
# The seed table holds the postures of a grid with this many points per joint
# across the limits of the shoulder lift, the elbow and the wrist flex, at
# two wrist rolls (see _seed_rolls). Fewer points leave more poses
# unreached: of 1500 random in-limit joint vectors (RNG seeds 21, 22 and
# 23) along their own approach axis, at the 21 pairs of bars of the class
# docstring, 15 points leave 35 unreached, 13 points 47, and 11 points 54.
APPROACH_SEED_GRID_POINTS = 15
# How many of the nearest seeds an approach, in each rank of its seeds, and
# a least distance start from in turn. Fewer leave more targets unreached:
# two seeds for an approach leave 9 of 450 random in-limit joint vectors
# (RNG seeds 101, 202 and 303) unreached along their own approach axis at
# bars of 0.002 m and 0.01 rad (three leave 2), and two for a least distance
# leave 12 of 11340 reachable positions up to 6.8 mm short (four leave 1,
# 0.65 mm short).
APPROACH_SEEDS = 3
APPROACH_DISTANCE_SEEDS = 4
# The ranks of the seeds of an approach, in the order the solver tries
# them: in each, how many metres between a seed's grasp point and the target
# weigh as much as a radian between its approach axis and the direction.
# The solver ranks the seeds a second time only when every descent from the
# seeds of the first rank fails. No rank depends on the caller's bars, so
# all bars try the same seeds. The first rank is at the ratio of the bars of
# a workspace answer, 0.01 m over 0.05 rad. The second puts the position
# first. Its ratio is chosen on 3450 random in-limit joint vectors (RNG
# seeds 101, 202, 303, 21, 22, 23, 31, 32 and 33), along their own approach
# axis at 21 pairs of bars, and in a random direction at 0.01 m and pi rad.
# Unreached, in that order: the first rank alone, 231 and 296; a second
# rank at 0.04, 161 and 70; at 0.02, 129 and 10; at 0.01, 98 and 0; at
# 0.005, 91 and 0; at 0, 91 and 0. Of the ratios that leave none at pi rad,
# 0.01 takes the fewest QP steps on a workspace grid. Most approaches of a
# workspace grid fail, because most grid points are out of reach, so each
# of these also descends from the seeds of the second rank: a full grid
# takes 1.6 to 2.1 times the QP steps of the first rank alone.
APPROACH_SEED_RANKS_METRES_PER_RADIAN = (0.01 / 0.05, 0.01)
# Two seeds of one posture at pans closer than this, in radians, start one
# descent twice: the solver takes the nearer and skips the other. With
# 0.02 rad, 4 of the 450 joint vectors above stay unreached along their own
# approach axis at bars of 0.002 m and 0.01 rad; with 0.1 rad, 2.
APPROACH_SEED_PAN_SPACING_RAD = 0.1
# The trust region of one QP step: the most it turns an arm joint, and the
# wrist roll. An unbounded step overshoots far from the target and
# oscillates about the edge of the reach. The roll swings the grasp point on
# a circle of a few millimetres, so it needs larger turns to move it.
APPROACH_STEP_RAD = 0.05
APPROACH_ROLL_STEP_RAD = 0.2
# A descent stops when its error has not shrunk by APPROACH_STALL_GAIN_M in
# APPROACH_STALL_STEPS steps, or after its step budget: APPROACH_ITERATIONS
# for each trade of an approach, APPROACH_DISTANCE_ITERATIONS for a least
# distance.
APPROACH_STALL_STEPS = 3
APPROACH_STALL_GAIN_M = 1e-5
APPROACH_ITERATIONS = 30
APPROACH_DISTANCE_ITERATIONS = 60
# How many trades between the position and the approach objectives an
# approach tries from one seed, and how far the second trade moves the
# weight before a bisection takes over (see _Trade).
APPROACH_TRADES = 9
APPROACH_TRADE_FACTOR = 4.0
# A least distance this short is a reach: the descent stops there.
APPROACH_DISTANCE_RESOLUTION_M = 1e-4
# The largest coordinate magnitude, in metres, of a target position. The
# solver's arithmetic holds at coordinates of 1e10 m on this model; at
# 1e11 m on each axis its QP fails.
APPROACH_MAX_COORDINATE_M = 1.0e6
# The range of the bars of an approach. Every grasp point is within the
# longest position bar of every target the solver takes, so a longer bar
# says nothing more. An approach axis is never more than pi from a direction.
# The narrowest angle bar is about the smallest angle the solver measures
# between two axes that is not zero (1.5e-8 rad, the arccosine of the
# largest number below 1). Inside these ranges the trade between the two
# objectives (see _Trade) stays below 1e34, so its arithmetic stays finite.
APPROACH_MAX_POSITION_BAR_M = 2.0 * APPROACH_MAX_COORDINATE_M
APPROACH_MIN_ANGLE_BAR_RAD = 1.0e-8
# The weight of the heavier of the two objectives in the QP of a descent;
# the trade sets the other (see ApproachSolver._weigh_approach). The QP
# meets its hard constraints, the joint limits and the trust region, only
# to an error that grows with its weights, not with their ratio. Measured
# on the 1293 descents from the seeds of both ranks of 200 random targets,
# each in a random direction (RNG seed 20261004): with the position
# objective at 1 and the approach objective at 1e5, a step breaks them by
# up to 3.8e-3 rad; at 1e7, by up to 0.37 rad, and then 72 of the descents
# get to joints where no step meets them, and the QP fails. With the
# heavier objective at 1, at trades of 0 (the QP of a least distance) and
# from 1e2 to 1e33, no step breaks them by more than 1e-7 rad, and no QP
# fails. Over 1500 random targets and directions (RNG seed 1234) at twelve
# pairs of bars, among them the four ends of their ranges and angle bars
# from 1e-8 to 1e-3 rad, no step breaks them by more than 1e-7 rad, and no
# QP fails. In the descents of 600 least distances to random targets (RNG
# seed 4242), no step breaks them by more than 1.1e-12 rad.
APPROACH_QP_MAX_WEIGHT = 1.0


def _bar(value: float) -> float:
    """A caller's acceptance bar: a positive finite distance or angle."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"tolerance must be finite and positive, got {value}")
    return value


def _arm_model(urdf_path: str):
    """The arm's placo model for the solvers, which move it by its kinematics
    alone: its collision meshes load with no self-collision pairs."""
    return placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)


def _set_joints(robot, positions_rad) -> None:
    """Puts the model at these joint radians, in the arm's order, and
    updates its frames."""
    for name, value in zip(ARM.joints, positions_rad, strict=True):
        robot.set_joint(name, float(value))
    robot.update_kinematics()


def _joints(robot) -> tuple[float, ...]:
    """The model's joint radians, in the arm's order."""
    return tuple(float(robot.get_joint(name)) for name in ARM.joints)


class Kinematics:
    """The arm's model loaded once, with the QP that solves the end effector's
    pose over it."""

    def __init__(self, urdf_path: str):
        self._robot = _arm_model(urdf_path)
        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.mask_fbase(True)
        self._tip = self._solver.add_frame_task(END_EFFECTOR_FRAME, np.eye(4))

    def jacobian(self, positions_rad: tuple[float, ...]):
        """The end-effector Jacobian at these joint positions: 6 rows by one
        column per arm joint in the arm's order, linear velocity in rows 0..2 and
        angular in rows 3..5, both in world-aligned axes.

        `J @ dq` is the twist a joint step produces, and it is linear in that
        step, which is what a speed cap needs: scaling a step by s scales its
        end-effector speed by exactly s. Measuring the same thing by moving
        the arm and differencing forward kinematics does not have that
        property, because forward kinematics is not linear.

        Local-world-aligned, not world: placo's world Jacobian gives the
        twist of the body frame referred to the world origin, whose linear
        part is not the tool point's velocity. Against a finite difference the
        two disagree by tens of percent at every step size."""
        # The QP carries state between solves, and moving the model out from
        # under it diverges a streaming solve by radians. This is a query, so
        # it puts the configuration back before returning.
        restore = _joints(self._robot)
        try:
            _set_joints(self._robot, positions_rad)
            columns = [self._robot.get_joint_v_offset(name) for name in ARM.joints]
            return np.asarray(
                self._robot.frame_jacobian(END_EFFECTOR_FRAME, "local_world_aligned")
            )[:, columns]
        finally:
            _set_joints(self._robot, restore)

    def forward_kinematics(self, positions_rad: tuple[float, ...]):
        """(position m, quaternion xyzw) of the end effector."""
        _set_joints(self._robot, positions_rad)
        return pose_from_matrix(self._robot.get_T_world_frame(END_EFFECTOR_FRAME))

    def inverse_kinematics(
        self,
        seed_rad: tuple[float, ...],
        position,
        orientation,
        *,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> tuple[float, ...] | None:
        """Joint radians reaching the pose, or None when no verified solution
        meets the caller's bars: the distance and the angle between the pose
        forward kinematics gives for the solution and the target. Five joints
        underactuate the three orientation degrees of freedom, so a tight
        orientation bar refuses almost every reachable pose; a caller that
        takes any orientation passes math.pi or more."""
        position_bar = _bar(position_tolerance_m)
        orientation_bar = _bar(orientation_tolerance_rad)
        target_matrix = matrix_from_pose(position, orientation)
        target_position = tuple(position)
        target_orientation = tuple(orientation)
        solution_rad = tuple(seed_rad)
        for _ in range(IK_MAX_ITERATIONS):
            solution_rad = self._step(solution_rad, target_matrix)
            reached, reached_orientation = self.forward_kinematics(solution_rad)
            if math.dist(reached, target_position) > position_bar:
                continue
            if relative_rotation_rad(reached_orientation, target_orientation) <= orientation_bar:
                return solution_rad
        return None

    def walk_line(
        self,
        seed_rad: tuple[float, ...],
        position,
        orientation,
        limits: JointLimits,
    ) -> tuple[tuple[float, ...], ...] | None:
        """The configurations of a straight line from the pose at `seed_rad` to
        `position`, from `seed_rad` on: one verified solve per sample, each seeded
        from the last, with the grasp point within LINE_POSITION_BAR_M of its
        sample. Five joints cannot hold a turn between two poses in general, so
        the orientation is free along the way, and the last sample lands within
        LINE_ORIENTATION_BAR_RAD of `orientation`. With `orientation` None the goal
        is the position alone, and the solves aim at the turn the line starts with.
        None when a sample has no solution inside `limits` or its solution jumps
        branch."""
        start_position, start_orientation = self.forward_kinematics(seed_rad)
        turn_to = start_orientation if orientation is None else orientation
        end_bar = ANY_ORIENTATION_RAD if orientation is None else LINE_ORIENTATION_BAR_RAD
        walk = [tuple(seed_rad)]
        for k in range(1, LINE_SAMPLES + 1):
            share = k / LINE_SAMPLES
            last = k == LINE_SAMPLES
            sample = tuple(a + (b - a) * share for a, b in zip(start_position, position, strict=True))
            solved = self.inverse_kinematics(
                walk[-1],
                tuple(position) if last else sample,
                slerp(start_orientation, turn_to, share),
                position_tolerance_m=LINE_POSITION_BAR_M,
                orientation_tolerance_rad=end_bar if last else ANY_ORIENTATION_RAD,
            )
            if solved is None or not limits.contains(solved):
                return None
            if any(abs(a - b) > MAX_LINE_STEP_RAD for a, b in zip(solved, walk[-1], strict=True)):
                return None
            walk.append(solved)
        return tuple(walk)

    def inverse_kinematics_streaming(
        self, seed_rad: tuple[float, ...], position, orientation
    ) -> tuple[float, ...] | None:
        """Best-effort step toward the pose for the streamed path: a bounded
        few QP iterations, unverified, so an out-of-reach target tracks the
        workspace boundary instead of freezing the arm. None only for a
        solution the solver corrupted (non-finite)."""
        target_matrix = matrix_from_pose(position, orientation)
        solution_rad = tuple(seed_rad)
        for _ in range(IK_STREAM_ITERATIONS):
            solution_rad = self._step(solution_rad, target_matrix)
        if not all(math.isfinite(v) for v in solution_rad):
            return None
        return solution_rad

    def _step(self, seed_rad, target_matrix) -> tuple[float, ...]:
        """One QP step from the seed toward the pose, both objectives soft
        at the weights above: the joint radians it lands on. The seed is
        written to the model without updating its kinematics: the QP
        linearises at the configuration the last query left, which within
        a solve is the seed itself, and on the first step of a solve is the
        pose queried last."""
        for name, value in zip(ARM.joints, seed_rad, strict=True):
            self._robot.set_joint(name, float(value))
        self._tip.T_world_frame = target_matrix
        self._tip.configure(END_EFFECTOR_FRAME, "soft", POSITION_WEIGHT, ORIENTATION_WEIGHT)
        self._solver.solve(True)
        self._robot.update_kinematics()
        return _joints(self._robot)


# The joint the solver turns to face a seed posture toward a target, and the
# joint whose turns move the grasp point least.
_PAN_JOINT = ARM.joints[0]
_ROLL_JOINT = ARM.joints[-1]
# The joints the seed table spans with a grid, by index in the arm's order.
_GRID_JOINTS = (1, 2, 3)
# The URDF joint types that turn their child link about the joint's origin,
# or hold it there, and so never move the child link's origin.
_JOINT_TYPES_THAT_KEEP_THE_ORIGIN = ("revolute", "continuous", "fixed")


@dataclass(frozen=True)
class ReachSphere:
    """A sphere that holds every point the grasp point can go to: about
    `centre`, the origin of the pan joint in the model's base frame, with
    radius `radius_m`, the sum of the lengths of the origin offsets of the
    joints after the pan joint, up to the joint whose child link is
    END_EFFECTOR_FRAME, each offset given in the frame of its parent link.
    This holds for every joint angle, by construction:

    1. A URDF joint puts the frame of its child link at the joint's origin
       in the frame of its parent link, turned about the joint's axis by
       the joint's angle. A revolute or continuous joint turns the axes of
       that frame about its origin; a fixed joint does not turn them. No
       angle moves the origin of the child link's frame.
    2. The origin of the next joint is at its offset in that frame, so the
       angles turn the offset and do not change its length.
    3. The grasp point is the origin of END_EFFECTOR_FRAME. It is thus the
       pan joint's origin plus one turned offset for each joint after the
       pan joint, the last the fixed joint of END_EFFECTOR_FRAME.
    4. By the triangle inequality, the length of a sum is at most the sum
       of the lengths. So the grasp point is at most `radius_m` from
       `centre`.

    On the SO-101 model the radius is 0.478 m. The farthest grasp point that
    a search of the joint space finds, a grid of 61 points across the limits
    of each joint after the pan refined about its farthest point, is 5.4 mm
    nearer to the centre.
    """

    centre: tuple[float, float, float]
    radius_m: float

    @classmethod
    def of(cls, urdf_path: str, robot) -> ReachSphere:
        """The sphere of the model loaded from `urdf_path` into `robot`: the
        radius from the origin offsets of the joints after the pan joint, up
        to the joint whose child link is END_EFFECTOR_FRAME, in the chain of
        joints of the URDF, and the centre where the model puts the pan
        joint's origin. Raises ValueError when the URDF has no chain of
        joints from the pan joint to END_EFFECTOR_FRAME, or when a joint
        after the pan joint in it can move its child link's origin."""
        joints_by_child = {
            joint.find("child").get("link"): joint
            for joint in ET.parse(urdf_path).getroot().findall("joint")
            if joint.find("child") is not None
        }
        radius_m = 0.0
        link = END_EFFECTOR_FRAME
        while (joint := joints_by_child.get(link)) is not None:
            if joint.get("name") == _PAN_JOINT:
                centre = robot.get_T_world_frame(_PAN_JOINT)[:3, 3]
                return cls(centre=tuple(float(value) for value in centre), radius_m=radius_m)
            if joint.get("type") not in _JOINT_TYPES_THAT_KEEP_THE_ORIGIN:
                raise ValueError(
                    f"URDF joint {joint.get('name')} is {joint.get('type')}: it can move "
                    f"the origin of {link}"
                )
            radius_m += math.hypot(*_origin_offset(joint))
            link = joint.find("parent").get("link")
        raise ValueError(f"URDF has no chain of joints from {_PAN_JOINT} to {END_EFFECTOR_FRAME}")

    def rules_out(self, position, position_tolerance_m: float) -> bool:
        """Whether `position` is farther from `centre` than `radius_m` plus
        `position_tolerance_m`: then no pose of the arm puts the grasp point
        within `position_tolerance_m` of it, in any orientation. False does
        not mean that a pose does.

        Raises ValueError for a position that is not three finite
        coordinates each within APPROACH_MAX_COORDINATE_M, and for a
        tolerance that is not above zero and at most
        APPROACH_MAX_POSITION_BAR_M."""
        return self.short_of(position) > _position_bar(position_tolerance_m)

    def short_of(self, position) -> float:
        """How far `position` stands outside the sphere, 0 inside it: no pose
        of the arm puts the grasp point nearer it, in any orientation.

        Raises ValueError for a position that is not three finite
        coordinates each within APPROACH_MAX_COORDINATE_M."""
        return max(0.0, math.dist(_target(position), self.centre) - self.radius_m)


def _origin_offset(joint) -> tuple[float, float, float]:
    """The offset of a URDF joint's origin in the frame of its parent link:
    its `origin` element's xyz, zero when the joint has no origin element or
    the element no xyz."""
    origin = joint.find("origin")
    xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
    x, y, z = (float(value) for value in xyz.split())
    return x, y, z


class ApproachSolver:
    """Where the grasp point can go, for a workspace answer: the least
    distance it comes to a position in any orientation (`least_distance`),
    and joints that bring it to a position with the gripper's approach axis
    along a direction (`approach`).

    The grasp point is the origin of END_EFFECTOR_FRAME and the approach axis
    is that frame's +Z, out of the gripper. Five joints leave the roll about
    the approach axis free, so a direction sets the approach axis alone; the
    roll the gripper ends at is not chosen.

    The solver has its own placo model and QP, apart from Kinematics, and
    its own seeds: a caller gives a position and a direction only. It is not
    thread safe: give each thread or process its own.

    How it searches. A solve is a descent of the QP from a seed, with a
    position task and an axis-align task on the grasp frame. Each step turns
    an arm joint by at most APPROACH_STEP_RAD and the wrist roll by at most
    APPROACH_ROLL_STEP_RAD. The seeds come from a table of postures across
    the joint space, taken once at construction at pan zero: the solver turns
    the table about the pan axis so that the arm's vertical plane holds the
    target (facing it, or reaching over the shoulder from behind) and, for a
    direction with a horizontal part, the direction. It then starts from the
    postures nearest the asked pose, and from one at the pan that faces the
    target, rank by rank: in each rank of
    APPROACH_SEED_RANKS_METRES_PER_RADIAN, a radian of the approach axis
    weighs as much as that many metres of the position (see _seeds). The
    seeds do not depend on the bars. Position and approach are soft
    objectives of the QP, the heavier at a weight of APPROACH_QP_MAX_WEIGHT,
    and an approach at the edge of the reach meets both bars only in a
    narrow range of their trade: the solver searches that trade from one
    seed (see _Trade).

    By construction, an answer is verified: the joints are moved into the
    joint limits, and forward kinematics must put them within both bars. A
    position that the reach sphere (`reach_sphere`, see ReachSphere) rules
    out at the position bar gets None at once: no pose meets that bar there.

    Measured on this model at the bars of a workspace answer, 0.01 m and
    0.05 rad, on samples not used to tune the search. Of 1800 random
    in-limit joint vectors (RNG seeds 9301, 9302 and 9303), it reaches the
    grasp point of each along its own approach axis, and in a direction
    0.022 to 0.042 rad from that axis. Of 6000 targets near the corner of
    both bars (RNG seeds 888001, 888002 and 888003), it leaves 2 unreached:
    each target is 9 mm from the grasp point of a random in-limit joint
    vector, in a random direction, and its direction is 0.045 rad from the
    approach axis of that joint vector.

    Measured at 21 pairs of bars on the same 1800 joint vectors. The pairs,
    as a position bar in metres with its angle bars in radians: 0.001 with
    0.005 and 0.05; 0.002 with 0.01, 0.02, 0.05 and 0.1; 0.005 with 0.025
    and 0.05; 0.01 with 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 1 and pi;
    0.02 with 0.05; 0.05 with 0.01, 0.05 and 0.5. Along their own approach
    axis, it leaves 14 unreached at 0.001 m and 0.005 rad, 11 at 0.002 m
    and 0.01 rad, 3 at 0.002 m and 0.02 rad, 1 at 0.01 m and 0.001 rad, and
    none at the other pairs. In a direction 0.022 to 0.042 rad from their
    approach axis, at the pairs with an angle bar of 0.05 rad, it leaves 1
    unreached at 0.001 m and 1 at 0.002 m. At 0.01 m and pi rad, where
    every direction meets the angle bar, it reaches each grasp point in a
    random direction. No pose reached at one pair is missed at a looser
    pair. This is a measurement, not a rule of the search: the seeds do not
    depend on the bars, but the trades of a descent start at the ratio of
    the bars.

    On the samples that the search is tuned on, at the bars of a workspace
    answer, it leaves none unreached: the grasp points of in-limit joint
    vectors whose approach axis is within 0.03 rad of straight down or
    straight forward, on two fixed joint grids (33465 targets), and of 3000
    random in-limit joint vectors (RNG seed 12345) along their own approach
    axis. On the workspace grid of 51 rows by 13 lanes above surfaces at
    seven heights from -0.2 to 0.2 m, also a sample that the search is
    tuned on, it reaches each grid point that a dense sampling of the joint
    space reaches, and 15 more at the corner of both bars.

    A full workspace grid (down, then forward, at each grid point the reach
    sphere does not rule out, and the least distance of each grid point no
    direction reaches) takes 0.9 to 1.8 s of CPU time on a loaded AMD EPYC
    Genoa, at surface heights from -0.3 to 0.4 m.
    """

    def __init__(self, urdf_path: str):
        self._limits = limits_from_description(DESCRIPTION)
        self._robot = _arm_model(urdf_path)
        _set_joints(self._robot, (0.0,) * len(ARM.joints))
        self.reach_sphere = ReachSphere.of(urdf_path, self._robot)
        self._pan_axis = _PanAxis.of(self._robot)
        self._seed_table = _SeedTable.of(self._robot, self._limits, self._pan_axis)
        self._side_rolls = _side_rolls(self._limits)

        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.mask_fbase(True)
        self._solver.enable_joint_limits(True)
        # With a time step of one, a joint's velocity limit is the most one
        # step turns it: the trust region.
        self._solver.enable_velocity_limits(True)
        self._solver.dt = 1.0
        for name in ARM.joints:
            self._robot.set_velocity_limit(name, APPROACH_STEP_RAD)
        self._robot.set_velocity_limit(_ROLL_JOINT, APPROACH_ROLL_STEP_RAD)
        self._position_task = self._solver.add_position_task(END_EFFECTOR_FRAME, np.zeros(3))
        self._approach_task = self._solver.add_axisalign_task(
            END_EFFECTOR_FRAME, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
        )

    def grasp_frame(
        self, joints: tuple[float, ...]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Where the grasp point stands with the arm at `joints`, joint
        radians in the arm's order, and the way its approach axis points
        there, in the model's base frame."""
        _set_joints(self._robot, joints)
        frame = self._robot.get_T_world_frame(END_EFFECTOR_FRAME)
        point, approach = frame[:3, 3], frame[:3, 2]
        return (
            (float(point[0]), float(point[1]), float(point[2])),
            (float(approach[0]), float(approach[1]), float(approach[2])),
        )

    def least_distance(self, position) -> float:
        """The least distance, in metres, the grasp point comes to `position`
        (x, y, z in metres, in the model's base frame) in any orientation,
        inside the joint limits.

        The least of the descents from the APPROACH_DISTANCE_SEEDS seeds
        nearest by the position alone, and from one at the pan that faces
        the target when none of them is there. When all of these stop above
        APPROACH_DISTANCE_RESOLUTION_M and the reach sphere does not rule
        the position out at that resolution, also from the nearest seed
        turned to each side roll (see _side_rolls): with the pan at a limit,
        only the wrist roll moves the grasp point to the side of the arm's
        plane, and a descent from the roll of a seed can stop at the limit
        of the roll on the wrong side. A descent stops as soon as it comes
        within APPROACH_DISTANCE_RESOLUTION_M.

        The answer is the distance at one step of a descent. The QP keeps
        the joints of a step inside the joint limits only to a measured
        error (see APPROACH_QP_MAX_WEIGHT). So the answer is a distance that
        the grasp point reaches at joints inside the limits to that error,
        and it is never shorter than the true least distance by more than
        that error moves the grasp point. Each descent is local, so the
        answer can be longer.

        Measured on 480000 random in-limit joint vectors, a sample not used
        to tune the search (RNG seeds 9101 to 9104, 60000 each; and 9201 to
        9204, 60000 each with the pan within 0.4 rad of a limit), whose
        grasp points the arm reaches: 40 answers are longer than the
        resolution (1 in 12000), 27 longer than 1 mm, 3 longer than 5 mm,
        and the longest is 6.4 mm. Of these 40, the grasp point of 29 is
        within 0.09 m of the pan axis; of the other 11, 7 have the pan
        within 0.05 rad of a limit, and 4 the shoulder lift, the elbow or
        the wrist flex within 0.15 rad of a limit. On 716 other targets, a
        sample that the search is tuned on (the workspace grid above a
        surface at 0 m, and 53 points around the reach and far past it), no
        answer is more than 1 mm longer than a dense sampling of the joint
        space.

        Raises ValueError for a position that is not three finite
        coordinates each within APPROACH_MAX_COORDINATE_M."""
        target = _target(position)
        seeds = list(self._seeds(target, None, APPROACH_DISTANCE_SEEDS))
        if not self.reach_sphere.rules_out(target, APPROACH_DISTANCE_RESOLUTION_M):
            seeds += [(*seeds[0][:-1], roll) for roll in self._side_rolls]
        least = math.inf
        for seed in seeds:
            least = min(least, self._least_distance_from(seed, target))
            if least <= APPROACH_DISTANCE_RESOLUTION_M:
                break
        return least

    def approach(
        self,
        position,
        direction,
        *,
        position_tolerance_m: float,
        angle_tolerance_rad: float,
    ) -> tuple[float, ...] | None:
        """Joint radians, in the arm's order and inside the joint limits, that put
        the grasp point within `position_tolerance_m` of `position` with the
        approach axis within `angle_tolerance_rad` of `direction`, verified by
        forward kinematics. None when no descent finds them, from the seeds
        of each rank (see _seeds): the APPROACH_SEEDS nearest, and one at the
        pan that faces the target when none of them is there. None at once
        when the reach sphere rules out the position at
        `position_tolerance_m`.

        `position` is x, y, z in metres in the model's base frame;
        `direction` is a vector in that frame, of any length.

        Raises ValueError for a position that is not three finite coordinates
        each within APPROACH_MAX_COORDINATE_M, for a direction that is not
        three finite components with a length, for a position tolerance that
        is not above zero and at most APPROACH_MAX_POSITION_BAR_M, and for an
        angle tolerance that is not from APPROACH_MIN_ANGLE_BAR_RAD to pi."""
        target = _target(position)
        axis = _unit_direction(direction)
        bars = _Bars.of(position_tolerance_m, angle_tolerance_rad)
        if self.reach_sphere.rules_out(target, bars.position_m):
            return None
        for seed in self._seeds(target, axis, APPROACH_SEEDS):
            joints = self._approach_from(seed, target, axis, bars)
            if joints is not None:
                return joints
        return None

    def _seeds(
        self, target: np.ndarray, direction: np.ndarray | None, count: int
    ) -> Iterator[tuple[float, ...]]:
        """The seeds of a descent toward `target`, in the order to try them,
        each once (see _SeedTable.nearest): the table's postures at each pan
        whose arm plane holds the target or the horizontal part of
        `direction`. With a direction, for each rank of
        APPROACH_SEED_RANKS_METRES_PER_RADIAN in turn, the `count` seeds
        nearest the asked pose and one at the pan that faces the target when
        none of them is there; the seeds of the next rank are computed only
        when those of the earlier ranks are all taken. Without a direction,
        one rank: the `count` seeds nearest by the position alone, and one at
        the pan that faces the target when none of them is there."""
        offset = target - self._pan_axis.origin
        pans = self._pan_axis.pans_holding(_bearing(offset), self._limits)
        if direction is not None and (direction[0] != 0.0 or direction[1] != 0.0):
            for pan in self._pan_axis.pans_holding(_bearing(direction), self._limits):
                if pan not in pans:
                    pans.append(pan)
        asked = [
            (
                pan,
                self._pan_axis.seen_at_pan_zero(offset, pan),
                None if direction is None else self._pan_axis.seen_at_pan_zero(direction, pan),
            )
            for pan in pans
        ]
        ranks = APPROACH_SEED_RANKS_METRES_PER_RADIAN if direction is not None else (0.0,)
        return self._seed_table.nearest(asked, ranks, count)

    def _least_distance_from(self, seed: tuple[float, ...], target: np.ndarray) -> float:
        """The least distance a descent from `seed` brings the grasp point
        to `target`, the approach axis free."""
        _set_joints(self._robot, seed)
        self._position_task.target_world = target
        self._weigh_approach(0.0)
        stall = _Stall()
        least = math.inf
        for _ in range(APPROACH_DISTANCE_ITERATIONS):
            distance = _distance_to(self._step(), target)
            least = min(least, distance)
            if least <= APPROACH_DISTANCE_RESOLUTION_M or stall.after(distance):
                break
        return least

    def _approach_from(
        self,
        seed: tuple[float, ...],
        target: np.ndarray,
        direction: np.ndarray,
        bars: _Bars,
    ) -> tuple[float, ...] | None:
        """Verified joints a descent from `seed` finds within both bars, at
        one of up to APPROACH_TRADES trades, or None."""
        _set_joints(self._robot, seed)
        self._position_task.target_world = target
        self._approach_task.targetAxis_world = direction
        trade = _Trade(bars)
        for _ in range(APPROACH_TRADES):
            self._weigh_approach(trade.weight)
            stopped_at = self._descend(target, direction, trade.weight, bars)
            if stopped_at is None:
                return self._verified(target, direction, bars)
            if not trade.retune(*stopped_at):
                return None
        return None

    def _descend(
        self, target: np.ndarray, direction: np.ndarray, weight: float, bars: _Bars
    ) -> tuple[float, float] | None:
        """Steps from the model's joints at one trade: None as soon as the
        grasp frame meets both bars, else the distance and the angle it
        stalls at."""
        stall = _Stall()
        for _ in range(APPROACH_ITERATIONS):
            frame = self._step()
            distance, angle = _distance_to(frame, target), _angle_to(frame, direction)
            if bars.met(distance, angle):
                return None
            if stall.after(math.sqrt(distance * distance + weight * angle * angle)):
                break
        return distance, angle

    def _weigh_approach(self, weight: float) -> None:
        """Weighs the approach objective `weight` times the position
        objective in the QP, the heavier of the two at
        APPROACH_QP_MAX_WEIGHT."""
        scale = APPROACH_QP_MAX_WEIGHT / max(1.0, weight)
        self._position_task.configure("position", "soft", scale)
        self._approach_task.configure("approach", "soft", weight * scale)

    def _step(self) -> np.ndarray:
        """One QP step from the model's joints: the grasp frame it lands on."""
        self._solver.solve(True)
        self._robot.update_kinematics()
        return self._robot.get_T_world_frame(END_EFFECTOR_FRAME)

    def _verified(
        self, target: np.ndarray, direction: np.ndarray, bars: _Bars
    ) -> tuple[float, ...] | None:
        """The model's joints moved into the joint limits, when forward
        kinematics then puts the grasp frame within both bars."""
        joints = self._limits.clamp(_joints(self._robot))
        _set_joints(self._robot, joints)
        frame = self._robot.get_T_world_frame(END_EFFECTOR_FRAME)
        if bars.met(_distance_to(frame, target), _angle_to(frame, direction)):
            return joints
        return None


@dataclass(frozen=True)
class _Bars:
    """How close an approach must come: the grasp point within `position_m`
    of the target, the approach axis within `angle_rad` of the direction."""

    position_m: float
    angle_rad: float

    @classmethod
    def of(cls, position_m: float, angle_rad: float) -> _Bars:
        """A caller's bars, inside the range the solver takes: the position
        bar above zero and at most APPROACH_MAX_POSITION_BAR_M, the angle bar
        from APPROACH_MIN_ANGLE_BAR_RAD to pi. Raises ValueError for a bar
        outside its range; NaN fails every comparison, so it is refused too."""
        if not APPROACH_MIN_ANGLE_BAR_RAD <= angle_rad <= math.pi:
            raise ValueError(
                f"an angle tolerance is from {APPROACH_MIN_ANGLE_BAR_RAD} rad to pi, "
                f"got {angle_rad}"
            )
        return cls(position_m=_position_bar(position_m), angle_rad=angle_rad)

    def met(self, distance: float, angle: float) -> bool:
        return distance <= self.position_m and angle <= self.angle_rad

    def metres_per_radian(self) -> float:
        """How many metres of a position miss weigh as much as one radian of
        an approach miss: the ratio of the bars."""
        return self.position_m / self.angle_rad


class _Trade:
    """The weight of the approach objective against the position objective
    in the QP of one approach. A descent at one weight stops where the two
    objectives balance, and a heavier weight gives up distance for angle.

    The first weight, (metres per radian)^2, makes a miss of either bar cost
    the same. When a descent stalls missing one bar only, the next weight
    moves toward that bar: by APPROACH_TRADE_FACTOR until a weight is known
    on each side, then to the geometric mean of the nearest weights known on
    each side, a bisection."""

    def __init__(self, bars: _Bars):
        self._bars = bars
        self.weight = bars.metres_per_radian() ** 2
        # The heaviest weight known to miss the angle bar, and the lightest
        # known to miss the position bar.
        self._misses_angle: float | None = None
        self._misses_position: float | None = None

    def retune(self, distance: float, angle: float) -> bool:
        """Moves the weight after a descent that stalled `distance` from the
        target with its approach axis `angle` off. False when it misses both
        bars: a heavier weight then costs more distance and a lighter one
        more angle, so no trade on this descent meets both."""
        if distance <= self._bars.position_m:
            self._misses_angle = self.weight
            self.weight = (
                self.weight * APPROACH_TRADE_FACTOR
                if self._misses_position is None
                else math.sqrt(self.weight * self._misses_position)
            )
            return True
        if angle <= self._bars.angle_rad:
            self._misses_position = self.weight
            self.weight = (
                self.weight / APPROACH_TRADE_FACTOR
                if self._misses_angle is None
                else math.sqrt(self.weight * self._misses_angle)
            )
            return True
        return False


class _Stall:
    """Whether a descent has stopped making headway: its error, in metres,
    has not shrunk by APPROACH_STALL_GAIN_M within APPROACH_STALL_STEPS
    steps."""

    def __init__(self):
        self._least = math.inf
        self._idle_steps = 0

    def after(self, error: float) -> bool:
        """Takes the error after one more step: True once the descent
        stalls."""
        if error < self._least - APPROACH_STALL_GAIN_M:
            self._idle_steps = 0
        else:
            self._idle_steps += 1
        self._least = min(self._least, error)
        return self._idle_steps >= APPROACH_STALL_STEPS


@dataclass(frozen=True)
class _PanAxis:
    """The vertical axis the shoulder pan turns the rest of the arm about.

    `origin` is the axis' point at height zero; the arm's bearing about the
    axis grows by `turn` (1 or -1) times the pan, from `bearing_at_zero`,
    the grasp point's bearing at the zero posture."""

    origin: np.ndarray
    turn: float
    bearing_at_zero: float

    @classmethod
    def of(cls, robot) -> _PanAxis:
        """Read from the model at its zero posture. Raises ValueError when
        the pan does not turn about a vertical axis."""
        pan_frame = robot.get_T_world_frame(_PAN_JOINT)
        axis = pan_frame[:3, 2]
        if not math.isclose(abs(axis[2]), 1.0, abs_tol=1e-9):
            raise ValueError(f"{_PAN_JOINT} does not turn about a vertical axis")
        origin = np.array([pan_frame[0, 3], pan_frame[1, 3], 0.0])
        grasp_point = robot.get_T_world_frame(END_EFFECTOR_FRAME)[:3, 3]
        return cls(
            origin=origin,
            turn=math.copysign(1.0, axis[2]),
            bearing_at_zero=_bearing(grasp_point - origin),
        )

    def pans_holding(self, bearing: float, limits: JointLimits) -> list[float]:
        """The pans that put the arm's vertical plane through `bearing` about
        the axis: the one that faces it and the one that reaches it over the
        shoulder from behind, each moved into the pan's limits, without
        repeats."""
        pans: list[float] = []
        for arm_bearing in (bearing, bearing + math.pi):
            pan = _clamped(
                _wrapped(self.turn * (arm_bearing - self.bearing_at_zero)),
                limits.lower[0],
                limits.upper[0],
            )
            if pan not in pans:
                pans.append(pan)
        return pans

    def seen_at_pan_zero(self, vector: np.ndarray, pan: float) -> np.ndarray:
        """`vector`, a world vector, turned back about the axis by what `pan`
        turns the arm: where it lies for the arm at pan zero."""
        angle = -self.turn * pan
        cos, sin = math.cos(angle), math.sin(angle)
        return np.array(
            [cos * vector[0] - sin * vector[1], sin * vector[0] + cos * vector[1], vector[2]]
        )


@dataclass(frozen=True)
class _SeedTable:
    """Postures on a grid across the limits of the shoulder lift, the elbow
    and the wrist flex, at pan zero and at each seed roll, with the grasp
    point each puts the arm at (an offset from the pan axis' origin) and its
    approach axis. Every posture is inside the joint limits, as the QP needs
    of a seed: the solver replaces the pan with one inside the limits."""

    postures: np.ndarray
    points: np.ndarray
    approaches: np.ndarray
    squared_point_norms: np.ndarray

    @classmethod
    def of(cls, robot, limits: JointLimits, pan_axis: _PanAxis) -> _SeedTable:
        """Taken by forward kinematics on the model; leaves the model at the
        last posture."""
        grids = [
            np.linspace(limits.lower[joint], limits.upper[joint], APPROACH_SEED_GRID_POINTS)
            for joint in _GRID_JOINTS
        ]
        postures, points, approaches = [], [], []
        for roll, lift, elbow, flex in itertools.product(_seed_rolls(limits), *grids):
            posture = (0.0, float(lift), float(elbow), float(flex), roll)
            _set_joints(robot, posture)
            frame = robot.get_T_world_frame(END_EFFECTOR_FRAME)
            postures.append(posture)
            points.append(frame[:3, 3] - pan_axis.origin)
            approaches.append(frame[:3, 2].copy())
        points_array = np.array(points)
        return cls(
            postures=np.array(postures),
            points=points_array,
            approaches=np.array(approaches),
            squared_point_norms=np.einsum("ij,ij->i", points_array, points_array),
        )

    def nearest(
        self,
        asked: list[tuple[float, np.ndarray, np.ndarray | None]],
        ranks: tuple[float, ...],
        count: int,
    ) -> Iterator[tuple[float, ...]]:
        """For each rank in turn, the seeds nearest the asked poses in that
        rank (see _nearest_in_rank), less those that repeat a seed of an
        earlier rank. Each asked pose is a pan with the target's offset and
        the direction (or None) seen at pan zero, the pan that faces the
        target first; a seed is a posture of the table at that pan. A rank
        is how many metres a radian between a posture's approach axis and
        the direction counts as. The seeds of a rank are computed only when
        those of the earlier ranks are all taken."""
        taken: list[tuple[int, float]] = []
        for metres_per_radian in ranks:
            for row, pan in self._nearest_in_rank(asked, metres_per_radian, count):
                if _repeats_a_seed(row, pan, taken):
                    continue
                taken.append((row, pan))
                yield self._seed(row, pan)

    def _nearest_in_rank(
        self,
        asked: list[tuple[float, np.ndarray, np.ndarray | None]],
        metres_per_radian: float,
        count: int,
    ) -> list[tuple[int, float]]:
        """The `count` seeds nearest the asked poses, nearest first, and one
        more at the pan that faces the target when none of them is there,
        each as a row of the table and a pan.

        A posture is as far from an asked pose as their points are, and with
        a direction their unit approach axes too, scaled so that a radian
        between them counts as `metres_per_radian` metres; the two add in
        squares. A seed that repeats a nearer seed (see _repeats_a_seed) is
        skipped. When no seed is at the pan that faces the target or closer
        than APPROACH_SEED_PAN_SPACING_RAD to it, the nearest posture at
        that pan comes last, so that every search tries the arm facing the
        target: the nearest seeds can all reach over the shoulder, or turn
        the arm's plane along the direction."""
        costs = self._costs(asked, metres_per_radian)
        flat = costs.ravel()
        # Each seed taken skips at most one posture at each other pan.
        candidates = min(count * len(asked), flat.size)
        nearest = np.argpartition(flat, candidates - 1)[:candidates]
        nearest = nearest[np.argsort(flat[nearest], kind="stable")]
        taken: list[tuple[int, float]] = []
        for index in nearest:
            row, column = divmod(int(index), len(asked))
            pan = asked[column][0]
            if not _repeats_a_seed(row, pan, taken):
                taken.append((row, pan))
            if len(taken) == count:
                break
        facing_pan = asked[0][0]
        if not any(_pans_close(pan, facing_pan) for _, pan in taken):
            taken.append((int(np.argmin(costs[:, 0])), facing_pan))
        return taken

    def _costs(
        self, asked: list[tuple[float, np.ndarray, np.ndarray | None]], metres_per_radian: float
    ) -> np.ndarray:
        """How far each posture of the table (a row) is from each asked pose
        (a column), as a rank: the squared distance less what is the same
        for every posture.

        |p - t|^2 + m^2 |a - d|^2 = |p|^2 - 2 p.t - 2 m^2 a.d + |t|^2 + 2 m^2:
        |t| is the same at every pan and |a| = |d| = 1, so the last two terms
        rank nothing."""
        axis_weight = 2.0 * metres_per_radian**2
        costs = np.empty((len(self.postures), len(asked)))
        for column, (_, offset, direction) in enumerate(asked):
            costs[:, column] = self.squared_point_norms - 2.0 * (self.points @ offset)
            if direction is not None:
                costs[:, column] -= axis_weight * (self.approaches @ direction)
        return costs

    def _seed(self, row: int, pan: float) -> tuple[float, ...]:
        """The posture of the table at `row`, turned to `pan`."""
        _, lift, elbow, flex, roll = (float(value) for value in self.postures[row])
        return (pan, lift, elbow, flex, roll)


def _repeats_a_seed(row: int, pan: float, taken: list[tuple[int, float]]) -> bool:
    """Whether the posture of the seed table at `row`, turned to `pan`,
    repeats a seed `taken` (a row and a pan each): the same posture at a pan
    closer than APPROACH_SEED_PAN_SPACING_RAD starts one descent twice."""
    return any(row == taken_row and _pans_close(pan, taken_pan) for taken_row, taken_pan in taken)


def _pans_close(pan: float, other: float) -> bool:
    """Whether two pans are closer than APPROACH_SEED_PAN_SPACING_RAD."""
    return abs(pan - other) < APPROACH_SEED_PAN_SPACING_RAD


def _seed_rolls(limits: JointLimits) -> tuple[float, float]:
    """The wrist rolls of the seed table: zero, and the roll the limits allow
    nearest a half turn, each inside the limits. The grasp point sits off
    the roll axis, so the roll swings it on a small circle, and at the edge
    of the reach a descent from one side of that circle seldom turns it to
    the other: the table seeds both sides."""
    lower, upper = limits.lower[-1], limits.upper[-1]
    half_turns = (_clamped(math.pi, lower, upper), _clamped(-math.pi, lower, upper))
    nearest_half_turn = min(half_turns, key=lambda roll: math.pi - abs(roll))
    return (_clamped(0.0, lower, upper), nearest_half_turn)


def _side_rolls(limits: JointLimits) -> tuple[float, float]:
    """The wrist rolls a quarter turn each way from zero, each inside the
    limits. The grasp point sits off the roll axis, so these rolls swing it
    farthest to each side of the arm's vertical plane."""
    lower, upper = limits.lower[-1], limits.upper[-1]
    return (_clamped(-math.pi / 2, lower, upper), _clamped(math.pi / 2, lower, upper))


def _target(position) -> np.ndarray:
    """A target position: three finite coordinates in metres, each within
    APPROACH_MAX_COORDINATE_M."""
    point = np.asarray(position, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"a position has three coordinates, got {position!r}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"a position has finite coordinates, got {position!r}")
    if np.max(np.abs(point)) > APPROACH_MAX_COORDINATE_M:
        raise ValueError(
            f"a position has coordinates within {APPROACH_MAX_COORDINATE_M} m, got {position!r}"
        )
    return point


def _position_bar(position_m: float) -> float:
    """A position bar inside the range the solver takes: above zero and at
    most APPROACH_MAX_POSITION_BAR_M. Raises ValueError for a bar outside
    it; NaN fails every comparison, so it is refused too."""
    if not 0.0 < position_m <= APPROACH_MAX_POSITION_BAR_M:
        raise ValueError(
            "a position tolerance is above 0 and at most "
            f"{APPROACH_MAX_POSITION_BAR_M} m, got {position_m}"
        )
    return position_m


def _unit_direction(direction) -> np.ndarray:
    """A direction as a unit vector: three finite components with a length.
    The vector is scaled by its largest component first, so that neither a
    tiny nor a huge vector loses its direction to rounding."""
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"a direction has three components, got {direction!r}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"a direction has finite components, got {direction!r}")
    largest = float(np.max(np.abs(vector)))
    if largest == 0.0:
        raise ValueError("a direction has a length, got the zero vector")
    scaled = vector / largest
    return scaled / np.linalg.norm(scaled)


def _distance_to(frame: np.ndarray, target: np.ndarray) -> float:
    """How far the grasp point of the grasp frame `frame` is from `target`,
    in metres."""
    return float(np.linalg.norm(frame[:3, 3] - target))


def _angle_to(frame: np.ndarray, direction: np.ndarray) -> float:
    """The angle, in radians, between the approach axis of the grasp frame
    `frame` and the unit vector `direction`."""
    cosine = float(frame[:3, 2] @ direction)
    return math.acos(min(max(cosine, -1.0), 1.0))


def _bearing(vector: np.ndarray) -> float:
    """The bearing of a vector's horizontal part, from +X toward +Y."""
    return math.atan2(vector[1], vector[0])


def _wrapped(angle: float) -> float:
    """`angle` wrapped into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamped(value: float, lower: float, upper: float) -> float:
    """`value` moved into [lower, upper]."""
    return min(max(value, lower), upper)
