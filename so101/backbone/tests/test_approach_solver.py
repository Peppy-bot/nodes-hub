"""The approach solver against the real SO-101 model: targets reachable by
construction, verified answers, the seeds, the reach sphere, the least
distance past the reach, and the inputs it refuses. Every target and joint
vector comes from a fixed grid, a fixed list or a fixed RNG seed. The
workspace grid, the bars and the grasp directions are those of a workspace
answer, from workspace_core_py."""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from so101.transforms import matrix_from_pose
from workspace_core_py import (
    GRASP_ANGLE_TOLERANCE,
    GRASP_DIRECTIONS,
    REACH_TOLERANCE,
    SurfaceHeight,
)

from so101_backbone import kinematics as kinematics_module
from so101_backbone.kinematics import (
    APPROACH_DISTANCE_RESOLUTION_M,
    APPROACH_MAX_COORDINATE_M,
    APPROACH_MAX_POSITION_BAR_M,
    APPROACH_MIN_ANGLE_BAR_RAD,
    APPROACH_SEED_PAN_SPACING_RAD,
    APPROACH_SEED_RANKS_METRES_PER_RADIAN,
    APPROACH_SEEDS,
    ApproachSolver,
    Kinematics,
    _SeedTable,
)
from so101_backbone.limits import from_description
from so101_backbone.model import DESCRIPTION, urdf_path

LIMITS = from_description(DESCRIPTION)
BARS = {"position_tolerance_m": REACH_TOLERANCE, "angle_tolerance_rad": GRASP_ANGLE_TOLERANCE}
APPROACHES = {direction.name: direction.approach for direction in GRASP_DIRECTIONS}
DOWN = APPROACHES["down"]
FORWARD = APPROACHES["forward"]
# How close to the direction the approach axis of a target reachable by
# construction is: well inside the angle bar.
CONSTRUCTION_ANGLE_RAD = 0.03


@pytest.fixture(scope="module")
def solver():
    return ApproachSolver(urdf_path())


@pytest.fixture(scope="module")
def kinematics():
    """An independent model to check answers with."""
    return Kinematics(urdf_path())


def grasp_frame(kinematics, joints):
    """The grasp point and the approach axis (the grasp frame's +Z) of
    `joints`, by forward kinematics."""
    position, orientation = kinematics.forward_kinematics(joints)
    return np.array(position), matrix_from_pose(position, orientation)[:3, 2]


def angle_between(a, b):
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.acos(min(max(cosine, -1.0), 1.0))


def assert_verified(
    kinematics,
    joints,
    target,
    direction,
    *,
    position_bar=REACH_TOLERANCE,
    angle_bar=GRASP_ANGLE_TOLERANCE,
):
    assert LIMITS.contains(joints), f"{joints} leaves the joint limits"
    point, approach_axis = grasp_frame(kinematics, joints)
    assert math.dist(point, target) <= position_bar, (joints, point, target)
    assert angle_between(approach_axis, direction) <= angle_bar, (joints, approach_axis)


def targets_reachable_by_construction(kinematics, direction, pans):
    """The grasp points of in-limit joint vectors on a fixed grid whose
    approach axis is within CONSTRUCTION_ANGLE_RAD of `direction`.

    The shoulder lift, the elbow and the wrist flex all pitch the approach
    axis in the arm's plane, which their sum tips below the horizon: the grid
    takes the wrist flex that makes the sum the angle `direction` dips, or
    that angle less a turn. Forward kinematics keeps only the joint vectors
    whose approach axis is close enough."""
    dip = math.asin(-direction[2])
    pitch_sums = (dip, dip - 2 * math.pi)
    lifts = np.linspace(LIMITS.lower[1], LIMITS.upper[1], 16)
    elbows = np.linspace(LIMITS.lower[2], LIMITS.upper[2], 16)
    rolls = (-2.6, -1.0, 0.5, 2.7)
    targets = []
    for pan, lift, elbow, pitch_sum, roll in itertools.product(
        pans, lifts, elbows, pitch_sums, rolls
    ):
        joints = (pan, float(lift), float(elbow), float(pitch_sum - lift - elbow), roll)
        if not LIMITS.contains(joints):
            continue
        point, approach_axis = grasp_frame(kinematics, joints)
        if angle_between(approach_axis, direction) <= CONSTRUCTION_ANGLE_RAD:
            targets.append((tuple(point), joints))
    return targets


@pytest.mark.parametrize(
    ("direction", "pans", "least_targets"),
    [
        (DOWN, (-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8), 3500),
        # Forward holds only at a pan that faces +X.
        (FORWARD, (-0.02, 0.0, 0.02), 2000),
    ],
    ids=["down", "forward"],
)
def test_every_target_of_a_joint_grid_reachable_by_construction_is_reached(
    solver, kinematics, direction, pans, least_targets
):
    targets = targets_reachable_by_construction(kinematics, direction, pans)
    # A grid that builds few targets would prove nothing.
    assert len(targets) >= least_targets
    unreached = []
    for target, built_from in targets:
        joints = solver.approach(target, direction, **BARS)
        if joints is None:
            unreached.append((target, built_from))
            continue
        assert_verified(kinematics, joints, target, direction)
    assert not unreached, f"{len(unreached)} of {len(targets)} unreached: {unreached[:5]}"


def test_every_pose_of_a_joint_grid_is_reached_along_its_own_approach_axis(solver, kinematics):
    # Any direction, not only down and forward: a direction that leans
    # sideways holds only at the pans that turn the arm's plane along it.
    grid = itertools.product(
        (-1.6, -0.5, 0.4, 1.5),
        np.linspace(-1.5, 1.5, 5),
        np.linspace(-1.5, 1.5, 5),
        np.linspace(-1.5, 1.5, 5),
        (-2.0, 0.7),
    )
    unreached = []
    for built_from in grid:
        target, direction = grasp_frame(kinematics, built_from)
        joints = solver.approach(target, direction, **BARS)
        if joints is None:
            unreached.append(built_from)
            continue
        assert_verified(kinematics, joints, target, direction)
    assert not unreached, f"{len(unreached)} unreached: {unreached[:5]}"


def workspace_grid_targets(surface_height_m):
    """The targets of the workspace grid above a surface, in the grid's
    order."""
    return SurfaceHeight.from_wire(surface_height_m).grid_targets()


# The grid points known reachable above a surface at each height, drawn row
# by row of the workspace grid from the nearest, each row from the robot's
# right to its left. "x": a sampling of the joint space every 0.015 rad of
# the lift and the elbow (three wrist flexes and eight rolls each) puts the
# grasp point within both bars, down or forward. "o": past that sampling, an
# answer of the solver meets both bars near their corner, verified by
# forward kinematics. ".": neither. The rows after the last one drawn are
# all ".".
KNOWN_REACHABLE_GRID_POINTS = {
    -0.1: (
        "xxxxx...xxxxx",
        "xxxxxxoxxxxxx",
        "xxxxxxxxxxxxx",
        "xxxxxxxxxxxxx",
        "xxxxxxxxxxxxx",
        "oxxxxxxxxxxxo",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        "..xxxxxxxxx..",
        "..xxxxxxxxx..",
        "..xxxxxxxxx..",
        "...xxxxxxx...",
        "...xxxxxxx...",
        "....xxxxx....",
        ".....oxo.....",
        "......x......",
        "......x......",
        "......x......",
        "......o......",
    ),
    0.0: (
        ".xxxx...xxxx.",
        ".xxxxx.xxxxx.",
        ".xxxxx.xxxxx.",
        ".xxxxx.xxxxx.",
        ".xxxxx.xxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        ".xxxxxxxxxxx.",
        "..xxxxxxxxx..",
        "..xxxxxxxxx..",
        "..xxxxxxxxx..",
        "...xxxxxxx...",
        "...xxxxxxx...",
        "....xxxxx....",
        ".....xxx.....",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
    ),
    0.05: (
        "..xxx...xxx..",
        "..xxo...oxx..",
        "..xx.....xx..",
        "..xxo...oxx..",
        "..xxx...xxx..",
        "..xxx...xxx..",
        "..xxx...xxx..",
        "..xxxxoxxxx..",
        "..xxxxxxxxx..",
        "...xxxxxxx...",
        "...xxxxxxx...",
        "...xxxxxxx...",
        "....xxxxx....",
        "....oxxxo....",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
        "......x......",
    ),
}


def known_reachable_targets(surface_height_m):
    """The targets of the grid points KNOWN_REACHABLE_GRID_POINTS draws
    reachable above a surface at this height."""
    targets = workspace_grid_targets(surface_height_m)
    rows = [list(row) for _, row in itertools.groupby(targets, key=lambda target: target[0])]
    drawing = KNOWN_REACHABLE_GRID_POINTS[surface_height_m]
    assert len(drawing) <= len(rows), "the drawing has more rows than the grid"
    known = []
    for row, marks in zip(rows[: len(drawing)], drawing, strict=True):
        assert len(marks) == len(row), "a drawn row has one mark per lane of the grid"
        known += [target for target, mark in zip(row, marks, strict=True) if mark != "."]
    return known


@pytest.fixture(
    scope="module",
    params=sorted(KNOWN_REACHABLE_GRID_POINTS),
    ids=lambda height: f"surface {height} m",
)
def surface_grid_answers(request, solver):
    """The surface height, and for each target of the workspace grid above
    it: its approach in each grasp direction, in the order of
    GRASP_DIRECTIONS."""
    height = request.param
    return height, [
        (
            target,
            tuple(
                solver.approach(target, direction.approach, **BARS)
                for direction in GRASP_DIRECTIONS
            ),
        )
        for target in workspace_grid_targets(height)
    ]


def test_every_answer_is_inside_the_joint_limits_and_within_both_bars(
    kinematics, surface_grid_answers
):
    _, answers = surface_grid_answers
    answered = 0
    for target, approaches in answers:
        for direction, joints in zip(GRASP_DIRECTIONS, approaches, strict=True):
            if joints is None:
                continue
            answered += 1
            assert_verified(kinematics, joints, target, direction.approach)
    assert answered > 0


def approached(approaches):
    """Whether a target has an approach in a grasp direction."""
    return any(joints is not None for joints in approaches)


def test_the_grid_points_known_reachable_are_reached(surface_grid_answers):
    height, answers = surface_grid_answers
    reached = {target for target, approaches in answers if approached(approaches)}
    unreached = [target for target in known_reachable_targets(height) if target not in reached]
    assert not unreached


def test_tighter_bars_are_the_bars_an_answer_meets(solver, kinematics):
    position_bar, angle_bar = 0.002, 0.01
    answered = 0
    for target in workspace_grid_targets(0.0)[::7]:
        joints = solver.approach(
            target, DOWN, position_tolerance_m=position_bar, angle_tolerance_rad=angle_bar
        )
        if joints is not None:
            answered += 1
            assert_verified(
                kinematics, joints, target, DOWN, position_bar=position_bar, angle_bar=angle_bar
            )
    assert answered > 0


# An in-limit joint vector whose grasp frame the verification of an answer is
# held to.
VERIFIED_POSE = (0.3, 0.4, 0.5, 0.2, 0.1)


def approach_verifying_one_seed(solver, monkeypatch, seed, target, direction):
    """approach() with `seed` as its one seed and a descent that stops at
    once, so the answer is what the verification makes of `seed` itself."""
    with monkeypatch.context() as patch:
        patch.setattr(solver, "_seeds", lambda *_: iter([seed]))
        patch.setattr(solver, "_descend", lambda *_: None)
        return solver.approach(target, direction, **BARS)


def test_joints_are_an_answer_only_when_forward_kinematics_puts_them_within_both_bars(
    solver, kinematics, monkeypatch
):
    point, axis = grasp_frame(kinematics, VERIFIED_POSE)
    across = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
    across /= np.linalg.norm(across)
    tilted = math.cos(0.2) * axis + math.sin(0.2) * across
    moved = point + 0.02 * across
    assert approach_verifying_one_seed(
        solver, monkeypatch, VERIFIED_POSE, point, axis
    ) == pytest.approx(VERIFIED_POSE)
    assert approach_verifying_one_seed(solver, monkeypatch, VERIFIED_POSE, point, tilted) is None
    assert approach_verifying_one_seed(solver, monkeypatch, VERIFIED_POSE, moved, axis) is None


# A joint grid whose poses are the targets of approaches at other bars than
# those of a workspace answer.
POSE_GRID = (
    (-1.2, 0.4, 1.6),
    (-1.2, -0.4, 0.4, 1.2),
    (-1.4, -0.5, 0.4, 1.3),
    (-1.2, -0.4, 0.4, 1.2),
    (-1.3, 0.5, 2.0),
)
NARROW_BARS = {"position_tolerance_m": 0.002, "angle_tolerance_rad": 0.01}
NARROW_ANGLE_BARS = {"position_tolerance_m": REACH_TOLERANCE, "angle_tolerance_rad": 0.01}
# Bars looser than NARROW_BARS.
LOOSER_THAN_NARROW_BARS = (
    {"position_tolerance_m": 0.002, "angle_tolerance_rad": 0.02},
    {"position_tolerance_m": 0.002, "angle_tolerance_rad": 0.05},
    NARROW_ANGLE_BARS,
    BARS,
)
WIDEST_ANGLE_BARS = {"position_tolerance_m": REACH_TOLERANCE, "angle_tolerance_rad": math.pi}
# The directions along the axes of the model's base frame.
AXIS_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
# Targets and directions that the seeds of two ranks, bars at the ends of
# their ranges, and narrow angle bars are tried on.
SPREAD_TARGETS = (
    (0.25, 0.0, 0.04),
    (0.2, 0.1, 0.1),
    (0.3, -0.1, 0.0),
    (0.1, 0.2, 0.3),
    (0.0, -0.3, 0.1),
    (0.35, 0.0, 0.2),
)
SPREAD_DIRECTIONS = (
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (-1.0, -1.0, -1.0),
    (1.0, 1.0, 0.0),
)
# In-limit joint vectors, each with a direction 0.022 to 0.042 rad from its
# approach axis: its grasp point in that direction meets the bars of a
# workspace answer by construction, and no descent from the seeds of the
# first rank meets them.
POSES_ONLY_THE_SECOND_RANK_OF_SEEDS_REACHES = (
    ((-1.2205, -0.9504, -1.4476, 1.4802, 2.0877), (0.23142, 0.57833, 0.78229)),
    ((-1.8811, 1.7234, 1.4109, 1.3635, -1.0832), (0.08693, -0.20225, 0.97547)),
    ((1.1715, -1.7196, 0.4147, -1.3148, -1.2569), (-0.3464, 0.77343, 0.53087)),
)


@pytest.fixture(scope="module")
def pose_grid(kinematics):
    """The joint vectors of POSE_GRID inside the joint limits, each with its
    grasp point and its approach axis."""
    return [
        (joints, *grasp_frame(kinematics, joints))
        for joints in itertools.product(*POSE_GRID)
        if LIMITS.contains(joints)
    ]


def unreached_along_their_own_axis(solver, poses, bars):
    """The joint vectors of `poses` whose grasp point and approach axis an
    approach at `bars` does not reach."""
    return {joints for joints, point, axis in poses if solver.approach(point, axis, **bars) is None}


def test_a_pose_of_a_joint_grid_reached_at_narrow_bars_is_reached_at_looser_ones(solver, pose_grid):
    # Measured on this grid, not a rule of the search: the seeds do not
    # depend on the bars, but the trades of a descent start at the ratio of
    # the bars (see _Trade). The class docstring gives the rate measured on
    # a random sample.
    unreached_narrow = unreached_along_their_own_axis(solver, pose_grid, NARROW_BARS)
    reached_narrow = [pose for pose in pose_grid if pose[0] not in unreached_narrow]
    # A grid the narrow bars seldom reach would prove nothing.
    assert len(reached_narrow) > len(pose_grid) // 2
    for looser in LOOSER_THAN_NARROW_BARS:
        assert not unreached_along_their_own_axis(solver, reached_narrow, looser), looser


def poses_only_the_second_rank_of_seeds_reaches(kinematics):
    """The grasp points of POSES_ONLY_THE_SECOND_RANK_OF_SEEDS_REACHES, each
    with its direction, checked to meet the bars of a workspace answer by
    construction."""
    targets = []
    for joints, direction in POSES_ONLY_THE_SECOND_RANK_OF_SEEDS_REACHES:
        point, approach_axis = grasp_frame(kinematics, joints)
        assert LIMITS.contains(joints)
        assert angle_between(approach_axis, direction) <= GRASP_ANGLE_TOLERANCE
        targets.append((point, direction))
    return targets


@pytest.mark.parametrize("position_bar", [0.002, 0.005, REACH_TOLERANCE, 0.02, 0.05])
def test_a_looser_position_bar_keeps_the_seeds_of_the_second_rank(solver, kinematics, position_bar):
    for target, direction in poses_only_the_second_rank_of_seeds_reaches(kinematics):
        joints = solver.approach(
            target,
            direction,
            position_tolerance_m=position_bar,
            angle_tolerance_rad=GRASP_ANGLE_TOLERANCE,
        )
        assert joints is not None, (target, direction)
        assert_verified(kinematics, joints, target, direction, position_bar=position_bar)


def test_the_seeds_of_the_first_rank_alone_leave_these_poses_unreached(
    solver, kinematics, monkeypatch
):
    # Without this, the poses above would not show what the second rank of
    # seeds adds.
    monkeypatch.setattr(
        kinematics_module,
        "APPROACH_SEED_RANKS_METRES_PER_RADIAN",
        APPROACH_SEED_RANKS_METRES_PER_RADIAN[:1],
    )
    for target, direction in poses_only_the_second_rank_of_seeds_reaches(kinematics):
        assert solver.approach(target, direction, **BARS) is None, (target, direction)


def seeds_of_an_approach_that_fails(solver, monkeypatch, target, direction, bars):
    """The seeds, in order, that `approach` starts a descent from when no
    descent finds joints."""
    seeds = []
    monkeypatch.setattr(solver, "_approach_from", lambda seed, *_: seeds.append(seed))
    assert solver.approach(target, direction, **bars) is None
    return seeds


def test_the_seeds_do_not_depend_on_the_bars(solver, monkeypatch):
    for target, direction in itertools.product(SPREAD_TARGETS, SPREAD_DIRECTIONS):
        seeds = seeds_of_an_approach_that_fails(solver, monkeypatch, target, direction, BARS)
        for bars in (NARROW_BARS, NARROW_ANGLE_BARS, WIDEST_ANGLE_BARS):
            assert (
                seeds_of_an_approach_that_fails(solver, monkeypatch, target, direction, bars)
                == seeds
            ), (target, direction, bars)


def ranks_of_an_approach(solver, monkeypatch, target, direction):
    """Whether `approach` finds joints, and the ranks it ranks the seed
    table at to find them, in order."""
    ranks = []
    nearest_in_rank = _SeedTable._nearest_in_rank

    def ranked(table, asked, metres_per_radian, count):
        ranks.append(metres_per_radian)
        return nearest_in_rank(table, asked, metres_per_radian, count)

    with monkeypatch.context() as patch:
        patch.setattr(_SeedTable, "_nearest_in_rank", ranked)
        joints = solver.approach(target, direction, **BARS)
    return joints is not None, ranks


def test_the_seeds_are_ranked_again_only_when_every_descent_from_the_first_rank_fails(
    solver, monkeypatch
):
    first_rank = list(APPROACH_SEED_RANKS_METRES_PER_RADIAN[:1])
    assert ranks_of_an_approach(solver, monkeypatch, (0.25, 0.05, 0.04), DOWN) == (
        True,
        first_rank,
    )
    monkeypatch.setattr(solver, "_approach_from", lambda *_: None)
    assert ranks_of_an_approach(solver, monkeypatch, (0.25, 0.05, 0.04), DOWN) == (
        False,
        list(APPROACH_SEED_RANKS_METRES_PER_RADIAN),
    )


def test_the_seeds_of_two_ranks_start_no_descent_twice(solver, monkeypatch):
    more_than_one_rank = 0
    for target, direction in itertools.product(SPREAD_TARGETS, SPREAD_DIRECTIONS):
        seeds = seeds_of_an_approach_that_fails(solver, monkeypatch, target, direction, BARS)
        # The first rank gives the APPROACH_SEEDS nearest and at most one more.
        more_than_one_rank += len(seeds) > APPROACH_SEEDS + 1
        for seed, other in itertools.combinations(seeds, 2):
            same_posture = seed[1:] == other[1:]
            assert not (same_posture and abs(seed[0] - other[0]) < APPROACH_SEED_PAN_SPACING_RAD)
    # Targets whose second rank adds no seed would prove nothing.
    assert more_than_one_rank > 0


def ranked_seeds_of_failing_approaches(solver, monkeypatch, targets_and_directions):
    """For each rank of the seed table that approaches to
    `targets_and_directions` take when no descent finds joints: the pan
    that faces the target, and the seeds of the rank, each a row of the
    table and a pan."""
    ranked = []
    nearest_in_rank = _SeedTable._nearest_in_rank

    def recorded(table, asked, metres_per_radian, count):
        taken = nearest_in_rank(table, asked, metres_per_radian, count)
        ranked.append((asked[0][0], taken))
        return taken

    with monkeypatch.context() as patch:
        patch.setattr(_SeedTable, "_nearest_in_rank", recorded)
        patch.setattr(solver, "_approach_from", lambda *_: None)
        for target, direction in targets_and_directions:
            solver.approach(target, direction, **BARS)
    return ranked


def pans_close(pan, other):
    return abs(pan - other) < APPROACH_SEED_PAN_SPACING_RAD


def test_each_rank_gives_its_nearest_seeds_once_and_one_facing_the_target(solver, monkeypatch):
    ranked = ranked_seeds_of_failing_approaches(
        solver, monkeypatch, itertools.product(SPREAD_TARGETS, SPREAD_DIRECTIONS)
    )
    facing_seed_added = 0
    for facing_pan, taken in ranked:
        nearest = taken[:APPROACH_SEEDS]
        assert len(nearest) == APPROACH_SEEDS
        for (row, pan), (other_row, other_pan) in itertools.combinations(taken, 2):
            assert not (row == other_row and pans_close(pan, other_pan))
        assert any(pans_close(pan, facing_pan) for _, pan in taken)
        facing_seed_added += not any(pans_close(pan, facing_pan) for _, pan in nearest)
    # Ranks whose nearest seeds all face the target would prove nothing.
    assert facing_seed_added > 0


def test_every_pose_of_a_joint_grid_is_reached_at_a_narrow_angle_bar(solver, pose_grid):
    # The seeds of the first rank alone leave 3 of these poses unreached, and
    # seeds ranked at the ratio of these bars alone leave 30.
    assert not unreached_along_their_own_axis(solver, pose_grid, NARROW_ANGLE_BARS)


def test_a_grasp_point_is_approached_in_any_direction_at_the_widest_angle_bar(solver, pose_grid):
    # Every direction meets an angle bar of pi: the position alone decides.
    # The seeds of the first rank put the approach axis first, where the
    # grasp point cannot follow to the position: alone, they leave 234 of
    # these approaches unreached. The second rank puts the position first.
    unreached = [
        (joints, direction)
        for joints, point, _ in pose_grid
        for direction in AXIS_DIRECTIONS
        if solver.approach(point, direction, **WIDEST_ANGLE_BARS) is None
    ]
    assert not unreached


def test_a_direction_the_arm_cannot_hold_there_is_not_approached(solver):
    # Forward needs the arm's plane along +X, and the grasp point leaves that
    # plane by less than a centimetre: 0.15 m to the side it cannot point
    # forward, though it reaches the position itself.
    target = (0.25, 0.15, 0.04)
    assert solver.least_distance(target) <= APPROACH_DISTANCE_RESOLUTION_M
    assert solver.approach(target, FORWARD, **BARS) is None
    assert solver.approach(target, DOWN, **BARS) is not None


def test_a_position_the_arm_reaches_has_a_least_distance_of_almost_nothing(solver, kinematics):
    # On this grid of 1296 positions, the descents stall short of the
    # resolution at two positions only, by less than 2 mm. The least_distance
    # docstring gives the rate on a random sample.
    grid = itertools.product(
        (-1.5, 0.0, 1.5),
        np.linspace(LIMITS.lower[1], LIMITS.upper[1], 6),
        np.linspace(LIMITS.lower[2], LIMITS.upper[2], 6),
        np.linspace(LIMITS.lower[3], LIMITS.upper[3], 4),
        (-2.0, 0.0, 2.0),
    )
    least_distances = [solver.least_distance(grasp_frame(kinematics, joints)[0]) for joints in grid]
    assert max(least_distances) <= 0.002
    within_resolution = sum(
        distance <= APPROACH_DISTANCE_RESOLUTION_M for distance in least_distances
    )
    assert within_resolution >= 0.99 * len(least_distances)


# In-limit joint vectors with the pan within 0.003 rad of a limit, so that
# the arm's plane cannot turn to their grasp point: the wrist roll swings it
# there from the side of the plane. The descents from the nearest seeds stop
# 9 to 10 mm short of it, and those from the side rolls reach it.
POSES_REACHED_FROM_THE_SIDE_ROLLS = (
    (-1.9191, 1.3545, -1.1182, -0.5204, -1.5323),
    (1.9175, 0.201, 1.426, 1.4509, -1.191),
    (-1.9171, 1.0886, -1.5738, 0.3932, -1.6282),
)


def test_a_grasp_point_at_a_pan_limit_is_reached_from_the_side_rolls(solver, kinematics):
    for joints in POSES_REACHED_FROM_THE_SIDE_ROLLS:
        least = solver.least_distance(grasp_frame(kinematics, joints)[0])
        assert least <= APPROACH_DISTANCE_RESOLUTION_M, (joints, least)


def test_the_nearest_seeds_alone_stop_short_of_these_grasp_points(solver, kinematics, monkeypatch):
    # Without this, the poses above would not show what the side rolls add.
    monkeypatch.setattr(solver, "_side_rolls", ())
    for joints in POSES_REACHED_FROM_THE_SIDE_ROLLS:
        least = solver.least_distance(grasp_frame(kinematics, joints)[0])
        assert least > 0.009, (joints, least)


def descents_of_a_least_distance(solver, monkeypatch, target):
    """The descents, in order, that `least_distance` makes: the seed of each,
    and the least distance it brings the grasp point to `target`."""
    descents = []
    least_distance_from = solver._least_distance_from

    def recorded(seed, target):
        distance = least_distance_from(seed, target)
        descents.append((seed, distance))
        return distance

    with monkeypatch.context() as patch:
        patch.setattr(solver, "_least_distance_from", recorded)
        solver.least_distance(target)
    return descents


def side_roll_seeds(solver, descents):
    return [seed for seed, _ in descents if seed[-1] in solver._side_rolls]


# A grasp point the arm reaches, which the first descent of a least distance
# comes within the resolution of.
GRASP_POINT_THE_FIRST_DESCENT_REACHES = (0.25, 0.05, 0.04)
# An in-limit joint vector: the first descent of a least distance to its
# grasp point stops 2.9 mm short, and the second comes within the
# resolution.
POSE_THE_SECOND_DESCENT_REACHES = (0.0, -0.35, 1.69, -1.65, -2.0)


def stops_at_the_first_descent_that_reaches(descents):
    """Whether the last of `descents` is the first that comes within
    APPROACH_DISTANCE_RESOLUTION_M."""
    distances = [distance for _, distance in descents]
    return distances[-1] <= APPROACH_DISTANCE_RESOLUTION_M and all(
        distance > APPROACH_DISTANCE_RESOLUTION_M for distance in distances[:-1]
    )


def test_the_side_rolls_are_tried_last_inside_the_reach_sphere_when_no_other_descent_reaches(
    solver, kinematics, monkeypatch
):
    sphere = solver.reach_sphere
    centre = np.array(sphere.centre)
    # Under the base, inside the sphere: no descent comes within the
    # resolution.
    inside = centre + (0.0, 0.0, -0.3)
    inside_descents = descents_of_a_least_distance(solver, monkeypatch, inside)
    assert all(distance > APPROACH_DISTANCE_RESOLUTION_M for _, distance in inside_descents)
    nearest = inside_descents[0][0]
    side_rolls = [(*nearest[:-1], roll) for roll in solver._side_rolls]
    assert [seed for seed, _ in inside_descents[-2:]] == side_rolls
    assert side_roll_seeds(solver, inside_descents) == side_rolls
    # Inside the sphere, a least distance stops at the first descent that
    # comes within the resolution, and so never gets to the side rolls.
    first_reaches = descents_of_a_least_distance(
        solver, monkeypatch, GRASP_POINT_THE_FIRST_DESCENT_REACHES
    )
    assert stops_at_the_first_descent_that_reaches(first_reaches)
    assert len(first_reaches) == 1
    assert LIMITS.contains(POSE_THE_SECOND_DESCENT_REACHES)
    second_reaches = descents_of_a_least_distance(
        solver, monkeypatch, grasp_frame(kinematics, POSE_THE_SECOND_DESCENT_REACHES)[0]
    )
    assert stops_at_the_first_descent_that_reaches(second_reaches)
    assert len(second_reaches) == 2
    assert not side_roll_seeds(solver, first_reaches + second_reaches)
    # Past the sphere at the resolution, no side roll.
    past = centre + (sphere.radius_m + 2 * APPROACH_DISTANCE_RESOLUTION_M, 0.0, 0.0)
    assert not side_roll_seeds(solver, descents_of_a_least_distance(solver, monkeypatch, past))


def sampled_least_distance(kinematics, target):
    """The least distance of the grasp points of a grid over the shoulder
    lift, the elbow, the wrist flex and the roll at pan zero, refined by a
    finer grid about the nearest: a distance the arm reaches, within about
    0.3 mm of the least one for a target ahead or above."""
    target = np.array(target)
    joints = (1, 2, 3, 4)

    def nearest(grids):
        return min(
            (math.dist(grasp_frame(kinematics, (0.0, *rest))[0], target), rest)
            for rest in itertools.product(*grids)
        )

    coarse = [np.linspace(LIMITS.lower[j], LIMITS.upper[j], 13) for j in joints[:3]]
    coarse.append(np.linspace(LIMITS.lower[4], LIMITS.upper[4], 5))
    _, best = nearest(coarse)
    fine = [
        np.clip(
            np.linspace(value - (grid[1] - grid[0]), value + (grid[1] - grid[0]), 9),
            LIMITS.lower[j],
            LIMITS.upper[j],
        )
        for value, grid, j in zip(best, coarse, joints, strict=True)
    ]
    distance, _ = nearest(fine)
    return distance


@pytest.mark.parametrize(
    "target",
    [(1000.0, 0.0, 0.0), (0.0, 0.0, 1000.0), (1.0, 0.0, 0.04)],
    ids=["a kilometre ahead", "a kilometre up", "a metre ahead"],
)
def test_a_point_beyond_the_reach_is_as_far_as_from_the_nearest_grasp_point(
    solver, kinematics, target
):
    # For a point far beyond the reach, the nearest grasp point is the one
    # farthest toward it.
    expected = sampled_least_distance(kinematics, target)
    assert solver.least_distance(target) == pytest.approx(expected, abs=0.002)


# The RNG seed of the random joint vectors the reach sphere is checked on.
REACH_SPHERE_SAMPLE_SEED = 20261004
# How far past the farthest grasp point found the reach sphere may stand:
# its radius is 5.4 mm more than the farthest grasp point a search of the
# joint space finds (see ReachSphere). A radius that adds an offset the
# chain does not have, as the pan joint's own (0.073 m), fails this.
REACH_SPHERE_MOST_SLACK_M = 0.01


def test_the_reach_sphere_holds_every_grasp_point_of_a_joint_grid_and_a_random_sample(
    solver, kinematics, record_property
):
    sphere = solver.reach_sphere
    grid = itertools.product(
        (LIMITS.lower[0], 0.0, LIMITS.upper[0]),
        *(np.linspace(LIMITS.lower[joint], LIMITS.upper[joint], 13) for joint in (1, 2, 3)),
        np.linspace(LIMITS.lower[4], LIMITS.upper[4], 7),
    )
    sample = np.random.default_rng(REACH_SPHERE_SAMPLE_SEED).uniform(
        LIMITS.lower, LIMITS.upper, (20000, len(LIMITS.lower))
    )
    farthest = max(
        math.dist(kinematics.forward_kinematics(tuple(map(float, joints)))[0], sphere.centre)
        for joints in itertools.chain(grid, sample)
    )
    slack = sphere.radius_m - farthest
    record_property("reach_sphere_slack_m", slack)
    assert farthest <= sphere.radius_m
    assert slack <= REACH_SPHERE_MOST_SLACK_M, f"{slack} m past the farthest grasp point found"


def test_the_reach_sphere_is_centred_on_the_pan_joint_origin(solver):
    # The origin of shoulder_pan in the URDF, in base_link, the model's base
    # frame.
    assert solver.reach_sphere.centre == pytest.approx((0.0388353, -8.97657e-09, 0.0624))


@pytest.mark.parametrize("position_bar", [REACH_TOLERANCE, 0.05])
@pytest.mark.parametrize(
    "direction", [*AXIS_DIRECTIONS, (1.0, 1.0, 1.0), (-1.0, 2.0, -0.5)], ids=repr
)
def test_a_position_past_the_reach_sphere_and_the_bar_gets_no_descent_and_one_within_does(
    solver, monkeypatch, direction, position_bar
):
    sphere = solver.reach_sphere
    unit = np.array(direction) / np.linalg.norm(direction)
    past = np.array(sphere.centre) + (sphere.radius_m + position_bar + 1e-6) * unit
    within = np.array(sphere.centre) + (sphere.radius_m + position_bar - 1e-6) * unit
    bars = {"position_tolerance_m": position_bar, "angle_tolerance_rad": GRASP_ANGLE_TOLERANCE}
    assert sphere.rules_out(past, position_bar)
    assert not sphere.rules_out(within, position_bar)
    # A least distance is a distance the grasp point reaches, to the error of
    # the QP.
    assert solver.least_distance(past) > position_bar
    assert not seeds_of_an_approach_that_fails(solver, monkeypatch, past, DOWN, bars)
    assert seeds_of_an_approach_that_fails(solver, monkeypatch, within, DOWN, bars)


# An in-limit joint vector whose grasp point is 5.4 mm inside the reach
# sphere, about as far from its centre as the arm reaches.
FAR_REACHING_POSE = (-1.6804, 0.2674, -1.2885, -0.0881, 0.0449)


@pytest.mark.parametrize("position_bar", [REACH_TOLERANCE, 0.02, 0.05])
def test_a_position_past_the_reach_sphere_within_the_bar_of_a_grasp_point_is_approached(
    solver, kinematics, position_bar
):
    # The reach sphere rules out a position only past its radius plus the
    # bar: the arm puts the grasp point within the bar of these positions.
    assert LIMITS.contains(FAR_REACHING_POSE)
    point, approach_axis = grasp_frame(kinematics, FAR_REACHING_POSE)
    centre = np.array(solver.reach_sphere.centre)
    outward = (point - centre) / np.linalg.norm(point - centre)
    target = point + (position_bar - 0.001) * outward
    assert math.dist(target, centre) > solver.reach_sphere.radius_m
    joints = solver.approach(
        target,
        approach_axis,
        position_tolerance_m=position_bar,
        angle_tolerance_rad=GRASP_ANGLE_TOLERANCE,
    )
    assert joints is not None
    assert_verified(kinematics, joints, target, approach_axis, position_bar=position_bar)


def urdf_with(tmp_path, old, new):
    """A copy of the model's URDF, in `tmp_path`, with `old` replaced by
    `new`, beside a link to its collision meshes, which the copy names by
    paths relative to it."""
    urdf = Path(urdf_path())
    text = urdf.read_text()
    assert old in text
    path = tmp_path / "so101.urdf"
    path.write_text(text.replace(old, new))
    (tmp_path / "meshes").symlink_to(urdf.parent / "meshes")
    return str(path)


def test_a_joint_of_the_chain_that_can_move_its_child_link_origin_is_refused(tmp_path):
    path = urdf_with(
        tmp_path,
        '<joint name="elbow_flex" type="revolute">',
        '<joint name="elbow_flex" type="prismatic">',
    )
    with pytest.raises(ValueError, match="elbow_flex is prismatic"):
        ApproachSolver(path)


def test_a_grasp_frame_off_the_chain_of_the_pan_joint_is_refused(tmp_path):
    path = urdf_with(
        tmp_path,
        '<parent link="gripper_link"/>\n    <child link="gripper_frame_link"/>',
        '<parent link="base_link"/>\n    <child link="gripper_frame_link"/>',
    )
    with pytest.raises(ValueError, match="no chain of joints"):
        ApproachSolver(path)


@pytest.mark.parametrize(
    "position_bar",
    [0.0, -0.01, math.nan, math.inf, APPROACH_MAX_POSITION_BAR_M * 1.001],
    ids=repr,
)
def test_the_reach_sphere_refuses_a_position_bar_outside_its_range(solver, position_bar):
    with pytest.raises(ValueError):
        solver.reach_sphere.rules_out((0.25, 0.0, 0.04), position_bar)


def test_a_point_beyond_the_reach_is_not_approached(solver):
    assert solver.approach((1000.0, 0.0, 0.0), DOWN, **BARS) is None
    assert solver.approach((1.0, 0.0, 0.04), FORWARD, **BARS) is None


def test_the_largest_coordinate_is_answered(solver):
    far = solver.least_distance((APPROACH_MAX_COORDINATE_M, 0.0, 0.0))
    assert math.isfinite(far)
    assert far == pytest.approx(APPROACH_MAX_COORDINATE_M - 0.48, abs=0.01)


def test_an_answer_does_not_depend_on_what_was_asked_before():
    targets = [(0.25, 0.05, 0.04), (0.0, -0.3, -0.01), (0.6, 0.2, 0.1), (0.32, 0.0, 0.04)]

    def answers(order):
        solver = ApproachSolver(urdf_path())
        return {
            index: (
                solver.least_distance(targets[index]),
                solver.approach(targets[index], DOWN, **BARS),
                solver.approach(targets[index], FORWARD, **BARS),
            )
            for index in order
        }

    in_order = answers(range(len(targets)))
    reversed_order = answers(reversed(range(len(targets))))
    assert in_order == reversed_order


@pytest.mark.parametrize(
    "length", [1e-300, 1e-6, 1.0, 1e6, 1e300], ids=lambda length: f"length {length:g}"
)
def test_a_direction_of_any_length_is_the_same_direction(solver, length):
    target = (0.25, 0.05, 0.04)
    unit = solver.approach(target, DOWN, **BARS)
    assert unit is not None
    assert solver.approach(target, tuple(length * c for c in DOWN), **BARS) == unit


BAD_POSITIONS = [
    (math.nan, 0.0, 0.0),
    (0.0, math.inf, 0.0),
    (0.0, 0.0, -math.inf),
    (APPROACH_MAX_COORDINATE_M * 1.001, 0.0, 0.0),
    (0.2, 0.0),
    (0.2, 0.0, 0.1, 0.0),
]


@pytest.mark.parametrize("position", BAD_POSITIONS, ids=repr)
def test_a_position_that_is_not_three_usable_coordinates_is_refused(solver, position):
    with pytest.raises(ValueError):
        solver.least_distance(position)
    with pytest.raises(ValueError):
        solver.approach(position, DOWN, **BARS)
    with pytest.raises(ValueError):
        solver.reach_sphere.rules_out(position, REACH_TOLERANCE)


@pytest.mark.parametrize(
    "direction",
    [(0.0, 0.0, 0.0), (math.nan, 0.0, -1.0), (0.0, math.inf, 0.0), (0.0, -1.0)],
    ids=repr,
)
def test_a_direction_that_is_not_three_finite_components_with_a_length_is_refused(
    solver, direction
):
    with pytest.raises(ValueError):
        solver.approach((0.25, 0.0, 0.04), direction, **BARS)


@pytest.mark.parametrize(
    ("position_bar", "angle_bar"),
    [
        (0.0, GRASP_ANGLE_TOLERANCE),
        (-0.01, GRASP_ANGLE_TOLERANCE),
        (math.nan, GRASP_ANGLE_TOLERANCE),
        (math.inf, GRASP_ANGLE_TOLERANCE),
        (APPROACH_MAX_POSITION_BAR_M * 1.001, GRASP_ANGLE_TOLERANCE),
        (1e200, 1.0),
        (1e100, 1e-100),
        (1e300, 1e-300),
        (REACH_TOLERANCE, 0.0),
        (REACH_TOLERANCE, -0.01),
        (REACH_TOLERANCE, math.nan),
        (REACH_TOLERANCE, math.inf),
        (REACH_TOLERANCE, APPROACH_MIN_ANGLE_BAR_RAD * 0.999),
        (REACH_TOLERANCE, 1e-160),
        (1.0, 1e-160),
        (1e6, 1e-150),
        (REACH_TOLERANCE, math.pi * 1.001),
    ],
    ids=repr,
)
def test_a_bar_outside_its_range_is_refused(solver, position_bar, angle_bar):
    with pytest.raises(ValueError):
        solver.approach(
            (0.25, 0.0, 0.04),
            DOWN,
            position_tolerance_m=position_bar,
            angle_tolerance_rad=angle_bar,
        )


@pytest.mark.parametrize(
    ("position_bar", "angle_bar"),
    [
        (APPROACH_MAX_POSITION_BAR_M, APPROACH_MIN_ANGLE_BAR_RAD),
        (APPROACH_MAX_POSITION_BAR_M, math.pi),
        (5e-324, APPROACH_MIN_ANGLE_BAR_RAD),
        (5e-324, math.pi),
        (0.05, 0.001),
        (REACH_TOLERANCE, 1e-5),
        (REACH_TOLERANCE, 1e-6),
        (REACH_TOLERANCE, APPROACH_MIN_ANGLE_BAR_RAD),
    ],
    ids=repr,
)
def test_bars_inside_their_ranges_give_joints_inside_the_limits_or_none(
    solver, position_bar, angle_bar
):
    # At these bars the approach objective weighs up to about 1e33 times the
    # position objective. The QP takes each such trade with the heavier
    # objective at APPROACH_QP_MAX_WEIGHT; with the position objective at 1
    # instead, it fails from an approach weight of about 1e7.
    for target, direction in itertools.product(SPREAD_TARGETS, SPREAD_DIRECTIONS):
        joints = solver.approach(
            target, direction, position_tolerance_m=position_bar, angle_tolerance_rad=angle_bar
        )
        assert joints is None or LIMITS.contains(joints), (target, direction, joints)


def test_every_grasp_point_meets_the_widest_bars(solver):
    for target, direction in (((0.25, 0.0, 0.04), DOWN), ((0.2, 0.1, 0.1), FORWARD)):
        joints = solver.approach(
            target,
            direction,
            position_tolerance_m=APPROACH_MAX_POSITION_BAR_M,
            angle_tolerance_rad=math.pi,
        )
        assert joints is not None
