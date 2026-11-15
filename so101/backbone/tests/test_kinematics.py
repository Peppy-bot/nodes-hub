"""Real placo FK/IK against the mini arm, a chain of the SO-101's shape with
placeholder geometry and the so101 description's names: the solver wiring, unit
conversions, and the verified-solution gate."""

import math
from pathlib import Path
from string import Template

import numpy as np
import pytest
from conftest import WIDE_LIMITS
from so101.transforms import relative_rotation_rad

from so101_backbone.kinematics import (
    LINE_ORIENTATION_BAR_RAD,
    LINE_POSITION_BAR_M,
    LINE_SAMPLES,
    MAX_LINE_STEP_RAD,
    Kinematics,
)
from so101_backbone.limits import JointLimits
from so101_backbone.model import ARM, END_EFFECTOR_FRAME, urdf_path

MINI_ARM = Path(__file__).parent / "assets" / "mini_arm.urdf.template"
HOME = (0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def mini_urdf(tmp_path_factory) -> str:
    """The mini arm's URDF, named as the so101 description names its arm's
    joints and its end-effector frame."""
    joints = {f"joint_{index}": name for index, name in enumerate(ARM.joints)}
    urdf = Template(MINI_ARM.read_text()).substitute(joints, end_effector=END_EFFECTOR_FRAME)
    path = tmp_path_factory.mktemp("mini_arm") / "mini_arm.urdf"
    path.write_text(urdf)
    return str(path)


@pytest.fixture(scope="module")
def kinematics(mini_urdf):
    return Kinematics(mini_urdf)


def test_fk_at_home_matches_link_stack(kinematics):
    position, orientation = kinematics.forward_kinematics(HOME)
    # The chain is a straight stack of z offsets at home.
    assert position == pytest.approx((0.0, 0.0, 0.05 + 0.03 + 0.11 + 0.10 + 0.05 + 0.08))
    assert orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_fk_varies_with_each_positioning_joint(kinematics):
    # Evaluated at a bent posture: at home the chain is coaxial, so the two
    # z-axis joints (shoulder_pan, wrist_roll) would spin the EE in place.
    bent_home = (0.2, -0.4, 0.6, 0.3, 0.0)
    reference, _ = kinematics.forward_kinematics(bent_home)
    for j in range(4):  # wrist_roll only reorients the EE frame on this chain
        nudged = tuple(q + (0.3 if i == j else 0.0) for i, q in enumerate(bent_home))
        position, _ = kinematics.forward_kinematics(nudged)
        assert not all(
            math.isclose(a, b, abs_tol=1e-9) for a, b in zip(position, reference, strict=True)
        ), f"joint {j} did not move the end effector"


def test_ik_reaches_a_known_fk_pose(kinematics):
    target_joints = (0.3, -0.4, 0.6, 0.2, 0.0)
    position, orientation = kinematics.forward_kinematics(target_joints)
    solution = kinematics.inverse_kinematics(HOME, position, orientation, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi)
    assert solution is not None
    reached, _ = kinematics.forward_kinematics(solution)
    assert math.dist(reached, position) <= 0.01


def test_unreachable_pose_is_refused(kinematics):
    # Twice the arm's full extension.
    solution = kinematics.inverse_kinematics(
        HOME, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), position_tolerance_m=0.01, orientation_tolerance_rad=math.pi
    )
    assert solution is None


def test_streaming_solver_steps_toward_a_reachable_pose(kinematics):
    target_joints = (0.3, -0.4, 0.6, 0.2, 0.0)
    position, orientation = kinematics.forward_kinematics(target_joints)
    joints = HOME
    for _ in range(60):  # a second of 60 Hz streaming
        joints = kinematics.inverse_kinematics_streaming(joints, position, orientation)
        assert joints is not None
    reached, _ = kinematics.forward_kinematics(joints)
    assert math.dist(reached, position) <= 0.02


def test_streaming_solver_tracks_the_workspace_boundary(kinematics):
    # Far out of reach: the stream must keep returning a finite best-effort
    # configuration (boundary tracking), never None, never non-finite.
    joints = HOME
    for _ in range(30):
        joints = kinematics.inverse_kinematics_streaming(
            joints, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)
        )
        assert joints is not None
        assert all(math.isfinite(j) for j in joints)


def test_ik_accepts_the_best_orientation_for_an_underactuated_pose(kinematics):
    # Five joints underactuate the three orientation degrees of freedom, so
    # acceptance is positional: a reachable position paired with an
    # orientation the chain cannot meet still solves, landing the position
    # and taking the solver's best orientation.
    target_joints = (0.3, -0.5, 0.7, 0.2, 0.0)
    position, reachable_orientation = kinematics.forward_kinematics(target_joints)
    impossible_orientation = (0.0, 0.0, 0.0, 1.0)
    assert reachable_orientation != pytest.approx(impossible_orientation)
    solution = kinematics.inverse_kinematics(HOME, position, impossible_orientation, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi)
    assert solution is not None
    reached, _ = kinematics.forward_kinematics(solution)
    assert math.dist(reached, position) <= 0.01


def rotated(orientation, axis, degrees):
    """`orientation` turned about one of its own axes."""
    half = math.radians(degrees) / 2.0
    unit = np.array(axis, float) / np.linalg.norm(axis)
    d = np.array([*(unit * math.sin(half)), math.cos(half)])
    x1, y1, z1, w1 = orientation
    x2, y2, z2, w2 = d
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def test_an_impossible_orientation_does_not_cost_the_position(kinematics):
    # Five joints cannot turn the grasp point about its own x axis, and both
    # pose objectives are soft, so the solver will trade position away to
    # chase an orientation it can never reach. A move asks first for a place
    # to be: the position it was given must survive the attempt.
    seed = (0.0, 0.3, 0.4, 0.2, 0.0)
    position, orientation = kinematics.forward_kinematics(seed)
    solution = kinematics.inverse_kinematics(
        seed, position, rotated(orientation, (1, 0, 0), 20), position_tolerance_m=0.01, orientation_tolerance_rad=math.pi
    )
    assert solution is not None
    reached, _ = kinematics.forward_kinematics(solution)
    assert math.dist(reached, position) < 0.001


def test_an_impossible_orientation_does_not_refuse_a_reachable_position(kinematics):
    # The position here is reachable by construction. Weighted hard enough,
    # the orientation objective drags the solve outside the position
    # tolerance and the caller is told the pose is unreachable, naming the
    # wrong cause.
    seed = (0.0, 0.3, 0.4, 0.2, 0.0)
    position, orientation = kinematics.forward_kinematics(seed)
    assert kinematics.inverse_kinematics(
        seed, position, rotated(orientation, (1, 0, 0), 90), position_tolerance_m=0.01, orientation_tolerance_rad=math.pi
    ) is not None


def test_a_reachable_orientation_is_still_honoured(kinematics):
    # The orientation weight is small, not absent: dropping it to zero would
    # abandon the wrist orientations this arm can actually hold.
    seed = (0.0, 0.3, 0.4, 0.2, 0.0)
    position, orientation = kinematics.forward_kinematics(seed)
    target = rotated(orientation, (0, 0, 1), 20)
    solution = kinematics.inverse_kinematics(seed, position, target, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi)
    assert solution is not None
    _, reached = kinematics.forward_kinematics(solution)
    assert math.degrees(relative_rotation_rad(reached, target)) < 1.0


def test_the_deployed_arm_still_reaches_an_orientation_it_can_hold():
    # Against the shipped model: the seed already satisfies
    # the requested position, so a solve that stopped at the first step
    # inside the position tolerance would return before the orientation
    # objective had moved the wrist at all.
    kinematics = Kinematics(urdf_path())
    seed = (0.0, 0.3, 0.4, 0.2, 0.0)
    position, orientation = kinematics.forward_kinematics(seed)
    target = rotated(orientation, (0, 0, 1), 20)
    solution = kinematics.inverse_kinematics(seed, position, target, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi)
    assert solution is not None
    reached_position, reached_orientation = kinematics.forward_kinematics(solution)
    assert math.degrees(relative_rotation_rad(reached_orientation, target)) < 1.0
    assert math.dist(reached_position, position) < 0.001


def test_a_caller_can_tighten_the_position_bar(kinematics):
    # A bar far below what one linearised step converges to inside the
    # iteration budget refuses.
    target = kinematics.forward_kinematics((0.4, 0.3, -0.5, 0.2, 0.1))
    assert kinematics.inverse_kinematics(HOME, *target, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi) is not None
    assert (
        kinematics.inverse_kinematics(
            HOME, *target, position_tolerance_m=1e-12, orientation_tolerance_rad=math.pi
        )
        is None
    )


def test_an_orientation_bar_refuses_what_the_wrist_cannot_reach(kinematics):
    # Five joints underactuate three rotational degrees of freedom, so some
    # reachable positions carry an unreachable orientation. The case is found
    # rather than pinned: the descent is mildly path dependent, so a hardcoded
    # pose can stop qualifying when the tests around it change.
    reachable_but_turned = None
    for seed_scale in range(1, 40):
        angle = seed_scale * 0.13
        position = kinematics.forward_kinematics(
            (0.2, 0.3 - angle / 8, angle / 2, 0.1, 0.0)
        )[0]
        turned = (
            math.sin(angle),
            math.cos(angle) * math.sin(angle),
            math.cos(angle),
            math.sin(angle / 2),
        )
        norm = math.sqrt(sum(c * c for c in turned))
        turned = tuple(c / norm for c in turned)
        solution = kinematics.inverse_kinematics(HOME, position, turned, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi)
        if solution is None:
            continue
        missed = relative_rotation_rad(
            kinematics.forward_kinematics(solution)[1], turned
        )
        if missed > 0.1:
            reachable_but_turned = (position, turned, missed)
            break
    assert reachable_but_turned is not None, "no position-reachable orientation miss found"
    position, turned, missed = reachable_but_turned
    assert (
        kinematics.inverse_kinematics(
            HOME, position, turned, position_tolerance_m=0.01, orientation_tolerance_rad=missed / 2
        )
        is None
    )


def test_a_generous_orientation_bar_still_accepts(kinematics):
    target = kinematics.forward_kinematics((0.3, 0.2, -0.4, 0.1, 0.2))
    assert (
        kinematics.inverse_kinematics(
            HOME, *target, position_tolerance_m=0.01, orientation_tolerance_rad=math.pi
        )
        is not None
    )


def test_the_jacobian_predicts_end_effector_motion(kinematics):
    # The property a speed cap depends on: J @ dq is the twist a joint step
    # produces. Checked against a finite difference small enough that the
    # second-order term is negligible.
    import numpy as np

    q = (0.3, -0.4, 0.6, 0.2, 0.5)
    direction = np.array([1.0, -0.7, 0.4, 0.2, -0.9])
    direction /= np.linalg.norm(direction)
    step = direction * 1e-6
    jacobian = kinematics.jacobian(q)
    assert jacobian.shape == (6, 5)

    predicted = np.linalg.norm((jacobian @ step)[:3])
    here = np.array(kinematics.forward_kinematics(q)[0])
    there = np.array(kinematics.forward_kinematics(tuple(np.array(q) + step))[0])
    assert predicted == pytest.approx(np.linalg.norm(there - here), rel=1e-3)


def test_the_jacobian_is_linear_in_the_step(kinematics):
    # Why the governor uses it rather than differencing forward kinematics:
    # halving the joint step must halve the end-effector speed exactly, so a
    # cap can be met by one division instead of a search.
    import numpy as np

    q = (0.1, -0.6, 0.9, 0.3, -0.2)
    jacobian = kinematics.jacobian(q)
    step = np.array([0.4, -0.3, 0.2, 0.1, -0.5])
    full = np.linalg.norm((jacobian @ step)[:3])
    for scale in (0.5, 0.25, 0.1):
        scaled = np.linalg.norm((jacobian @ (step * scale))[:3])
        assert scaled == pytest.approx(scale * full, rel=1e-12)


def test_the_jacobian_is_taken_at_the_tool_point(kinematics):
    # The trap this closes: placo's world Jacobian refers the twist to the
    # world origin, and its linear rows are not the tool's velocity. A pure
    # shoulder rotation moves the tool, so the linear part cannot vanish.
    import numpy as np

    q = (0.0, -0.5, 0.8, 0.3, 0.0)
    jacobian = kinematics.jacobian(q)
    pan_only = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    here = np.array(kinematics.forward_kinematics(q)[0])
    radius = math.hypot(here[0], here[1])
    assert radius > 0.05, "pick a posture where the tool is off the pan axis"
    assert np.linalg.norm((jacobian @ pan_only)[:3]) == pytest.approx(radius, rel=1e-3)


def test_asking_for_the_jacobian_does_not_disturb_a_streaming_solve(mini_urdf):
    # placo's solver carries state between solves: a query that moved the
    # model and left it moved diverges an interleaved streaming solve by
    # radians.
    import numpy as np

    def run(interleave: bool):
        kinematics = Kinematics(mini_urdf)
        seed, out = HOME, []
        for step in range(8):
            target = (0.2 + step * 0.05, -0.3, 0.5, 0.1, 0.0)
            position, orientation = kinematics.forward_kinematics(target)
            if interleave:
                kinematics.jacobian((0.9, -0.2, 0.4, 0.1, 0.3))
            seed = kinematics.inverse_kinematics_streaming(seed, position, orientation)
            out.append(seed)
        return out

    for undisturbed, interleaved in zip(run(False), run(True), strict=True):
        assert np.allclose(undisturbed, interleaved, atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.01, math.nan, math.inf])
def test_an_unusable_tolerance_is_refused(kinematics, bad):
    target = kinematics.forward_kinematics((0.1, 0.1, -0.2, 0.0, 0.0))
    with pytest.raises(ValueError):
        kinematics.inverse_kinematics(
            HOME, *target, position_tolerance_m=bad, orientation_tolerance_rad=math.pi
        )
    with pytest.raises(ValueError):
        kinematics.inverse_kinematics(
            HOME, *target, position_tolerance_m=0.01, orientation_tolerance_rad=bad
        )


BENT = (0.3, -0.4, 0.5, -0.3, 0.2)
OTHER_BENT = (0.4, -0.3, 0.4, -0.2, 0.1)


def off_the_segment(point, start, end):
    along = np.subtract(end, start)
    t = np.clip(np.dot(np.subtract(point, start), along) / np.dot(along, along), 0.0, 1.0)
    return float(np.linalg.norm(np.subtract(point, np.add(start, along * t))))


def test_a_walk_keeps_the_grasp_point_on_the_line_and_lands_on_the_pose(kinematics):
    start, _ = kinematics.forward_kinematics(BENT)
    position, orientation = kinematics.forward_kinematics(OTHER_BENT)
    walk = kinematics.walk_line(BENT, position, orientation, WIDE_LIMITS)
    assert walk is not None and len(walk) == LINE_SAMPLES + 1
    assert walk[0] == BENT
    for q in walk:
        assert off_the_segment(kinematics.forward_kinematics(q)[0], start, position) <= LINE_POSITION_BAR_M
    for a, b in zip(walk, walk[1:], strict=False):
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) <= MAX_LINE_STEP_RAD
    landed, turned = kinematics.forward_kinematics(walk[-1])
    assert math.dist(landed, position) <= LINE_POSITION_BAR_M
    assert relative_rotation_rad(turned, orientation) <= LINE_ORIENTATION_BAR_RAD


def test_a_position_out_of_reach_has_no_walk(kinematics):
    assert kinematics.walk_line(BENT, (5.0, 0.0, 0.0), None, WIDE_LIMITS) is None


def test_an_orientation_the_arm_cannot_end_on_has_no_walk(kinematics):
    position, _ = kinematics.forward_kinematics(OTHER_BENT)
    assert kinematics.walk_line(BENT, position, (1.0, 0.0, 0.0, 0.0), WIDE_LIMITS) is None


def test_an_orientation_a_little_past_the_bar_has_no_walk(kinematics):
    # The pose's own orientation walks; turned 0.05 rad about the world's x axis, the
    # end stands past the 0.01 rad bar, and the arm's joints cannot take it up.
    position, orientation = kinematics.forward_kinematics(OTHER_BENT)
    s, c = math.sin(0.025), math.cos(0.025)
    x, y, z, w = orientation
    turned = (c * x + s * w, c * y - s * z, c * z + s * y, c * w - s * x)
    assert kinematics.walk_line(BENT, position, turned, WIDE_LIMITS) is None


def test_a_position_alone_ends_near_the_turn_the_line_starts_with(kinematics):
    _, start_orientation = kinematics.forward_kinematics(BENT)
    position, _ = kinematics.forward_kinematics(OTHER_BENT)
    walk = kinematics.walk_line(BENT, position, None, WIDE_LIMITS)
    assert walk is not None
    assert relative_rotation_rad(kinematics.forward_kinematics(walk[-1])[1], start_orientation) < 1.0


def test_a_solution_past_the_limits_has_no_walk(kinematics):
    position, orientation = kinematics.forward_kinematics(OTHER_BENT)
    narrow = JointLimits(lower=WIDE_LIMITS.lower, upper=(0.31, *WIDE_LIMITS.upper[1:]))
    assert kinematics.walk_line(BENT, position, orientation, narrow) is None
