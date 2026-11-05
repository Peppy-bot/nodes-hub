//! The field cases of the OpenArm's Cartesian plan: the postures and moves
//! reported from the robot, planned on the bundled models at the launcher's
//! limits, as regressions.

use hegel::TestCase;
use hegel::generators as gs;
use srs_model::chain_kinematics::interpolate;
use srs_model::nalgebra::{Isometry3, Translation3, Unit, UnitQuaternion, Vector3};
use srs_model::{Arm, ELBOW, Elbow, JointVec, LineCursor, LinePlan, PlanLimits, plan_line};

mod helpers;

use helpers::fixtures::*;
use std::time::Duration;

/// The configuration a planned line walks to, from `start` to `end`.
fn walked_to_end(
    model: &Arm,
    line: &LinePlan,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
) -> JointVec {
    LineCursor::new(line.start_q, line.swivel)
        .advance(model, |s| interpolate(start, end, s), 1.0)
        .expect("the plan's line tracks to its end")
}

// The field repro no held elbow could track: swiveling the elbow at a constant
// rate along the pull tracks it as a straight line, landing on the goal at the
// request.
#[test]
fn the_cross_body_pull_is_a_line_with_the_elbow_swiveled() {
    let model = v2_right_arm();
    let (start, end, seed) = cross_body_pull(&model);
    let line = plan_line(
        &model,
        &end,
        seed,
        &v2_limits(),
        Duration::from_secs_f64(10.0),
    )
    .expect("the pull tracks as a line");
    let duration_s = line.duration.as_secs_f64();
    assert_eq!(line.elbow, Elbow::Manipulability);
    assert!(
        (duration_s - 10.0).abs() < EPS,
        "planned at {duration_s:.3}s"
    );
    assert_lands_on(&model, &walked_to_end(&model, &line, &start, &end), &end);
}

// The user-reported pull: x back to -0.2 from Ready, which a stream carries out
// live, fired as an action.
#[test]
fn pull_from_ready_to_x_minus_02_plans_a_line() {
    let model = v2_right_arm();
    // The recorded pull is a tip coordinate, so -0.2 is placed there and carried
    // to the tool: the same physical motion, in the frame commanded.
    let mut end_tip = tip_world(&model, &READY);
    end_tip.translation.vector.x = -0.2;
    let end = end_tip * model.tool();
    let line = plan_line(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
    )
    .expect("the pull tracks as a line");
    let start = model.world_ee_pose(&READY);
    assert_lands_on(&model, &walked_to_end(&model, &line, &start, &end), &end);
}

// After a swiveled line parks the arm at an unusual arm angle, ordinary nudges
// from that posture plan as quiet held-elbow lines, the elbow tried first.
#[test]
fn small_nudge_after_a_swiveled_line_stays_a_quiet_line() {
    let model = v2_right_arm();
    let (start, end, seed) = cross_body_pull(&model);
    let pull = plan_line(&model, &end, seed, &v2_limits(), Duration::ZERO)
        .expect("the pull tracks as a line");
    let parked = walked_to_end(&model, &pull, &start, &end);
    let mut nudge_end = model.world_ee_pose(&parked);
    nudge_end.translation.vector.x += 0.03;
    let line = plan_line(
        &model,
        &nudge_end,
        parked,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
    )
    .expect("an ordinary nudge must stay a line");
    let duration_s = line.duration.as_secs_f64();
    assert_eq!(
        line.elbow,
        Elbow::Held,
        "an ordinary nudge must hold the elbow"
    );
    assert!(
        (duration_s - 2.0).abs() < EPS,
        "nudge stays at the request, got {duration_s:.3}s"
    );
}

// An easy short move plans as a line at exactly the requested duration with the
// elbow held: well-conditioned paths spend no swivel.
#[test]
fn easy_move_plans_the_line_at_the_requested_duration() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.translation.vector.z += 0.05;
    let line = plan_line(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
    )
    .expect("small lift from ready is planned");
    let duration_s = line.duration.as_secs_f64();
    assert!(
        (duration_s - 2.0).abs() < EPS,
        "easy move must stay at the request, got {duration_s:.3}s"
    );
    assert_eq!(
        line.elbow,
        Elbow::Held,
        "an easy move must not swivel the elbow"
    );
}

// The 25 cm lift from this field-recorded grasp posture of the v2 left arm: the
// held elbow carried joint 6 onto its limit a third of the way up, where the line
// turned a corner the arm snapped through. The swivel toward the widest limit
// margin tracks the line at the request with every joint off its limits the whole
// way up.
#[test]
fn the_field_lift_swivels_the_elbow_off_the_wrist_limit() {
    let model = v2_left_arm();
    let seed = READY_LEFT;
    let start = model.world_pose(&model.at(&seed).ee_pose());
    let mut end = start;
    end.translation.vector.z += 0.25;
    let line = plan_line(
        &model,
        &end,
        seed,
        &v2_limits(),
        Duration::from_secs_f64(10.0),
    )
    .expect("the lift must stay a line");
    let duration_s = line.duration.as_secs_f64();
    assert_eq!(
        line.elbow,
        Elbow::LimitMargin,
        "the held elbow pins the wrist"
    );
    assert!(
        (duration_s - 10.0).abs() < EPS,
        "the lift must stay at the request, got {duration_s:.3}s"
    );
    let limits = model.limits();
    let mut cursor = LineCursor::new(line.start_q, line.swivel);
    for k in 1..=100 {
        let q = cursor
            .advance(
                &model,
                |s| interpolate(&start, &end, s),
                f64::from(k) / 100.0,
            )
            .expect("the plan's line tracks");
        for (i, (&v, limit)) in q.iter().zip(&limits).enumerate() {
            assert!(
                v > limit.lo && v < limit.hi,
                "joint {i} on a limit {} of the way up",
                f64::from(k) / 100.0
            );
        }
    }
}

// A turn in place about the grasp point has a line too: the wrist orbits the
// point, the end effector's progress runs in radians, and the line lands on the
// orientation. From Ready the elbow reaches its singularity floor beyond ~0.8
// rad, so this turns as far as that posture allows.
#[test]
fn a_pure_reorientation_plans_a_line() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.7) * end.rotation;
    let line = plan_line(&model, &end, READY, &v2_limits(), Duration::ZERO)
        .expect("a 0.7 rad turn from Ready tracks");
    assert!(line.duration > Duration::ZERO);
    assert_lands_on(&model, &walked_to_end(&model, &line, &start, &end), &end);
}

#[test]
fn an_unreachable_goal_has_no_line() {
    let arm = v1_left_arm();
    let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
    let mut unreachable = arm.world_pose(&arm.at(&seed).ee_pose());
    unreachable.translation.vector.x += 10.0;
    assert_eq!(
        plan_line(&arm, &unreachable, seed, &v1_limits(), Duration::ZERO),
        None,
        "10 m away is out of reach"
    );
}

/// A unit vector.
fn drawn_unit(tc: &TestCase) -> Unit<Vector3<f64>> {
    let drawn: [f64; 3] = tc.draw(gs::arrays(gs::floats().min_value(-1.0).max_value(1.0)));
    let drawn = Vector3::from(drawn);
    tc.assume(drawn.norm() > 1e-3);
    Unit::new_normalize(drawn)
}

/// A v2 arm, a posture anywhere in its limits or beside its wrist or elbow singular
/// surface or at Ready, and a pose up to 50 cm and 0.8 rad from where that posture
/// puts the grasp point.
fn drawn_move(tc: &TestCase) -> (Arm, JointVec, Isometry3<f64>) {
    let right: bool = tc.draw(gs::booleans());
    let model = if right { v2_right_arm() } else { v2_left_arm() };
    let limits = model.limits();
    let mut seed: JointVec = std::array::from_fn(|i| {
        tc.draw(gs::floats().min_value(limits[i].lo).max_value(limits[i].hi))
    });
    match tc.draw(gs::integers::<u8>().max_value(3)) {
        1 => seed[WRIST_BEND] = tc.draw(gs::floats().min_value(-0.05).max_value(0.05)),
        2 => {
            let floor = limits[ELBOW].lo;
            seed[ELBOW] = tc.draw(gs::floats().min_value(floor).max_value(floor + 0.2));
        }
        3 => seed = if right { READY } else { READY_LEFT },
        _ => {}
    }
    let (direction, axis) = (drawn_unit(tc), drawn_unit(tc));
    let travel: f64 = tc.draw(gs::floats().min_value(0.0).max_value(0.5));
    let turn: f64 = tc.draw(gs::floats().min_value(0.0).max_value(0.8));
    let start = model.world_ee_pose(&seed);
    let end = Isometry3::from_parts(
        Translation3::from(start.translation.vector + direction.into_inner() * travel),
        UnitQuaternion::from_axis_angle(&axis, turn) * start.rotation,
    );
    (model, seed, end)
}

// The acceleration limit sets how long a line takes and nothing else: at ten times
// the launcher's limit a move plans the same kind of line, until the launcher's limit
// runs that line past the longest a move may take.
#[hegel::test(test_cases = 100)]
fn the_acceleration_limit_changes_the_kind_of_line_only_past_the_longest_move(tc: TestCase) {
    let (model, seed, end) = drawn_move(&tc);
    let plan = |acceleration: JointVec, max_duration: Duration| {
        let limits = PlanLimits {
            max_joint_acceleration_rad_s2: acceleration,
            max_duration,
            ..v2_limits()
        };
        plan_line(&model, &end, seed, &limits, Duration::ZERO)
    };
    let (gentle, brisk) = (
        plan(A_MAX, LONGEST_MOVE),
        plan(A_MAX.map(|a| a * 10.0), LONGEST_MOVE),
    );
    let elbow = |line: &Option<LinePlan>| line.as_ref().map(|l| l.elbow);
    if elbow(&gentle) == elbow(&brisk) {
        return;
    }
    let unbounded = plan(A_MAX, 100 * LONGEST_MOVE);
    assert_eq!(elbow(&unbounded), elbow(&brisk), "{seed:?} to {end:?}");
    let line = unbounded.expect("the brisk line tracks");
    assert!(line.duration > LONGEST_MOVE, "{seed:?} to {end:?}");
}
