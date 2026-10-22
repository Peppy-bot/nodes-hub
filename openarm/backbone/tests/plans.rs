//! The field cases of the OpenArm's Cartesian plan: the postures and moves
//! reported from the robot, planned on the bundled models at the launcher's
//! limits, as regressions.

use srs_model::chain_kinematics::{MAX_SERVO_S, ServoTolerances, interpolate};
use srs_model::nalgebra::{Isometry3, UnitQuaternion, Vector3};
use srs_model::{Arm, CartesianPlan, Elbow, LineCursor, LinePlan, PlanRejection, plan_cartesian};

mod helpers;

use helpers::fixtures::*;
use std::time::Duration;

// The field repro that no held or steered elbow could track and the servo
// had to take: swiveling the elbow at a constant rate along the pull tracks
// it as a straight line, landing on the goal at the request.
#[test]
fn the_cross_body_pull_is_a_line_with_the_elbow_swiveled() {
    let model = v2_right_arm();
    let (start, end, seed) = cross_body_pull(&model);
    let plan = plan_cartesian(
        &model,
        &end,
        seed,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
        Ok(default_tolerance()),
    )
    .expect("the pull is planned");
    let CartesianPlan::Line(LinePlan {
        duration,
        elbow,
        swivel,
        start_q,
        ..
    }) = plan
    else {
        panic!("the pull tracks as a line");
    };
    let duration_s = duration.as_secs_f64();
    assert_eq!(elbow, Elbow::Manipulability);
    assert!(
        (duration_s - 2.0).abs() < EPS,
        "planned at {duration_s:.3}s"
    );
    let mut cursor = LineCursor::new(start_q, swivel);
    let landed = cursor
        .advance(&model, |s| interpolate(&start, &end, s), 1.0)
        .expect("the line tracks to its end");
    assert_lands_within(&model, &landed, &end, default_tolerance());
}

// The user-reported gap made a regression test: pulling x back to -0.2 from
// Ready works in streaming, so the same intent fired as an action reaches the
// pose too, via the servo when no line tracks.
#[test]
fn pull_from_ready_to_x_minus_02_reaches_via_servo() {
    let model = v2_right_arm();
    // The recorded pull is a tip coordinate, so -0.2 is placed there and carried
    // to the tool: the same physical motion, in the frame commanded.
    let mut end_tip = tip_world(&model, &READY);
    end_tip.translation.vector.x = -0.2;
    let end = end_tip * model.tool();
    let plan = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
        Ok(default_tolerance()),
    )
    .expect("the pull must be reachable, as streaming proves live");
    if matches!(plan, CartesianPlan::Servo(_)) {
        let (_, _, last) = played_servo(plan);
        assert_lands_within(&model, &last, &end, default_tolerance());
    }
}

// The incident regression: after a servo move parks the arm at an unusual arm
// angle, ordinary nudges from that posture plan as quiet held-elbow lines, the
// tier tried first.
#[test]
fn small_nudge_after_a_servo_move_stays_a_quiet_line() {
    let model = v2_right_arm();
    let (start, end, seed) = cross_body_pull(&model);
    let parked = roll_servo_to_convergence(&model, &start, &end, seed, default_tolerance());
    let nudge_start = model.world_pose(&model.at(&parked).ee_pose());
    let mut nudge_end = nudge_start;
    nudge_end.translation.vector.x += 0.03;
    let plan = plan_cartesian(
        &model,
        &nudge_end,
        parked,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
        Ok(default_tolerance()),
    )
    .expect("nudge from the parked posture is reachable");
    let CartesianPlan::Line(LinePlan {
        duration, elbow, ..
    }) = plan
    else {
        panic!("an ordinary nudge must stay a line");
    };
    let duration_s = duration.as_secs_f64();
    assert_eq!(elbow, Elbow::Held, "an ordinary nudge must hold the elbow");
    assert!(
        (duration_s - 2.0).abs() < EPS,
        "nudge stays at the request, got {duration_s:.3}s"
    );
}

// An easy short move plans as a line at exactly the requested duration with
// the elbow held: well-conditioned paths spend no swivel and never degrade to
// the servo.
#[test]
fn easy_move_plans_the_line_at_the_requested_duration() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.translation.vector.z += 0.05;
    let plan = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
        Ok(default_tolerance()),
    )
    .expect("small lift from ready is planned");
    let CartesianPlan::Line(LinePlan {
        duration, elbow, ..
    }) = plan
    else {
        panic!("a trackable line must plan as a line");
    };
    let duration_s = duration.as_secs_f64();
    assert!(
        (duration_s - 2.0).abs() < EPS,
        "easy move must stay at the request, got {duration_s:.3}s"
    );
    assert_eq!(elbow, Elbow::Held, "an easy move must not swivel the elbow");
}

// The 25 cm lift from this field-recorded grasp posture of the v2 left arm:
// the held elbow carried joint 6 onto its limit a third of the way up, where
// the line turned a corner the arm snapped through. The swivel toward the
// widest limit margin tracks the line at the request with every joint off its
// limits the whole way up.
#[test]
fn the_field_lift_swivels_the_elbow_off_the_wrist_limit() {
    let model = v2_left_arm();
    let seed = [-0.1537, -0.39547, 0.4808, 0.95, 0.0008, -0.0046, 0.0008];
    let start = model.world_pose(&model.at(&seed).ee_pose());
    let mut end = start;
    end.translation.vector.z += 0.25;
    let plan = plan_cartesian(
        &model,
        &end,
        seed,
        &v2_limits(),
        Duration::from_secs_f64(5.0),
        Ok(default_tolerance()),
    )
    .expect("the lift is planned");
    let CartesianPlan::Line(LinePlan {
        duration,
        elbow,
        swivel,
        start_q,
        ..
    }) = plan
    else {
        panic!("the lift must stay a line");
    };
    let duration_s = duration.as_secs_f64();
    assert_eq!(elbow, Elbow::LimitMargin, "the held elbow pins the wrist");
    assert!(
        (duration_s - 5.0).abs() < EPS,
        "the lift must stay at the request, got {duration_s:.3}s"
    );
    let limits = model.limits();
    let mut cursor = LineCursor::new(start_q, swivel);
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

#[test]
fn plan_cartesian_rejects_an_unreachable_goal() {
    let arm = v1_left_arm();
    let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
    let start = arm.world_pose(&arm.at(&seed).ee_pose());
    let mut unreachable = start;
    unreachable.translation.vector.x += 10.0;
    assert_eq!(
        plan_cartesian(
            &arm,
            &unreachable,
            seed,
            &v1_limits(),
            Duration::from_secs_f64(0.0),
            Ok(default_tolerance())
        ),
        Err(PlanRejection::Unreachable),
        "10 m away is out of reach, not merely slow or too tightly toleranced"
    );
}

/// A pose the sub-millimeter default refuses and a 5 mm / 5 deg request
/// admits, found by sweeping the workspace around Ready: the servo rollout
/// stalls against the tight bar and converges against the loose one.
/// Offsets are from the Ready grasp point, in world meters.
const TOLERANCE_SENSITIVE_OFFSET: [f64; 3] = [-0.32, -0.16, 0.08];

fn tolerance_sensitive_goal(model: &Arm) -> (Isometry3<f64>, Isometry3<f64>) {
    let start = ready_pose(model);
    let mut end = start;
    end.translation.vector.x += TOLERANCE_SENSITIVE_OFFSET[0];
    end.translation.vector.y += TOLERANCE_SENSITIVE_OFFSET[1];
    end.translation.vector.z += TOLERANCE_SENSITIVE_OFFSET[2];
    (start, end)
}

/// The move the servo tier plans: the tolerance-sensitive goal at the panel's
/// slack, from Ready.
fn servo_case(model: &Arm) -> (Isometry3<f64>, Isometry3<f64>, ServoTolerances) {
    let (start, end) = tolerance_sensitive_goal(model);
    (start, end, panel_tolerance())
}

// The caller's slack decides admission: the same goal, arm and seed, refused
// at the default bar and planned at the panel's.
#[test]
fn a_loose_tolerance_admits_a_goal_the_default_refuses() {
    let model = v2_right_arm();
    let (_, end) = tolerance_sensitive_goal(&model);
    assert!(
        plan_cartesian(
            &model,
            &end,
            READY,
            &v2_limits(),
            Duration::from_secs_f64(2.0),
            Ok(default_tolerance())
        )
        .is_err(),
        "this goal is chosen because the default tolerance refuses it"
    );
    let plan = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
        Ok(panel_tolerance()),
    )
    .expect("the panel's slack must admit it");
    played_servo(plan);
}

// A goal admitted at a tolerance is planned to within that tolerance.
#[test]
fn an_admitted_goal_is_planned_within_the_tolerance_it_was_admitted_at() {
    let model = v2_right_arm();
    let (_, end) = tolerance_sensitive_goal(&model);
    let tolerance = panel_tolerance();
    let plan = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(2.0),
        Ok(tolerance),
    )
    .expect("admitted at the panel's slack");
    let (_, _, landed) = played_servo(plan);
    assert_lands_within(&model, &landed, &end, tolerance);
}

// A servo move takes the time asked for, as a line does: requested at several
// times what its path needs from rest to rest, the move is planned at exactly
// the request and plays the path the rollout took.
#[test]
fn a_servo_guided_move_takes_the_requested_duration() {
    let model = v2_right_arm();
    let (_, end, tolerance) = servo_case(&model);
    let as_fast_as_it_can = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(0.0),
        Ok(tolerance),
    )
    .expect("the move converges");
    let path_s = played_servo(as_fast_as_it_can).0;
    for requested in [2.0 * path_s, 10.0 * path_s] {
        let plan = plan_cartesian(
            &model,
            &end,
            READY,
            &v2_limits(),
            Duration::from_secs_f64(requested),
            Ok(tolerance),
        )
        .expect("servo reaches the pose");
        let duration_s = played_servo(plan).0;
        assert!(
            (duration_s - requested).abs() < EPS,
            "asked for {requested:.2}s, the move was planned at {duration_s:.2}s"
        );
    }
}

// As fast as the limits allow, a request of 0, takes exactly what the path
// needs from rest to rest.
#[test]
fn a_servo_guided_move_asked_for_no_duration_takes_what_its_path_needs() {
    let model = v2_right_arm();
    let (_, end, tolerance) = servo_case(&model);
    let plan = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(0.0),
        Ok(tolerance),
    )
    .expect("servo reaches the pose");
    let CartesianPlan::Servo(servo) = &plan else {
        panic!("the move runs the servo");
    };
    let path_s = servo.path.duration_s();
    assert!(
        (servo.duration.as_secs_f64() - path_s).abs() < EPS,
        "a {}s plan for a {path_s}s path",
        servo.duration.as_secs_f64()
    );
}

// The ceiling bounds the law's convergence, not the move, so a servo move
// asked to take longer than the ceiling itself is admitted at its request.
#[test]
fn a_servo_guided_move_requested_past_the_ceiling_is_admitted() {
    let model = v2_right_arm();
    let (_, end, tolerance) = servo_case(&model);
    let requested = MAX_SERVO_S + 15.0;
    let plan = plan_cartesian(
        &model,
        &end,
        READY,
        &v2_limits(),
        Duration::from_secs_f64(requested),
        Ok(tolerance),
    )
    .expect("a long request is admitted");
    let duration_s = played_servo(plan).0;
    assert!(
        (duration_s - requested).abs() < EPS,
        "asked for {requested:.2}s, the move was planned at {duration_s:.2}s"
    );
}

// A move that ends as pure rotation (position converges early, orientation
// keeps slewing) does not trip the stall guard: rotational progress counts as
// progress. Exercises the law directly with a large in-place reorientation,
// where reference advance and position shrink both go quiet while the wrist is
// still turning. In place means about the grasp point: the wrist orbits it,
// and from Ready the elbow reaches its singularity floor beyond ~0.8 rad, so
// this turns as far as that posture allows.
#[test]
fn pure_reorientation_converges_in_the_servo_law() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.7) * end.rotation;
    roll_servo_to_convergence(&model, &start, &end, READY, default_tolerance());
}

// Convergence is judged against the caller's bar, so a wider bar is met
// strictly sooner than a narrower one on the same goal.
#[test]
fn a_wider_tolerance_converges_no_later_than_a_narrower_one() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.translation.vector.x -= 0.2;
    let steps = |tolerance: ServoTolerances| {
        servo_rollout(&model, &start, &end, READY, tolerance)
            .expect("the pull converges")
            .len()
    };
    let (tight, loose) = (steps(default_tolerance()), steps(panel_tolerance()));
    // Strict: the bar is read exactly when the lengths differ. The rollout is
    // deterministic, so the gap is reproducible.
    assert!(
        loose < tight,
        "the panel's bar converged in {loose} steps, the default's in {tight}"
    );
}

// The zero-length-line detector measures geometry, not acceptance: it is the
// law's own floor, so a pure reorientation starts its reference fully advanced
// no matter how wide the caller's position slack is.
#[test]
fn a_pure_reorientation_is_unaffected_by_a_wide_position_tolerance() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.2) * start.rotation;
    // A meter of position slack: far past the zero-length-line threshold.
    let sloppy = ServoTolerances::new(1.0, 1e-3).expect("a meter of slack is reachable");
    let configs = servo_rollout(&model, &start, &end, READY, sloppy)
        .expect("a 0.2 rad reorientation converges");
    assert!(
        configs.len() > 1,
        "a 0.2 rad reorientation is not converged at a 1e-3 rad bar"
    );
    assert!(
        configs[1]
            .iter()
            .zip(READY.iter())
            .any(|(a, b)| (a - b).abs() > 0.0),
        "a pure reorientation must still command motion"
    );
}
