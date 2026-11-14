//! The field cases of the OpenArm's Cartesian plan: the postures and moves
//! reported from the robot, planned on the bundled models at the launcher's
//! limits, as regressions; and what a line played as a joint trajectory holds.

use std::time::{Duration, Instant};

use hegel::TestCase;
use hegel::generators as gs;
use robot_description::states;
use srs_model::chain_kinematics::{interpolate, pose_miss};
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, Unit, UnitQuaternion, Vector3};
use srs_model::{Arm, ArmAnglePolicy, ELBOW, JointVec, plan_line};

mod helpers;

use helpers::fixtures::*;

/// The joints of an arm.
const ARM_JOINTS: usize = 7;

/// How far (m) the hand of a played line may stray from the exact line.
const STRAY_M: f64 = 2e-3;
/// How far a played line's hand may turn from the exact line's.
const STRAY_RAD: f64 = 2e-2;
/// The most a joint of a short, well-conditioned line swings beyond its net turn: a
/// change a joint path takes as none.
const QUIET_SWING_RAD: f64 = control_core::path_timing::MIN_JOINT_CHANGE_RAD;

/// The furthest the hand of `walk` strays from the exact line from `start` to `end`,
/// played at the launcher's limits: in meters off the segment, and in radians off the
/// turn from `start` to `end`.
fn strays(
    model: &Arm,
    walk: &[JointVec],
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
) -> (f64, f64) {
    let t0 = Instant::now();
    let traj = played(walk, t0);
    let duration = traj.clock().duration();
    let ab = end.translation.vector - start.translation.vector;
    (0..=400u32)
        .map(|k| {
            let ee = model.world_ee_pose(&traj.sample(t0 + duration * k / 400));
            let from_start = ee.translation.vector - start.translation.vector;
            let along = if ab.norm_squared() > 0.0 {
                (from_start.dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let off_the_turn = (0..=100)
                .map(|j| {
                    let on_the_turn = interpolate(start, end, f64::from(j) / 100.0).rotation;
                    ee.rotation.angle_to(&on_the_turn)
                })
                .fold(f64::INFINITY, f64::min);
            ((from_start - ab * along).norm(), off_the_turn)
        })
        .fold((0.0, 0.0), |(m, rad), (off_m, off_rad)| {
            (off_m.max(m), off_rad.max(rad))
        })
}

fn time_of(walk: &[JointVec]) -> f64 {
    played(walk, Instant::now())
        .clock()
        .duration()
        .as_secs_f64()
}

/// The most any joint of `walk` travels back and forth beyond its net turn.
fn swing_of(walk: &[JointVec]) -> f64 {
    (0..ARM_JOINTS)
        .map(|i| {
            let travel: f64 = walk.windows(2).map(|w| (w[1][i] - w[0][i]).abs()).sum();
            travel - (walk[walk.len() - 1][i] - walk[0][i]).abs()
        })
        .fold(0.0, f64::max)
}

// The field repro: the seeded walk tracks it as a straight line, landing on the goal.
#[test]
fn the_cross_body_pull_is_a_line() {
    let model = v2_right_arm();
    let (start, end, seed) = cross_body_pull(&model);
    let walk = plan_line(&model, &end, seed).expect("the pull tracks as a line");
    assert_lands_on(&model, &walk[walk.len() - 1], &end);
    let (off_m, off_rad) = strays(&model, &walk, &start, &end);
    assert!(
        off_m <= STRAY_M && off_rad <= STRAY_RAD,
        "{off_m} m, {off_rad} rad"
    );
}

// The user-reported pull: x back to -0.2 from the posture it was reported from, which a
// stream carries out live, fired as an action.
#[test]
fn the_reported_pull_to_x_minus_02_plans_a_line() {
    let model = v2_right_arm();
    // The recorded pull is a tip coordinate, so -0.2 is placed there and carried to the
    // tool: the same physical motion, in the frame commanded.
    let mut end_tip = tip_world(&model, &REPORTED_START);
    end_tip.translation.vector.x = -0.2;
    let end = end_tip * model.tool();
    let walk = plan_line(&model, &end, REPORTED_START).expect("the pull tracks as a line");
    assert_lands_on(&model, &walk[walk.len() - 1], &end);
}

// After the pull parks the arm at an unusual arm angle, an ordinary nudge from that posture
// is a line too, and sweeps no joint.
#[test]
fn small_nudge_after_the_cross_body_pull_stays_a_quiet_line() {
    let model = v2_right_arm();
    let (_, end, seed) = cross_body_pull(&model);
    let pull = plan_line(&model, &end, seed).expect("the pull tracks as a line");
    let parked = pull[pull.len() - 1];
    let mut nudge_end = model.world_ee_pose(&parked);
    nudge_end.translation.vector.x += 0.03;
    let walk = plan_line(&model, &nudge_end, parked).expect("an ordinary nudge must stay a line");
    let swing = swing_of(&walk);
    assert!(swing <= QUIET_SWING_RAD, "the nudge swings {swing} rad");
}

// An easy short move sweeps no joint.
#[test]
fn an_easy_move_sweeps_no_joint() {
    let model = v2_right_arm();
    let mut end = ready_pose(&model);
    end.translation.vector.z += 0.05;
    let walk = plan_line(&model, &end, posture(V2, "right_arm", states::READY))
        .expect("small lift from ready is planned");
    let swing = swing_of(&walk);
    assert!(swing <= QUIET_SWING_RAD, "the lift swings {swing} rad");
}

// The 25 cm lift from this field-recorded grasp posture of the v2 left arm: the walk stays
// inside the limits, and the hand on the line the whole way up.
#[test]
fn the_field_lift_stays_on_the_line_inside_the_limits() {
    let model = v2_left_arm();
    let mut end = model.world_ee_pose(&REPORTED_START_LEFT);
    end.translation.vector.z += 0.25;
    let walk = plan_line(&model, &end, REPORTED_START_LEFT).expect("the lift must stay a line");
    let limits = model.limits();
    for (k, q) in walk.iter().enumerate().skip(1) {
        for (i, (&v, limit)) in q.iter().zip(&limits).enumerate() {
            assert!(
                limit.contains(v),
                "joint {i} outside its limits at state {k}"
            );
        }
    }
    let (off_m, off_rad) = strays(
        &model,
        &walk,
        &model.world_ee_pose(&REPORTED_START_LEFT),
        &end,
    );
    assert!(
        off_m <= STRAY_M && off_rad <= STRAY_RAD,
        "{off_m} m, {off_rad} rad"
    );
}

// A turn in place about the grasp point has a line too: the wrist orbits the point and the
// line lands on the orientation.
#[test]
fn a_pure_reorientation_plans_a_line() {
    let model = v2_right_arm();
    let start = ready_pose(&model);
    let mut end = start;
    end.rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.7) * end.rotation;
    let walk = plan_line(&model, &end, posture(V2, "right_arm", states::READY))
        .expect("a 0.7 rad turn from Ready tracks");
    assert!(time_of(&walk) > 0.0);
    assert_lands_on(&model, &walk[walk.len() - 1], &end);
}

#[test]
fn an_unreachable_goal_has_no_line() {
    let arm = v1_left_arm();
    let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
    let mut unreachable = arm.world_pose(&arm.at(&seed).ee_pose());
    unreachable.translation.vector.x += 10.0;
    assert_eq!(
        plan_line(&arm, &unreachable, seed),
        None,
        "10 m away is out of reach"
    );
}

// The move a search for slow lines found on the right arm, a turn in place with a joint on its
// limit: it takes the time its walk needs, on the line.
#[test]
fn a_turn_in_place_on_a_joint_limit_takes_the_time_of_its_walk() {
    let model = v2_right_arm();
    let seed: JointVec = [
        -0.5,
        3.3160999999999996,
        0.0,
        2.30859375,
        1.0,
        0.0,
        0.01953125,
    ];
    let end = Isometry3::from_parts(
        Translation3::new(
            0.22708814676410083,
            -0.16227560836854987,
            0.5360335470985803,
        ),
        UnitQuaternion::from_quaternion(Quaternion::new(
            0.35492938102695715,
            -0.8141879802039343,
            0.4594481026635893,
            0.005523434989812087,
        )),
    );
    let walk = plan_line(&model, &end, seed).expect("the turn tracks");
    let time_s = time_of(&walk);
    // 2.34 s at the launcher's limits.
    assert!(time_s < 3.0, "the walk takes {time_s} s");
    let (off_m, off_rad) = strays(&model, &walk, &model.world_ee_pose(&seed), &end);
    assert!(
        off_m <= STRAY_M && off_rad <= STRAY_RAD,
        "{off_m} m, {off_rad} rad"
    );
}

// A move through the shoulder's near-gimbal, j1 and j3 almost parallel: the seeded walk swings
// no joint past a radian.
#[test]
fn a_line_past_the_shoulders_near_gimbal_sweeps_no_joint_out_and_back() {
    let model = v2_right_arm();
    let seed: JointVec = [
        0.30536698277083985,
        1.6329557795173737,
        0.9236719980886103,
        1.1731904203527241,
        -0.34474169198561144,
        -0.4483644938931237,
        -0.3831886672587488,
    ];
    let end = Isometry3::from_parts(
        Translation3::new(
            -0.02075165570287052,
            -0.7003013061991421,
            0.8784807024580981,
        ),
        UnitQuaternion::from_quaternion(Quaternion::new(
            0.44957322108241293,
            0.19680448605906553,
            0.3446837531922017,
            -0.8002156105937623,
        )),
    );
    let walk = plan_line(&model, &end, seed).expect("the line tracks");
    let swing = swing_of(&walk);
    assert!(swing <= 1.0, "a joint swings {swing} rad");
}

// A move toward a straight elbow whose line no descent from the state before continues: the
// goal has a configuration, and it has no plan.
#[test]
fn a_line_no_descent_continues_has_no_plan() {
    let model = v2_right_arm();
    let seed: JointVec = [
        1.8492735835567866,
        1.6619512835286385,
        0.407590169470458,
        0.5030512159414721,
        -0.3894408746599456,
        -0.15439797023539903,
        -0.06501276773723075,
    ];
    let end = Isometry3::from_parts(
        Translation3::new(
            -0.06948327918718664,
            -0.7426881708286679,
            0.7202940376338689,
        ),
        UnitQuaternion::from_quaternion(Quaternion::new(
            0.6215032972114822,
            0.584661916495689,
            0.2716032202436591,
            -0.445124460918649,
        )),
    );
    assert!(
        model
            .solve_ik(&model.base_pose(&end), ArmAnglePolicy::FromSeed, &seed)
            .is_some(),
        "the goal has a configuration"
    );
    assert_eq!(plan_line(&model, &end, seed), None);
}

// A move whose steps leave the line between their states: the walk halves them, and the hand
// stays on it. Its steps alone play 19.6 mm off.
#[test]
fn a_line_halves_the_steps_that_leave_the_line() {
    let model = v2_right_arm();
    let seed: JointVec = [
        0.6184468721467242,
        1.7813815400246413,
        0.5486687029630122,
        0.6134055346379973,
        -0.07764002139704163,
        0.18239601407670425,
        0.00530395842377529,
    ];
    let end = Isometry3::from_parts(
        Translation3::new(
            -0.19096469789186654,
            -0.37974457088305036,
            0.899612817452504,
        ),
        UnitQuaternion::from_quaternion(Quaternion::new(
            0.18270058971218298,
            0.3174263631020636,
            0.44440760712646815,
            -0.8175346336729374,
        )),
    );
    let walk = plan_line(&model, &end, seed).expect("the line tracks");
    let (off_m, off_rad) = strays(&model, &walk, &model.world_ee_pose(&seed), &end);
    assert!(
        off_m <= STRAY_M && off_rad <= STRAY_RAD,
        "the hand strays {off_m} m and {off_rad} rad"
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
/// surface or at Ready, and a pose up to 50 cm and 0.8 rad from where that posture puts the
/// grasp point.
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
            let straight = limits[ELBOW].lo;
            seed[ELBOW] = tc.draw(gs::floats().min_value(straight).max_value(straight + 0.2));
        }
        3 => {
            seed = posture(
                V2,
                if right { "right_arm" } else { "left_arm" },
                states::READY,
            )
        }
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

// A line played as a joint trajectory at the launcher's limits keeps the hand on the exact
// line between the samples of its walk and lands on the goal, beside the singular surfaces too.
#[hegel::test(test_cases = 200, seed = Some(69))]
fn a_played_line_keeps_the_hand_on_the_line(tc: TestCase) {
    let (model, seed, end) = drawn_move(&tc);
    let Some(walk) = plan_line(&model, &end, seed) else {
        tc.event("no line");
        return;
    };
    let (off_m, off_rad) = strays(&model, &walk, &model.world_ee_pose(&seed), &end);
    tc.target(off_m);
    assert!(
        off_m <= STRAY_M,
        "{off_m} m off the line from {seed:?} to {end:?}"
    );
    assert!(
        off_rad <= STRAY_RAD,
        "{off_rad} rad off the turn from {seed:?} to {end:?}"
    );
    let t0 = Instant::now();
    let traj = played(&walk, t0);
    let landed =
        model.world_ee_pose(&traj.sample(t0 + traj.clock().duration() + Duration::from_secs(1)));
    let miss = pose_miss(&landed, &end);
    assert!(
        miss.position_m <= 1e-5 && miss.orientation_rad <= 1e-5,
        "{miss:?}"
    );
}
