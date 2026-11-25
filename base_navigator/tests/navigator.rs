//! The navigator booted in-process on a simulated clock the test ticks,
//! against a mock base of drive_link that executes its setpoints through
//! base_state's limiter, and cameras and 2-D lidars that render boxes: the
//! refusals of a goal in base_navigation's order, the drives around what the
//! cameras see, the 2-D lidar's grid, the recoveries, a plan held across
//! steps, the replacement, the cancels and the holds, the release burst after
//! an ending, the silences, the start-over of the odometry, and a clock that
//! lags the messages or leaps, each judged on the node's own clock.

mod helpers;

use std::f64::consts::{FRAC_PI_2, PI};

use control_core::drive_link::FollowState;
use helpers::log::{keep_the_log, logged_with};
use helpers::mock_base::{MockBase, ModeReport, Planar};
use helpers::outcome;
use helpers::scene::{Box3, DepthLens, Scene, pit};
use helpers::*;
use peppygen::exposed_actions::base_navigation::navigate_to_pose::GoalTarget;
use peppygen::fixtures::exposed_actions::base_navigation::{
    back_up, drive_on_heading, navigate_to_pose, spin,
};

use navigate_to_pose::{FailureKind, GoalRequestData};

/// A box ahead of the base, its near face 1.0 m out: 0.3 m deep, 0.6 m wide
/// and 0.4 m tall, which the front camera sees from where the base starts.
const BOX_AHEAD: Box3 = Box3::new([1.0, -0.3, 0.0], [1.3, 0.3, 0.4]);

/// A goal 0.7 m behind [`BOX_AHEAD`].
const BEHIND_THE_BOX: [f64; 2] = [2.0, 0.0];

/// A goal 0.1 m before [`BOX_AHEAD`]'s near face: within the padded
/// footprint's inscribed radius, 0.21 m, of it.
const AT_THE_BOX: [f64; 2] = [0.9, 0.0];

/// The side of a cell of the tests' grid (m).
const CELL_M: f64 = 0.05;

/// How many steps a test leaves a source silent before a goal: 1 s, twice
/// the source timeout and drive_link's 0.5 s, so the node's clock has passed
/// them whichever tick it read last when the goal arrives.
const SILENT_STEPS: usize = 20;

/// The goal of a pen, inside it, 0.45 m from each wall.
const PEN_GOAL: [f64; 2] = [1.6, 0.0];

/// A closed pen of walls 0.2 m tall around [`PEN_GOAL`], its front wall
/// first, low enough for a camera to see over the front wall to the back one.
fn pen() -> Vec<Box3> {
    vec![
        Box3::new([1.1, -0.5, 0.0], [1.15, 0.5, 0.2]),
        Box3::new([2.05, -0.5, 0.0], [2.1, 0.5, 0.2]),
        Box3::new([1.1, -0.5, 0.0], [2.1, -0.45, 0.2]),
        Box3::new([1.1, 0.45, 0.0], [2.1, 0.5, 0.2]),
    ]
}

/// A navigation to `point` in odom heading +x, as [`goal`] gives it.
fn to(point: [f64; 2]) -> GoalRequestData {
    goal(point[0], point[1])
}

/// A wall across the way 0.1 m deep and 4.8 m wide, `gap_m` ahead of the
/// front edge of `base`, which heads along +x.
fn wall_ahead_of(base: &MockBase, gap_m: f64) -> Box3 {
    let front = base.truth.x + FOOTPRINT[0][0];
    Box3::new([front + gap_m, -2.4, 0.0], [front + gap_m + 0.1, 2.4, 0.4])
}

/// Asserts that `message` holds each of `parts`.
fn assert_names(message: &str, parts: &[&str]) {
    for part in parts {
        assert!(message.contains(part), "{message:?} does not name {part:?}");
    }
}

/// The steps of a goal in flight up to the one that ends it, and the
/// setpoints streamed from that step on: the release burst after it, if any.
async fn ending(rig: &mut Rig) -> (Vec<StepSeen>, Vec<(i64, Setpoint)>) {
    let ran = rig.run_until_ended().await;
    let after = rig.burst_after(ended_ns(&ran)).await;
    (ran, after)
}

/// The time of the last of `steps`, the one that ended a goal.
fn ended_ns(steps: &[StepSeen]) -> i64 {
    steps.last().expect("the step that ended the goal").at_ns
}

/// Asserts that `after`, the setpoints streamed from `ended_ns` on, at which
/// a goal ended, are the release burst: zero velocity until at least 3
/// setpoints and 0.25 s of the clock from the ending, then nothing.
fn assert_release_burst(ended_ns: i64, after: &[(i64, Setpoint)]) {
    assert!(
        after.iter().all(|(_, setpoint)| is_zero(setpoint)),
        "the burst streams zero velocity: {after:?}"
    );
    let completes = |index: usize| index + 1 >= 3 && after[index].0 - ended_ns >= 250_000_000;
    let done = (0..after.len())
        .find(|&index| completes(index))
        .unwrap_or_else(|| panic!("the burst streams 3 setpoints over 0.25 s: {after:?}"));
    assert_eq!(
        done + 1,
        after.len(),
        "nothing streams after the burst: {after:?}"
    );
}

/// Asserts that `streamed` holds no setpoint.
fn assert_streams_nothing(streamed: &[(i64, Setpoint)], when: &str) {
    assert!(
        streamed.is_empty(),
        "{when}: the node streamed {streamed:?}"
    );
}

/// The setpoints of `steps` that arrived, by stamp.
fn setpoints(rig: &Rig, steps: &[StepSeen]) -> Vec<Setpoint> {
    steps
        .iter()
        .filter_map(|step| rig.world.setpoint_at(step.at_ns).cloned())
        .collect()
}

/// Step until the base rests, at most `steps` steps.
async fn until_at_rest(rig: &mut Rig, steps: usize) {
    for _ in 0..steps {
        if rig.world.base.rests() {
            return;
        }
        rig.step().await;
    }
    assert!(
        rig.world.base.rests(),
        "the base rests: {:?}",
        rig.world.base.velocity
    );
}

// ---------------------------------------------------------------------------
// The refusals of a goal, in base_navigation's order.
// ---------------------------------------------------------------------------

// A goal that arrives before any drive_states waits up to the source
// timeout on the node's clock, then is refused with resource_unavailable
// naming what it waited for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_before_any_drive_states_waits_the_source_timeout_then_is_refused() {
    let mut rig = Rig::booted(params(), Scene::open(), &[FRONT]).await;
    rig.world.base_reports = false;
    rig.answer_describe_base(measured());
    let sent_ns = rig.world.now_ns;
    let refusal = rig.refusal_stepping(&goal(1.0, 0.0)).await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(&refusal.message, &["no drive_states arrived", "0.5 s"]);
    assert!(
        rig.world.now_ns - sent_ns >= 500_000_000,
        "the wait ran on the node's clock up to the source timeout"
    );
    rig.shutdown().await;
}

// A goal whose describe_base read does not answer within
// source_timeout_s is refused with resource_unavailable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_whose_describe_base_read_does_not_answer_in_the_source_timeout_is_refused() {
    let mut rig = Rig::booted(params(), Scene::open(), &[FRONT]).await;
    // The first drive_states arrives and describe_base is asked, never to
    // answer.
    rig.step().await;
    let sent_ns = rig.world.now_ns;
    let refusal = rig.refusal_stepping(&goal(1.0, 0.0)).await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(
        &refusal.message,
        &["describe_base did not answer", "source_timeout_s 0.5 s"],
    );
    assert!(rig.world.now_ns - sent_ns >= 500_000_000);
    rig.shutdown().await;
}

// A goal for a base whose describe_base names commanded odometry is
// refused with precondition_failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_for_a_base_of_commanded_odometry_is_refused_precondition_failed() {
    let mut rig = Rig::start_with(
        params(),
        Scene::open(),
        &[FRONT],
        base_answer(LIMITS, "commanded"),
    )
    .await;
    let refusal = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(refusal.kind, FailureKind::PreconditionFailed, "{refusal:?}");
    assert_names(&refusal.message, &["commanded odometry"]);
    rig.shutdown().await;
}

// Goals with a max_speed_m_s of -0.1, one above describe_base's linear_limits
// along x and one between 0 and the deadband, a tolerance of 0, a NaN, an
// orientation that is no unit quaternion and a target off the floor plane are
// refused with rejected, each message naming the value.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn goals_with_a_value_the_navigator_does_not_take_are_refused_rejected() {
    let mut rig = Rig::start(Scene::open()).await;
    let cases = [
        (
            GoalRequestData {
                max_speed_m_s: -0.1,
                ..goal(1.0, 0.0)
            },
            "max_speed_m_s",
        ),
        (
            GoalRequestData {
                max_speed_m_s: LIMITS.max_linear_m_s + 0.01,
                ..goal(1.0, 0.0)
            },
            "max_speed_m_s",
        ),
        (
            GoalRequestData {
                max_speed_m_s: LIMITS.linear_deadband_m_s / 2.0,
                ..goal(1.0, 0.0)
            },
            "deadband",
        ),
        (
            GoalRequestData {
                position_tolerance_m: 0.0,
                ..goal(1.0, 0.0)
            },
            "position_tolerance_m",
        ),
        (
            GoalRequestData {
                orientation_tolerance_rad: 0.0,
                ..goal(1.0, 0.0)
            },
            "orientation_tolerance_rad",
        ),
        (
            GoalRequestData {
                max_speed_m_s: f64::NAN,
                ..goal(1.0, 0.0)
            },
            "max_speed_m_s",
        ),
        (
            GoalRequestData {
                target: GoalTarget {
                    position: [f64::NAN, 0.0, 0.0],
                    ..in_odom(1.0, 0.0, 0.0)
                },
                ..goal(1.0, 0.0)
            },
            "the target",
        ),
        (
            GoalRequestData {
                target: GoalTarget {
                    orientation: [0.0, 0.0, 0.0, 2.0],
                    ..in_odom(1.0, 0.0, 0.0)
                },
                ..goal(1.0, 0.0)
            },
            "the target",
        ),
        (
            GoalRequestData {
                target: GoalTarget {
                    position: [1.0, 0.0, 0.1],
                    ..in_odom(1.0, 0.0, 0.0)
                },
                ..goal(1.0, 0.0)
            },
            "send a target in odom or the robot frame",
        ),
    ];
    for (data, names) in &cases {
        let refusal = rig.refusal(data).await;
        assert_eq!(refusal.kind, FailureKind::Rejected, "{data:?}: {refusal:?}");
        assert_names(&refusal.message, &[*names]);
    }
    rig.shutdown().await;
}

// A goal outside the grid centered on the base is refused with rejected,
// the message naming the grid's half side, and one inside it is accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_outside_the_grid_centered_on_the_base_is_refused_rejected() {
    let mut rig = Rig::start(Scene::open()).await;
    for (x, y) in [
        (grid_half_side_m() + 0.1, 0.0),
        (0.0, -grid_half_side_m() - 0.1),
    ] {
        let refusal = rig.refusal(&goal(x, y)).await;
        assert_eq!(
            refusal.kind,
            FailureKind::Rejected,
            "({x}, {y}): {refusal:?}"
        );
        assert_names(
            &refusal.message,
            &["outside the grid", "2.5 m", "send a nearer target"],
        );
    }
    let inside = rig.send_when_ready(&goal(2.0, 0.0)).await;
    assert!(inside.accepted, "{:?}", inside.refusal);
    rig.shutdown().await;
}

// A goal in a frame the node's tree does not hold is refused with
// unknown_frame, once the placement's wait on the node's clock ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_in_a_frame_the_tree_does_not_hold_is_refused_unknown_frame() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    let kitchen = GoalRequestData {
        target: GoalTarget {
            frame: Some("kitchen".to_string()),
            ..in_odom(1.0, 0.0, 0.0)
        },
        ..goal(1.0, 0.0)
    };
    let refusal = rig.refusal_stepping(&kitchen).await;
    assert_eq!(refusal.kind, FailureKind::UnknownFrame, "{refusal:?}");
    assert_names(&refusal.message, &["kitchen"]);
    rig.shutdown().await;
}

// base_navigation's refusals: a goal in the map frame is refused with
// not_localized, saying that no map is loaded and to send the target in odom
// or the robot frame; one off the floor plane in the map frame is rejected
// first, saying to send a planar target in map or the robot frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_in_the_map_frame_is_refused_not_localized_saying_what_to_send() {
    let mut rig = Rig::start(Scene::open()).await;
    let in_map = |z: f64| GoalRequestData {
        target: GoalTarget {
            frame: Some("map".to_string()),
            position: [1.0, 0.0, z],
            ..in_odom(1.0, 0.0, 0.0)
        },
        ..goal(1.0, 0.0)
    };
    let refusal = rig.refusal(&in_map(0.0)).await;
    assert_eq!(refusal.kind, FailureKind::NotLocalized, "{refusal:?}");
    assert_names(
        &refusal.message,
        &["no map is loaded", "odom or the robot frame"],
    );
    let lifted = rig.refusal(&in_map(0.5)).await;
    assert_eq!(lifted.kind, FailureKind::Rejected, "{lifted:?}");
    assert_names(
        &lifted.message,
        &["placed in map", "send a target in map or the robot frame"],
    );
    rig.shutdown().await;
}

// A goal with two causes gets the kind of the earlier check in
// base_navigation's order: while a robot stop holds the base, a map goal is
// refused with not_localized, a NaN speed and a target outside the grid with
// rejected, and silent drive_states with resource_unavailable, each before
// the hold's precondition_failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_with_two_causes_gets_the_kind_of_the_earlier_check() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    rig.world.base.follow_state = FollowState::RobotStop;
    rig.step().await;
    let held = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(held.kind, FailureKind::PreconditionFailed, "the hold alone");
    let in_map = GoalRequestData {
        target: GoalTarget {
            frame: Some("map".to_string()),
            ..in_odom(1.0, 0.0, 0.0)
        },
        ..goal(1.0, 0.0)
    };
    let nan_speed = GoalRequestData {
        max_speed_m_s: f64::NAN,
        ..goal(1.0, 0.0)
    };
    let outside = goal(grid_half_side_m() + 0.1, 0.0);
    for (data, kind) in [
        (&in_map, FailureKind::NotLocalized),
        (&nan_speed, FailureKind::Rejected),
        (&outside, FailureKind::Rejected),
    ] {
        let refusal = rig.refusal(data).await;
        assert_eq!(refusal.kind, kind, "{data:?}: {refusal:?}");
    }
    rig.world.base_reports = false;
    rig.steps(SILENT_STEPS).await;
    let silent = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(silent.kind, FailureKind::ResourceUnavailable, "{silent:?}");
    assert_names(&silent.message, &["drive_states"]);
    rig.shutdown().await;
}

// A goal with no drive_states in more than 0.5 s is refused with
// resource_unavailable, the message naming drive_states and what to do.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_drive_states_are_silent_is_refused_resource_unavailable() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    rig.world.base_reports = false;
    rig.steps(SILENT_STEPS).await;
    let refusal = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(
        &refusal.message,
        &["sent no drive_states", "wait for the base"],
    );
    rig.shutdown().await;
}

// A goal while follow_state is base_silence is refused with
// resource_unavailable, saying to send it again 0.6 s after the
// refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_a_silence_of_the_bases_sensing_holds_it_is_refused_resource_unavailable() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    rig.world.base.follow_state = FollowState::BaseSilence;
    rig.step().await;
    let refusal = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(
        &refusal.message,
        &["base_silence", "more than 0.6 s after this refusal"],
    );
    rig.shutdown().await;
}

// A goal while a camera the grid reads is silent is refused with
// resource_unavailable, naming the camera.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_a_camera_is_silent_is_refused_resource_unavailable_naming_it() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    rig.world.cameras_on = false;
    rig.steps(SILENT_STEPS).await;
    let refusal = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(
        &refusal.message,
        &["the depth camera front", "source_timeout_s"],
    );
    rig.shutdown().await;
}

// A goal while follow_state is follower_goal or robot_stop is refused
// with precondition_failed, saying what to do, and accepted once the base
// follows again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_follower_goal_or_robot_stop_holds_the_base_is_refused_precondition_failed() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    for (hold, says) in [
        (FollowState::FollowerGoal, "once that goal ends"),
        (FollowState::RobotStop, "more than 0.6 s after this refusal"),
    ] {
        rig.world.base.follow_state = hold;
        rig.step().await;
        let refusal = rig.refusal(&goal(1.0, 0.0)).await;
        assert_eq!(
            refusal.kind,
            FailureKind::PreconditionFailed,
            "{hold:?}: {refusal:?}"
        );
        assert_names(&refusal.message, &[hold.as_str(), says]);
    }
    rig.world.base.follow_state = FollowState::Following;
    rig.step().await;
    let accepted = rig.send(&goal(1.0, 0.0)).await;
    assert!(accepted.accepted, "{:?}", accepted.refusal);
    rig.shutdown().await;
}

// base_navigation's refusals (base_mode): a goal while follow_state is
// base_mode is refused with precondition_failed, naming the mode and what
// brings the base to standing; while the mode is standing, saying to send it
// again 0.6 s after the refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_the_bases_mode_holds_it_is_refused_naming_the_mode() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    rig.world.base.follow_state = FollowState::BaseMode;
    for (mode, says) in [
        ("fallen", "walking_robot's stand"),
        ("standing", "more than 0.6 s after this refusal"),
    ] {
        rig.world.base.mode = ModeReport::Fixed(mode);
        rig.step().await;
        let refusal = rig.refusal(&goal(1.0, 0.0)).await;
        assert_eq!(
            refusal.kind,
            FailureKind::PreconditionFailed,
            "{mode}: {refusal:?}"
        );
        assert_names(&refusal.message, &["base_mode", mode, says]);
    }
    rig.shutdown().await;
}

// base_navigation's refusals (parts_within_footprint): a goal while
// drive_states reports parts_within_footprint false is refused with
// precondition_failed, naming postures' move_to_home.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_a_limb_is_outside_the_footprint_is_refused_naming_move_to_home() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.ready().await;
    rig.world.base.parts_within_footprint = false;
    rig.step().await;
    let refusal = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(refusal.kind, FailureKind::PreconditionFailed, "{refusal:?}");
    assert_names(
        &refusal.message,
        &["parts_within_footprint false", "move_to_home"],
    );
    rig.shutdown().await;
}

// A spin whose angle_rad is above 0 while describe_base's angular_limits
// max_rate is 0, a drive_on_heading whose end lies outside the grid, a
// back_up with distance_m 0 and a drive_on_heading with speed_m_s below the
// deadband are refused with rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn moves_the_base_cannot_run_are_refused_rejected() {
    let mut clockwise_only = measured();
    clockwise_only.angular_limits.max_rate = 0.0;
    let mut rig = Rig::start_with(params(), Scene::open(), &[FRONT], clockwise_only).await;
    let counterclockwise = rig.send_spin(&spin_by(1.0, 0.0)).await;
    let refused = counterclockwise
        .refusal
        .expect("a counterclockwise spin is refused");
    assert_eq!(refused.kind, spin::FailureKind::Rejected, "{refused:?}");
    assert_names(&refused.message, &["angle_rad 1", "max_rate 0"]);
    let long = rig
        .send_drive(&ahead(grid_half_side_m() + 0.1, 0.2, 0.0))
        .await;
    let refused = long.refusal.expect("a drive past the grid is refused");
    assert_eq!(
        refused.kind,
        drive_on_heading::FailureKind::Rejected,
        "{refused:?}"
    );
    assert_names(
        &refused.message,
        &["outside the grid", "send a shorter distance"],
    );
    let slow = rig
        .send_drive(&ahead(0.5, LIMITS.linear_deadband_m_s / 2.0, 0.0))
        .await;
    let refused = slow.refusal.expect("a drive below the deadband is refused");
    assert_eq!(
        refused.kind,
        drive_on_heading::FailureKind::Rejected,
        "{refused:?}"
    );
    assert_names(&refused.message, &["speed_m_s", "deadband"]);
    let none = rig.send_back_up(&back(0.0, 0.1, 0.0)).await;
    let refused = none.refusal.expect("a back-up of no distance is refused");
    assert_eq!(refused.kind, back_up::FailureKind::Rejected, "{refused:?}");
    assert_names(&refused.message, &["distance_m"]);
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// Drives around what the cameras see.
// ---------------------------------------------------------------------------

// A goal within the inscribed radius of a box face the camera sees fails
// with unreachable before the base moves, streaming no setpoint, and the
// first goal that fails before it streams brings no release burst.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_at_a_seen_box_fails_unreachable_before_the_base_moves_and_streams_nothing() {
    let mut rig = Rig::start(Scene::with(&[BOX_AHEAD])).await;
    let handle = rig.send_when_ready(&to(AT_THE_BOX)).await;
    assert!(handle.accepted, "{:?}", handle.refusal);
    let ran = rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable
    );
    let sent_ns = ran[0].at_ns;
    let after = rig.burst_after(sent_ns).await;
    assert_streams_nothing(&after, "the failed navigation, and after it");
    assert_eq!(
        rig.world.base.truth,
        Planar::default(),
        "the base never moved"
    );
    rig.shutdown().await;
}

// base_navigation's tries: a navigation whose plan finds the base in
// collision fails with start_in_collision. The odometry slips the base onto
// a box face the camera saw, the camera dark since, so no reading clears the
// footprint there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_navigation_that_starts_in_collision_fails_start_in_collision() {
    let mut rig = Rig::start(Scene::with(&[BOX_AHEAD])).await;
    rig.ready().await;
    rig.world.cameras_on = false;
    rig.world.base.slip_odometry(AT_THE_BOX[0], 0.0);
    rig.step().await;
    let handle = rig.send(&goal(-1.0, 0.0)).await;
    assert!(handle.accepted, "{:?}", handle.refusal);
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::StartInCollision
    );
    rig.shutdown().await;
}

// A navigation drives the base around a box to a
// goal behind it, without contact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_navigation_drives_around_a_seen_box_to_a_goal_behind_it_without_contact() {
    let mut rig = Rig::start(Scene::with(&[BOX_AHEAD])).await;
    let handle = rig.send_when_ready(&to(BEHIND_THE_BOX)).await;
    rig.run_checking(|base| {
        assert!(
            !touches(base, &BOX_AHEAD),
            "the base touched the box: {:?}",
            base.truth
        )
    })
    .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result);
    let [x, y, _] = data.final_pose.position;
    assert!(
        (x - BEHIND_THE_BOX[0]).hypot(y - BEHIND_THE_BOX[1]) < 0.2,
        "{data:?}"
    );
    rig.shutdown().await;
}

/// botshare's footprint, 0.78 m long and 0.58 m wide about the robot frame's
/// origin, which the 0.01 m padding grows to 0.80 m by 0.60 m.
const BOTSHARE_FOOTPRINT: [[f64; 2]; 4] =
    [[0.39, -0.29], [0.39, 0.29], [-0.39, 0.29], [-0.39, -0.29]];

/// The padded footprint's corners about the robot frame's origin.
const BOTSHARE_PADDED_CORNERS: [[f64; 2]; 4] = [[0.4, -0.3], [0.4, 0.3], [-0.4, 0.3], [-0.4, -0.3]];

/// botshare's depth camera at its front, 0.144 m up, pitched 10 degrees
/// down.
const BOTSHARE_FRONT: Mount = Mount {
    name: "front",
    position: [0.387, 0.0, 0.144],
    yaw_rad: 0.0,
    pitch_rad: 10.0 * PI / 180.0,
    lens: LENS,
};

/// describe_base's answer for the rig's base with botshare's footprint.
fn botshare_answer() -> peppygen::consumed_services::base_state::describe_base::ResponseData {
    let mut answer = measured();
    answer.footprint.points = BOTSHARE_FOOTPRINT
        .iter()
        .map(|&[x, y]| {
            peppygen::consumed_services::base_state::describe_base::ResponseDataFootprintPointsItem {
                position: [x, y, 0.0],
            }
        })
        .collect();
    answer
}

/// How far along odom's x the padded footprint of botshare reaches where
/// `base` stands.
fn padded_reach_x(base: &MockBase) -> f64 {
    let Planar { x, yaw, .. } = base.odometry;
    let (sin, cos) = yaw.sin_cos();
    BOTSHARE_PADDED_CORNERS
        .iter()
        .map(|[px, py]| x + cos * px - sin * py)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The base of `rig`, botshare's, drives to a goal 2.1 m ahead, 0.45 m inside
/// the edge of the grid centered on the start, and turns there by a quarter
/// turn to the goal's heading, -pi/2. Its padded footprint's corners lie 0.5 m
/// from the robot frame's origin, so they leave that square during the turn,
/// and the navigation completes within its tolerances.
async fn turn_at_a_goal_near_the_grids_first_edge(mut rig: Rig) {
    let quarter_turn = GoalRequestData {
        target: in_odom(2.1, 0.0, -FRAC_PI_2),
        position_tolerance_m: 0.02,
        ..goal(2.1, 0.0)
    };
    let handle = rig.send_when_ready(&quarter_turn).await;
    let mut reach = f64::NEG_INFINITY;
    rig.run_checking(|base| reach = reach.max(padded_reach_x(base)))
        .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result);
    let first_edge = grid_half_side_m() + CELL_M;
    assert!(
        reach > first_edge,
        "the padded footprint reached x {reach}, inside the first square's edge at {first_edge}"
    );
    let [x, y, _] = data.final_pose.position;
    assert!((x - 2.1).hypot(y) <= 0.02, "{data:?}");
    assert!(
        (yaw_of(data.final_pose.orientation) + FRAC_PI_2).abs() <= 0.2,
        "{data:?}"
    );
    rig.shutdown().await;
}

// base_navigation's grid: botshare's base, its camera reading the floor,
// turns at a goal near the first grid's edge as
// `turn_at_a_goal_near_the_grids_first_edge` gives it: the grid centers on
// the base with each reading while the goal runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_navigation_turning_at_a_goal_near_the_grids_first_edge_completes() {
    let rig = Rig::start_with(
        params(),
        Scene::open(),
        &[BOTSHARE_FRONT],
        botshare_answer(),
    )
    .await;
    turn_at_a_goal_near_the_grids_first_edge(rig).await;
}

// An obstacle placed ahead of the moving base slows it and stops it
// before contact; the navigation then fails, and the base comes to rest.
// The test judges the approach up to the step its forward motion stops:
// the recoveries then run at the wall, where each clearing empties the
// cells the front camera no longer sees, as Nav2's default tree clears its
// costmap entirely.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_obstacle_placed_ahead_of_the_moving_base_slows_it_and_stops_it_before_contact() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    let cruise = rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let cruise_m_s = setpoints(&rig, &cruise)
        .iter()
        .map(forward)
        .fold(0.0, f64::max);
    let wall = wall_ahead_of(&rig.world.base, 0.8);
    rig.world.scene.boxes.push(wall);
    let mut stopped = false;
    let ran = rig
        .run_checking(|base| {
            stopped = stopped || base.velocity.linear_x() == 0.0;
            assert!(
                stopped || !touches(base, &wall),
                "the base touched the wall before it stopped: {:?}",
                base.truth
            )
        })
        .await;
    assert!(stopped, "the base stops before the navigation ends");
    rig.world.until_setpoint_from(ended_ns(&ran)).await;
    let speeds: Vec<f64> = setpoints(&rig, &ran).iter().map(forward).collect();
    assert!(
        speeds
            .iter()
            .any(|&speed| speed > 0.0 && speed < cruise_m_s),
        "the base slows: {speeds:?}"
    );
    assert!(speeds.contains(&0.0), "the base stops: {speeds:?}");
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    assert!(
        matches!(result.outcome, navigate_to_pose::ResultOutcome::Failed(_)),
        "{:?}",
        result.outcome
    );
    until_at_rest(&mut rig, 20).await;
    rig.shutdown().await;
}

// A step down across the way ahead stops the base before its edge.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_step_down_ahead_of_the_base_stops_it_before_the_edge() {
    let (min, max) = ([1.0, -2.4], [2.0, 2.4]);
    let mut rig = Rig::start(Scene::with_pit(min, max, 0.25)).await;
    let handle = rig.send_when_ready(&goal(2.3, 0.0)).await;
    rig.run_checking(|base| {
        assert!(
            !touches(base, &pit(min, max)),
            "the base drove over the drop: {:?}",
            base.truth
        )
    })
    .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    assert!(
        matches!(result.outcome, navigate_to_pose::ResultOutcome::Failed(_)),
        "{:?}",
        result.outcome
    );
    rig.shutdown().await;
}

// A goal beyond a step down, with a way around it, is reached around it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_beyond_a_seen_step_down_is_reached_around_it() {
    let (min, max) = ([0.8, -0.6], [1.6, 0.6]);
    let mut rig = Rig::start(Scene::with_pit(min, max, 0.25)).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_checking(|base| {
        assert!(
            !touches(base, &pit(min, max)),
            "the base drove over the drop: {:?}",
            base.truth
        )
    })
    .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

/// botshare's base camera, base_camera_joint of botshare_base.urdf: 0.387 m
/// ahead of the robot frame's origin and 0.144 m up, pitched 0.1745 rad up,
/// with the lens Waldo renders it by: 640 x 400 pixels, 313.937 pixels a unit
/// along each axis, millimeter counts, to 20 m.
const BOTSHARE_BASE_CAMERA: Mount = Mount {
    name: "base_camera",
    position: [0.387, 0.0, 0.144],
    yaw_rad: 0.0,
    pitch_rad: -0.1745,
    lens: DepthLens {
        width: 640,
        height: 400,
        fx: 313.937,
        fy: 313.937,
        cx: 319.5,
        cy: 199.5,
        unit_m: 0.001,
        max_depth_m: 20.0,
    },
};

/// How far (rad) Waldo rests botshare's base nose-up on a slab: the turn of
/// -0.0080 rad about the base's own y axis in the orientation
/// objects.object_states gives the base there.
const WALDO_REST_PITCH_UP_RAD: f64 = 0.008;

/// around_a_drop's L in the base's start frame: two slabs whose tops lie at
/// the floor's plane, the base on the first, and the room's floor 0.2 m below.
fn around_a_drop() -> Scene {
    Scene {
        boxes: vec![
            Box3::new([-30.0, -30.0, -1.2], [30.0, 30.0, -0.2]),
            Box3::new([-3.4, -0.5, -1.2], [1.1, 2.6, 0.0]),
            Box3::new([1.1, -1.9, -1.2], [4.2, 2.6, 0.0]),
        ],
    }
}

/// A goal on around_a_drop's second slab, 0.475 m inside its east edge at y
/// -1.9, at the center of cell (93, 21) of the grid centered on the start.
/// The room floor's returns past that edge, 4 m and more out and past the
/// obstacle range, cross the floor's plane, placed by the level body, in
/// cells 0.025 m to 0.225 m inside the edge, (2.175, -1.675) and
/// (2.325, -1.675) among them, 0.25 m and 0.29 m from the goal: within the
/// padded footprint's inscribed radius, 0.30 m. The cells that hold the edge
/// itself, 0.025 m inside it, lie 0.45 m or more from the goal, outside it.
const NEAR_THE_EAST_EDGE: [f64; 2] = [2.175, -1.425];

// botshare's base rests 0.008 rad nose-up, as Waldo rests it, and the navigator places
// each ray of its camera by the level body. The room floor's returns past the second
// slab's east edge lie past the obstacle range and mark no drop on the slab, so the
// navigation to a goal 0.475 m inside that edge is accepted and completes there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tilted_base_reaches_a_goal_near_a_slabs_far_edge() {
    let mut rig = Rig::booted(params(), around_a_drop(), &[BOTSHARE_BASE_CAMERA]).await;
    rig.world.body_pitch_up_rad = WALDO_REST_PITCH_UP_RAD;
    rig.answer_describe_base(botshare_answer());
    rig.step().await;
    let handle = rig.send_when_ready(&to(NEAR_THE_EAST_EDGE)).await;
    assert!(handle.accepted, "{:?}", handle.refusal);
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result);
    let [x, y, _] = data.final_pose.position;
    let [goal_x, goal_y] = NEAR_THE_EAST_EDGE;
    assert!((x - goal_x).hypot(y - goal_y) <= 0.1, "{data:?}");
    rig.shutdown().await;
}

// A camera that stops publishing brings the base to rest and fails the
// navigation with resource_unavailable, naming the camera; the release burst
// follows the failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_camera_that_stops_publishing_fails_the_navigation_and_brings_the_base_to_rest() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.1).await;
    rig.world.cameras_on = false;
    let ran = rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &[FRONT.name]);
    rig.world.cameras_on = true;
    until_at_rest(&mut rig, 20).await;
    let fence_ns = rig.fence().await;
    assert_release_burst(
        ended_ns(&ran),
        &rig.world.setpoints_in(ended_ns(&ran)..fence_ns),
    );
    rig.shutdown().await;
}

// Each frame's placement by the tree: the odometry reaches the node a frame
// period after each camera's frame of the same time, and each frame waits in
// its intake for the transform of its time, so the grid still takes each one:
// a goal at a seen box is admitted and fails with unreachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn odometry_that_lags_the_frames_by_a_frame_period_still_places_each_frame() {
    let mut rig = Rig::start(Scene::with(&[BOX_AHEAD])).await;
    rig.world.odometry_lag_steps = 1;
    let handle = rig.send_when_ready(&to(AT_THE_BOX)).await;
    assert!(handle.accepted, "{:?}", handle.refusal);
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable
    );
    rig.shutdown().await;
}

// A table whose top lies above max_obstacle_height_m is driven under.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_table_whose_top_lies_above_the_obstacle_band_is_driven_under() {
    let table = Scene::table([1.0, -0.6], [2.0, 0.6], 0.55);
    let mut rig = Rig::start(Scene::with(&table)).await;
    let handle = rig.send_when_ready(&goal(1.5, 0.0)).await;
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result);
    let [x, y, _] = data.final_pose.position;
    assert!((x - 1.5).hypot(y) < 0.2, "{data:?}");
    rig.shutdown().await;
}

/// Where a navigation to (2, 0) heading -pi/2 completes, and how far it
/// drove, with a camera on top of the base whose view holds `deck_half` of
/// the base's deck, or none.
async fn top_camera_drive(deck_half: Option<[f64; 3]>) -> ([f64; 3], f64, f64) {
    let mut rig = Rig::start_with(params(), Scene::open(), &[TOP], measured()).await;
    rig.world.deck_half = deck_half;
    let turn = GoalRequestData {
        target: in_odom(2.0, 0.0, -FRAC_PI_2),
        ..goal(2.0, 0.0)
    };
    let handle = rig.send_when_ready(&turn).await;
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result).clone();
    rig.shutdown().await;
    (
        data.final_pose.position,
        yaw_of(data.final_pose.orientation),
        data.distance_driven_m,
    )
}

/// How far (m) a drive's length may differ from another's of the same goal
/// whose steps took their inputs at other steps: a voxel, well below any
/// detour around an occupied cell.
const SAME_DRIVE_M: f64 = 0.05;

// With the depth camera on top of the base, seeing the base's own deck, the
// base drives as with no deck in view: to the goal within its tolerances,
// turning to the goal's heading in place, over the same length. The deck
// that reaches the footprint's outline puts returns at the outline itself,
// inside the padding band of Nav2's 0.01 m: the grid forgets the returns
// within the unpadded outline and clears the cells the padded footprint
// covers, and this run shows whether a return left between the two marks the
// base.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_camera_on_top_of_the_base_that_sees_its_deck_drives_as_with_no_deck_in_view() {
    let (_, _, unseen_m) = top_camera_drive(None).await;
    for deck_half in [None, Some(DECK_INSIDE_HALF), Some(DECK_AT_OUTLINE_HALF)] {
        let ([x, y, _], yaw, driven_m) = top_camera_drive(deck_half).await;
        assert!(
            (x - 2.0).hypot(y) <= 0.1 && (yaw + FRAC_PI_2).abs() <= 0.2,
            "the deck {deck_half:?}: the drive ends at ({x}, {y}) heading {yaw}"
        );
        assert!(
            (driven_m - unseen_m).abs() < SAME_DRIVE_M,
            "the deck {deck_half:?} changes the drive: {driven_m} m against {unseen_m} m"
        );
    }
}

// With the top camera's view holding the deck at the footprint's
// outline, at Nav2's 0.01 m padding, the drive around a box completes
// without contact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_camera_on_top_of_the_base_that_sees_its_deck_drives_around_a_box_without_contact() {
    let mut rig = Rig::start_with(params(), Scene::with(&[BOX_AHEAD]), &[TOP], measured()).await;
    rig.world.deck_half = Some(DECK_AT_OUTLINE_HALF);
    let handle = rig.send_when_ready(&to(BEHIND_THE_BOX)).await;
    rig.run_checking(|base| {
        assert!(
            !touches(base, &BOX_AHEAD),
            "the base touched the box: {:?}",
            base.truth
        )
    })
    .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

// Requirement, the depth frames of each camera it plans with: each feeds the
// grid, so a box only the left camera sees makes a goal at it unreachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_camera_feeds_the_grid_so_a_box_only_the_left_camera_sees_makes_a_goal_unreachable() {
    let left_box = Box3::new([-0.25, 1.0, 0.0], [0.25, 1.3, 0.4]);
    let mut rig = Rig::start_with(
        params(),
        Scene::with(&[left_box]),
        &[FRONT, LEFT],
        measured(),
    )
    .await;
    let handle = rig.send_when_ready(&goal(0.0, 0.9)).await;
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable
    );
    rig.shutdown().await;
}

/// The distance to the goal of each step of a navigation to (1.8, 0) heading
/// +y within `tolerance`, where the drive_states the step read placed the
/// robot frame, as its feedback gives it, and the forward speed of the
/// step's setpoint.
async fn approach(tolerance: f64) -> Vec<(f64, f64)> {
    let mut rig = Rig::start(Scene::open()).await;
    let target = GoalRequestData {
        target: in_odom(1.8, 0.0, FRAC_PI_2),
        position_tolerance_m: tolerance,
        orientation_tolerance_rad: 0.1,
        ..goal(1.8, 0.0)
    };
    let handle = rig.send_when_ready(&target).await;
    let ran = rig.run_until_ended().await;
    rig.world.until_setpoint_from(ended_ns(&ran)).await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    let steps = ran
        .iter()
        .filter_map(|step| {
            let [x, y, _] = step.navigation()?.current_pose.position;
            let ahead = rig.world.setpoint_at(step.at_ns).map_or(0.0, forward);
            Some(((x - 1.8).hypot(y), ahead))
        })
        .collect();
    rig.shutdown().await;
    steps
}

// A navigation with position_tolerance_m 0.3 stops commanding forward
// motion once the robot frame's origin stands within 0.3 m of the goal, and
// one with 0.05 only within 0.05 m.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forward_motion_stops_once_the_origin_stands_within_the_position_tolerance() {
    for tolerance in [0.3, 0.05] {
        let steps = approach(tolerance).await;
        let reached = steps
            .iter()
            .position(|(distance, _)| *distance <= tolerance)
            .expect("the base reaches the position");
        assert!(
            steps[reached..].iter().all(|(_, ahead)| *ahead <= 0.0),
            "{tolerance}: forward motion at or after the step that found the base within it: {:?}",
            &steps[reached..]
        );
        if tolerance < 0.3 {
            assert!(
                steps.iter().any(|(distance, ahead)| *distance < 0.3
                    && *distance > tolerance
                    && *ahead > 0.0),
                "the navigation drives on within 0.3 m"
            );
        }
    }
}

// With max_speed_m_s 0.2 the norm of every setpoint's linear_velocity
// stays at or below 0.2 m/s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_speed_caps_the_norm_of_every_setpoints_linear_velocity() {
    let mut rig = Rig::start(Scene::open()).await;
    let slow = GoalRequestData {
        max_speed_m_s: 0.2,
        ..goal(2.0, 0.0)
    };
    let handle = rig.send_when_ready(&slow).await;
    let steps = rig.run_until_ended().await;
    rig.world.until_setpoint_from(ended_ns(&steps)).await;
    let ran = setpoints(&rig, &steps);
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    assert!(ran.iter().all(|s| speed(s) <= 0.2), "{ran:?}");
    assert!(
        ran.iter().any(|s| speed(s) > 0.15),
        "the base drives near the cap: {ran:?}"
    );
    rig.shutdown().await;
}

// base_navigation's feedback: once per step, where the robot frame stood in
// odom at the timestamp of the drive_states the step read, the path's length
// left (the straight line before any path), the angle left, the estimate of
// the time left from that drive_states' speed, and no recovery on an open
// floor.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feedback_at_each_step_gives_the_stamped_pose_the_path_left_and_the_time_left() {
    let mut rig = Rig::start(Scene::open()).await;
    let target = [1.5, 0.0];
    let handle = rig.send_when_ready(&to(target)).await;
    let ran = rig.run_until_ended().await;
    let seen: Vec<(&StepSeen, &navigate_to_pose::Feedback)> = ran
        .iter()
        .filter_map(|step| step.navigation().map(|feedback| (step, feedback)))
        .collect();
    assert_eq!(
        seen.len() + 1,
        ran.len(),
        "feedback at each step but the last"
    );
    assert!(seen.len() > 20, "{} steps", seen.len());
    let mut distances = Vec::new();
    for (step, feedback) in &seen {
        let pose_record = &feedback.current_pose;
        assert_eq!(pose_record.frame, ODOMETRY_FRAME);
        assert!(
            ns(pose_record.timestamp) <= step.at_ns,
            "a drive_states of the step's time or before"
        );
        let (pose, velocity) = rig.world.reported_at(pose_record.timestamp);
        assert_eq!(pose_record.position, [pose.x, pose.y, 0.0]);
        assert_eq!(feedback.recoveries, 0);
        assert!(
            (0.0..=PI).contains(&feedback.angle_remaining_rad),
            "{feedback:?}"
        );
        assert!(
            feedback.distance_remaining_m >= pose.distance_to(target) - 1e-9,
            "a path is no shorter than the straight line: {feedback:?}"
        );
        let measured_m_s = velocity.linear_x().hypot(velocity.linear_y());
        let expected_s = if measured_m_s <= 0.01 || feedback.distance_remaining_m <= 0.1 {
            0.0
        } else {
            feedback.distance_remaining_m / measured_m_s
        };
        assert!(
            (feedback.estimated_time_remaining_s - expected_s).abs() < 1e-9,
            "{feedback:?} at {measured_m_s} m/s"
        );
        distances.push(feedback.distance_remaining_m);
    }
    let (_, first) = seen[0];
    let (start, _) = rig.world.reported_at(first.current_pose.timestamp);
    assert_eq!(
        first.distance_remaining_m,
        start.distance_to(target),
        "the straight line before the first path"
    );
    assert!(
        distances[0] > distances[distances.len() - 1],
        "{distances:?}"
    );
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

/// The forward speed the navigation commands once the base's origin passes
/// 0.5 m on its way to a goal 0.6 m before a wall 2.2 m out, for a base that
/// decelerates at `deceleration`, the controller's cost regulation reaching
/// 1 cm alone.
async fn speed_past_half_a_meter(deceleration: f64) -> f64 {
    let mut settings = params();
    settings.controller.cost_scaling_dist_m = 0.01;
    let wall = Box3::new([2.2, -2.4, 0.0], [2.3, 2.4, 0.4]);
    let mut rig = Rig::start_with_limits(
        settings,
        Scene::with(&[wall]),
        &[FRONT],
        decelerating(deceleration),
    )
    .await;
    let _navigation = rig.send_when_ready(&goal(1.6, 0.0)).await;
    rig.run_until(|base| base.odometry.x >= 0.5).await;
    let step = rig.run_on(2).await;
    rig.world.until_setpoint_from(step[1].at_ns).await;
    let speed = rig.world.setpoint_at(step[0].at_ns).map_or(0.0, forward);
    rig.shutdown().await;
    speed
}

// A describe_base answer whose bound exceeds time_before_collision_s
// lengthens the stop's projection to that bound: decelerating at 0.5, the
// 3 s the turn takes to rest plus the 0.5 s source timeout slow the base
// 1.5 m before the wall, where the 1 s of time_before_collision_s, which a
// base decelerating at 3 keeps, leaves it at speed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_describe_base_bound_above_time_before_collision_lengthens_the_stops_projection() {
    let long = speed_past_half_a_meter(0.5).await;
    let short = speed_past_half_a_meter(3.0).await;
    assert!(long < short, "{long} m/s against {short} m/s");
    assert!(short > 0.45, "{short} m/s");
}

// ---------------------------------------------------------------------------
// The 2-D lidar: the grid of the floor read from a lidar alone.
// ---------------------------------------------------------------------------

/// The rig with the lidar over the base's origin alone before `scene`, the
/// rig's base measuring its odometry.
async fn lidar_rig(scene: Scene) -> Rig {
    Rig::start_with_lidars(params(), scene, &[], &[LIDAR], measured()).await
}

/// The lidar, which describe_sensors names carried by the left arm.
const CARRIED_LIDAR: LidarMount = LidarMount {
    carried_by: "left_arm",
    ..LIDAR
};

/// A room's four walls, 1 m tall and 0.2 m thick, from -4 m to 6 m along x
/// and -4 m to 4 m along y: the base's start and [`BEHIND_THE_BOX`] lie inside
/// it, and every ray of [`LIDAR_LENS`] from the base on its way there meets a
/// wall within its 8 m range.
const ROOM_WALLS: [Box3; 4] = [
    Box3::new([-4.0, -4.0, 0.0], [6.0, -3.8, 1.0]),
    Box3::new([-4.0, 3.8, 0.0], [6.0, 4.0, 1.0]),
    Box3::new([-4.0, -4.0, 0.0], [-3.8, 4.0, 1.0]),
    Box3::new([5.8, -4.0, 0.0], [6.0, 4.0, 1.0]),
];

/// botshare's lidar, lidar_frame on lidar_joint of botshare_base.urdf: 0.33 m
/// ahead of the robot frame's origin, 0.222 m up, facing backward.
const BOTSHARE_LIDAR: LidarMount = LidarMount {
    name: "lidar",
    position: [0.33, 0.0, 0.222],
    yaw_rad: PI,
    lens: LIDAR_LENS,
    carried_by: "",
};

// base_navigation's sensors: a 2-D lidar alone feeds the grid, so in a room
// whose walls the lidar sees, a navigation drives the base around a box the
// lidar sees to a goal behind it, without contact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lidar_alone_feeds_the_grid_and_a_box_it_sees_is_driven_around() {
    let mut rig = lidar_rig(Scene::with(&[&ROOM_WALLS[..], &[BOX_AHEAD]].concat())).await;
    let handle = rig.send_when_ready(&to(BEHIND_THE_BOX)).await;
    rig.run_checking(|base| {
        assert!(
            !touches(base, &BOX_AHEAD),
            "the base touched the box: {:?}",
            base.truth
        )
    })
    .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result);
    let [x, y, _] = data.final_pose.position;
    assert!(
        (x - BEHIND_THE_BOX[0]).hypot(y - BEHIND_THE_BOX[1]) < 0.2,
        "{data:?}"
    );
    rig.shutdown().await;
}

// A goal while the lidar the grid reads is silent is refused with
// resource_unavailable, naming the lidar.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_the_lidar_is_silent_is_refused_resource_unavailable_naming_it() {
    let mut rig = lidar_rig(Scene::open()).await;
    rig.ready().await;
    rig.world.lidars_on = false;
    rig.steps(SILENT_STEPS).await;
    let refusal = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(&refusal.message, &["the lidar lidar", "source_timeout_s"]);
    rig.shutdown().await;
}

// A lidar that stops publishing fails the navigation in flight with
// resource_unavailable, naming the lidar.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lidar_that_stops_publishing_fails_the_navigation_resource_unavailable_naming_it() {
    let mut rig = lidar_rig(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.1).await;
    rig.world.lidars_on = false;
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["the lidar lidar"]);
    rig.shutdown().await;
}

// sensor_mounts' frames: a scan whose header names a frame other than the one
// describe_sensors gives the lidar is dropped, and a goal waits for the lidar,
// the refusal naming both frames; once its scans name its own frame, the
// navigator takes goals.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scan_naming_another_frame_is_dropped_and_the_wait_says_why() {
    let mut rig = lidar_rig(Scene::open()).await;
    rig.world.scan_frame = Some("rear_lidar_frame");
    let refusal = rig
        .refusal_once(&goal(1.0, 0.0), |refusal| {
            refusal.message.contains("rear_lidar_frame")
        })
        .await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(
        &refusal.message,
        &[
            "the lidar lidar",
            "its scans arrive and are dropped",
            "\"rear_lidar_frame\"",
            "\"lidar_frame\"",
        ],
    );
    rig.world.scan_frame = None;
    rig.ready().await;
    rig.shutdown().await;
}

// base_navigation's sensors: a lidar that describe_sensors names carried by
// a part is refused, and a goal waits for it, the refusal naming the lidar,
// its carrier and what to run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lidar_a_part_carries_is_refused_naming_its_carrier() {
    let mut rig =
        Rig::start_with_lidars(params(), Scene::open(), &[], &[CARRIED_LIDAR], measured()).await;
    let refusal = rig
        .refusal_once(&goal(1.0, 0.0), |refusal| {
            refusal.message.contains("carries")
        })
        .await;
    assert_eq!(
        refusal.kind,
        FailureKind::ResourceUnavailable,
        "{refusal:?}"
    );
    assert_names(
        &refusal.message,
        &[
            "the lidar lidar",
            "\"left_arm\" carries",
            "a lidar fixed to the base",
        ],
    );
    rig.shutdown().await;
}

// base_navigation's grid, read from botshare's lidar alone: botshare's base
// turns at a goal near the first grid's edge as
// `turn_at_a_goal_near_the_grids_first_edge` gives it, its padded
// footprint's corners leaving the square the grid held when the goal was
// accepted: the grid centers on the base with each scan while the goal runs,
// where the base stood at the scan's first ray.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lidar_only_grid_rolls_with_the_base_while_a_goal_runs() {
    let rig = Rig::start_with_lidars(
        params(),
        Scene::open(),
        &[],
        &[BOTSHARE_LIDAR],
        botshare_answer(),
    )
    .await;
    turn_at_a_goal_near_the_grids_first_edge(rig).await;
}

// ---------------------------------------------------------------------------
// Recoveries, the follow's retry, and the near-goal path.
// ---------------------------------------------------------------------------

/// The feedback's recoveries and the setpoint of each step a navigation ran
/// on, the step that ended it left out, every setpoint stamped at or before
/// the last of `steps` having arrived.
fn by_recoveries(rig: &Rig, steps: &[StepSeen]) -> Vec<(u32, Option<Setpoint>)> {
    steps
        .iter()
        .filter_map(|step| {
            step.navigation().map(|feedback| {
                (
                    feedback.recoveries,
                    rig.world.setpoint_at(step.at_ns).cloned(),
                )
            })
        })
        .collect()
}

/// Step the navigation in flight until it ends, then take every setpoint
/// through the step that ended it, the first of its release burst: the
/// steps.
async fn run_to_its_burst(rig: &mut Rig) -> Vec<StepSeen> {
    let steps = rig.run_until_ended().await;
    rig.world.until_setpoint_from(ended_ns(&steps)).await;
    steps
}

/// `counted` as runs of the recoveries read and the setpoint streamed,
/// `count:forward/turn` or `count:none`, each with how many steps it held,
/// for a failing assertion's message.
fn timeline(counted: &[(u32, Option<Setpoint>)]) -> String {
    let label = |(count, setpoint): &(u32, Option<Setpoint>)| match setpoint {
        Some(s) => format!("{count}:{:.3}/{:.3}", forward(s), turn(s)),
        None => format!("{count}:none"),
    };
    let mut runs: Vec<(String, usize)> = Vec::new();
    for step in counted {
        let step = label(step);
        match runs.last_mut() {
            Some((last, held)) if *last == step => *held += 1,
            _ => runs.push((step, 1)),
        }
    }
    runs.iter()
        .map(|(step, held)| format!("{step} x{held}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The setpoints `counted` streamed while the feedback's recoveries was
/// `count`.
fn while_counting(counted: &[(u32, Option<Setpoint>)], count: u32) -> Vec<Setpoint> {
    counted
        .iter()
        .filter(|(at, _)| *at == count)
        .filter_map(|(_, setpoint)| setpoint.clone())
        .collect()
}

/// The values the feedback's recoveries took, in order, each once.
fn counts_taken(counted: &[(u32, Option<Setpoint>)]) -> Vec<u32> {
    let mut counts: Vec<u32> = counted.iter().map(|(count, _)| *count).collect();
    assert!(
        counts.windows(2).all(|pair| pair[0] <= pair[1]),
        "the recoveries only grow: {}",
        timeline(counted)
    );
    counts.dedup();
    counts
}

/// Asserts that the recovery wait of `counted`, which starts at the first
/// step whose recoveries read `count`, streams zero velocity at each step for
/// its 5 s: the try after the wait follows under the same count.
fn assert_the_wait(counted: &[(u32, Option<Setpoint>)], count: u32) {
    let waited = counted
        .iter()
        .skip_while(|(at, _)| *at != count)
        .take_while(|(at, setpoint)| *at == count && setpoint.as_ref().is_some_and(is_zero))
        .count();
    assert!(
        waited >= helpers::mock_base::steps_in(5.0),
        "the wait streams zero for its 5 s, one setpoint a step, {waited} steps: {}",
        timeline(counted)
    );
}

// A navigation whose plans find no path runs the clearing, the turn, the
// wait and the back-up, in base_navigation's order, once each, with a try
// after each of the first three, then fails with no_path. The feedback's
// recoveries counts each recovery and each retry's clearing: 1 the first
// plan's retry, 2 the clearing, 3 the retry of the try after it, 4 the turn,
// 5 the retry of the try after it, 6 the wait, 7 the retry of the try after
// it, 8 the back-up. The tries before the turn stream nothing, and no
// setpoint has a linear_velocity x below 0 before the back-up. A camera on
// each edge sees the pen whatever the base's heading.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plans_that_find_no_path_run_each_recovery_once_in_order_then_fail_no_path() {
    let mut rig = Rig::start_with(params(), Scene::with(&pen()), &RING, measured()).await;
    let handle = rig.send_when_ready(&to(PEN_GOAL)).await;
    let steps = run_to_its_burst(&mut rig).await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(failure.kind, FailureKind::NoPath, "{failure:?}");
    assert_names(
        &failure.message,
        &["after the clearing, the turn, the wait and the back-up"],
    );
    let counted = by_recoveries(&rig, &steps);
    assert_eq!(counts_taken(&counted), (0..=8).collect::<Vec<u32>>());
    let before_the_turn: Vec<Setpoint> = (0..4)
        .flat_map(|count| while_counting(&counted, count))
        .collect();
    assert!(
        before_the_turn.is_empty(),
        "the tries plan: {before_the_turn:?}"
    );
    assert!(
        while_counting(&counted, 4)
            .iter()
            .any(|s| turns_in_place(s) && turn(s) > 0.0),
        "the turn turns the base in place counterclockwise"
    );
    assert_the_wait(&counted, 6);
    let before_the_back_up: Vec<Setpoint> = (0..8)
        .flat_map(|count| while_counting(&counted, count))
        .collect();
    assert!(
        before_the_back_up.iter().all(|s| forward(s) >= 0.0),
        "only the back-up moves the base backward"
    );
    assert!(
        while_counting(&counted, 8).iter().any(|s| forward(s) < 0.0),
        "the back-up drives backward"
    );
    rig.shutdown().await;
}

// navigate_to_pose's max_speed_m_s caps the back-up recovery too: a
// navigation into the pen with max_speed_m_s 0.1, below the back-up speed of
// tests/parameters.json, 0.15 m/s, reaches its back-up, which drives backward,
// and no setpoint's linear_velocity has a norm above 0.1 m/s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_speed_caps_the_back_up_recovery() {
    let mut rig = Rig::start_with(params(), Scene::with(&pen()), &RING, measured()).await;
    let slow = GoalRequestData {
        max_speed_m_s: 0.1,
        ..to(PEN_GOAL)
    };
    let handle = rig.send_when_ready(&slow).await;
    let steps = run_to_its_burst(&mut rig).await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(failure.kind, FailureKind::NoPath, "{failure:?}");
    let counted = by_recoveries(&rig, &steps);
    assert!(
        while_counting(&counted, 8).iter().any(|s| forward(s) < 0.0),
        "the back-up drives backward: {}",
        timeline(&counted)
    );
    let fastest = counted
        .iter()
        .filter_map(|(_, setpoint)| setpoint.as_ref())
        .map(speed)
        .fold(0.0, f64::max);
    assert!(fastest <= 0.1, "{fastest} m/s: {}", timeline(&counted));
    rig.shutdown().await;
}

// base_navigation's wait: a base that executes no motion while its path
// stays open fails the navigation with blocked after the recoveries: each
// follow makes no progress, and the turn and the back-up run out of their
// time allowances, 1 s here. The recoveries read 1 for the first follow's
// retry, 2 the clearing, 3 its try's retry, 4 the turn, which fails and
// passes straight to the wait, 5, then 6 the retry of the try after the
// wait, 7 the back-up. The wait streams zero for its 5 s and the navigation
// goes on after it, also on a base that never measures rest.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_base_that_executes_no_motion_fails_blocked_after_the_recoveries() {
    let quick = || {
        let mut quick = params();
        quick.recoveries.turn_time_allowance_s = 1.0;
        quick.recoveries.back_up_time_allowance_s = 1.0;
        quick
    };
    for mode in [ModeReport::ByRest, ModeReport::Fixed("moving")] {
        let mut rig = Rig::start_with(quick(), Scene::open(), &[FRONT], measured()).await;
        rig.world.base.executes = false;
        rig.world.base.mode = mode;
        let handle = rig.send_when_ready(&goal(1.5, 0.0)).await;
        let steps = run_to_its_burst(&mut rig).await;
        let result = outcome::navigate_to_pose::result_of(&handle).await;
        let failure = outcome::navigate_to_pose::failure(&result);
        assert_eq!(failure.kind, FailureKind::Blocked, "{mode:?}: {failure:?}");
        let counted = by_recoveries(&rig, &steps);
        assert_eq!(
            counts_taken(&counted),
            (0..=7).collect::<Vec<u32>>(),
            "{mode:?}"
        );
        assert_the_wait(&counted, 5);
        let before_the_back_up: Vec<Setpoint> = (0..7)
            .flat_map(|count| while_counting(&counted, count))
            .collect();
        assert!(
            before_the_back_up.iter().all(|s| forward(s) >= 0.0),
            "{mode:?}: no setpoint has a linear_velocity x below 0 before the back-up"
        );
        assert!(
            while_counting(&counted, 7).iter().any(|s| forward(s) < 0.0),
            "{mode:?}: the back-up commands backward"
        );
        rig.shutdown().await;
    }
}

// After each recovery a try runs: a pen that opens during the wait is
// driven into by the try after the wait, and the navigation completes with
// the recoveries it ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_try_after_the_wait_drives_into_a_pen_that_opened_during_it() {
    let mut rig = Rig::start_with(params(), Scene::with(&pen()), &RING, measured()).await;
    let handle = rig.send_when_ready(&to(PEN_GOAL)).await;
    rig.run_until_feedback(|feedback| feedback.recoveries == 6)
        .await;
    rig.world.scene = Scene::with(&pen()[1..]);
    let steps = run_to_its_burst(&mut rig).await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let data = outcome::navigate_to_pose::completed(&result);
    let [x, y, _] = data.final_pose.position;
    assert!((x - PEN_GOAL[0]).hypot(y - PEN_GOAL[1]) < 0.2, "{data:?}");
    let last = by_recoveries(&rig, &steps).last().map(|(count, _)| *count);
    assert_eq!(last, Some(6), "no recovery after the wait");
    rig.shutdown().await;
}

/// A box just behind the back edge, left of the center line: 0.03 m past the
/// padded footprint, 0.26 m from the robot frame's origin, outside the padded
/// footprint's inscribed radius, in cells the padded footprint's back left
/// corner sweeps once the base has turned about 0.5 rad counterclockwise, and
/// in the back camera's view, which hides nothing of the pen ahead.
const BOX_AT_THE_BACK_EDGE: Box3 = Box3::new([-0.31, 0.0, 0.0], [-0.26, 0.06, 0.4]);

// A turn recovery beside a box fails when its sweep meets the box, ending
// short of its angle, and the wait follows at once: the recoveries read 4 for
// the turn and 5 for the wait, with no try between, then the try after the
// wait and its retry, 6. The back-up, which the box behind the base blocks at
// its first step, ends the navigation in the step it starts, with no_path, as
// no follow ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_turn_recovery_whose_sweep_meets_a_box_fails_and_the_wait_follows() {
    let mut boxes = pen();
    boxes.push(BOX_AT_THE_BACK_EDGE);
    let mut rig = Rig::start_with(params(), Scene::with(&boxes), &RING, measured()).await;
    let handle = rig.send_when_ready(&to(PEN_GOAL)).await;
    let steps = run_to_its_burst(&mut rig).await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let counted = by_recoveries(&rig, &steps);
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::NoPath,
        "{failure:?}: {}",
        timeline(&counted)
    );
    assert_names(&failure.message, &["the back-up"]);
    assert_eq!(
        counts_taken(&counted),
        (0..=6).collect::<Vec<u32>>(),
        "{}",
        timeline(&counted)
    );
    let wait_starts = steps
        .iter()
        .filter_map(StepSeen::navigation)
        .find(|feedback| feedback.recoveries == 5)
        .expect("the wait runs");
    let turned = yaw_of(wait_starts.current_pose.orientation);
    assert!(
        turned < 0.6,
        "the turn ends short of its 1.57 rad: {turned} rad"
    );
    assert_the_wait(&counted, 5);
    rig.shutdown().await;
}

// base_navigation's follow: a wall that appears 0.05 m before the base's front
// edge once its first plan, straight ahead, was asked leaves the controller no
// command that keeps the padded footprint off it along its arc. Once the plan
// arrives, the follow streams zero velocity for the failure tolerance, 0.3 s,
// then fails, empties the grid and follows the path once more on the readings
// after the clearing, the recoveries reading 1; that follow fails the same
// way, and the clearing recovery runs, 2. The base never touches the wall.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_follow_with_no_clear_command_fails_after_its_tolerance_then_follows_once_more() {
    let mut rig = Rig::start_held(Scene::open()).await;
    rig.world.release_plans(1);
    let _handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    let planned = rig.run_on(1).await;
    let wall = wall_ahead_of(&rig.world.base, 0.05);
    rig.world.scene.boxes.push(wall);
    let mut steps = planned;
    for _ in 0..DRIVE_STEPS {
        let step = rig.step().await;
        assert!(!step.goal_ended, "the navigation goes on");
        assert!(
            !touches(&rig.world.base, &wall),
            "{:?}",
            rig.world.base.truth
        );
        let count = step
            .navigation()
            .expect("the navigation's feedback")
            .recoveries;
        steps.push(step);
        if count == 2 {
            break;
        }
    }
    rig.fence().await;
    let counted = by_recoveries(&rig, &steps);
    assert_eq!(
        counts_taken(&counted),
        vec![0, 1, 2],
        "{}",
        timeline(&counted)
    );
    let followed: Vec<&(u32, Option<Setpoint>)> = counted
        .iter()
        .skip_while(|(_, setpoint)| setpoint.is_none())
        .filter(|(count, _)| *count < 2)
        .collect();
    assert!(
        followed
            .iter()
            .all(|(_, setpoint)| setpoint.as_ref().is_some_and(is_zero)),
        "zero velocity at each step of the follow and its retry: {}",
        timeline(&counted)
    );
    let tolerance_steps = helpers::mock_base::steps_in(0.3);
    for count in [0, 1] {
        let zero_run = followed.iter().filter(|(at, _)| *at == count).count();
        assert!(
            zero_run > tolerance_steps,
            "follow {count} streams zero past the failure tolerance: {}",
            timeline(&counted)
        );
    }
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// The controller and the stop run apart from the planner.
// ---------------------------------------------------------------------------

// While the harness holds a plan unfinished, the controller and the stop
// send a setpoint at each 50 ms of the clock, the feedback goes out at each
// step, and a wall that the depth frames show ahead of the base slows it and
// stops it before contact, the plan still held. The first plan runs; the
// replan due a planning period into the follow is the one held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn while_a_plan_is_held_each_step_sends_a_setpoint_and_a_wall_ahead_stops_the_base() {
    let mut rig = Rig::start_held(Scene::open()).await;
    rig.world.release_plans(1);
    let _handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    let mut cruise = rig.run_until(|base| base.velocity.linear_x() > 0.2).await;
    cruise.extend(rig.run_on(helpers::mock_base::steps_in(1.25)).await);
    let cruise_m_s = setpoints(&rig, &cruise)
        .iter()
        .map(forward)
        .fold(0.0, f64::max);
    let wall = wall_ahead_of(&rig.world.base, 0.8);
    rig.world.scene.boxes.push(wall);
    let mut speeds = Vec::new();
    let mut last_ns = rig.world.node_ns(rig.world.now_ns);
    for _ in 0..helpers::mock_base::steps_in(3.0) {
        let step = rig.step().await;
        assert!(!step.goal_ended, "the navigation goes on");
        assert!(step.navigation().is_some(), "feedback at each step");
        assert_eq!(
            step.at_ns - last_ns,
            STEP_NS,
            "a step at each 50 ms of the clock"
        );
        last_ns = step.at_ns;
        rig.world.until_setpoint_from(step.at_ns).await;
        let setpoint = rig
            .world
            .setpoint_at(step.at_ns)
            .expect("a setpoint at each step while the plan is held");
        speeds.push(forward(setpoint));
        assert!(
            !touches(&rig.world.base, &wall),
            "{:?}",
            rig.world.base.truth
        );
        if rig.world.base.velocity.linear_x() == 0.0 {
            break;
        }
    }
    assert_eq!(
        rig.world.base.velocity.linear_x(),
        0.0,
        "the base stops before the wall: {speeds:?}"
    );
    assert!(
        speeds
            .iter()
            .any(|&speed| speed > 0.0 && speed < cruise_m_s),
        "the base slows: {speeds:?}"
    );
    rig.shutdown().await;
}

// While the harness holds the first plan of a navigation, each step gives
// the feedback, the straight line to the goal, and streams nothing, as a
// navigation streams nothing before its first command; once the plan is let
// go, the navigation streams its first command, and no setpoint came before
// it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn while_the_first_plan_is_held_each_step_gives_feedback_and_streams_nothing() {
    let mut rig = Rig::start_held(Scene::open()).await;
    let _handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    let held = rig.run_on(helpers::mock_base::steps_in(1.5)).await;
    for step in &held {
        let feedback = step.navigation().expect("feedback at each step");
        assert_eq!(feedback.distance_remaining_m, 2.0, "the straight line");
    }
    rig.world.release_plans(1);
    let released_ns = rig.world.node_ns(rig.world.now_ns);
    let mut first_ns = None;
    for _ in 0..READY_STEPS {
        let step = rig.run_on(1).await;
        if let Some((at_ns, _)) = rig.world.setpoints_in(..).first() {
            first_ns = Some(*at_ns);
            break;
        }
        assert!(step[0].navigation().is_some(), "feedback at each step");
    }
    let first_ns = first_ns.expect("the first command once the plan finished");
    assert!(
        first_ns > released_ns,
        "the first setpoint at {first_ns} ns"
    );
    assert_streams_nothing(
        &rig.world.setpoints_in(held[0].at_ns..first_ns),
        "while the first plan is held",
    );
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// clear_grid.
// ---------------------------------------------------------------------------

// clear_grid empties the grid, so a goal at a box it held plans and moves the
// base; the box the camera sees again is marked again, and the navigation
// then fails with unreachable. The source timeout of 2.5 s keeps the dark
// camera from falling silent while the plan runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clear_grid_empties_the_grid_and_a_box_the_camera_sees_again_is_marked_again() {
    let mut dark = params();
    dark.source_timeout_s = 2.5;
    let mut rig = Rig::start_with(dark, Scene::with(&[BOX_AHEAD]), &[FRONT], measured()).await;
    let first = rig.send_when_ready(&to(AT_THE_BOX)).await;
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&first).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable,
        "the grid holds the box"
    );
    rig.world.cameras_on = false;
    rig.steps(2).await;
    rig.clear_grid().await;
    let second = rig.send(&to(AT_THE_BOX)).await;
    assert!(second.accepted, "{:?}", second.refusal);
    rig.run_until(|base| base.velocity.linear_x() > 0.0).await;
    rig.world.cameras_on = true;
    rig.run_checking(|base| assert!(!touches(base, &BOX_AHEAD), "{:?}", base.truth))
        .await;
    let result = outcome::navigate_to_pose::result_of(&second).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable,
        "the box is marked again"
    );
    rig.shutdown().await;
}

/// One step of the navigation in flight, which goes on, the base clear of
/// `obstacle`: the step's feedback.
async fn step_clear_of(rig: &mut Rig, obstacle: &Box3) -> navigate_to_pose::Feedback {
    let step = rig.run_on(1).await.remove(0);
    assert!(
        !touches(&rig.world.base, obstacle),
        "{:?}",
        rig.world.base.truth
    );
    step.navigation()
        .expect("a navigation's feedback at each step")
        .clone()
}

/// How much longer (m) than the straight way to [`BEHIND_THE_BOX`] a path
/// around [`BOX_AHEAD`], 0.6 m wide, is at least from where the base stands
/// in the clear_grid test, its first path and the one after the box is seen
/// again; a path through where the box stands is longer by less, its cells
/// along a line askew to the grid.
const AROUND_THE_BOX_M: f64 = 0.4;

/// How much longer than the straight way to [`BEHIND_THE_BOX`] from where
/// the drive_states its step read placed the robot frame the path of
/// `feedback` is (m).
fn detour(feedback: &navigate_to_pose::Feedback) -> f64 {
    let [x, y, _] = feedback.current_pose.position;
    let straight = (x - BEHIND_THE_BOX[0]).hypot(y - BEHIND_THE_BOX[1]);
    feedback.distance_remaining_m - straight
}

// clear_grid during a navigation empties the grid: the next plan, the camera
// dark, runs straight through where the box stands, while the stop still reads
// the camera's newest frame; once the camera sees the box again it is marked
// again, the next plan goes around it, and the navigation completes without
// contact. The source timeout of 2.5 s keeps the dark camera from falling
// silent within a planning period and the plan after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clear_grid_during_a_navigation_plans_on_the_emptied_grid_until_the_box_is_seen_again() {
    let mut dark = params();
    dark.source_timeout_s = 2.5;
    let mut rig = Rig::start_with(dark, Scene::with(&[BOX_AHEAD]), &[FRONT], measured()).await;
    let handle = rig.send_when_ready(&to(BEHIND_THE_BOX)).await;
    let mut around = false;
    for _ in 0..helpers::mock_base::steps_in(2.0) {
        let feedback = step_clear_of(&mut rig, &BOX_AHEAD).await;
        if detour(&feedback) > AROUND_THE_BOX_M {
            around = true;
            break;
        }
    }
    assert!(around, "the first path goes around the box");
    rig.world.cameras_on = false;
    rig.clear_grid().await;
    let mut straight = false;
    for _ in 0..helpers::mock_base::steps_in(2.0) {
        let feedback = step_clear_of(&mut rig, &BOX_AHEAD).await;
        if detour(&feedback) < AROUND_THE_BOX_M {
            straight = true;
            break;
        }
    }
    assert!(
        straight,
        "a plan on the emptied grid runs straight through the box"
    );
    rig.world.cameras_on = true;
    let mut around_again = false;
    for _ in 0..helpers::mock_base::steps_in(2.0) {
        let feedback = step_clear_of(&mut rig, &BOX_AHEAD).await;
        if detour(&feedback) > AROUND_THE_BOX_M {
            around_again = true;
            break;
        }
    }
    assert!(
        around_again,
        "the box seen again is marked again, and the path goes around it"
    );
    rig.run_checking(|base| assert!(!touches(base, &BOX_AHEAD), "{:?}", base.truth))
        .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// How a goal ends: the replacement, the cancel, the holds, the silences and
// the release burst.
// ---------------------------------------------------------------------------

/// The stamp of the first setpoint at or after `from_ns` that is no zero,
/// stepping the goal in flight until one arrives.
async fn first_command_from(rig: &mut Rig, from_ns: i64) -> i64 {
    for _ in 0..DRIVE_STEPS {
        let command = rig
            .world
            .setpoints_in(from_ns..)
            .into_iter()
            .find(|(_, setpoint)| !is_zero(setpoint));
        if let Some((at_ns, _)) = command {
            return at_ns;
        }
        rig.run_on(1).await;
    }
    panic!("no command in {DRIVE_STEPS} steps");
}

// A second goal replaces a running navigation: the replaced navigation ends
// cancelled, its reason saying a new goal replaced it and its result where a
// drive_states the node took by then placed the robot frame, at that
// drive_states' timestamp; its burst streams zero velocity while the new goal
// plans, and the new goal's first command ends it, the base driving on into
// the new goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_goal_replaces_a_running_navigation_and_its_first_command_ends_the_burst() {
    let mut rig = Rig::start(Scene::open()).await;
    let first = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let replaced_ns = rig.world.node_ns(rig.world.now_ns);
    let second = rig.send(&goal(2.0, 0.6)).await;
    assert!(second.accepted, "{:?}", second.refusal);
    let replaced = outcome::navigate_to_pose::result_of(&first).await;
    let (reason, data) = outcome::navigate_to_pose::cancelled(&replaced);
    assert_names(reason, &["a new goal replaced"]);
    assert!(ns(data.final_pose.timestamp) <= replaced_ns, "{data:?}");
    let (at, _) = rig.world.reported_at(data.final_pose.timestamp);
    assert_eq!(data.final_pose.position, [at.x, at.y, 0.0]);
    let command_ns = first_command_from(&mut rig, replaced_ns + 1).await;
    let burst = rig.world.setpoints_in(replaced_ns + 1..command_ns);
    assert!(
        !burst.is_empty() && burst.iter().all(|(_, setpoint)| is_zero(setpoint)),
        "the replaced goal's burst streams while the new goal plans: {burst:?}"
    );
    let driving = rig.run_on(10).await;
    rig.world.until_setpoint_from(ended_ns(&driving)).await;
    let commands = rig.world.setpoints_in(command_ns..);
    assert!(
        commands.iter().all(|(_, setpoint)| !is_zero(setpoint)),
        "the new goal's commands end the burst: {commands:?}"
    );
    assert!(
        rig.world.base.velocity.linear_x() > 0.0,
        "the base drives into the new goal"
    );
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&second).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

/// A navigation that `hold`, a follow_state other than following, ends: it
/// ends cancelled, its reason naming the hold and its result where the
/// drive_states that first reported the hold placed the robot frame, at its
/// timestamp; the ending brings no release burst, and the navigator streams
/// nothing and refuses a goal while the hold lasts, then accepts one once the
/// base follows again.
async fn a_hold_cancels_the_navigation(hold: FollowState) {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base.follow_state = hold;
    let held = rig.step().await;
    let (first_report, first_ns) = (rig.world.base.odometry, rig.world.now_ns);
    if !held.goal_ended {
        rig.run_until_ended().await;
    }
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let (reason, data) = outcome::navigate_to_pose::cancelled(&result);
    assert_names(reason, &[hold.as_str()]);
    assert_eq!(
        data.final_pose.position,
        [first_report.x, first_report.y, 0.0]
    );
    assert_eq!(ns(data.final_pose.timestamp), first_ns);
    rig.steps(10).await;
    let refused = rig.refusal(&goal(1.0, 0.0)).await;
    assert_eq!(refused.kind, FailureKind::PreconditionFailed, "{refused:?}");
    rig.world.base.follow_state = FollowState::Following;
    rig.step().await;
    let next = rig.send_when_ready(&goal(1.0, 0.0)).await;
    assert!(next.accepted, "{:?}", next.refusal);
    let resumed_ns = first_command_from(&mut rig, held.at_ns).await;
    assert_streams_nothing(
        &rig.world.setpoints_in(held.at_ns..resumed_ns),
        &format!("{hold:?}, from its first report until the next navigation streams"),
    );
    rig.shutdown().await;
}

// A robot stop ends the navigation cancelled, its result where
// drive_states first reported it, and the navigator streams no setpoint
// until follow_state is following again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_stop_cancels_the_navigation_and_nothing_streams_until_the_base_follows() {
    a_hold_cancels_the_navigation(FollowState::RobotStop).await;
}

// base_navigation's holds: a goal of the base's node, follower_goal, ends the
// navigation the same way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_follower_goal_cancels_the_navigation_and_nothing_streams_until_the_base_follows() {
    a_hold_cancels_the_navigation(FollowState::FollowerGoal).await;
}

// A cancel during a navigation ends it cancelled at once, its result where
// the newest drive_states placed the robot frame, at its timestamp; the
// release burst follows, then nothing, and the base comes to rest.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancel_ends_the_navigation_at_once_then_the_release_burst_and_nothing() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    outcome::navigate_to_pose::cancel(&handle).await;
    let ended = rig.step().await;
    assert!(
        ended.goal_ended,
        "the cancel ends the navigation at the next step"
    );
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let (_, data) = outcome::navigate_to_pose::cancelled(&result);
    assert!(ns(data.final_pose.timestamp) <= ended.at_ns, "{data:?}");
    let (at, _) = rig.world.reported_at(data.final_pose.timestamp);
    assert_eq!(data.final_pose.position, [at.x, at.y, 0.0]);
    let after = rig.burst_after(ended.at_ns).await;
    assert_release_burst(ended.at_ns, &after);
    rig.shutdown().await;
}

// base_navigation's order of endings: a cancel comes before a new goal's
// acceptance, so a navigation whose caller cancelled it before a new goal
// arrived ends cancelled by that cancel, its result where the newest
// drive_states placed the robot frame, and the new goal replaces no goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancel_before_a_new_goal_ends_the_goal_by_the_cancel() {
    let mut rig = Rig::start(Scene::open()).await;
    let first = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let cancel_ns = rig.world.now_ns;
    outcome::navigate_to_pose::cancel(&first).await;
    let second = rig.send(&goal(1.0, 0.0)).await;
    assert!(second.accepted, "{:?}", second.refusal);
    let result = outcome::navigate_to_pose::result_of(&first).await;
    let (reason, data) = outcome::navigate_to_pose::cancelled(&result);
    assert!(!reason.contains("a new goal replaced it"), "{reason}");
    assert!(ns(data.final_pose.timestamp) <= cancel_ns, "{data:?}");
    let (at, _) = rig.world.reported_at(data.final_pose.timestamp);
    assert_eq!(data.final_pose.position, [at.x, at.y, 0.0]);
    rig.shutdown().await;
}

/// A navigation driving at speed, cancelled: the step that ended it, the
/// first of its release burst.
async fn cancelled_at_speed(rig: &mut Rig, to: [f64; 2]) -> StepSeen {
    let handle = rig.send_when_ready(&goal(to[0], to[1])).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    outcome::navigate_to_pose::cancel(&handle).await;
    let ended = rig.step().await;
    assert!(ended.goal_ended, "the cancel ends the navigation");
    rig.world.until_setpoint_from(ended.at_ns).await;
    ended
}

// base_navigation's release burst: a drive_states that reports robot_stop
// during the burst ends it, short of its 3 setpoints over 0.25 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_stop_report_during_the_release_burst_ends_it() {
    let mut rig = Rig::start(Scene::open()).await;
    let ended = cancelled_at_speed(&mut rig, [2.0, 0.0]).await;
    rig.world.base.follow_state = FollowState::RobotStop;
    rig.steps(6).await;
    rig.world.base.follow_state = FollowState::Following;
    rig.steps(2).await;
    let fence_ns = rig.fence().await;
    let burst = rig.world.setpoints_in(ended.at_ns..fence_ns);
    let span_ns = burst.last().map_or(0, |(at_ns, _)| at_ns - ended.at_ns);
    assert!(
        burst.iter().all(|(_, setpoint)| is_zero(setpoint))
            && (burst.len() < 3 || span_ns < 250_000_000),
        "the robot stop's report ends the burst: {burst:?}"
    );
    rig.shutdown().await;
}

// base_navigation's release burst: a goal accepted during the burst ends it
// at its first setpoint, and the burst goes on while the goal plans.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_accepted_during_the_burst_ends_it_at_its_first_setpoint() {
    let mut rig = Rig::start(Scene::open()).await;
    let ended = cancelled_at_speed(&mut rig, [2.0, 0.0]).await;
    let second = rig.send(&goal(1.5, 0.3)).await;
    assert!(second.accepted, "{:?}", second.refusal);
    let command_ns = first_command_from(&mut rig, ended.at_ns).await;
    let burst = rig.world.setpoints_in(ended.at_ns..command_ns);
    assert!(
        burst.len() >= 2 && burst.iter().all(|(_, setpoint)| is_zero(setpoint)),
        "the burst goes on while the new goal plans: {burst:?}"
    );
    let following = rig.run_on(5).await;
    rig.world.until_setpoint_from(ended_ns(&following)).await;
    let commands = rig.world.setpoints_in(command_ns..);
    assert!(
        commands.iter().all(|(_, setpoint)| !is_zero(setpoint)),
        "the new goal's setpoints, the burst over: {commands:?}"
    );
    rig.shutdown().await;
}

// A replacing goal whose first plan fails starts the release burst again
// from its failure: the replaced navigation ends cancelled and its burst
// streams while the new one plans; the new one fails with unreachable before
// its own first setpoint, and the burst starts again from that failure,
// 0.25 s and at least 3 setpoints, then nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replacing_goal_whose_first_plan_fails_starts_the_burst_again() {
    let side = Box3::new([1.5, 0.5, 0.0], [1.8, 0.8, 0.4]);
    let mut rig = Rig::start(Scene::with(&[side])).await;
    let first = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let second = rig.send(&goal(1.65, 0.65)).await;
    assert!(second.accepted, "{:?}", second.refusal);
    let replaced = outcome::navigate_to_pose::result_of(&first).await;
    outcome::navigate_to_pose::cancelled(&replaced);
    let (ran, after) = ending(&mut rig).await;
    let result = outcome::navigate_to_pose::result_of(&second).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable
    );
    let planning = setpoints(&rig, &ran[..ran.len() - 1]);
    assert!(
        !planning.is_empty() && planning.iter().all(is_zero),
        "the burst streams while the replacing goal plans: {planning:?}"
    );
    assert_release_burst(ended_ns(&ran), &after);
    rig.shutdown().await;
}

// base_navigation's replacement: a goal accepted after a burst while the base
// still moves counts as before the base moves until its first setpoint, so
// its goal in collision fails it with unreachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_accepted_while_the_base_still_moves_counts_as_before_the_base_moves() {
    let mut rig = Rig::start(Scene::with(&[BOX_AHEAD])).await;
    cancelled_at_speed(&mut rig, [1.8, -1.2]).await;
    rig.steps(6).await;
    assert!(
        !rig.world.base.rests(),
        "the base still moves after the burst"
    );
    let second = rig.send(&to(AT_THE_BOX)).await;
    assert!(second.accepted, "{:?}", second.refusal);
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&second).await;
    assert_eq!(
        outcome::navigate_to_pose::failure(&result).kind,
        FailureKind::Unreachable
    );
    rig.shutdown().await;
}

// A mock base whose drive_states fall silent fails the navigation with
// resource_unavailable, the navigator's setpoints falling to 0: the newest
// drive_states reported following and the navigator's stream was live, so
// the release burst follows the failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drive_states_that_fall_silent_fail_the_navigation_and_the_burst_follows() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base_reports = false;
    let ran = rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["sent no drive_states"]);
    rig.world.base_reports = true;
    let after = rig.burst_after(ended_ns(&ran)).await;
    assert_release_burst(ended_ns(&ran), &after);
    rig.shutdown().await;
}

// A mock base whose odometry_start_timestamp changes fails the navigation
// with resource_unavailable, the setpoints falling to 0 in the burst, and
// describe_base is read again: a goal waits for that read and is refused
// while it does not answer, then accepted once it does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_new_odometry_start_fails_the_navigation_and_describe_base_is_read_again() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let restart_ns = rig.world.now_ns + STEP_NS;
    rig.world.base.start_odometry_over(restart_ns);
    let ran = rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["odometry started over"]);
    let unread = rig.refusal_stepping(&goal(1.0, 0.0)).await;
    assert_eq!(unread.kind, FailureKind::ResourceUnavailable, "{unread:?}");
    assert_names(&unread.message, &["describe_base did not answer"]);
    within(
        "the read of the new start",
        rig.world
            .base_state
            .describe_base
            .next_request(ANSWER_BUDGET),
    )
    .await
    .expect("the node asks describe_base for the new start")
    .respond(measured())
    .await
    .expect("the answer goes");
    let fence_ns = rig.fence().await;
    assert_release_burst(
        ended_ns(&ran),
        &rig.world.setpoints_in(ended_ns(&ran)..fence_ns),
    );
    rig.shutdown().await;
}

/// A goal ended at `ended_ns` by a follow_state other than following, which
/// then ended: nothing streamed from that step until the fence after it.
async fn assert_no_burst_after(rig: &mut Rig, ended_ns: i64) {
    rig.world.base.follow_state = FollowState::Following;
    rig.steps(2).await;
    let fence_ns = rig.fence().await;
    assert_streams_nothing(
        &rig.world.setpoints_in(ended_ns..fence_ns),
        "from the hold's ending",
    );
}

// drive_states that report base_silence fail a navigation, and a spin with
// timeout_s 0, with resource_unavailable naming the silence; the ending
// brings no burst during the hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_silence_of_the_bases_sensing_fails_a_navigation_and_an_unbounded_spin() {
    let mut rig = Rig::start(Scene::open()).await;
    let navigation = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base.follow_state = FollowState::BaseSilence;
    let held = rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&navigation).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["base_silence", "silence"]);
    rig.steps(5).await;
    assert_no_burst_after(&mut rig, ended_ns(&held)).await;
    rig.shutdown().await;

    let mut rig = Rig::start(Scene::open()).await;
    let turning = rig.send_spin_when_ready(&spin_by(PI, 0.0)).await;
    rig.run_on(5).await;
    rig.world.base.follow_state = FollowState::BaseSilence;
    rig.run_until_ended().await;
    let result = outcome::spin::result_of(&turning).await;
    let failure = outcome::spin::failure(&result);
    assert_eq!(
        failure.kind,
        spin::FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["silence"]);
    rig.shutdown().await;
}

// drive_states that report base_mode fail a navigation and
// a spin with attempt_failed, naming the mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_bases_mode_holding_it_fails_a_navigation_and_a_spin_naming_the_mode() {
    let mut rig = Rig::start(Scene::open()).await;
    let navigation = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base.follow_state = FollowState::BaseMode;
    rig.world.base.mode = ModeReport::Fixed("damped");
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&navigation).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(failure.kind, FailureKind::AttemptFailed, "{failure:?}");
    assert_names(&failure.message, &["base_mode", "damped"]);
    rig.shutdown().await;

    let mut rig = Rig::start(Scene::open()).await;
    let turning = rig.send_spin_when_ready(&spin_by(PI, 0.0)).await;
    rig.run_on(5).await;
    rig.world.base.follow_state = FollowState::BaseMode;
    rig.world.base.mode = ModeReport::Fixed("damped");
    rig.run_until_ended().await;
    let result = outcome::spin::result_of(&turning).await;
    let failure = outcome::spin::failure(&result);
    assert_eq!(
        failure.kind,
        spin::FailureKind::AttemptFailed,
        "{failure:?}"
    );
    assert_names(&failure.message, &["damped"]);
    rig.shutdown().await;
}

// base_navigation's limbs: while drive_states reports following and
// parts_within_footprint false, the navigation fails with attempt_failed, the
// burst following; parts_within_footprint false during a hold ends the goal
// by its follow_state alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_limb_outside_the_footprint_fails_the_navigation_unless_a_hold_ends_it_first() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base.parts_within_footprint = false;
    let ran = rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(failure.kind, FailureKind::AttemptFailed, "{failure:?}");
    assert_names(&failure.message, &["a limb left the footprint"]);
    rig.world.base.parts_within_footprint = true;
    let after = rig.burst_after(ended_ns(&ran)).await;
    assert_release_burst(ended_ns(&ran), &after);
    rig.shutdown().await;

    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base.parts_within_footprint = false;
    rig.world.base.follow_state = FollowState::RobotStop;
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let (reason, _) = outcome::navigate_to_pose::cancelled(&result);
    assert_names(reason, &["robot_stop"]);
    rig.shutdown().await;
}

/// What happens in the step whose drive_states measures a spin's whole turn.
#[derive(Clone, Copy, Debug)]
enum AtTheTurnsEnd {
    /// The caller cancels the spin before that step.
    Cancel,
    /// That drive_states reports this follow_state.
    Holds(FollowState),
    /// That drive_states reports parts_within_footprint false.
    LimbLeaves,
    /// That drive_states reports a new odometry_start_timestamp.
    OdometryStartsOver,
}

/// The heading (rad) from which a spin of pi/2 from heading 0 has turned its
/// whole angle, within Nav2's 1e-6 rad.
const QUARTER_TURNED_RAD: f64 = FRAC_PI_2 - 1e-6;

/// A spin of pi/2 within `timeout_s`, `at_the_end` happening in the step
/// whose drive_states first measures its whole turn: the number of the step
/// that ended the spin from the acceptance, and the spin's result. What
/// happens there lasts, so it ends the spin by its rule whichever step the
/// node takes that drive_states at.
async fn spin_ending_with(
    timeout_s: f64,
    at_the_end: Option<AtTheTurnsEnd>,
) -> (usize, spin::ResultResponse) {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig
        .send_spin_when_ready(&spin_by(FRAC_PI_2, timeout_s))
        .await;
    let mut happened = false;
    for step in 1..=DRIVE_STEPS {
        rig.world.advance();
        if !happened && rig.world.base.odometry.yaw >= QUARTER_TURNED_RAD {
            happened = true;
            let base = &mut rig.world.base;
            match at_the_end {
                Some(AtTheTurnsEnd::Cancel) => outcome::spin::cancel(&handle).await,
                Some(AtTheTurnsEnd::Holds(hold)) => base.follow_state = hold,
                Some(AtTheTurnsEnd::LimbLeaves) => base.parts_within_footprint = false,
                Some(AtTheTurnsEnd::OdometryStartsOver) => {
                    base.odometry_start_ns = rig.world.now_ns;
                }
                None => {}
            }
        }
        let seen = rig.world.publish_and_see(&rig.harness.clock).await;
        if seen.goal_ended {
            let result = outcome::spin::result_of(&handle).await;
            rig.shutdown().await;
            return (step, result);
        }
    }
    panic!("the spin ran {DRIVE_STEPS} steps");
}

// base_navigation's order of endings within a step: in the step whose
// drive_states measures a spin's whole turn, a cancel ends the spin
// cancelled, a robot_stop report cancels it naming the hold,
// parts_within_footprint false fails it with attempt_failed, a new
// odometry_start_timestamp with resource_unavailable, and a timeout_s that
// passes in that step with timeout, each before its completion; a timeout_s
// that passes three steps later leaves it completed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_ending_in_the_step_that_measures_a_spins_turn_comes_before_its_completion() {
    let (steps, completed) = spin_ending_with(0.0, None).await;
    outcome::spin::completed(&completed);
    let (_, cancelled) = spin_ending_with(0.0, Some(AtTheTurnsEnd::Cancel)).await;
    outcome::spin::cancelled(&cancelled);
    let (_, held) = spin_ending_with(0.0, Some(AtTheTurnsEnd::Holds(FollowState::RobotStop))).await;
    let (reason, _) = outcome::spin::cancelled(&held);
    assert_names(reason, &["robot_stop"]);
    let (_, limb) = spin_ending_with(0.0, Some(AtTheTurnsEnd::LimbLeaves)).await;
    let failure = outcome::spin::failure(&limb);
    assert_eq!(
        failure.kind,
        spin::FailureKind::AttemptFailed,
        "{failure:?}"
    );
    assert_names(&failure.message, &["a limb left the footprint"]);
    let (_, over) = spin_ending_with(0.0, Some(AtTheTurnsEnd::OdometryStartsOver)).await;
    let failure = outcome::spin::failure(&over);
    assert_eq!(
        failure.kind,
        spin::FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["odometry started over"]);
    let in_the_step_s = steps as f64 * DT - 0.001;
    let (_, timed_out) = spin_ending_with(in_the_step_s, None).await;
    let failure = outcome::spin::failure(&timed_out);
    assert_eq!(failure.kind, spin::FailureKind::Timeout, "{failure:?}");
    let later_s = (steps + 3) as f64 * DT;
    outcome::spin::completed(&spin_ending_with(later_s, None).await.1);
}

// base_navigation's endings: a goal completes at its end, a navigation once
// its goal checker finds the goal reached and a spin once the odometry
// measures its whole turn, on a base whose mode never reports it at rest, and
// the release burst follows each completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_completes_at_its_end_and_the_release_burst_follows() {
    let mut rig = Rig::start(Scene::open()).await;
    rig.world.base.mode = ModeReport::Fixed("moving");
    let navigation = rig.send_when_ready(&goal(1.0, 0.0)).await;
    let (ran, after) = ending(&mut rig).await;
    let result = outcome::navigate_to_pose::result_of(&navigation).await;
    let data = outcome::navigate_to_pose::completed(&result);
    assert!(
        (data.final_pose.position[0] - 1.0).abs() <= 0.1,
        "within the tolerance: {data:?}"
    );
    assert_release_burst(ended_ns(&ran), &after);
    let turning = rig.send_spin_when_ready(&spin_by(FRAC_PI_2, 0.0)).await;
    let (ran, after) = ending(&mut rig).await;
    let result = outcome::spin::result_of(&turning).await;
    outcome::spin::completed(&result);
    assert_release_burst(ended_ns(&ran), &after);
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// The moves: a spin, a drive_on_heading and a back_up.
// ---------------------------------------------------------------------------

/// What each of `steps` that a move ran on measured of its turn (rad) or its
/// distance (m), as its feedback gives it.
fn moved(steps: &[StepSeen]) -> Vec<f64> {
    steps
        .iter()
        .filter_map(|step| match &step.feedback {
            Some(Feedback::Spin(feedback)) => Some(feedback.angle_turned_rad),
            Some(Feedback::DriveOnHeading(feedback)) => Some(feedback.distance_traveled_m),
            Some(Feedback::BackUp(feedback)) => Some(feedback.distance_traveled_m),
            _ => None,
        })
        .collect()
}

/// Asserts that a move gave feedback at each of `steps` but the one that
/// ended it, what it measured growing and short of `end` until then.
fn assert_moved_short_of(steps: &[StepSeen], end: f64) {
    let moved = moved(steps);
    assert_eq!(
        moved.len() + 1,
        steps.len(),
        "feedback at each step: {moved:?}"
    );
    assert!(
        moved.windows(2).all(|pair| pair[0] <= pair[1]) && moved.iter().all(|&m| m < end),
        "the move measures its way to {end}: {moved:?}"
    );
}

// base_navigation's moves: a spin of pi/2 turns the base until the odometry
// measures pi/2 and completes at that step, its result where that
// drive_states placed the robot frame, at its timestamp; the base comes to
// rest after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_spin_turns_until_the_odometry_measures_its_angle_and_completes_there() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_spin_when_ready(&spin_by(FRAC_PI_2, 0.0)).await;
    let steps = rig.run_until_ended().await;
    assert_moved_short_of(&steps, QUARTER_TURNED_RAD);
    let result = outcome::spin::result_of(&handle).await;
    let data = outcome::spin::completed(&result);
    let (end, _) = rig.world.reported_at(data.final_pose.timestamp);
    assert_eq!(data.final_pose.position, [end.x, end.y, 0.0]);
    assert!(end.yaw >= QUARTER_TURNED_RAD, "{data:?}");
    assert!(
        yaw_of(data.final_pose.orientation) >= QUARTER_TURNED_RAD,
        "{data:?}"
    );
    assert!(!rig.world.base.rests(), "the base still turns at the end");
    until_at_rest(&mut rig, 20).await;
    rig.shutdown().await;
}

// A drive_on_heading of 1 m and a back_up of 0.3 m each cover their
// distance as the odometry measures it and complete there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drive_on_heading_and_a_back_up_cover_their_distance_and_complete_there() {
    let mut rig = Rig::start(Scene::open()).await;
    let forward_drive = rig.send_drive_when_ready(&ahead(1.0, 0.3, 0.0)).await;
    let steps = rig.run_until_ended().await;
    assert_moved_short_of(&steps, 1.0);
    rig.world.until_setpoint_from(ended_ns(&steps)).await;
    let ran = setpoints(&rig, &steps);
    assert!(
        ran.iter().all(|s| forward(s) >= 0.0 && turn(s) == 0.0),
        "{ran:?}"
    );
    let result = outcome::drive_on_heading::result_of(&forward_drive).await;
    let data = outcome::drive_on_heading::completed(&result);
    assert!(data.final_pose.position[0] >= 1.0 - 1e-9, "{data:?}");
    until_at_rest(&mut rig, 20).await;
    let start_x = rig.world.base.odometry.x;
    let backward_drive = rig.send_back_up(&back(0.3, 0.15, 0.0)).await;
    assert!(backward_drive.accepted, "{:?}", backward_drive.refusal);
    let steps = rig.run_until_ended().await;
    assert_moved_short_of(&steps, 0.3);
    rig.world.until_setpoint_from(ended_ns(&steps)).await;
    let ran = setpoints(&rig, &steps);
    assert!(ran.iter().all(|s| forward(s) <= 0.0), "{ran:?}");
    let result = outcome::back_up::result_of(&backward_drive).await;
    let data = outcome::back_up::completed(&result);
    assert!(
        start_x - data.final_pose.position[0] >= 0.3 - 1e-9,
        "{data:?}"
    );
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// A clock that lags the messages, and one that leaps.
// ---------------------------------------------------------------------------

/// How far the lagging clock reads behind the messages' stamps (ns): 0.4 s,
/// most of drive_link's 0.5 s.
const CLOCK_LAG_NS: i64 = 400_000_000;

// The kickoff's clock lesson: on a clock that reads 0.4 s behind every
// message's stamp, the tree places each frame and the navigation, and the
// base drives to the goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_a_clock_that_lags_the_messages_a_navigation_completes() {
    let mut rig = Rig::start_lagging(Scene::with(&[BOX_AHEAD]), CLOCK_LAG_NS).await;
    let handle = rig.send_when_ready(&to(BEHIND_THE_BOX)).await;
    rig.run_checking(|base| assert!(!touches(base, &BOX_AHEAD), "{:?}", base.truth))
        .await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

// The kickoff's clock lesson, drive_link's deadman: the 0.5 s of silence
// count from the arrival of the newest drive_states on the node's clock,
// never from its stamp. The newest drive_states, stamped at the rig's step S,
// arrives while the node's clock reads S less a step less the lag, or a step
// later; the goal fails at the tenth or eleventh step after S, when the
// node's clock passes that arrival by more than 0.5 s; the clock passes the
// stamp by 0.5 s only at the nineteenth.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_a_lagging_clock_the_drive_states_deadman_counts_from_arrival() {
    let mut rig = Rig::start_lagging(Scene::open(), CLOCK_LAG_NS).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    rig.world.base_reports = false;
    let ran = rig.run_until_ended().await;
    assert!(
        (10..=11).contains(&ran.len()),
        "the goal ends at the tenth or eleventh silent step, at {}",
        ran.len()
    );
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["sent no drive_states"]);
    rig.shutdown().await;
}

// The kickoff's clock lesson: a clock that leaps 0.65 s, the base's
// drive_states arriving meanwhile while it stood still, finds the newest
// arrival more than 0.5 s old: the navigation fails with resource_unavailable
// at the leap's step, and with the navigator's newest setpoint as old, its
// stream ended, no burst follows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clock_that_leaps_past_half_a_second_fails_the_navigation_with_no_burst() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let leaped = rig.leap(13).await;
    assert!(leaped.goal_ended, "the leap's step ends the navigation");
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    let failure = outcome::navigate_to_pose::failure(&result);
    assert_eq!(
        failure.kind,
        FailureKind::ResourceUnavailable,
        "{failure:?}"
    );
    assert_names(&failure.message, &["sent no drive_states"]);
    rig.steps(5).await;
    let fence_ns = rig.fence().await;
    assert_streams_nothing(
        &rig.world.setpoints_in(leaped.at_ns..fence_ns),
        "from the leap's step: its stream had ended",
    );
    rig.shutdown().await;
}

// The kickoff's clock lesson: a leap of 0.4 s leaves the navigation running:
// the leap's step streams a command, and the navigation completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clock_that_leaps_less_than_half_a_second_leaves_the_navigation_running() {
    let mut rig = Rig::start(Scene::open()).await;
    let handle = rig.send_when_ready(&goal(2.0, 0.0)).await;
    rig.run_until(|base| base.velocity.linear_x() > 0.3).await;
    let leaped = rig.leap(8).await;
    assert!(!leaped.goal_ended, "the navigation goes on");
    rig.world.until_setpoint_from(leaped.at_ns).await;
    assert!(
        rig.world.setpoint_at(leaped.at_ns).is_some(),
        "the leap's step streams"
    );
    rig.run_until_ended().await;
    let result = outcome::navigate_to_pose::result_of(&handle).await;
    outcome::navigate_to_pose::completed(&result);
    rig.shutdown().await;
}

// ---------------------------------------------------------------------------
// What the node logs.
// ---------------------------------------------------------------------------

// The setup logs a warning that names approach_velocity_scaling_dist
// and half the grid's side when the first exceeds the second, as Nav2's
// regulated pure pursuit warns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_setup_warns_when_the_approach_scaling_distance_exceeds_half_the_grid() {
    keep_the_log();
    let mut wide = params();
    wide.controller.approach_velocity_scaling_dist_m = 2.75;
    let rig = Rig::start_with(wide, Scene::open(), &[FRONT], measured()).await;
    let warned = logged_with(&[
        "approach_velocity_scaling_dist_m 2.75 m",
        "half of grid.side_m, 2.5 m",
    ]);
    assert_eq!(warned.len(), 1, "{warned:?}");
    rig.shutdown().await;
}

// Each describe_base read logs both of the stop's values, the time before a
// collision and the bound from source_timeout_s and the base's limits, and
// the larger, which the stop projects: decelerating at 0.5, the turn takes
// 3 s to rest from 1.5 rad/s, so the stop projects 3.5 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_describe_base_read_logs_both_horizons_of_the_stop() {
    keep_the_log();
    let mut slow = base_answer(decelerating(0.5), "measured");
    slow.base_name = "slow_base".to_string();
    let mut rig = Rig::start_with(params(), Scene::open(), &[FRONT], slow).await;
    let handle = rig.send_when_ready(&goal(1.0, 0.0)).await;
    assert!(handle.accepted, "the base is read");
    let logged = logged_with(&[
        "base slow_base: the stop projects 3.5 s",
        "time_before_collision_s 1 s",
        "source_timeout_s 0.5 s",
        "3 s to rest",
    ]);
    assert_eq!(logged.len(), 1, "{logged:?}");
    rig.shutdown().await;
}
