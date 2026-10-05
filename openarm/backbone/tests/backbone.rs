//! Integration tests over the generated harness: the backbone in-process, the
//! robot initializer, the leading node and the four limb followers all played
//! by generated mocks over the real wire.
//!
//! Fixture notes, from what the node actually does:
//!
//! - `startup::wait_until_ready` gates the subscriptions, publishers, actions
//!   and the coordinator on `robot_init/is_ready` answering `ready: true`, so
//!   every test pumps that mock service. The `get_limb_names`,
//!   `get_camera_poses` and workspace services stand outside that gate and
//!   answer from bringup.
//! - `coordinator::seed_all` then gates streaming on a first measured state
//!   from BOTH arms and BOTH grippers, and `liveness` freezes a limb that goes
//!   silent for four periods of `follower_state_rate_hz`. The harness pins every pairing slot to a
//!   mock (only the `collision_ctrl` and `perception_geometry` dependency
//!   slots have a `_vacant` knob), so
//!   each test pumps all four follower back-channels at a rate inside the
//!   stale limit. The unselected slots (pose pairs, upstream gripper pairs,
//!   and any leader pair a test does not drive) stay silent, which is exactly
//!   what an unbound optional slot delivers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use peppygen::fixtures::exposed_actions::limb_motion::{move_arm_joints, move_gripper};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::perception_geometry::{get_color_intrinsics, get_depth_intrinsics};
use tokio::sync::{mpsc, oneshot, watch};

/// How long a mock pump waits parked for the node's next poll or for the
/// node's subscription to appear; only expires once the harness is gone.
const PUMP_TIMEOUT: Duration = Duration::from_secs(120);

/// Control rate for the tests: 20 ms cycle (Nyquist 25 Hz, above the 15 Hz
/// velocity-filter default).
const CONTROL_RATE_HZ: u32 = 50;

/// Declared follower rate for the tests: four of its periods make an 80 ms
/// stale window, comfortably above the state pumps even on a loaded machine.
const FOLLOWER_STATE_RATE_HZ: u32 = 50;

/// Follower back-channel period; must stay well inside the 80 ms follower
/// stale window or the coordinator freezes that limb.
const STATE_PUMP_PERIOD: Duration = Duration::from_millis(10);

/// Rest pose both arms are seeded at: elbow (j4) exactly at the description's
/// 0.05 rad singularity floor, so the planner's limit clamp is the identity
/// and held setpoints echo it bit-exact. Same pose as the governor's own
/// unit-test `home()`, whose nearest checked pair sits outside the validated
/// band on v1 (the governor test `far_apart_is_unthrottled`).
const HOME: [f64; 7] = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0];

/// Outer deadline for every bounded convergence loop.
const DEADLINE: Duration = Duration::from_secs(90);

/// Per-read window inside a convergence loop: several coordinator readout /
/// stream periods, so an elapsed window means "nothing arrived", not jitter.
const READ_WINDOW: Duration = Duration::from_secs(2);

/// The node's full parameter set (most have no schema default, so every test
/// passes them explicitly): v1 hardware, joints-mode upstream, the validated
/// collision band, and a permissive EE cap so streamed chases converge fast.
fn params() -> peppygen::Parameters {
    peppygen::Parameters {
        collision_governor_enabled: true,
        control_rate_hz: CONTROL_RATE_HZ,
        d_safe_m: 0.02,
        d_stop_m: 0.005,
        follower_state_rate_hz: FOLLOWER_STATE_RATE_HZ,
        hardware_version: "v1".to_string(),
        max_ee_angular_velocity_rad_s: 0.8,
        max_ee_velocity_m_s: 2.0,
        max_gripper_rate_frac_s: 6.0,
        max_joint_velocity_rad_s_1: 16.754666,
        max_joint_velocity_rad_s_2: 16.754666,
        max_joint_velocity_rad_s_3: 5.445426,
        max_joint_velocity_rad_s_4: 5.445426,
        max_joint_velocity_rad_s_5: 20.943946,
        max_joint_velocity_rad_s_6: 20.943946,
        max_joint_velocity_rad_s_7: 20.943946,
        upstream_mode: "joints".to_string(),
        velocity_filter_cutoff_hz: 15.0,
    }
}

/// The node's parameter set on v2 hardware, whose design carries the
/// cameras.
fn v2_params() -> peppygen::Parameters {
    peppygen::Parameters {
        hardware_version: "v2".to_string(),
        ..params()
    }
}

/// Answers every `robot_init/is_ready` poll with the current value of `flag`,
/// until the mock's session closes (the node polls every 500 ms).
fn pump_is_ready(
    service: peppygen::mock::deps::robot_init::is_ready::Service,
    flag: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        use peppygen::mock::deps::robot_init::is_ready::ResponseData;
        while let Ok(responder) = service.next_request(PUMP_TIMEOUT).await {
            let ready = flag.load(Ordering::SeqCst);
            if responder.respond(ResponseData { ready }).await.is_err() {
                break;
            }
        }
    });
}

/// Plays one follower's state back-channel: waits for the node's pinned
/// subscription (opened only once the readiness gate passes), then publishes
/// `$make()` every [`STATE_PUMP_PERIOD`]. Publishing before the match would
/// hit the publisher's own 10 s readiness timeout while a test still holds
/// the node not-ready, so the wait is explicit and effectively unbounded.
macro_rules! pump_states {
    ($publisher:expr, $make:expr) => {{
        let publisher = $publisher;
        tokio::spawn(async move {
            if !matches!(publisher.wait_for_subscriber(PUMP_TIMEOUT).await, Ok(true)) {
                return;
            }
            let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
            loop {
                ticker.tick().await;
                if publisher.publish(&$make()).await.is_err() {
                    return;
                }
            }
        });
    }};
}

/// Static joint-state pump for one arm follower, parked at [`HOME`].
macro_rules! pump_arm_at_home {
    ($publisher:expr, $message:path) => {
        pump_states!($publisher, || {
            use $message as message;
            message::Message {
                timestamp: SystemTime::now(),
                positions: HOME.to_vec(),
                velocities: vec![0.0; 7],
                efforts: Vec::new(),
            }
        })
    };
}

/// Static aperture pump for one gripper follower, at `$opening`.
macro_rules! pump_gripper_at {
    ($publisher:expr, $message:path, $opening:expr) => {
        pump_states!($publisher, || {
            use $message as message;
            message::Message {
                timestamp: SystemTime::now(),
                opening: $opening,
                effort: 0.0,
                max_effort: 1.0,
            }
        })
    };
}

/// Static aperture pump for one gripper follower, half open.
macro_rules! pump_gripper_half_open {
    ($publisher:expr, $message:path) => {
        pump_gripper_at!($publisher, $message, 0.5)
    };
}

/// The three static followers every test needs alongside whatever it does
/// with the left arm: right arm at [`HOME`], both grippers half open.
macro_rules! pump_right_arm_and_grippers {
    ($mocks:ident) => {
        pump_arm_at_home!(
            $mocks.pairings.right_arm.joint_states,
            peppygen::paired_topics::right_arm::joint_states
        );
        pump_gripper_half_open!(
            $mocks.pairings.left_gripper.gripper_states,
            peppygen::paired_topics::left_gripper::gripper_states
        );
        pump_gripper_half_open!(
            $mocks.pairings.right_gripper.gripper_states,
            peppygen::paired_topics::right_gripper::gripper_states
        );
    };
}

/// One leader `joint_setpoints` command at `positions` (velocities/efforts
/// ride empty: the backbone plans its own velocity shaping).
fn leader_command(
    positions: [f64; 7],
) -> peppygen::paired_topics::leader_left_arm::joint_setpoints::Message {
    peppygen::paired_topics::leader_left_arm::joint_setpoints::Message {
        timestamp: SystemTime::now(),
        positions: positions.to_vec(),
        velocities: Vec::new(),
        efforts: Vec::new(),
    }
}

/// One arm played by [`spawn_arm_follower!`]: the latest position it
/// adopted, and its brake.
struct ArmFollower {
    followed: watch::Receiver<[f64; 7]>,
    brake: mpsc::Sender<oneshot::Sender<[f64; 7]>>,
}

impl ArmFollower {
    /// Stop the arm where it stands: from now on it adopts no setpoint and
    /// publishes the position this returns.
    async fn brake(&self) -> [f64; 7] {
        let (reply, held) = oneshot::channel();
        self.brake
            .send(reply)
            .await
            .expect("the follower takes the brake");
        held.await.expect("the follower answers the brake")
    }
}

/// Plays one arm as a perfect follower, starting at [`HOME`]:
/// - it publishes its measured state every [`STATE_PUMP_PERIOD`] on
///   `$states`, a publisher of the `$message` slot;
/// - it adopts each governed setpoint that the node streams down
///   `$setpoints` as the new measurement, until the test brakes it.
///
/// The watch it gives carries the latest adopted position. Thus a test can
/// assert what motion the arm mock observed.
macro_rules! spawn_arm_follower {
    ($states:expr, $setpoints:expr, $message:path) => {{
        let states = $states;
        let mut setpoints = $setpoints;
        let (tx, followed) = watch::channel(HOME);
        let (brake, mut brakes) = mpsc::channel::<oneshot::Sender<[f64; 7]>>(1);
        tokio::spawn(async move {
            use $message as message;
            if !matches!(states.wait_for_subscriber(PUMP_TIMEOUT).await, Ok(true)) {
                return;
            }
            let mut positions = HOME;
            let mut braked = false;
            let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let message = message::Message {
                            timestamp: SystemTime::now(),
                            positions: positions.to_vec(),
                            velocities: vec![0.0; 7],
                            efforts: Vec::new(),
                        };
                        if states.publish(&message).await.is_err() {
                            return;
                        }
                    }
                    received = setpoints.next() => match received {
                        Ok(Some(m)) if !braked && m.positions.len() == 7 => {
                            for (slot, v) in positions.iter_mut().zip(&m.positions) {
                                *slot = *v;
                            }
                            tx.send_replace(positions);
                        }
                        Ok(Some(_)) | Err(_) => {}
                        Ok(None) => return,
                    },
                    Some(reply) = brakes.recv(), if !braked => {
                        braked = true;
                        let _ = reply.send(positions);
                    }
                }
            }
        });
        ArmFollower { followed, brake }
    }};
}

/// Plays the left arm as a perfect follower that is never braked (see
/// [`spawn_arm_follower!`]). The returned watch carries the latest adopted
/// position.
fn spawn_left_arm_follower(
    states: peppygen::mock::pairings::left_arm::joint_states::Publisher,
    setpoints: peppygen::mock::pairings::left_arm::joint_setpoints::Subscription,
) -> watch::Receiver<[f64; 7]> {
    spawn_arm_follower!(
        states,
        setpoints,
        peppygen::paired_topics::left_arm::joint_states
    )
    .followed
}

/// Jaw travel (opening fraction) the lagging gripper follower covers per
/// [`STATE_PUMP_PERIOD`]: 1.0 per second against the 6.0 per second the
/// backbone commands at, so the jaws arrive many state periods after the
/// commanded ramp has landed.
const LAGGING_GRIPPER_STEP: f64 = 0.01;

/// One gripper setpoint as the follower receives it: the opening fraction
/// and the effort cap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GripperSetpoint {
    opening: f64,
    max_effort: f64,
}

/// The left gripper played by [`spawn_left_gripper_follower`]: the opening
/// it publishes, and every setpoint it received, in order.
struct LeftGripper {
    jaws: watch::Receiver<f64>,
    received: Arc<Mutex<Vec<GripperSetpoint>>>,
}

impl LeftGripper {
    /// How many setpoints the gripper received so far.
    fn received_count(&self) -> usize {
        self.received
            .lock()
            .expect("no panic while recording")
            .len()
    }

    /// The setpoints the gripper received after the first `count`.
    fn received_since(&self, count: usize) -> Vec<GripperSetpoint> {
        self.received.lock().expect("no panic while recording")[count..].to_vec()
    }

    /// The last setpoint the gripper received: what drives it while the
    /// node sends nothing.
    fn last_received(&self) -> Option<GripperSetpoint> {
        self.received
            .lock()
            .expect("no panic while recording")
            .last()
            .copied()
    }
}

/// Plays the left gripper as a follower whose jaws lag their commands: every
/// [`STATE_PUMP_PERIOD`] the opening moves at most [`LAGGING_GRIPPER_STEP`]
/// toward the latest setpoint the node streamed down, landing on it exactly,
/// and is published as the measured state. An object between the jaws stops
/// them at `object_at`, the opening they cannot close past (0.0 for no
/// object). It starts half open like the static gripper pumps, and records
/// every setpoint it receives.
fn spawn_left_gripper_follower(
    states: peppygen::mock::pairings::left_gripper::gripper_states::Publisher,
    mut setpoints: peppygen::mock::pairings::left_gripper::gripper_setpoints::Subscription,
    object_at: f64,
) -> LeftGripper {
    let mut opening = 0.5;
    let (tx, jaws) = watch::channel(opening);
    let received = Arc::new(Mutex::new(Vec::new()));
    let record = received.clone();
    tokio::spawn(async move {
        if !matches!(states.wait_for_subscriber(PUMP_TIMEOUT).await, Ok(true)) {
            return;
        }
        let mut commanded = opening;
        let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let remaining: f64 = commanded - opening;
                    let next = if remaining.abs() <= LAGGING_GRIPPER_STEP {
                        commanded
                    } else {
                        opening + LAGGING_GRIPPER_STEP.copysign(remaining)
                    };
                    opening = next.max(object_at);
                    let message = peppygen::paired_topics::left_gripper::gripper_states::Message {
                        timestamp: SystemTime::now(),
                        opening,
                        effort: 0.0,
                        max_effort: 1.0,
                    };
                    if states.publish(&message).await.is_err() {
                        return;
                    }
                    tx.send_replace(opening);
                }
                received = setpoints.next() => match received {
                    Ok(Some(m)) => {
                        commanded = m.opening;
                        record.lock().expect("no panic while recording").push(GripperSetpoint {
                            opening: m.opening,
                            max_effort: m.max_effort,
                        });
                    }
                    Err(_) => {}
                    Ok(None) => return,
                }
            }
        }
    });
    LeftGripper { jaws, received }
}

/// Plays the left gripper with nothing between its jaws (see
/// [`spawn_left_gripper_follower`]). The returned watch carries the latest
/// published opening.
fn spawn_lagging_left_gripper_follower(
    states: peppygen::mock::pairings::left_gripper::gripper_states::Publisher,
    setpoints: peppygen::mock::pairings::left_gripper::gripper_setpoints::Subscription,
) -> watch::Receiver<f64> {
    spawn_left_gripper_follower(states, setpoints, 0.0).jaws
}

/// The full boot with the robot reporting ready and both dependency slots
/// vacant (their `zero_or_one` empty binding): the launch-time band stands,
/// exactly the branch the cardinality exists for, and the workspace answers
/// leave the view unchecked.
async fn start_ready_vacant(
    parameters: peppygen::Parameters,
) -> peppygen::Result<(Harness, peppygen::fixtures::harness::Mocks)> {
    let (harness, mocks) = Harness::start_with(
        Config {
            parameters: Some(parameters),
            collision_ctrl_vacant: true,
            perception_geometry_vacant: true,
            ..Default::default()
        },
        openarm_backbone::setup,
    )
    .await?;
    assert!(
        mocks.deps.collision_ctrl.is_none(),
        "a vacant collision_ctrl slot must start no mock"
    );
    assert!(
        mocks.deps.perception_geometry.is_none(),
        "a vacant perception_geometry slot must start no mock"
    );
    Ok((harness, mocks))
}

/// Wait for the first governed setpoint on the right arm's wire. The
/// coordinator has then seeded every limb. Thus a goal sent from now on
/// runs, and the seed wait does not refuse it.
async fn await_streaming(
    mut right_wire: peppygen::mock::pairings::right_arm::joint_setpoints::Subscription,
) -> peppygen::Result<()> {
    tokio::time::timeout(DEADLINE, right_wire.next())
        .await
        .expect("streaming never began")?
        .expect("right arm subscription open");
    Ok(())
}

/// The left arm and the left gripper of a node that
/// [`start_with_left_followers`] started.
struct LeftFollowers {
    /// The latest position the left arm adopted.
    arm: watch::Receiver<[f64; 7]>,
    gripper: LeftGripper,
}

/// Start the node with the robot ready and wait until streaming begins.
/// - The left arm is a perfect follower (see [`spawn_left_arm_follower`]).
/// - The left gripper lags its commands, and an object stops its jaws at
///   `object_at` (see [`spawn_left_gripper_follower`]).
/// - The right arm stands at [`HOME`] and the right gripper is half open.
async fn start_with_left_followers(object_at: f64) -> peppygen::Result<(Harness, LeftFollowers)> {
    let (harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let arm = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_arm_at_home!(
        mocks.pairings.right_arm.joint_states,
        peppygen::paired_topics::right_arm::joint_states
    );
    pump_gripper_half_open!(
        mocks.pairings.right_gripper.gripper_states,
        peppygen::paired_topics::right_gripper::gripper_states
    );
    let gripper = spawn_left_gripper_follower(
        mocks.pairings.left_gripper.gripper_states,
        mocks.pairings.left_gripper.gripper_setpoints,
        object_at,
    );
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;
    Ok((harness, LeftFollowers { arm, gripper }))
}

/// The next limb_states snapshot the node emits. The node emits one only
/// once it has seeded every limb, so the first one also says that streaming
/// began.
async fn next_snapshot(
    harness: &mut Harness,
) -> peppygen::Result<peppygen::fixtures::emitted_topics::limb_state::limb_states::Message> {
    Ok(
        tokio::time::timeout(DEADLINE, harness.emitted.limb_state_limb_states.next())
            .await
            .expect("no limb_states snapshot before the deadline")?
            .expect("limb_states subscription open"),
    )
}

/// Assert that `got` holds as many values as `expected`, each within
/// `tolerance` of its counterpart.
fn assert_all_close(what: &str, got: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(got.len(), expected.len(), "{what}: {got:?}");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= tolerance,
            "{what}[{i}] = {g}, expected {e} within {tolerance}: {got:?}"
        );
    }
}

/// The result of a posture goal, as either postures action gives it.
#[derive(Debug)]
struct PostureResult {
    success: bool,
    message: String,
    arm_names: Vec<String>,
    positions: Vec<f64>,
    orientations: Vec<f64>,
}

/// Send one goal of `$action` (move_to_ready or move_to_home) with
/// `$duration_s`, and give the node's answer to it.
macro_rules! send_posture_goal {
    ($harness:expr, $action:ident, $duration_s:expr) => {{
        use peppygen::fixtures::exposed_actions::postures::$action;
        $action::send_goal(
            &$harness,
            &$action::GoalRequestData {
                duration_s: $duration_s,
            },
            peppygen::QoSProfile::Reliable,
            DEADLINE,
        )
        .await?
    }};
}

/// Send one goal of `$action` with `$duration_s`, and assert that the node
/// refuses it as an invalid duration.
macro_rules! assert_posture_refused {
    ($harness:expr, $action:ident, $duration_s:expr) => {{
        let goal = send_posture_goal!($harness, $action, $duration_s);
        assert!(
            !goal.accepted,
            "a {} s {} must be refused",
            $duration_s,
            stringify!($action)
        );
        assert_eq!(goal.reason.as_deref(), Some("invalid duration"));
    }};
}

/// Send one goal of `$action` with `$duration_s`, and wait for the result
/// it completes with.
macro_rules! complete_posture {
    ($harness:expr, $action:ident, $duration_s:expr) => {{
        use peppygen::fixtures::exposed_actions::postures::$action;
        let goal = send_posture_goal!($harness, $action, $duration_s);
        assert!(
            goal.accepted,
            "{} rejected: {:?}",
            stringify!($action),
            goal.reason
        );
        match goal.get_result(DEADLINE).await?.outcome {
            $action::ResultOutcome::Completed(data) => PostureResult {
                success: data.success,
                message: data.message,
                arm_names: data.arm_names,
                positions: data.positions,
                orientations: data.orientations,
            },
            other => panic!("{} did not complete: {other:?}", stringify!($action)),
        }
    }};
}

/// Send one `move_arm_joints` goal for the left arm to `target` over
/// `duration_s`, and assert that it completes with success.
async fn complete_left_arm_joint_move(
    harness: &Harness,
    target: [f64; 7],
    duration_s: f64,
) -> peppygen::Result<move_arm_joints::ResultData> {
    let goal = move_arm_joints::send_goal(
        harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "left_arm".to_string(),
            joint_positions: target,
            duration_s,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal rejected: {:?}", goal.reason);
    match goal.get_result(DEADLINE).await?.outcome {
        move_arm_joints::ResultOutcome::Completed(data) => {
            assert!(data.success, "{}", data.message);
            Ok(data)
        }
        other => panic!("move_arm_joints did not complete: {other:?}"),
    }
}

/// Send one `move_gripper` goal for the left gripper to `opening` under
/// `max_effort`, and wait for the result it completes with.
async fn complete_left_gripper_move(
    harness: &Harness,
    opening: f64,
    max_effort: f64,
) -> peppygen::Result<move_gripper::ResultData> {
    let goal = move_gripper::send_goal(
        harness,
        &move_gripper::GoalRequestData {
            gripper_name: "left_gripper".to_string(),
            opening,
            max_effort,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal rejected: {:?}", goal.reason);
    match goal.get_result(DEADLINE).await?.outcome {
        move_gripper::ResultOutcome::Completed(data) => Ok(data),
        other => panic!("move_gripper did not complete: {other:?}"),
    }
}

/// Readiness gate + minimal fan-through: with the robot ready, `collision_ctrl`
/// vacant and only the left leader driving, the leading node's joint command
/// crosses the whole pipeline (listener -> chase -> governor -> pairing wire)
/// to the left arm mock, while the uncommanded right arm holds its seeded pose
/// bit-exact and the unselected slots stay silent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leader_command_fans_through_to_the_governed_arm_wire() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params()).await?;

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    let mut left_wire = mocks.pairings.left_arm.joint_setpoints;
    let mut right_wire = mocks.pairings.right_arm.joint_setpoints;
    let leader = mocks.pairings.leader_left_arm.joint_setpoints;

    // Streaming begins (both arms + both grippers seeded): before any command,
    // every published setpoint is the held seed. The uncommanded right side
    // holds it bit-exact through the whole governed pipeline.
    let first_right = tokio::time::timeout(DEADLINE, right_wire.next())
        .await
        .expect("the right arm wire must start streaming once all followers report")?
        .expect("right arm subscription open");
    assert_eq!(first_right.positions.len(), 7);
    for (published, seeded) in first_right.positions.iter().zip(HOME.iter()) {
        assert!(
            (published - seeded).abs() < 1e-9,
            "an uncommanded arm must hold its seeded pose, got {:?}",
            first_right.positions
        );
    }

    // The leading node commands an elbow bend on the left arm only. The watch
    // keeps the latest command, so one delivery suffices; the republish on an
    // empty read window covers a best-effort drop.
    let mut target = HOME;
    target[3] = 0.4;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    leader.publish(&leader_command(target)).await?;
    loop {
        match tokio::time::timeout(READ_WINDOW, left_wire.next()).await {
            Ok(setpoint) => {
                let setpoint = setpoint?.expect("left arm subscription open");
                assert_eq!(setpoint.positions.len(), 7);
                if (setpoint.positions[3] - target[3]).abs() < 0.05 {
                    break;
                }
            }
            Err(_) => leader.publish(&leader_command(target)).await?,
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "governed setpoints never converged on the leader's command"
        );
    }

    harness.shutdown().await
}

/// Exposed action end-to-end: `move_arm_joints` through the real action
/// engine, with the left arm mock playing a perfect follower (it adopts every
/// governed setpoint as its measured state). The goal must be admitted, the
/// trajectory streamed down the pairing wire, and the result must report
/// success with the measured (echoed) pose on the commanded target.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_arm_joints_streams_a_trajectory_the_arm_follows_to_the_target() -> peppygen::Result<()>
{
    // collision_ctrl bound to its (silent) mock this time: the listener runs
    // against a live producer that never publishes, and the launch band stands.
    let (harness, mocks) = Harness::start_with(
        Config {
            parameters: Some(params()),
            collision_ctrl_vacant: false,
            perception_geometry_vacant: true,
            ..Default::default()
        },
        openarm_backbone::setup,
    )
    .await?;
    assert!(mocks.deps.collision_ctrl.is_some());

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);

    // Gate the goal on streaming having begun: a goal that reaches the
    // coordinator while `seed_all` still waits for first states is refused
    // ("the follower has not reported its first state yet"), so wait for the
    // first governed setpoint on the (uncommanded) right wire first.
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    // A known-good working posture (the coordinator's own unit tests hold it),
    // well clear of the other arm, so the tiny validated band never throttles.
    let target = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
    // The result parks until the trajectory completes. This action exposes no
    // feedback stream (peppy.json5 defines none), so the result is the whole
    // terminal protocol.
    let data = complete_left_arm_joint_move(&harness, target, 1.0).await?;
    assert!(data.action_time > 0.0);
    // `final_joint_positions` is the measured pose at the terminal, i.e. what
    // the arm mock echoed back: the follower observed the commanded motion
    // arrive at the target.
    for (joint, (reached, commanded)) in data
        .final_joint_positions
        .iter()
        .zip(target.iter())
        .enumerate()
    {
        assert!(
            (reached - commanded).abs() < 0.05,
            "joint {joint} ended at {reached}, commanded {commanded}"
        );
    }
    let observed = *followed.borrow();
    for (observed, commanded) in observed.iter().zip(target.iter()) {
        assert!((observed - commanded).abs() < 0.05);
    }

    harness.shutdown().await
}

/// A joint move that the collision governor holds short still completes
/// with success when its time runs out. Its result gives the joints that
/// the arm measured then, not the goal. Here the band is wide, so the rest
/// pose sits under d_stop. A left wrist sent toward the centerline never
/// gets there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_move_the_governor_holds_completes_with_the_measured_pose() -> peppygen::Result<()>
{
    let mut wide = params();
    wide.d_stop_m = 0.8;
    wide.d_safe_m = 1.0;
    let (harness, mocks) = start_ready_vacant(wide).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    let mut inward = HOME;
    inward[2] = 1.5;
    inward[3] = 0.4;
    let data = complete_left_arm_joint_move(&harness, inward, 1.0).await?;
    assert_eq!(data.message, "trajectory complete");
    assert!(
        (data.final_joint_positions[2] - inward[2]).abs() > 0.5,
        "the governor let the wrist reach its goal: {:?}",
        data.final_joint_positions
    );
    let observed = *followed.borrow();
    for (joint, (reported, observed)) in data
        .final_joint_positions
        .iter()
        .zip(observed.iter())
        .enumerate()
    {
        assert!(
            (reported - observed).abs() < 0.05,
            "joint {joint}: the result gives {reported}, the arm stands at {observed}"
        );
    }
    harness.shutdown().await
}

/// A move_arm result gives the grasp pose of the joints that the arm
/// measured when the move ended. Here the left arm stands still at Ready.
/// A move 5 cm up runs to its end with success. Its result gives where the
/// arm stands, as limb_state does, not the goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pose_move_reports_the_grasp_point_of_the_measured_joints() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::limb_motion::move_arm;
    use peppygen::fixtures::exposed_services::limb_motion::check_arm_move;

    let (mut harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let ready = openarm_description::ready(openarm_description::Side::Left);
    pump_states!(mocks.pairings.left_arm.joint_states, || {
        peppygen::paired_topics::left_arm::joint_states::Message {
            timestamp: SystemTime::now(),
            positions: ready.to_vec(),
            velocities: vec![0.0; 7],
            efforts: Vec::new(),
        }
    });
    pump_right_arm_and_grippers!(mocks);

    let snapshot = next_snapshot(&mut harness).await?;
    let start = [
        snapshot.positions[0],
        snapshot.positions[1],
        snapshot.positions[2],
    ];
    let orientation = [
        snapshot.orientations[0],
        snapshot.orientations[1],
        snapshot.orientations[2],
        snapshot.orientations[3],
    ];
    let raised = [start[0], start[1], start[2] + 0.05];
    let check = check_arm_move::poll(
        &harness,
        &check_arm_move::RequestData {
            arm_name: "left_arm".to_string(),
            position: raised,
            orientation,
            duration_s: 1.0,
            plan_position_tolerance_m: 0.0,
            plan_orientation_tolerance_rad: 0.0,
        },
        DEADLINE,
    )
    .await?;
    assert!(check.success, "{}", check.message);

    let goal = move_arm::send_goal(
        &harness,
        &move_arm::GoalRequestData {
            arm_name: "left_arm".to_string(),
            position: raised,
            orientation,
            duration_s: 1.0,
            plan_position_tolerance_m: 0.0,
            plan_orientation_tolerance_rad: 0.0,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal rejected: {:?}", goal.reason);
    let data = match goal.get_result(DEADLINE).await?.outcome {
        move_arm::ResultOutcome::Completed(data) => data,
        other => panic!("move_arm did not complete: {other:?}"),
    };
    assert!(data.success, "{}", data.message);
    assert_all_close("final_position", &data.final_position, &start, 1e-9);
    assert_all_close(
        "final_orientation",
        &data.final_orientation,
        &orientation,
        1e-9,
    );
    let short_by = data
        .final_position
        .iter()
        .zip(raised)
        .map(|(reported, goal)| (reported - goal).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        short_by > 0.04,
        "the result gives the goal, not the arm: {short_by} m"
    );
    harness.shutdown().await
}

/// Exposed gripper action end-to-end: `move_gripper` with the left gripper
/// played by a follower whose jaws lag their commands. The commanded ramp
/// lands on the target while the jaws are still most of the way out; the
/// result must wait until they stand still and report the opening they stopped
/// at, not the one measured when the last setpoint went out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_gripper_answers_with_the_opening_the_lagging_jaws_stopped_at() -> peppygen::Result<()>
{
    let (harness, mocks) = start_ready_vacant(params()).await?;

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_arm_at_home!(
        mocks.pairings.right_arm.joint_states,
        peppygen::paired_topics::right_arm::joint_states
    );
    pump_gripper_half_open!(
        mocks.pairings.right_gripper.gripper_states,
        peppygen::paired_topics::right_gripper::gripper_states
    );
    let followed = spawn_lagging_left_gripper_follower(
        mocks.pairings.left_gripper.gripper_states,
        mocks.pairings.left_gripper.gripper_setpoints,
    );

    // Gate the goal on streaming having begun, as the arm move above does: a
    // goal that reaches the coordinator during the seed wait is refused.
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    // Close from half open: the commanded ramp covers the 0.4 travel in about
    // 70 ms, the lagging jaws need 40 state periods.
    let target = 0.1;
    let data = complete_left_gripper_move(&harness, target, 0.0).await?;
    assert!(data.success, "move failed: {}", data.message);
    assert_eq!(data.message, "move complete");
    // The jaws stand still on the target once the result is out, so the
    // follower's last opening is the one the backbone measured at the end.
    assert_eq!(
        data.final_opening,
        *followed.borrow(),
        "final_opening must be the opening the jaws stopped at"
    );
    assert!(
        (data.final_opening - target).abs() < 1e-9,
        "the jaws stopped at {}, commanded {target}",
        data.final_opening
    );

    harness.shutdown().await
}

/// After a move_gripper ends, the node sends that gripper nothing new. Thus
/// the gripper keeps the move's opening and effort cap, and keeps squeezing
/// what it holds. Here:
/// - an object stops the jaws at 0.3 on their way to 0.0;
/// - the move ends with success, short of its target;
/// - through a later arm move, each setpoint that the gripper gets is still
///   0.0 at the effort cap of 1.5, which the node relays unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gripper_held_by_an_object_keeps_its_closing_command_after_the_move()
-> peppygen::Result<()> {
    let (harness, LeftFollowers { gripper, .. }) = start_with_left_followers(0.3).await?;

    let squeeze = GripperSetpoint {
        opening: 0.0,
        max_effort: 1.5,
    };
    let data = complete_left_gripper_move(&harness, squeeze.opening, squeeze.max_effort).await?;
    assert!(data.success, "{}", data.message);
    assert_eq!(
        data.message,
        "move complete: the gripper stopped at 0.300, short of the target 0.000"
    );
    assert_eq!(data.final_opening, 0.3);
    let during = gripper.received_since(0);
    assert!(!during.is_empty(), "the move commanded the gripper");
    for setpoint in &during {
        assert_eq!(setpoint.max_effort, squeeze.max_effort, "{during:?}");
    }

    let after_move = gripper.received_count();
    complete_left_arm_joint_move(&harness, [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0], 1.0).await?;
    for setpoint in gripper.received_since(after_move) {
        assert_eq!(setpoint, squeeze, "the gripper got a new command");
    }
    assert_eq!(gripper.last_received(), Some(squeeze));
    assert_eq!(*gripper.jaws.borrow(), 0.3, "the object stays held");
    harness.shutdown().await
}

/// The not-ready hold: while `robot_init` answers `ready: false` the backbone
/// must neither subscribe to the leading node nor stream a single setpoint;
/// flipping to ready opens the gate and governed streaming starts. The gate is
/// observed deterministically: the node cannot subscribe before the flip, so
/// the bounded windows measure observation cost, not luck.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_coordinator_holds_everything_until_robot_init_reports_ready() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params()).await?;

    let ready = Arc::new(AtomicBool::new(false));
    pump_is_ready(mocks.deps.robot_init.is_ready, ready.clone());
    // The follower pumps park on wait_for_subscriber while the gate holds.
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    // Not ready: the leading node finds no subscriber (the listener is behind
    // the gate), so there is nothing a streamed command could even reach.
    let leader = mocks.pairings.leader_left_arm.joint_setpoints;
    assert!(
        !leader.wait_for_subscriber(Duration::from_secs(2)).await?,
        "the backbone subscribed to the leader stream before the robot was ready"
    );
    // And the downstream wire stays silent: no publisher, no setpoints.
    let mut left_wire = mocks.pairings.left_arm.joint_setpoints;
    assert!(
        tokio::time::timeout(READ_WINDOW, left_wire.next())
            .await
            .is_err(),
        "a governed setpoint escaped while the robot was not ready"
    );

    // Flip: the next 500 ms readiness poll passes, the listeners subscribe,
    // the pumps un-park and seed both arms and both grippers, and governed
    // streaming begins.
    ready.store(true, Ordering::SeqCst);
    assert!(
        leader.wait_for_subscriber(DEADLINE).await?,
        "the backbone never subscribed to the leader stream after ready"
    );
    let first = tokio::time::timeout(DEADLINE, left_wire.next())
        .await
        .expect("no governed setpoint streamed after the robot became ready")?
        .expect("left arm subscription open");
    assert_eq!(first.positions.len(), 7);

    harness.shutdown().await
}

/// The collision governor as a black box: with a band widened until the rest
/// pose already sits under `d_stop`, a leading node driving both wrists toward
/// the centerline commands a closing motion the governor must fully deny, and
/// the emitted `collision_status` readout must report it `stopped`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_closing_command_inside_d_stop_reads_stopped_on_collision_status() -> peppygen::Result<()>
{
    // Widen the band: the rest-pose clearance (a few cm) is far under d_stop.
    // Thus the governor denies any closing candidate outright (Stopped, not
    // Throttling), and the measured-state tripwire is armed from the first tick.
    let mut wide = params();
    wide.d_stop_m = 0.8;
    wide.d_safe_m = 1.0;
    let d_stop_m = wide.d_stop_m;
    let (mut harness, mocks) = start_ready_vacant(wide).await?;

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    // Both wrists commanded toward the centerline (the governor unit tests'
    // own closing geometry: left j3 positive, right j3 negative), so the
    // candidate motion unambiguously closes the gap.
    let left_leader = mocks.pairings.leader_left_arm.joint_setpoints;
    let right_leader = mocks.pairings.leader_right_arm.joint_setpoints;
    let mut inward_left = HOME;
    inward_left[2] = 1.5;
    inward_left[3] = 0.4;
    let mut inward_right = HOME;
    inward_right[2] = -1.5;
    inward_right[3] = 0.4;

    let publish_commands = || async {
        left_leader.publish(&leader_command(inward_left)).await?;
        right_leader
            .publish(
                &peppygen::paired_topics::leader_right_arm::joint_setpoints::Message {
                    timestamp: SystemTime::now(),
                    positions: inward_right.to_vec(),
                    velocities: Vec::new(),
                    efforts: Vec::new(),
                },
            )
            .await
    };

    // The ~20 Hz readout publishes the guard continuously; drain it until the
    // commanded motion reads stopped (the first messages may predate the
    // command and read clear).
    let deadline = tokio::time::Instant::now() + DEADLINE;
    publish_commands().await?;
    loop {
        match tokio::time::timeout(
            READ_WINDOW,
            harness.emitted.collision_status_collision_status.next(),
        )
        .await
        {
            Ok(status) => {
                let status = status?.expect("collision_status subscription open");
                assert!(status.distance.is_finite());
                if status.stopped {
                    assert!(
                        status.distance < d_stop_m,
                        "stopped outside the stop floor: d={}",
                        status.distance
                    );
                    assert!(!status.link_a.is_empty() && !status.link_b.is_empty());
                    break;
                }
            }
            Err(_) => publish_commands().await?,
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "collision_status never reported the closing command stopped"
        );
    }

    harness.shutdown().await
}

/// The whole-robot limb_state readout: with every follower live (both arms at
/// [`HOME`], both grippers half open), the emitted snapshot names both limbs,
/// slices per the contract's rule, and carries the measured values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limb_states_snapshots_carry_names_counts_and_measured_state() -> peppygen::Result<()> {
    let (mut harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    let snapshot = next_snapshot(&mut harness).await?;

    assert_eq!(snapshot.arm_names, ["left_arm", "right_arm"]);
    assert_eq!(snapshot.joints_per_arm, [7, 7]);
    assert_eq!(snapshot.joint_positions.len(), 14);
    // Both arms park at HOME, so each 7-entry slice echoes the pumps.
    for (i, v) in snapshot.joint_positions.iter().enumerate() {
        assert!(
            (v - HOME[i % 7]).abs() < 1e-9,
            "joint_positions[{i}] = {v}, pumped {}",
            HOME[i % 7]
        );
    }
    assert_eq!(snapshot.positions.len(), 6, "3 per arm");
    assert_eq!(snapshot.orientations.len(), 8, "4 per arm");
    for (arm, quat) in snapshot.orientations.chunks(4).enumerate() {
        let norm = quat.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "arm {arm} orientation is not a unit quaternion (norm {norm})"
        );
    }
    assert_eq!(snapshot.gripper_names, ["left_gripper", "right_gripper"]);
    assert_eq!(snapshot.gripper_openings, [0.5, 0.5]);
    Ok(())
}

/// The same name tables, answered on demand with the robot reporting
/// not-ready and every follower silent: the service is the source a consumer
/// reads a robot's limbs from, and it carries the joint counts that size the
/// snapshot's arrays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limb_names_are_answered_before_the_robot_is_ready() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::limb_state::get_limb_names;

    let (harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(false)),
    );

    let names = get_limb_names::poll(&harness, DEADLINE).await?;

    assert_eq!(names.arm_names, ["left_arm", "right_arm"]);
    assert_eq!(names.joints_per_arm, [7, 7]);
    assert_eq!(names.gripper_names, ["left_gripper", "right_gripper"]);
    harness.shutdown().await
}

/// A goal naming no limb of this robot is refused at admission with the name
/// table quoted, for the arm and gripper moves alike.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn goals_for_unknown_limb_names_are_refused() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    // Wait for streaming so the refusal below is the name check, not the
    // seed gate's blanket refusal.
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    let goal = move_arm_joints::send_goal(
        &harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "torso".to_string(),
            joint_positions: HOME,
            duration_s: 1.0,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(!goal.accepted, "a goal for \"torso\" must be refused");
    assert_eq!(
        goal.reason.as_deref(),
        Some(r#"unknown arm_name: this robot's arms are "left_arm" and "right_arm""#)
    );

    // A gripper goal addressed with an ARM name is a mixup, not a gripper.
    let goal = move_gripper::send_goal(
        &harness,
        &move_gripper::GoalRequestData {
            gripper_name: "left_arm".to_string(),
            opening: 0.5,
            max_effort: 0.0,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(
        !goal.accepted,
        "a gripper goal for \"left_arm\" must be refused"
    );
    assert_eq!(
        goal.reason.as_deref(),
        Some(
            r#"unknown gripper_name: this robot's grippers are "left_gripper" and "right_gripper""#
        )
    );
    Ok(())
}

/// The stop service ends the moves in flight, whoever started them: an arm
/// move and a gripper move both end as cancelled with the stop's message,
/// the answer names both limbs, a stop with nothing moving names none, and
/// a goal that arrives after the stop runs to its end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_ends_every_move_in_flight_and_a_later_goal_runs() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::limb_motion::stop;

    let (harness, LeftFollowers { arm, gripper }) = start_with_left_followers(0.0).await?;

    // Nothing moves: the stop stops nothing.
    let idle = stop::poll(
        &harness,
        &stop::RequestData {
            reason: "nothing to stop".to_string(),
        },
        DEADLINE,
    )
    .await?;
    assert!(idle.success, "{}", idle.message);
    assert!(idle.stopped.is_empty(), "{:?}", idle.stopped);
    assert_eq!(idle.message, "nothing was moving");

    // A long arm move and a gripper move whose jaws lag: both are in flight
    // when the stop arrives. The stop is gated on the jaws having moved,
    // so it ends moves that run, not ones that wait to be admitted.
    let target = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
    let arm_goal = move_arm_joints::send_goal(
        &harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "left_arm".to_string(),
            joint_positions: target,
            duration_s: 30.0,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(arm_goal.accepted, "goal rejected: {:?}", arm_goal.reason);
    let gripper_goal = move_gripper::send_goal(
        &harness,
        &move_gripper::GoalRequestData {
            gripper_name: "left_gripper".to_string(),
            opening: 0.1,
            max_effort: 0.0,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(
        gripper_goal.accepted,
        "goal rejected: {:?}",
        gripper_goal.reason
    );
    let mut moving = gripper.jaws.clone();
    tokio::time::timeout(
        DEADLINE,
        moving.wait_for(|opening| (opening - 0.5).abs() > 0.05),
    )
    .await
    .expect("the jaws never started moving")
    .expect("follower watch open");

    let answer = stop::poll(
        &harness,
        &stop::RequestData {
            reason: "the operator asked".to_string(),
        },
        DEADLINE,
    )
    .await?;
    assert!(answer.success, "{}", answer.message);
    assert_eq!(answer.stopped, ["left_arm", "left_gripper"]);
    assert_eq!(answer.message, "stopped left_arm, left_gripper");

    let result = arm_goal.get_result(DEADLINE).await?;
    let data = match result.outcome {
        move_arm_joints::ResultOutcome::Cancelled(data) => data,
        other => panic!("the stopped arm move did not end cancelled: {other:?}"),
    };
    assert!(!data.success);
    assert_eq!(data.message, "stopped: the operator asked");
    let result = gripper_goal.get_result(DEADLINE).await?;
    let data = match result.outcome {
        move_gripper::ResultOutcome::Cancelled(data) => data,
        other => panic!("the stopped gripper move did not end cancelled: {other:?}"),
    };
    assert!(!data.success);
    assert_eq!(data.message, "stopped: the operator asked");
    // The arm holds short of the target it was on its way to.
    let held = *arm.borrow();
    assert!(
        (held[1] - target[1]).abs() > 0.1,
        "the arm went on to the target after the stop: {held:?}"
    );

    // The stop does not latch: a goal that arrives after it runs to its end.
    complete_left_arm_joint_move(&harness, target, 1.0).await?;

    harness.shutdown().await
}

/// robot.stop opens no gripper. Here a settled move closed a gripper to
/// 0.1. A stop names no gripper. Through a later arm move, the last opening
/// that the gripper got is still 0.1, where its jaws stay.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_leaves_a_settled_gripper_closed() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::limb_motion::stop;

    let (harness, LeftFollowers { gripper, .. }) = start_with_left_followers(0.0).await?;

    let data = complete_left_gripper_move(&harness, 0.1, 0.0).await?;
    assert!(data.success, "{}", data.message);
    assert_eq!(*gripper.jaws.borrow(), 0.1);

    let before_stop = gripper.received_count();
    let answer = stop::poll(
        &harness,
        &stop::RequestData {
            reason: "the operator asked".to_string(),
        },
        DEADLINE,
    )
    .await?;
    assert!(answer.success, "{}", answer.message);
    assert!(
        !answer.stopped.iter().any(|limb| limb == "left_gripper"),
        "{:?}",
        answer.stopped
    );

    complete_left_arm_joint_move(&harness, [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0], 1.0).await?;
    for setpoint in gripper.received_since(before_stop) {
        assert_eq!(setpoint.opening, 0.1, "the gripper got a new opening");
    }
    assert_eq!(gripper.last_received().map(|s| s.opening), Some(0.1));
    assert_eq!(*gripper.jaws.borrow(), 0.1, "the jaws stay closed");
    harness.shutdown().await
}

/// A posture result gives arm_names and, in that order, the grasp pose of
/// each arm measured when the goal completes. limb_state gives the same
/// pose for the same joints. Its success and its message say that the
/// move's time ran out, not that the arms arrived. Here both arms stand
/// still at HOME, so move_to_ready succeeds with neither arm at Ready.
/// move_to_home gives its result the same way.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_result_reports_each_arms_measured_grasp_point() -> peppygen::Result<()> {
    let (mut harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);
    let snapshot = next_snapshot(&mut harness).await?;

    let ready = complete_posture!(harness, move_to_ready, 0.0);
    let home = complete_posture!(harness, move_to_home, 0.0);
    for (result, done) in [
        (ready, "the move to ready ran its time"),
        (home, "the move to home ran its time"),
    ] {
        assert!(result.success, "{}", result.message);
        assert_eq!(result.message, done);
        assert_eq!(result.arm_names, ["left_arm", "right_arm"]);
        assert_all_close("positions", &result.positions, &snapshot.positions, 1e-9);
        assert_all_close(
            "orientations",
            &result.orientations,
            &snapshot.orientations,
            1e-9,
        );
    }
    harness.shutdown().await
}

/// Before the arms report their first state, a posture goal ends failed
/// with no grasp pose, and its message says why.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_goal_before_the_first_measurement_reports_no_pose() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    // No follower reports, so the coordinator waits for the first states.
    let result = complete_posture!(harness, move_to_ready, 1.0);
    assert!(!result.success);
    assert!(
        result.message.ends_with(
            "the follower has not reported its first state yet; \
             no arm poses: left_arm has not measured its joints"
        ),
        "{}",
        result.message
    );
    assert!(result.arm_names.is_empty(), "{:?}", result.arm_names);
    assert!(result.positions.is_empty() && result.orientations.is_empty());
    harness.shutdown().await
}

/// A posture goal whose duration is above the robot's limit is refused at
/// admission, for both postures.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_goal_above_the_duration_ceiling_is_refused() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    assert_posture_refused!(harness, move_to_ready, 601.0);
    assert_posture_refused!(harness, move_to_home, 601.0);
    harness.shutdown().await
}

/// After robot.stop, a posture result gives where each arm stopped:
/// - the left arm, played by a follower that the test brakes on its way to
///   Ready;
/// - the right arm, which stands at HOME.
///
/// The test waits until the robot measures the braked left arm before the
/// stop. Thus each arm rests where the robot measured it. The goal ends
/// cancelled, with success false and with the poses that limb_state gives
/// for those joints.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopped_posture_reports_where_the_arms_stopped() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::postures::move_to_ready;
    use peppygen::fixtures::exposed_services::limb_motion::stop;

    let (mut harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let left = spawn_arm_follower!(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);
    next_snapshot(&mut harness).await?;

    let goal = send_posture_goal!(harness, move_to_ready, 30.0);
    assert!(goal.accepted, "goal rejected: {:?}", goal.reason);
    let mut moving = left.followed.clone();
    tokio::time::timeout(
        DEADLINE,
        moving.wait_for(|q| q.iter().zip(HOME).any(|(q, home)| (q - home).abs() > 0.01)),
    )
    .await
    .expect("the left arm never started moving")
    .expect("follower watch open");

    // The arm stops where it stands; the stop comes once the robot measures
    // it there.
    let stopped_at = left.brake().await;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    let snapshot = loop {
        let snapshot = next_snapshot(&mut harness).await?;
        if snapshot.joint_positions[..7] == stopped_at {
            break snapshot;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the robot never measured the braked arm at {stopped_at:?}"
        );
    };
    let answer = stop::poll(
        &harness,
        &stop::RequestData {
            reason: "the operator asked".to_string(),
        },
        DEADLINE,
    )
    .await?;
    assert!(answer.success, "{}", answer.message);
    assert_eq!(answer.stopped, ["left_arm", "right_arm"]);

    let data = match goal.get_result(DEADLINE).await?.outcome {
        move_to_ready::ResultOutcome::Cancelled(data) => data,
        other => panic!("the stopped posture did not end cancelled: {other:?}"),
    };
    assert!(!data.success);
    assert_eq!(data.message, "left: stopped: the operator asked");
    assert_eq!(data.arm_names, ["left_arm", "right_arm"]);
    assert_all_close("positions", &data.positions, &snapshot.positions, 1e-9);
    assert_all_close(
        "orientations",
        &data.orientations,
        &snapshot.orientations,
        1e-9,
    );
    harness.shutdown().await
}

/// A posture move sends no command to the grippers. A gripper that a
/// move_gripper left at 0.2 stays there through move_to_ready and
/// move_to_home. It gets no opening other than that one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_move_leaves_the_grippers_alone() -> peppygen::Result<()> {
    let (harness, LeftFollowers { gripper, .. }) = start_with_left_followers(0.0).await?;

    let data = complete_left_gripper_move(&harness, 0.2, 0.0).await?;
    assert!(data.success, "{}", data.message);
    assert_eq!(*gripper.jaws.borrow(), 0.2);
    let after_move = gripper.received_count();

    let ready = complete_posture!(harness, move_to_ready, 1.0);
    assert!(ready.success, "{}", ready.message);
    assert_eq!(*gripper.jaws.borrow(), 0.2, "move_to_ready moved the jaws");
    let home = complete_posture!(harness, move_to_home, 1.0);
    assert!(home.success, "{}", home.message);
    assert_eq!(*gripper.jaws.borrow(), 0.2, "move_to_home moved the jaws");
    for setpoint in gripper.received_since(after_move) {
        assert_eq!(
            setpoint.opening, 0.2,
            "a posture move commanded the gripper"
        );
    }
    harness.shutdown().await
}

/// The plan check answers whether a Cartesian goal has a plan and moves
/// nothing: a pose a little above the grasp point has one, with the time the
/// move would take; a pose two metres away is refused with the words a
/// `move_arm` refusal gives; an unknown arm is refused; and the arm mock
/// observes no motion throughout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plan_check_answers_without_moving_the_arm() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::limb_motion::check_arm_move;

    let (mut harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);

    // Where the left grasp point stands, from the robot's own snapshot.
    let snapshot = next_snapshot(&mut harness).await?;
    let position = [
        snapshot.positions[0],
        snapshot.positions[1],
        snapshot.positions[2],
    ];
    let orientation = [
        snapshot.orientations[0],
        snapshot.orientations[1],
        snapshot.orientations[2],
        snapshot.orientations[3],
    ];
    let check = |arm_name: &str, position: [f64; 3]| check_arm_move::RequestData {
        arm_name: arm_name.to_string(),
        position,
        orientation,
        duration_s: 0.0,
        plan_position_tolerance_m: 0.0,
        plan_orientation_tolerance_rad: 0.0,
    };

    let near = check("left_arm", [position[0], position[1], position[2] + 0.02]);
    let answer = check_arm_move::poll(&harness, &near, DEADLINE).await?;
    assert!(answer.success, "{}", answer.message);
    assert!(answer.duration_s > 0.0, "{}", answer.duration_s);

    let far = check("left_arm", [position[0] + 2.0, position[1], position[2]]);
    let answer = check_arm_move::poll(&harness, &far, DEADLINE).await?;
    assert!(!answer.success);
    assert!(
        answer.message.starts_with("goal pose not planned within"),
        "{}",
        answer.message
    );
    assert_eq!(answer.duration_s, 0.0);

    let unknown = check("middle_arm", position);
    let answer = check_arm_move::poll(&harness, &unknown, DEADLINE).await?;
    assert!(!answer.success);
    assert!(
        answer.message.starts_with("unknown arm_name"),
        "{}",
        answer.message
    );

    // Nothing moved: the follower still adopts HOME from every setpoint.
    let observed = *followed.borrow();
    for (joint, (observed, home)) in observed.iter().zip(HOME.iter()).enumerate() {
        assert!(
            (observed - home).abs() < 1e-9,
            "joint {joint} moved to {observed} during the checks"
        );
    }
    harness.shutdown().await
}

/// The camera mounts service answers from bringup: refused until the
/// coordinator has measured the arms, then every camera of the generation
/// under the stamp of the snapshot the poses come from, a v1 robot with no
/// camera at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn camera_poses_are_answered_once_the_arms_are_measured() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::camera_mounts::get_camera_poses;

    let (harness, mocks) = start_ready_vacant(params()).await?;
    let ready = Arc::new(AtomicBool::new(false));
    pump_is_ready(mocks.deps.robot_init.is_ready, ready.clone());

    let refused = get_camera_poses::poll(&harness, DEADLINE).await?;
    assert!(!refused.success);
    assert_eq!(refused.message, "the robot has not measured its joints yet");
    assert!(refused.camera_names.is_empty());

    ready.store(true, Ordering::SeqCst);
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    let answer = loop {
        let answer = get_camera_poses::poll(&harness, DEADLINE).await?;
        if answer.success {
            break answer;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "no measurement reached the service: {}",
            answer.message
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    // v1 carries no camera: a success with nothing listed.
    assert_eq!(answer.message, "0 cameras");
    assert!(answer.camera_names.is_empty() && answer.carried_by.is_empty());
    assert!(answer.positions.is_empty() && answer.orientations.is_empty());
    harness.shutdown().await
}

/// A v2 robot lists its three cameras, the chest camera at the design's
/// numbers and each wrist camera carried by its arm, under the stamp of a
/// limb_states snapshot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_v2_robot_lists_its_three_cameras_in_the_robot_frame() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::camera_mounts::get_camera_poses;

    let (mut harness, mocks) = start_ready_vacant(v2_params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states
    );
    pump_right_arm_and_grippers!(mocks);
    let snapshot = next_snapshot(&mut harness).await?;

    let answer = get_camera_poses::poll(&harness, DEADLINE).await?;
    assert!(answer.success, "{}", answer.message);
    assert!(answer.timestamp >= snapshot.timestamp);
    assert_eq!(answer.camera_names, ["wrist_left", "wrist_right", "chest"]);
    assert_eq!(answer.carried_by, ["left_arm", "right_arm", ""]);
    assert_eq!(answer.positions.len(), 9);
    assert_eq!(answer.orientations.len(), 12);
    assert_eq!(&answer.positions[6..9], [0.0792, 0.0315, 0.7941]);
    // The chest camera looks along its optical +Z, to the robot's front and
    // down: the image's +Y (down the picture) points forward and down.
    let [x, y, z, w] = [
        answer.orientations[8],
        answer.orientations[9],
        answer.orientations[10],
        answer.orientations[11],
    ];
    // Rotate the unit +Z by the quaternion: v' = q v q*.
    let view_z = 1.0 - 2.0 * (x * x + y * y);
    let view_x = 2.0 * (x * z + w * y);
    assert!(
        view_x > 0.4 && view_z < -0.8,
        "the chest camera looks forward and down: ({view_x}, {view_z})"
    );
    harness.shutdown().await
}

/// Answers every poll of a request-less mock service with `$answer`, until
/// the mock's session closes.
macro_rules! pump_answers {
    ($service:expr, $answer:expr) => {{
        let service = $service;
        let answer = $answer;
        tokio::spawn(async move {
            while let Ok(responder) = service.next_request(PUMP_TIMEOUT).await {
                if responder.respond(answer.clone()).await.is_err() {
                    break;
                }
            }
        });
    }};
}

/// Plays the mock bound to `perception_geometry` as the chest camera's
/// camera_geometry, answering `$colour` and `$depth`, and holds the robot not
/// ready. The other mocks stay with the test.
macro_rules! answer_as_the_chest_camera {
    ($mocks:ident, $colour:expr, $depth:expr) => {
        let geometry = $mocks
            .deps
            .perception_geometry
            .expect("a bound perception_geometry slot starts its mock");
        pump_answers!(geometry.get_color_intrinsics, $colour);
        pump_answers!(geometry.get_depth_intrinsics, $depth);
        pump_is_ready(
            $mocks.deps.robot_init.is_ready,
            Arc::new(AtomicBool::new(false)),
        );
    };
}

/// The v2 boot with the `perception_geometry` slot bound to its mock, which
/// a test plays as the chest camera's camera_geometry, and `collision_ctrl`
/// vacant.
async fn start_v2_with_camera_geometry()
-> peppygen::Result<(Harness, peppygen::fixtures::harness::Mocks)> {
    Harness::start_with(
        Config {
            parameters: Some(v2_params()),
            collision_ctrl_vacant: true,
            perception_geometry_vacant: false,
            ..Default::default()
        },
        openarm_backbone::setup,
    )
    .await
}

/// The chest camera's colour intrinsics as the simulated camera gives them:
/// 1280 by 720 pixels over a 52 degree vertical field of view, the optical
/// axis through the middle of the image.
fn chest_colour_intrinsics() -> get_color_intrinsics::ResponseData {
    let focal = 360.0 / 26f64.to_radians().tan();
    get_color_intrinsics::ResponseData {
        success: true,
        message: String::new(),
        width: 1280,
        height: 720,
        fx: focal,
        fy: focal,
        cx: 639.5,
        cy: 359.5,
        distortion_model: "none".to_string(),
        distortion: Vec::new(),
    }
}

/// The chest camera's depth intrinsics, aligned to the colour stream and
/// measuring depths from 0.1 to 10 m.
fn chest_depth_intrinsics() -> get_depth_intrinsics::ResponseData {
    let colour = chest_colour_intrinsics();
    get_depth_intrinsics::ResponseData {
        success: true,
        message: String::new(),
        width: colour.width,
        height: colour.height,
        fx: colour.fx,
        fy: colour.fy,
        cx: colour.cx,
        cy: colour.cy,
        distortion_model: colour.distortion_model,
        distortion: colour.distortion,
        depth_model: "z".to_string(),
        min_depth_m: 0.1,
        max_depth_m: 10.0,
        align_mode: "depth_to_color".to_string(),
    }
}

/// What a camera that gives colour alone answers when asked for its depth
/// intrinsics.
fn no_depth_intrinsics() -> get_depth_intrinsics::ResponseData {
    get_depth_intrinsics::ResponseData {
        success: false,
        message: "this camera has no depth stream".to_string(),
        width: 0,
        height: 0,
        fx: 0.0,
        fy: 0.0,
        cx: 0.0,
        cy: 0.0,
        distortion_model: String::new(),
        distortion: Vec::new(),
        depth_model: String::new(),
        min_depth_m: 0.0,
        max_depth_m: 0.0,
        align_mode: String::new(),
    }
}

/// With no camera geometry linked, describe_workspace answers from the
/// design before the robot is ready: v2 still names the chest camera as its
/// perception camera, reach alone decides, no part is said to be seen, and
/// the message says why. A height that is not a number is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_camera_geometry_the_workspace_is_described_without_a_view_check()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::describe_workspace;

    let (harness, mocks) = start_ready_vacant(v2_params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(false)),
    );

    let table = describe_workspace::RequestData {
        surface_height: 0.45,
    };
    let answer = describe_workspace::poll(&harness, &table, DEADLINE).await?;
    assert!(answer.success, "{}", answer.message);
    assert_eq!(answer.perception_camera, "chest");
    assert!(answer.workable && answer.area > 0.0, "{}", answer.message);
    assert!(answer.rectangle.is_some() && answer.reach.is_some());
    assert_eq!(answer.view, None);
    assert!(
        answer.message.ends_with(
            "The view is not checked: no camera geometry is linked for the chest camera."
        ),
        "{}",
        answer.message
    );

    let not_a_number = describe_workspace::RequestData {
        surface_height: f64::NAN,
    };
    let refused = describe_workspace::poll(&harness, &not_a_number, DEADLINE).await?;
    assert!(!refused.success);
    assert_eq!(refused.message, "surface_height must be a finite number");
    harness.shutdown().await
}

/// With the chest camera's geometry linked, check_positions judges each
/// point by reach and by what the camera sees through its own intrinsics: a
/// point 0.30 m ahead at table height is workable, one a metre ahead is out
/// of reach by how far the closest arm stops. A list that does not split
/// into points is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_camera_geometry_linked_check_positions_reports_what_the_chest_camera_sees()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(mocks, chest_colour_intrinsics(), chest_depth_intrinsics());

    let points = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45, 1.0, 0.0, 0.45],
    };
    let answer = check_positions::poll(&harness, &points, DEADLINE).await?;
    assert!(answer.success, "{}", answer.message);
    assert_eq!(answer.perception_camera, "chest");
    let [near, far] = answer.results.as_slice() else {
        panic!("one result per point: {:?}", answer.results);
    };
    assert_eq!(near.position, [0.30, 0.0, 0.45]);
    assert!(
        near.workable && near.reachable && near.in_view,
        "{}",
        near.message
    );
    assert_eq!((near.view.as_str(), near.short_by), ("seen", 0.0));
    assert_eq!(
        near.message,
        format!(
            "Workable: {} reaches it and the chest camera sees it.",
            near.arm
        )
    );
    assert_eq!(far.position, [1.0, 0.0, 0.45]);
    assert!(!far.workable && !far.reachable, "{}", far.message);
    assert!(
        far.arm.is_empty() && far.short_by > 0.01,
        "{}",
        far.short_by
    );
    assert!(!answer.all_workable);
    assert_eq!(answer.message, "1 of the 2 points is workable.");

    let ragged = check_positions::RequestData {
        positions: vec![0.30, 0.0],
    };
    let refused = check_positions::poll(&harness, &ragged, DEADLINE).await?;
    assert!(!refused.success && refused.results.is_empty());
    assert_eq!(
        refused.message,
        "positions must hold 3 values (x, y, z) per point"
    );
    harness.shutdown().await
}

/// A linked camera that gives no depth cannot be the chest camera, the
/// perception camera: the answer is refused, naming it and saying why.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_without_depth_is_refused_naming_the_chest_camera() -> peppygen::Result<()>
{
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(mocks, chest_colour_intrinsics(), no_depth_intrinsics());

    let point = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45],
    };
    let answer = check_positions::poll(&harness, &point, DEADLINE).await?;
    assert!(!answer.success);
    assert_eq!(
        answer.message,
        "the camera linked as the chest camera, the perception camera, gives no depth: \
         this camera has no depth stream"
    );
    assert!(answer.results.is_empty() && answer.perception_camera.is_empty());
    harness.shutdown().await
}

/// A linked camera whose depth stream gives a depth model camera_geometry:v1
/// does not name leaves the chest camera's depth range without a meaning:
/// the answer is refused, naming the model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_of_an_unknown_depth_model_is_refused_naming_the_model()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(
        mocks,
        chest_colour_intrinsics(),
        get_depth_intrinsics::ResponseData {
            depth_model: "disparity".to_string(),
            ..chest_depth_intrinsics()
        }
    );

    let point = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45],
    };
    let answer = check_positions::poll(&harness, &point, DEADLINE).await?;
    assert!(!answer.success);
    assert_eq!(
        answer.message,
        "cannot read the depth intrinsics of the chest camera, the perception camera: \
         unknown depth model 'disparity': camera_geometry:v1 names \"z\" and \"range\""
    );
    assert!(answer.results.is_empty() && answer.perception_camera.is_empty());
    harness.shutdown().await
}

/// What a simulated camera answers for its colour intrinsics before it
/// hears from the simulation.
fn unready_colour_intrinsics() -> get_color_intrinsics::ResponseData {
    get_color_intrinsics::ResponseData {
        success: false,
        message: "no camera geometry has been received from the simulation".to_string(),
        width: 0,
        height: 0,
        fx: 0.0,
        fy: 0.0,
        cx: 0.0,
        cy: 0.0,
        distortion_model: String::new(),
        distortion: Vec::new(),
    }
}

/// A linked camera that cannot give its colour intrinsics now leaves the
/// chest camera's view unknown: the answer is refused with the camera's
/// reason, and says nothing of what the robot reaches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_without_its_colour_intrinsics_is_refused_with_its_reason()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::describe_workspace;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(mocks, unready_colour_intrinsics(), no_depth_intrinsics());

    let table = describe_workspace::RequestData {
        surface_height: 0.45,
    };
    let answer = describe_workspace::poll(&harness, &table, DEADLINE).await?;
    assert!(!answer.success);
    assert_eq!(
        answer.message,
        "cannot read the colour intrinsics of the chest camera, the perception camera: \
         no camera geometry has been received from the simulation"
    );
    assert!(!answer.workable && answer.area == 0.0);
    assert_eq!(
        (answer.rectangle, answer.reach, answer.view),
        (None, None, None)
    );
    harness.shutdown().await
}
