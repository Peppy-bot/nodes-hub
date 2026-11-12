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

use peppygen::exposed_actions::limb_motion::execute_joint_path::{
    GoalArmsItem, GoalArmsItemWaypointsItem,
};
use peppygen::fixtures::exposed_actions::limb_motion::execute_joint_path;
use peppygen::fixtures::exposed_actions::limb_motion::{move_arm_joints, move_gripper};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::perception_geometry::{get_color_intrinsics, get_depth_intrinsics};
use tokio::sync::{mpsc, oneshot, watch};

mod helpers;
use helpers::MODEL;
use helpers::fixtures::posture;
use helpers::params::params;
use robot_description::states;

/// The arms of the robot the tests launch, by the names the wire carries, in
/// the order of limb_state's arm names.
const ARM_NAMES: [&str; 2] = ["left_arm", "right_arm"];

/// The model of the v2 robot, whose design carries the cameras.
const V2: &str = "openarm_v2";

/// The posture `state` of the arm `arm_name` of the model the tests launch.
fn launched_posture(arm_name: &str, state: &str) -> [f64; 7] {
    posture(MODEL, arm_name, state)
}

/// How long a mock pump waits parked for the node's next poll or for the
/// node's subscription to appear; only expires once the harness is gone.
const PUMP_TIMEOUT: Duration = Duration::from_secs(120);

/// Follower back-channel period; must stay well inside the 80 ms follower
/// stale window or the coordinator freezes that limb.
const STATE_PUMP_PERIOD: Duration = Duration::from_millis(10);

/// The Home posture of the arm `arm_name` of the launched model, where the
/// tests seed its follower: inside every joint's limits, so the planner's
/// limit clamp leaves it as it is and held setpoints echo it bit-exact. Its
/// nearest checked pair sits outside the validated band on v1 (the governor
/// test `far_apart_is_unthrottled`).
fn home_of(arm_name: &str) -> [f64; 7] {
    launched_posture(arm_name, states::HOME)
}

/// The Home posture of the left arm ([`home_of`]).
fn left_home() -> [f64; 7] {
    home_of("left_arm")
}

/// The Home posture of the right arm ([`home_of`]).
fn right_home() -> [f64; 7] {
    home_of("right_arm")
}

/// Outer deadline for every bounded convergence loop.
const DEADLINE: Duration = Duration::from_secs(90);

/// Per-read window inside a convergence loop: several coordinator readout /
/// stream periods, so an elapsed window means "nothing arrived", not jitter.
const READ_WINDOW: Duration = Duration::from_secs(2);

/// The names of the cameras openarm_v2's description carries, in its order.
fn v2_camera_names() -> Vec<String> {
    robot_description::model("openarm_v2")
        .expect("a bundled model")
        .description()
        .cameras()
        .iter()
        .map(|camera| camera.name.clone())
        .collect()
}

/// Answers the `robot_identity/get_identity` call the node makes at start with
/// `model` as the robot's model, and returns once it answered.
async fn answer_identity(
    mocks: &peppygen::fixtures::harness::Mocks,
    model: &str,
) -> peppygen::Result<()> {
    use peppygen::mock::deps::robot_identity::get_identity::ResponseData;
    mocks
        .deps
        .robot_identity
        .get_identity
        .next_request(DEADLINE)
        .await?
        .respond(ResponseData {
            robot: "openarm".to_owned(),
            model: model.to_owned(),
            core_node: "core".to_owned(),
        })
        .await
}

/// Answers the `robot_identity/get_description` call the node makes at start
/// with the description of `model` as robot_description bundles it, or with
/// `digest` in place of its digest, and returns once it answered.
async fn answer_description(
    mocks: &peppygen::fixtures::harness::Mocks,
    model: &str,
    digest: Option<&str>,
) -> peppygen::Result<()> {
    use peppygen::mock::deps::robot_identity::get_description::ResponseData;
    let bundled = robot_description::model(model).expect("a bundled model");
    mocks
        .deps
        .robot_identity
        .get_description
        .next_request(DEADLINE)
        .await?
        .respond(ResponseData {
            urdf: bundled.urdf().to_owned(),
            srdf: bundled.srdf().to_owned(),
            meshes: Vec::new(),
            digest: digest.unwrap_or(bundled.digest()).to_owned(),
        })
        .await
}

/// Answers the node's questions of robot_identity at start as the initializer of a
/// robot of the model `model` built with this backbone's libraries answers them.
async fn answer_robot_identity(
    mocks: &peppygen::fixtures::harness::Mocks,
    model: &str,
) -> peppygen::Result<()> {
    answer_identity(mocks, model).await?;
    answer_description(mocks, model, None).await
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

/// Static joint-state pump for one arm follower, parked at `$home`, its arm's
/// Home ([`home_of`]).
macro_rules! pump_arm_at_home {
    ($publisher:expr, $message:path, $home:expr $(,)?) => {
        pump_states!($publisher, || {
            use $message as message;
            message::Message {
                timestamp: SystemTime::now(),
                positions: $home.to_vec(),
                velocities: vec![0.0; 7],
                efforts: Vec::new(),
            }
        })
    };
}

/// Static aperture pump for one gripper follower, half open.
macro_rules! pump_gripper_half_open {
    ($publisher:expr, $message:path) => {
        pump_states!($publisher, || {
            use $message as message;
            message::Message {
                timestamp: SystemTime::now(),
                opening: 0.5,
                effort: 0.0,
                max_effort: 1.0,
            }
        })
    };
}

/// The three static followers every test needs alongside whatever it does
/// with the left arm: right arm at [`home`], both grippers half open.
macro_rules! pump_right_arm_and_grippers {
    ($mocks:ident) => {
        pump_arm_at_home!(
            $mocks.pairings.right_arm.joint_states,
            peppygen::paired_topics::right_arm::joint_states,
            right_home(),
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

/// An arm played by [`spawn_arm_follower`]: the latest position it adopted,
/// every setpoint it received, in order, and its brake.
struct ArmFollower {
    followed: watch::Receiver<[f64; 7]>,
    received: Arc<Mutex<Vec<[f64; 7]>>>,
    brake: mpsc::Sender<oneshot::Sender<[f64; 7]>>,
}

impl ArmFollower {
    /// Every setpoint the arm received so far.
    fn received(&self) -> Vec<[f64; 7]> {
        self.received
            .lock()
            .expect("no panic while recording")
            .clone()
    }

    /// Wait until the arm adopts its first setpoint: the backbone streams to
    /// an arm only while its follower delivers fresh states, so a goal sent
    /// then finds the arm's measurement fresh.
    async fn first_setpoint(&self) {
        let mut followed = self.followed.clone();
        tokio::time::timeout(DEADLINE, followed.changed())
            .await
            .expect("the backbone never streamed to the arm")
            .expect("the follower runs");
    }

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

/// Plays an arm as a follower that stands `$standing_error` off each
/// setpoint, as a servo held against gravity stands off its command,
/// starting at [`home`]:
/// - it publishes its measured state, a `$message`, every
///   [`STATE_PUMP_PERIOD`] on `$states`;
/// - it adopts each governed setpoint that the node streams down
///   `$setpoints`, plus the standing error, as the new measurement, until
///   the test brakes it.
///
/// Its `followed` watch carries the latest adopted position, and `received`
/// every setpoint in order. Thus a test can assert what motion the arm mock
/// observed.
macro_rules! spawn_arm_follower {
    ($states:expr, $setpoints:expr, $message:path, $standing_error:expr, $home:expr $(,)?) => {{
        let states = $states;
        let mut setpoints = $setpoints;
        let standing_error: [f64; 7] = $standing_error;
        let home: [f64; 7] = $home;
        let (tx, followed) = watch::channel(home);
        let (brake, mut brakes) = mpsc::channel::<oneshot::Sender<[f64; 7]>>(1);
        let received = Arc::new(Mutex::new(Vec::new()));
        let record = received.clone();
        tokio::spawn(async move {
            if !matches!(states.wait_for_subscriber(PUMP_TIMEOUT).await, Ok(true)) {
                return;
            }
            let mut positions = home;
            let mut braked = false;
            let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        use $message as message;
                        let state = message::Message {
                            timestamp: SystemTime::now(),
                            positions: positions.to_vec(),
                            velocities: vec![0.0; 7],
                            efforts: Vec::new(),
                        };
                        if states.publish(&state).await.is_err() {
                            return;
                        }
                    }
                    received = setpoints.next() => match received {
                        Ok(Some(m)) if !braked && m.positions.len() == 7 => {
                            let setpoint: [f64; 7] =
                                m.positions.try_into().expect("seven joint positions");
                            positions = std::array::from_fn(|i| setpoint[i] + standing_error[i]);
                            record.lock().expect("no panic while recording").push(setpoint);
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
        ArmFollower {
            followed,
            received,
            brake,
        }
    }};
}

/// The standing error of a perfect follower: it stands at each setpoint.
const NO_STANDING_ERROR: [f64; 7] = [0.0; 7];

/// Plays the left arm as a perfect follower ([`spawn_arm_follower`]).
fn spawn_left_arm_follower(
    states: peppygen::mock::pairings::left_arm::joint_states::Publisher,
    setpoints: peppygen::mock::pairings::left_arm::joint_setpoints::Subscription,
) -> ArmFollower {
    spawn_left_arm_follower_off_by(states, setpoints, NO_STANDING_ERROR)
}

/// Plays the left arm as a follower that stands `standing_error` off each
/// setpoint ([`spawn_arm_follower`]).
fn spawn_left_arm_follower_off_by(
    states: peppygen::mock::pairings::left_arm::joint_states::Publisher,
    setpoints: peppygen::mock::pairings::left_arm::joint_setpoints::Subscription,
    standing_error: [f64; 7],
) -> ArmFollower {
    spawn_arm_follower!(
        states,
        setpoints,
        peppygen::paired_topics::left_arm::joint_states,
        standing_error,
        left_home(),
    )
}

/// Plays the right arm as a perfect follower ([`spawn_arm_follower`]).
fn spawn_right_arm_follower(
    states: peppygen::mock::pairings::right_arm::joint_states::Publisher,
    setpoints: peppygen::mock::pairings::right_arm::joint_setpoints::Subscription,
) -> ArmFollower {
    spawn_right_arm_follower_off_by(states, setpoints, NO_STANDING_ERROR)
}

/// Plays the right arm as a follower that stands `standing_error` off each
/// setpoint ([`spawn_arm_follower`]).
fn spawn_right_arm_follower_off_by(
    states: peppygen::mock::pairings::right_arm::joint_states::Publisher,
    setpoints: peppygen::mock::pairings::right_arm::joint_setpoints::Subscription,
    standing_error: [f64; 7],
) -> ArmFollower {
    spawn_arm_follower!(
        states,
        setpoints,
        peppygen::paired_topics::right_arm::joint_states,
        standing_error,
        right_home(),
    )
}

/// Jaw travel (opening fraction) the lagging gripper follower covers per
/// [`STATE_PUMP_PERIOD`]: 1.0 per second against the 6.0 per second the
/// backbone commands at, so the jaws arrive many state periods after the
/// commanded ramp has landed.
const LAGGING_GRIPPER_STEP: f64 = 0.01;

/// Jaw travel per [`STATE_PUMP_PERIOD`] of a gripper follower whose jaws
/// stand on the latest setpoint at each state it publishes: the whole jaw
/// travel. The jaws of such a gripper stand where the backbone commanded
/// them as its commanded travel ends, so where a close leaves them does not
/// depend on how fast the host runs.
const PROMPT_GRIPPER_STEP: f64 = 1.0;

/// One gripper setpoint as the follower receives it: the opening fraction
/// and the effort cap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GripperSetpoint {
    opening: f64,
    max_effort: f64,
}

/// A gripper played by [`spawn_gripper_follower`]: the opening it publishes,
/// the opening of the last setpoint it received, every setpoint it
/// received, in order, and whether it stopped reporting.
struct GripperFollower {
    jaws: watch::Receiver<f64>,
    commanded: watch::Receiver<f64>,
    received: Arc<Mutex<Vec<GripperSetpoint>>>,
    silent: Arc<AtomicBool>,
}

impl GripperFollower {
    /// Stop reporting: from now on the gripper publishes no state, as a
    /// gripper whose motor faulted. It still records the setpoints it gets.
    fn stop_reporting(&self) {
        self.silent.store(true, Ordering::SeqCst);
    }

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

    /// Wait until the gripper receives a setpoint under `opening`.
    async fn commanded_under(&self, opening: f64) {
        let mut commanded = self.commanded.clone();
        tokio::time::timeout(DEADLINE, commanded.wait_for(|setpoint| *setpoint < opening))
            .await
            .unwrap_or_else(|_| panic!("the gripper never got a setpoint under {opening}"))
            .expect("the follower runs");
    }

    /// Wait until the jaws stand at `opening`.
    async fn jaws_at(&self, opening: f64) {
        let mut jaws = self.jaws.clone();
        tokio::time::timeout(DEADLINE, jaws.wait_for(|stand| *stand == opening))
            .await
            .unwrap_or_else(|_| panic!("the jaws never stood at {opening}"))
            .expect("the follower runs");
    }
}

/// Plays a gripper as a follower whose jaws lag their commands: every
/// [`STATE_PUMP_PERIOD`] the opening moves at most `$step` toward the latest
/// setpoint the node streamed down `$setpoints`, landing on it exactly, and
/// is published on `$states` as the measured state, a `$message`. An object
/// between the jaws stops them at `$object_at`, the opening they cannot
/// close past (0.0 for no object). It starts half open like the static
/// gripper pumps, and records every setpoint it receives. It publishes no
/// state once the test makes it stop reporting.
macro_rules! spawn_gripper_follower {
    ($states:expr, $setpoints:expr, $message:path, $step:expr, $object_at:expr) => {{
        let states = $states;
        let mut setpoints = $setpoints;
        let step: f64 = $step;
        let object_at: f64 = $object_at;
        let mut opening = 0.5;
        let (tx, jaws) = watch::channel(opening);
        let (commanded_tx, commanded_opening) = watch::channel(opening);
        let received = Arc::new(Mutex::new(Vec::new()));
        let record = received.clone();
        let silent = Arc::new(AtomicBool::new(false));
        let silenced = silent.clone();
        tokio::spawn(async move {
            if !matches!(states.wait_for_subscriber(PUMP_TIMEOUT).await, Ok(true)) {
                return;
            }
            let mut commanded = opening;
            let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        if silenced.load(Ordering::SeqCst) {
                            continue;
                        }
                        use $message as message;
                        let remaining: f64 = commanded - opening;
                        let next = if remaining.abs() <= step {
                            commanded
                        } else {
                            opening + step.copysign(remaining)
                        };
                        opening = next.max(object_at);
                        let state = message::Message {
                            timestamp: SystemTime::now(),
                            opening,
                            effort: 0.0,
                            max_effort: 1.0,
                        };
                        if states.publish(&state).await.is_err() {
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
                            commanded_tx.send_replace(m.opening);
                        }
                        Err(_) => {}
                        Ok(None) => return,
                    }
                }
            }
        });
        GripperFollower {
            jaws,
            commanded: commanded_opening,
            received,
            silent,
        }
    }};
}

/// Plays the left gripper as a lagging follower whose jaws an object stops
/// at `object_at` ([`spawn_gripper_follower`]).
fn spawn_left_gripper_follower(
    states: peppygen::mock::pairings::left_gripper::gripper_states::Publisher,
    setpoints: peppygen::mock::pairings::left_gripper::gripper_setpoints::Subscription,
    object_at: f64,
) -> GripperFollower {
    spawn_left_gripper_follower_at(states, setpoints, LAGGING_GRIPPER_STEP, object_at)
}

/// Plays the left gripper as a follower whose jaws cover `step` per
/// [`STATE_PUMP_PERIOD`] and that an object stops at `object_at`
/// ([`spawn_gripper_follower`]).
fn spawn_left_gripper_follower_at(
    states: peppygen::mock::pairings::left_gripper::gripper_states::Publisher,
    setpoints: peppygen::mock::pairings::left_gripper::gripper_setpoints::Subscription,
    step: f64,
    object_at: f64,
) -> GripperFollower {
    spawn_gripper_follower!(
        states,
        setpoints,
        peppygen::paired_topics::left_gripper::gripper_states,
        step,
        object_at
    )
}

/// Plays the right gripper as a follower whose jaws cover `step` per
/// [`STATE_PUMP_PERIOD`], with nothing between them
/// ([`spawn_gripper_follower`]).
fn spawn_right_gripper_follower_at(
    states: peppygen::mock::pairings::right_gripper::gripper_states::Publisher,
    setpoints: peppygen::mock::pairings::right_gripper::gripper_setpoints::Subscription,
    step: f64,
) -> GripperFollower {
    spawn_gripper_follower!(
        states,
        setpoints,
        peppygen::paired_topics::right_gripper::gripper_states,
        step,
        0.0
    )
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

/// The full boot of a robot of `model` with both dependency slots vacant (their
/// `zero_or_one` empty binding): the launch-time band stands, exactly the
/// branch the cardinality exists for, and the workspace answers leave the view
/// unchecked.
async fn start_ready_vacant(
    parameters: peppygen::Parameters,
    model: &str,
) -> peppygen::Result<(Harness, peppygen::fixtures::harness::Mocks)> {
    let (harness, mocks, _fault) = helpers::start(Config {
        parameters: Some(parameters),
        collision_ctrl_vacant: true,
        perception_geometry_vacant: true,
        ..Default::default()
    })
    .await?;
    assert!(
        mocks.deps.collision_ctrl.is_none(),
        "a vacant collision_ctrl slot must start no mock"
    );
    assert!(
        mocks.deps.perception_geometry.is_none(),
        "a vacant perception_geometry slot must start no mock"
    );
    answer_robot_identity(&mocks, model).await?;
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
    gripper: GripperFollower,
}

/// Start the node with the robot ready and wait until streaming begins.
/// - The left arm is a perfect follower (see [`spawn_left_arm_follower`]).
/// - The left gripper lags its commands, and an object stops its jaws at
///   `object_at` (see [`spawn_left_gripper_follower`]).
/// - The right arm stands at [`home`] and the right gripper is half open.
async fn start_with_left_followers(object_at: f64) -> peppygen::Result<(Harness, LeftFollowers)> {
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let arm = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    )
    .followed;
    pump_arm_at_home!(
        mocks.pairings.right_arm.joint_states,
        peppygen::paired_topics::right_arm::joint_states,
        right_home(),
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

/// The first limb_states snapshot the node emits with each arm at its posture
/// `state` of the launched model, within 1e-9 rad in every joint.
async fn snapshot_with_both_arms_at(
    harness: &mut Harness,
    state: &str,
) -> peppygen::Result<peppygen::fixtures::emitted_topics::limb_state::limb_states::Message> {
    let joints: Vec<f64> = ARM_NAMES
        .into_iter()
        .flat_map(|arm_name| launched_posture(arm_name, state))
        .collect();
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let snapshot = next_snapshot(harness).await?;
        let at_posture = snapshot.joint_positions.len() == joints.len()
            && snapshot
                .joint_positions
                .iter()
                .zip(&joints)
                .all(|(measured, posture)| (measured - posture).abs() <= 1e-9);
        if at_posture {
            return Ok(snapshot);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the robot never measured both arms at {joints:?}"
        );
    }
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

/// The error reply that `answer` holds, read by `of`, the `Failure::of` of
/// the member. Fails the test on a response, or on an error that is not an
/// error reply of the member.
fn error_reply<T: std::fmt::Debug, F>(
    answer: peppygen::Result<T>,
    of: impl Fn(&peppygen::Error) -> Option<F>,
) -> F {
    match answer {
        Ok(response) => panic!("a response, not an error reply: {response:?}"),
        Err(e) => of(&e).unwrap_or_else(|| panic!("not an error reply of the member: {e}")),
    }
}

/// The result of a posture goal, as either postures action gives it.
#[derive(Debug)]
struct PostureResult {
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
            peppygen::DEFAULT_FEEDBACK_BOUND,
            DEADLINE,
        )
        .await?
    }};
}

/// Send one goal of `$action` with `$duration_s`, and assert that the node
/// refuses it with the failure kind `$kind` and the message `$reason`.
macro_rules! assert_posture_refused {
    ($harness:expr, $action:ident, $duration_s:expr, $kind:ident, $reason:expr) => {{
        use peppygen::fixtures::exposed_actions::postures::$action;
        let goal = send_posture_goal!($harness, $action, $duration_s);
        let refusal = goal.refusal.unwrap_or_else(|| {
            panic!(
                "a {} s {} must be refused",
                $duration_s,
                stringify!($action)
            )
        });
        assert_eq!(
            refusal,
            $action::Failure::new($action::FailureKind::$kind, $reason)
        );
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
            "{} refused: {:?}",
            stringify!($action),
            goal.refusal
        );
        match goal.get_result(DEADLINE).await?.outcome {
            $action::ResultOutcome::Completed(data) => PostureResult {
                arm_names: data.arm_names,
                positions: data.positions,
                orientations: data.orientations,
            },
            other => panic!("{} did not complete: {other:?}", stringify!($action)),
        }
    }};
}

/// Send one `move_arm_joints` goal for the left arm to `target` over
/// `duration_s`, and assert that it completes.
async fn complete_left_arm_joint_move(
    harness: &Harness,
    target: [f64; 7],
    duration_s: f64,
) -> peppygen::Result<move_arm_joints::ResultData> {
    match left_arm_joint_move(harness, target, duration_s).await? {
        move_arm_joints::ResultOutcome::Completed(data) => Ok(data),
        other => panic!("move_arm_joints did not complete: {other:?}"),
    }
}

/// Send one `move_arm_joints` goal for the left arm to `target` over
/// `duration_s`, assert that the node accepts it, and wait for how it ends.
async fn left_arm_joint_move(
    harness: &Harness,
    target: [f64; 7],
    duration_s: f64,
) -> peppygen::Result<move_arm_joints::ResultOutcome> {
    arm_joint_move(harness, "left_arm", target, duration_s).await
}

/// The right arm folded in front of the chest, its grasp point at about
/// [0.34, -0.195, 0.28] m, 0.39 m from the left arm's at its mirror image:
/// where the joint paths of the collision guard start, a small turn of a
/// shoulder from the other hand.
const FOLDED_IN_RIGHT: [f64; 7] = [0.15, 0.40, -0.48, 0.95, 0.0, 0.0, 0.0];

/// [`FOLDED_IN_RIGHT`]'s mirror image on the left arm: its first three
/// joints turned the other way.
const FOLDED_IN_LEFT: [f64; 7] = [-0.15, -0.40, 0.48, 0.95, 0.0, 0.0, 0.0];

/// Move the right arm to [`FOLDED_IN_RIGHT`] and the left arm to
/// [`FOLDED_IN_LEFT`], one `move_arm_joints` goal each, and assert that both
/// complete.
async fn fold_both_arms_in(harness: &Harness) -> peppygen::Result<()> {
    let folded_in = [("left_arm", FOLDED_IN_LEFT), ("right_arm", FOLDED_IN_RIGHT)];
    for (arm_name, target) in folded_in {
        match arm_joint_move(harness, arm_name, target, 0.0).await? {
            move_arm_joints::ResultOutcome::Completed(_) => {}
            other => panic!("move_arm_joints of {arm_name} did not complete: {other:?}"),
        }
    }
    Ok(())
}

/// Send one `move_arm_joints` goal for `arm_name` to `target` over
/// `duration_s`, assert that the node accepts it, and wait for how it ends.
async fn arm_joint_move(
    harness: &Harness,
    arm_name: &str,
    target: [f64; 7],
    duration_s: f64,
) -> peppygen::Result<move_arm_joints::ResultOutcome> {
    let goal = move_arm_joints::send_goal(
        harness,
        &move_arm_joints::GoalRequestData {
            arm_name: arm_name.to_string(),
            joint_positions: target,
            duration_s,
            settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    Ok(goal.get_result(DEADLINE).await?.outcome)
}

/// The settle the moves of these tests ask for: a slack the followers'
/// tracking stays within, and a timeout that keeps a failed settle short.
const SETTLE_TOLERANCE_RAD: f64 = 0.05;
const SETTLE_POSITION_TOLERANCE_M: f64 = 0.005;
const SETTLE_ORIENTATION_TOLERANCE_RAD: f64 = 0.05;
const SETTLE_TIMEOUT_S: f64 = 0.5;

/// The seconds a settle timeout's message names after the planned end. The
/// node formats the time it measured, so a stalled test host reads higher
/// than the settle timeout, never lower.
fn seconds_after_planned_end(message: &str) -> f64 {
    let (_, tail) = message
        .rsplit_once(", ")
        .expect("a settle failure names the time since the planned end");
    tail.strip_suffix(" s after the planned end")
        .expect("the time ends the message")
        .parse()
        .expect("a number of seconds")
}

/// The settle timeout the gripper moves of these tests ask for: well past the
/// lagging follower's longest travel here, half open to closed at
/// [`LAGGING_GRIPPER_STEP`] per [`STATE_PUMP_PERIOD`] (0.5 s), plus the still
/// window, so every move completes on stillness.
const GRIPPER_SETTLE_TIMEOUT_S: f64 = 2.0;

/// Send one `move_gripper` goal for the left gripper to `opening` under
/// `max_effort`, and wait for the result it completes with.
async fn complete_left_gripper_move(
    harness: &Harness,
    opening: f64,
    max_effort: f64,
) -> peppygen::Result<move_gripper::ResultData> {
    complete_left_gripper_move_within(harness, opening, max_effort, GRIPPER_SETTLE_TIMEOUT_S).await
}

/// Send one `move_gripper` goal for the left gripper to `opening` under
/// `max_effort` with `settle_timeout_s`, and wait for the result it
/// completes with.
async fn complete_left_gripper_move_within(
    harness: &Harness,
    opening: f64,
    max_effort: f64,
    settle_timeout_s: f64,
) -> peppygen::Result<move_gripper::ResultData> {
    complete_gripper_move_within(
        harness,
        "left_gripper",
        opening,
        max_effort,
        settle_timeout_s,
    )
    .await
}

/// Send one `move_gripper` goal for `gripper_name` to `opening` under
/// `max_effort` with `settle_timeout_s`, and wait for the result it
/// completes with.
async fn complete_gripper_move_within(
    harness: &Harness,
    gripper_name: &str,
    opening: f64,
    max_effort: f64,
    settle_timeout_s: f64,
) -> peppygen::Result<move_gripper::ResultData> {
    let goal = move_gripper::send_goal(
        harness,
        &move_gripper::GoalRequestData {
            gripper_name: gripper_name.to_string(),
            opening,
            max_effort,
            settle_timeout_s,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    match goal.get_result(DEADLINE).await?.outcome {
        move_gripper::ResultOutcome::Completed(data) => Ok(data),
        other => panic!("move_gripper of {gripper_name} did not complete: {other:?}"),
    }
}

/// Readiness gate + minimal fan-through: with the robot ready, `collision_ctrl`
/// vacant and only the left leader driving, the leading node's joint command
/// crosses the whole pipeline (listener -> chase -> governor -> pairing wire)
/// to the left arm mock, while the uncommanded right arm holds its seeded pose
/// bit-exact and the unselected slots stay silent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leader_command_fans_through_to_the_governed_arm_wire() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);

    let mut left_wire = mocks.pairings.left_arm.joint_setpoints;
    let mut right_wire = mocks.pairings.right_arm.joint_setpoints;
    let leader = mocks.pairings.leader_left_arm.joint_setpoints;

    // Streaming begins (both arms + both grippers seeded): before any command,
    // every published setpoint is the held seed. The uncommanded right arm
    // holds it bit-exact through the whole governed pipeline.
    let first_right = tokio::time::timeout(DEADLINE, right_wire.next())
        .await
        .expect("the right arm wire must start streaming once all followers report")?
        .expect("right arm subscription open");
    assert_eq!(first_right.positions.len(), 7);
    for (published, seeded) in first_right.positions.iter().zip(right_home().iter()) {
        assert!(
            (published - seeded).abs() < 1e-9,
            "an uncommanded arm must hold its seeded pose, got {:?}",
            first_right.positions
        );
    }

    // The leading node commands an elbow bend on the left arm only. The watch
    // keeps the latest command, so one delivery suffices; the republish on an
    // empty read window covers a best-effort drop.
    let mut target = left_home();
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

/// A leader that is streaming has the limb: a goal on it is refused with
/// `leader_streaming`, a plan check on it gets an error reply of that kind,
/// and the goal is admitted once the stream has stopped for the streaming
/// window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_goal_on_an_arm_a_leader_streams_waits_for_the_stream_to_stop() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::limb_motion::check_arm_move;

    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let mut followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    )
    .followed;
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    // The leader streams a bent elbow every state period, as a leader arm
    // does, until the test lets go.
    let mut target = left_home();
    target[3] = 0.4;
    let leader = mocks.pairings.leader_left_arm.joint_setpoints;
    let (let_go, mut released) = oneshot::channel::<()>();
    let streaming = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
        loop {
            tokio::select! {
                _ = &mut released => return,
                _ = ticker.tick() => {
                    if leader.publish(&leader_command(target)).await.is_err() {
                        return;
                    }
                }
            }
        }
    });
    // The stream has the arm once the governed wire moves toward it.
    tokio::time::timeout(
        DEADLINE,
        followed.wait_for(|q| q[3] - left_home()[3] >= 0.01),
    )
    .await
    .expect("the leader's command never reached the governed wire")
    .expect("the left arm follower runs");

    let refused = move_arm_joints::send_goal(
        &harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "left_arm".to_string(),
            joint_positions: left_home(),
            duration_s: 0.2,
            settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    let refusal = refused
        .refusal
        .expect("a streaming leader must keep the arm");
    assert_eq!(refusal.kind, move_arm_joints::FailureKind::LeaderStreaming);
    assert!(
        refusal
            .message
            .starts_with("a leader is streaming arm commands"),
        "the refusal must say a leader streams: {}",
        refusal.message
    );

    let check = check_arm_move::RequestData {
        arm_name: "left_arm".to_string(),
        position: [0.3, 0.2, 0.3],
        orientation: Some([0.0, 0.0, 0.0, 1.0]),
        duration_s: 0.0,
    };
    let reply = error_reply(
        check_arm_move::poll(&harness, &check, DEADLINE).await,
        check_arm_move::Failure::of,
    );
    assert_eq!(
        (reply.kind, reply.message),
        (
            check_arm_move::FailureKind::LeaderStreaming,
            refusal.message
        )
    );

    // The leader lets go: past the 250 ms streaming window, the arm takes
    // the goal.
    let_go
        .send(())
        .expect("the leader streams until it lets go");
    streaming.await.expect("the leader stops streaming");
    tokio::time::sleep(Duration::from_millis(400)).await;
    complete_left_arm_joint_move(&harness, left_home(), 0.2).await?;
    harness.shutdown().await
}

/// Admission end-to-end: a goal on a limb whose follower has not reported is
/// refused at the door with `resource_unavailable`, before the coordinator
/// has seeded, in the words the so101 backbone gives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_goal_on_a_limb_with_no_fresh_follower_state_is_refused_at_admission()
-> peppygen::Result<()> {
    let (harness, mocks, _fault) = helpers::start(Config {
        parameters: Some(params()),
        collision_ctrl_vacant: false,
        perception_geometry_vacant: true,
        ..Default::default()
    })
    .await?;
    answer_robot_identity(&mocks, MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    // The right arm and both grippers report; the left arm never does.
    pump_right_arm_and_grippers!(mocks);

    let goal = move_arm_joints::send_goal(
        &harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "left_arm".to_string(),
            joint_positions: left_home(),
            duration_s: 1.0,
            settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert_eq!(
        goal.refusal
            .expect("a goal on an unreported arm was accepted"),
        move_arm_joints::Failure::new(
            move_arm_joints::FailureKind::ResourceUnavailable,
            "no fresh follower state to anchor the move from"
        )
    );
    harness.shutdown().await
}

/// Exposed action end-to-end: `move_arm_joints` through the real action
/// engine, with the left arm mock playing a perfect follower (it adopts every
/// governed setpoint as its measured state). The goal must be admitted, the
/// trajectory streamed down the pairing wire, and the goal must complete with
/// the measured (echoed) pose on the commanded target.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_arm_joints_streams_a_trajectory_the_arm_follows_to_the_target() -> peppygen::Result<()>
{
    // collision_ctrl bound to its (silent) mock this time: the listener runs
    // against a live producer that never publishes, and the launch band stands.
    let (harness, mocks, _fault) = helpers::start(Config {
        parameters: Some(params()),
        collision_ctrl_vacant: false,
        perception_geometry_vacant: true,
        ..Default::default()
    })
    .await?;
    answer_robot_identity(&mocks, MODEL).await?;
    assert!(mocks.deps.collision_ctrl.is_some());

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    )
    .followed;
    pump_right_arm_and_grippers!(mocks);

    // Gate the goal on streaming having begun: a goal sent before the
    // coordinator's first tick is refused at admission ("no fresh follower
    // state to anchor the move from"), so wait for the first governed
    // setpoint on the (uncommanded) right wire first.
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

/// A joint move that the collision governor holds short fails with
/// `timeout` when the settle timeout runs out after the planned end, naming
/// the held joint, and gives no result. Here the band is wide, so the rest
/// pose sits under d_stop. A left wrist sent toward the centerline never
/// gets there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_move_the_governor_holds_fails_its_settle_with_timeout() -> peppygen::Result<()> {
    let mut wide = params();
    wide.d_stop_m = 0.8;
    wide.d_safe_m = 1.0;
    let (harness, mocks) = start_ready_vacant(wide, MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    )
    .followed;
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    let mut inward = left_home();
    inward[2] = 1.5;
    inward[3] = 0.4;
    let failure = match left_arm_joint_move(&harness, inward, 1.0).await? {
        move_arm_joints::ResultOutcome::Failed(failure) => failure,
        other => panic!("a held arm cannot have settled on its target: {other:?}"),
    };
    assert_eq!(failure.kind, move_arm_joints::FailureKind::Timeout);
    assert!(
        failure.message.starts_with("still ")
            && failure
                .message
                .contains(" rad from the target at joint 3, "),
        "the failure names the held joint: {}",
        failure.message
    );
    assert!(
        seconds_after_planned_end(&failure.message) >= SETTLE_TIMEOUT_S,
        "the failure came no sooner than the settle timeout: {}",
        failure.message
    );
    let observed = *followed.borrow();
    assert!(
        (observed[2] - inward[2]).abs() > 0.5,
        "the governor let the wrist reach its goal: {observed:?}"
    );
    harness.shutdown().await
}

/// A move_arm goal ends on the grasp point of the joints that the arm
/// measures. Here the left arm stands still at Ready:
/// - a plan check of a pose 5 cm up answers that a plan reaches it;
/// - a move there runs to its planned end and fails with `timeout`, naming
///   the 50 mm the grasp point still stands from the target, and gives no
///   result;
/// - the same move, cancelled by its caller, ends cancelled for that reason
///   with where the arm stands, as limb_state gives it, not the goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pose_move_that_cannot_settle_fails_and_a_cancelled_one_reports_the_measured_grasp_point()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::limb_motion::move_arm;
    use peppygen::fixtures::exposed_services::limb_motion::check_arm_move;

    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let ready = launched_posture("left_arm", states::READY);
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
            orientation: Some(orientation),
            duration_s: 1.0,
        },
        DEADLINE,
    )
    .await?;
    assert!(check.reachable, "{}", check.reason);
    assert!(check.reason.is_empty(), "{}", check.reason);

    let raise = |duration_s: f64| move_arm::GoalRequestData {
        arm_name: "left_arm".to_string(),
        position: raised,
        orientation: Some(orientation),
        duration_s,
        settle_position_tolerance_m: SETTLE_POSITION_TOLERANCE_M,
        settle_orientation_tolerance_rad: SETTLE_ORIENTATION_TOLERANCE_RAD,
        settle_timeout_s: SETTLE_TIMEOUT_S,
    };
    let goal = move_arm::send_goal(
        &harness,
        &raise(1.0),
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    let failure = match goal.get_result(DEADLINE).await?.outcome {
        move_arm::ResultOutcome::Failed(failure) => failure,
        other => panic!("an arm that never moved cannot have settled: {other:?}"),
    };
    assert_eq!(failure.kind, move_arm::FailureKind::Timeout);
    assert!(
        failure
            .message
            .starts_with("still 50.0 mm / 0.00 rad from the target, "),
        "the failure names the remaining error: {}",
        failure.message
    );
    assert!(
        seconds_after_planned_end(&failure.message) >= SETTLE_TIMEOUT_S,
        "the failure came no sooner than the settle timeout: {}",
        failure.message
    );

    let goal = move_arm::send_goal(
        &harness,
        &raise(30.0),
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    goal.cancel_goal(DEADLINE).await?;
    let (reason, data) = match goal.get_result(DEADLINE).await?.outcome {
        move_arm::ResultOutcome::Cancelled { reason, data } => (reason, data),
        other => panic!("the cancelled move did not end cancelled: {other:?}"),
    };
    assert_eq!(reason, "cancelled by its caller");
    assert_all_close("final_position", &data.final_position, &start, 1e-9);
    assert_all_close(
        "final_orientation",
        &data.final_orientation,
        &orientation,
        1e-9,
    );
    harness.shutdown().await
}

/// Exposed gripper action end-to-end: `move_gripper` with the left gripper
/// played by a follower whose jaws lag their commands. The commanded ramp
/// lands on the target while the jaws are still most of the way out; the
/// result comes once they have stood still on it, says they settled, and
/// reports the opening measured then.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_gripper_answers_with_the_opening_the_lagging_jaws_stopped_at() -> peppygen::Result<()>
{
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_arm_at_home!(
        mocks.pairings.right_arm.joint_states,
        peppygen::paired_topics::right_arm::joint_states,
        right_home(),
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
    // 70 ms, the lagging jaws need 40 state periods and then stand still.
    let target = 0.1;
    let data = complete_left_gripper_move(&harness, target, 0.0).await?;
    assert!(data.settled, "the jaws stood still on the target: {data:?}");
    assert_eq!(
        data.final_opening, target,
        "final_opening must be the opening the jaws arrived at"
    );
    assert_eq!(*followed.borrow(), target, "the jaws stand on the target");

    harness.shutdown().await
}

/// A gripper move whose jaws do not stand still by its settle timeout
/// completes where they stand, and says they did not settle. Here the
/// timeout is 0 s: the commanded ramp lands on the target long before the
/// lagging jaws, which cannot have stood still for the 0.25 s window on the
/// tick it lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gripper_move_whose_jaws_are_not_still_by_its_timeout_completes_unsettled()
-> peppygen::Result<()> {
    let (harness, _) = start_with_left_followers(0.0).await?;

    let target = 0.1;
    let data = complete_left_gripper_move_within(&harness, target, 0.0, 0.0).await?;
    assert!(!data.settled, "the jaws cannot have stood still: {data:?}");
    assert!(
        (target..=0.5).contains(&data.final_opening),
        "the jaws stand between where they were and the target: {data:?}"
    );
    harness.shutdown().await
}

/// After a move_gripper ends, the node sends that gripper nothing new. Thus
/// the gripper keeps the move's opening and effort cap, and keeps squeezing
/// what it holds. Here:
/// - an object stops the jaws at 0.3 on their way to 0.0;
/// - the move completes once the jaws stand still there and reports where
///   they stopped;
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
    assert!(data.settled, "the held jaws stood still: {data:?}");
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
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;

    let ready = Arc::new(AtomicBool::new(false));
    pump_is_ready(mocks.deps.robot_init.is_ready, ready.clone());
    // The follower pumps park on wait_for_subscriber while the gate holds.
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
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
    let (mut harness, mocks) = start_ready_vacant(wide, MODEL).await?;

    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);

    // Both wrists commanded toward the centerline (the governor unit tests'
    // own closing geometry: left j3 positive, right j3 negative), so the
    // candidate motion unambiguously closes the gap.
    let left_leader = mocks.pairings.leader_left_arm.joint_setpoints;
    let right_leader = mocks.pairings.leader_right_arm.joint_setpoints;
    let mut inward_left = left_home();
    inward_left[2] = 1.5;
    inward_left[3] = 0.4;
    let mut inward_right = right_home();
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
/// [`home`], both grippers half open), the emitted snapshot names both limbs,
/// slices per the contract's rule, and carries the measured values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limb_states_snapshots_carry_names_counts_and_measured_state() -> peppygen::Result<()> {
    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);

    let snapshot = next_snapshot(&mut harness).await?;

    assert_eq!(snapshot.arm_names, ["left_arm", "right_arm"]);
    assert_eq!(snapshot.joints_per_arm, [7, 7]);
    assert_eq!(snapshot.joint_positions.len(), 14);
    // Both arms park at HOME, so each 7-entry slice echoes the pumps.
    for (i, v) in snapshot.joint_positions.iter().enumerate() {
        assert!(
            (v - home_of(ARM_NAMES[i / 7])[i % 7]).abs() < 1e-9,
            "joint_positions[{i}] = {v}, pumped {}",
            home_of(ARM_NAMES[i / 7])[i % 7]
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

    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
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

/// A goal naming no limb of this robot is refused at admission with
/// `rejected` and the name table quoted, for the arm and gripper moves
/// alike.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn goals_for_unknown_limb_names_are_refused() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);

    // Wait for streaming so the refusal below is the name check, not the
    // seed gate's blanket refusal.
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    let goal = move_arm_joints::send_goal(
        &harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "torso".to_string(),
            joint_positions: left_home(),
            duration_s: 1.0,
            settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert_eq!(
        goal.refusal.expect("a goal for \"torso\" must be refused"),
        move_arm_joints::Failure::new(
            move_arm_joints::FailureKind::Rejected,
            r#"arm_name names "torso", no arm of this robot: its arms are "left_arm" and "right_arm""#
        )
    );

    // A gripper goal addressed with an ARM name is a mixup, not a gripper.
    let goal = move_gripper::send_goal(
        &harness,
        &move_gripper::GoalRequestData {
            gripper_name: "left_arm".to_string(),
            opening: 0.5,
            max_effort: 0.0,
            settle_timeout_s: GRIPPER_SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert_eq!(
        goal.refusal
            .expect("a gripper goal for \"left_arm\" must be refused"),
        move_gripper::Failure::new(
            move_gripper::FailureKind::Rejected,
            r#"gripper_name names "left_arm", no gripper of this robot: its grippers are "left_gripper" and "right_gripper""#
        )
    );
    Ok(())
}

/// The stop service ends the moves in flight, whoever started them: an arm
/// move and a gripper move both end cancelled, for a reason that names the
/// stop and the reason of its request, the answer names both limbs, a stop
/// with nothing moving names none, and a goal that arrives after the stop
/// runs to its end.
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
    assert!(idle.stopped.is_empty(), "{:?}", idle.stopped);

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
            settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(arm_goal.accepted, "goal refused: {:?}", arm_goal.refusal);
    let gripper_goal = move_gripper::send_goal(
        &harness,
        &move_gripper::GoalRequestData {
            gripper_name: "left_gripper".to_string(),
            opening: 0.1,
            max_effort: 0.0,
            settle_timeout_s: GRIPPER_SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(
        gripper_goal.accepted,
        "goal refused: {:?}",
        gripper_goal.refusal
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
    assert_eq!(answer.stopped, ["left_arm", "left_gripper"]);

    match arm_goal.get_result(DEADLINE).await?.outcome {
        move_arm_joints::ResultOutcome::Cancelled { reason, .. } => {
            assert_eq!(reason, "stopped: the operator asked");
        }
        other => panic!("the stopped arm move did not end cancelled: {other:?}"),
    }
    match gripper_goal.get_result(DEADLINE).await?.outcome {
        move_gripper::ResultOutcome::Cancelled { reason, data } => {
            assert_eq!(reason, "stopped: the operator asked");
            assert!(!data.settled, "a stopped gripper move did not settle");
        }
        other => panic!("the stopped gripper move did not end cancelled: {other:?}"),
    }
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
    assert!(data.settled, "{data:?}");
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
/// pose for the same joints. Here both arms follow their setpoints, so each
/// posture completes once both arms stand at it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_result_reports_each_arms_measured_grasp_point() -> peppygen::Result<()> {
    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let _left = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    let _right = spawn_right_arm_follower(
        mocks.pairings.right_arm.joint_states,
        mocks.pairings.right_arm.joint_setpoints,
    );
    pump_gripper_half_open!(
        mocks.pairings.left_gripper.gripper_states,
        peppygen::paired_topics::left_gripper::gripper_states
    );
    pump_gripper_half_open!(
        mocks.pairings.right_gripper.gripper_states,
        peppygen::paired_topics::right_gripper::gripper_states
    );
    next_snapshot(&mut harness).await?;

    let ready = complete_posture!(harness, move_to_ready, 0.0);
    let at_ready = snapshot_with_both_arms_at(&mut harness, states::READY).await?;
    let home = complete_posture!(harness, move_to_home, 0.0);
    let at_home = snapshot_with_both_arms_at(&mut harness, states::HOME).await?;
    for (result, snapshot) in [(ready, at_ready), (home, at_home)] {
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

/// A posture that an arm does not reach fails with `timeout`, its message
/// led by the name of that arm and naming the joint and how far it stands,
/// no sooner than 2 s after the planned end. Here the right arm stands still
/// at HOME, as an arm that a bench in front of the robot stops does: the
/// robot does not see the bench. move_to_home, where both arms already
/// stand, completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_an_arm_stops_short_of_fails_with_timeout() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::postures::move_to_ready;

    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let _left = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);
    next_snapshot(&mut harness).await?;

    let goal = send_posture_goal!(harness, move_to_ready, 0.0);
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    let failure = match goal.get_result(DEADLINE).await?.outcome {
        move_to_ready::ResultOutcome::Failed(failure) => failure,
        other => panic!("a right arm held at HOME cannot reach Ready: {other:?}"),
    };
    assert_eq!(failure.kind, move_to_ready::FailureKind::Timeout);
    assert!(
        failure.message.starts_with("right_arm: still ")
            && failure.message.contains(" rad from the target at joint "),
        "the failure names the arm and its joint: {}",
        failure.message
    );
    assert!(
        seconds_after_planned_end(&failure.message) >= 2.0,
        "the failure came no sooner than the posture's settle timeout: {}",
        failure.message
    );
    complete_posture!(harness, move_to_home, 0.0);
    harness.shutdown().await
}

/// Before the arms report their first state, a posture goal is refused at
/// admission with `resource_unavailable`, in the words every arm goal gets
/// then.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_goal_before_the_first_measurement_is_refused() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    // No follower reports, so the coordinator waits for the first states.
    let refusal = "no fresh follower state to anchor the move from";
    assert_posture_refused!(harness, move_to_ready, 1.0, ResourceUnavailable, refusal);
    assert_posture_refused!(harness, move_to_home, 1.0, ResourceUnavailable, refusal);
    harness.shutdown().await
}

/// A posture goal whose duration is above the robot's limit is refused at
/// admission with `rejected`, for both postures.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_posture_goal_above_the_duration_ceiling_is_refused() -> peppygen::Result<()> {
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;

    let ceiling = "duration_s must be a number of seconds from 0 to 60";
    assert_posture_refused!(harness, move_to_ready, 61.0, Rejected, ceiling);
    assert_posture_refused!(harness, move_to_home, 61.0, Rejected, ceiling);
    harness.shutdown().await
}

/// After robot.stop, a posture result gives where each arm stopped:
/// - the left arm, played by a follower that the test brakes on its way to
///   Ready;
/// - the right arm, which stands at HOME.
///
/// The test waits until the robot measures the braked left arm before the
/// stop. Thus each arm rests where the robot measured it. The goal ends
/// cancelled for the stop's reason, with the poses that limb_state gives
/// for those joints.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopped_posture_reports_where_the_arms_stopped() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::postures::move_to_ready;
    use peppygen::fixtures::exposed_services::limb_motion::stop;

    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let left = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);
    next_snapshot(&mut harness).await?;

    let goal = send_posture_goal!(harness, move_to_ready, 30.0);
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    let mut moving = left.followed.clone();
    tokio::time::timeout(
        DEADLINE,
        moving.wait_for(|q| {
            q.iter()
                .zip(left_home())
                .any(|(q, home)| (q - home).abs() > 0.01)
        }),
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
    assert_eq!(answer.stopped, ["left_arm", "right_arm"]);

    let data = match goal.get_result(DEADLINE).await?.outcome {
        move_to_ready::ResultOutcome::Cancelled { reason, data } => {
            assert_eq!(reason, "stopped: the operator asked");
            data
        }
        other => panic!("the stopped posture did not end cancelled: {other:?}"),
    };
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

/// Every limb of a node that [`start_with_every_follower`] started: both
/// arms follow their setpoints, and the jaws of both grippers lag theirs.
struct EveryFollower {
    left_arm: ArmFollower,
    right_arm: ArmFollower,
    left_gripper: GripperFollower,
    right_gripper: GripperFollower,
}

impl EveryFollower {
    fn grippers(&self) -> [(&'static str, &GripperFollower); 2] {
        [
            ("left_gripper", &self.left_gripper),
            ("right_gripper", &self.right_gripper),
        ]
    }

    fn arms(&self) -> [&ArmFollower; 2] {
        [&self.left_arm, &self.right_arm]
    }
}

/// Start the node with `parameters` on a robot of `model` that is ready, and
/// wait until it has seeded every limb. Both arms are perfect followers. The jaws of each
/// gripper cover `gripper_step` per [`STATE_PUMP_PERIOD`], and an object
/// stops the left jaws at `left_object_at`.
async fn start_with_every_follower(
    parameters: peppygen::Parameters,
    model: &str,
    gripper_step: f64,
    left_object_at: f64,
) -> peppygen::Result<(Harness, EveryFollower)> {
    let (mut harness, mocks) = start_ready_vacant(parameters, model).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followers = EveryFollower {
        left_arm: spawn_left_arm_follower(
            mocks.pairings.left_arm.joint_states,
            mocks.pairings.left_arm.joint_setpoints,
        ),
        right_arm: spawn_right_arm_follower(
            mocks.pairings.right_arm.joint_states,
            mocks.pairings.right_arm.joint_setpoints,
        ),
        left_gripper: spawn_left_gripper_follower_at(
            mocks.pairings.left_gripper.gripper_states,
            mocks.pairings.left_gripper.gripper_setpoints,
            gripper_step,
            left_object_at,
        ),
        right_gripper: spawn_right_gripper_follower_at(
            mocks.pairings.right_gripper.gripper_states,
            mocks.pairings.right_gripper.gripper_setpoints,
            gripper_step,
        ),
    };
    next_snapshot(&mut harness).await?;
    Ok((harness, followers))
}

/// move_to_ready sends no command to the grippers. A gripper that a
/// move_gripper left at 0.2 stays there through move_to_ready. It gets no
/// opening other than that one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_to_ready_sends_the_grippers_no_command() -> peppygen::Result<()> {
    let (harness, followers) =
        start_with_every_follower(params(), MODEL, LAGGING_GRIPPER_STEP, 0.0).await?;
    let gripper = &followers.left_gripper;

    let data = complete_left_gripper_move(&harness, 0.2, 0.0).await?;
    assert!(data.settled, "{data:?}");
    assert_eq!(*gripper.jaws.borrow(), 0.2);
    let after_move = gripper.received_count();
    let right_before = followers.right_gripper.received_count();

    complete_posture!(harness, move_to_ready, 1.0);
    assert_eq!(*gripper.jaws.borrow(), 0.2, "move_to_ready moved the jaws");
    for setpoint in gripper.received_since(after_move) {
        assert_eq!(setpoint.opening, 0.2, "move_to_ready commanded the gripper");
    }
    assert_eq!(
        followers.right_gripper.received_since(right_before),
        [],
        "move_to_ready commanded a gripper that no move commanded"
    );
    harness.shutdown().await
}

/// From Ready with both grippers fully open, move_to_home closes both
/// grippers, then brings both arms to Home, on a robot of `model`: its result
/// gives both arms where they stand at Home, and the jaws of both grippers are
/// closed.
async fn home_closes_fully_open_grippers(model: &str) -> peppygen::Result<()> {
    let (mut harness, followers) =
        start_with_every_follower(params(), model, PROMPT_GRIPPER_STEP, 0.0).await?;

    complete_posture!(harness, move_to_ready, 0.0);
    for (gripper_name, gripper) in followers.grippers() {
        let opened = complete_gripper_move_within(
            &harness,
            gripper_name,
            1.0,
            0.0,
            GRIPPER_SETTLE_TIMEOUT_S,
        )
        .await?;
        assert!(opened.settled, "{gripper_name}: {opened:?}");
        assert_eq!(*gripper.jaws.borrow(), 1.0, "{gripper_name} is not open");
    }

    let home = complete_posture!(harness, move_to_home, 0.0);
    let at_home = snapshot_with_both_arms_at(&mut harness, states::HOME).await?;
    assert_eq!(home.arm_names, ["left_arm", "right_arm"]);
    assert_all_close("positions", &home.positions, &at_home.positions, 1e-9);
    for (_, gripper) in followers.grippers() {
        gripper.jaws_at(0.0).await;
    }
    harness.shutdown().await
}

/// See [`home_closes_fully_open_grippers`], on an openarm_v1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_to_home_closes_fully_open_grippers_and_the_arms_stand_at_left_home()
-> peppygen::Result<()> {
    home_closes_fully_open_grippers(MODEL).await
}

/// See [`home_closes_fully_open_grippers`], on an openarm_v2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_to_home_closes_fully_open_grippers_on_an_openarm_v2() -> peppygen::Result<()> {
    home_closes_fully_open_grippers(V2).await
}

/// A gripper that holds an item closes on it, and the arms go home with it,
/// on a robot of `model`. Here a grasp under an effort cap of
/// 0.4 holds an object that stops the left jaws at 0.3:
/// - move_to_home completes with both arms at Home;
/// - the left jaws stand on the object, and the last setpoint that the
///   gripper got is closed under the cap of the grasp, so it keeps squeezing
///   the object as the grasp did;
/// - every setpoint of the close of the left gripper carries that cap.
async fn home_closes_on_an_item(model: &str) -> peppygen::Result<()> {
    const GRASP_EFFORT: f64 = 0.4;
    const OBJECT_AT: f64 = 0.3;
    let (mut harness, followers) =
        start_with_every_follower(params(), model, PROMPT_GRIPPER_STEP, OBJECT_AT).await?;
    let held = &followers.left_gripper;

    complete_posture!(harness, move_to_ready, 0.0);
    let grasp = complete_gripper_move_within(
        &harness,
        "left_gripper",
        0.0,
        GRASP_EFFORT,
        GRIPPER_SETTLE_TIMEOUT_S,
    )
    .await?;
    assert!(grasp.settled, "{grasp:?}");
    assert_eq!(grasp.final_opening, OBJECT_AT);
    let after_grasp = held.received_count();

    complete_posture!(harness, move_to_home, 0.0);
    snapshot_with_both_arms_at(&mut harness, states::HOME).await?;
    followers.right_gripper.jaws_at(0.0).await;
    assert_eq!(*held.jaws.borrow(), OBJECT_AT);
    let close = held.received_since(after_grasp);
    assert!(!close.is_empty(), "the close sent the gripper no setpoint");
    for setpoint in &close {
        assert_eq!(setpoint.max_effort, GRASP_EFFORT, "{setpoint:?}");
    }
    assert_eq!(
        held.last_received(),
        Some(GripperSetpoint {
            opening: 0.0,
            max_effort: GRASP_EFFORT,
        })
    );
    harness.shutdown().await
}

/// See [`home_closes_on_an_item`], on an openarm_v1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_to_home_closes_on_an_item_and_the_arms_go_home_with_it() -> peppygen::Result<()> {
    home_closes_on_an_item(MODEL).await
}

/// See [`home_closes_on_an_item`], on an openarm_v2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_to_home_closes_on_an_item_on_an_openarm_v2() -> peppygen::Result<()> {
    home_closes_on_an_item(V2).await
}

/// move_to_home parks the arms of a robot whose gripper stopped reporting.
/// Here the right gripper goes silent, half open, with both arms at Ready:
/// - once the backbone refuses a move of that gripper for its missing
///   state, move_to_home completes with both arms at Home;
/// - the silent gripper gets no setpoint, and the left gripper is closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn move_to_home_parks_the_arms_of_a_robot_whose_gripper_stopped_reporting()
-> peppygen::Result<()> {
    let (harness, followers) =
        start_with_every_follower(params(), MODEL, PROMPT_GRIPPER_STEP, 0.0).await?;
    complete_posture!(harness, move_to_ready, 0.0);

    let silent = &followers.right_gripper;
    silent.stop_reporting();
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let goal = move_gripper::send_goal(
            &harness,
            &move_gripper::GoalRequestData {
                gripper_name: "right_gripper".to_string(),
                opening: 0.5,
                max_effort: 0.0,
                settle_timeout_s: GRIPPER_SETTLE_TIMEOUT_S,
            },
            peppygen::DEFAULT_FEEDBACK_BOUND,
            DEADLINE,
        )
        .await?;
        if let Some(refusal) = &goal.refusal {
            assert_eq!(
                refusal.message, "no fresh gripper state to anchor the move from",
                "{refusal:?}"
            );
            break;
        }
        goal.get_result(DEADLINE).await?;
        assert!(
            tokio::time::Instant::now() < deadline,
            "the backbone never missed the state of the silent gripper"
        );
    }
    let before_home = silent.received_count();

    let home = complete_posture!(harness, move_to_home, 0.0);
    assert_eq!(home.arm_names, ["left_arm", "right_arm"]);
    for (follower, arm_name) in followers.arms().into_iter().zip(ARM_NAMES) {
        let last = follower.received().pop().expect("the arm got a setpoint");
        assert_all_close(
            "the last arm setpoint",
            &last,
            &launched_posture(arm_name, states::HOME),
            1e-9,
        );
    }
    assert_eq!(
        silent.received_since(before_home),
        [],
        "the silent gripper got a setpoint"
    );
    followers.left_gripper.jaws_at(0.0).await;
    harness.shutdown().await
}

/// A stop during the close of a move_to_home ends the goal cancelled before
/// the arms move. Here both arms stand at Ready, and the backbone ramps a
/// gripper opening at 0.01 of the jaw travel per second, so each close
/// travels for 50 s from the half open jaws:
/// - the stop comes once both grippers got a setpoint that closes them, and
///   names both grippers and no arm;
/// - the goal ends cancelled for the stop's reason;
/// - neither arm got a setpoint other than Ready.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_during_the_close_of_move_to_home_ends_it_before_the_arms_move()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::postures::move_to_home;
    use peppygen::fixtures::exposed_services::limb_motion::stop;

    let slow_grippers = peppygen::Parameters {
        max_gripper_rate_frac_s: 0.01,
        ..params()
    };
    let (harness, followers) =
        start_with_every_follower(slow_grippers, MODEL, LAGGING_GRIPPER_STEP, 0.0).await?;
    complete_posture!(harness, move_to_ready, 0.0);
    let at_ready = followers.arms().map(|arm| arm.received().len());

    let goal = send_posture_goal!(harness, move_to_home, 0.0);
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    for (_, gripper) in followers.grippers() {
        gripper.commanded_under(0.5).await;
    }
    let answer = stop::poll(
        &harness,
        &stop::RequestData {
            reason: "the operator asked".to_string(),
        },
        DEADLINE,
    )
    .await?;
    assert_eq!(answer.stopped, ["left_gripper", "right_gripper"]);

    match goal.get_result(DEADLINE).await?.outcome {
        move_to_home::ResultOutcome::Cancelled { reason, data } => {
            assert_eq!(reason, "stopped: the operator asked");
            assert_eq!(data.arm_names, ["left_arm", "right_arm"]);
        }
        other => panic!("the stopped move_to_home did not end cancelled: {other:?}"),
    }
    for ((follower, arm_name), ready_count) in
        followers.arms().into_iter().zip(ARM_NAMES).zip(at_ready)
    {
        let ready = launched_posture(arm_name, states::READY);
        for setpoint in &follower.received()[ready_count..] {
            assert_all_close("an arm setpoint after the stop", setpoint, &ready, 1e-9);
        }
    }
    harness.shutdown().await
}

/// The plan check answers whether a Cartesian goal has a plan and moves
/// nothing: a pose a little above the grasp point has one, with the time the
/// move would take; a pose two meters away has none, and a move_arm goal to
/// it fails with `unreachable` in the same words; an unknown arm gets an
/// error reply with `rejected`; and the arm mock observes no motion
/// throughout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plan_check_answers_without_moving_the_arm() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::limb_motion::move_arm;
    use peppygen::fixtures::exposed_services::limb_motion::check_arm_move;

    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    )
    .followed;
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
        orientation: Some(orientation),
        duration_s: 0.0,
    };

    let near = check("left_arm", [position[0], position[1], position[2] + 0.02]);
    let answer = check_arm_move::poll(&harness, &near, DEADLINE).await?;
    assert!(answer.reachable, "{}", answer.reason);
    assert!(answer.reason.is_empty(), "{}", answer.reason);
    assert!(answer.duration_s > 0.0, "{}", answer.duration_s);

    let far_position = [position[0] + 2.0, position[1], position[2]];
    let far = check("left_arm", far_position);
    let answer = check_arm_move::poll(&harness, &far, DEADLINE).await?;
    assert!(!answer.reachable);
    assert!(
        answer.reason.starts_with("goal pose not planned"),
        "{}",
        answer.reason
    );
    assert_eq!(answer.duration_s, 0.0);

    let goal = move_arm::send_goal(
        &harness,
        &move_arm::GoalRequestData {
            arm_name: "left_arm".to_string(),
            position: far_position,
            orientation: Some(orientation),
            duration_s: 0.0,
            settle_position_tolerance_m: SETTLE_POSITION_TOLERANCE_M,
            settle_orientation_tolerance_rad: SETTLE_ORIENTATION_TOLERANCE_RAD,
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    match goal.get_result(DEADLINE).await?.outcome {
        move_arm::ResultOutcome::Failed(failure) => assert_eq!(
            failure,
            move_arm::Failure::new(move_arm::FailureKind::Unreachable, answer.reason)
        ),
        other => panic!("a move no plan reaches did not fail: {other:?}"),
    }

    let unknown = check("middle_arm", position);
    let reply = error_reply(
        check_arm_move::poll(&harness, &unknown, DEADLINE).await,
        check_arm_move::Failure::of,
    );
    assert_eq!(reply.kind, check_arm_move::FailureKind::Rejected);
    assert!(
        reply
            .message
            .starts_with("arm_name names \"middle_arm\", no arm of this robot"),
        "{}",
        reply.message
    );

    // Nothing moved: the follower still adopts HOME from every setpoint.
    let observed = *followed.borrow();
    for (joint, (observed, home)) in observed.iter().zip(left_home().iter()).enumerate() {
        assert!(
            (observed - home).abs() < 1e-9,
            "joint {joint} moved to {observed} during the checks"
        );
    }
    harness.shutdown().await
}

/// A goal with no orientation is a position alone: it completes there with the hand
/// at the turn it starts with.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_position_goal_completes_at_the_turn_the_hand_starts_with() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::limb_motion::move_arm;

    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let _followed = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    )
    .followed;
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
    let lifted = [start[0], start[1], start[2] + 0.02];
    let goal = move_arm::send_goal(
        &harness,
        &move_arm::GoalRequestData {
            arm_name: "left_arm".to_string(),
            position: lifted,
            orientation: None,
            duration_s: 0.0,
            settle_position_tolerance_m: SETTLE_POSITION_TOLERANCE_M,
            settle_orientation_tolerance_rad: SETTLE_ORIENTATION_TOLERANCE_RAD,
            settle_timeout_s: SETTLE_TIMEOUT_S,
        },
        peppygen::DEFAULT_FEEDBACK_BOUND,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal refused: {:?}", goal.refusal);
    let data = match goal.get_result(DEADLINE).await?.outcome {
        move_arm::ResultOutcome::Completed(data) => data,
        other => panic!("a position goal did not complete: {other:?}"),
    };
    assert_all_close(
        "final_position",
        &data.final_position,
        &lifted,
        SETTLE_POSITION_TOLERANCE_M,
    );
    assert_all_close(
        "final_orientation",
        &data.final_orientation,
        &orientation,
        SETTLE_ORIENTATION_TOLERANCE_RAD,
    );
    harness.shutdown().await
}

/// The camera mounts service answers from bringup: an error reply with
/// `precondition_failed` until the coordinator has measured the arms, then
/// every camera of the generation under the stamp of the snapshot the poses
/// come from, a v1 robot with no camera at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn camera_poses_are_answered_once_the_arms_are_measured() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::camera_mounts::get_camera_poses;

    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    let ready = Arc::new(AtomicBool::new(false));
    pump_is_ready(mocks.deps.robot_init.is_ready, ready.clone());

    let refused = error_reply(
        get_camera_poses::poll(&harness, DEADLINE).await,
        get_camera_poses::Failure::of,
    );
    assert_eq!(
        refused,
        get_camera_poses::Failure::new(
            get_camera_poses::FailureKind::PreconditionFailed,
            "the robot has not measured its joints yet"
        )
    );

    ready.store(true, Ordering::SeqCst);
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    let answer = loop {
        let refused = match get_camera_poses::poll(&harness, DEADLINE).await {
            Ok(answer) => break answer,
            Err(e) => get_camera_poses::Failure::of(&e).ok_or(e)?,
        };
        assert!(
            tokio::time::Instant::now() < deadline,
            "no measurement reached the service: {refused}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    // v1 carries no camera: an answer with nothing listed.
    assert!(answer.camera_names.is_empty() && answer.carried_by.is_empty());
    assert!(answer.positions.is_empty() && answer.orientations.is_empty());
    harness.shutdown().await
}

/// A v2 robot lists the cameras of its description, in its order, one fixed
/// to the base and looking forward and down, and one carried by each arm,
/// under the stamp of a limb_states snapshot. Where each camera stands is the
/// unit tests' (`camera_mounts`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_v2_robot_lists_its_three_cameras_in_the_robot_frame() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::camera_mounts::get_camera_poses;

    let (mut harness, mocks) = start_ready_vacant(params(), V2).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm.joint_states,
        peppygen::paired_topics::left_arm::joint_states,
        left_home(),
    );
    pump_right_arm_and_grippers!(mocks);
    let snapshot = next_snapshot(&mut harness).await?;

    let answer = get_camera_poses::poll(&harness, DEADLINE).await?;
    assert!(answer.timestamp >= snapshot.timestamp);
    let described = v2_camera_names();
    assert_eq!(answer.camera_names, described);
    let cameras = described.len();
    for carrier in ["", "left_arm", "right_arm"] {
        assert_eq!(
            answer.carried_by.iter().filter(|c| *c == carrier).count(),
            1,
            "one camera carried by {carrier:?}"
        );
    }
    assert_eq!(answer.positions.len(), 3 * cameras);
    assert_eq!(answer.orientations.len(), 4 * cameras);
    // The base camera looks along its optical +Z, to the robot's front and
    // down: the image's +Y (down the picture) points forward and down.
    let base = answer
        .carried_by
        .iter()
        .position(String::is_empty)
        .expect("a camera fixed to the base");
    let [x, y, z, w] = [0, 1, 2, 3].map(|axis| answer.orientations[4 * base + axis]);
    // Rotate the unit +Z by the quaternion: v' = q v q*.
    let view_z = 1.0 - 2.0 * (x * x + y * y);
    let view_x = 2.0 * (x * z + w * y);
    assert!(
        view_x > 0.4 && view_z < -0.8,
        "the base camera looks forward and down: ({view_x}, {view_z})"
    );
    harness.shutdown().await
}

/// Answers every poll of a request-less mock service with `$answer`, until
/// the mock's session closes: `Ok` a response, `Err` an error reply with the
/// failure's kind and message.
macro_rules! pump_answers {
    ($service:expr, $answer:expr) => {{
        let service = $service;
        let answer = $answer;
        tokio::spawn(async move {
            while let Ok(responder) = service.next_request(PUMP_TIMEOUT).await {
                let answered = match answer.clone() {
                    Ok(response) => responder.respond(response).await,
                    Err(failure) => {
                        responder
                            .respond_failure(failure.kind, failure.message)
                            .await
                    }
                };
                if answered.is_err() {
                    break;
                }
            }
        });
    }};
}

/// Plays the mock bound to `perception_geometry` as the chest camera's
/// camera_geometry, answering `$colour` and `$depth` (each a response or an
/// error reply), and holds the robot not ready. The other mocks stay with
/// the test.
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
    let (harness, mocks, _fault) = helpers::start(Config {
        parameters: Some(params()),
        collision_ctrl_vacant: true,
        perception_geometry_vacant: false,
        ..Default::default()
    })
    .await?;
    answer_robot_identity(&mocks, V2).await?;
    Ok((harness, mocks))
}

/// The chest camera's colour intrinsics as the simulated camera gives them:
/// 1280 by 720 pixels over a 52 degree vertical field of view, the optical
/// axis through the middle of the image.
fn chest_colour_intrinsics() -> get_color_intrinsics::ResponseData {
    let focal = 360.0 / 26f64.to_radians().tan();
    get_color_intrinsics::ResponseData {
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

/// The error reply of a camera that gives colour alone, asked for its depth
/// intrinsics.
fn no_depth_intrinsics() -> get_depth_intrinsics::Failure {
    get_depth_intrinsics::Failure::new(
        get_depth_intrinsics::FailureKind::Rejected,
        "this camera has no depth stream",
    )
}

/// With no camera geometry linked, describe_workspace answers from the
/// design before the robot is ready: v2 still names one of its cameras as its
/// perception camera, reach alone decides, and no part is said to be seen.
/// A height that is not a number gets an error reply with `rejected`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_camera_geometry_the_workspace_is_described_without_a_view_check()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::describe_workspace;

    let (harness, mocks) = start_ready_vacant(params(), V2).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(false)),
    );

    let table = describe_workspace::RequestData {
        surface_height: 0.45,
    };
    let answer = describe_workspace::poll(&harness, &table, DEADLINE).await?;
    assert!(
        v2_camera_names().contains(&answer.perception_camera),
        "{answer:?}"
    );
    assert!(answer.workable && answer.area > 0.0, "{answer:?}");
    assert!(answer.rectangle.is_some() && answer.reach.is_some());
    assert_eq!(answer.view, None);

    let not_a_number = describe_workspace::RequestData {
        surface_height: f64::NAN,
    };
    let refused = error_reply(
        describe_workspace::poll(&harness, &not_a_number, DEADLINE).await,
        describe_workspace::Failure::of,
    );
    assert_eq!(
        refused,
        describe_workspace::Failure::new(
            describe_workspace::FailureKind::Rejected,
            "surface_height must be a finite number"
        )
    );
    harness.shutdown().await
}

/// With the chest camera's geometry linked, check_positions judges each
/// point by reach and by what the camera sees through its own intrinsics: a
/// point 0.30 m ahead at table height is workable, one a meter ahead is out
/// of reach by how far the closest arm stops. A list that does not split
/// into points gets an error reply with `rejected`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_camera_geometry_linked_check_positions_reports_what_the_chest_camera_sees()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(
        mocks,
        Ok::<_, get_color_intrinsics::Failure>(chest_colour_intrinsics()),
        Ok::<_, get_depth_intrinsics::Failure>(chest_depth_intrinsics())
    );

    let points = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45, 1.0, 0.0, 0.45],
    };
    let answer = check_positions::poll(&harness, &points, DEADLINE).await?;
    assert!(
        v2_camera_names().contains(&answer.perception_camera),
        "{answer:?}"
    );
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

    let ragged = check_positions::RequestData {
        positions: vec![0.30, 0.0],
    };
    let refused = error_reply(
        check_positions::poll(&harness, &ragged, DEADLINE).await,
        check_positions::Failure::of,
    );
    assert_eq!(
        refused,
        check_positions::Failure::new(
            check_positions::FailureKind::Rejected,
            "positions must hold 3 values (x, y, z) per point"
        )
    );
    harness.shutdown().await
}

/// A linked camera that gives no depth cannot be the chest camera, the
/// perception camera: the answer is an error reply with
/// `precondition_failed`, naming it and saying why.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_without_depth_is_refused_naming_the_chest_camera() -> peppygen::Result<()>
{
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(
        mocks,
        Ok::<_, get_color_intrinsics::Failure>(chest_colour_intrinsics()),
        Err::<get_depth_intrinsics::ResponseData, _>(no_depth_intrinsics())
    );

    let point = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45],
    };
    let refused = error_reply(
        check_positions::poll(&harness, &point, DEADLINE).await,
        check_positions::Failure::of,
    );
    assert_eq!(
        refused,
        check_positions::Failure::new(
            check_positions::FailureKind::PreconditionFailed,
            "the camera linked as the chest camera, the perception camera, gives no depth: \
             this camera has no depth stream"
        )
    );
    harness.shutdown().await
}

/// A linked camera whose depth stream gives a depth model camera_geometry:v1
/// does not name leaves the chest camera's depth range without a meaning:
/// the answer is an error reply with `precondition_failed`, naming the
/// model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_of_an_unknown_depth_model_is_refused_naming_the_model()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(
        mocks,
        Ok::<_, get_color_intrinsics::Failure>(chest_colour_intrinsics()),
        Ok::<_, get_depth_intrinsics::Failure>(get_depth_intrinsics::ResponseData {
            depth_model: "disparity".to_string(),
            ..chest_depth_intrinsics()
        })
    );

    let point = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45],
    };
    let refused = error_reply(
        check_positions::poll(&harness, &point, DEADLINE).await,
        check_positions::Failure::of,
    );
    assert_eq!(
        refused,
        check_positions::Failure::new(
            check_positions::FailureKind::PreconditionFailed,
            "cannot read the depth intrinsics of the chest camera, the perception camera: \
             unknown depth_model 'disparity' (expected 'z' or 'range')"
        )
    );
    harness.shutdown().await
}

/// A linked camera whose colour stream gives a lens camera_geometry:v1 does
/// not name leaves the chest camera's image without a model: the answer is
/// an error reply with `precondition_failed`, naming the lens.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_of_an_unknown_colour_lens_is_refused_naming_the_lens()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::check_positions;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(
        mocks,
        Ok::<_, get_color_intrinsics::Failure>(get_color_intrinsics::ResponseData {
            distortion_model: "kannala_brandt".to_string(),
            distortion: vec![0.1, 0.0, 0.0, 0.0],
            ..chest_colour_intrinsics()
        }),
        Ok::<_, get_depth_intrinsics::Failure>(chest_depth_intrinsics())
    );

    let point = check_positions::RequestData {
        positions: vec![0.30, 0.0, 0.45],
    };
    let refused = error_reply(
        check_positions::poll(&harness, &point, DEADLINE).await,
        check_positions::Failure::of,
    );
    assert_eq!(
        refused,
        check_positions::Failure::new(
            check_positions::FailureKind::PreconditionFailed,
            "cannot read the colour intrinsics of the chest camera, the perception camera: \
             unknown distortion 'kannala_brandt' with coefficients [0.1, 0.0, 0.0, 0.0] \
             (expected 'none' with none, or 'plumb_bob' or 'inverse_plumb_bob' with the five \
             numbers k1, k2, p1, p2, k3)"
        )
    );
    harness.shutdown().await
}

/// The error reply of a simulated camera asked for its colour intrinsics
/// before it hears from the simulation.
fn unready_colour_intrinsics() -> get_color_intrinsics::Failure {
    get_color_intrinsics::Failure::new(
        get_color_intrinsics::FailureKind::PreconditionFailed,
        "no camera geometry has been received from the simulation",
    )
}

/// A linked camera that cannot give its colour intrinsics now leaves the
/// chest camera's view unknown: the answer is an error reply with
/// `precondition_failed` and the camera's reason, and says nothing of what
/// the robot reaches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_linked_camera_without_its_colour_intrinsics_is_refused_with_its_reason()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::workspace::describe_workspace;

    let (harness, mocks) = start_v2_with_camera_geometry().await?;
    answer_as_the_chest_camera!(
        mocks,
        Err::<get_color_intrinsics::ResponseData, _>(unready_colour_intrinsics()),
        Err::<get_depth_intrinsics::ResponseData, _>(no_depth_intrinsics())
    );

    let table = describe_workspace::RequestData {
        surface_height: 0.45,
    };
    let refused = error_reply(
        describe_workspace::poll(&harness, &table, DEADLINE).await,
        describe_workspace::Failure::of,
    );
    assert_eq!(
        refused,
        describe_workspace::Failure::new(
            describe_workspace::FailureKind::PreconditionFailed,
            "cannot read the colour intrinsics of the chest camera, the perception camera: \
             no camera geometry has been received from the simulation"
        )
    );
    harness.shutdown().await
}

/// Static aperture pump for one gripper follower, standing at `$opening`.
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

/// One arm of an `execute_joint_path` goal: `arm_name` through `waypoints`,
/// with no expected start and these tests' settle tolerance on every joint.
fn path_arm(arm_name: &str, waypoints: &[[f64; 7]]) -> GoalArmsItem {
    GoalArmsItem {
        arm_name: arm_name.to_string(),
        waypoints: waypoints
            .iter()
            .map(|&joint_positions| GoalArmsItemWaypointsItem { joint_positions })
            .collect(),
        expected_start_joint_positions: Vec::new(),
        settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
    }
}

/// An `execute_joint_path` goal of `arms`, in that order, over `duration_s`,
/// with these tests' settle timeout.
fn joint_path(arms: Vec<GoalArmsItem>, duration_s: f64) -> execute_joint_path::GoalRequestData {
    execute_joint_path::GoalRequestData {
        arms,
        duration_s,
        settle_timeout_s: SETTLE_TIMEOUT_S,
    }
}

/// An `execute_joint_path` goal of the left arm through `waypoints` over
/// `duration_s` (see [`joint_path`]).
fn left_joint_path(waypoints: &[[f64; 7]], duration_s: f64) -> execute_joint_path::GoalRequestData {
    joint_path(vec![path_arm("left_arm", waypoints)], duration_s)
}

/// Send `goal` and give the node's answer to it.
async fn send_path_goal(
    harness: &Harness,
    goal: &execute_joint_path::GoalRequestData,
) -> peppygen::Result<execute_joint_path::GoalHandle> {
    execute_joint_path::send_goal(harness, goal, peppygen::DEFAULT_FEEDBACK_BOUND, DEADLINE).await
}

/// Send `goal` and assert that the node accepts it.
async fn send_joint_path(
    harness: &Harness,
    goal: &execute_joint_path::GoalRequestData,
) -> peppygen::Result<execute_joint_path::GoalHandle> {
    let sent = send_path_goal(harness, goal).await?;
    assert!(sent.accepted, "goal refused: {:?}", sent.refusal);
    Ok(sent)
}

/// Send `goal`, assert that the node accepts it, and wait for how it ends.
async fn run_joint_path(
    harness: &Harness,
    goal: &execute_joint_path::GoalRequestData,
) -> peppygen::Result<execute_joint_path::ResultOutcome> {
    let sent = send_joint_path(harness, goal).await?;
    Ok(sent.get_result(DEADLINE).await?.outcome)
}

/// Send `goal`, and assert that it completes: the result it completes with.
async fn complete_joint_path(
    harness: &Harness,
    goal: &execute_joint_path::GoalRequestData,
) -> peppygen::Result<execute_joint_path::ResultData> {
    match run_joint_path(harness, goal).await? {
        execute_joint_path::ResultOutcome::Completed(data) => Ok(data),
        other => panic!("execute_joint_path did not complete: {other:?}"),
    }
}

/// Send `goal`, and assert that it fails: the failure it fails with.
async fn fail_joint_path(
    harness: &Harness,
    goal: &execute_joint_path::GoalRequestData,
) -> peppygen::Result<execute_joint_path::Failure> {
    match run_joint_path(harness, goal).await? {
        execute_joint_path::ResultOutcome::Failed(failure) => Ok(failure),
        other => panic!("execute_joint_path did not fail: {other:?}"),
    }
}

/// A move_arm_joints goal on an arm that executes another move is refused
/// with `precondition_failed` in the words of a busy arm, which name no arm:
/// the goal names its one arm itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_move_arm_joints_goal_on_a_busy_arm_is_refused_in_the_words_of_a_busy_arm()
-> peppygen::Result<()> {
    let (harness, _arm) = start_with_left_arm().await?;
    let goal = move_arm_joints::GoalRequestData {
        arm_name: "left_arm".to_string(),
        joint_positions: [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0],
        duration_s: 30.0,
        settle_tolerances_rad: [SETTLE_TOLERANCE_RAD; 7],
        settle_timeout_s: SETTLE_TIMEOUT_S,
    };
    let send =
        || move_arm_joints::send_goal(&harness, &goal, peppygen::DEFAULT_FEEDBACK_BOUND, DEADLINE);
    let running = send().await?;
    assert!(running.accepted, "goal refused: {:?}", running.refusal);
    let refused = send().await?;
    assert_eq!(
        refused.refusal,
        Some(move_arm_joints::Failure::new(
            move_arm_joints::FailureKind::PreconditionFailed,
            "arm is already executing a motion"
        ))
    );
    running.cancel_goal(DEADLINE).await?;
    assert!(matches!(
        running.get_result(DEADLINE).await?.outcome,
        move_arm_joints::ResultOutcome::Cancelled { .. }
    ));
    harness.shutdown().await
}

/// A joint path whose arm stops short of its last waypoint fails with
/// `timeout` when the settle timeout runs out after the planned end, naming
/// the joint farthest from its target, and gives no result. Here the arm is
/// braked at HOME before the path starts, so it never moves.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_whose_arm_stops_short_fails_its_settle_with_timeout() -> peppygen::Result<()>
{
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let arm = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;
    assert_eq!(arm.brake().await, left_home());

    let mut out = left_home();
    out[2] = 0.3;
    out[3] = 0.4;
    let failure = fail_joint_path(&harness, &left_joint_path(&[out], 1.0)).await?;
    assert_eq!(failure.kind, execute_joint_path::FailureKind::Timeout);
    assert!(
        failure.message.starts_with("still ")
            && failure
                .message
                .contains(" rad from the target at joint 4, "),
        "the failure names the joint farthest from its target: {}",
        failure.message
    );
    assert!(
        seconds_after_planned_end(&failure.message) >= SETTLE_TIMEOUT_S,
        "the failure came no sooner than the settle timeout: {}",
        failure.message
    );
    harness.shutdown().await
}

// --- Joint paths (execute_joint_path) -------------------------------------

/// How far a commanded position may stand from the rounded path (rad), in
/// the path tests: a tick the guard slows lands on the chord between two
/// points of the path, a fraction of a milliradian off the arc it cuts.
const ON_PATH_RAD: f64 = 1e-3;

/// How far apart the points stand that the path tests draw a rounded path
/// with (rad): close enough that the polyline through them lies within a
/// micro-radian of the arcs.
const PATH_SPACING_RAD: f64 = 1e-3;

/// Five waypoints of the left arm out from [`home`] to the working posture
/// and back toward it, every corner turning, clear of the robot's body.
const FIVE_WAYPOINTS: [[f64; 7]; 5] = [
    [0.0, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0],
    [0.3, -0.5, 0.0, 0.9, 0.0, 0.0, 0.0],
    [0.2, -0.8, 0.2, 1.2, 0.0, 0.0, 0.0],
    [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0],
    [0.0, -0.5, 0.0, 0.9, 0.0, 0.0, 0.0],
];

fn distance_between(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The distance in joint space from `q` to the nearest of the straight
/// segments between consecutive `points`.
fn distance_to_segments(q: &[f64], points: &[Vec<f64>]) -> f64 {
    points
        .windows(2)
        .map(|w| {
            let run: Vec<f64> = w[1].iter().zip(&w[0]).map(|(b, a)| b - a).collect();
            let length_sq: f64 = run.iter().map(|r| r * r).sum();
            let along = q
                .iter()
                .zip(&w[0])
                .zip(&run)
                .map(|((q, a), r)| (q - a) * r)
                .sum::<f64>();
            let t = if length_sq > 0.0 {
                (along / length_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let nearest: Vec<f64> = w[0].iter().zip(&run).map(|(a, r)| a + t * r).collect();
            distance_between(q, &nearest)
        })
        .fold(f64::INFINITY, f64::min)
}

/// The rounded path a joint path takes from `start` through `waypoints`,
/// drawn as points [`PATH_SPACING_RAD`] apart along it.
fn rounded_path(start: &[f64], waypoints: &[Vec<f64>]) -> Vec<Vec<f64>> {
    use control_core::path_timing::RoundedPath;
    use control_core::positive_finite::PositiveFinite;
    let points: Vec<Vec<f64>> = std::iter::once(start.to_vec())
        .chain(waypoints.iter().cloned())
        .collect();
    let path = RoundedPath::joint_path(&points);
    path.arcs_every(PositiveFinite::parse(PATH_SPACING_RAD).expect("a positive spacing"))
        .iter()
        .map(|&s| path.at(s))
        .collect()
}

/// The left arm's waypoints as the arrays of a goal of the left arm.
fn left_waypoints(waypoints: &[[f64; 7]]) -> Vec<Vec<f64>> {
    waypoints.iter().map(|q| q.to_vec()).collect()
}

/// The trajectory of one arm through `points`, the first where the arm
/// stands, at the fastest the joint velocity and acceleration caps of
/// `parameters` allow, starting
/// now: the timing the node gives a path of that arm.
fn fastest_path(
    parameters: &peppygen::Parameters,
    points: &[[f64; 7]],
) -> control_core::path_timing::JointTrajectory<7> {
    let p = parameters;
    let velocity = [
        p.max_joint_velocity_rad_s_1,
        p.max_joint_velocity_rad_s_2,
        p.max_joint_velocity_rad_s_3,
        p.max_joint_velocity_rad_s_4,
        p.max_joint_velocity_rad_s_5,
        p.max_joint_velocity_rad_s_6,
        p.max_joint_velocity_rad_s_7,
    ];
    let acceleration = [
        p.max_joint_acceleration_rad_s2_1,
        p.max_joint_acceleration_rad_s2_2,
        p.max_joint_acceleration_rad_s2_3,
        p.max_joint_acceleration_rad_s2_4,
        p.max_joint_acceleration_rad_s2_5,
        p.max_joint_acceleration_rad_s2_6,
        p.max_joint_acceleration_rad_s2_7,
    ];
    control_core::path_timing::JointTrajectory::through(
        points,
        &velocity,
        &acceleration,
        Duration::ZERO,
        std::time::Instant::now(),
    )
}

/// The step `traj` takes each control tick of `parameters` when it plays at
/// full pace, each with the arc length, along the path, where it starts.
fn unslowed_steps(
    traj: &control_core::path_timing::JointTrajectory<7>,
    parameters: &peppygen::Parameters,
) -> Vec<(f64, f64)> {
    let period = Duration::from_secs(1) / parameters.control_rate_hz;
    let start = traj.clock().motion_start();
    let ticks = traj.clock().duration().div_duration_f64(period).ceil() as u32;
    let points: Vec<[f64; 7]> = (0..=ticks)
        .map(|k| traj.sample(start + period * k))
        .collect();
    steps_along(&points)
}

/// The steps between consecutive positions of `points`, each with the arc
/// length, along them, where it starts.
fn steps_along(points: &[[f64; 7]]) -> Vec<(f64, f64)> {
    points
        .windows(2)
        .scan(0.0, |arc, w| {
            let step = distance_between(&w[0], &w[1]);
            let from = *arc;
            *arc += step;
            Some((from, step))
        })
        .collect()
}

/// The steps between consecutive positions of `trail`.
fn steps(trail: &[[f64; 7]]) -> Vec<f64> {
    trail
        .windows(2)
        .map(|w| distance_between(&w[0], &w[1]))
        .collect()
}

/// The index of the position of `trail` nearest to `q`.
fn nearest_index(trail: &[[f64; 7]], q: &[f64; 7]) -> usize {
    (0..trail.len())
        .min_by(|&a, &b| distance_between(&trail[a], q).total_cmp(&distance_between(&trail[b], q)))
        .expect("the arm received setpoints")
}

/// The larger of the steps into and out of position `k` of a trail with
/// `steps`: how fast the arm passes there.
fn speed_at(steps: &[f64], k: usize) -> f64 {
    steps[k.saturating_sub(1)].max(steps[k.min(steps.len() - 1)])
}

/// Assert that every position of `trail` lies within [`ON_PATH_RAD`] of
/// `path`, and that some position cuts a corner of the straight segments
/// between `corners` by more than ten times that: the path is the rounded
/// one, not its segments.
fn assert_on_rounded_path(trail: &[Vec<f64>], path: &[Vec<f64>], corners: &[Vec<f64>]) {
    for (k, q) in trail.iter().enumerate() {
        let off = distance_to_segments(q, path);
        assert!(
            off <= ON_PATH_RAD,
            "setpoint {k} lies {off} rad off the rounded path: {q:?}"
        );
    }
    let cut = trail
        .iter()
        .map(|q| distance_to_segments(q, corners))
        .fold(0.0, f64::max);
    assert!(
        cut > 10.0 * ON_PATH_RAD,
        "no setpoint cuts a corner, so this case shows nothing: {cut} rad at most"
    );
}

/// Start the node with the robot ready, the left arm a perfect follower
/// (see [`spawn_left_arm_follower`]), the right arm at [`home`] and both
/// grippers half open, and wait until the backbone streams to both arms.
async fn start_with_left_arm() -> peppygen::Result<(Harness, ArmFollower)> {
    let (harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let arm = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);
    await_streaming(mocks.pairings.right_arm.joint_setpoints).await?;
    arm.first_setpoint().await;
    Ok((harness, arm))
}

/// A joint path of five waypoints keeps the arm moving through each inner
/// waypoint and settles at the last, and every position the governor
/// commands lies on the rounded path from where the arm stood.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_keeps_moving_through_its_inner_waypoints_on_its_rounded_path()
-> peppygen::Result<()> {
    let (harness, arm) = start_with_left_arm().await?;

    let data = complete_joint_path(&harness, &left_joint_path(&FIVE_WAYPOINTS, 0.0)).await?;
    assert_all_close(
        "final_joint_positions",
        &data.arms[0].final_joint_positions,
        &FIVE_WAYPOINTS[4],
        SETTLE_TOLERANCE_RAD,
    );
    let trail = arm.received();
    let corners: Vec<Vec<f64>> = std::iter::once(left_home().to_vec())
        .chain(left_waypoints(&FIVE_WAYPOINTS))
        .collect();
    assert_on_rounded_path(
        &left_waypoints(&trail),
        &rounded_path(&left_home(), &left_waypoints(&FIVE_WAYPOINTS)),
        &corners,
    );
    let steps = steps(&trail);
    let peak = steps.iter().copied().fold(0.0, f64::max);
    for waypoint in &FIVE_WAYPOINTS[..4] {
        let through = speed_at(&steps, nearest_index(&trail, waypoint));
        assert!(
            through > 0.1 * peak,
            "the arm slowed to {through} rad a tick at {waypoint:?}, against a peak of {peak}"
        );
    }
    harness.shutdown().await
}

/// A joint path whose expected start lies past the start tolerance of where
/// the arm measures fails with `precondition_failed` before the arm moves,
/// naming the arm and the joint. The same path with no expected start runs
/// from wherever the arm stands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expected_start_off_the_measured_arm_fails_the_path_before_it_moves()
-> peppygen::Result<()> {
    let (harness, arm) = start_with_left_arm().await?;

    let mut expected = left_home();
    expected[1] -= 0.2;
    let mut goal = left_joint_path(&FIVE_WAYPOINTS[..1], 0.0);
    goal.arms[0].expected_start_joint_positions = expected.to_vec();
    let failure = fail_joint_path(&harness, &goal).await?;
    assert_eq!(
        failure,
        execute_joint_path::Failure::new(
            execute_joint_path::FailureKind::PreconditionFailed,
            "left_arm stands 0.200 rad from expected_start_joint_positions at joint 2, past the \
             0.010 rad start tolerance; plan the path from where the arm stands"
        )
    );
    assert!(
        arm.received().iter().all(|q| *q == left_home()),
        "the arm moved: {:?}",
        arm.received()
    );

    goal.arms[0].expected_start_joint_positions = Vec::new();
    complete_joint_path(&harness, &goal).await?;
    harness.shutdown().await
}

/// A joint path whose straight legs keep the left hand clear of the right hand,
/// folded in toward it, but whose rounded corner cuts across the right hand
/// fails with `unreachable`, running none of the path, naming the stop
/// distance; the same path with its corner repeated stops there, runs its
/// straight legs and stays clear. The path runs in the plane of the left arm's
/// first two joints, around the right hand: out from under it and back up
/// beside it. Both grippers stand fully open, the widest fingers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_whose_rounded_corner_cuts_the_other_arm_fails_unreachable()
-> peppygen::Result<()> {
    let (mut harness, mocks) = start_ready_vacant(params(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let left = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    let right = spawn_right_arm_follower(
        mocks.pairings.right_arm.joint_states,
        mocks.pairings.right_arm.joint_setpoints,
    );
    pump_gripper_at!(
        mocks.pairings.left_gripper.gripper_states,
        peppygen::paired_topics::left_gripper::gripper_states,
        1.0
    );
    pump_gripper_at!(
        mocks.pairings.right_gripper.gripper_states,
        peppygen::paired_topics::right_gripper::gripper_states,
        1.0
    );
    // The first limb_states snapshot says that every limb reported, so a
    // joint move admits each arm.
    next_snapshot(&mut harness).await?;
    fold_both_arms_in(&harness).await?;
    let folded_left = *left.followed.borrow();
    let shift = |by_j1: f64, by_j2: f64| {
        let mut to = folded_left;
        to[0] += by_j1;
        to[1] += by_j2;
        to
    };
    let mut right_in = *right.followed.borrow();
    right_in[1] -= 0.42;
    complete_joint_path(
        &harness,
        &joint_path(vec![path_arm("right_arm", &[right_in])], 0.0),
    )
    .await?;
    let (start, corner, end) = (shift(-0.46, 0.51), shift(-0.09, 0.09), shift(0.33, 0.26));
    complete_joint_path(&harness, &left_joint_path(&[shift(-0.46, 0.0)], 0.0)).await?;
    complete_joint_path(&harness, &left_joint_path(&[start], 0.0)).await?;
    // The move's last setpoint reaches the follower before the failing path is sent, so
    // all it receives from here on is that path's.
    const SAME_RAD: f64 = 1e-9;
    let mut followed = left.followed.clone();
    tokio::time::timeout(
        DEADLINE,
        followed.wait_for(|q| {
            q.iter()
                .zip(&start)
                .all(|(q, to)| (q - to).abs() < SAME_RAD)
        }),
    )
    .await
    .expect("the move's last setpoint arrives")
    .expect("the follower runs");

    let before = left.received().len();
    let failure = fail_joint_path(&harness, &left_joint_path(&[corner, end], 0.0)).await?;
    assert_eq!(failure.kind, execute_joint_path::FailureKind::Unreachable);
    assert!(
        failure.message.contains("stop distance"),
        "{}",
        failure.message
    );
    let stood = left.received()[before - 1];
    assert!(
        left.received()[before..].iter().all(|q| *q == stood),
        "the arm moved before the path failed"
    );

    // The same path stopping at its corner runs its straight legs clear.
    complete_joint_path(&harness, &left_joint_path(&[corner, corner, end], 0.0)).await?;
    harness.shutdown().await
}

/// While the right arm moves in and the collision guard slows the left arm
/// on a joint path, every position the governor commands the left arm stays
/// on the rounded path, and the path still ends at its last waypoint. The
/// band is wide, 30 mm to 200 mm: with the right arm moving in alongside, the
/// guard holds some tick of the left arm, where the path runs at speed, to
/// under half the step the path takes there unslowed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_the_guard_slows_stays_on_its_rounded_path_and_ends_at_its_last_waypoint()
-> peppygen::Result<()> {
    let mut wide = params();
    wide.d_stop_m = 0.03;
    wide.d_safe_m = 0.2;
    let (mut harness, mocks) = start_ready_vacant(wide.clone(), MODEL).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let left = spawn_left_arm_follower(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
    );
    let right = spawn_right_arm_follower(
        mocks.pairings.right_arm.joint_states,
        mocks.pairings.right_arm.joint_setpoints,
    );
    pump_gripper_at!(
        mocks.pairings.left_gripper.gripper_states,
        peppygen::paired_topics::left_gripper::gripper_states,
        0.5
    );
    pump_gripper_at!(
        mocks.pairings.right_gripper.gripper_states,
        peppygen::paired_topics::right_gripper::gripper_states,
        0.5
    );
    // The first limb_states snapshot says that every limb reported, so a
    // joint move admits each arm.
    next_snapshot(&mut harness).await?;
    fold_both_arms_in(&harness).await?;

    let folded_left = *left.followed.borrow();
    let mut right_in = *right.followed.borrow();
    right_in[1] -= 0.3;
    let shift = |from: [f64; 7], joint: usize, by: f64| {
        let mut to = from;
        to[joint] += by;
        to
    };
    let w1 = shift(folded_left, 0, -0.3);
    let w2 = shift(w1, 1, 0.2);
    let w3 = shift(w2, 0, 0.3);
    let waypoints = [w1, w2, w3];

    let before = left.received().len();
    let mut right_goal = joint_path(vec![path_arm("right_arm", &[right_in])], 1.0);
    right_goal.settle_timeout_s = 30.0;
    let right_goal = send_joint_path(&harness, &right_goal).await?;
    let mut goal = left_joint_path(&waypoints, 0.0);
    goal.settle_timeout_s = 30.0;
    let left_goal = send_joint_path(&harness, &goal).await?;
    let data = match left_goal.get_result(DEADLINE).await?.outcome {
        execute_joint_path::ResultOutcome::Completed(data) => data,
        other => panic!("the slowed path did not end at its last waypoint: {other:?}"),
    };
    assert_all_close(
        "final_joint_positions",
        &data.arms[0].final_joint_positions,
        &w3,
        SETTLE_TOLERANCE_RAD,
    );
    let trail = &left.received()[before..];
    let unslowed = unslowed_steps(&fastest_path(&wide, &[folded_left, w1, w2, w3]), &wide);
    let peak = unslowed.iter().map(|&(_, step)| step).fold(0.0, f64::max);
    let unslowed_at = |arc: f64| {
        unslowed
            .iter()
            .take_while(|&&(from, _)| from <= arc)
            .last()
            .map_or(0.0, |&(_, step)| step)
    };
    let slowed = steps_along(trail)
        .into_iter()
        .filter(|&(from, _)| unslowed_at(from) >= 0.5 * peak)
        .map(|(from, step)| step / unslowed_at(from))
        .fold(f64::INFINITY, f64::min);
    assert!(
        slowed < 0.5,
        "the guard never held the left arm under half its unslowed step where the path \
         runs at speed: {slowed} of it at the least"
    );
    let corners: Vec<Vec<f64>> = std::iter::once(folded_left.to_vec())
        .chain(left_waypoints(&waypoints))
        .collect();
    assert_on_rounded_path(
        &left_waypoints(trail),
        &rounded_path(&folded_left, &left_waypoints(&waypoints)),
        &corners,
    );
    assert!(matches!(
        right_goal.get_result(DEADLINE).await?.outcome,
        execute_joint_path::ResultOutcome::Completed(_)
    ));
    harness.shutdown().await
}

/// A joint path that turns back at a waypoint stops there and turns back,
/// and a waypoint within 0.001 rad of the one before it is a stop: the arm
/// comes to rest at it before it goes on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_stops_where_it_turns_back_and_at_a_repeated_waypoint() -> peppygen::Result<()>
{
    let (harness, arm) = start_with_left_arm().await?;

    let rests_at = |trail: &[[f64; 7]], waypoint: &[f64; 7], what: &str| {
        let k = nearest_index(trail, waypoint);
        assert!(
            distance_between(&trail[k], waypoint) <= 1e-3,
            "the arm {what} {} rad short of the waypoint",
            distance_between(&trail[k], waypoint)
        );
        let steps = steps(trail);
        let peak = steps.iter().copied().fold(0.0, f64::max);
        let at = steps[k.saturating_sub(1)].min(steps[k.min(steps.len() - 1)]);
        assert!(
            at <= 0.1 * peak,
            "the arm {what} at {at} rad a tick, against a peak of {peak}"
        );
    };

    let out = FIVE_WAYPOINTS[0];
    complete_joint_path(&harness, &left_joint_path(&[out, left_home()], 0.0)).await?;
    rests_at(&arm.received(), &out, "turned back");

    let corner = FIVE_WAYPOINTS[1];
    let mut repeat = corner;
    repeat[0] += 0.0005;
    let before = arm.received().len();
    complete_joint_path(
        &harness,
        &left_joint_path(&[FIVE_WAYPOINTS[0], corner, repeat, FIVE_WAYPOINTS[2]], 0.0),
    )
    .await?;
    rests_at(&arm.received()[before..], &corner, "stopped at the repeat");
    harness.shutdown().await
}

/// A cancel and a stop each end a joint path, the result giving where the
/// arm stood; a second goal on the arm while one runs is refused; and a path
/// of 4 s at full speed asked to take 10 s ends no earlier than 10 s after
/// the test sends it, on the test's own clock.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_ends_on_a_cancel_or_a_stop_and_takes_the_duration_asked()
-> peppygen::Result<()> {
    use peppygen::fixtures::exposed_services::limb_motion::stop;

    let (harness, LeftFollowers { arm, .. }) = start_with_left_followers(0.0).await?;
    let mut moving = arm.clone();

    let goal = send_joint_path(&harness, &left_joint_path(&FIVE_WAYPOINTS, 30.0)).await?;
    tokio::time::timeout(
        DEADLINE,
        moving.wait_for(|q| distance_between(q, &left_home()) > 0.01),
    )
    .await
    .expect("the arm never started moving")
    .expect("follower watch open");
    for second in [
        left_joint_path(&FIVE_WAYPOINTS, 0.0),
        joint_path(
            vec![
                path_arm("right_arm", &[right_home()]),
                path_arm("left_arm", &[left_home()]),
            ],
            0.0,
        ),
    ] {
        let refused = send_path_goal(&harness, &second).await?;
        assert!(!refused.accepted, "a second goal on a busy arm ran");
        assert_eq!(
            refused.refusal.as_ref().map(|r| r.kind),
            Some(execute_joint_path::FailureKind::PreconditionFailed),
            "{:?}",
            refused.refusal
        );
    }
    goal.cancel_goal(DEADLINE).await?;
    let (reason, data) = match goal.get_result(DEADLINE).await?.outcome {
        execute_joint_path::ResultOutcome::Cancelled { reason, data } => (reason, data),
        other => panic!("the cancelled path did not end cancelled: {other:?}"),
    };
    assert_eq!(reason, "cancelled by its caller");
    let stood = *arm.borrow();
    assert_all_close(
        "final_joint_positions",
        &data.arms[0].final_joint_positions,
        &stood,
        0.05,
    );
    assert!(
        distance_between(&stood, &left_home()) > 0.01,
        "the arm never moved"
    );

    let goal = send_joint_path(&harness, &left_joint_path(&[left_home()], 30.0)).await?;
    tokio::time::timeout(
        DEADLINE,
        moving.wait_for(|q| distance_between(q, &stood) > 0.01),
    )
    .await
    .expect("the arm never started back")
    .expect("follower watch open");
    let answer = stop::poll(
        &harness,
        &stop::RequestData {
            reason: "the operator asked".to_string(),
        },
        DEADLINE,
    )
    .await?;
    assert_eq!(answer.stopped, ["left_arm"]);
    let (reason, data) = match goal.get_result(DEADLINE).await?.outcome {
        execute_joint_path::ResultOutcome::Cancelled { reason, data } => (reason, data),
        other => panic!("the stopped path did not end cancelled: {other:?}"),
    };
    assert_eq!(reason, "stopped: the operator asked");
    assert_all_close(
        "final_joint_positions",
        &data.arms[0].final_joint_positions,
        &*arm.borrow(),
        0.05,
    );

    // Back and forth over 0.8 rad of the second joint, stopping at each
    // turn: five runs of 0.8 s each at the arm's acceleration limit.
    let here = *arm.borrow();
    let mut out = here;
    out[1] -= 0.8;
    let shuttle = [out, here, out, here, out];
    let fastest = fastest_path(&params(), &[here, out, here, out, here, out])
        .clock()
        .duration()
        .as_secs_f64();
    assert!(
        (3.9..4.1).contains(&fastest),
        "the shuttle takes {fastest} s at full speed"
    );
    let asked_s = 10.0;
    let sent = std::time::Instant::now();
    complete_joint_path(&harness, &left_joint_path(&shuttle, asked_s)).await?;
    let took_s = sent.elapsed().as_secs_f64();
    assert!(
        took_s >= asked_s,
        "a path asked to take {asked_s} s ended {took_s} s after it was sent"
    );
    harness.shutdown().await
}

/// A goal the backbone does not take is refused at admission with
/// `rejected`, before any arm moves: an arm named twice, arms through
/// different numbers of waypoints, an expected start of the wrong length, a
/// non-finite expected start, and a waypoint outside the joint limits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_goal_the_backbone_does_not_take_is_refused_at_admission()
-> peppygen::Result<()> {
    let (harness, arm) = start_with_left_arm().await?;

    let twice = joint_path(
        vec![
            path_arm("left_arm", &[left_home()]),
            path_arm("left_arm", &[left_home()]),
        ],
        0.0,
    );
    let uneven = joint_path(
        vec![
            path_arm("right_arm", &[right_home()]),
            path_arm("left_arm", &[left_home(), left_home()]),
        ],
        0.0,
    );
    let mut short_start = left_joint_path(&[left_home()], 0.0);
    short_start.arms[0].expected_start_joint_positions = left_home()[..6].to_vec();
    let mut not_finite = left_joint_path(&[left_home()], 0.0);
    not_finite.arms[0].expected_start_joint_positions = vec![f64::NAN; 7];
    let mut beyond = left_home();
    beyond[1] = 1.0;
    let past_limits = left_joint_path(&[beyond], 0.0);
    let refusals = [
        (twice, "arms names \"left_arm\" twice: name each arm once"),
        (
            uneven,
            "left_arm holds 2 waypoints, where right_arm holds 1: every arm passes through as \
             many",
        ),
        (
            short_start,
            "expected_start_joint_positions of left_arm holds 6 values, where the arm has 7 \
             joints: give none, or one per joint",
        ),
        (
            not_finite,
            "expected_start_joint_positions of left_arm holds a non-finite value",
        ),
    ];
    for (goal, reason) in refusals {
        let refused = send_path_goal(&harness, &goal).await?;
        assert_eq!(
            refused.refusal,
            Some(execute_joint_path::Failure::new(
                execute_joint_path::FailureKind::Rejected,
                reason
            ))
        );
    }
    let refused = send_path_goal(&harness, &past_limits).await?;
    let refusal = refused.refusal.expect("a waypoint past the limits");
    assert_eq!(refusal.kind, execute_joint_path::FailureKind::Rejected);
    assert!(
        refusal
            .message
            .starts_with("waypoint 1 of left_arm puts joint 2 at 1 rad, outside its limits ["),
        "{}",
        refusal.message
    );
    assert!(
        arm.received().iter().all(|q| *q == left_home()),
        "an arm moved"
    );
    harness.shutdown().await
}

/// A joint path the joint velocity and acceleration caps cannot run within 60 s
/// fails with `rejected`, running none of the path, the message giving the time
/// they take: a hundred runs of 0.8 rad of the second joint, the arm stopping
/// at each turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_the_caps_cannot_run_within_60_s_fails_rejected() -> peppygen::Result<()> {
    let (harness, arm) = start_with_left_arm().await?;

    let mut out = left_home();
    out[1] -= 0.8;
    let shuttle: Vec<[f64; 7]> = (0..100)
        .map(|run| if run % 2 == 0 { out } else { left_home() })
        .collect();
    let points: Vec<[f64; 7]> = std::iter::once(left_home())
        .chain(shuttle.iter().copied())
        .collect();
    let time_s = fastest_path(&params(), &points)
        .clock()
        .duration()
        .as_secs_f64();
    assert!(time_s > 60.0, "the shuttle takes {time_s} s at full speed");
    let failure = fail_joint_path(&harness, &left_joint_path(&shuttle, 0.0)).await?;
    assert_eq!(
        failure,
        execute_joint_path::Failure::new(
            execute_joint_path::FailureKind::Rejected,
            format!(
                "the joint velocity and acceleration caps take {time_s:.1} s to run this path, \
                 past the 60 s a move may take; split it into paths of fewer waypoints"
            )
        )
    );
    assert!(
        arm.received().iter().all(|q| *q == left_home()),
        "the arm moved: {:?}",
        arm.received()
    );
    harness.shutdown().await
}

/// Where the paths of both arms take the arms from [`home`], each clear of
/// the robot's body: through the first waypoint of [`FIVE_WAYPOINTS`] and
/// the second, the right arm the mirror of half the left arm's travel.
const BOTH_ARMS_WAYPOINTS: [([f64; 7], [f64; 7]); 2] = [
    (
        [0.0, -0.3, 0.0, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.15, 0.0, 0.325, 0.0, 0.0, 0.0],
    ),
    (
        [0.3, -0.5, 0.0, 0.9, 0.0, 0.0, 0.0],
        [-0.15, 0.25, 0.0, 0.475, 0.0, 0.0, 0.0],
    ),
];

/// Start the node with the robot ready, both arms perfect followers (see
/// [`spawn_arm_follower`]) and both grippers half open, and wait for the
/// first limb_states snapshot, which says that streaming began.
async fn start_with_arm_followers() -> peppygen::Result<(Harness, ArmFollower, ArmFollower)> {
    start_with_arm_followers_off_by(MODEL, NO_STANDING_ERROR, NO_STANDING_ERROR).await
}

/// Start the node on a robot of `model` as [`start_with_arm_followers`] does,
/// the left arm standing `left_error` off each setpoint and the right arm
/// `right_error`.
async fn start_with_arm_followers_off_by(
    model: &str,
    left_error: [f64; 7],
    right_error: [f64; 7],
) -> peppygen::Result<(Harness, ArmFollower, ArmFollower)> {
    let (mut harness, mocks) = start_ready_vacant(params(), model).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    let left = spawn_left_arm_follower_off_by(
        mocks.pairings.left_arm.joint_states,
        mocks.pairings.left_arm.joint_setpoints,
        left_error,
    );
    let right = spawn_right_arm_follower_off_by(
        mocks.pairings.right_arm.joint_states,
        mocks.pairings.right_arm.joint_setpoints,
        right_error,
    );
    pump_gripper_at!(
        mocks.pairings.left_gripper.gripper_states,
        peppygen::paired_topics::left_gripper::gripper_states,
        0.5
    );
    pump_gripper_at!(
        mocks.pairings.right_gripper.gripper_states,
        peppygen::paired_topics::right_gripper::gripper_states,
        0.5
    );
    next_snapshot(&mut harness).await?;
    Ok((harness, left, right))
}

/// The arrays of the path through [`BOTH_ARMS_WAYPOINTS`] through both
/// arms' joints, the right arm's first.
fn right_first(waypoints: &[([f64; 7], [f64; 7])]) -> Vec<Vec<f64>> {
    waypoints
        .iter()
        .map(|(left, right)| [*right, *left].concat())
        .collect()
}

/// A goal of both arms through [`BOTH_ARMS_WAYPOINTS`], naming the right arm
/// first.
fn both_arms_path() -> execute_joint_path::GoalRequestData {
    let (left, right): (Vec<[f64; 7]>, Vec<[f64; 7]>) = BOTH_ARMS_WAYPOINTS.into_iter().unzip();
    joint_path(
        vec![path_arm("right_arm", &right), path_arm("left_arm", &left)],
        0.0,
    )
}

/// Both arms through one path of two waypoints: from the tick they leave
/// HOME, every tick the two arms' governed setpoints, together, lie on the
/// rounded path through both arms' joints, so the arms pass each waypoint
/// together; they reach the last waypoint on the same tick; and the result
/// gives both arms' joints in the order of the goal's arms.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_arms_move_along_one_path_and_reach_each_waypoint_together() -> peppygen::Result<()> {
    let (harness, left, right) = start_with_arm_followers().await?;

    let data = complete_joint_path(&harness, &both_arms_path()).await?;
    let (left_end, right_end) = BOTH_ARMS_WAYPOINTS[1];
    let names: Vec<&str> = data.arms.iter().map(|arm| arm.arm_name.as_str()).collect();
    assert_eq!(names, ["right_arm", "left_arm"]);
    for (arm, end) in data.arms.iter().zip([right_end, left_end]) {
        assert_all_close(
            "final_joint_positions",
            &arm.final_joint_positions,
            &end,
            SETTLE_TOLERANCE_RAD,
        );
    }
    // From the tick the arms leave HOME, which both do on the path's first
    // step: each follower's trail starts at the first setpoint it received.
    let moving = |follower: &ArmFollower, home: [f64; 7]| -> Vec<[f64; 7]> {
        follower
            .received()
            .into_iter()
            .skip_while(|q| *q == home)
            .collect()
    };
    let (left_trail, right_trail) = (moving(&left, left_home()), moving(&right, right_home()));
    let together: Vec<Vec<f64>> = right_trail
        .iter()
        .zip(&left_trail)
        .map(|(right, left)| [*right, *left].concat())
        .collect();
    let start = [right_home(), left_home()].concat();
    let corners: Vec<Vec<f64>> = std::iter::once(start.clone())
        .chain(right_first(&BOTH_ARMS_WAYPOINTS))
        .collect();
    assert_on_rounded_path(
        &together,
        &rounded_path(&start, &right_first(&BOTH_ARMS_WAYPOINTS)),
        &corners,
    );
    let arrived = |trail: &[[f64; 7]], end: &[f64; 7]| {
        trail.iter().position(|q| distance_between(q, end) < 1e-9)
    };
    assert_eq!(
        arrived(&left_trail, &left_end),
        arrived(&right_trail, &right_end)
    );
    assert!(
        arrived(&left_trail, &left_end).is_some(),
        "the arms reach the end"
    );
    harness.shutdown().await
}

/// A path of both arms whose expected start lies past the start tolerance of
/// either arm fails with `precondition_failed` before either arm moves,
/// naming the arm and its joint.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expected_start_of_either_arm_off_its_tolerance_fails_the_path_before_either_moves()
-> peppygen::Result<()> {
    let (harness, left, right) = start_with_arm_followers().await?;

    // The second joints run the other way on the two arms, so each start
    // stays inside its arm's limits.
    let shoulder_by = |home: [f64; 7], by: f64| {
        let mut q = home;
        q[1] += by;
        q
    };
    let off_starts = [
        ((shoulder_by(left_home(), -0.2), right_home()), "left_arm"),
        ((left_home(), shoulder_by(right_home(), 0.2)), "right_arm"),
    ];
    for ((left_start, right_start), arm) in off_starts {
        let mut goal = both_arms_path();
        goal.arms[0].expected_start_joint_positions = right_start.to_vec();
        goal.arms[1].expected_start_joint_positions = left_start.to_vec();
        let failure = fail_joint_path(&harness, &goal).await?;
        assert_eq!(
            failure,
            execute_joint_path::Failure::new(
                execute_joint_path::FailureKind::PreconditionFailed,
                format!(
                    "{arm} stands 0.200 rad from expected_start_joint_positions at joint 2, past \
                     the 0.010 rad start tolerance; plan the path from where the arm stands"
                )
            )
        );
    }
    for (arm, follower, home) in [
        ("left", &left, left_home()),
        ("right", &right, right_home()),
    ] {
        assert!(
            follower.received().iter().all(|q| *q == home),
            "the {arm} arm moved"
        );
    }
    harness.shutdown().await
}

// --- Leaving the stop distance ---------------------------------------------

/// Both arms with the elbow bent square, the pose the paths that leave the
/// stop distance start from.
const ELBOW_SQUARE: [f64; 7] = [0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0];

/// How far the left arm stands off each setpoint in the tests that hold it
/// at the stop distance: 1 mrad on at its third joint, which turns its hand
/// in toward the right arm from [`ELBOW_SQUARE`].
const LEFT_STANDS_IN: [f64; 7] = [0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 0.0];

/// How far the right arm stands off each setpoint in those tests: 1 mrad
/// back at its third joint, which turns its hand in toward the left arm.
const RIGHT_STANDS_IN: [f64; 7] = [0.0, 0.0, -1e-3, 0.0, 0.0, 0.0, 0.0];

/// [`ELBOW_SQUARE`] with the third joint at `j3`.
fn square_turned(j3: f64) -> [f64; 7] {
    let mut q = ELBOW_SQUARE;
    q[2] = j3;
    q
}

/// The command the node holds `arm` at once the move that brought it there
/// has ended: the first setpoint the arm receives twice in a row. While the
/// collision guard holds a move short, the held command creeps on by less
/// each tick; once the move ends, the command stands still.
async fn held_command(arm: &ArmFollower) -> [f64; 7] {
    let mut followed = arm.followed.clone();
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        if let [.., last, next] = arm.received()[..]
            && last == next
        {
            return next;
        }
        tokio::time::timeout_at(deadline, followed.changed())
            .await
            .expect("the arm's command never stood still")
            .expect("the follower runs");
    }
}

/// Start the node on v2 with the robot ready, the left arm standing
/// `left_error` off each setpoint and the right arm `right_error`
/// ([`start_with_arm_followers_off_by`]); bring both arms to
/// [`ELBOW_SQUARE`] on one path; then send the left hand in toward the right
/// arm at its third joint, where the collision guard holds the left arm's
/// command at the stop distance and the move fails with `timeout`. Gives the
/// harness, both followers and the left arm's held command.
async fn start_with_the_left_arm_held_at_the_stop_distance(
    left_error: [f64; 7],
    right_error: [f64; 7],
) -> peppygen::Result<(Harness, ArmFollower, ArmFollower, [f64; 7])> {
    let (harness, left, right) =
        start_with_arm_followers_off_by(V2, left_error, right_error).await?;
    let both_square = joint_path(
        vec![
            path_arm("left_arm", &[ELBOW_SQUARE]),
            path_arm("right_arm", &[ELBOW_SQUARE]),
        ],
        0.0,
    );
    complete_joint_path(&harness, &both_square).await?;
    match left_arm_joint_move(&harness, square_turned(1.2), 1.0).await? {
        move_arm_joints::ResultOutcome::Failed(failure) => assert_eq!(
            failure.kind,
            move_arm_joints::FailureKind::Timeout,
            "{}",
            failure.message
        ),
        other => panic!("the guard holds the left hand short of the right arm: {other:?}"),
    }
    let held = held_command(&left).await;
    assert!(
        held[2] > 0.3 && held[2] < 1.1,
        "the guard holds the left arm on its way in: {held:?}"
    );
    Ok((harness, left, right, held))
}

/// An arm the collision guard holds at the stop distance leaves it on a
/// joint path. The left arm, sent in toward the right arm, stands 1 mrad
/// further in than its held command, so it measures inside the stop
/// distance: a path from there further in fails with `unreachable` before
/// the arm moves, its command left where the guard held it, and a path back
/// out completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_leaves_the_stop_distance_the_guard_holds_an_arm_at() -> peppygen::Result<()> {
    let (harness, left, _right, held) =
        start_with_the_left_arm_held_at_the_stop_distance(LEFT_STANDS_IN, NO_STANDING_ERROR)
            .await?;

    let before = left.received().len();
    let further_in = left_joint_path(&[square_turned(held[2] + 0.2)], 0.0);
    let failure = fail_joint_path(&harness, &further_in).await?;
    assert_eq!(
        failure.kind,
        execute_joint_path::FailureKind::Unreachable,
        "{}",
        failure.message
    );
    assert!(
        failure.message.contains("stop distance"),
        "{}",
        failure.message
    );
    assert!(
        left.received()[before..].iter().all(|q| *q == held),
        "the refused path moved the left arm's command off where the guard held it"
    );

    let data = complete_joint_path(&harness, &left_joint_path(&[ELBOW_SQUARE], 0.0)).await?;
    assert_all_close(
        "final_joint_positions",
        &data.arms[0].final_joint_positions,
        &ELBOW_SQUARE,
        SETTLE_TOLERANCE_RAD,
    );
    harness.shutdown().await
}

/// The other arm leaves the stop distance the collision guard holds an arm
/// at on a joint path of its own: with the left arm held there, the right
/// arm, standing 1 mrad in toward the left arm, turns its hand away along a
/// path of the right arm alone, which completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_of_the_other_arm_leaves_the_stop_distance_an_arm_is_held_at()
-> peppygen::Result<()> {
    let (harness, _left, _right, _held) =
        start_with_the_left_arm_held_at_the_stop_distance(LEFT_STANDS_IN, RIGHT_STANDS_IN).await?;

    let away = [square_turned(0.3), square_turned(0.6)];
    let data = complete_joint_path(
        &harness,
        &joint_path(vec![path_arm("right_arm", &away)], 0.0),
    )
    .await?;
    assert_all_close(
        "final_joint_positions",
        &data.arms[0].final_joint_positions,
        &away[1],
        SETTLE_TOLERANCE_RAD,
    );
    harness.shutdown().await
}

/// A joint path of both arms leaves the stop distance the collision guard
/// holds an arm at: with the left arm held there and both arms standing 1
/// mrad in toward each other, a path of both arms that keeps the left arm
/// where it stands and turns the right hand away completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joint_path_of_both_arms_leaves_the_stop_distance_an_arm_is_held_at()
-> peppygen::Result<()> {
    let (harness, left, _right, _held) =
        start_with_the_left_arm_held_at_the_stop_distance(LEFT_STANDS_IN, RIGHT_STANDS_IN).await?;

    let left_stands = *left.followed.borrow();
    let away = square_turned(0.6);
    let data = complete_joint_path(
        &harness,
        &joint_path(
            vec![
                path_arm("left_arm", &[left_stands]),
                path_arm("right_arm", &[away]),
            ],
            0.0,
        ),
    )
    .await?;
    assert_all_close(
        "final_joint_positions",
        &data.arms[1].final_joint_positions,
        &away,
        SETTLE_TOLERANCE_RAD,
    );
    harness.shutdown().await
}

/// Waits for the fault that ends the node and gives its reason with each error it
/// comes from.
async fn fault_reason(harness: &Harness, fault: &control_core::fault::Fault) -> String {
    tokio::time::timeout(
        DEADLINE,
        harness.node_runner().cancellation_token().cancelled(),
    )
    .await
    .expect("the fault shuts the node down");
    let reason = fault.take().expect("the node ends with a fault");
    std::iter::successors(Some(reason.as_ref() as &dyn std::error::Error), |&error| {
        error.source()
    })
    .map(ToString::to_string)
    .collect::<Vec<_>>()
    .join(": ")
}

/// A backbone whose robot's initializer names a model of other arms ends with a
/// fault that names the served model and the models the backbone drives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backbone_whose_initializer_serves_a_model_it_does_not_drive_ends() -> peppygen::Result<()>
{
    let (harness, mocks, fault) = helpers::start(Config {
        parameters: Some(params()),
        collision_ctrl_vacant: true,
        perception_geometry_vacant: true,
        ..Default::default()
    })
    .await?;
    answer_robot_identity(&mocks, "so101").await?;
    let reason = fault_reason(&harness, &fault).await;
    assert!(
        reason.starts_with("the model the robot's initializer serves: model so101 describes")
            && reason.ends_with("the models of that shape are \"openarm_v1\" and \"openarm_v2\""),
        "{reason}"
    );
    harness.shutdown().await
}

/// A backbone whose robot's initializer serves the description of another
/// build of the model ends with a fault that names both descriptions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backbone_whose_initializer_serves_another_description_ends() -> peppygen::Result<()> {
    let (harness, mocks, fault) = helpers::start(Config {
        parameters: Some(params()),
        collision_ctrl_vacant: true,
        perception_geometry_vacant: true,
        ..Default::default()
    })
    .await?;
    let other = "0".repeat(64);
    answer_identity(&mocks, MODEL).await?;
    answer_description(&mocks, MODEL, Some(&other)).await?;
    let reason = fault_reason(&harness, &fault).await;
    let built = robot_description::model(MODEL)
        .expect("a bundled model")
        .digest();
    assert_eq!(
        reason,
        format!(
            "the robot's initializer serves the description {other} of {MODEL}, and this \
             backbone was built with {built}: build the backbone and the initializer from one \
             public-peppy-libs commit"
        )
    );
    harness.shutdown().await
}
