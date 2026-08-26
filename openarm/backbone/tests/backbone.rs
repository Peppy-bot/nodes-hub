//! Integration tests over the generated harness: the backbone in-process, the
//! robot initializer, the leading node and the four limb followers all played
//! by generated mocks over the real wire.
//!
//! Fixture notes, from what the node actually does:
//!
//! - `startup::wait_until_ready` gates everything (subscriptions, publishers,
//!   actions, the coordinator) on `robot_init/is_ready` answering `ready: true`,
//!   so every test pumps that mock service.
//! - `coordinator::seed_all` then gates streaming on a first measured state
//!   from BOTH arms and BOTH grippers, and `liveness` freezes a limb that goes
//!   silent for 4 control periods. The harness pins every pairing slot to a
//!   mock (only the `collision_ctrl` dependency slot has a `_vacant` knob), so
//!   each test pumps all four follower back-channels at a rate inside the
//!   stale limit. The unselected slots (pose pairs, upstream gripper pairs,
//!   and any leader pair a test does not drive) stay silent, which is exactly
//!   what an unbound optional slot delivers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use peppygen::fixtures::harness::{Config, Harness};
use tokio::sync::watch;

/// How long a mock pump waits parked for the node's next poll or for the
/// node's subscription to appear; only expires once the harness is gone.
const PUMP_TIMEOUT: Duration = Duration::from_secs(120);

/// Control rate for the tests: 20 ms cycle (Nyquist 25 Hz, above the 15 Hz
/// velocity-filter default) and an 80 ms stale limit, comfortably above the
/// 10 ms state pumps even on a loaded machine.
const CONTROL_RATE_HZ: u32 = 50;

/// Follower back-channel period; must stay well inside the 4-cycle (80 ms)
/// liveness stale limit or the coordinator freezes that limb.
const STATE_PUMP_PERIOD: Duration = Duration::from_millis(10);

/// Rest pose both arms are seeded at: elbow (j4) exactly at the description's
/// 0.05 rad singularity floor, so the planner's limit clamp is the identity
/// and held setpoints echo it bit-exact. Same pose as the governor's own
/// unit-test `home()`; nearest-pair clearance there is ~24 mm.
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
/// with the left arm: right arm at [`HOME`], both grippers half open.
macro_rules! pump_right_arm_and_grippers {
    ($mocks:ident) => {
        pump_arm_at_home!(
            $mocks.pairings.right_arm_link.joint_states,
            peppygen::paired_topics::right_arm_link::joint_states
        );
        pump_gripper_half_open!(
            $mocks.pairings.left_gripper_link.gripper_states,
            peppygen::paired_topics::left_gripper_link::gripper_states
        );
        pump_gripper_half_open!(
            $mocks.pairings.right_gripper_link.gripper_states,
            peppygen::paired_topics::right_gripper_link::gripper_states
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

/// Plays the left arm as a perfect follower: publishes its measured state
/// every [`STATE_PUMP_PERIOD`], adopting each governed setpoint the node
/// streams down as the new measurement. The returned watch carries the latest
/// adopted position, so a test can assert what motion the arm mock observed.
fn spawn_left_arm_follower(
    states: peppygen::mock::pairings::left_arm_link::joint_states::Publisher,
    mut setpoints: peppygen::mock::pairings::left_arm_link::joint_setpoints::Subscription,
) -> watch::Receiver<[f64; 7]> {
    let (tx, rx) = watch::channel(HOME);
    tokio::spawn(async move {
        if !matches!(states.wait_for_subscriber(PUMP_TIMEOUT).await, Ok(true)) {
            return;
        }
        let mut positions = HOME;
        let mut ticker = tokio::time::interval(STATE_PUMP_PERIOD);
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let message = peppygen::paired_topics::left_arm_link::joint_states::Message {
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
                    Ok(Some(m)) if m.positions.len() == 7 => {
                        for (slot, v) in positions.iter_mut().zip(&m.positions) {
                            *slot = *v;
                        }
                        tx.send_replace(positions);
                    }
                    Ok(Some(_)) | Err(_) => {}
                    Ok(None) => return,
                }
            }
        }
    });
    rx
}

/// The full boot with the robot reporting ready and the `collision_ctrl`
/// dependency slot vacant (its `zero_or_one` empty binding): the launch-time
/// band stands, exactly the branch the cardinality exists for.
async fn start_ready_vacant(
    parameters: peppygen::Parameters,
) -> peppygen::Result<(Harness, peppygen::fixtures::harness::Mocks)> {
    let (harness, mocks) = Harness::start_with(
        Config {
            parameters: Some(parameters),
            collision_ctrl_vacant: true,
            ..Default::default()
        },
        openarm_backbone::setup,
    )
    .await?;
    assert!(
        mocks.deps.collision_ctrl.is_none(),
        "a vacant collision_ctrl slot must start no mock"
    );
    Ok((harness, mocks))
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
        mocks.pairings.left_arm_link.joint_states,
        peppygen::paired_topics::left_arm_link::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    let mut left_wire = mocks.pairings.left_arm_link.joint_setpoints;
    let mut right_wire = mocks.pairings.right_arm_link.joint_setpoints;
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
    use peppygen::fixtures::exposed_actions::limb_motion::move_arm_joints;

    // collision_ctrl bound to its (silent) mock this time: the listener runs
    // against a live producer that never publishes, and the launch band stands.
    let (harness, mocks) = Harness::start_with(
        Config {
            parameters: Some(params()),
            collision_ctrl_vacant: false,
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
        mocks.pairings.left_arm_link.joint_states,
        mocks.pairings.left_arm_link.joint_setpoints,
    );
    pump_right_arm_and_grippers!(mocks);

    // Gate the goal on streaming having begun: a goal that reaches the
    // coordinator while `seed_all` still waits for first states is refused
    // ("the follower has not reported its first state yet"), so wait for the
    // first governed setpoint on the (uncommanded) right wire first.
    let mut right_wire = mocks.pairings.right_arm_link.joint_setpoints;
    tokio::time::timeout(DEADLINE, right_wire.next())
        .await
        .expect("streaming never began before the goal")?
        .expect("right arm subscription open");

    // A known-good working posture (the coordinator's own unit tests hold it),
    // well clear of the other arm, so the tiny validated band never throttles.
    let target = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
    let goal = move_arm_joints::send_goal(
        &harness,
        &move_arm_joints::GoalRequestData {
            arm_name: "left_arm".to_string(),
            joint_positions: target,
            duration_s: 1.0,
        },
        peppygen::QoSProfile::Reliable,
        DEADLINE,
    )
    .await?;
    assert!(goal.accepted, "goal rejected: {:?}", goal.reason);

    // The result parks until the trajectory completes. This action exposes no
    // feedback stream (peppy.json5 defines none), so the result is the whole
    // terminal protocol.
    let result = goal.get_result(DEADLINE).await?;
    let data = match result.outcome {
        move_arm_joints::ResultOutcome::Completed(data) => data,
        other => panic!("move_arm_joints did not complete: {other:?}"),
    };
    assert!(data.success, "move failed: {}", data.message);
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
        mocks.pairings.left_arm_link.joint_states,
        peppygen::paired_topics::left_arm_link::joint_states
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
    let mut left_wire = mocks.pairings.left_arm_link.joint_setpoints;
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
    // Widen the band: rest-pose clearance (~24 mm) is far under d_stop, so any
    // closing candidate is denied outright (Stopped, not Throttling) and the
    // measured-state tripwire is armed from the first tick.
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
        mocks.pairings.left_arm_link.joint_states,
        peppygen::paired_topics::left_arm_link::joint_states
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
        mocks.pairings.left_arm_link.joint_states,
        peppygen::paired_topics::left_arm_link::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    let snapshot = tokio::time::timeout(DEADLINE, harness.emitted.limb_state_limb_states.next())
        .await
        .expect("no limb_states snapshot before the deadline")?
        .expect("limb_states subscription open");

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

/// A goal naming no limb of this robot is refused at admission with the name
/// table quoted, for the arm and gripper moves alike.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn goals_for_unknown_limb_names_are_refused() -> peppygen::Result<()> {
    use peppygen::fixtures::exposed_actions::limb_motion::{move_arm_joints, move_gripper};

    let (harness, mocks) = start_ready_vacant(params()).await?;
    pump_is_ready(
        mocks.deps.robot_init.is_ready,
        Arc::new(AtomicBool::new(true)),
    );
    pump_arm_at_home!(
        mocks.pairings.left_arm_link.joint_states,
        peppygen::paired_topics::left_arm_link::joint_states
    );
    pump_right_arm_and_grippers!(mocks);

    // Wait for streaming so the refusal below is the name check, not the
    // seed gate's blanket refusal.
    let mut right_wire = mocks.pairings.right_arm_link.joint_setpoints;
    tokio::time::timeout(DEADLINE, right_wire.next())
        .await
        .expect("streaming never began")?
        .expect("right arm subscription open");

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
