//! Taking this robot's seat in a simulation and holding it: the robot
//! attaches, its limbs' setpoints go out as one command at the command rate,
//! and every state that comes back is published on the limb it measures.
//!
//! The seat lasts as long as the robot's stay in the scene. Taking it fails
//! when the simulation refuses the robot, so the robot reports no readiness
//! it cannot back. Losing the seat stops the node, which `peppy stack list`
//! then reports failed; `peppy stack join` puts the copy back.

use std::sync::Arc;
use std::time::Duration;

use peppygen::consumed_actions::simulation::attach;
use peppygen::consumed_services::simulation::command;
use peppygen::paired_topics::{
    left_arm_link, left_gripper_link, right_arm_link, right_gripper_link,
};
use peppygen::{NodeRunner, Parameters, QoSProfile, Result};
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;
use tracing::{error, info, warn};

use crate::limbs::{ARMS, GRIPPERS, Seat, arm_command, gripper_command};
use crate::refused;

/// How long the robot waits for the simulation to seat it. The engine
/// rebuilds its world around the joining robot, which on a mesh-heavy model
/// takes a moment.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one command may take. The loop waits for the reply before its
/// next tick, so a simulation that stops answering costs this much before
/// the robot tries again.
const COMMAND_TIMEOUT: Duration = Duration::from_millis(250);
/// How long the robot waits for the engine to take it out of the scene, or
/// to say why it did.
const LEAVE_TIMEOUT: Duration = Duration::from_secs(2);
/// How long the shutdown hook waits for the seat to come back, inside
/// peppy's shutdown grace (5 s by default, `lifecycle.shutdown_grace_secs`).
const SEAT_RELEASE_TIMEOUT: Duration = Duration::from_secs(3);
// The wait outlasts what it waits for, so a robot that does leave is not
// reported as one that did not.
const _: () = assert!(SEAT_RELEASE_TIMEOUT.as_millis() > LEAVE_TIMEOUT.as_millis());
/// Pause after a receive error before retrying, so a persistently broken
/// subscription cannot hot-spin a consumer or flood the log.
const RECEIVE_ERROR_BACKOFF: Duration = Duration::from_millis(100);
/// The command rates the engine can be driven at, the upper end being the
/// fastest loop any OpenArm runs.
const COMMAND_RATE_HZ: std::ops::RangeInclusive<u32> = 1..=1000;

/// The simulation that seats this robot: where its attachment goes and where
/// its commands go. Both address the one `simulation` slot, so a robot that
/// drives its own hardware resolves neither and takes no seat.
struct Simulation {
    attach_to: ProducerRef,
    command_to: ProducerRef,
}

impl Simulation {
    fn bound(runner: &NodeRunner) -> Option<Self> {
        let (attach_to, command_to) = (
            attach::bound_producer(runner)?,
            command::bound_producer(runner)?,
        );
        Some(Self {
            attach_to: attach_to.clone(),
            command_to: command_to.clone(),
        })
    }
}

/// Takes this robot's seat in the simulation the launcher bound it to. A
/// robot with no simulation bound drives the limbs the launcher gave it and
/// returns without a seat.
pub async fn take(params: &Parameters, runner: &Arc<NodeRunner>) -> Result<()> {
    let Some(simulation) = Simulation::bound(runner) else {
        info!("no simulation seats this robot, so it takes no seat");
        return Ok(());
    };
    let model = params.model.clone();
    if model.is_empty() {
        return Err(refused(
            "this robot takes a seat in a simulation and names no model for it to stand: set `model` to an id of that simulation's robot catalogue, such as openarm_v2",
        ));
    }
    peppygen::clock::init(runner).await?;
    let token = runner.cancellation_token().clone();
    let rate = params.command_rate_hz;
    if !COMMAND_RATE_HZ.contains(&rate) {
        return Err(refused(format!(
            "command_rate_hz must be between {} and {}, and this robot's is {rate}",
            COMMAND_RATE_HZ.start(),
            COMMAND_RATE_HZ.end()
        )));
    }
    let period = Duration::from_secs_f64(1.0 / f64::from(rate));
    let placement =
        (!params.placement.auto).then_some(attach::SimulationRobotAttachActionGoalPlacement {
            position: [params.placement.x, params.placement.y, params.placement.z],
            yaw: params.placement.yaw,
        });

    // The seat: the simulation stands this robot and says which limbs it
    // gave it. Feedback is a state stream the engine publishes on its own
    // grid without waiting for anyone, so it is read as sensor data.
    let goal = attach::ActionHandle::fire_goal(
        runner,
        &simulation.attach_to,
        ATTACH_TIMEOUT,
        attach::GoalRequest {
            model: model.clone(),
            placement,
        },
        QoSProfile::SensorData,
    )
    .await?;
    if !goal.accepted {
        return Err(refused(format!(
            "the simulation refused this robot: {}",
            goal.reason.unwrap_or_else(|| "no reason given".into())
        )));
    }
    let seated = goal
        .data
        .clone()
        .ok_or_else(|| refused("the simulation seated this robot without naming its limbs"))?;
    let seat = Seat::of(&seated).map_err(refused)?;
    info!(
        "{model} joined the simulation with arms [{}] and grippers [{}]",
        seated.arm_names.join(", "),
        seated.gripper_names.join(", ")
    );

    spawn_arm_consumers(runner, &seat, &token).await?;
    spawn_gripper_consumers(runner, &seat, &token).await?;
    let (left, has_left) = tokio::sync::oneshot::channel();
    let states = spawn_state_publishers(runner, seat.clone(), goal, left, token.clone()).await?;
    let commands = tokio::spawn(command_loop(
        runner.clone(),
        simulation.command_to,
        seat,
        period,
        token.clone(),
    ));

    // Shutting this robot down takes it out of the scene: the wind-down
    // gives its seat back, and the node waits for that, so the body leaves
    // with the robot.
    let shutdown_token = token.clone();
    runner.on_shutdown(async move {
        shutdown_token.cancel();
        if tokio::time::timeout(SEAT_RELEASE_TIMEOUT, has_left)
            .await
            .is_err()
        {
            warn!("this robot did not leave the scene within {SEAT_RELEASE_TIMEOUT:?}");
        }
    });

    // The robot's stay is over when either half of the seat ends: stop the
    // node, so a robot with no seat stops serving the readiness that stands
    // for one.
    tokio::spawn(async move {
        tokio::select! {
            _ = states => {}
            _ = commands => {}
        }
        token.cancel();
    });
    Ok(())
}

/// Sends the limbs' latest setpoints to the engine at the command rate. The
/// call is also this robot's heartbeat, so it goes out whether or not a
/// setpoint changed.
async fn command_loop(
    runner: Arc<NodeRunner>,
    command_to: ProducerRef,
    seat: Seat,
    period: Duration,
    token: CancellationToken,
) {
    let mut ticker = tokio::time::interval(period);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut failing = false;
    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            _ = ticker.tick() => {}
        }
        let answered = command::poll(&runner, &command_to, COMMAND_TIMEOUT, seat.request()).await;
        let outcome = match answered {
            Ok(response) if response.data.success => Ok(()),
            // The engine is still standing this robot: its first commands
            // land once it is seated, and the robot holds its start pose
            // meanwhile.
            Ok(response) if response.data.message.contains("still joining") => Ok(()),
            Ok(response) => Err(response.data.message),
            Err(e) => Err(e.to_string()),
        };
        match outcome {
            Ok(()) => failing = false,
            Err(e) if !failing => {
                failing = true;
                warn!("commanding the simulation is failing, suppressing repeats: {e}");
            }
            Err(_) => {}
        }
    }
}

/// Encodes one arm's measured state for the slot that publishes it.
type ArmStateBuilder =
    fn(std::time::SystemTime, Vec<f64>, Vec<f64>, Vec<f64>) -> Result<peppylib::Payload>;
/// Encodes one gripper's measured state for the slot that publishes it.
type GripperStateBuilder = fn(std::time::SystemTime, f64, f64, f64) -> Result<peppylib::Payload>;

/// Publishes every state the simulation sends back on the limb it measures,
/// and ends when the robot's stay does.
async fn spawn_state_publishers(
    runner: &Arc<NodeRunner>,
    seat: Seat,
    mut goal: attach::ActionHandle,
    left: tokio::sync::oneshot::Sender<()>,
    token: CancellationToken,
) -> Result<tokio::task::JoinHandle<()>> {
    let arms = [
        (
            ARMS[0].0,
            left_arm_link::joint_states::declare_publisher(runner).await?,
            left_arm_link::joint_states::build_message as ArmStateBuilder,
        ),
        (
            ARMS[1].0,
            right_arm_link::joint_states::declare_publisher(runner).await?,
            right_arm_link::joint_states::build_message as ArmStateBuilder,
        ),
    ];
    let grippers = [
        (
            GRIPPERS[0].0,
            left_gripper_link::gripper_states::declare_publisher(runner).await?,
            left_gripper_link::gripper_states::build_message as GripperStateBuilder,
        ),
        (
            GRIPPERS[1].0,
            right_gripper_link::gripper_states::declare_publisher(runner).await?,
            right_gripper_link::gripper_states::build_message as GripperStateBuilder,
        ),
    ];
    Ok(tokio::spawn(async move {
        let mut failing = false;
        let mut first = true;
        let mut ended_by_engine = false;
        loop {
            let feedback = tokio::select! {
                _ = token.cancelled() => break,
                feedback = goal.on_next_feedback_message() => feedback,
            };
            let feedback = match feedback {
                Ok(feedback) => feedback,
                // The stay is over on the engine's side: it completed the
                // goal, or it is gone. Its own account of why comes from
                // the goal's result.
                Err(e) => {
                    info!("this robot's seat in the simulation ended: {e}");
                    ended_by_engine = true;
                    break;
                }
            };
            let mut published = Ok(());
            for (slot, publisher, build) in &arms {
                let Some(state) = seat.arm_state(slot, &feedback) else {
                    continue;
                };
                let message = build(
                    feedback.timestamp,
                    state.positions.clone(),
                    state.velocities.clone(),
                    Vec::new(),
                );
                published = published.and(publish(publisher, message).await);
            }
            for (slot, publisher, build) in &grippers {
                let Some(state) = seat.gripper_state(slot, &feedback) else {
                    continue;
                };
                let message = build(
                    feedback.timestamp,
                    state.opening,
                    state.effort,
                    seat.gripper_max_effort(slot),
                );
                published = published.and(publish(publisher, message).await);
            }
            match published {
                Ok(()) => {
                    failing = false;
                    if first {
                        first = false;
                        info!("first simulated state published on this robot's limbs");
                    }
                }
                Err(e) if !failing => {
                    failing = true;
                    warn!("publishing simulated state is failing, suppressing repeats: {e}");
                }
                Err(_) => {}
            }
        }
        if ended_by_engine {
            report_removal(&goal).await;
        } else if let Err(e) = goal.cancel_goal(LEAVE_TIMEOUT).await {
            // Leaving on the way out frees the robot's seat at once, rather
            // than when its lease lapses.
            warn!("this robot could not tell the simulation it is leaving: {e}");
        }
        let _ = left.send(());
    }))
}

/// Reports the engine's own account of why this robot left the scene.
async fn report_removal(goal: &attach::ActionHandle) {
    let outcome = match goal.get_result(LEAVE_TIMEOUT).await {
        Ok(result) => result.outcome,
        Err(e) => {
            warn!("the simulation did not say why this robot left: {e}");
            return;
        }
    };
    match outcome {
        attach::ResultOutcome::Completed(data) | attach::ResultOutcome::Cancelled(data)
            if data.success =>
        {
            info!(
                "the simulation took this robot out of the scene: {}",
                data.message
            );
        }
        attach::ResultOutcome::Completed(data) | attach::ResultOutcome::Cancelled(data) => {
            warn!(
                "the simulation took this robot out of the scene: {}",
                data.message
            );
        }
        attach::ResultOutcome::Abandoned => warn!("the simulation abandoned this robot's seat"),
        attach::ResultOutcome::Expired => warn!("this robot's seat expired before it was read"),
    }
}

async fn publish(
    publisher: &peppylib::TopicPublisher,
    message: Result<peppylib::Payload>,
) -> std::result::Result<(), String> {
    let payload = message.map_err(|e| e.to_string())?;
    publisher.publish(payload).await.map_err(|e| e.to_string())
}

/// One consumer per arm slot, holding that arm's latest setpoint.
async fn spawn_arm_consumers(
    runner: &Arc<NodeRunner>,
    seat: &Seat,
    token: &CancellationToken,
) -> Result<()> {
    macro_rules! arm_consumer {
        ($slot:ident) => {{
            let mut sub = $slot::joint_setpoints::subscribe(runner).await?;
            let (seat, token) = (seat.clone(), token.clone());
            tokio::spawn(async move {
                let mut unusable = false;
                loop {
                    let received = tokio::select! {
                        _ = token.cancelled() => return,
                        received = sub.next() => received,
                    };
                    let message = match received {
                        Ok(Some((_, message))) => message,
                        Ok(None) => {
                            warn!(
                                "{} joint_setpoints ended, so nothing drives this limb",
                                stringify!($slot)
                            );
                            return;
                        }
                        Err(e) => {
                            error!("{} joint_setpoints receive: {e}", stringify!($slot));
                            tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                            continue;
                        }
                    };
                    match arm_command(message.positions, message.velocities) {
                        Some(command) => {
                            unusable = false;
                            seat.set_arm(stringify!($slot), command);
                        }
                        None if !unusable => {
                            unusable = true;
                            warn!(
                                "dropping unusable {} setpoints, suppressing repeats",
                                stringify!($slot)
                            );
                        }
                        None => {}
                    }
                }
            });
        }};
    }
    arm_consumer!(left_arm_link);
    arm_consumer!(right_arm_link);
    Ok(())
}

/// One consumer per gripper slot.
async fn spawn_gripper_consumers(
    runner: &Arc<NodeRunner>,
    seat: &Seat,
    token: &CancellationToken,
) -> Result<()> {
    macro_rules! gripper_consumer {
        ($slot:ident) => {{
            let mut sub = $slot::gripper_setpoints::subscribe(runner).await?;
            let (seat, token) = (seat.clone(), token.clone());
            tokio::spawn(async move {
                let mut unusable = false;
                loop {
                    let received = tokio::select! {
                        _ = token.cancelled() => return,
                        received = sub.next() => received,
                    };
                    let message = match received {
                        Ok(Some((_, message))) => message,
                        Ok(None) => {
                            warn!(
                                "{} gripper_setpoints ended, so nothing drives this limb",
                                stringify!($slot)
                            );
                            return;
                        }
                        Err(e) => {
                            error!("{} gripper_setpoints receive: {e}", stringify!($slot));
                            tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                            continue;
                        }
                    };
                    match gripper_command(message.opening, message.max_effort) {
                        Some(command) => {
                            unusable = false;
                            seat.set_gripper(stringify!($slot), command);
                        }
                        None if !unusable => {
                            unusable = true;
                            warn!(
                                "dropping unusable {} setpoints, suppressing repeats",
                                stringify!($slot)
                            );
                        }
                        None => {}
                    }
                }
            });
        }};
    }
    gripper_consumer!(left_gripper_link);
    gripper_consumer!(right_gripper_link);
    Ok(())
}
