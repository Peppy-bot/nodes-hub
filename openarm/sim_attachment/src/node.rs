//! Taking the seat and holding it: the robot attaches, its limbs' setpoints
//! go out as one command at the command rate, and every state that comes
//! back is published on the limb it measures.
//!
//! The node's life is the robot's stay in the scene. It fails to start when
//! the simulation refuses the robot, and it stops when the seat ends (the
//! engine took the robot out, or the goal could no longer be driven), so
//! the runtime restarts it and the robot rejoins rather than standing there
//! unattached.

use std::sync::Arc;
use std::time::Duration;

use peppygen::consumed_actions::engine::attach;
use peppygen::consumed_services::engine::command;
use peppygen::paired_topics::{left_arm, left_gripper, right_arm, right_gripper};
use peppygen::{NodeRunner, Parameters, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use tracing::{error, info, warn};

use crate::limbs::{ARMS, GRIPPERS, Seat, arm_command, gripper_command, model_of};

/// How long the robot waits for the simulation to seat it. The engine
/// rebuilds its world around the joining robot, which on a mesh-heavy model
/// takes a moment.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one command may take. Well under a command period at the rates
/// a robot runs, so a lost reply is retried on the next tick.
const COMMAND_TIMEOUT: Duration = Duration::from_millis(250);
/// How long the robot waits for the engine to acknowledge that it left.
const LEAVE_TIMEOUT: Duration = Duration::from_secs(5);
/// Pause after a receive error before retrying, so a persistently broken
/// subscription cannot hot-spin a consumer or flood the log.
const RECEIVE_ERROR_BACKOFF: Duration = Duration::from_millis(100);

fn refused(message: impl Into<String>) -> peppygen::Error {
    peppygen::Error::Node(std::io::Error::other(message.into()).into())
}

/// The node's entry point: the closure `NodeBuilder::run` takes, named so
/// the test harness can boot the node in-process.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    peppygen::clock::init(&node_runner).await?;
    let token = node_runner.cancellation_token().clone();
    let model = model_of(&params.hardware_version).map_err(refused)?;
    let rate = params.command_rate_hz;
    if !(1..=1000).contains(&rate) {
        return Err(refused(format!(
            "command_rate_hz must be between 1 and 1000, and this robot's is {rate}"
        )));
    }
    let period = Duration::from_secs_f64(1.0 / f64::from(rate));
    let placement =
        (!params.placement.auto).then_some(attach::SimulationRobotAttachActionGoalPlacement {
            position: [params.placement.x, params.placement.y, params.placement.z],
            yaw: params.placement.yaw,
        });

    // The seat: the simulation stands this robot and says which limbs it
    // gave it. Feedback is a state stream, so it is read latest-wins.
    let goal = attach::ActionHandle::fire_goal(
        &node_runner,
        attach::bound_producer(&node_runner),
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

    spawn_arm_consumers(&node_runner, &seat, &token).await?;
    spawn_gripper_consumers(&node_runner, &seat, &token).await?;
    let (left, has_left) = tokio::sync::oneshot::channel();
    let states =
        spawn_state_publishers(&node_runner, seat.clone(), goal, left, token.clone()).await?;
    let commands = tokio::spawn(command_loop(
        node_runner.clone(),
        seat,
        period,
        token.clone(),
    ));

    // Shutting this robot down takes it out of the scene: the wind-down
    // gives its seat back, and the node waits for that, so the body leaves
    // with the robot.
    let shutdown_token = token.clone();
    node_runner.on_shutdown(async move {
        shutdown_token.cancel();
        if tokio::time::timeout(LEAVE_TIMEOUT, has_left).await.is_err() {
            warn!("this robot did not leave the scene within {LEAVE_TIMEOUT:?}");
        }
    });

    // The robot's stay is over when either half of the seat ends: cancel the
    // node so the runtime restarts it and it rejoins.
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
        let answered = command::poll(
            &runner,
            command::bound_producer(&runner),
            COMMAND_TIMEOUT,
            seat.request(),
        )
        .await;
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

/// Publishes every state the simulation sends back on the limb it measures,
/// and ends when the robot's stay does.
async fn spawn_state_publishers(
    runner: &Arc<NodeRunner>,
    seat: Seat,
    mut goal: attach::ActionHandle,
    left: tokio::sync::oneshot::Sender<()>,
    token: CancellationToken,
) -> Result<tokio::task::JoinHandle<()>> {
    let arms = vec![
        (
            ARMS[0].0,
            left_arm::joint_states::declare_publisher(runner).await?,
        ),
        (
            ARMS[1].0,
            right_arm::joint_states::declare_publisher(runner).await?,
        ),
    ];
    let grippers = vec![
        (
            GRIPPERS[0].0,
            left_gripper::gripper_states::declare_publisher(runner).await?,
        ),
        (
            GRIPPERS[1].0,
            right_gripper::gripper_states::declare_publisher(runner).await?,
        ),
    ];
    let runner = runner.clone();
    Ok(tokio::spawn(async move {
        let mut failing = false;
        let mut first = true;
        loop {
            let feedback = tokio::select! {
                _ = token.cancelled() => break,
                feedback = goal.on_next_feedback_message() => feedback,
            };
            let feedback = match feedback {
                Ok(feedback) => feedback,
                // The stay is over: the engine completed the goal, or it is
                // gone. Either way this robot is no longer in the scene.
                Err(e) => {
                    info!("this robot's seat in the simulation ended: {e}");
                    break;
                }
            };
            let mut published = Ok(());
            for (slot, publisher) in &arms {
                let Some(state) = seat.arm_state(slot, &feedback) else {
                    continue;
                };
                let message = left_arm::joint_states::build_message(
                    feedback.timestamp,
                    state.positions.clone(),
                    state.velocities.clone(),
                    Vec::new(),
                );
                published = published.and(publish(publisher, message).await);
            }
            for (slot, publisher) in &grippers {
                let Some(state) = seat.gripper_state(slot, &feedback) else {
                    continue;
                };
                let message = left_gripper::gripper_states::build_message(
                    feedback.timestamp,
                    state.opening,
                    state.effort,
                    0.0,
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
        // Leaving on the way out frees the robot's seat at once, rather
        // than when its lease lapses.
        if let Err(e) = goal.cancel_goal(LEAVE_TIMEOUT).await {
            warn!("this robot could not tell the simulation it is leaving: {e}");
        }
        let _ = left.send(());
        drop(runner);
    }))
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
                loop {
                    let received = tokio::select! {
                        _ = token.cancelled() => return,
                        received = sub.next() => received,
                    };
                    let message = match received {
                        Ok(Some((_, message))) => message,
                        Ok(None) => return,
                        Err(e) => {
                            error!("{} joint_setpoints receive: {e}", stringify!($slot));
                            tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                            continue;
                        }
                    };
                    match arm_command(message.positions, message.velocities) {
                        Some(command) => seat.set_arm(stringify!($slot), command),
                        None => warn!("dropping unusable {} setpoints", stringify!($slot)),
                    }
                }
            });
        }};
    }
    arm_consumer!(left_arm);
    arm_consumer!(right_arm);
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
                loop {
                    let received = tokio::select! {
                        _ = token.cancelled() => return,
                        received = sub.next() => received,
                    };
                    let message = match received {
                        Ok(Some((_, message))) => message,
                        Ok(None) => return,
                        Err(e) => {
                            error!("{} gripper_setpoints receive: {e}", stringify!($slot));
                            tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                            continue;
                        }
                    };
                    match gripper_command(message.opening, message.max_effort) {
                        Some(command) => seat.set_gripper(stringify!($slot), command),
                        None => warn!("dropping unusable {} setpoints", stringify!($slot)),
                    }
                }
            });
        }};
    }
    gripper_consumer!(left_gripper);
    gripper_consumer!(right_gripper);
    Ok(())
}
