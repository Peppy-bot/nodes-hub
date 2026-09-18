//! Joining a simulation: this robot attaches as the copy it runs as, naming
//! the model to stand and where, and holds that goal for as long as it is in
//! the scene.
//!
//! Its limbs pair to the engine through the backbone, and the copy the engine reads on each pair is
//! the one this goal named, so a setpoint reaches the robot it was meant for
//! and its measured state comes back on the same pair.
//!
//! Joining fails loudly, so a robot the scene refused never serves the
//! readiness it cannot back, and `peppy stack list` reports the instance
//! failed with the engine's reason. The node serves readiness from the
//! moment the engine admits the robot, which is what the launch's health
//! check waits for; the engine says the robot stands on the goal's
//! feedback, and a goal the engine ends before that is a robot the scene
//! could not stand: the node logs the engine's reason and stops, as it does
//! when the goal ends later, and `peppy stack list` reports the instance
//! finished; `peppy stack join` puts the copy back.
//!
//! A node that stops takes its robot out and waits for the engine's result,
//! which arrives once the robot is out of the scene and its name is free:
//! removing a copy returns only then, so a copy joined straight back under
//! the same name finds its name free.

use std::sync::Arc;
use std::time::Duration;

use peppygen::consumed_actions::simulation::attach;
use peppygen::parameters::placement::Placement;
use peppygen::{NodeRunner, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use tracing::{error, info, warn};

use crate::refused;

/// How long the robot waits for the simulation to admit it. The engine
/// answers once it has checked the model and the spot, and stands the robot
/// after, which on a cold cache takes it tens of seconds.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the account of a stay the engine ended may take: the goal is
/// terminal, so the engine answers at once.
const RESULT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long leaving may take in all: the engine acknowledging the cancel,
/// then its result saying the robot is out of the scene, which takes the
/// engine a scene rebuild.
const LEAVE_TIMEOUT: Duration = Duration::from_secs(4);
/// How long the shutdown hook waits for the robot to be out, inside peppy's
/// shutdown grace (5 s by default, `lifecycle.shutdown_grace_secs`).
const LEAVE_REPORT_TIMEOUT: Duration = Duration::from_millis(4500);
// The wait outlasts the leave it waits for, so a robot that does leave is
// not reported as one that did not.
const _: () = assert!(LEAVE_REPORT_TIMEOUT.as_millis() > LEAVE_TIMEOUT.as_millis());

/// Where the launcher asks this robot to stand, as the engine takes it.
/// `auto` leaves the spot to the engine. A spot that is not a number is
/// refused here, before it reaches the wire.
fn spot_of(
    placement: &Placement,
) -> std::result::Result<Option<attach::SimulationRobotAttachActionGoalPlacement>, String> {
    if placement.auto {
        return Ok(None);
    }
    let parts = [placement.x, placement.y, placement.z, placement.yaw];
    if !parts.iter().all(|part| part.is_finite()) {
        let [x, y, z, yaw] = parts;
        return Err(format!(
            "placement x, y, z and yaw must each be a finite number, and this robot's are {x}, {y}, {z} and {yaw}"
        ));
    }
    Ok(Some(attach::SimulationRobotAttachActionGoalPlacement {
        position: [placement.x, placement.y, placement.z],
        yaw: placement.yaw,
    }))
}

/// The name this robot stands under in the scene: the copy the launch put it
/// in, or its own instance id for a robot launched outside one. It is the
/// name the engine reads on every one of this robot's limb pairs, which is
/// how it tells one robot's limbs from another's.
fn robot_name(runner: &NodeRunner) -> String {
    runner
        .copy()
        .unwrap_or_else(|| runner.processor().bound_instance_id())
        .to_string()
}

/// Joins the simulation the launcher bound this robot to, standing the
/// model of the robot's generation. A robot with no simulation bound drives
/// its own hardware and returns without joining.
pub async fn scene(
    generation: &str,
    placement: &Placement,
    runner: &Arc<NodeRunner>,
) -> Result<()> {
    let Some(simulation) = attach::bound_producer(runner).cloned() else {
        info!("no simulation stands this robot, so it joins none");
        return Ok(());
    };
    let model = format!("openarm_{generation}");
    let placement = spot_of(placement).map_err(refused)?;
    let robot = robot_name(runner);

    let goal = attach::ActionHandle::fire_goal(
        runner,
        &simulation,
        ATTACH_TIMEOUT,
        attach::GoalRequest {
            robot: robot.clone(),
            model: model.clone(),
            placement,
        },
        QoSProfile::Reliable,
    )
    .await?;
    if !goal.accepted {
        return Err(refused(format!(
            "the simulation refused this robot: {}",
            goal.reason.unwrap_or_else(|| "no reason given".into())
        )));
    }
    let stood = goal
        .data
        .clone()
        .ok_or_else(|| refused("the simulation stood this robot without naming its limbs"))?;
    info!(
        "'{robot}' joined the simulation as {model}, with arms [{}] and grippers [{}]",
        stood.arm_names.join(", "),
        stood.gripper_names.join(", ")
    );

    hold(runner, goal, robot);
    Ok(())
}

/// Holds the robot's place in the scene for as long as the node runs: the
/// goal ends when the engine takes the robot out, and shutting the node down
/// tells the engine to and waits until the robot is out.
fn hold(runner: &Arc<NodeRunner>, goal: attach::ActionHandle, robot: String) {
    let token = runner.cancellation_token().clone();
    let (left, has_left) = tokio::sync::oneshot::channel();
    tokio::spawn(stay(goal, robot, left, token.clone()));
    runner.on_shutdown(async move {
        token.cancel();
        if tokio::time::timeout(LEAVE_REPORT_TIMEOUT, has_left)
            .await
            .is_err()
        {
            warn!("this robot did not leave the scene within {LEAVE_REPORT_TIMEOUT:?}");
        }
    });
}

/// One robot's stay: it ends when the engine ends the goal (the goal's
/// feedback stream closes with it) or the engine is gone, or when this
/// node is shutting down and leaves. Either way the node stops, because a
/// robot that is not in the scene has no readiness to serve. A goal the
/// engine ends before it ever said the robot stands is a robot the scene
/// could not stand, reported as such.
async fn stay(
    mut goal: attach::ActionHandle,
    robot: String,
    left: tokio::sync::oneshot::Sender<()>,
    token: CancellationToken,
) {
    let mut stood = false;
    let ended = tokio::select! {
        _ = token.cancelled() => false,
        () = stay_ends(&mut goal, &robot, &mut stood) => true,
    };
    if ended {
        report(&robot, stood, account(&goal).await);
        token.cancel();
    } else {
        leave(&goal, &robot, stood).await;
    }
    let _ = left.send(());
}

/// Resolves once the engine's side of the stay is over: the engine sends
/// one feedback message when the robot stands, which sets `stood`, and the
/// stream closes when the goal ends or the engine is gone. `stood` is what
/// the node reports the stay by, whichever way the stay ends.
async fn stay_ends(goal: &mut attach::ActionHandle, robot: &str, stood: &mut bool) {
    loop {
        match goal.on_next_feedback_message().await {
            Ok(feedback) if feedback.standing => {
                info!("'{robot}' stands in the scene");
                *stood = true;
            }
            Ok(_) => {}
            Err(peppygen::Error::ActionFeedbackChannelClosed) => return,
            Err(peppygen::Error::ActionFeedbackProducerGone { .. }) => {
                warn!("the simulation standing '{robot}' is gone");
                return;
            }
            Err(e) => {
                warn!("the stay of '{robot}' cannot be followed: {e}");
                return;
            }
        }
    }
}

/// The engine's account of a stay that has ended.
async fn account(
    goal: &attach::ActionHandle,
) -> std::result::Result<attach::ResultOutcome, String> {
    goal.get_result(RESULT_TIMEOUT)
        .await
        .map(|result| result.outcome)
        .map_err(|e| e.to_string())
}

/// Takes the robot out of the scene on the way out: cancelling the goal asks
/// the engine to, and the engine's result says the robot is out and its name
/// is free. Both share [`LEAVE_TIMEOUT`].
async fn leave(goal: &attach::ActionHandle, robot: &str, stood: bool) {
    let deadline = tokio::time::Instant::now() + LEAVE_TIMEOUT;
    let remaining = || deadline.saturating_duration_since(tokio::time::Instant::now());
    if let Err(e) = goal.cancel_goal(remaining()).await {
        warn!("'{robot}' could not tell the simulation it is leaving: {e}");
        return;
    }
    match goal.get_result(remaining()).await {
        Ok(result) => report(robot, stood, Ok(result.outcome)),
        Err(e) => warn!("the simulation did not say '{robot}' is out of the scene: {e}"),
    }
}

/// How a robot's stay ended, with what the engine said of it.
#[derive(Debug, PartialEq, Eq)]
enum Ending {
    /// The engine never stood the robot.
    NeverStood(String),
    /// The engine took the robot out of the scene, as it was asked to.
    TakenOut(String),
    /// The engine took the robot out for a reason of its own.
    Dropped(String),
    /// The engine did not say what became of the robot.
    Unsaid(String),
}

/// What the engine's account says became of a robot that `stood` or never
/// did.
fn ending(stood: bool, outcome: std::result::Result<attach::ResultOutcome, String>) -> Ending {
    let (asked_for, said) = match outcome {
        Err(e) => return Ending::Unsaid(e),
        Ok(attach::ResultOutcome::Completed(data) | attach::ResultOutcome::Cancelled(data)) => {
            (data.success, data.message)
        }
        Ok(attach::ResultOutcome::Abandoned) => (false, "the simulation abandoned it".to_owned()),
        Ok(attach::ResultOutcome::Expired) => {
            (false, "it left before the reason was read".to_owned())
        }
    };
    match (stood, asked_for) {
        (false, _) => Ending::NeverStood(said),
        (true, true) => Ending::TakenOut(said),
        (true, false) => Ending::Dropped(said),
    }
}

/// Reports how this robot's stay ended: a robot that stood was taken out of
/// the scene, and one that never stood could not be stood.
fn report(robot: &str, stood: bool, outcome: std::result::Result<attach::ResultOutcome, String>) {
    match ending(stood, outcome) {
        Ending::NeverStood(why) => error!("the simulation did not stand '{robot}': {why}"),
        Ending::TakenOut(why) => {
            info!("the simulation took '{robot}' out of the scene: {why}");
        }
        Ending::Dropped(why) => warn!("the simulation took '{robot}' out of the scene: {why}"),
        Ending::Unsaid(why) => warn!("the simulation did not say why '{robot}' left: {why}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn said(success: bool, message: &str) -> attach::ResultResponseData {
        attach::ResultResponseData {
            success,
            message: message.to_owned(),
        }
    }

    #[test]
    fn a_stay_is_reported_by_whether_the_robot_ever_stood() {
        // A goal the engine ends before the robot stands: the scene could
        // not stand it, whatever the engine's own verdict on the goal.
        assert_eq!(
            ending(
                false,
                Ok(attach::ResultOutcome::Completed(said(false, "no files")))
            ),
            Ending::NeverStood("no files".to_owned())
        );
        assert_eq!(
            ending(
                false,
                Ok(attach::ResultOutcome::Completed(said(true, "cleared")))
            ),
            Ending::NeverStood("cleared".to_owned())
        );
        assert_eq!(
            ending(false, Ok(attach::ResultOutcome::Abandoned)),
            Ending::NeverStood("the simulation abandoned it".to_owned())
        );

        // A robot that stood: the engine took it out as asked, or for a
        // reason of its own.
        assert_eq!(
            ending(
                true,
                Ok(attach::ResultOutcome::Cancelled(said(true, "left")))
            ),
            Ending::TakenOut("left".to_owned())
        );
        assert_eq!(
            ending(
                true,
                Ok(attach::ResultOutcome::Completed(said(false, "lapsed")))
            ),
            Ending::Dropped("lapsed".to_owned())
        );
        assert_eq!(
            ending(true, Ok(attach::ResultOutcome::Expired)),
            Ending::Dropped("it left before the reason was read".to_owned())
        );
        assert_eq!(
            ending(true, Err("timed out".to_owned())),
            Ending::Unsaid("timed out".to_owned())
        );
    }
}
