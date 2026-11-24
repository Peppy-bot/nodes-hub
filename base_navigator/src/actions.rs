//! base_navigation's four actions: each goal is checked in the contract's
//! order against what the navigator knows when it arrives
//! ([`Admission`]), refused with the kind of the earliest check it fails, or
//! accepted and handed to the control loop, where it replaces the goal in
//! flight. A refused goal leaves the goal in flight as it was.

use std::sync::Arc;

use control_core::robot_tree;
use control_core::sync::lock_ignoring_poison;
use peppygen::exposed_actions::base_navigation::{
    back_up, drive_on_heading, navigate_to_pose, spin,
};
use peppygen::{NodeRunner, Result};
use tokio::sync::mpsc;
use tracing::error;

use crate::admission::Admission;
use crate::control::{AcceptedGoal, Event};

/// Expose the action `$action` as `$fn_name`: check each goal with the
/// admission's `$check`, and hand an accepted one to the control loop as
/// `AcceptedGoal::$variant`. Written once here because the four generated
/// action modules carry distinct types with an identical surface.
macro_rules! goal_runner {
    ($fn_name:ident, $action:ident, $check:ident, $variant:ident) => {
        pub async fn $fn_name(
            runner: Arc<NodeRunner>,
            admission: Admission,
            events: mpsc::Sender<Event>,
        ) -> Result<()> {
            let mut handle = $action::ActionHandle::expose(&runner).await?;
            loop {
                // The admitted goal passes out of the decider, which the
                // generated handle calls once per goal.
                let admitted = std::sync::Mutex::new(None);
                let accepted = handle
                    .handle_goal_next_request(|request| {
                        Ok(match admission.$check(&request.data) {
                            Ok(goal) => {
                                *lock_ignoring_poison(&admitted) = Some(goal);
                                $action::GoalDecision::accept()
                            }
                            Err(refusal) => {
                                let (kind, message) = robot_tree::TargetFailure::from(refusal)
                                    .for_member($action::FailureKind::parse);
                                $action::GoalDecision::refuse(kind, message)
                            }
                        })
                    })
                    .await?;
                let Some(ctx) = accepted else { return Ok(()) };
                let (request, admitted) = lock_ignoring_poison(&admitted)
                    .take()
                    .expect("the decider keeps the goal it accepts");
                let goal = AcceptedGoal::$variant {
                    request,
                    ctx: Box::new(ctx),
                    admitted,
                };
                if events.send(Event::Goal(Box::new(goal))).await.is_err() {
                    error!("{}: the control loop is not running", stringify!($action));
                    return Ok(());
                }
            }
        }
    };
}

goal_runner!(run_navigate_to_pose, navigate_to_pose, navigation, Navigate);
goal_runner!(run_spin, spin, spin, Spin);
goal_runner!(
    run_drive_on_heading,
    drive_on_heading,
    drive_on_heading,
    DriveOnHeading
);
goal_runner!(run_back_up, back_up, back_up, BackUp);
