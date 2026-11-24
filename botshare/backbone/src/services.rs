//! The services the base answers: what the base is (describe_base), from the
//! parameters and the robot's model, and the robot stop of robot_stop, which
//! the control loop carries out.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking::ask_loop;
use control_core::serving;
use control_core::sync::wait_for;
use peppygen::NodeRunner;
use peppygen::exposed_services::base_state::describe_base;
use peppygen::exposed_services::robot_stop::stop;
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::config::{BaseConfig, MotionLimits};
use crate::control::{BASE_NAME, Event};
use crate::transforms::ROBOT_FRAME;

/// What describe_base answers: the base's name, its measured odometry, its
/// limits ([`BaseConfig::motion_limits`]), and its outline in the robot
/// frame, which holds at every time.
fn base_description(config: &BaseConfig) -> describe_base::Response {
    let MotionLimits { linear, angular } = config.motion_limits();
    describe_base::Response::new(
        BASE_NAME.to_string(),
        "measured".to_string(),
        describe_base::ResponseLinearLimits {
            min_velocity: linear.min_velocity,
            max_velocity: linear.max_velocity,
            max_acceleration: linear.max_acceleration,
            max_deceleration: linear.max_deceleration,
            deadband: linear.deadband,
            rest_velocity: linear.rest_velocity,
        },
        describe_base::ResponseAngularLimits {
            min_rate: angular.min_rate,
            max_rate: angular.max_rate,
            max_acceleration: angular.max_acceleration,
            max_deceleration: angular.max_deceleration,
            deadband: angular.deadband,
            rest_rate: angular.rest_rate,
        },
        describe_base::ResponseFootprint {
            frame: ROBOT_FRAME.to_owned(),
            timestamp: None,
            points: config
                .footprint
                .points()
                .iter()
                .map(|&[x, y]| describe_base::ResponseFootprintPointsItem {
                    position: [x, y, 0.0],
                })
                .collect(),
        },
    )
}

/// Serve describe_base until the node cancels.
pub async fn serve_describe_base(runner: Arc<NodeRunner>, config: Arc<BaseConfig>) {
    let token = runner.cancellation_token().clone();
    serving::serve("describe_base", token.cancelled(), || {
        describe_base::handle_requests(&runner, |_request| Ok(base_description(&config)))
    })
    .await;
}

/// Hand the robot stop to the control loop and return the parts it stopped,
/// or why it could not, the loop taken and answered within `timeout`
/// ([`ask_loop`]): the stop waits within `timeout` for its place in the loop's
/// one queue, which the wheels' measurements and the leader's setpoints share.
fn stop_moves(
    events: &mpsc::Sender<Event>,
    reason: String,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let (reply, answer) = oneshot::channel();
    wait_for(ask_loop(
        "control loop",
        events,
        Event::Stop { reason, reply },
        answer,
        timeout,
    ))
    .inspect_err(|reason| error!("stop: {reason}"))
}

/// Serve robot_stop's `stop` until `token` cancels: [`stop_moves`] within
/// `answer_timeout`, answered with the parts it stopped, or an error reply
/// with `attempt_failed`.
pub async fn run_robot_stop(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    events: mpsc::Sender<Event>,
    answer_timeout: Duration,
) {
    serving::serve("robot_stop", token.cancelled(), || {
        stop::handle_requests(&runner, |request| {
            stop_moves(&events, request.data.reason, answer_timeout)
                .map(stop::Response::new)
                .map_err(|reason| stop::Failure::new(stop::FailureKind::AttemptFailed, reason))
        })
    })
    .await;
}
