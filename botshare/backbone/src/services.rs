//! The services the base answers: what the base is (describe_base), from the
//! parameters and the robot's model, and the robot stop of robot_stop, which
//! the control loop carries out.

use std::sync::Arc;
use std::time::Duration;

use control_core::serving;
use control_core::sync::wait_for;
use peppygen::NodeRunner;
use peppygen::exposed_services::base_state::describe_base;
use peppygen::exposed_services::robot_stop::stop;
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::config::{AxisSpec, BaseConfig};
use crate::control::{BASE_NAME, Event};
use crate::transforms::ROBOT_FRAME;

/// What describe_base answers: the base's name, its measured odometry, the
/// limits of x and y (y still) and of the turn, each with its rest bound, and
/// its outline in the robot frame, which holds at every time.
fn base_description(config: &BaseConfig) -> describe_base::Response {
    let (x, y, yaw) = (config.axes.x, AxisSpec::STILL, config.axes.yaw);
    describe_base::Response::new(
        BASE_NAME.to_string(),
        "measured".to_string(),
        describe_base::ResponseLinearLimits {
            min_velocity: [x.min_velocity, y.min_velocity],
            max_velocity: [x.max_velocity, y.max_velocity],
            max_acceleration: [x.max_acceleration, y.max_acceleration],
            max_deceleration: [x.max_deceleration, y.max_deceleration],
            deadband: [x.deadband, y.deadband],
            rest_velocity: [x.rest, y.rest],
        },
        describe_base::ResponseAngularLimits {
            min_rate: yaw.min_velocity,
            max_rate: yaw.max_velocity,
            max_acceleration: yaw.max_acceleration,
            max_deceleration: yaw.max_deceleration,
            deadband: yaw.deadband,
            rest_rate: yaw.rest,
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
/// or why it could not, the loop taken and answered within `timeout`. The
/// stop waits within `timeout` for its place in the loop's one queue, which
/// the wheels' measurements and the leader's setpoints share:
/// - the loop is not running, or it dropped the stop;
/// - it did not take and answer the stop within `timeout`.
fn stop_moves(
    events: &mpsc::Sender<Event>,
    reason: String,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let (reply, answer) = oneshot::channel();
    wait_for(async {
        let asked = async {
            events
                .send(Event::Stop { reason, reply })
                .await
                .map_err(|_| "the control loop is not running".to_string())?;
            answer
                .await
                .map_err(|_| "the control loop dropped the stop".to_string())
        };
        tokio::time::timeout(timeout, asked)
            .await
            .unwrap_or_else(|_| {
                Err(format!(
                    "the control loop did not answer within {timeout:?}"
                ))
            })
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A control loop that takes no stop within the answer time, its one
    /// queue full: the stop fails naming that time, and the loop never takes
    /// it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_the_loop_does_not_take_in_its_answer_time_fails_naming_it() {
        let (events, mut queued) = mpsc::channel(1);
        let (held, _answer) = oneshot::channel();
        events
            .try_send(Event::Stop {
                reason: "a stop before".to_string(),
                reply: held,
            })
            .expect("a free queue");
        let timeout = Duration::from_millis(10);
        assert_eq!(
            stop_moves(&events, "the operator asked".to_string(), timeout),
            Err(format!(
                "the control loop did not answer within {timeout:?}"
            ))
        );
        let Ok(Event::Stop { reason, .. }) = queued.try_recv() else {
            panic!("the stop before stays queued");
        };
        assert_eq!(reason, "a stop before");
        assert!(
            queued.try_recv().is_err(),
            "the timed-out stop never queued"
        );
    }

    /// A control loop that takes the stop answers it with the parts it
    /// stopped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_the_loop_takes_answers_the_parts_it_stopped() {
        let (events, mut queued) = mpsc::channel(1);
        tokio::spawn(async move {
            if let Some(Event::Stop { reply, .. }) = queued.recv().await {
                let _ = reply.send(vec![BASE_NAME.to_string()]);
            }
        });
        assert_eq!(
            stop_moves(
                &events,
                "the operator asked".to_string(),
                Duration::from_secs(30)
            ),
            Ok(vec![BASE_NAME.to_string()])
        );
    }
}
