//! The wiring: the links of the node, the topics it follows and the services it asks,
//! and the parse of each message into a value of the model, once, at the edge. The
//! wiring is the only code that knows peppy.

pub mod arms;
pub mod camera;
pub mod clock;
pub mod planner_map;
pub mod robot;
pub mod scene;
pub mod tree;

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::latched_log::LatchedLog;
use frame_tree::FrameName;
use peppygen::NodeRunner;
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use tokio::sync::watch;

use crate::model::arms::ArmsDescription;
use crate::model::robot::RobotBody;

/// How long the viewer waits for a service to answer.
pub const SERVICE_TIMEOUT: Duration = Duration::from_secs(5);

/// What one follower learns that another draws with.
pub struct Shared {
    /// The arms as the motion planner describes them: the joints' names for the plots
    /// and the grasp links that carry objects. None until the planner answers.
    pub arms: watch::Sender<Option<ArmsDescription>>,
    /// The robot's body as its description gives it, whose links the arms follower
    /// places. None until the description is read.
    pub body: watch::Sender<Option<Arc<RobotBody>>>,
    /// The robot frame as sensor_mounts' describe_sensors gives it, which the viewer draws
    /// every value in, or the ask of it that runs.
    robot_frame: Mutex<RobotFrame>,
    /// The failed asks of the robot frame.
    robot_frame_asks: LatchedLog,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            arms: watch::Sender::new(None),
            body: watch::Sender::new(None),
            robot_frame: Mutex::new(RobotFrame::Unasked),
            robot_frame_asks: LatchedLog::warn("asking sensor_mounts for the robot frame"),
        }
    }
}

/// What the viewer knows of the robot frame.
enum RobotFrame {
    /// No ask runs: none ran yet, or the last one failed.
    Unasked,
    /// One ask of sensor_mounts runs in the background.
    Asking,
    /// The robot frame sensor_mounts gave.
    Known(FrameName),
}

/// Why a value that the viewer draws in the robot frame is dropped before sensor_mounts
/// gives the robot frame.
const ROBOT_FRAME_UNKNOWN: &str =
    "the robot frame is unknown until sensor_mounts answers describe_sensors";

impl Shared {
    /// The robot frame, once sensor_mounts' describe_sensors gave it. Until then a call is
    /// refused at once, so its follower drops the value that needs the frame and goes on,
    /// and a call while no ask runs starts one in the background, for every follower: the
    /// first call, and the first after a failed ask.
    pub fn robot_frame(self: &Arc<Self>, runner: &Arc<NodeRunner>) -> Result<FrameName, String> {
        let mut known = lock(&self.robot_frame);
        match &*known {
            RobotFrame::Known(robot_frame) => return Ok(robot_frame.clone()),
            RobotFrame::Asking => return Err(ROBOT_FRAME_UNKNOWN.to_owned()),
            RobotFrame::Unasked => *known = RobotFrame::Asking,
        }
        tokio::spawn(learn_robot_frame(Arc::clone(self), Arc::clone(runner)));
        Err(ROBOT_FRAME_UNKNOWN.to_owned())
    }

    /// Holds `frame`, the frame the report `report` gives its values in, to the robot frame
    /// the viewer draws them in, refused while the robot frame is unknown. With the
    /// sensor_mounts link vacant, the viewer reads no robot frame and takes the report's
    /// frame, which its contract gives as the robot frame.
    pub fn hold_to_robot_frame(
        self: &Arc<Self>,
        runner: &Arc<NodeRunner>,
        report: &'static str,
        frame: &str,
    ) -> Result<(), String> {
        if describe_sensors::bound_producer(runner).is_none() {
            return Ok(());
        }
        let robot_frame = self.robot_frame(runner)?;
        check_report_frame(report, frame, &robot_frame).map_err(|e| e.to_string())
    }
}

fn lock(robot_frame: &Mutex<RobotFrame>) -> MutexGuard<'_, RobotFrame> {
    robot_frame.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One ask of the robot frame, whose answer `shared` keeps: the frame, or, after a failed
/// ask, that no ask runs.
async fn learn_robot_frame(shared: Arc<Shared>, runner: Arc<NodeRunner>) {
    let learned = match ask_robot_frame(&runner).await {
        Ok(robot_frame) => {
            shared.robot_frame_asks.success();
            RobotFrame::Known(robot_frame)
        }
        Err(reason) => {
            shared.robot_frame_asks.failure(reason);
            RobotFrame::Unasked
        }
    };
    *lock(&shared.robot_frame) = learned;
}

/// A report in another frame than the robot frame, which the viewer drops.
#[derive(Debug, thiserror::Error, PartialEq)]
#[error(
    "{report} names the frame {frame:?} where sensor_mounts gives the robot frame \
     {robot_frame:?}: link the viewer to the instances of one robot copy"
)]
struct OtherFrame {
    report: &'static str,
    frame: String,
    robot_frame: String,
}

/// `frame`, the frame the report `report` gives its values in, held to `robot_frame`.
fn check_report_frame(
    report: &'static str,
    frame: &str,
    robot_frame: &FrameName,
) -> Result<(), OtherFrame> {
    if frame == robot_frame.as_str() {
        return Ok(());
    }
    Err(OtherFrame {
        report,
        frame: frame.to_owned(),
        robot_frame: robot_frame.as_str().to_owned(),
    })
}

/// One ask of the robot frame.
async fn ask_robot_frame(runner: &NodeRunner) -> Result<FrameName, String> {
    let mounts =
        describe_sensors::bound_producer(runner).ok_or("the sensor_mounts link is vacant")?;
    let sensors = sendable(describe_sensors::poll(runner, mounts, SERVICE_TIMEOUT))
        .await
        .map_err(|e| format!("describe_sensors: {e}"))?;
    FrameName::parse(&sensors.data.robot_frame)
        .map_err(|e| format!("describe_sensors gives a robot_frame: {e}"))
}

/// Now, on the node's clock: the simulation's under `--clock simulation`. An error
/// until the clock is ready, such as before the simulation's first tick.
pub fn node_now() -> Result<SystemTime, String> {
    peppygen::clock::now_ns()
        .map(|nanoseconds| UNIX_EPOCH + Duration::from_nanos(nanoseconds))
        .map_err(|e| format!("the node's clock: {e}"))
}

/// A future that ends when the node stops, holding no borrow of `runner`.
pub fn stopped(runner: &NodeRunner) -> impl Future<Output = ()> + Send + 'static {
    let token = runner.cancellation_token().clone();
    async move { token.cancelled().await }
}

/// `future`, boxed as a future that can move between threads. The futures of peppy's
/// `subscribe` and `poll` hold their `&str` and `&ProducerRef` arguments, and rustc
/// cannot prove such a future `Send` for every lifetime inside a spawned task; boxed
/// where it is made, its lifetimes are those of that place.
pub fn sendable<'a, T>(
    future: impl Future<Output = T> + Send + 'a,
) -> Pin<Box<dyn Future<Output = T> + Send + 'a>> {
    Box::pin(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_in_the_robot_frame_is_held_and_one_in_another_names_both_frames() {
        let robot_frame = FrameName::parse("base_link").expect("a frame name");
        assert_eq!(
            check_report_frame("limb_states", "base_link", &robot_frame),
            Ok(())
        );
        assert_eq!(
            check_report_frame("limb_states", "odom", &robot_frame).map_err(|e| e.to_string()),
            Err(
                "limb_states names the frame \"odom\" where sensor_mounts gives the robot \
                 frame \"base_link\": link the viewer to the instances of one robot copy"
                    .to_string()
            )
        );
    }
}
