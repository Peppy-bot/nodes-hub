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
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use peppygen::NodeRunner;
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
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            arms: watch::Sender::new(None),
            body: watch::Sender::new(None),
        }
    }
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
