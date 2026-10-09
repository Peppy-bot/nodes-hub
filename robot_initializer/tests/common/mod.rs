//! What this node's tests boot a robot with. Each test binary compiles its
//! own copy, so a builder that only one of them calls is unused in the
//! others.
#![allow(dead_code)]

use std::pin::Pin;
use std::sync::{Arc, LazyLock, Once};
use std::time::Duration;

use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::simulation::attach;
use peppygen::parameters::placement::Placement;
use peppygen::{NodeRunner, Parameters};
use robot_initializer::{Moment, StandClock};
use tokio::sync::{mpsc, watch};

/// Joints of one OpenArm arm, as the engine reports them.
const ARM_DOF: u32 = 7;

/// How long a test waits for the node's `setup` to return.
const SETUP_BUDGET: Duration = Duration::from_secs(30);

/// How often a test checks whether the node's `setup` has returned.
const SETUP_POLL: Duration = Duration::from_millis(50);

/// How long a test waits for the node to log a line it is about to log.
const LOGGING: Duration = Duration::from_secs(10);

/// The four limbs of an OpenArm, each with a driver answering for it.
pub const LIMBS: usize = 4;

/// A robot of `model`, standing wherever a simulation parks it.
pub fn parameters(model: &str) -> Parameters {
    Parameters {
        model: model.into(),
        placement: Placement {
            auto: true,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
        },
    }
}

/// The same robot, standing at the placement it is given.
pub fn standing_at(model: &str, x: f64, y: f64, z: f64, yaw: f64) -> Parameters {
    Parameters {
        placement: Placement {
            auto: false,
            x,
            y,
            z,
            yaw,
        },
        ..parameters(model)
    }
}

/// A robot that joins a simulation: the `simulation` slot is bound and the
/// robot has no drivers of its own, so the node attaches before it does
/// anything else and the simulation answers for its readiness.
pub fn joining_a_simulation(model: &str) -> Config {
    Config {
        parameters: Some(parameters(model)),
        ..Config::default()
    }
}

/// An OpenArm with no simulation bound: the `simulation` slot is vacant, so
/// the node joins no scene and serves readiness alone, over the drivers of
/// its four limbs.
pub fn on_its_own_hardware() -> Config {
    Config {
        parameters: Some(parameters("openarm_v2")),
        simulation_vacant: true,
        limbs_instances: LIMBS,
        ..Config::default()
    }
}

/// An SO-101 with no simulation bound: its single `so101_follower` answers
/// for the whole robot.
pub fn an_so101_on_its_own_hardware() -> Config {
    Config {
        parameters: Some(parameters("so101")),
        limbs_instances: 1,
        ..on_its_own_hardware()
    }
}

/// The engine's answer to a robot it stood: the model's limbs, listed right
/// before left, which is not the order of this robot's own slots.
pub fn stood() -> attach::GoalResponseData {
    attach::GoalResponseData {
        arm_names: vec!["right".into(), "left".into()],
        arm_joints: vec![ARM_DOF, ARM_DOF],
        gripper_names: vec!["right".into(), "left".into()],
    }
}

/// The engine's mock, which a robot joining a simulation always has bound.
pub fn simulation_of(
    mocks: &mut peppygen::fixtures::harness::Mocks,
) -> peppygen::mock::deps::simulation::Mock {
    mocks
        .deps
        .simulation
        .take()
        .expect("a robot joining a simulation has one bound")
}

/// Waits for the node's `setup` to return, failing the test after
/// `SETUP_BUDGET`.
pub async fn await_setup_return(harness: &Harness) {
    let deadline = tokio::time::Instant::now() + SETUP_BUDGET;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "setup must return within {SETUP_BUDGET:?}"
        );
        tokio::time::sleep(SETUP_POLL).await;
    }
}

/// Tears the harness down once the node's `setup` has returned, so a setup
/// error reaches the caller. Teardown aborts a setup still running after the
/// shutdown grace and reports it as a clean stop.
pub async fn shutdown_once_setup_returns(harness: Harness) -> peppygen::Result<()> {
    await_setup_return(&harness).await;
    harness.shutdown().await
}

/// A clock that the test drives: the robot's wait for standing hears the
/// moments the test sends, and no other, so no test reaches the stand
/// budget unless it spends it.
pub struct ScriptedClock {
    moments: mpsc::UnboundedReceiver<Moment>,
    elapsed: Duration,
}

impl StandClock for ScriptedClock {
    fn elapsed(&self) -> Duration {
        self.elapsed
    }

    async fn next(&mut self) -> Moment {
        let Some(moment) = self.moments.recv().await else {
            return std::future::pending().await;
        };
        if let Moment::Progress(elapsed) = moment {
            self.elapsed = elapsed;
        }
        moment
    }
}

/// The test's end of a [`ScriptedClock`].
pub struct Clock(mpsc::UnboundedSender<Moment>);

impl Clock {
    /// Tells the robot that it does not stand yet, `elapsed` after its goal
    /// was sent.
    pub fn progress(&self, elapsed: Duration) {
        self.send(Moment::Progress(elapsed));
    }

    /// Spends the robot's stand budget.
    pub fn spend(&self) {
        self.send(Moment::Spent);
    }

    fn send(&self, moment: Moment) {
        self.0
            .send(moment)
            .expect("the robot's wait for standing hears its clock");
    }
}

/// A clock for the test to drive, and the end of it that the node reads.
pub fn scripted_clock() -> (Clock, ScriptedClock) {
    let (moments, heard) = mpsc::unbounded_channel();
    (
        Clock(moments),
        ScriptedClock {
            moments: heard,
            elapsed: Duration::ZERO,
        },
    )
}

/// What the harness runs as the node's setup.
pub type Setup = Pin<Box<dyn Future<Output = peppygen::Result<()>> + Send>>;

/// The node's own setup, its wait for standing on `clock`.
pub fn setup_on(clock: ScriptedClock) -> impl FnOnce(Parameters, Arc<NodeRunner>) -> Setup {
    move |params, runner| {
        Box::pin(robot_initializer::setup_with_clock(
            params,
            runner,
            move |_budget| clock,
        ))
    }
}

/// Every line logged in this test binary, as an operator reads it.
static LOGGED: LazyLock<watch::Sender<Vec<String>>> =
    LazyLock::new(|| watch::Sender::new(Vec::new()));

/// One logged line on its way to [`LOGGED`]: the formatter writes an event
/// whole, then drops the line.
#[derive(Default)]
struct Line(Vec<u8>);

impl std::io::Write for Line {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for Line {
    fn drop(&mut self) {
        let line = String::from_utf8_lossy(&self.0).trim_end().to_owned();
        LOGGED.send_modify(|lines| lines.push(line));
    }
}

/// Captures every line logged in this test binary from now on, for
/// [`logged`]. Each harness runs its node under a name of its own, so a test
/// reads its own robot's lines by that name.
pub fn capture_logs() {
    static CAPTURING: Once = Once::new();
    CAPTURING.call_once(|| {
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(Line::default)
            .init();
    });
}

/// Waits until a line that holds `text` is logged, failing the test after
/// [`LOGGING`].
pub async fn logged(text: &str) {
    let mut lines = LOGGED.subscribe();
    tokio::time::timeout(
        LOGGING,
        lines.wait_for(|lines| lines.iter().any(|line| line.contains(text))),
    )
    .await
    .unwrap_or_else(|_| panic!("the node logs {text:?} within {LOGGING:?}"))
    .expect("the log stays open for as long as the test binary runs");
}
