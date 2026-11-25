//! What the base is, as base_state's describe_base answers it: its name, the
//! odometry source a goal needs measured, the limits of its forward motion
//! and of its turn, and its footprint, padded and inflated as the grid reads
//! it. The navigator reads it each time drive_states reports a new
//! odometry_start_timestamp, the first drive_states included
//! ([`run_reader`]), and a goal waits for the read of the newest start up to
//! the source timeout ([`BaseReads::newest`]). Each read logs both of the
//! stop's horizons and the one the stop projects, and builds the grid of the
//! floor for the footprint it gives.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use control_core::asking;
use control_core::footprint::{Footprint, FootprintError};
use control_core::limiter::{AxisLimits, AxisLimitsParams, LimiterError};
use control_core::node_clock::NodeClock;
use control_core::non_negative_finite::NonNegativeFinite;
use control_core::sync::lock_ignoring_poison;
use control_core::time::seconds;
use navigation_core::inflation::{Inflation, InflationBelowInscribed};
use navigation_core::{PaddedFootprint, PaddingError};
use peppygen::NodeRunner;
use peppygen::consumed_services::base_state::describe_base;
use tokio::sync::watch;
use tracing::{error, info};

use crate::clock;
use crate::config::{NavigatorConfig, StopSettings};
use crate::failure::{Failure, FailureKind};
use crate::follow::ROBOT_FRAME;
use crate::grid::{GridSlot, LiveGrid};

/// How long one ask of a description waits for its answer, describe_base's and a depth
/// camera's geometry and stream info; an ask that runs out is asked again.
pub(crate) const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);

/// Where the base's odometry comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdometrySource {
    Measured,
    Commanded,
}

/// The base as describe_base gives it, its footprint padded by the launch's
/// padding and inflated by its inflation.
#[derive(Clone, Debug)]
pub struct Base {
    pub name: String,
    pub odometry: OdometrySource,
    /// The limits along x, forward positive.
    pub linear_x: AxisLimits,
    /// The limits of the turn, counterclockwise positive.
    pub angular: AxisLimits,
    pub padded_footprint: PaddedFootprint,
    pub inflation: Inflation,
}

/// Why describe_base's answer describes no base the navigator drives.
#[derive(Debug, thiserror::Error)]
pub enum BaseError {
    #[error(
        "describe_base answers the odometry_source {0:?}, neither \"measured\" nor \"commanded\""
    )]
    Odometry(String),

    #[error(
        "describe_base answers a {name} of {value} for {axis}, which is no number of at least 0"
    )]
    Limit {
        axis: &'static str,
        name: &'static str,
        value: f64,
    },

    #[error("describe_base answers {axis} limits base_state's limiter refuses")]
    Axis {
        axis: &'static str,
        #[source]
        source: LimiterError,
    },

    #[error(
        "describe_base answers a linear_limits max_velocity along x of {0} m/s: the navigator \
         drives a base forward"
    )]
    CannotDrive(f64),

    #[error(
        "describe_base answers a footprint in {0:?}, where base_state gives it in {ROBOT_FRAME:?}"
    )]
    FootprintFrame(String),

    #[error("describe_base answers a footprint the navigator cannot place")]
    Footprint(#[from] FootprintError),

    #[error("describe_base's footprint padded by footprint_padding_m")]
    Padding(#[from] PaddingError),

    #[error("inflation_radius_m against describe_base's footprint")]
    Inflation(#[from] InflationBelowInscribed),
}

/// The limits of one axis, as describe_base gives them, by the names of the
/// limiter's parameters.
struct AxisAnswer {
    axis: &'static str,
    min_velocity: f64,
    max_velocity: f64,
    max_acceleration: f64,
    max_deceleration: f64,
    deadband: f64,
}

impl AxisAnswer {
    fn parse(self) -> Result<AxisLimits, BaseError> {
        let axis = self.axis;
        let limit = |name: &'static str, value: f64| {
            NonNegativeFinite::parse(value).map_err(|_| BaseError::Limit { axis, name, value })
        };
        AxisLimits::parse(AxisLimitsParams {
            min_velocity: self.min_velocity,
            max_velocity: self.max_velocity,
            max_acceleration: limit("max_acceleration", self.max_acceleration)?,
            max_deceleration: limit("max_deceleration", self.max_deceleration)?,
            deadband: limit("deadband", self.deadband)?,
        })
        .map_err(|source| BaseError::Axis { axis, source })
    }
}

/// The larger magnitude of `axis`'s slowest and fastest velocity.
pub fn fastest(axis: &AxisLimits) -> f64 {
    let (min, max) = axis.velocity_range();
    min.abs().max(max)
}

/// The time (s) `axis`'s deceleration takes to bring the axis to rest from
/// its fastest velocity; 0 on an axis of range 0 to 0.
fn time_to_stop_s(axis: &AxisLimits) -> f64 {
    let fastest = fastest(axis);
    if fastest == 0.0 {
        return 0.0;
    }
    fastest / axis.max_deceleration()
}

impl Base {
    /// The base `answer` describes, its footprint padded and inflated as
    /// `config` says.
    pub fn parse(
        answer: &describe_base::ResponseData,
        config: &NavigatorConfig,
    ) -> Result<Self, BaseError> {
        let odometry = match answer.odometry_source.as_str() {
            "measured" => OdometrySource::Measured,
            "commanded" => OdometrySource::Commanded,
            other => return Err(BaseError::Odometry(other.to_string())),
        };
        let linear = &answer.linear_limits;
        let linear_x = AxisAnswer {
            axis: "linear_limits along x",
            min_velocity: linear.min_velocity[0],
            max_velocity: linear.max_velocity[0],
            max_acceleration: linear.max_acceleration[0],
            max_deceleration: linear.max_deceleration[0],
            deadband: linear.deadband[0],
        }
        .parse()?;
        let turn = &answer.angular_limits;
        let angular = AxisAnswer {
            axis: "angular_limits",
            min_velocity: turn.min_rate,
            max_velocity: turn.max_rate,
            max_acceleration: turn.max_acceleration,
            max_deceleration: turn.max_deceleration,
            deadband: turn.deadband,
        }
        .parse()?;
        let forward = linear_x.velocity_range().1;
        if forward <= 0.0 {
            return Err(BaseError::CannotDrive(forward));
        }
        let footprint = &answer.footprint;
        if footprint.frame != ROBOT_FRAME {
            return Err(BaseError::FootprintFrame(footprint.frame.clone()));
        }
        let points: Vec<[f64; 2]> = footprint
            .points
            .iter()
            .map(|point| [point.position[0], point.position[1]])
            .collect();
        let padded_footprint =
            PaddedFootprint::new(&Footprint::parse(&points)?, config.grid.footprint_padding)?;
        let inflation = Inflation::parse(config.grid.inflation, &padded_footprint)?;
        Ok(Self {
            name: answer.base_name.clone(),
            odometry,
            linear_x,
            angular,
            padded_footprint,
            inflation,
        })
    }

    /// The fastest forward speed (m/s) the base executes.
    pub fn forward_speed_m_s(&self) -> f64 {
        self.linear_x.velocity_range().1
    }

    /// The longer of the times (s) that the deceleration along x and the
    /// turn's take to bring the base to rest from the fastest velocity of
    /// their axis.
    fn time_to_stop_s(&self) -> f64 {
        time_to_stop_s(&self.linear_x).max(time_to_stop_s(&self.angular))
    }

    /// The stop's horizon from the limits (s): the source timeout and the
    /// time to stop.
    fn limits_horizon_s(&self, stop: &StopSettings) -> f64 {
        stop.source_timeout_s.get() + self.time_to_stop_s()
    }

    /// How far ahead (s) the stop projects the padded footprint: the larger
    /// of the time before a collision and the horizon from the limits.
    pub fn stop_horizon_s(&self, stop: &StopSettings) -> f64 {
        stop.time_before_collision_s
            .get()
            .max(self.limits_horizon_s(stop))
    }
}

/// describe_base's answer for one start of the odometry, or why it describes
/// no base the navigator drives.
#[derive(Clone, Debug)]
pub struct BaseRead {
    pub odometry_start: SystemTime,
    pub answer: Result<Arc<Base>, String>,
}

/// The newest start of the odometry, as drive_states reports it, and the
/// newest read of describe_base, as a goal waits for them on the node's
/// clock.
#[derive(Clone)]
pub struct BaseReads {
    pub starts: watch::Receiver<Option<SystemTime>>,
    pub reads: watch::Receiver<Option<BaseRead>>,
    pub clock: NodeClock,
}

impl BaseReads {
    /// The base of the read for the newest start of the odometry: at once
    /// when it is read, or once it is, waiting up to `timeout_ns` on the
    /// node's clock; refused with `resource_unavailable` when the wait runs
    /// out and when the answer describes a base the navigator cannot drive.
    pub async fn newest(&mut self, timeout_ns: i64) -> Result<Arc<Base>, Failure> {
        let deadline_ns = clock::now_ns(&self.clock).saturating_add(timeout_ns);
        loop {
            let start = *self.starts.borrow_and_update();
            let read = self.reads.borrow_and_update().clone();
            if let Some(read) = read.filter(|read| Some(read.odometry_start) == start) {
                return read.answer.map_err(|reason| {
                    unavailable(format!(
                        "describe_base's answer describes a base the navigator cannot drive: \
                         {reason}"
                    ))
                });
            }
            if clock::now_ns(&self.clock) >= deadline_ns {
                return Err(unavailable(unanswered(start.is_some(), timeout_ns)));
            }
            let deadline = u64::try_from(deadline_ns).unwrap_or(u64::MAX);
            tokio::select! {
                () = self.clock.until(deadline) => {}
                changed = self.starts.changed() => if changed.is_err() {
                    return Err(unavailable(unanswered(start.is_some(), timeout_ns)));
                },
                changed = self.reads.changed() => if changed.is_err() {
                    return Err(unavailable(unanswered(start.is_some(), timeout_ns)));
                },
            }
        }
    }
}

fn unavailable(message: String) -> Failure {
    Failure::new(FailureKind::ResourceUnavailable, message)
}

/// What a goal waited for and did not get within `timeout_ns`: drive_states,
/// or once one arrived, describe_base's answer for its odometry.
fn unanswered(drive_states_arrived: bool, timeout_ns: i64) -> String {
    let within = seconds(timeout_ns);
    if drive_states_arrived {
        return format!(
            "describe_base did not answer for the base's newest odometry start within \
             source_timeout_s {within} s: wait for the base"
        );
    }
    format!(
        "no drive_states arrived from the base within source_timeout_s {within} s, so \
         describe_base is not read: wait for the base"
    )
}

/// Log the stop's horizons for `base`: the time before a collision, the
/// horizon from the base's limits, and the larger, which the stop projects.
fn log_horizons(config: &NavigatorConfig, base: &Base) {
    let stop = &config.stop;
    info!(
        "base {}: the stop projects {} s, the larger of time_before_collision_s {} s and the \
         sum of source_timeout_s {} s and the base's {} s to rest from its fastest velocity, \
         {} s",
        base.name,
        base.stop_horizon_s(stop),
        stop.time_before_collision_s.get(),
        stop.source_timeout_s.get(),
        base.time_to_stop_s(),
        base.limits_horizon_s(stop)
    );
}

/// The read of `answer`: the base, or why it describes none the navigator
/// drives, with the grid of the floor that `grid` takes for it, built for
/// `sensors` sensors around odom's origin.
fn read(
    answer: &describe_base::ResponseData,
    config: &NavigatorConfig,
    sensors: NonZeroUsize,
    grid: &Mutex<GridSlot>,
) -> Result<Arc<Base>, String> {
    let base = Base::parse(answer, config).map_err(|e| match std::error::Error::source(&e) {
        Some(source) => format!("{e}: {source}"),
        None => e.to_string(),
    })?;
    log_horizons(config, &base);
    let built = LiveGrid::new(config, &base, sensors, [0.0, 0.0])
        .map_err(|e| format!("the grid of the floor for its footprint: {e}"))?;
    lock_ignoring_poison(grid).set(built);
    Ok(Arc::new(base))
}

/// Read describe_base at each start of the odometry that `starts` gives,
/// asked until it answers or the odometry starts over again, and publish
/// each read on `reads`, until the node cancels.
pub async fn run_reader(
    runner: Arc<NodeRunner>,
    config: Arc<NavigatorConfig>,
    sensors: NonZeroUsize,
    mut starts: watch::Receiver<Option<SystemTime>>,
    reads: watch::Sender<Option<BaseRead>>,
    grid: Arc<Mutex<GridSlot>>,
) {
    let token = runner.cancellation_token().clone();
    loop {
        let unread = |start: &Option<SystemTime>| {
            start.is_some_and(|start| {
                reads
                    .borrow()
                    .as_ref()
                    .is_none_or(|read| read.odometry_start != start)
            })
        };
        let start = tokio::select! {
            () = token.cancelled() => return,
            start = starts.wait_for(unread) => match start {
                Ok(start) => (*start).expect("a start the reader has not read"),
                Err(_) => return,
            },
        };
        let mut superseded = starts.clone();
        let stopped = token.clone();
        let asker = runner.clone();
        let answer = asking::ask(
            "describe_base",
            async move {
                tokio::select! {
                    () = stopped.cancelled() => {}
                    _ = superseded.wait_for(|newest| *newest != Some(start)) => {}
                }
            },
            async move || {
                describe_base::poll(
                    &asker,
                    describe_base::bound_producer(&asker),
                    ANSWER_TIMEOUT,
                )
                .await
                .map(|response| response.data)
            },
        )
        .await;
        let Some(answer) = answer else {
            continue;
        };
        let read = read(&answer, &config, sensors, &grid);
        if let Err(reason) = &read {
            error!("describe_base: {reason}");
        }
        reads.send_replace(Some(BaseRead {
            odometry_start: start,
            answer: read,
        }));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use describe_base::{
        ResponseDataAngularLimits, ResponseDataFootprint, ResponseDataFootprintPointsItem,
        ResponseDataLinearLimits,
    };

    use crate::config::tests::config;

    /// describe_base's answer for a differential base of 0.5 m/s forward and
    /// backward and 1.5 rad/s, decelerating at `deceleration` m/s^2 and
    /// rad/s^2, 0.44 m long and 0.40 m wide about the robot frame's origin.
    pub(crate) fn answer(deceleration: f64) -> describe_base::ResponseData {
        describe_base::ResponseData {
            base_name: "base".to_string(),
            odometry_source: "measured".to_string(),
            linear_limits: ResponseDataLinearLimits {
                min_velocity: [-0.5, 0.0],
                max_velocity: [0.5, 0.0],
                max_acceleration: [1.0, 0.0],
                max_deceleration: [deceleration, 0.0],
                deadband: [0.02, 0.0],
                rest_velocity: [0.01, 0.0],
            },
            angular_limits: ResponseDataAngularLimits {
                min_rate: -1.5,
                max_rate: 1.5,
                max_acceleration: 3.0,
                max_deceleration: deceleration,
                deadband: 0.05,
                rest_rate: 0.01,
            },
            footprint: ResponseDataFootprint {
                frame: ROBOT_FRAME.to_string(),
                timestamp: None,
                points: [[0.22, -0.2], [0.22, 0.2], [-0.22, 0.2], [-0.22, -0.2]]
                    .iter()
                    .map(|&[x, y]| ResponseDataFootprintPointsItem {
                        position: [x, y, 0.0],
                    })
                    .collect(),
            },
        }
    }

    pub(crate) fn base(deceleration: f64) -> Base {
        Base::parse(&answer(deceleration), &config()).expect("a base")
    }

    #[test]
    fn a_base_is_its_odometry_limits_and_padded_footprint() {
        let base = base(1.0);
        assert_eq!(base.name, "base");
        assert_eq!(base.odometry, OdometrySource::Measured);
        assert_eq!(base.forward_speed_m_s(), 0.5);
        assert_eq!(fastest(&base.angular), 1.5);
        // The tests' padding of 0.01 m grows the 0.40 m wide outline.
        assert!((base.padded_footprint.inscribed_radius_m() - 0.21).abs() < 1e-9);
    }

    // The stop projects the larger of time_before_collision_s, 1 s, and the
    // sum of source_timeout_s, 0.5 s, and the longer time to stop: 1.5 / 1 +
    // 0.5 = 2 s, and 1.5 / 6 + 0.5 = 0.75 s.
    #[test]
    fn a_limits_horizon_above_time_before_collision_lengthens_the_stop() {
        let stop = config().stop;
        assert_eq!(base(1.0).stop_horizon_s(&stop), 2.0);
        assert_eq!(base(6.0).limits_horizon_s(&stop), 0.75);
        assert_eq!(base(6.0).stop_horizon_s(&stop), 1.0);
    }

    #[test]
    fn an_answer_of_no_odometry_no_forward_speed_or_another_frame_is_refused() {
        let unknown = describe_base::ResponseData {
            odometry_source: "wheels".to_string(),
            ..answer(1.0)
        };
        assert!(matches!(
            Base::parse(&unknown, &config()),
            Err(BaseError::Odometry(_))
        ));
        let mut backward = answer(1.0);
        backward.linear_limits.max_velocity[0] = 0.0;
        assert!(matches!(
            Base::parse(&backward, &config()),
            Err(BaseError::CannotDrive(_))
        ));
        let mut elsewhere = answer(1.0);
        elsewhere.footprint.frame = "odom".to_string();
        assert!(matches!(
            Base::parse(&elsewhere, &config()),
            Err(BaseError::FootprintFrame(_))
        ));
    }

    // A goal that arrives before drive_states is refused naming
    // drive_states once its wait runs out; a read of an older start does not
    // answer it, and the read of the newest start does.
    #[tokio::test]
    async fn a_goal_waits_for_the_read_of_the_newest_start() {
        let (clock, feed) = NodeClock::simulation();
        feed.tick(1_000);
        let (starts_tx, starts) = watch::channel(None);
        let (reads_tx, reads) = watch::channel(None);
        let mut base_reads = BaseReads {
            starts,
            reads,
            clock,
        };
        let refused = base_reads.newest(0).await.expect_err("no drive_states");
        assert!(
            refused.message.contains("no drive_states"),
            "{}",
            refused.message
        );
        let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let new = SystemTime::UNIX_EPOCH + Duration::from_secs(2);
        starts_tx.send_replace(Some(new));
        reads_tx.send_replace(Some(BaseRead {
            odometry_start: old,
            answer: Ok(Arc::new(base(1.0))),
        }));
        let unanswered = base_reads
            .newest(0)
            .await
            .expect_err("an older start's read");
        assert!(
            unanswered
                .message
                .starts_with("describe_base did not answer"),
            "{}",
            unanswered.message
        );
        reads_tx.send_replace(Some(BaseRead {
            odometry_start: new,
            answer: Ok(Arc::new(base(2.0))),
        }));
        let read = base_reads.newest(0).await.expect("the newest read");
        assert_eq!(read.time_to_stop_s(), 0.75);
    }
}
