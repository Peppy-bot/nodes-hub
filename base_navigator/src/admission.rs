//! A goal of base_navigation checked in the contract's order, against what
//! the navigator knows when it arrives. A goal with several causes gets the
//! kind of the earliest check:
//!
//! 1. it waits up to the source timeout for describe_base's read of the newest
//!    odometry start, the first drive_states included, and is refused with
//!    `resource_unavailable` when the wait runs out or the answer describes a
//!    base the navigator cannot drive, and with `precondition_failed` while
//!    the answer names commanded odometry;
//! 2. `rejected` for a value it does not take, each action's own;
//! 3. a navigation whose target is in the map frame: `not_localized`, as no
//!    map is loaded;
//! 4. a navigation whose target is in another frame: placed in odom by the
//!    robot's tree at its timestamp, and refused with the lookup's kind, then
//!    `rejected` when the placed target is not planar or lies outside the grid
//!    centered on the base;
//! 5. `resource_unavailable` while more than 0.5 s passed since the newest
//!    drive_states arrived, while it reports base_silence, and while a sensor
//!    the grid reads is silent, the refusal naming each that holds;
//! 6. `precondition_failed` while the newest drive_states reports base_mode,
//!    parts_within_footprint false, robot_stop or follower_goal.
//!
//! base_navigation's `leader_streaming` holds while a leader that the
//! navigator follows streams; this navigator follows no leader.

use std::sync::{Arc, Mutex};

use control_core::drive_link::FollowState;
use control_core::geometry::{GeometryError, Pose, Stamped, target_off_the_floor};
use control_core::node_clock::NodeClock;
use control_core::planar_pose::PlanarPose;
use control_core::positive_finite::PositiveFinite;
use control_core::robot_tree::{Placer, Unplaced};
use control_core::sync::{lock_ignoring_poison, wait_for};
use control_core::time::{seconds, wait_from_secs};
use frame_tree::arithmetic::normalize_angle;
use peppygen::exposed_actions::base_navigation::{
    back_up, drive_on_heading, navigate_to_pose, spin,
};

use crate::base::{Base, BaseReads, OdometrySource, fastest};
use crate::cameras::Sensors;
use crate::clock;
use crate::config::{NavigatorConfig, duration_ns};
use crate::drive::NavigationRequest;
use crate::failure::{
    Failure, FailureKind, NavigationFailure, NavigationKind, TargetFailure, of_navigation,
};
use crate::follow::odometry_frame;
use crate::grid::GridSlot;
use crate::link::{Link, Received};
use crate::recovery::{Direction, SpinRequest, StraightRequest};

/// The frame of the loaded map, as transforms names it.
const MAP_FRAME: &str = "map";

/// How long after a refusal for a hold that lasts on after its cause a goal
/// is sent again, as base_navigation says: the hold ends more than 0.5 s
/// after the navigator's newest setpoint, and the base reports following at
/// its next drive_states, at most 0.1 s later.
const AFTER_A_HOLD: &str = "send the goal again more than 0.6 s after this refusal";

/// A goal the checks admit: the base it drives, and the newest drive_states
/// when it was accepted.
#[derive(Clone, Debug)]
pub struct Admitted {
    pub base: Arc<Base>,
    pub at: Received,
}

/// What the goals' checks and the control loop share: the navigator's
/// settings, the node's clock, drive_states, the grid of the floor and its
/// sensors.
#[derive(Clone)]
pub struct Shared {
    pub config: Arc<NavigatorConfig>,
    pub clock: NodeClock,
    pub link: Arc<Mutex<Link>>,
    pub grid: Arc<Mutex<GridSlot>>,
    pub sensors: Arc<Sensors>,
}

/// What a goal is checked against: what it shares with the control loop,
/// describe_base's reads, and the robot's tree.
#[derive(Clone)]
pub struct Admission {
    pub shared: Shared,
    pub base_reads: BaseReads,
    pub placer: Arc<Placer>,
}

fn rejected(message: impl Into<String>) -> Failure {
    Failure::new(FailureKind::Rejected, message)
}

fn precondition_failed(message: impl Into<String>) -> Failure {
    Failure::new(FailureKind::PreconditionFailed, message)
}

fn resource_unavailable(message: impl Into<String>) -> Failure {
    Failure::new(FailureKind::ResourceUnavailable, message)
}

/// `value` of `field`, a finite number.
fn finite(field: &str, value: f64) -> Result<f64, Failure> {
    if value.is_finite() {
        return Ok(value);
    }
    Err(rejected(format!(
        "{field} must be a finite number, got {value}"
    )))
}

/// `value` of `field`, a positive finite number.
fn positive(field: &str, value: f64) -> Result<PositiveFinite, Failure> {
    PositiveFinite::parse_field(field, value).map_err(rejected)
}

/// A goal's `timeout_s`: none for 0, no bound; refused below 0 and when not
/// finite.
fn timeout(value: f64) -> Result<Option<i64>, Failure> {
    let wait = wait_from_secs(value).map_err(|e| rejected(format!("timeout_s {e}")))?;
    Ok((!wait.is_zero()).then(|| duration_ns(wait)))
}

/// The refusal of a target that, in `frame`, the frame it runs in, lies off
/// the floor plane, naming the part, its value there, and what to send.
fn off_the_floor(refusal: &GeometryError, frame: &str) -> Failure {
    rejected(target_off_the_floor(refusal, frame))
}

/// The goal's `max_speed_m_s` held to describe_base's limits along x: none
/// for 0, the base's limits alone.
fn max_speed(max_speed_m_s: f64, base: &Base) -> Result<Option<PositiveFinite>, Failure> {
    let speed = finite("max_speed_m_s", max_speed_m_s)?;
    let largest = fastest(&base.linear_x);
    let deadband = base.linear_x.deadband();
    if speed < 0.0 || speed > largest || (speed > 0.0 && speed < deadband) {
        return Err(rejected(format!(
            "max_speed_m_s must be 0, for the base's limits alone, or lie from describe_base's \
             linear_limits deadband along x, {deadband} m/s, to the larger magnitude of its \
             velocities along x, {largest} m/s, got {speed} m/s"
        )));
    }
    if speed == 0.0 {
        return Ok(None);
    }
    Ok(Some(
        PositiveFinite::parse(speed).expect("a finite speed above 0"),
    ))
}

/// The controller's speed: the base's forward limit, at most the goal's
/// `max_speed`.
fn navigation_speed(max_speed: Option<PositiveFinite>, base: &Base) -> PositiveFinite {
    let forward = base.forward_speed_m_s();
    let speed = max_speed.map_or(forward, |cap| cap.get().min(forward));
    PositiveFinite::parse(speed).expect("a positive speed up to a positive forward limit")
}

impl Admission {
    /// Check 1: describe_base's read for the newest start of the odometry,
    /// and the newest drive_states.
    fn base(&self) -> Result<(Arc<Base>, Received), Failure> {
        let mut reads = self.base_reads.clone();
        let base = wait_for(reads.newest(self.shared.config.stop.source_timeout_ns()))?;
        if base.odometry == OdometrySource::Commanded {
            return Err(precondition_failed(
                "describe_base names commanded odometry: a goal of base_navigation needs a base \
                 that measures its motion",
            ));
        }
        let newest = lock_ignoring_poison(&self.shared.link)
            .newest()
            .expect("describe_base is read once drive_states arrived");
        Ok((base, newest))
    }

    /// Checks 5 and 6: drive_states, the sensors, and the holds, with the
    /// newest drive_states the goal is admitted at.
    fn follows(&self, base: Arc<Base>) -> Result<Admitted, Failure> {
        let now_ns = clock::now_ns(&self.shared.clock);
        let at = lock_ignoring_poison(&self.shared.link)
            .newest()
            .expect("describe_base is read once drive_states arrived");
        let state = at.state;
        let silent = lock_ignoring_poison(&self.shared.grid)
            .silent_sensors(now_ns, self.shared.sensors.count().get());
        let waits: Vec<String> = [
            at.silent_at(now_ns).then(|| {
                format!(
                    "the base sent no drive_states for {} s: wait for the base",
                    seconds(now_ns - at.arrived_ns)
                )
            }),
            (state.follow == FollowState::BaseSilence).then(|| {
                format!(
                    "drive_states reports follow_state base_silence, a silence of the base's \
                     motion sensing that holds the base: {AFTER_A_HOLD}"
                )
            }),
            (!silent.is_empty()).then(|| {
                format!(
                    "the navigator waits for {}",
                    self.shared
                        .sensors
                        .waits(&silent, self.shared.config.stop.source_timeout_s.get())
                )
            }),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !waits.is_empty() {
            return Err(resource_unavailable(waits.join("; ")));
        }
        if state.follow == FollowState::BaseMode {
            let mode = state.mode.as_str();
            return Err(precondition_failed(match state.mode.to_standing() {
                Some(what) => format!(
                    "drive_states reports follow_state base_mode, the base's mode {mode} holding \
                     it: {what}, then send the goal again"
                ),
                None => format!(
                    "drive_states reports follow_state base_mode while the base's mode is \
                     {mode}: {AFTER_A_HOLD}"
                ),
            }));
        }
        if !state.parts_within_footprint {
            return Err(precondition_failed(
                "drive_states reports parts_within_footprint false, a limb outside the base's \
                 footprint: bring the limbs home with postures' move_to_home, then send the \
                 goal again",
            ));
        }
        match state.follow {
            FollowState::RobotStop => Err(precondition_failed(format!(
                "drive_states reports follow_state robot_stop, a robot stop that holds the base: \
                 {AFTER_A_HOLD}"
            ))),
            FollowState::FollowerGoal => Err(precondition_failed(
                "drive_states reports follow_state follower_goal, a goal of the base's node that \
                 holds the base: send the goal again once that goal ends",
            )),
            FollowState::Following => Ok(Admitted { base, at }),
            FollowState::BaseMode | FollowState::BaseSilence => {
                unreachable!("base_mode and base_silence are refused before the limbs")
            }
        }
    }

    /// A navigate_to_pose goal, checked in base_navigation's order.
    pub fn navigation(
        &self,
        data: &navigate_to_pose::GoalRequestData,
    ) -> Result<(NavigationRequest, Admitted), TargetFailure> {
        let navigation = |failure: Failure| TargetFailure::from(of_navigation(failure));
        let (base, newest) = self.base().map_err(navigation)?;
        let target = &data.target;
        let stamped = Pose::parse(target.position, target.orientation)
            .and_then(|pose| {
                Stamped::parse(
                    self.placer.frame_of(target.frame.as_deref()),
                    target.timestamp,
                    pose,
                )
            })
            .map_err(|e| navigation(rejected(format!("the target: {e}"))))?;
        let position_tolerance_m =
            positive("position_tolerance_m", data.position_tolerance_m).map_err(navigation)?;
        let orientation_tolerance_rad =
            positive("orientation_tolerance_rad", data.orientation_tolerance_rad)
                .map_err(navigation)?;
        let max_speed_m_s = max_speed(data.max_speed_m_s, &base).map_err(navigation)?;
        let speed_m_s = navigation_speed(max_speed_m_s, &base);
        if stamped.frame().as_str() == MAP_FRAME {
            stamped
                .value()
                .planar()
                .map_err(|e| navigation(off_the_floor(&e, MAP_FRAME)))?;
            return Err(TargetFailure::from(NavigationFailure::new(
                NavigationKind::NotLocalized,
                "no map is loaded: send the target in odom or the robot frame",
            )));
        }
        let odom = odometry_frame();
        let placed = self
            .placer
            .place_in_now(&stamped, &odom)
            .map_err(|e| Unplaced::of("the target", &e))?
            .value
            .planar()
            .map_err(|e| navigation(off_the_floor(&e, odom.as_str())))?;
        let target = PlanarPose::new(placed.x(), placed.y(), normalize_angle(placed.yaw()))
            .expect("a finite heading wraps to a finite heading");
        let grid = &self.shared.config.grid;
        let half_side_m = grid.half_side_m();
        let base_at = [newest.state.pose.x(), newest.state.pose.y()];
        if !grid.square_holds(base_at, [target.x(), target.y()]) {
            return Err(navigation(rejected(format!(
                "the target, placed in odom at x {} m, y {} m, lies outside the grid of the floor \
                 centered on the base, {half_side_m} m from it along odom's x and y: send a \
                 nearer target",
                target.x(),
                target.y()
            ))));
        }
        let admitted = self.follows(base).map_err(navigation)?;
        Ok((
            NavigationRequest {
                target,
                position_tolerance_m,
                orientation_tolerance_rad,
                max_speed_m_s,
                speed_m_s,
            },
            admitted,
        ))
    }

    /// A spin goal, checked in base_navigation's order.
    pub fn spin(&self, data: &spin::GoalRequestData) -> Result<(SpinRequest, Admitted), Failure> {
        let (base, _) = self.base()?;
        let angle_rad = finite("angle_rad", data.angle_rad)?;
        let timeout_ns = timeout(data.timeout_s)?;
        let (clockwise, counterclockwise) = base.angular.velocity_range();
        if (angle_rad > 0.0 && counterclockwise == 0.0) || (angle_rad < 0.0 && clockwise == 0.0) {
            return Err(rejected(format!(
                "angle_rad {angle_rad} turns the base the way describe_base's angular_limits \
                 give a rate of 0, min_rate {clockwise} and max_rate {counterclockwise} rad/s: \
                 send a turn the base executes"
            )));
        }
        let admitted = self.follows(base)?;
        Ok((
            SpinRequest {
                angle_rad,
                timeout_ns,
            },
            admitted,
        ))
    }

    /// A drive_on_heading (`Forward`) or a back_up (`Backward`) of
    /// `distance_m` at `speed_m_s` within `timeout_s`, checked in
    /// base_navigation's order.
    fn straight(
        &self,
        direction: Direction,
        distance_m: f64,
        speed_m_s: f64,
        timeout_s: f64,
    ) -> Result<(StraightRequest, Admitted), Failure> {
        let (base, newest) = self.base()?;
        let distance = positive("distance_m", distance_m)?;
        let speed = positive("speed_m_s", speed_m_s)?;
        let timeout_ns = timeout(timeout_s)?;
        let deadband = base.linear_x.deadband();
        let limit = direction.fastest_m_s(&base);
        if speed.get() < deadband || speed.get() > limit {
            let (name, bound) = match direction {
                Direction::Forward => ("max_velocity", "drive_on_heading"),
                Direction::Backward => ("the magnitude of min_velocity", "back_up"),
            };
            return Err(rejected(format!(
                "speed_m_s of a {bound} must lie from describe_base's linear_limits deadband along \
                 x, {deadband} m/s, to {name} along x, {limit} m/s, got {} m/s",
                speed.get()
            )));
        }
        let pose = newest.state.pose;
        let grid = &self.shared.config.grid;
        let half_side_m = grid.half_side_m();
        let end = direction.end_of(pose, distance.get());
        if !grid.square_holds([pose.x(), pose.y()], end) {
            return Err(rejected(format!(
                "distance_m {} m ends the drive outside the grid of the floor centered on the \
                 base, {half_side_m} m from it along odom's x and y: send a shorter distance",
                distance.get()
            )));
        }
        let admitted = self.follows(base)?;
        Ok((
            StraightRequest {
                distance_m: distance,
                speed_m_s: speed,
                timeout_ns,
            },
            admitted,
        ))
    }

    /// A drive_on_heading goal, checked in base_navigation's order.
    pub fn drive_on_heading(
        &self,
        data: &drive_on_heading::GoalRequestData,
    ) -> Result<(StraightRequest, Admitted), Failure> {
        self.straight(
            Direction::Forward,
            data.distance_m,
            data.speed_m_s,
            data.timeout_s,
        )
    }

    /// A back_up goal, checked in base_navigation's order.
    pub fn back_up(
        &self,
        data: &back_up::GoalRequestData,
    ) -> Result<(StraightRequest, Admitted), Failure> {
        self.straight(
            Direction::Backward,
            data.distance_m,
            data.speed_m_s,
            data.timeout_s,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::{Duration, SystemTime};

    use control_core::robot_tree::RobotTree;
    use frame_tree::arithmetic::quaternion_of_yaw;
    use frame_tree::{MovingMessage, Publisher, TransformRecord};
    use hegel::TestCase;
    use hegel::generators as gs;
    use tokio::sync::watch;

    use crate::base::BaseRead;
    use crate::base::tests::answer;
    use crate::cameras::tests::camera_slot;
    use crate::config::tests::config;
    use crate::follow::{ROBOT_FRAME, robot_frame};
    use crate::grid::tests::grid;
    use crate::link::Mode;
    use crate::link::tests::state;

    const NOW_NS: i64 = 10_000_000_000;

    /// The holds a test reports, at most one: drive_states carry one
    /// follow_state.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Hold {
        None,
        BaseSilence,
        BaseMode,
        RobotStop,
    }

    /// The causes a test sets: each true breaks one check of the order.
    #[derive(Clone, Copy, Debug)]
    struct Causes {
        no_read: bool,
        commanded: bool,
        rejected: bool,
        map: bool,
        silent_link: bool,
        silent_sensor: bool,
        parts_outside: bool,
        hold: Hold,
    }

    /// The kind of each cause, in base_navigation's order.
    fn earliest(c: Causes) -> Option<navigate_to_pose::FailureKind> {
        use navigate_to_pose::FailureKind as K;
        [
            (c.no_read, K::ResourceUnavailable),
            (c.commanded, K::PreconditionFailed),
            (c.rejected, K::Rejected),
            (c.map, K::NotLocalized),
            (c.silent_link, K::ResourceUnavailable),
            (c.hold == Hold::BaseSilence, K::ResourceUnavailable),
            (c.silent_sensor, K::ResourceUnavailable),
            (c.hold == Hold::BaseMode, K::PreconditionFailed),
            (c.parts_outside, K::PreconditionFailed),
            (c.hold == Hold::RobotStop, K::PreconditionFailed),
        ]
        .into_iter()
        .find_map(|(set, kind)| set.then_some(kind))
    }

    /// The tree of a base whose odometry places the robot frame at x 1 m,
    /// y 2 m, heading 0.5 rad in odom.
    fn placer() -> Arc<Placer> {
        let tree = RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window");
        let backbone = Publisher {
            core_node: "core".to_owned(),
            instance_id: "backbone".to_owned(),
        };
        let stamp = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        tree.take_moving(
            &backbone,
            &MovingMessage {
                timestamp: stamp,
                transforms: vec![TransformRecord::new(
                    "odom",
                    ROBOT_FRAME,
                    [1.0, 2.0, 0.0],
                    quaternion_of_yaw(0.5),
                )],
                start_timestamp: SystemTime::UNIX_EPOCH,
            },
        )
        .expect("odom to the robot frame");
        Arc::new(Placer::new(Arc::new(tree), robot_frame(), Duration::ZERO))
    }

    fn admission(c: Causes) -> Admission {
        // The clock keeps its tick once its feed is gone.
        let (clock, feed) = NodeClock::simulation();
        feed.tick(NOW_NS as u64);
        let mut described = answer(1.0);
        if c.commanded {
            described.odometry_source = "commanded".to_string();
        }
        let base = Arc::new(Base::parse(&described, &config()).expect("a base"));
        let mut newest = state(0, 10, 1.0, FollowState::Following);
        newest.pose = PlanarPose::new(1.0, 2.0, 0.5).unwrap();
        newest.follow = match c.hold {
            Hold::None => FollowState::Following,
            Hold::BaseSilence => FollowState::BaseSilence,
            Hold::BaseMode => FollowState::BaseMode,
            Hold::RobotStop => FollowState::RobotStop,
        };
        if c.hold == Hold::BaseMode {
            newest.mode = Mode::Damped;
        }
        newest.parts_within_footprint = !c.parts_outside;
        let mut link = Link::default();
        let arrived_ns = if c.silent_link {
            NOW_NS - 500_000_001
        } else {
            NOW_NS
        };
        link.take(newest, arrived_ns);
        let (_, starts) = watch::channel(Some(newest.odometry_start));
        let read = (!c.no_read).then(|| BaseRead {
            odometry_start: newest.odometry_start,
            answer: Ok(base.clone()),
        });
        let (_, reads) = watch::channel(read);
        let mut live = grid();
        let reading_ns = if c.silent_sensor {
            NOW_NS - 500_000_001
        } else {
            NOW_NS
        };
        live.insert_depth_frame(
            0,
            &[],
            PlanarPose::IDENTITY,
            frame_tree::Transform::IDENTITY,
            PositiveFinite::parse(4.0).unwrap(),
            reading_ns,
        )
        .expect("source 0");
        let mut slot = GridSlot::default();
        slot.set(live);
        Admission {
            shared: Shared {
                config: Arc::new(config()),
                clock: clock.clone(),
                link: Arc::new(Mutex::new(link)),
                grid: Arc::new(Mutex::new(slot)),
                sensors: Arc::new(Sensors::new(vec![camera_slot("front")])),
            },
            base_reads: BaseReads {
                starts,
                reads,
                clock,
            },
            placer: placer(),
        }
    }

    /// A navigation 1 m ahead of the robot frame, in the frame `frame`.
    fn goal(frame: Option<&str>, max_speed_m_s: f64) -> navigate_to_pose::GoalRequestData {
        navigate_to_pose::GoalRequestData {
            target: navigate_to_pose::GoalTarget {
                frame: frame.map(str::to_owned),
                timestamp: None,
                position: [1.0, 0.0, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
            },
            position_tolerance_m: 0.1,
            orientation_tolerance_rad: 0.2,
            max_speed_m_s,
        }
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("a runtime")
    }

    // A goal with several causes gets the kind of the earliest check,
    // and a goal with none is admitted, its target placed in odom.
    #[hegel::test]
    fn a_goal_with_several_causes_gets_the_kind_of_the_earliest(tc: TestCase) {
        let draw = || tc.draw(gs::booleans());
        let index: usize = tc.draw(gs::integers().min_value(0).max_value(3));
        let hold = [
            Hold::None,
            Hold::BaseSilence,
            Hold::BaseMode,
            Hold::RobotStop,
        ][index];
        let causes = Causes {
            no_read: draw(),
            commanded: draw(),
            rejected: draw(),
            map: draw(),
            silent_link: draw(),
            silent_sensor: draw(),
            parts_outside: draw(),
            hold,
        };
        let admission = admission(causes);
        let data = goal(
            causes.map.then_some(MAP_FRAME),
            if causes.rejected { -0.1 } else { 0.0 },
        );
        let admitted = runtime().block_on(async { admission.navigation(&data) });
        let refused = admitted
            .as_ref()
            .err()
            .cloned()
            .map(|refusal| refusal.for_member(navigate_to_pose::FailureKind::parse));
        assert_eq!(
            refused.as_ref().map(|(kind, _)| *kind),
            earliest(causes),
            "{causes:?}: {refused:?}"
        );
        if let Ok((request, _)) = admitted {
            let expected = [1.0 + 0.5_f64.cos(), 2.0 + 0.5_f64.sin()];
            assert!(
                (request.target.x() - expected[0]).abs() < 1e-9
                    && (request.target.y() - expected[1]).abs() < 1e-9,
                "the target 1 m ahead of the robot frame, in odom: {request:?}"
            );
        }
    }

    // The refusals of a hold say when to send the goal again, that of a map
    // target says what to send, and that of resources names each the goal
    // waits for.
    #[test]
    fn the_refusals_say_what_to_do() {
        let no_cause = Causes {
            no_read: false,
            commanded: false,
            rejected: false,
            map: false,
            silent_link: false,
            silent_sensor: false,
            parts_outside: false,
            hold: Hold::None,
        };
        let message = |causes: Causes, frame: Option<&str>| {
            let admission = admission(causes);
            let data = goal(frame, 0.0);
            runtime()
                .block_on(async { admission.navigation(&data) })
                .expect_err("a refusal")
                .for_member(navigate_to_pose::FailureKind::parse)
                .1
        };
        let stop = message(
            Causes {
                hold: Hold::RobotStop,
                ..no_cause
            },
            None,
        );
        assert!(stop.ends_with(AFTER_A_HOLD), "{stop}");
        let silence = message(
            Causes {
                hold: Hold::BaseSilence,
                ..no_cause
            },
            None,
        );
        assert!(silence.ends_with(AFTER_A_HOLD), "{silence}");
        let mode = message(
            Causes {
                hold: Hold::BaseMode,
                ..no_cause
            },
            None,
        );
        assert!(
            mode.contains("damped") && mode.contains("emergency stop"),
            "{mode}"
        );
        let map = message(no_cause, Some(MAP_FRAME));
        assert_eq!(
            map,
            "no map is loaded: send the target in odom or the robot frame"
        );
        let sensor = message(
            Causes {
                silent_sensor: true,
                ..no_cause
            },
            None,
        );
        assert!(sensor.contains("the camera front"), "{sensor}");
        let every_resource = message(
            Causes {
                silent_link: true,
                silent_sensor: true,
                hold: Hold::BaseSilence,
                ..no_cause
            },
            None,
        );
        for cause in ["no drive_states for", "base_silence", "the camera front"] {
            assert!(every_resource.contains(cause), "{cause}: {every_resource}");
        }
    }

    // A target off the floor in the map frame is rejected before not_localized,
    // naming the part and what to send: a planar target in map or the robot
    // frame.
    #[test]
    fn a_map_target_off_the_floor_is_rejected_naming_the_part() {
        let refusal = off_the_floor(&GeometryError::OffFloor { z: 0.2 }, MAP_FRAME);
        assert_eq!(refusal.kind, FailureKind::Rejected);
        assert_eq!(
            refusal.message,
            "the target's position, placed in map, lies 0.2 m above the floor plane: send a \
             target in map or the robot frame with a position [x, y, 0] and an orientation \
             [0, 0, sin(yaw/2), cos(yaw/2)]"
        );
    }

    // A max_speed_m_s of 0 sets no cap and drives at the forward limit; a
    // speed inside the deadband, below 0 or above the limits is rejected.
    #[test]
    fn a_navigation_speed_is_held_to_the_limits_along_x() {
        let base = Base::parse(&answer(1.0), &config()).expect("a base");
        let speed = |max_speed_m_s| {
            max_speed(max_speed_m_s, &base).map(|cap| {
                (
                    cap.map(PositiveFinite::get),
                    navigation_speed(cap, &base).get(),
                )
            })
        };
        assert_eq!(speed(0.0), Ok((None, 0.5)));
        assert_eq!(speed(0.3), Ok((Some(0.3), 0.3)));
        for refused in [-0.1, 0.01, 0.51, f64::NAN] {
            assert!(max_speed(refused, &base).is_err(), "{refused}");
        }
    }

    // A straight drive whose end leaves the grid centered on the base is
    // rejected; a timeout of 0 sets no bound and one below 0 is rejected.
    #[test]
    fn a_straight_drive_ends_inside_the_grid_and_takes_its_timeout() {
        let base = PlanarPose::new(1.0, 2.0, 0.5).unwrap();
        let grid = config().grid;
        let center = [base.x(), base.y()];
        assert!(grid.square_holds(center, Direction::Backward.end_of(base, 2.0)));
        assert!(!grid.square_holds(center, Direction::Forward.end_of(base, 3.0)));
        assert_eq!(timeout(0.0), Ok(None));
        assert_eq!(timeout(1.5), Ok(Some(1_500_000_000)));
        assert!(timeout(-1.0).is_err());
    }
}
