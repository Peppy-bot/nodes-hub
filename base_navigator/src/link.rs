//! drive_link as the leader reads it: each drive_states the base streams back,
//! where its odometry places the robot frame in odom, the velocity it executes,
//! when its odometry started, its mode, whether its limbs stand within its
//! footprint, and whether it follows the navigator. [`Link`] numbers each
//! state in arrival order and keeps the newest with its arrival on the node's
//! clock and the first state that reported the base not following since it
//! last followed, and the states of the last drive_link window that reported
//! an ending of a goal, so a goal reads each report once; [`run_listener`]
//! feeds it and, when the odometry starts over, empties the grid of the floor
//! and hands the new start to describe_base's reader.

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use control_core::drive_link::{FollowState, WINDOW_NS};
use control_core::geometry::{Pose, Twist};
use control_core::latched_log::LatchedLog;
use control_core::node_clock::NodeClock;
use control_core::planar_pose::PlanarPose;
use control_core::receiving::receive;
use control_core::stream_window::StreamWindow;
use control_core::sync::lock_ignoring_poison;
use control_core::twist::PlanarTwist;
use peppygen::NodeRunner;
use peppygen::paired_topics::base::drive_states;
use tokio::sync::watch;
use tracing::{error, info};

use crate::clock;
use crate::follow::{ODOMETRY_FRAME, ROBOT_FRAME};
use crate::grid::GridSlot;

/// The navigator's deadman on drive_states: the base counts silent once more
/// than 0.5 s pass with no drive_states, on the node's clock, which refuses a
/// goal and fails the goal in flight.
const DRIVE_STATES_DEADMAN: StreamWindow<i64> = StreamWindow::new(WINDOW_NS);

/// What the base can do, as base_state names its modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Damped,
    Posed,
    Transition,
    Upright,
    Standing,
    Moving,
    Fallen,
}

impl Mode {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "off" => Self::Off,
            "damped" => Self::Damped,
            "posed" => Self::Posed,
            "transition" => Self::Transition,
            "upright" => Self::Upright,
            "standing" => Self::Standing,
            "moving" => Self::Moving,
            "fallen" => Self::Fallen,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Damped => "damped",
            Self::Posed => "posed",
            Self::Transition => "transition",
            Self::Upright => "upright",
            Self::Standing => "standing",
            Self::Moving => "moving",
            Self::Fallen => "fallen",
        }
    }

    /// What brings a base in this mode to standing, as base_state names the
    /// modes; none in a mode in which the base takes velocity, standing or
    /// moving.
    pub fn to_standing(self) -> Option<&'static str> {
        match self {
            Self::Off | Self::Damped => {
                Some("release what holds the base, such as its emergency stop, and arm its drives")
            }
            Self::Posed | Self::Upright | Self::Transition => {
                Some("bring the base to standing with walking_robot's stand")
            }
            Self::Fallen => {
                Some("have a person right the base, then arm it with walking_robot's stand")
            }
            Self::Standing | Self::Moving => None,
        }
    }
}

/// One drive_states, parsed: where the odometry places the robot frame in
/// odom at `timestamp`, the velocity the base executes, and what the base
/// reports beside them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveState {
    pub timestamp: SystemTime,
    pub odometry_start: SystemTime,
    pub pose: PlanarPose,
    pub velocity: PlanarTwist,
    pub mode: Mode,
    pub parts_within_footprint: bool,
    pub follow: FollowState,
}

impl DriveState {
    /// Whether this drive_states reports an ending of a goal: a follow_state
    /// other than following, or parts_within_footprint false.
    fn reports_an_ending(&self) -> bool {
        self.follow != FollowState::Following || !self.parts_within_footprint
    }
}

/// A drive_states as the link took it: its number in arrival order, when it
/// arrived on the node's clock, and, while it reports a follow_state other
/// than following, the first drive_states that reported the base not
/// following since it last followed, under the same odometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Received {
    pub state: DriveState,
    pub seq: u64,
    pub arrived_ns: i64,
    pub left_following: Option<DriveState>,
}

impl Received {
    /// Whether this drive_states arrived more than 0.5 s before `now_ns`.
    pub fn silent_at(&self, now_ns: i64) -> bool {
        !DRIVE_STATES_DEADMAN.streams(Some(self.arrived_ns), now_ns)
    }
}

/// What a drive_states changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Taken {
    /// The odometry goes on.
    Continued,
    /// The odometry started over at this new start, or this is its first
    /// state.
    Started(SystemTime),
}

/// The first drive_states after a given one that reported each ending of a
/// goal a drive_states gives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reports {
    /// The first that reported a follow_state other than following.
    pub hold: Option<Received>,
    /// The first that reported parts_within_footprint false.
    pub limb_outside: Option<Received>,
}

/// drive_states as the navigator keeps them.
#[derive(Debug, Default)]
pub struct Link {
    newest: Option<Received>,
    /// How many drive_states the link took.
    taken: u64,
    /// The drive_states that reported an ending of a goal and arrived within
    /// the deadman's 0.5 s of the newest, oldest first.
    reports: VecDeque<Received>,
}

impl Link {
    pub fn newest(&self) -> Option<Received> {
        self.newest
    }

    /// The reports of the drive_states taken after the one numbered `after`.
    pub fn reports_after(&self, after: u64) -> Reports {
        let later = || self.reports.iter().filter(move |report| report.seq > after);
        Reports {
            hold: later()
                .find(|report| report.state.follow != FollowState::Following)
                .copied(),
            limb_outside: later()
                .find(|report| !report.state.parts_within_footprint)
                .copied(),
        }
    }

    /// Take `state`, arrived at `arrived_ns`.
    pub fn take(&mut self, state: DriveState, arrived_ns: i64) -> Taken {
        let previous = self.newest;
        let started = previous.is_none_or(|p| p.state.odometry_start != state.odometry_start);
        let left_following = match (state.follow, previous) {
            (FollowState::Following, _) => None,
            (_, Some(p)) if p.state.follow != FollowState::Following && !started => {
                p.left_following
            }
            _ => Some(state),
        };
        let received = Received {
            state,
            seq: self.taken,
            arrived_ns,
            left_following,
        };
        self.taken += 1;
        self.newest = Some(received);
        if state.reports_an_ending() {
            self.reports.push_back(received);
        }
        while self.reports.front().is_some_and(|report| {
            !DRIVE_STATES_DEADMAN.streams(Some(report.arrived_ns), arrived_ns)
        }) {
            self.reports.pop_front();
        }
        if started {
            Taken::Started(state.odometry_start)
        } else {
            Taken::Continued
        }
    }
}

/// One drive_states as the navigator takes it, or why it does not: its pose
/// of the robot frame in odom and its twist on the floor plane.
fn parse(message: &drive_states::Message) -> Result<DriveState, String> {
    if message.frame != ODOMETRY_FRAME || message.child_frame != ROBOT_FRAME {
        return Err(format!(
            "the pose of {:?} in {:?}, where drive_link gives {ROBOT_FRAME:?} in \
             {ODOMETRY_FRAME:?}",
            message.child_frame, message.frame
        ));
    }
    let pose = Pose::parse(message.pose.position, message.pose.orientation)
        .and_then(Pose::planar)
        .map_err(|e| format!("the pose: {e}"))?;
    let velocity = Twist::parse(
        message.twist.linear_velocity,
        message.twist.angular_velocity,
    )
    .and_then(Twist::planar)
    .map_err(|e| format!("the twist: {e}"))?;
    let mode = Mode::parse(&message.mode).ok_or_else(|| format!("the mode {:?}", message.mode))?;
    let follow = FollowState::parse(&message.follow_state).map_err(|e| e.to_string())?;
    Ok(DriveState {
        timestamp: message.timestamp,
        odometry_start: message.odometry_start_timestamp,
        pose,
        velocity,
        mode,
        parts_within_footprint: message.parts_within_footprint,
        follow,
    })
}

/// Receive drive_states until the node cancels: keep each in `link` with its
/// arrival on `clock`, and at each start of the odometry empty `grid` and
/// hand the start to `starts`.
pub async fn run_listener(
    runner: Arc<NodeRunner>,
    clock: NodeClock,
    link: Arc<Mutex<Link>>,
    grid: Arc<Mutex<GridSlot>>,
    starts: watch::Sender<Option<SystemTime>>,
) {
    let mut subscription = match drive_states::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("drive_states subscribe: {e}"),
    };
    let dropped = LatchedLog::warn("drive_states");
    receive(
        "drive_states",
        runner.cancellation_token().cancelled(),
        async move || subscription.next().await,
        async move |(_peer, message): (peppygen::PeerInfo, drive_states::Message)| {
            let arrived_ns = clock::now_ns(&clock);
            let state = match parse(&message) {
                Ok(state) => state,
                Err(reason) => {
                    dropped.failure(format!("dropping a state: {reason}"));
                    return ControlFlow::Continue(());
                }
            };
            dropped.success();
            let taken = lock_ignoring_poison(&link).take(state, arrived_ns);
            if let Taken::Started(start) = taken {
                info!("the base's odometry started at {start:?}: the grid starts empty");
                lock_ignoring_poison(&grid).empty();
                starts.send_replace(Some(start));
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::time::Duration;

    /// A drive_states of the odometry started at `start_s`, stamped at
    /// `stamp_s`, the robot frame at `x` along odom's x, at rest, reporting
    /// `follow`.
    pub(crate) fn state(start_s: u64, stamp_s: u64, x: f64, follow: FollowState) -> DriveState {
        DriveState {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(stamp_s),
            odometry_start: SystemTime::UNIX_EPOCH + Duration::from_secs(start_s),
            pose: PlanarPose::new(x, 0.0, 0.0).expect("a finite pose"),
            velocity: PlanarTwist::ZERO,
            mode: Mode::Standing,
            parts_within_footprint: true,
            follow,
        }
    }

    // A new odometry_start_timestamp starts the odometry over; the first
    // state is a start too.
    #[test]
    fn a_new_odometry_start_is_a_start() {
        let mut link = Link::default();
        let first = state(5, 10, 1.0, FollowState::Following);
        assert_eq!(link.take(first, 10), Taken::Started(first.odometry_start));
        assert_eq!(
            link.take(state(5, 20, 2.0, FollowState::Following), 20),
            Taken::Continued
        );
        let over = state(25, 30, 0.0, FollowState::Following);
        assert_eq!(link.take(over, 30), Taken::Started(over.odometry_start));
    }

    // The first state that reports the base not following stays the one a
    // cancelled goal measures, until the base follows again.
    #[test]
    fn the_first_state_that_leaves_following_is_kept_until_following() {
        let mut link = Link::default();
        link.take(state(5, 10, 1.0, FollowState::Following), 10);
        link.take(state(5, 20, 2.0, FollowState::RobotStop), 20);
        link.take(state(5, 30, 3.0, FollowState::FollowerGoal), 30);
        let left = link.newest().and_then(|l| l.left_following);
        assert_eq!(left.map(|s| s.pose.x()), Some(2.0));
        link.take(state(5, 40, 4.0, FollowState::Following), 40);
        assert_eq!(link.newest().and_then(|l| l.left_following), None);
    }

    // Each report of an ending after a given drive_states is read once a
    // later one replaced it, the first of each kind, and a report goes once
    // it lies more than the deadman's 0.5 s before the newest.
    #[test]
    fn the_first_report_of_each_ending_after_a_state_is_kept() {
        let mut link = Link::default();
        let following = |stamp_s| state(5, stamp_s, 0.0, FollowState::Following);
        link.take(state(5, 1, 0.0, FollowState::RobotStop), 0);
        link.take(following(2), 0);
        let admitted = link.newest().expect("a state").seq;
        link.take(state(5, 3, 0.0, FollowState::BaseMode), 0);
        let limb_outside = DriveState {
            parts_within_footprint: false,
            ..following(4)
        };
        link.take(limb_outside, 0);
        link.take(state(5, 5, 0.0, FollowState::RobotStop), 0);
        link.take(following(6), 0);
        let reports = link.reports_after(admitted);
        assert_eq!(
            reports.hold.map(|report| (report.state.follow, report.seq)),
            Some((FollowState::BaseMode, admitted + 1))
        );
        assert_eq!(
            reports.limb_outside.map(|report| report.state),
            Some(limb_outside)
        );
        let newest = link.newest().expect("a state").seq;
        assert_eq!(link.reports_after(newest), Reports::default());
        link.take(following(7), WINDOW_NS + 1);
        assert_eq!(link.reports_after(admitted), Reports::default());
    }

    #[test]
    fn a_state_older_than_half_a_second_is_silent() {
        let mut link = Link::default();
        link.take(state(5, 10, 0.0, FollowState::Following), 1_000);
        let newest = link.newest().expect("a state");
        assert!(!newest.silent_at(1_000 + WINDOW_NS));
        assert!(newest.silent_at(1_001 + WINDOW_NS));
    }

    #[test]
    fn every_mode_reads_back_its_name() {
        for mode in [
            Mode::Off,
            Mode::Damped,
            Mode::Posed,
            Mode::Transition,
            Mode::Upright,
            Mode::Standing,
            Mode::Moving,
            Mode::Fallen,
        ] {
            assert_eq!(Mode::parse(mode.as_str()), Some(mode));
        }
    }
}
