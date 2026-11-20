//! What the node reads of the robot, filled by its listeners and read by its members and
//! its worker: the base's odometry and its facts at the newest odometry start, read from
//! describe_base, and the lidar's newest scan and its place on the base.

use std::sync::Mutex;

use control_core::node_clock::NodeClock;
use control_core::sync::lock_ignoring_poison;
use frame_tree::Transform;
use navigation_core::PaddedFootprint;
use tokio::sync::{Notify, watch};

use crate::lidar::Arrival;
use crate::odometry::{Odometry, OdometryStart};

/// What describe_base says of the base at an odometry start.
#[derive(Debug, Clone)]
pub struct BaseFacts {
    /// Whether its odometry_source names commanded odometry.
    pub commanded: bool,
    /// Its footprint grown by the node's padding.
    pub footprint: PaddedFootprint,
}

/// The read of describe_base at the newest odometry start.
#[derive(Debug, Clone)]
pub enum FactsRead {
    /// No base_states has arrived yet.
    NoOdometry,
    /// The read of the start runs.
    Reading(OdometryStart),
    Read(BaseFacts),
    /// The read did not answer within the source timeout, or its answer was refused; the
    /// text says why.
    Unanswered(String),
}

/// Where the lidar stands on the base.
#[derive(Debug, Clone)]
pub enum Mount {
    /// No scan has named its frame yet.
    Unknown,
    /// The lookup of the frame in base_footprint failed; the text names its failure.
    Failing(String),
    Placed {
        frame: String,
        base_from_lidar: Transform,
    },
}

/// The node's inputs.
#[derive(Debug)]
pub struct Inputs {
    pub clock: NodeClock,
    pub odometry: Mutex<Odometry>,
    /// When the newest base_states arrived (ns, the node's clock, no earlier than its
    /// timestamp).
    pub base_arrived_ns: Mutex<Option<u64>>,
    pub facts: watch::Sender<FactsRead>,
    /// When the newest scan arrived (ns, the node's clock, no earlier than its timestamp).
    pub scan_arrived_ns: Mutex<Option<u64>>,
    /// The number of scans arrived.
    scans_arrived: Mutex<Arrival>,
    pub mount: Mutex<Mount>,
    /// Told of each change that a scan waiting for its odometry or for describe_base reads.
    pub changed: Notify,
}

impl Inputs {
    pub fn new(clock: NodeClock) -> Self {
        Self {
            clock,
            odometry: Mutex::default(),
            base_arrived_ns: Mutex::default(),
            facts: watch::Sender::new(FactsRead::NoOdometry),
            scan_arrived_ns: Mutex::default(),
            scans_arrived: Mutex::default(),
            mount: Mutex::new(Mount::Unknown),
            changed: Notify::new(),
        }
    }

    /// Counts a scan arrived: its place in the order of arrival.
    pub fn scan_arrives(&self) -> Arrival {
        let mut arrived = lock_ignoring_poison(&self.scans_arrived);
        arrived.0 += 1;
        *arrived
    }

    /// The number of scans arrived now.
    pub fn scans_arrived(&self) -> Arrival {
        *lock_ignoring_poison(&self.scans_arrived)
    }

    /// When a message stamped `stamp_ns` arrived (ns, the node's clock): the node's clock
    /// now, or the message's own time while the node's clock has not reached it. A message
    /// arrives no earlier than its timestamp, and a simulation's clock reaches the node on a
    /// topic of its own, behind the messages stamped on it.
    pub fn arrival_of(&self, stamp_ns: i64) -> u64 {
        self.clock
            .now_ns()
            .max(u64::try_from(stamp_ns).unwrap_or(0))
    }

    /// The time (ns) since `arrived`, none for no arrival.
    fn since(&self, arrived: &Mutex<Option<u64>>) -> Option<u64> {
        let now = self.clock.now_ns();
        lock_ignoring_poison(arrived).map(|at| now.saturating_sub(at))
    }

    /// The time (ns) since the newest base_states arrived.
    pub fn since_base(&self) -> Option<u64> {
        self.since(&self.base_arrived_ns)
    }

    /// The time (ns) since the newest scan arrived.
    pub fn since_scan(&self) -> Option<u64> {
        self.since(&self.scan_arrived_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_arrives_at_the_clock_or_at_its_own_time_when_the_clock_lags_it() {
        let (clock, feed) = NodeClock::simulation();
        feed.tick(5_000);
        let inputs = Inputs::new(clock);
        assert_eq!(
            inputs.arrival_of(3_000),
            5_000,
            "measured before the clock's time"
        );
        assert_eq!(
            inputs.arrival_of(9_000),
            9_000,
            "stamped past the lagging clock"
        );
        *lock_ignoring_poison(&inputs.scan_arrived_ns) = Some(inputs.arrival_of(9_000));
        assert_eq!(
            inputs.since_scan(),
            Some(0),
            "heard while the clock catches up"
        );
        feed.tick(10_000);
        assert_eq!(inputs.since_scan(), Some(1_000));
    }

    #[test]
    fn scans_count_in_their_order_of_arrival() {
        let (clock, _feed) = NodeClock::simulation();
        let inputs = Inputs::new(clock);
        assert_eq!(inputs.scans_arrived(), Arrival(0));
        assert_eq!(
            [inputs.scan_arrives(), inputs.scan_arrives()],
            [Arrival(1), Arrival(2)]
        );
        assert_eq!(inputs.scans_arrived(), Arrival(2));
    }
}
