//! Whether the base follows its leader of drive_link, by drive_link's rules,
//! on the node's clock: the arrival of each setpoint counts, never its stamp.
//!
//! The leader streams from each setpoint that arrives until more than
//! [`drive_link::WINDOW_NS`] pass with no setpoint. The base follows the leader's
//! setpoints while no hold keeps the base from it; a setpoint that arrives
//! during a hold is dropped, and counts for every rule here. The holds, in
//! the order drive_states names the first that holds:
//!
//! - base silence: the wheels' silence holds the base; when the leader
//!   streamed as the silence started, the hold lasts on after it ends until
//!   more than [`drive_link::WINDOW_NS`] pass with no setpoint. The node
//!   senses no state of the base's drives, so its mode is standing or moving
//!   while the wheels report, and drive_link's base_mode never holds.
//! - robot stop: from robot_stop's stop until more than [`drive_link::WINDOW_NS`]
//!   pass, from the stop on, with no setpoint.
//! - follower goal: a goal of base_motion holds the base while it runs; the
//!   caller says whether one does.

use control_core::drive_link::{self, FollowState};

/// The leader's stream and the holds that wait on it.
#[derive(Clone, Debug)]
pub struct Follower {
    /// When the leader's newest setpoint arrived (ns on the node's clock).
    last_ns: Option<i64>,
    /// Whether the wheels are silent: before their first measurement and
    /// while their newest is too old.
    silent: bool,
    /// Whether the silence's hold lasts on until the leader's stream ends:
    /// the leader streamed as the silence started.
    silence_lingers: bool,
    /// When the robot stop that holds the base came (ns), while it holds.
    stop_ns: Option<i64>,
}

impl Default for Follower {
    /// Before the wheels' first measurement: silent, with no leader heard.
    fn default() -> Self {
        Self {
            last_ns: None,
            silent: true,
            silence_lingers: false,
            stop_ns: None,
        }
    }
}

impl Follower {
    /// Whether the leader streams at `now_ns`.
    pub fn streams_at(&self, now_ns: i64) -> bool {
        drive_link::LEADER_STREAM.streams(self.last_ns, now_ns)
    }

    /// Ends each hold whose pause has passed by `now_ns`, for good: a later
    /// setpoint starts no hold again.
    fn release(&mut self, now_ns: i64) {
        if let Some(stop_ns) = self.stop_ns {
            let quiet_from = self.last_ns.map_or(stop_ns, |last_ns| last_ns.max(stop_ns));
            if !drive_link::LEADER_STREAM.streams(Some(quiet_from), now_ns) {
                self.stop_ns = None;
            }
        }
        if self.silence_lingers && !self.silent && !self.streams_at(now_ns) {
            self.silence_lingers = false;
        }
    }

    /// What drive_states tells the leader at `now_ns`, `goal_holds` saying
    /// whether a goal of the node holds the base.
    pub fn state(&mut self, now_ns: i64, goal_holds: bool) -> FollowState {
        self.release(now_ns);
        if self.silent || self.silence_lingers {
            return FollowState::BaseSilence;
        }
        if self.stop_ns.is_some() {
            return FollowState::RobotStop;
        }
        if goal_holds {
            return FollowState::FollowerGoal;
        }
        FollowState::Following
    }

    /// A setpoint of the leader arrived at `now_ns`: whether the base
    /// follows it, by [`Follower::state`].
    pub fn setpoint(&mut self, now_ns: i64, goal_holds: bool) -> bool {
        self.release(now_ns);
        self.last_ns = Some(now_ns);
        self.state(now_ns, goal_holds) == FollowState::Following
    }

    /// The wheels fell silent at `now_ns`: the hold lasts on after the
    /// silence when the leader streams now.
    pub fn silence_starts(&mut self, now_ns: i64) {
        self.release(now_ns);
        if !self.silent {
            self.silence_lingers = self.streams_at(now_ns);
            self.silent = true;
        }
    }

    /// The wheels report again at `now_ns`.
    pub fn silence_ends(&mut self, now_ns: i64) {
        self.silent = false;
        self.release(now_ns);
    }

    /// The robot stop at `now_ns`: whether the leader streamed at it.
    pub fn stop(&mut self, now_ns: i64) -> bool {
        self.release(now_ns);
        self.stop_ns = Some(now_ns);
        self.streams_at(now_ns)
    }

    /// The node's clock went back to `now_ns`: the leader's newest setpoint
    /// and the robot stop, when they came after it, take `now_ns` as their
    /// time, so the stream and each hold last on from it.
    pub fn clock_went_back(&mut self, now_ns: i64) {
        self.last_ns = self.last_ns.map(|last_ns| last_ns.min(now_ns));
        self.stop_ns = self.stop_ns.map(|stop_ns| stop_ns.min(now_ns));
    }
}

#[cfg(test)]
mod tests;
