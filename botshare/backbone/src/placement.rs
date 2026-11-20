//! The placement of each goal and setpoint by the robot's tree, at its timestamp, as
//! base_motion and drive_link place one. A move_base target is placed in odom and in the
//! robot frame through [`RobotTree::place`]: with no timestamp where its frame stands at
//! the latest time, and at a time where its frame stood then, through the root of its
//! frame's chain, so a robot-frame target from before a move names the same place on the
//! floor after it. A drive_base velocity and each drive_link setpoint are placed in the
//! robot frame at their own time, by the place of their frame in the robot frame then, so a
//! velocity in the robot frame stands in it at any time, as transforms looks a frame up in
//! itself.
//!
//! Each placement waits up to the lookup timeout on the node's clock for the tree to
//! answer, and a placement that fails gives the failure of the lookup, or the refusal of a
//! placed value no contract takes.

use std::sync::Arc;
use std::time::Duration;

use control_core::geometry::{Placed, Stamped, Twist};
use control_core::robot_tree::{PlaceError, PlacedValue, RobotTree};
use control_core::sync::wait_for;
use frame_tree::FrameName;

/// The robot's tree, the robot frame a velocity is placed in, the odometry frame a
/// target is placed in, and how long a placement waits for the tree.
pub struct Placer {
    tree: Arc<RobotTree>,
    robot_frame: FrameName,
    odometry_frame: FrameName,
    wait: Duration,
}

impl Placer {
    pub fn new(
        tree: Arc<RobotTree>,
        robot_frame: FrameName,
        odometry_frame: FrameName,
        wait: Duration,
    ) -> Self {
        Self {
            tree,
            robot_frame,
            odometry_frame,
            wait,
        }
    }

    /// `stamped` in the robot frame at the latest time, with the time of the robot
    /// frame's place.
    async fn place<T: Placed>(&self, stamped: &Stamped<T>) -> Result<PlacedValue<T>, PlaceError> {
        self.tree.place(stamped, &self.robot_frame, self.wait).await
    }

    /// [`Placer::place`] from a handler the generated code calls synchronously: the wait
    /// blocks in place on the runtime.
    pub fn place_now<T: Placed>(&self, stamped: &Stamped<T>) -> Result<PlacedValue<T>, PlaceError> {
        wait_for(self.place(stamped))
    }

    /// The velocity `stamped` in the robot frame at its own time: its vectors turned by
    /// the place of its frame in the robot frame then.
    pub async fn place_velocity(&self, stamped: &Stamped<Twist>) -> Result<Twist, PlaceError> {
        let found = self
            .tree
            .lookup_within(
                &self.robot_frame,
                stamped.frame(),
                stamped.time(),
                self.wait,
            )
            .await?;
        Ok(stamped.value().placed_by(&found.placement)?)
    }

    /// [`Placer::place_velocity`] from a handler the generated code calls synchronously.
    pub fn place_velocity_now(&self, stamped: &Stamped<Twist>) -> Result<Twist, PlaceError> {
        wait_for(self.place_velocity(stamped))
    }

    /// `stamped` in the odometry frame at the latest time, from a handler the generated
    /// code calls synchronously, as [`Placer::place_now`] places in the robot frame.
    pub fn place_in_odometry_now<T: Placed>(
        &self,
        stamped: &Stamped<T>,
    ) -> Result<PlacedValue<T>, PlaceError> {
        wait_for(self.tree.place(stamped, &self.odometry_frame, self.wait))
    }
}
