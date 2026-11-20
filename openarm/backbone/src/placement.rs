//! The placement of each command's point or pose in the robot frame by the robot's tree,
//! at the command's timestamp, as the contracts place a command: a move_arm goal, a
//! check_arm_move request and the points of check_positions, all of one request by one
//! placement, and each pose setpoint of pose_link, each through [`RobotTree::place`]. A
//! value with no timestamp is placed where its frame stands at the latest time. A value at a
//! time is placed where its frame stood then, through the root of its frame's chain, so a
//! robot-frame target or setpoint from before a base move names the same place on the floor
//! after it, as limb_motion and pose_link place one.
//!
//! Each placement waits up to the lookup timeout on the node's clock for the tree to
//! answer, and a placement that fails gives the failure of the lookup, or the refusal of a
//! placed value no contract takes.

use std::sync::Arc;
use std::time::Duration;

use control_core::geometry::{Placed, Stamped};
use control_core::robot_tree::{PlaceError, PlacedValue, RobotTree};
use control_core::sync::wait_for;
use frame_tree::FrameName;

/// The robot's tree, the robot frame every command is placed in, and how long a
/// placement waits for the tree.
pub struct Placer {
    tree: Arc<RobotTree>,
    robot_frame: FrameName,
    wait: Duration,
}

impl Placer {
    pub fn new(tree: Arc<RobotTree>, robot_frame: FrameName, wait: Duration) -> Self {
        Self {
            tree,
            robot_frame,
            wait,
        }
    }

    /// The robot frame, the root link of the robot's URDF, which every report names.
    pub fn robot_frame(&self) -> &FrameName {
        &self.robot_frame
    }

    /// The frame a command names: `frame`, or the robot frame for a command that
    /// names none, as the contracts read a command with no frame.
    pub fn frame_of<'a>(&'a self, frame: Option<&'a str>) -> &'a str {
        frame.unwrap_or(self.robot_frame.as_str())
    }

    /// `stamped` in the robot frame at the latest time, with the time of the robot
    /// frame's place.
    pub async fn place<T: Placed>(
        &self,
        stamped: &Stamped<T>,
    ) -> Result<PlacedValue<T>, PlaceError> {
        self.tree.place(stamped, &self.robot_frame, self.wait).await
    }

    /// [`Placer::place`] from a handler the generated code calls synchronously: the wait
    /// blocks in place on the runtime.
    pub fn place_now<T: Placed>(&self, stamped: &Stamped<T>) -> Result<PlacedValue<T>, PlaceError> {
        wait_for(self.place(stamped))
    }
}
