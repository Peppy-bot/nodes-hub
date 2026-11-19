//! motion_planner: its parameters, the robots it plans for and their arms, each robot's own
//! body, the grippers' fingers and the other bodies the arms carry, the robot's measured
//! joint positions and gripper openings, the parsing of its requests, the robot as the map
//! cuts it out of a depth frame, the planning scene's members on the wire, the self model
//! of the planned arms of the robot its description describes, the node's clock and the
//! followers that fill the robot's tree of frames, and the node that serves
//! motion_planning on them.

mod clock;
pub mod config;
pub mod fingers;
mod follow;
pub mod history;
pub mod intake;
pub mod node;
pub mod own_body;
pub mod request;
pub mod robot;
pub mod scene;
pub mod self_model;
