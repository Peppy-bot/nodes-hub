//! motion_planner: its parameters, the robots it plans for, the grippers' fingers and
//! the other bodies the arms carry, the robot's measured joint positions and gripper
//! openings, the parsing of its requests, the robot as the map cuts it out of a depth
//! frame, the planning scene's members on the wire, the self model of the planned arm,
//! and the node that serves motion_planning on them.

pub mod config;
pub mod fingers;
pub mod history;
pub mod intake;
pub mod node;
pub mod request;
pub mod robot;
pub mod scene;
pub mod self_model;
