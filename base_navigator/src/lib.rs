//! base_navigator: drives a robot's mobile base to a pose around what its
//! 2-D lidars and depth cameras see, turns it in place, and drives it straight
//! ahead or back clear of the same, as Nav2's NavigateToPose, Spin,
//! DriveOnHeading and BackUp drive a base, on navigation_core's ports of Nav2.
//! It implements base_navigation and leads the base over drive_link.
//!
//! - [`link`]: drive_states, the base's odometry, mode and whether it
//!   follows.
//! - [`base`]: describe_base's answer, read at each start of the odometry.
//! - [`follow`]: the robot's tree, from transforms and static_transforms.
//! - [`sensors`], [`cameras`], [`intake`], [`grid`]: the depth cameras and the
//!   2-D lidars fixed to the base, in one list, a depth camera's own
//!   description, their frames and scans placed by the tree, and
//!   navigation_core's `LiveObstacles`, the grid of the floor in odom, with its
//!   one occupancy query and the stop.
//! - [`admission`]: a goal checked in base_navigation's order.
//! - [`drive`], [`following`], [`recovery`], [`moves`], [`planning`]: a
//!   navigation's tries and recoveries, its follow, the turns and straight
//!   drives, the moves, and the planner on a task of its own, each a function
//!   of its inputs and of time.
//! - [`control`]: the loop that steps the goal on the node's clock, ends it by
//!   what drive_states reports, and streams the release burst.
//! - [`actions`], [`services`]: the four actions and clear_grid.
//!
//! The library crate exists so the harness tests boot the node in-process:
//! [`setup`] is the node's whole entry point, and `src/main.rs` only hands it
//! to `NodeBuilder::run`; [`setup_with`] takes the [`Planner`] whose
//! [`PlanGate`] holds a plan running.

#![forbid(unsafe_code)]

mod actions;
mod admission;
mod base;
mod cameras;
mod clock;
mod config;
mod control;
mod drive;
mod failure;
mod follow;
mod following;
mod grid;
mod intake;
mod link;
mod moves;
mod node;
mod planning;
mod recovery;
mod sensors;
mod services;

pub use node::{NodeError, setup, setup_with};
pub use planning::{PlanGate, Planner};
