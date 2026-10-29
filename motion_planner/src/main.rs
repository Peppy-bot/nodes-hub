//! motion_planner: keeps a voxel map of what the perception camera sees and plans the
//! robot's arm paths on it. See README.md.

use motion_planner::node::setup;
use peppygen::{NodeBuilder, Result};

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    NodeBuilder::new().run(setup)
}
