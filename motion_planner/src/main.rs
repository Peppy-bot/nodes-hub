//! motion_planner: keeps a voxel map of what the perception camera sees and plans the
//! robot's arm paths on it. See README.md.

use control_core::fault::Fault;
use motion_planner::node::setup;
use peppygen::{NodeBuilder, Result};

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let fault = Fault::default();
    let raised = fault.clone();
    NodeBuilder::new().run(move |params, runner| setup(params, runner, raised))?;
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
