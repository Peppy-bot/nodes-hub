//! transform_server: holds a robot's tree of frames and answers lookups in it. See
//! README.md.

use control_core::fault::Fault;
use peppygen::{NodeBuilder, Result};
use transform_server::setup;

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let fault = Fault::default();
    let raised = fault.clone();
    NodeBuilder::new().run(move |params, runner| setup(params, runner, raised))?;
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
