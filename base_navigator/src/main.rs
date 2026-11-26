//! Binary entry point for the base navigator: tracing, then hand the
//! library's [`base_navigator::setup`] to the node runtime with the node's
//! fault. All node logic lives in the library so the harness tests boot the
//! same entry point in-process.

#![forbid(unsafe_code)]

use control_core::fault::Fault;

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let fault = Fault::default();
    let raised = fault.clone();
    peppygen::NodeBuilder::new()
        .run(move |params, runner| base_navigator::setup(params, runner, raised))?;

    // The runtime has returned, so its shutdown hooks have run; exiting
    // non-zero here makes the daemon record a node that a fault ended as
    // failed.
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
