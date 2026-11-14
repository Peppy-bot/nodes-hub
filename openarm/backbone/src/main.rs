//! Binary entry point for the bimanual backbone: tracing, then hand the
//! library's [`openarm_backbone::setup`] to the node runtime, and exit with the
//! fault that ended the node, if one did. All node logic lives in `src/lib.rs` so
//! the harness-driven integration tests can boot the same entry point in-process.

#![forbid(unsafe_code)]

use control_core::fault::Fault;

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let fault = Fault::default();
    let raised = fault.clone();
    peppygen::NodeBuilder::new()
        .run(move |params, runner| openarm_backbone::setup(params, runner, raised))?;
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
