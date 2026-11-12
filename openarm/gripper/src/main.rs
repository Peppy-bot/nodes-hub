#![forbid(unsafe_code)]

use control_core::fault::Fault;

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let fault = Fault::default();
    let raised = fault.clone();
    peppygen::NodeBuilder::new()
        .run(move |params, runner| openarm_gripper::setup(params, runner, raised))?;

    // The runtime has returned, so the shutdown hooks (motor disable, lock
    // release) have already run; exiting non-zero here makes the daemon
    // record a node that a fault ended, a hard CAN fault among them, as
    // failed instead of finished.
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
