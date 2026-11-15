#![forbid(unsafe_code)]

use control_core::fault::Fault;

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let fault = Fault::default();
    let raised = fault.clone();
    peppygen::NodeBuilder::new()
        .run(move |params, runner| openarm_ker::setup(params, runner, raised))?;

    // The runtime has returned, so the shutdown hooks (instance lock release)
    // have already run; exiting non-zero here makes the daemon record a node
    // that a fault ended as failed instead of finished.
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
