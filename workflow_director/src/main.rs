//! Binary shell: tracing init plus the runtime boot of the library's `setup`,
//! and the exit error of a fault that ended the node.

#![forbid(unsafe_code)]

use peppygen::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let fault = workflow_director::Fault::default();
    let raised = fault.clone();
    peppygen::NodeBuilder::new()
        .run(move |params, runner| workflow_director::setup(params, runner, raised))?;
    fault.end()
}
