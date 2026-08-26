//! Binary shell: tracing init plus the runtime boot of the library's `setup`.

#![forbid(unsafe_code)]

use peppygen::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    peppygen::NodeBuilder::new().run(openarm_gripper_sim::setup)
}
