#![forbid(unsafe_code)]

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    peppygen::NodeBuilder::new().run(openarm_arm::setup)?;

    // The runtime has returned, so the shutdown hooks (motor disable via the
    // control loop, lock release) have already run; exiting non-zero here
    // makes the daemon record a hard CAN fault as failed instead of finished.
    if openarm_arm::hard_fault_latched() {
        return Err(openarm_arm::NodeError::HardFault.into());
    }
    Ok(())
}
