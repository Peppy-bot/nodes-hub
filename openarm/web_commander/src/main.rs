#![forbid(unsafe_code)]

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    peppygen::NodeBuilder::new().run(openarm_commander::setup)?;

    // The runtime has returned cleanly as far as it knows; a panel that died
    // is the reason the token was cancelled, so exit as failed with it.
    if let Some(fault) = openarm_commander::ui_failed() {
        return Err(openarm_commander::NodeError::Ui(fault).into());
    }
    Ok(())
}
