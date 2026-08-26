#![forbid(unsafe_code)]

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    peppygen::NodeBuilder::new().run(openarm_ker::setup)?;

    // The runtime has returned, so the shutdown hooks (instance lock release)
    // have already run; exiting non-zero here makes the daemon record a task
    // that stopped on its own as failed instead of finished.
    if let Some(task) = openarm_ker::task_failed() {
        return Err(openarm_ker::NodeError::TaskStopped(task).into());
    }
    Ok(())
}
