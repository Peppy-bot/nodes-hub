use peppygen::{NodeBuilder, Result};

fn main() -> Result<()> {
    // INFO by default, but honour RUST_LOG so the device-open path's debug
    // lines (resolved index, validation, requested format) can be turned on
    // when a camera refuses to open.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    NodeBuilder::new().run(uvc_camera_linux::setup)
}
