//! Binary shell: tracing init, the runtime boot of the library's `setup`, and
//! the non-zero exit a relay leg's death earns.

#![forbid(unsafe_code)]

use peppygen::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    peppygen::NodeBuilder::new().run(sim_rgbd_camera::setup)?;
    if sim_rgbd_camera::leg_died() {
        return Err(std::io::Error::other("a relay leg died; failing the instance").into());
    }
    Ok(())
}
