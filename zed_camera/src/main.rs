//! Stereolabs ZED as an rgbd_camera producer without the ZED SDK. The
//! binary shell: tracing init and the runtime boot of [`node::setup`].

#[cfg(not(target_os = "linux"))]
compile_error!("zed_camera captures over V4L2 and builds for Linux only");

mod node;

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    peppygen::NodeBuilder::new().run(node::setup)
}
