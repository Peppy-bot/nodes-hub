//! Binary shell: tracing init and the runtime boot of the library's `setup`.

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    peppygen::NodeBuilder::new().run(realsense_d4xx::setup)
}
