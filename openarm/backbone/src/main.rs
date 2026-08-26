//! Binary entry point for the bimanual backbone: tracing, then hand the
//! library's [`openarm_backbone::setup`] to the node runtime. All node logic
//! lives in `src/lib.rs` so the harness-driven integration tests can boot the
//! same entry point in-process.

#![forbid(unsafe_code)]

fn main() -> peppygen::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    peppygen::NodeBuilder::new().run(openarm_backbone::setup)
}
