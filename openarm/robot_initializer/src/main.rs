#![forbid(unsafe_code)]

use peppygen::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    peppygen::NodeBuilder::new().run(openarm_robot_initializer::setup)
}
