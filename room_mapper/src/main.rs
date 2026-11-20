//! room_mapper: maps rooms with a mobile base's 2-D lidar and keeps the base localized in
//! the loaded map. See README.md.

use control_core::fault::Fault;
use peppygen::{NodeBuilder, Result};
use room_mapper::setup;

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let fault = Fault::default();
    let raised = fault.clone();
    NodeBuilder::new().run(move |params, runner| setup(params, runner, raised))?;
    fault
        .take()
        .map_or(Ok(()), |reason| Err(peppygen::Error::Node(reason)))
}
