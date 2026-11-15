//! rerun_viewer: draws a robot stack's camera, arms and planning scene, live, in the
//! Rerun web viewer. See README.md.

use std::sync::Arc;

use peppygen::{NodeBuilder, Result};
use rerun_viewer::canvas::RerunCanvas;
use rerun_viewer::config::Config;
use rerun_viewer::node::setup;

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    NodeBuilder::new().run(|params, runner| async move {
        let config = Config::parse(&params).map_err(|e| peppygen::Error::Node(e.into()))?;
        let canvas =
            RerunCanvas::serve(&config.viewer).map_err(|e| peppygen::Error::Node(e.into()))?;
        setup(config, runner, Arc::new(canvas)).await
    })
}
