#![forbid(unsafe_code)]

use std::sync::Arc;

use peppygen::{NodeRunner, Parameters, Result};

mod service;

pub async fn setup(_args: Parameters, runner: Arc<NodeRunner>) -> Result<()> {
    let token = runner.cancellation_token().clone();
    tokio::spawn(async move {
        service::run(runner, token).await;
    });
    Ok(())
}
