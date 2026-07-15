//! Synthetic joint_states + joint_commands producer for smoke-testing
//! consumers of the generic robot contracts. Each joint follows a slow sine
//! wave; the command leads the measured state by a small phase so state and
//! action differ. Not for real use.

use std::time::{Duration, Instant};

use peppygen::emitted_topics::joint_commands::v1::joint_commands;
use peppygen::emitted_topics::joint_states::v1::joint_states;
use peppygen::{NodeBuilder, Parameters, Result};
use tracing::{error, info};

fn wave(joint: usize, t: f64, phase: f64) -> f64 {
    let freq = 0.2 + 0.05 * joint as f64;
    (t * freq * std::f64::consts::TAU + phase + joint as f64).sin()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    NodeBuilder::new().run(|params: Parameters, runner| async move {
        let joints = params.joint_count.max(1) as usize;
        let period = Duration::from_secs_f64(1.0 / params.rate_hz.max(1) as f64);
        info!("mock_joint_source: {joints} joints at {} Hz", params.rate_hz);

        let states_runner = runner.clone();
        tokio::spawn(async move {
            let publisher = match joint_states::declare_publisher(&states_runner).await {
                Ok(p) => p,
                Err(e) => {
                    error!("declare joint_states publisher: {e}");
                    return;
                }
            };
            let start = Instant::now();
            let mut ticker = tokio::time::interval(period);
            loop {
                ticker.tick().await;
                let t = start.elapsed().as_secs_f64();
                let positions: Vec<f64> = (0..joints).map(|j| wave(j, t, 0.0)).collect();
                let velocities: Vec<f64> = if params.report_velocities {
                    (0..joints).map(|j| wave(j, t, std::f64::consts::FRAC_PI_2)).collect()
                } else {
                    Vec::new()
                };
                let message = joint_states::build_message(positions, velocities, Vec::new());
                if let Ok(payload) = message
                    && let Err(e) = publisher.publish(payload).await
                {
                    error!("publish joint_states: {e}");
                }
            }
        });

        let commands_runner = runner.clone();
        tokio::spawn(async move {
            let publisher = match joint_commands::declare_publisher(&commands_runner).await {
                Ok(p) => p,
                Err(e) => {
                    error!("declare joint_commands publisher: {e}");
                    return;
                }
            };
            let start = Instant::now();
            let mut ticker = tokio::time::interval(period);
            loop {
                ticker.tick().await;
                let t = start.elapsed().as_secs_f64();
                // Command leads the state by a small phase so action != state.
                let positions: Vec<f64> = (0..joints).map(|j| wave(j, t, 0.15)).collect();
                let message = joint_commands::build_message(positions);
                if let Ok(payload) = message
                    && let Err(e) = publisher.publish(payload).await
                {
                    error!("publish joint_commands: {e}");
                }
            }
        });

        Ok(())
    })
}
