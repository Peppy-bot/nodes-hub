use peppygen::consumed_topics::joint_commands;
use peppygen::emitted_topics::joint_states;
use peppygen::{NodeBuilder, Parameters, Result};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

async fn publish_joint_states(node_runner: Arc<peppygen::NodeRunner>) {
    loop {
        let now = SystemTime::now();
        let secs = now
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let positions = [secs.sin(), (secs * 1.3).sin(), (secs * 0.7).sin()];
        let velocities = [0.1, 0.2, 0.1];
        if let Err(e) = joint_states::emit(&node_runner, positions, velocities, now).await {
            eprintln!("[arm] emit joint_states error: {e:?}");
            break;
        }
        println!("[arm] published joint_states: positions={positions:.3?}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn receive_joint_commands(node_runner: Arc<peppygen::NodeRunner>) {
    loop {
        match joint_commands::on_next_message_received(&node_runner, None, None).await {
            Ok((_instance_id, msg)) => {
                println!(
                    "[arm] received joint_commands: target_positions={:.3?} max_velocity={}",
                    msg.target_positions, msg.max_velocity
                );
            }
            Err(e) => {
                eprintln!("[arm] joint_commands subscription error: {e:?}");
                break;
            }
        }
    }
}

fn main() -> Result<()> {
    NodeBuilder::new().run(|_args: Parameters, node_runner| async move {
        tokio::spawn(publish_joint_states(Arc::clone(&node_runner)));
        tokio::spawn(receive_joint_commands(Arc::clone(&node_runner)));
        Ok(())
    })
}
