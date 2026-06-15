use peppygen::emitted_topics::joint_states;
use peppygen::exposed_actions::move_arm;
use peppygen::{NodeBuilder, Parameters, Result};
use peppylib::runtime::CancellationToken;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

async fn publish_joint_states(
    node_runner: Arc<peppygen::NodeRunner>,
    current_position: Arc<Mutex<[i32; 3]>>,
) {
    let token = node_runner.cancellation_token().clone();
    loop {
        let now = SystemTime::now();
        let positions = {
            let p = *current_position
                .lock()
                .expect("current_position lock poisoned");
            [p[0] as f64, p[1] as f64, p[2] as f64]
        };
        let velocities = [0.0, 0.0, 0.0];
        if let Err(e) = joint_states::emit(&node_runner, positions, velocities, now).await {
            eprintln!("[arm] emit joint_states error: {e:?}");
            break;
        }
        println!("[arm] published joint_states: positions={positions:.3?}");
        tokio::select! {
            _ = token.cancelled() => break,
            _ = tokio::time::sleep(Duration::from_millis(500)) => {}
        }
    }
}

async fn run_action(
    node_runner: Arc<peppygen::NodeRunner>,
    current_position: Arc<Mutex<[i32; 3]>>,
) -> Result<()> {
    println!("[arm] move_arm action handler started");
    let mut action = move_arm::ActionHandle::expose(&node_runner).await?;
    let token = node_runner.cancellation_token().clone();
    // Single arm per instance, so just one in-flight goal at a time.
    let busy: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

    loop {
        let busy_for_decider = Arc::clone(&busy);
        let next = tokio::select! {
            _ = token.cancelled() => break,
            next = action.handle_goal_next_request(move |request| {
                println!(
                    "[arm] received move_arm goal: {:?}",
                    request.data.desired_position
                );
                let mut flag = busy_for_decider.lock().expect("busy lock poisoned");
                if *flag {
                    Ok(move_arm::GoalResponse::reject(
                        "arm is already moving".to_string(),
                    ))
                } else {
                    *flag = true;
                    Ok(move_arm::GoalResponse::accept())
                }
            }) => next?,
        };
        let Some(ctx) = next else {
            println!("[arm] move_arm action handler closed");
            break;
        };
        let position = Arc::clone(&current_position);
        let busy_clone = Arc::clone(&busy);
        let goal_token = token.clone();
        tokio::spawn(async move {
            drive_goal(ctx, position, busy_clone, goal_token).await;
        });
    }
    Ok(())
}

async fn drive_goal(
    ctx: move_arm::GoalContext,
    current_position: Arc<Mutex<[i32; 3]>>,
    busy: Arc<Mutex<bool>>,
    token: CancellationToken,
) {
    let target = ctx.request().data.desired_position;
    let start = *current_position
        .lock()
        .expect("current_position lock poisoned");
    let duration = choose_action_duration();

    let outcome = tokio::select! {
        final_position = execute_goal(&ctx, start, target, duration, Arc::clone(&current_position)) => Some(final_position),
        _ = ctx.cancel_signal() => None,
        _ = token.cancelled() => None,
    };

    match outcome {
        Some(final_position) => {
            println!("[arm] move_arm completed at position: {final_position:?}");
            if let Err(e) = ctx.complete(final_position).await {
                eprintln!("[arm] complete error: {e:?}");
            }
        }
        None => {
            let last_known = *current_position
                .lock()
                .expect("current_position lock poisoned");
            println!("[arm] move_arm cancelled at position: {last_known:?}");
            if let Err(e) = ctx.complete_cancelled(last_known).await {
                eprintln!("[arm] complete_cancelled error: {e:?}");
            }
        }
    }

    *busy.lock().expect("busy lock poisoned") = false;
}

async fn execute_goal(
    ctx: &move_arm::GoalContext,
    start: [i32; 3],
    target: [i32; 3],
    duration: Duration,
    current_position: Arc<Mutex<[i32; 3]>>,
) -> [i32; 3] {
    let (steps, step_duration) = feedback_plan(duration);
    for step in 1..=steps {
        tokio::time::sleep(step_duration).await;
        let ratio = step as f32 / steps as f32;
        let current = interpolate_position(start, target, ratio);
        *current_position
            .lock()
            .expect("current_position lock poisoned") = current;
        let _ = ctx.publish_feedback(current).await;
    }
    target
}

fn choose_action_duration() -> Duration {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or_default();
    let millis = 1_000 + (nanos % 2_000) as u64;
    Duration::from_millis(millis)
}

fn feedback_plan(duration: Duration) -> (u32, Duration) {
    let total_ms = duration.as_millis().max(1);
    let steps = (total_ms / 200).max(1) as u32;
    let step_ms = (total_ms / steps as u128).max(1) as u64;
    (steps, Duration::from_millis(step_ms))
}

fn interpolate_position(start: [i32; 3], target: [i32; 3], ratio: f32) -> [i32; 3] {
    [
        lerp_i32(start[0], target[0], ratio),
        lerp_i32(start[1], target[1], ratio),
        lerp_i32(start[2], target[2], ratio),
    ]
}

fn lerp_i32(start: i32, target: i32, ratio: f32) -> i32 {
    let delta = (target - start) as f32;
    (start as f32 + delta * ratio).round() as i32
}

fn main() -> Result<()> {
    NodeBuilder::<Parameters>::new().run(|_args: Parameters, node_runner| async move {
        let current_position: Arc<Mutex<[i32; 3]>> = Arc::new(Mutex::new([0; 3]));
        let states_runner = Arc::clone(&node_runner);
        let states_position = Arc::clone(&current_position);
        let action_runner = Arc::clone(&node_runner);
        let action_position = Arc::clone(&current_position);

        tokio::spawn(publish_joint_states(states_runner, states_position));
        tokio::spawn(async move {
            if let Err(error) = run_action(action_runner, action_position).await {
                tracing::error!("move_arm action error: {error:?}");
            }
        });

        // Log when the shutdown/cancel signal is received so it is visible in
        // the node's stdout.
        node_runner.on_shutdown(async move {
            println!("[arm] Shutdown signal received");
        });
        Ok(())
    })
}
