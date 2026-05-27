use peppygen::consumed_topics::{left_robot_arm_joint_states, right_robot_arm_joint_states};
use peppygen::emitted_topics::joint_positions;
use peppygen::exposed_actions::move_arm;
use peppygen::{NodeBuilder, Parameters, Result};
use peppylib::runtime::CancellationToken;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

const ARM_ID_LEFT: u16 = 0;
const ARM_ID_RIGHT: u16 = 1;
const ARM_COUNT: usize = 2;

fn arm_side(arm_id: u16) -> &'static str {
    match arm_id {
        ARM_ID_LEFT => "Left",
        ARM_ID_RIGHT => "Right",
        _ => "Unknown",
    }
}

fn arm_slot(arm_id: u16) -> usize {
    (arm_id as usize).min(ARM_COUNT - 1)
}

async fn next_goal(action: &mut move_arm::ActionHandle) -> Result<Option<move_arm::GoalRequest>> {
    let goal_holder = Arc::new(Mutex::new(None));
    let goal_holder_clone = Arc::clone(&goal_holder);
    let handled = action
        .handle_goal_next_request(move |request| {
            *goal_holder_clone.lock().expect("goal lock poisoned") = Some(request);
            Ok(move_arm::GoalResponse::new(true))
        })
        .await?;
    if !handled {
        return Ok(None);
    }
    Ok(goal_holder.lock().expect("goal lock poisoned").take())
}

async fn run_action(
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
) -> Result<()> {
    println!("[controller] move_arm action handler started");
    let mut action = move_arm::ActionHandle::expose(&node_runner).await?;
    let last_positions: Arc<Mutex<[[i32; 3]; ARM_COUNT]>> =
        Arc::new(Mutex::new([[0; 3]; ARM_COUNT]));

    // Per-arm goal channels: dispatcher forwards goals to the worker for the
    // matching arm_id so left and right are processed concurrently.
    let (arm_txs, arm_rxs): (Vec<_>, Vec<_>) = (0..ARM_COUNT)
        .map(|_| mpsc::unbounded_channel::<move_arm::GoalRequest>())
        .unzip();
    let (results_tx, mut results_rx) = mpsc::unbounded_channel::<[i32; 3]>();

    for (arm_id, rx) in arm_rxs.into_iter().enumerate() {
        let runner = Arc::clone(&node_runner);
        let positions = Arc::clone(&last_positions);
        let tx = results_tx.clone();
        tokio::spawn(async move {
            arm_worker(runner, arm_id as u16, rx, positions, tx).await;
        });
    }
    drop(results_tx);

    let mut pending_results: VecDeque<[i32; 3]> = VecDeque::new();
    let mut workers_alive = ARM_COUNT;

    loop {
        if cancel_token.is_cancelled() {
            println!("[controller] move_arm shutdown requested");
            break;
        }

        if pending_results.is_empty() {
            if workers_alive == 0 {
                // No workers left to produce results; only accept new goals
                // (though with no workers, goals won't progress; this branch
                // mostly exists to gracefully wind down on shutdown).
                match next_goal(&mut action).await? {
                    Some(goal) => {
                        dispatch_goal(&arm_txs, goal);
                    }
                    None => {
                        println!("[controller] move_arm action handler closed");
                        break;
                    }
                }
                continue;
            }
            tokio::select! {
                goal_outcome = next_goal(&mut action) => {
                    match goal_outcome? {
                        Some(goal) => dispatch_goal(&arm_txs, goal),
                        None => {
                            println!("[controller] move_arm action handler closed");
                            break;
                        }
                    }
                }
                maybe_result = results_rx.recv() => {
                    match maybe_result {
                        Some(final_position) => pending_results.push_back(final_position),
                        None => workers_alive = 0,
                    }
                }
            }
        } else {
            // Hold a pending result; respond to the next result request.
            // FIFO matches the order the brain typically issues get_result
            // calls when both arms complete concurrently.
            let final_position = pending_results[0];
            let result_future = action.handle_result_next_request(move |_request| {
                Ok(move_arm::ResultResponse::new(final_position))
            });
            match tokio::time::timeout(Duration::from_secs(10), result_future).await {
                Ok(Ok(true)) => {
                    pending_results.pop_front();
                }
                Ok(Ok(false)) => {
                    println!("[controller] move_arm result stream closed");
                    break;
                }
                Ok(Err(e)) => {
                    eprintln!("[controller] result request error: {e}");
                    break;
                }
                Err(_) => {
                    println!(
                        "[controller] result request timed out, discarding final_position={:?}",
                        final_position
                    );
                    pending_results.pop_front();
                }
            }
        }
    }

    // Closing arm_txs signals workers to exit; results_rx then closes on its
    // own once each worker drops its results_tx clone.
    drop(arm_txs);
    Ok(())
}

fn dispatch_goal(
    arm_txs: &[mpsc::UnboundedSender<move_arm::GoalRequest>],
    goal: move_arm::GoalRequest,
) {
    let slot = arm_slot(goal.data.arm_id);
    if let Err(e) = arm_txs[slot].send(goal) {
        eprintln!("[controller] arm worker channel closed: {e}");
    }
}

async fn arm_worker(
    node_runner: Arc<peppygen::NodeRunner>,
    arm_id: u16,
    mut rx: mpsc::UnboundedReceiver<move_arm::GoalRequest>,
    last_positions: Arc<Mutex<[[i32; 3]; ARM_COUNT]>>,
    results_tx: mpsc::UnboundedSender<[i32; 3]>,
) {
    let side = arm_side(arm_id);
    let slot = arm_slot(arm_id);
    while let Some(goal_request) = rx.recv().await {
        let desired_position = goal_request.data.desired_position;
        println!("[controller] {side} arm received goal: {desired_position:?}");

        let cmd_positions = desired_position.map(|v| v as f64);
        if let Err(e) = joint_positions::emit(&node_runner, arm_id, cmd_positions, 1.0).await {
            eprintln!("[controller] {side} emit joint_positions error: {e:?}");
        } else {
            println!(
                "[controller] {side} published joint_positions: arm_id={arm_id} target={cmd_positions:.3?} max_vel=1.0"
            );
        }

        let start_position = last_positions.lock().expect("last_positions lock poisoned")[slot];
        let duration = choose_action_duration();

        let final_position = execute_goal(
            &node_runner,
            arm_id,
            start_position,
            desired_position,
            duration,
        )
        .await;

        println!("[controller] {side} arm completed at position: {final_position:?}");
        last_positions.lock().expect("last_positions lock poisoned")[slot] = final_position;

        if results_tx.send(final_position).is_err() {
            // Main loop has shut down; stop pulling new goals.
            break;
        }
    }
}

async fn execute_goal(
    node_runner: &Arc<peppygen::NodeRunner>,
    arm_id: u16,
    start: [i32; 3],
    target: [i32; 3],
    duration: Duration,
) -> [i32; 3] {
    let (steps, step_duration) = feedback_plan(duration);

    for step in 1..=steps {
        tokio::time::sleep(step_duration).await;
        let ratio = step as f32 / steps as f32;
        let current = interpolate_position(start, target, ratio);
        let cmd_positions = current.map(|v| v as f64);
        let _ = joint_positions::emit(node_runner, arm_id, cmd_positions, 1.0).await;
    }

    target
}

fn choose_action_duration() -> Duration {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos())
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
    NodeBuilder::<Parameters>::new().run(|_args, node_runner| async move {
        let left_states_runner = Arc::clone(&node_runner);
        let right_states_runner = Arc::clone(&node_runner);
        let action_runner = Arc::clone(&node_runner);
        let action_cancel_token = node_runner.cancellation_token().clone();

        tokio::spawn(async move {
            loop {
                match left_robot_arm_joint_states::on_next_message_received(
                    &left_states_runner,
                    None,
                )
                .await
                {
                    Ok((_id, msg)) => println!(
                        "[controller] left joint_states update: positions={:.3?} velocities={:.3?}",
                        msg.positions, msg.velocities
                    ),
                    Err(e) => {
                        eprintln!("[controller] left joint_states subscription closed: {e:?}");
                        break;
                    }
                }
            }
        });

        tokio::spawn(async move {
            loop {
                match right_robot_arm_joint_states::on_next_message_received(
                    &right_states_runner,
                    None,
                )
                .await
                {
                    Ok((_id, msg)) => println!(
                        "[controller] right joint_states update: positions={:.3?} velocities={:.3?}",
                        msg.positions, msg.velocities
                    ),
                    Err(e) => {
                        eprintln!("[controller] right joint_states subscription closed: {e:?}");
                        break;
                    }
                }
            }
        });

        tokio::spawn(async move {
            if let Err(error) = run_action(action_runner, action_cancel_token).await {
                tracing::error!("move_arm action error: {error:?}");
            }
        });

        Ok(())
    })
}
