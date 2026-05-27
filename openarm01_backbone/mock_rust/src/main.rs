use peppygen::consumed_topics::{left_robot_arm_joint_states, right_robot_arm_joint_states};
use peppygen::emitted_topics::joint_positions;
use peppygen::exposed_actions::move_arm;
use peppygen::{NodeBuilder, Parameters, Result};
use peppylib::runtime::CancellationToken;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ARM_ID_LEFT: u16 = 0;
const ARM_ID_RIGHT: u16 = 1;

#[derive(Debug, Clone, Copy)]
enum ActionOutcome {
    Completed([i32; 3]),
    Cancelled([i32; 3]),
    Closed,
}

#[derive(Debug, Clone, Copy)]
enum CancelPoll {
    None,
    Cancelled,
    Closed,
}

fn arm_side(arm_id: u16) -> &'static str {
    match arm_id {
        ARM_ID_LEFT => "Left",
        ARM_ID_RIGHT => "Right",
        _ => "Unknown",
    }
}

async fn next_goal(
    action: &mut move_arm::ActionHandle,
) -> Result<Option<move_arm::GoalRequest>> {
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

async fn check_cancel(action: &mut move_arm::ActionHandle) -> Result<CancelPoll> {
    match tokio::time::timeout(
        Duration::from_millis(0),
        action.handle_cancel_next_request(|_request| {
            Ok(move_arm::CancelResponse::new(true, None))
        }),
    )
    .await
    {
        Ok(result) => match result? {
            true => Ok(CancelPoll::Cancelled),
            false => Ok(CancelPoll::Closed),
        },
        Err(_) => Ok(CancelPoll::None),
    }
}

async fn run_action(
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
) -> Result<()> {
    println!("[controller] move_arm action handler started");
    let mut action = move_arm::ActionHandle::expose(&node_runner).await?;
    // Per-arm last position, indexed by arm_id (0=left, 1=right).
    let mut last_positions: [[i32; 3]; 2] = [[0, 0, 0], [0, 0, 0]];

    loop {
        if cancel_token.is_cancelled() {
            println!("[controller] move_arm shutdown requested");
            break;
        }

        let Some(goal_request) = next_goal(&mut action).await? else {
            println!("[controller] move_arm action handler closed");
            break;
        };

        let arm_id = goal_request.data.arm_id;
        let side = arm_side(arm_id);
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

        let arm_slot = (arm_id as usize).min(last_positions.len() - 1);
        let start_position = last_positions[arm_slot];
        let duration = choose_action_duration();

        let outcome = execute_goal(
            &mut action,
            &node_runner,
            arm_id,
            start_position,
            desired_position,
            duration,
        )
        .await?;

        let final_position = match outcome {
            ActionOutcome::Completed(position) => {
                println!("[controller] {side} arm completed at position: {position:?}");
                last_positions[arm_slot] = position;
                position
            }
            ActionOutcome::Cancelled(position) => {
                println!("[controller] {side} arm cancelled at position: {position:?}");
                last_positions[arm_slot] = position;
                position
            }
            ActionOutcome::Closed => {
                println!("[controller] {side} arm action closed");
                break;
            }
        };

        // Use timeout to avoid blocking forever if client doesn't request result
        let result_timeout = Duration::from_secs(10);
        let result_future = action.handle_result_next_request(move |_request| {
            Ok(move_arm::ResultResponse::new(final_position))
        });
        match tokio::time::timeout(result_timeout, result_future).await {
            Ok(Ok(true)) => {}
            Ok(Ok(false)) => {
                println!("[controller] {side} arm action handle closed");
                break;
            }
            Ok(Err(e)) => {
                eprintln!("[controller] {side} arm result request error: {e}");
                break;
            }
            Err(_) => {
                println!(
                    "[controller] {side} arm result request timed out, continuing to next goal"
                );
            }
        }
    }

    Ok(())
}

async fn execute_goal(
    action: &mut move_arm::ActionHandle,
    node_runner: &Arc<peppygen::NodeRunner>,
    arm_id: u16,
    start: [i32; 3],
    target: [i32; 3],
    duration: Duration,
) -> Result<ActionOutcome> {
    action.emit_feedback(start).await?;

    match check_cancel(action).await? {
        CancelPoll::Cancelled => return Ok(ActionOutcome::Cancelled(start)),
        CancelPoll::Closed => return Ok(ActionOutcome::Closed),
        CancelPoll::None => {}
    }

    let (steps, step_duration) = feedback_plan(duration);
    let mut current = start;

    for step in 1..=steps {
        tokio::time::sleep(step_duration).await;

        match check_cancel(action).await? {
            CancelPoll::Cancelled => return Ok(ActionOutcome::Cancelled(current)),
            CancelPoll::Closed => return Ok(ActionOutcome::Closed),
            CancelPoll::None => {}
        }

        let ratio = step as f32 / steps as f32;
        current = interpolate_position(start, target, ratio);
        let cmd_positions = current.map(|v| v as f64);
        let _ = joint_positions::emit(node_runner, arm_id, cmd_positions, 1.0).await;
        action.emit_feedback(current).await?;

        match check_cancel(action).await? {
            CancelPoll::Cancelled => return Ok(ActionOutcome::Cancelled(current)),
            CancelPoll::Closed => return Ok(ActionOutcome::Closed),
            CancelPoll::None => {}
        }
    }

    Ok(ActionOutcome::Completed(target))
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
