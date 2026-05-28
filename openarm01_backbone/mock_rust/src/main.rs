use peppygen::consumed_topics::{left_robot_arm_joint_states, right_robot_arm_joint_states};
use peppygen::emitted_topics::joint_positions;
use peppygen::exposed_actions::move_arm;
use peppygen::{NodeBuilder, Parameters, Result};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

async fn run_action(node_runner: Arc<peppygen::NodeRunner>) -> Result<()> {
    println!("[controller] move_arm action handler started");
    let mut action = move_arm::ActionHandle::expose(&node_runner).await?;
    let last_positions: Arc<Mutex<[[i32; 3]; ARM_COUNT]>> =
        Arc::new(Mutex::new([[0; 3]; ARM_COUNT]));
    let busy_arms: Arc<Mutex<HashSet<u16>>> = Arc::new(Mutex::new(HashSet::new()));

    loop {
        let busy_for_decider = Arc::clone(&busy_arms);
        let next = action
            .handle_goal_next_request(move |request| {
                let side = arm_side(request.data.arm_id);
                println!(
                    "[controller] {side} arm received goal: {:?}",
                    request.data.desired_position
                );
                // Reject a second goal for the same arm; drive_goal releases the slot when done.
                if busy_for_decider
                    .lock()
                    .expect("busy lock poisoned")
                    .insert(request.data.arm_id)
                {
                    Ok(move_arm::GoalResponse::accept())
                } else {
                    Ok(move_arm::GoalResponse::reject(format!(
                        "arm {} is already moving",
                        request.data.arm_id
                    )))
                }
            })
            .await?;
        let Some(ctx) = next else {
            println!("[controller] move_arm action handler closed");
            break;
        };
        let runner = Arc::clone(&node_runner);
        let positions = Arc::clone(&last_positions);
        let busy = Arc::clone(&busy_arms);
        tokio::spawn(async move {
            drive_goal(runner, ctx, positions, busy).await;
        });
    }
    Ok(())
}

async fn drive_goal(
    node_runner: Arc<peppygen::NodeRunner>,
    ctx: move_arm::GoalContext,
    last_positions: Arc<Mutex<[[i32; 3]; ARM_COUNT]>>,
    busy_arms: Arc<Mutex<HashSet<u16>>>,
) {
    let arm_id = ctx.request().data.arm_id;
    let side = arm_side(arm_id);
    let slot = arm_slot(arm_id);
    let desired_position = ctx.request().data.desired_position;

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
    // execute_goal updates this in place; the cancel branch reads the last stepped value.
    let current_position: Arc<Mutex<[i32; 3]>> = Arc::new(Mutex::new(start_position));

    let outcome = tokio::select! {
        final_position = execute_goal(
            &node_runner,
            &ctx,
            arm_id,
            start_position,
            desired_position,
            duration,
            Arc::clone(&current_position),
        ) => Some(final_position),
        _ = ctx.cancel_signal() => None,
    };

    match outcome {
        Some(final_position) => {
            println!("[controller] {side} arm completed at position: {final_position:?}");
            last_positions.lock().expect("last_positions lock poisoned")[slot] = final_position;
            if let Err(e) = ctx.complete(final_position).await {
                eprintln!("[controller] {side} complete error: {e:?}");
            }
        }
        None => {
            let last_known = *current_position.lock().expect("current_position lock poisoned");
            last_positions.lock().expect("last_positions lock poisoned")[slot] = last_known;
            println!("[controller] {side} arm cancelled at position: {last_known:?}");
            if let Err(e) = ctx.complete_cancelled(last_known).await {
                eprintln!("[controller] {side} complete_cancelled error: {e:?}");
            }
        }
    }

    busy_arms.lock().expect("busy lock poisoned").remove(&arm_id);
}

async fn execute_goal(
    node_runner: &Arc<peppygen::NodeRunner>,
    ctx: &move_arm::GoalContext,
    arm_id: u16,
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
        *current_position.lock().expect("current_position lock poisoned") = current;
        let cmd_positions = current.map(|v| v as f64);
        let _ = joint_positions::emit(node_runner, arm_id, cmd_positions, 1.0).await;
        let _ = ctx.publish_feedback(current).await;
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
            if let Err(error) = run_action(action_runner).await {
                tracing::error!("move_arm action error: {error:?}");
            }
        });

        Ok(())
    })
}
