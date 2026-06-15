use peppygen::consumed_actions::robot_controller_move_arm as arm;
use peppygen::consumed_topics::camera_video_stream as video_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use std::sync::Arc;
use std::time::Duration;

fn log_arm_result(side: &str, result: arm::ResultResponse) {
    // get_result returns a typed terminal outcome rather than erroring.
    // Completed/Cancelled carry the result payload; Abandoned/Expired do not.
    match result.outcome {
        arm::ResultOutcome::Completed(data) => println!(
            "[brain] {side} arm completed at position: {:?}",
            data.final_position
        ),
        arm::ResultOutcome::Cancelled(data) => println!(
            "[brain] {side} arm cancelled at position: {:?}",
            data.final_position
        ),
        arm::ResultOutcome::Abandoned => {
            println!("[brain] {side} arm abandoned the goal without a result")
        }
        arm::ResultOutcome::Expired => {
            println!("[brain] {side} arm result expired before it was fetched")
        }
    }
}

async fn ai_process(node_runner: Arc<NodeRunner>, cancel_token: CancellationToken) {
    println!("[brain] AI process started, waiting for video frames...");
    loop {
        // Select the token against the iteration so shutdown interrupts the
        // unbounded frame wait and the multi-second goal/result awaits, not
        // just the gap between iterations.
        tokio::select! {
            _ = cancel_token.cancelled() => {
                println!("[brain] Shutdown requested, stopping AI process");
                return;
            }
            _ = process_next_frame(&node_runner) => {}
        }
    }
}

async fn process_next_frame(node_runner: &NodeRunner) {
    // Subscribe to video frames from the camera
    let frame_result = video_stream::on_next_message_received(node_runner).await;

    let (_producer, frame) = match frame_result {
        Ok(msg) => {
            println!("[brain] Received video frame");
            msg
        }
        Err(e) => {
            eprintln!("Failed to receive video frame: {e}");
            return;
        }
    };

    // Process the frame and generate fake arm positions
    let fake_position = [
        frame.frame[0] as i32,
        frame.frame[1] as i32,
        frame.frame[2] as i32,
    ];
    println!("[brain] Generated arm position: {:?}", fake_position);

    // Fire action goals to both arms concurrently
    println!("[brain] Firing goals to both arms...");
    let left_goal = arm::GoalRequest {
        arm_id: 0,
        desired_position: fake_position,
    };
    let right_goal = arm::GoalRequest {
        arm_id: 1,
        desired_position: fake_position,
    };

    let goal_timeout = Duration::from_secs(5);
    let result_timeout = Duration::from_secs(10);

    // Fire goals to both arms concurrently
    let (left_goal_result, right_goal_result) = tokio::join!(
        arm::ActionHandle::fire_goal(node_runner, goal_timeout, left_goal, QoSProfile::Standard),
        arm::ActionHandle::fire_goal(node_runner, goal_timeout, right_goal, QoSProfile::Standard),
    );

    // Get the action handles from accepted goals
    let left_handle = match left_goal_result {
        Ok(handle) if handle.data.accepted => {
            println!("[brain] Left arm goal accepted");
            Some(handle)
        }
        Ok(_) => {
            eprintln!("[brain] Left arm goal rejected");
            None
        }
        Err(e) => {
            eprintln!("Failed to fire left arm goal: {e}");
            None
        }
    };

    let right_handle = match right_goal_result {
        Ok(handle) if handle.data.accepted => {
            println!("[brain] Right arm goal accepted");
            Some(handle)
        }
        Ok(_) => {
            eprintln!("[brain] Right arm goal rejected");
            None
        }
        Err(e) => {
            eprintln!("Failed to fire right arm goal: {e}");
            None
        }
    };

    // Wait for results from both arms concurrently (only if goals were accepted)
    match (left_handle, right_handle) {
        (Some(left_h), Some(right_h)) => {
            let (left_result, right_result): (
                peppygen::Result<arm::ResultResponse>,
                peppygen::Result<arm::ResultResponse>,
            ) = tokio::join!(
                left_h.get_result(result_timeout),
                right_h.get_result(result_timeout),
            );

            match left_result {
                Ok(result) => log_arm_result("Left", result),
                Err(e) => eprintln!("[brain] Failed to get left arm result: {e}"),
            }

            match right_result {
                Ok(result) => log_arm_result("Right", result),
                Err(e) => eprintln!("[brain] Failed to get right arm result: {e}"),
            }
        }
        (Some(left_h), None) => match left_h.get_result(result_timeout).await {
            Ok(result) => log_arm_result("Left", result),
            Err(e) => eprintln!("[brain] Failed to get left arm result: {e}"),
        },
        (None, Some(right_h)) => match right_h.get_result(result_timeout).await {
            Ok(result) => log_arm_result("Right", result),
            Err(e) => eprintln!("[brain] Failed to get right arm result: {e}"),
        },
        (None, None) => {
            eprintln!("[brain] Both arm goals failed, skipping result wait");
        }
    }
}

fn main() -> Result<()> {
    NodeBuilder::<Parameters>::new().run(|_args, node_runner| async move {
        let cancel_token = node_runner.cancellation_token().clone();
        // Log when the shutdown/cancel signal is received so it is visible in
        // the node's stdout.
        node_runner.on_shutdown(async move {
            println!("[brain] Shutdown signal received");
        });
        tokio::spawn(async move {
            ai_process(node_runner, cancel_token).await;
        });
        Ok(())
    })
}
