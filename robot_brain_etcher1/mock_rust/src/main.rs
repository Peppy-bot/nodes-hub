use peppygen::consumed_actions::robot_controller_move_left_arm as left_arm;
use peppygen::consumed_actions::robot_controller_move_right_arm as right_arm;
use peppygen::consumed_topics::camera_stream_video_stream as video_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use std::sync::Arc;
use std::time::Duration;

async fn ai_process(node_runner: Arc<NodeRunner>, cancel_token: CancellationToken) {
    println!("[brain] AI process started, waiting for video frames...");
    loop {
        if cancel_token.is_cancelled() {
            println!("[brain] Shutdown requested, stopping AI process");
            return;
        }

        // Subscribe to video frames from the camera
        let frame_result = video_stream::on_next_message_received(&node_runner, None, None).await;

        let (_instance_id, frame) = match frame_result {
            Ok(msg) => {
                println!("[brain] Received video frame");
                msg
            }
            Err(e) => {
                eprintln!("Failed to receive video frame: {e}");
                continue;
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
        let left_goal = left_arm::GoalRequest {
            arm_id: 0,
            desired_position: fake_position,
        };
        let right_goal = right_arm::GoalRequest {
            arm_id: 1,
            desired_position: fake_position,
        };

        let goal_timeout = Duration::from_secs(5);
        let result_timeout = Duration::from_secs(10);

        // Fire goals to both arms concurrently
        let (left_goal_result, right_goal_result) = tokio::join!(
            left_arm::ActionHandle::fire_goal(
                &node_runner,
                goal_timeout,
                None,
                None,
                left_goal,
                QoSProfile::Standard
            ),
            right_arm::ActionHandle::fire_goal(
                &node_runner,
                goal_timeout,
                None,
                None,
                right_goal,
                QoSProfile::Standard
            ),
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
                    peppygen::Result<left_arm::ResultResponse>,
                    peppygen::Result<right_arm::ResultResponse>,
                ) = tokio::join!(
                    left_h.get_result(result_timeout),
                    right_h.get_result(result_timeout),
                );

                match left_result {
                    Ok(result) => println!(
                        "[brain] Left arm completed at position: {:?}",
                        result.data.final_position
                    ),
                    Err(e) => eprintln!("[brain] Failed to get left arm result: {e}"),
                }

                match right_result {
                    Ok(result) => println!(
                        "[brain] Right arm completed at position: {:?}",
                        result.data.final_position
                    ),
                    Err(e) => eprintln!("[brain] Failed to get right arm result: {e}"),
                }
            }
            (Some(left_h), None) => match left_h.get_result(result_timeout).await {
                Ok(result) => println!(
                    "[brain] Left arm completed at position: {:?}",
                    result.data.final_position
                ),
                Err(e) => eprintln!("[brain] Failed to get left arm result: {e}"),
            },
            (None, Some(right_h)) => match right_h.get_result(result_timeout).await {
                Ok(result) => println!(
                    "[brain] Right arm completed at position: {:?}",
                    result.data.final_position
                ),
                Err(e) => eprintln!("[brain] Failed to get right arm result: {e}"),
            },
            (None, None) => {
                eprintln!("[brain] Both arm goals failed, skipping result wait");
            }
        }
    }
}

fn main() -> Result<()> {
    NodeBuilder::<Parameters>::new().run(|_args, node_runner| async move {
        let cancel_token = node_runner.cancellation_token().clone();
        tokio::spawn(async move {
            ai_process(node_runner, cancel_token).await;
        });
        Ok(())
    })
}
