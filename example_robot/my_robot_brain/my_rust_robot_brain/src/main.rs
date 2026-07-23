use peppygen::consumed_actions::robot_controller::move_arm as arm;
use peppygen::consumed_topics::camera::video_stream as video_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

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

async fn read_video_frames(
    node_runner: Arc<NodeRunner>,
    latest_frame: watch::Sender<Option<Arc<video_stream::Message>>>,
    cancel_token: CancellationToken,
) {
    // Own the subscription in a dedicated task so action waits never stop the
    // transport from being drained. The watch channel coalesces every burst to
    // one Arc-backed latest frame instead of retaining a stale frame backlog.
    let mut subscription = match video_stream::subscribe(&node_runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            eprintln!("Failed to subscribe to video stream: {e}");
            return;
        }
    };
    while !latest_frame.is_closed() {
        tokio::select! {
            _ = cancel_token.cancelled() => return,
            received = subscription.next() => match received {
                Ok(Some((_producer, frame))) => {
                    latest_frame.send_replace(Some(Arc::new(frame)));
                }
                Ok(None) => return,
                Err(e) => eprintln!("Failed to receive video frame: {e}"),
            },
        }
    }
}

async fn ai_process(node_runner: Arc<NodeRunner>, cancel_token: CancellationToken) {
    println!("[brain] AI process started, waiting for video frames...");
    let (latest_frame_tx, mut latest_frame_rx) = watch::channel(None);
    let reader_node_runner = Arc::clone(&node_runner);
    let reader_cancel_token = cancel_token.clone();
    let reader_task = tokio::spawn(async move {
        read_video_frames(reader_node_runner, latest_frame_tx, reader_cancel_token).await;
    });

    loop {
        let changed = tokio::select! {
            _ = cancel_token.cancelled() => {
                println!("[brain] Shutdown requested, stopping AI process");
                break;
            }
            changed = latest_frame_rx.changed() => changed,
        };
        if changed.is_err() {
            eprintln!("Video stream closed");
            break;
        }

        // `borrow_and_update` marks every frame published so far as observed;
        // if several arrived during the prior action, only the newest remains.
        let Some(frame) = latest_frame_rx.borrow_and_update().clone() else {
            continue;
        };
        println!("[brain] Received latest video frame");

        // Shutdown still interrupts multi-second goal/result waits while the
        // independent reader continues draining frames in the other branch.
        tokio::select! {
            _ = cancel_token.cancelled() => {
                println!("[brain] Shutdown requested, stopping AI process");
                break;
            }
            _ = process_frame(&node_runner, frame.as_ref()) => {}
        }
    }

    reader_task.abort();
    if let Err(e) = reader_task.await
        && !e.is_cancelled()
    {
        eprintln!("Video frame reader failed: {e}");
    }
}

async fn process_frame(node_runner: &NodeRunner, frame: &video_stream::Message) {
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
        arm::ActionHandle::fire_goal(
            node_runner,
            arm::bound_producer(node_runner),
            goal_timeout,
            left_goal,
            QoSProfile::Standard,
        ),
        arm::ActionHandle::fire_goal(
            node_runner,
            arm::bound_producer(node_runner),
            goal_timeout,
            right_goal,
            QoSProfile::Standard,
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
