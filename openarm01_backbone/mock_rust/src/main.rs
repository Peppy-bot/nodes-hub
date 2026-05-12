use peppygen::consumed_topics::robot_arm_joint_states;
use peppygen::emitted_topics::joint_commands;
use peppygen::exposed_actions::{move_left_arm, move_right_arm};
use peppygen::{NodeBuilder, Parameters, Result};
use peppylib::runtime::CancellationToken;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

trait ArmAction: Sized + Send {
    type GoalRequest: Send;

    fn goal_desired_position(request: &Self::GoalRequest) -> [i32; 3];

    fn expose_action(
        node_runner: &peppygen::NodeRunner,
    ) -> impl Future<Output = Result<Self>> + Send + '_;

    async fn next_goal(&mut self) -> Result<Option<Self::GoalRequest>>;

    async fn check_cancel(&mut self) -> Result<CancelPoll>;

    fn send_feedback(&mut self, position: [i32; 3])
    -> impl Future<Output = Result<()>> + Send + '_;

    fn send_result(
        &mut self,
        final_position: [i32; 3],
    ) -> impl Future<Output = Result<bool>> + Send + '_;
}

impl ArmAction for move_left_arm::ActionHandle {
    type GoalRequest = move_left_arm::GoalRequest;

    fn goal_desired_position(request: &Self::GoalRequest) -> [i32; 3] {
        request.data.desired_position
    }

    fn expose_action(
        node_runner: &peppygen::NodeRunner,
    ) -> impl Future<Output = Result<Self>> + Send + '_ {
        Self::expose(node_runner)
    }

    async fn next_goal(&mut self) -> Result<Option<Self::GoalRequest>> {
        let goal_holder = Arc::new(Mutex::new(None));
        let goal_holder_clone = Arc::clone(&goal_holder);
        let handled = self
            .handle_goal_next_request(move |request| {
                *goal_holder_clone.lock().expect("goal lock poisoned") = Some(request);
                Ok(move_left_arm::GoalResponse::new(true))
            })
            .await?;
        if !handled {
            return Ok(None);
        }
        Ok(goal_holder.lock().expect("goal lock poisoned").take())
    }

    async fn check_cancel(&mut self) -> Result<CancelPoll> {
        match tokio::time::timeout(
            Duration::from_millis(0),
            self.handle_cancel_next_request(|_request| {
                Ok(move_left_arm::CancelResponse::new(true, None))
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

    fn send_feedback(
        &mut self,
        position: [i32; 3],
    ) -> impl Future<Output = Result<()>> + Send + '_ {
        self.emit_feedback(position)
    }

    fn send_result(
        &mut self,
        final_position: [i32; 3],
    ) -> impl Future<Output = Result<bool>> + Send + '_ {
        self.handle_result_next_request(move |_request| {
            Ok(move_left_arm::ResultResponse::new(final_position))
        })
    }
}

impl ArmAction for move_right_arm::ActionHandle {
    type GoalRequest = move_right_arm::GoalRequest;

    fn goal_desired_position(request: &Self::GoalRequest) -> [i32; 3] {
        request.data.desired_position
    }

    fn expose_action(
        node_runner: &peppygen::NodeRunner,
    ) -> impl Future<Output = Result<Self>> + Send + '_ {
        Self::expose(node_runner)
    }

    async fn next_goal(&mut self) -> Result<Option<Self::GoalRequest>> {
        let goal_holder = Arc::new(Mutex::new(None));
        let goal_holder_clone = Arc::clone(&goal_holder);
        let handled = self
            .handle_goal_next_request(move |request| {
                *goal_holder_clone.lock().expect("goal lock poisoned") = Some(request);
                Ok(move_right_arm::GoalResponse::new(true))
            })
            .await?;
        if !handled {
            return Ok(None);
        }
        Ok(goal_holder.lock().expect("goal lock poisoned").take())
    }

    async fn check_cancel(&mut self) -> Result<CancelPoll> {
        match tokio::time::timeout(
            Duration::from_millis(0),
            self.handle_cancel_next_request(|_request| {
                Ok(move_right_arm::CancelResponse::new(true, None))
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

    fn send_feedback(
        &mut self,
        position: [i32; 3],
    ) -> impl Future<Output = Result<()>> + Send + '_ {
        self.emit_feedback(position)
    }

    fn send_result(
        &mut self,
        final_position: [i32; 3],
    ) -> impl Future<Output = Result<bool>> + Send + '_ {
        self.handle_result_next_request(move |_request| {
            Ok(move_right_arm::ResultResponse::new(final_position))
        })
    }
}

async fn run_arm_action<A: ArmAction>(
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
    side: &str,
) -> Result<()> {
    println!("[controller] {side} arm action handler started");
    let mut action = A::expose_action(&node_runner).await?;
    let mut last_position = [0, 0, 0];

    loop {
        if cancel_token.is_cancelled() {
            println!("[controller] {side} arm shutdown requested");
            break;
        }

        let Some(goal_request) = action.next_goal().await? else {
            println!("[controller] {side} arm action handler closed");
            break;
        };

        let desired_position = A::goal_desired_position(&goal_request);
        println!("[controller] {side} arm received goal: {desired_position:?}");
        let cmd_positions = desired_position.map(|v| v as f64);
        if let Err(e) = joint_commands::emit(&node_runner, cmd_positions, 1.0).await {
            eprintln!("[controller] {side} emit joint_commands error: {e:?}");
        } else {
            println!(
                "[controller] {side} published joint_commands: target={cmd_positions:.3?} max_vel=1.0"
            );
        }
        let duration = choose_action_duration();

        let outcome = execute_goal(
            &mut action,
            &node_runner,
            last_position,
            desired_position,
            duration,
        )
        .await?;

        let final_position = match outcome {
            ActionOutcome::Completed(position) => {
                println!("[controller] {side} arm completed at position: {position:?}");
                last_position = position;
                position
            }
            ActionOutcome::Cancelled(position) => {
                println!("[controller] {side} arm cancelled at position: {position:?}");
                last_position = position;
                position
            }
            ActionOutcome::Closed => {
                println!("[controller] {side} arm action closed");
                break;
            }
        };

        // Use timeout to avoid blocking forever if client doesn't request result
        let result_timeout = Duration::from_secs(10);
        match tokio::time::timeout(result_timeout, action.send_result(final_position)).await {
            Ok(Ok(true)) => {} // Result was requested and handled
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

async fn execute_goal<A: ArmAction>(
    action: &mut A,
    node_runner: &Arc<peppygen::NodeRunner>,
    start: [i32; 3],
    target: [i32; 3],
    duration: Duration,
) -> Result<ActionOutcome> {
    action.send_feedback(start).await?;

    match action.check_cancel().await? {
        CancelPoll::Cancelled => return Ok(ActionOutcome::Cancelled(start)),
        CancelPoll::Closed => return Ok(ActionOutcome::Closed),
        CancelPoll::None => {}
    }

    let (steps, step_duration) = feedback_plan(duration);
    let mut current = start;

    for step in 1..=steps {
        tokio::time::sleep(step_duration).await;

        match action.check_cancel().await? {
            CancelPoll::Cancelled => return Ok(ActionOutcome::Cancelled(current)),
            CancelPoll::Closed => return Ok(ActionOutcome::Closed),
            CancelPoll::None => {}
        }

        let ratio = step as f32 / steps as f32;
        current = interpolate_position(start, target, ratio);
        let cmd_positions = current.map(|v| v as f64);
        let _ = joint_commands::emit(node_runner, cmd_positions, 1.0).await;
        action.send_feedback(current).await?;

        match action.check_cancel().await? {
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
        let left_runner = Arc::clone(&node_runner);
        let right_runner = Arc::clone(&node_runner);
        let states_runner = Arc::clone(&node_runner);
        let left_cancel_token = node_runner.cancellation_token().clone();
        let right_cancel_token = node_runner.cancellation_token().clone();

        tokio::spawn(async move {
            loop {
                match robot_arm_joint_states::on_next_message_received(&states_runner, None, None)
                    .await
                {
                    Ok((_id, msg)) => println!(
                        "[controller] joint_states update: positions={:.3?} velocities={:.3?}",
                        msg.positions, msg.velocities
                    ),
                    Err(e) => {
                        eprintln!("[controller] joint_states subscription closed: {e:?}");
                        break;
                    }
                }
            }
        });

        tokio::spawn(async move {
            if let Err(error) = run_arm_action::<move_left_arm::ActionHandle>(
                left_runner,
                left_cancel_token,
                "Left",
            )
            .await
            {
                tracing::error!("Left arm action error: {error:?}");
            }
        });

        tokio::spawn(async move {
            if let Err(error) = run_arm_action::<move_right_arm::ActionHandle>(
                right_runner,
                right_cancel_token,
                "Right",
            )
            .await
            {
                tracing::error!("Right arm action error: {error:?}");
            }
        });

        Ok(())
    })
}
