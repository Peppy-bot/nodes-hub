use peppygen::consumed_actions::{left_robot_arm_move_arm, right_robot_arm_move_arm};
use peppygen::exposed_actions::move_arm;
use peppygen::{NodeBuilder, Parameters, QoSProfile, Result};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const ARM_ID_LEFT: u16 = 0;
const ARM_ID_RIGHT: u16 = 1;

const GOAL_TIMEOUT: Duration = Duration::from_secs(5);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);
const RESULT_TIMEOUT: Duration = Duration::from_secs(30);
// Bound each wait for the next feedback message. The feedback end-of-stream
// sentinel is an ordinary message that can be lost or delayed; without a bound,
// a lost sentinel would block feedback draining forever and the result would
// never be fetched, timing out the client. On idle/end/error we stop relaying
// feedback and fetch the result, which parks server-side until the goal truly
// completes — so breaking early never loses the result, it only stops relaying
// intermediate progress. Keep this comfortably below typical client result
// timeouts and well above the arm's feedback cadence.
const FEEDBACK_IDLE_TIMEOUT: Duration = Duration::from_secs(2);

fn arm_side(arm_id: u16) -> &'static str {
    match arm_id {
        ARM_ID_LEFT => "Left",
        ARM_ID_RIGHT => "Right",
        _ => "Unknown",
    }
}

// The per-arm consumed_actions modules generate distinct types with the same
// shape. Wrap them in an enum so the forwarding loop only has to be written
// once; the per-arm code is just the constructor and trivial delegations.
enum ArmHandle {
    Left(left_robot_arm_move_arm::ActionHandle),
    Right(right_robot_arm_move_arm::ActionHandle),
}

impl ArmHandle {
    async fn fire(
        node_runner: &peppygen::NodeRunner,
        arm_id: u16,
        desired: [i32; 3],
    ) -> Result<Self> {
        match arm_id {
            ARM_ID_LEFT => {
                let request = left_robot_arm_move_arm::GoalRequest {
                    desired_position: desired,
                };
                let handle = left_robot_arm_move_arm::ActionHandle::fire_goal(
                    node_runner,
                    GOAL_TIMEOUT,
                    request,
                    QoSProfile::Standard,
                )
                .await?;
                Ok(ArmHandle::Left(handle))
            }
            ARM_ID_RIGHT => {
                let request = right_robot_arm_move_arm::GoalRequest {
                    desired_position: desired,
                };
                let handle = right_robot_arm_move_arm::ActionHandle::fire_goal(
                    node_runner,
                    GOAL_TIMEOUT,
                    request,
                    QoSProfile::Standard,
                )
                .await?;
                Ok(ArmHandle::Right(handle))
            }
            _ => unreachable!("decider rejects unknown arm_id before fire"),
        }
    }

    fn accepted(&self) -> bool {
        match self {
            ArmHandle::Left(h) => h.data.accepted,
            ArmHandle::Right(h) => h.data.accepted,
        }
    }

    fn rejection_reason(&self) -> Option<&str> {
        match self {
            ArmHandle::Left(h) => h.data.error_message.as_deref(),
            ArmHandle::Right(h) => h.data.error_message.as_deref(),
        }
    }

    async fn next_feedback(&mut self) -> Result<[i32; 3]> {
        match self {
            ArmHandle::Left(h) => h
                .on_next_feedback_message()
                .await
                .map(|fb| fb.current_position),
            ArmHandle::Right(h) => h
                .on_next_feedback_message()
                .await
                .map(|fb| fb.current_position),
        }
    }

    async fn cancel(&self, timeout: Duration) -> Result<()> {
        match self {
            ArmHandle::Left(h) => h.cancel_goal(timeout).await.map(|_| ()),
            ArmHandle::Right(h) => h.cancel_goal(timeout).await.map(|_| ()),
        }
    }

    async fn final_position(&self, timeout: Duration) -> Result<[i32; 3]> {
        match self {
            ArmHandle::Left(h) => h.get_result(timeout).await.map(|r| r.data.final_position),
            ArmHandle::Right(h) => h.get_result(timeout).await.map(|r| r.data.final_position),
        }
    }
}

async fn forward(backbone_ctx: move_arm::GoalContext, mut handle: ArmHandle, side: &str) {
    // The decider already accepted on the arm's behalf, so no accept check
    // here — forward feedback/cancels, then complete with the arm's result.
    // Result delivery must NOT depend on the feedback stream ending: each wait
    // for the next feedback message is bounded (see FEEDBACK_IDLE_TIMEOUT), so a
    // lost/delayed end-of-stream sentinel can never wedge this goal.
    let mut cancelled = false;
    loop {
        tokio::select! {
            fb = tokio::time::timeout(FEEDBACK_IDLE_TIMEOUT, handle.next_feedback()) => match fb {
                Ok(Ok(fp)) => {
                    let _ = backbone_ctx.publish_feedback(fp).await;
                }
                // Stream ended (Err) or went idle (Elapsed): stop draining and
                // go fetch the authoritative result.
                Ok(Err(_)) | Err(_) => break,
            },
            _ = backbone_ctx.cancel_signal(), if !cancelled => {
                cancelled = true;
                if let Err(e) = handle.cancel(CANCEL_TIMEOUT).await {
                    eprintln!("[controller] {side} cancel_goal error: {e:?}");
                }
            }
        }
    }

    match handle.final_position(RESULT_TIMEOUT).await {
        Ok(fp) => {
            println!("[controller] {side} arm completed at position: {fp:?}");
            let complete_result = if cancelled {
                backbone_ctx.complete_cancelled(fp).await
            } else {
                backbone_ctx.complete(fp).await
            };
            if let Err(e) = complete_result {
                eprintln!("[controller] {side} complete error: {e:?}");
            }
        }
        Err(e) => {
            eprintln!("[controller] {side} get_result error: {e:?}");
            let _ = backbone_ctx.complete_cancelled([0, 0, 0]).await;
        }
    }
}

async fn run_action(node_runner: Arc<peppygen::NodeRunner>) -> Result<()> {
    println!("[controller] move_arm action handler started");
    let mut action = move_arm::ActionHandle::expose(&node_runner).await?;
    let busy_arms: Arc<Mutex<HashSet<u16>>> = Arc::new(Mutex::new(HashSet::new()));
    // Decider stashes the accepted arm handle here; drive_goal picks it up
    // after the backbone-side accept clears.
    let pending_handles: Arc<Mutex<HashMap<u16, ArmHandle>>> = Arc::new(Mutex::new(HashMap::new()));

    loop {
        let busy_for_decider = Arc::clone(&busy_arms);
        let pending_for_decider = Arc::clone(&pending_handles);
        let runner_for_decider = Arc::clone(&node_runner);
        let next = action
            .handle_goal_next_request(move |request| {
                let arm_id = request.data.arm_id;
                let side = arm_side(arm_id);
                println!(
                    "[controller] {side} arm received goal: {:?}",
                    request.data.desired_position
                );
                if arm_id != ARM_ID_LEFT && arm_id != ARM_ID_RIGHT {
                    return Ok(move_arm::GoalResponse::reject(format!(
                        "unknown arm_id {arm_id}"
                    )));
                }
                if busy_for_decider
                    .lock()
                    .expect("busy lock poisoned")
                    .contains(&arm_id)
                {
                    return Ok(move_arm::GoalResponse::reject(format!(
                        "arm {arm_id} is already moving"
                    )));
                }
                // Pre-fire at the arm so we can mirror its accept/reject. The
                // decider is sync, so bridge into async with block_in_place +
                // block_on — safe here because NodeBuilder uses a multi-thread
                // tokio runtime.
                let desired = request.data.desired_position;
                let runner = Arc::clone(&runner_for_decider);
                let arm_handle_result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(ArmHandle::fire(&runner, arm_id, desired))
                });
                let arm_handle = match arm_handle_result {
                    Ok(h) => h,
                    Err(e) => {
                        return Ok(move_arm::GoalResponse::reject(format!(
                            "{side} fire_goal error: {e:?}"
                        )));
                    }
                };
                if !arm_handle.accepted() {
                    let reason = arm_handle
                        .rejection_reason()
                        .unwrap_or("arm rejected")
                        .to_string();
                    println!("[controller] {side} arm rejected forwarded goal: {reason}");
                    return Ok(move_arm::GoalResponse::reject(reason));
                }
                println!("[controller] {side} arm accepted forwarded goal");
                busy_for_decider
                    .lock()
                    .expect("busy lock poisoned")
                    .insert(arm_id);
                pending_for_decider
                    .lock()
                    .expect("pending lock poisoned")
                    .insert(arm_id, arm_handle);
                Ok(move_arm::GoalResponse::accept())
            })
            .await?;
        let Some(ctx) = next else {
            println!("[controller] move_arm action handler closed");
            break;
        };
        let arm_id = ctx.request().data.arm_id;
        let arm_handle = pending_handles
            .lock()
            .expect("pending lock poisoned")
            .remove(&arm_id)
            .expect("decider stashed handle for accepted goal");
        let busy = Arc::clone(&busy_arms);
        tokio::spawn(async move {
            drive_goal(arm_handle, ctx, busy, arm_id).await;
        });
    }
    Ok(())
}

async fn drive_goal(
    arm_handle: ArmHandle,
    backbone_ctx: move_arm::GoalContext,
    busy_arms: Arc<Mutex<HashSet<u16>>>,
    arm_id: u16,
) {
    let side = arm_side(arm_id);
    forward(backbone_ctx, arm_handle, side).await;
    busy_arms
        .lock()
        .expect("busy lock poisoned")
        .remove(&arm_id);
}

fn main() -> Result<()> {
    NodeBuilder::<Parameters>::new().run(|_args, node_runner| async move {
        tokio::spawn(async move {
            if let Err(error) = run_action(node_runner).await {
                tracing::error!("move_arm action error: {error:?}");
            }
        });
        Ok(())
    })
}
