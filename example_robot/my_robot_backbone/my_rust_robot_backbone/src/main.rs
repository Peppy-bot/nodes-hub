use peppygen::consumed_actions::{left_robot_arm_move_arm, right_robot_arm_move_arm};
use peppygen::exposed_actions::move_arm;
use peppygen::{NodeBuilder, Parameters, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;

const ARM_ID_LEFT: u16 = 0;
const ARM_ID_RIGHT: u16 = 1;

const GOAL_TIMEOUT: Duration = Duration::from_secs(5);
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);
const RESULT_TIMEOUT: Duration = Duration::from_secs(30);

fn arm_side(arm_id: u16) -> &'static str {
    match arm_id {
        ARM_ID_LEFT => "Left",
        ARM_ID_RIGHT => "Right",
        _ => "Unknown",
    }
}

// The two arm modules generate distinct `ResultOutcome` enums with the same
// shape; collapse them into one type so the forwarding loop maps the outcome
// once. Completed/Cancelled carry the final position; Abandoned/Expired do not.
enum ArmOutcome {
    Completed([i32; 3]),
    Cancelled([i32; 3]),
    Abandoned,
    Expired,
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
                    left_robot_arm_move_arm::bound_producer(node_runner),
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
                    right_robot_arm_move_arm::bound_producer(node_runner),
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

    async fn result(&self, timeout: Duration) -> Result<ArmOutcome> {
        match self {
            ArmHandle::Left(h) => Ok(match h.get_result(timeout).await?.outcome {
                left_robot_arm_move_arm::ResultOutcome::Completed(d) => {
                    ArmOutcome::Completed(d.final_position)
                }
                left_robot_arm_move_arm::ResultOutcome::Cancelled(d) => {
                    ArmOutcome::Cancelled(d.final_position)
                }
                left_robot_arm_move_arm::ResultOutcome::Abandoned => ArmOutcome::Abandoned,
                left_robot_arm_move_arm::ResultOutcome::Expired => ArmOutcome::Expired,
            }),
            ArmHandle::Right(h) => Ok(match h.get_result(timeout).await?.outcome {
                right_robot_arm_move_arm::ResultOutcome::Completed(d) => {
                    ArmOutcome::Completed(d.final_position)
                }
                right_robot_arm_move_arm::ResultOutcome::Cancelled(d) => {
                    ArmOutcome::Cancelled(d.final_position)
                }
                right_robot_arm_move_arm::ResultOutcome::Abandoned => ArmOutcome::Abandoned,
                right_robot_arm_move_arm::ResultOutcome::Expired => ArmOutcome::Expired,
            }),
        }
    }
}

/// Goals currently forwarded to the arms, keyed by arm id. Entries live from
/// the decider's accept until `drive_goal` finishes, so the map doubles as the
/// busy-arm set. Each handle sits behind an async mutex shared with the
/// shutdown hook, which cancels whatever is still in flight at shutdown.
type ActiveHandles = Arc<Mutex<HashMap<u16, Arc<AsyncMutex<ArmHandle>>>>>;

async fn forward(
    backbone_ctx: move_arm::GoalContext,
    handle: Arc<AsyncMutex<ArmHandle>>,
    side: &str,
    token: CancellationToken,
) {
    // The decider already accepted on the arm's behalf, so no accept check
    // here — forward feedback/cancels, then relay the arm's typed outcome.
    // get_result parks until the arm reaches a terminal state, so result
    // delivery is always definitive and never depends on feedback timing; the
    // feedback drain just ends when the arm closes its stream (on completion,
    // cancel, or abandonment).
    //
    // The handle guard is held for the whole drive. Every long await below
    // selects on the cancellation token and returns without cleanup: that
    // releases the guard, and the shutdown hook — the owner of cleanup —
    // takes it to cancel the arm goal.
    let mut handle = handle.lock().await;
    let mut cancelled = false;
    loop {
        tokio::select! {
            fb = handle.next_feedback() => match fb {
                Ok(fp) => {
                    let _ = backbone_ctx.publish_feedback(fp).await;
                }
                Err(_) => break,
            },
            _ = backbone_ctx.cancel_signal(), if !cancelled => {
                cancelled = true;
                if let Err(e) = handle.cancel(CANCEL_TIMEOUT).await {
                    eprintln!("[controller] {side} cancel_goal error: {e:?}");
                }
            }
            _ = token.cancelled() => return,
        }
    }

    // Mirror the arm's outcome onto our own goal. For Abandoned/Expired we leave
    // backbone_ctx uncompleted, which the engine reports to our client as
    // Abandoned.
    let result = tokio::select! {
        result = handle.result(RESULT_TIMEOUT) => result,
        _ = token.cancelled() => return,
    };
    match result {
        Ok(ArmOutcome::Completed(fp)) => {
            println!("[controller] {side} arm completed at position: {fp:?}");
            if let Err(e) = backbone_ctx.complete(fp).await {
                eprintln!("[controller] {side} complete error: {e:?}");
            }
        }
        Ok(ArmOutcome::Cancelled(fp)) => {
            println!("[controller] {side} arm cancelled at position: {fp:?}");
            if let Err(e) = backbone_ctx.complete_cancelled(fp).await {
                eprintln!("[controller] {side} complete error: {e:?}");
            }
        }
        Ok(ArmOutcome::Abandoned) => {
            eprintln!("[controller] {side} arm abandoned its goal; abandoning forwarded goal");
        }
        Ok(ArmOutcome::Expired) => {
            eprintln!("[controller] {side} arm result expired; abandoning forwarded goal");
        }
        Err(e) => {
            eprintln!("[controller] {side} get_result error: {e:?}");
        }
    }
}

async fn run_action(
    node_runner: Arc<peppygen::NodeRunner>,
    active_handles: ActiveHandles,
) -> Result<()> {
    println!("[controller] move_arm action handler started");
    let mut action = move_arm::ActionHandle::expose(&node_runner).await?;
    let token = node_runner.cancellation_token().clone();

    loop {
        let active_for_decider = Arc::clone(&active_handles);
        let runner_for_decider = Arc::clone(&node_runner);
        let token_for_decider = token.clone();
        let next_request = action.handle_goal_next_request(move |request| {
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
            if active_for_decider
                .lock()
                .expect("active lock poisoned")
                .contains_key(&arm_id)
            {
                return Ok(move_arm::GoalResponse::reject(format!(
                    "arm {arm_id} is already moving"
                )));
            }
            // Pre-fire at the arm so we can mirror its accept/reject. The
            // decider is sync, so bridge into async with block_in_place +
            // block_on — safe here because NodeBuilder uses a multi-thread
            // tokio runtime. The bridge must never outlive the cancellation
            // token: after the shutdown hooks run, Runtime::drop blocks until
            // block_in_place sections return, so an unbounded fire here would
            // blow the shutdown grace window.
            let desired = request.data.desired_position;
            let runner = Arc::clone(&runner_for_decider);
            let active = Arc::clone(&active_for_decider);
            let token = token_for_decider.clone();
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let arm_handle = tokio::select! {
                        fired = ArmHandle::fire(&runner, arm_id, desired) => match fired {
                            Ok(handle) => handle,
                            Err(e) => {
                                return Ok(move_arm::GoalResponse::reject(format!(
                                    "{side} fire_goal error: {e:?}"
                                )));
                            }
                        },
                        _ = token.cancelled() => {
                            return Ok(move_arm::GoalResponse::reject("node is shutting down"));
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
                    // Register under the registry lock with a token re-check:
                    // the shutdown hook snapshots the registry right after the
                    // token fires, so a handle whose fire raced shutdown would
                    // be missed by the hook and must be cancelled here instead
                    // of registered.
                    {
                        let mut active = active.lock().expect("active lock poisoned");
                        if !token.is_cancelled() {
                            println!("[controller] {side} arm accepted forwarded goal");
                            active.insert(arm_id, Arc::new(AsyncMutex::new(arm_handle)));
                            return Ok(move_arm::GoalResponse::accept());
                        }
                    }
                    if let Err(e) = arm_handle.cancel(CANCEL_TIMEOUT).await {
                        eprintln!("[controller] {side} cancel_goal error: {e:?}");
                    }
                    Ok(move_arm::GoalResponse::reject("node is shutting down"))
                })
            })
        });
        let next = tokio::select! {
            next = next_request => next?,
            // Shutdown: stop accepting goals. In-flight forwarded goals are
            // cancelled by the on_shutdown hook, not here.
            _ = token.cancelled() => break,
        };
        let Some(ctx) = next else {
            println!("[controller] move_arm action handler closed");
            break;
        };
        let arm_id = ctx.request().data.arm_id;
        let arm_handle = active_handles
            .lock()
            .expect("active lock poisoned")
            .get(&arm_id)
            .cloned()
            .expect("decider stashed handle for accepted goal");
        let active = Arc::clone(&active_handles);
        let goal_token = token.clone();
        tokio::spawn(async move {
            drive_goal(arm_handle, ctx, active, arm_id, goal_token).await;
        });
    }
    Ok(())
}

async fn drive_goal(
    arm_handle: Arc<AsyncMutex<ArmHandle>>,
    backbone_ctx: move_arm::GoalContext,
    active_handles: ActiveHandles,
    arm_id: u16,
    token: CancellationToken,
) {
    let side = arm_side(arm_id);
    forward(backbone_ctx, arm_handle, side, token.clone()).await;
    // On shutdown the entry must stay registered: forward stopped without
    // cleanup, and the shutdown hook cancels the arm goal via the registry.
    if !token.is_cancelled() {
        active_handles
            .lock()
            .expect("active lock poisoned")
            .remove(&arm_id);
    }
}

fn main() -> Result<()> {
    NodeBuilder::<Parameters>::new().run(|_args, node_runner| async move {
        let active_handles: ActiveHandles = Arc::new(Mutex::new(HashMap::new()));

        // The arms keep executing a forwarded goal even after this controller
        // dies, so cancelling whatever is still in flight is an awaited
        // shutdown obligation, not task cleanup: it lives in a hook, which the
        // runtime runs while the messenger is still connected.
        let active_for_shutdown = Arc::clone(&active_handles);
        node_runner.on_shutdown(async move {
            let handles: Vec<(u16, Arc<AsyncMutex<ArmHandle>>)> = active_for_shutdown
                .lock()
                .expect("active lock poisoned")
                .iter()
                .map(|(arm_id, handle)| (*arm_id, Arc::clone(handle)))
                .collect();
            for (arm_id, handle) in handles {
                let side = arm_side(arm_id);
                // forward() releases the handle guard once the token fires, so
                // this acquire resolves promptly.
                let handle = handle.lock().await;
                match handle.cancel(CANCEL_TIMEOUT).await {
                    Ok(()) => {
                        println!("[controller] {side} arm goal cancelled at shutdown");
                    }
                    Err(e) => {
                        eprintln!("[controller] {side} shutdown cancel_goal error: {e:?}");
                    }
                }
            }
        });

        // Log when the shutdown/cancel signal is received so it is visible in
        // the node's stdout.
        node_runner.on_shutdown(async move {
            println!("[controller] Shutdown signal received");
        });

        tokio::spawn(async move {
            if let Err(error) = run_action(node_runner, active_handles).await {
                tracing::error!("move_arm action error: {error:?}");
            }
        });
        Ok(())
    })
}
