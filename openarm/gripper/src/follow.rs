// Ambient following of a streamed gripper opening fraction: drive the motor
// toward the latest command; with none yet, hold (the motor keeps its last
// setpoint, so we simply do not re-command). Either way the loop
// receives and refreshes the motor state every tick, so the always-on state
// publisher serves a live reading rather than one frozen at bring-up until the
// first command (the arm control loop reads state every tick the same way).
// The opening is commanded directly; the motor eases to it. Which control mode
// carries that command, and whether the effort cap reaches the wire, is the
// gripper's own business (hardware.rs), so this loop is the same for both
// generations.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::motor_health::{NOT_DRIVING_ESCALATE_AFTER, STATE_STALE_AFTER};
use openarm_can::CanErrorThrottle;
use peppylib::runtime::CancellationToken;
use tokio::sync::{oneshot, watch};
use tokio::time::MissedTickBehavior;
use tracing::error;

use crate::command_stream::GripperCommand;
use crate::drive::{self, Verdict};
use crate::hardware::Gripper;

/// Set when the loop stops on a hard fault, so main can exit non-zero after
/// the shutdown hooks have run and the daemon records the instance as failed
/// rather than finished.
pub static HARD_FAULT: AtomicBool = AtomicBool::new(false);

const CONTEXT: &str = "gripper follow";

#[derive(Clone)]
pub struct ControlConfig {
    pub cycle_period: Duration,
    pub recv_timeout_us: u32,
}

pub async fn run(
    gripper: Arc<Mutex<Gripper>>,
    cmd: watch::Receiver<Option<GripperCommand>>,
    cfg: ControlConfig,
    token: CancellationToken,
    started_tx: oneshot::Sender<()>,
) {
    // Readiness gates on this: the loop is entered, not merely spawned.
    let _ = started_tx.send(());
    let mut ticker = tokio::time::interval(cfg.cycle_period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut throttle = CanErrorThrottle::new();
    let stale_after_ticks = drive::ticks_within(STATE_STALE_AFTER, cfg.cycle_period);
    let escalate_after_ticks = drive::ticks_within(NOT_DRIVING_ESCALATE_AFTER, cfg.cycle_period);
    let mut not_driving = 0u32;

    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            _ = ticker.tick() => {}
        }

        let command = *cmd.borrow();

        // unwrap_or_else: drive even if the mutex was poisoned by a panic
        // elsewhere, so a transient fault doesn't strand the follow loop.
        let mut g = gripper.lock().unwrap_or_else(|e| e.into_inner());
        // Checked under the lock the disable hook shares: cancellation
        // precedes the hooks, so a tick that passed the select before
        // shutdown cannot issue CAN work the hook would then race.
        if token.is_cancelled() {
            return;
        }
        // Receive first and unconditionally, consuming the replies solicited
        // by the previous tick's refresh: the decode pass is what advances
        // the silence count, on error passes included.
        let received = g.recv_all(cfg.recv_timeout_us);
        let state = g.get_state();
        let silent = state.passes_since_state >= stale_after_ticks;
        match drive::assess(state.status, silent, &mut not_driving, escalate_after_ticks) {
            Verdict::Continue => {}
            Verdict::Reenable => {
                if let Err(e) = g.reenable() {
                    error!("re-enable motor: {e}");
                }
            }
            Verdict::HardFault(reason) => {
                // The motor is already limp or unaccounted for, so stop the
                // node without commanding this tick. The shutdown hook
                // disables the motor, and is_ready drops via the latch.
                error!("{reason}; stopping node");
                HARD_FAULT.store(true, Ordering::SeqCst);
                token.cancel();
                return;
            }
        }

        // Command only when there is a target; refresh state every tick either way.
        let sent = (|| {
            if let Some(command) = command {
                g.command(command.opening, command.max_effort)?;
            }
            g.refresh_all()
        })();
        // A driver fault costs this tick's frames, not the loop: the motor
        // holds its last commanded target, the next tick re-sends an absolute
        // command, and the disable that stopping would imply travels over the
        // same socket that just failed.
        match received.and(sent) {
            Ok(()) => throttle.success(CONTEXT),
            Err(e) => throttle.failure(CONTEXT, &e),
        }
    }
}
