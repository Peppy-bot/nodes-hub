// The state owner: a single task that owns `UiState` outright, so nothing else ever
// locks it. Everyone talks to it over channels instead of a shared mutex.
//
//   - the WS server sends operator input as `UiMsg` (a wire `Command` or a
//     disconnect) on the command channel;
//   - the state streams (arm/gripper/collision) and the discrete goal tasks send
//     measured state and goal outcomes as `Feedback`;
//   - the owner publishes the browser snapshot (pre-serialized) and the per-tick
//     `CommandFrame` (the setpoints to stream) on two watch channels.
//
// The owner is also the single motion authority: once per tick it advances each
// side's jog (the same pure step the panel drove before) and republishes the frame.
// Because it is the only writer, the tick decision and every reducer are ordinary
// synchronous functions on owned state, with no lock discipline to reason about.

use std::sync::Arc;
use std::time::{Duration, Instant};

use peppygen::NodeRunner;
use peppylib::runtime::CancellationToken;
use srs_model::nalgebra::{Quaternion, UnitQuaternion};
use tokio::sync::{mpsc, watch};
use tokio::time::{MissedTickBehavior, interval};
use tokio_util::sync::CancellationToken as GoalToken;

use crate::gestures::{BakedGesture, Registry, advance_gesture_step, lead_in_duration};
use crate::pose::{
    ArmModels, CartesianJog, Jog, JogCaps, JogMode, JogStep, JointJogStep, Pose, dist3, jog_tick,
    joint_jog_tick,
};
use crate::state::{
    ARM_DOF, Alert, ArmHealth, ArmTarget, BySide, GesturePhase, GesturePlayback, GripperHealth,
    Proximity, RecordingEpisode, SIDES, Side, UiState,
};
use crate::ui::{
    Command, build_snapshot_json, clamp_to_limits, ee_speed_floored, gripper_limits, sane_duration,
    valid_governor_band,
};
use crate::{move_arm, move_arm_joints, move_gripper, record};

/// The browser snapshot cadence (10 Hz); the command tick runs far faster, so the
/// FK-heavy snapshot is built here rather than every tick.
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);
/// After a preempted goal reports done, the backbone needs a beat to release its
/// single-flight gate before it will accept the queued move; the queued goal task
/// waits this long before firing.
pub const PREEMPT_GRACE: Duration = Duration::from_millis(50);

/// Operator input to the owner: a decoded wire command, or the WS closing (which drops
/// the deadman and restores the governor default, same as a released panel).
pub enum UiMsg {
    Command(Command),
    Disconnect,
}

/// State reported back to the owner from the always-on streams and the discrete goal
/// tasks. The owner is the only writer, so these are the sole way measured state and
/// goal outcomes reach `UiState`.
pub enum Feedback {
    ArmMeasured {
        side: Side,
        joints: [f64; ARM_DOF],
    },
    GripperMeasured {
        side: Side,
        opening: f64,
        // The gripper's reported effort ceiling (0 = no effort control).
        max_effort: f64,
    },
    Proximity(Proximity),
    // Boxed: seven readings dwarf every other variant, and this channel also
    // carries the high-rate measured-state stream.
    MotorHealth {
        side: Side,
        health: Box<ArmHealth>,
    },
    GripperMotorHealth {
        side: Side,
        health: GripperHealth,
    },
    Alert(Alert),
    ArmGoalDone {
        side: Side,
        summary: String,
    },
    GripperGoalDone {
        side: Side,
        summary: String,
    },
    // Live frame count from the in-flight record_episode goal's feedback.
    RecordProgress {
        frames: u64,
    },
    // The record goal reached a terminal state; the summary names the saved
    // (or discarded) episode.
    RecordDone {
        summary: String,
    },
    // finish_session answered; the summary names the finished session or the
    // refusal reason.
    SessionFinished {
        summary: String,
    },
}

/// The setpoints to stream this tick. `None` for a side means its deadman is off, so
/// the publisher emits nothing and the backbone holds the last setpoint. Recomputed
/// every command tick and read by the publisher tasks on their own cadence.
#[derive(Clone, Copy, Debug)]
pub struct CommandFrame {
    pub arms: BySide<Option<[f64; ARM_DOF]>>,
    pub grippers: BySide<Option<GripperFrame>>,
    pub governor: GovernorFrame,
}

/// One gripper's streamed setpoint: the opening fraction and the operator's
/// effort cap in its wire encoding (0 = no preference).
#[derive(Clone, Copy, Debug)]
pub struct GripperFrame {
    pub opening: f64,
    pub max_effort: f64,
}

/// The operator's self-collision governor controls, streamed continuously (no deadman:
/// the backbone must always know the operator's intent).
#[derive(Clone, Copy, Debug)]
pub struct GovernorFrame {
    pub collision_enabled: bool,
    pub d_stop: f64,
    pub d_safe: f64,
    pub max_ee_velocity_m_s: f64,
    pub max_gripper_rate_frac_s: f64,
}

impl CommandFrame {
    pub(crate) fn from_state(s: &UiState) -> Self {
        let arm = |side: Side| s.side_active(side).then_some(s.arms[side].joints);
        let grip = |side: Side| {
            s.side_active(side).then_some(GripperFrame {
                opening: s.grippers[side].position,
                max_effort: s.grippers[side].max_effort.unwrap_or(0.0),
            })
        };
        Self {
            arms: BySide::new(arm(Side::Left), arm(Side::Right)),
            grippers: BySide::new(grip(Side::Left), grip(Side::Right)),
            governor: GovernorFrame {
                collision_enabled: s.collision_enabled,
                d_stop: s.d_stop,
                d_safe: s.d_safe,
                max_ee_velocity_m_s: s.max_ee_velocity_m_s,
                max_gripper_rate_frac_s: s.max_gripper_rate_frac_s,
            },
        }
    }
}

/// A discrete arm move, resolved to what the goal task needs. A move that preempts an
/// in-flight one is stashed as one of these until the old goal reports done.
#[derive(Clone, Copy)]
enum ArmGoal {
    Joints {
        joints: [f64; ARM_DOF],
        duration_s: f64,
    },
    Pose {
        position: [f64; 3],
        orientation: [f64; 4],
        target_joints: [f64; ARM_DOF],
        duration_s: f64,
    },
}

impl ArmGoal {
    // The joint target to preview on the sliders while the move runs.
    fn preview(&self) -> [f64; ARM_DOF] {
        match self {
            Self::Joints { joints, .. } => *joints,
            Self::Pose { target_joints, .. } => *target_joints,
        }
    }

    fn action(&self) -> &'static str {
        match self {
            Self::Joints { .. } => "move_arm_joints",
            Self::Pose { .. } => "move_arm",
        }
    }
}

struct Owner {
    state: UiState,
    models: ArmModels,
    registry: Registry,
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    // Cloned into each spawned goal task so it can report its outcome back.
    feedback_tx: mpsc::Sender<Feedback>,
    // A discrete arm move queued behind the in-flight one it preempted; fired when that
    // one reports done (the backbone is single-flight, so they must not overlap).
    pending: BySide<Option<ArmGoal>>,
}

/// The owner's channel ends, grouped for [`run`]: operator commands and feedback
/// flow in, the command frame and the browser snapshot flow out. `feedback_tx` is
/// the sender end the owner clones into each goal task it spawns.
pub struct Channels {
    pub command_rx: mpsc::Receiver<UiMsg>,
    pub feedback_rx: mpsc::Receiver<Feedback>,
    pub feedback_tx: mpsc::Sender<Feedback>,
    pub frame_tx: watch::Sender<CommandFrame>,
    pub snapshot_tx: watch::Sender<String>,
}

/// Run the owner until shutdown. Owns `state`; every other task holds only a channel
/// end, so this is the one place `UiState` is read or written.
pub async fn run(
    state: UiState,
    models: ArmModels,
    registry: Registry,
    runner: Arc<NodeRunner>,
    command_period: Duration,
    token: CancellationToken,
    channels: Channels,
) {
    let Channels {
        mut command_rx,
        mut feedback_rx,
        feedback_tx,
        frame_tx,
        snapshot_tx,
    } = channels;
    let mut owner = Owner {
        state,
        models,
        registry,
        runner,
        token: token.clone(),
        feedback_tx,
        pending: BySide::new(None, None),
    };
    owner.state.recorder.available = record::available(&owner.runner);
    owner.state.health_bound = crate::motor_health::available(&owner.runner);
    owner.state.alerts_bound = crate::alerts::available(&owner.runner);

    // Publish the starting frame and snapshot before the first tick, so the publishers
    // and any already-connected browser see real state at once rather than after a tick.
    let _ = frame_tx.send(CommandFrame::from_state(&owner.state));
    if let Ok(json) =
        build_snapshot_json(&owner.state, Instant::now(), &owner.models, &owner.registry)
    {
        let _ = snapshot_tx.send(json);
    }

    // The jog integrator steps by exactly the tick it is driven at, so both come
    // from the one period parsed at startup rather than from the rate twice.
    let tick_dt_s = command_period.as_secs_f64();
    let mut command_tick = interval(command_period);
    command_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut snapshot_tick = interval(SNAPSHOT_INTERVAL);
    snapshot_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            msg = command_rx.recv() => match msg {
                Some(msg) => owner.reduce_ui(msg),
                None => return, // the WS server is gone; nothing left to command
            },
            Some(fb) = feedback_rx.recv() => owner.reduce_feedback(fb),
            _ = command_tick.tick() => {
                let gesture_finished = owner.advance_gesture(tick_dt_s);
                owner.advance_jogs(tick_dt_s);
                let _ = frame_tx.send(CommandFrame::from_state(&owner.state));
                if gesture_finished {
                    owner.finish_gesture();
                }
            }
            _ = snapshot_tick.tick() => {
                match build_snapshot_json(&owner.state, Instant::now(), &owner.models, &owner.registry) {
                    Ok(json) => { let _ = snapshot_tx.send(json); }
                    Err(e) => tracing::warn!(error = %e, "serialize snapshot"),
                }
            }
        }
    }
}

impl Owner {
    // Step the playing gesture one tick and write its samples into the involved
    // sides' targets; the frame publisher then streams them because playback keeps
    // those sides active. Runs before the jog advance, which skips gesture sides
    // (their deadmen are off). Returns true on the completing tick; the caller
    // clears playback only after the frame carrying the final samples is
    // published, so the last sample reaches the wire.
    fn advance_gesture(&mut self, tick_dt_s: f64) -> bool {
        let Some(pb) = self.state.gesture.as_ref() else {
            return false;
        };
        let (next, samples) = advance_gesture_step(pb, tick_dt_s);
        let done = next.is_none();
        if !done {
            self.state.gesture = next;
        }
        for side in SIDES {
            if let Some((joints, gripper)) = samples[side] {
                self.state.arms[side].joints = joints;
                if let Some(opening) = gripper {
                    self.state.grippers[side].position = opening;
                }
            }
        }
        done
    }

    // Release the completed gesture: called after the final frame is published.
    fn finish_gesture(&mut self) {
        if let Some(pb) = self.state.gesture.take() {
            self.state
                .set_status(format!("gesture '{}' complete", pb.gesture.label));
        }
    }

    // Step each enabled side's jog one tick and apply it. Only enabled sides advance
    // (a disabled side streams nothing, so walking its target would just drift the panel
    // off the measured pose). The step is pure (see `advance_jog`); caps re-derive from
    // the live EE-speed cap so retuning the knob changes jog speed.
    fn advance_jogs(&mut self, tick_dt_s: f64) {
        let caps = JogCaps::per_tick(
            tick_dt_s,
            self.state.max_ee_velocity_m_s,
            self.state.joint_jog_acceleration_rad_s2,
        );
        for side in SIDES {
            if !self.state.enabled[side] {
                continue;
            }
            let advance = advance_jog(&self.state.arms[side], side, &self.models, caps);
            apply_jog(&mut self.state, side, advance);
        }
    }

    fn reduce_ui(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Command(cmd) => self.reduce_command(cmd),
            UiMsg::Disconnect => self.on_disconnect(),
        }
    }

    fn on_disconnect(&mut self) {
        reset_on_disconnect(&mut self.state);
    }

    fn reduce_command(&mut self, cmd: Command) {
        match cmd {
            Command::FireArm {
                side,
                mut joints,
                duration_s,
            } => {
                let side: Side = side.into();
                // A discrete move preempts the live stream, so refuse one while enabled.
                if self.state.enabled[side] {
                    self.status(side, "disable before a discrete move");
                    return;
                }
                if self.state.gesture_holds(side) {
                    self.status(side, "gesture playing, not firing");
                    return;
                }
                clamp_to_limits(&mut joints, side);
                let target_ee = self.ee_position(side, &joints);
                let duration_s = self.floored_duration(side, target_ee, duration_s);
                self.fire_or_queue(side, ArmGoal::Joints { joints, duration_s });
            }
            Command::FireArmPose {
                side,
                position,
                orientation,
                duration_s,
            } => {
                let side: Side = side.into();
                if self.state.enabled[side] {
                    self.status(side, "disable before a discrete move");
                    return;
                }
                if self.state.gesture_holds(side) {
                    self.status(side, "gesture playing, not firing");
                    return;
                }
                if !position.iter().all(|v| v.is_finite()) {
                    self.status(side, "invalid position, not firing");
                    return;
                }
                let Some(rotation) = unit_quat_from_wire(orientation) else {
                    self.status(side, "invalid orientation, not firing");
                    return;
                };
                let seed = self.state.arms[side].joints;
                // Preview the pose as joints (seeded from the current target) so both the
                // sliders and the FK readout show where it is going, and reject an
                // unreachable pose up front rather than firing a goal the backbone refuses.
                let Some(mut target_joints) = self.models.solve_ik(side, position, rotation, &seed)
                else {
                    self.status(side, "pose unreachable, not firing");
                    return;
                };
                clamp_to_limits(&mut target_joints, side);
                let duration_s = self.floored_duration(side, position, duration_s);
                // Send the backbone the normalized quaternion, not the raw wire values.
                let orientation = [rotation.i, rotation.j, rotation.k, rotation.w];
                self.fire_or_queue(
                    side,
                    ArmGoal::Pose {
                        position,
                        orientation,
                        target_joints,
                        duration_s,
                    },
                );
            }
            Command::SetEnabled { side, on } => {
                let side: Side = side.into();
                if on {
                    // A discrete move owns the arm until its result lands; enabling now
                    // would fight it and snap back when it completes.
                    if self.state.arms[side].in_flight {
                        self.status(side, "move in flight, not enabling");
                        return;
                    }
                    if self.state.gesture_holds(side) {
                        self.status(side, "gesture playing, not enabling");
                        return;
                    }
                    // Refuse until measurements exist so the first emitted command holds
                    // position instead of a stale default. The arm target keeps its
                    // retained value (never re-seeded from the sagging measured pose);
                    // the gripper does not sag, so seed it from measured.
                    let (Some(_), Some(gripper_measured)) = (
                        self.state.arms[side].last_feedback,
                        self.state.grippers[side].last_feedback,
                    ) else {
                        self.status(side, "no measured pose yet, not enabling");
                        return;
                    };
                    self.state.grippers[side].position = gripper_measured;
                }
                // A jog must not survive across a deadman edge in either direction.
                self.state.arms[side].jog = None;
                self.state.arms[side].jog_blocked = false;
                self.state.enabled[side] = on;
                // Side-level (arm + gripper share the deadman), so no "arm" prefix.
                self.state.set_status(format!(
                    "{}: {}",
                    side.label(),
                    if on {
                        "ENABLED, streaming arm + gripper"
                    } else {
                        "disabled"
                    }
                ));
            }
            Command::SetArmTarget { side, mut joints } => {
                let side: Side = side.into();
                clamp_to_limits(&mut joints, side);
                if self.state.enabled[side] {
                    // Arm (or re-target) a joint jog: the tick ramps the setpoint toward
                    // the slider under a velocity/acceleration cap. Preserve the carried
                    // jog velocity when a live drag re-targets, so a continuous drag keeps
                    // its momentum; start from rest when arming fresh or switching from a
                    // Cartesian jog (the spaces must not fight). Its status latch resets.
                    let vel = match self.state.arms[side].jog {
                        Some(Jog::Joints { vel, .. }) => vel,
                        _ => [0.0; ARM_DOF],
                    };
                    self.state.arms[side].jog = Some(Jog::Joints {
                        target: joints,
                        vel,
                    });
                    self.state.arms[side].jog_blocked = false;
                }
            }
            Command::SetArmPose {
                side,
                position,
                orientation,
                arm_angle,
                mode,
            } => {
                let side: Side = side.into();
                if !position
                    .iter()
                    .chain(std::iter::once(&arm_angle))
                    .all(|v| v.is_finite())
                {
                    return;
                }
                let Some(rotation) = unit_quat_from_wire(orientation) else {
                    return;
                };
                // Store the desired pose as euler for the jog, which re-derives a
                // quaternion each step (so the euler encoding never drives interpolation).
                let (roll, pitch, yaw) = rotation.euler_angles();
                let pose: Pose = [position[0], position[1], position[2], roll, pitch, yaw];
                if self.state.enabled[side] {
                    self.state.arms[side].jog = Some(Jog::Cartesian(CartesianJog {
                        mode: mode.into(),
                        desired: pose,
                        arm_angle,
                    }));
                }
            }
            Command::SetGripperTarget { side, position } => {
                let side: Side = side.into();
                let [lo, hi] = gripper_limits();
                let position = position.clamp(lo, hi);
                if self.state.enabled[side] {
                    self.state.grippers[side].position = position;
                }
            }
            Command::SetGripperEffort { side, max_effort } => {
                let side: Side = side.into();
                if !max_effort.is_finite() || max_effort <= 0.0 {
                    return;
                }
                // Applies in both modes (streamed setpoints and discrete moves), so
                // it is not gated on the deadman the way the opening target is.
                // Clamped to the reported ceiling; without one the gripper's own
                // min() against its ceiling still bounds it.
                let ceiling = self.state.grippers[side].effort_ceiling;
                let capped = ceiling
                    .filter(|c| *c > 0.0)
                    .map_or(max_effort, |c| max_effort.min(c));
                self.state.grippers[side].max_effort = Some(capped);
            }
            Command::FireGripper { side, position } => {
                let side: Side = side.into();
                if !position.is_finite() {
                    return;
                }
                let [lo, hi] = gripper_limits();
                let position = position.clamp(lo, hi);
                // Disabled for a discrete move, and only one gripper goal in flight;
                // refuse rather than preempt (the moves are short).
                if self.state.enabled[side] {
                    self.status_gripper(side, "disable before a discrete move");
                    return;
                }
                if self.state.gesture_holds(side) {
                    self.status_gripper(side, "gesture playing, not firing");
                    return;
                }
                if self.state.grippers[side].in_flight {
                    self.status_gripper(side, "previous move still finishing");
                    return;
                }
                self.state.grippers[side].in_flight = true;
                self.state.grippers[side].position = position;
                self.status_gripper(side, "firing move_gripper");
                move_gripper::spawn(
                    self.runner.clone(),
                    self.feedback_tx.clone(),
                    self.token.clone(),
                    side,
                    position,
                    self.state.grippers[side].max_effort.unwrap_or(0.0),
                );
            }
            Command::RunGesture { name } => {
                let Some(gesture) = self.registry.find(&name) else {
                    self.state.set_status(format!("unknown gesture '{name}'"));
                    return;
                };
                let message = match start_gesture(&mut self.state, &self.models, gesture) {
                    Ok(m) | Err(m) => m,
                };
                self.state.set_status(message);
            }
            Command::StopGesture => match self.state.gesture.take() {
                Some(pb) => {
                    // The involved sides go inactive next frame; the backbone holds
                    // their last governed setpoint, the same release as a jog deadman.
                    self.state
                        .set_status(format!("gesture '{}' stopped", pb.gesture.label));
                }
                None => self.state.set_status("no gesture playing"),
            },
            Command::SetCollision { enabled } => {
                self.state.collision_enabled = enabled;
                self.state
                    .set_status(format!("collision avoidance {}", on_off(enabled)));
            }
            Command::StartRecording { task } => {
                if !self.state.recorder.available {
                    self.state.set_status("no recorder in this deployment");
                    return;
                }
                if self.state.recorder.episode.is_some() {
                    self.state.set_status("already recording");
                    return;
                }
                if self.state.recorder.finishing {
                    self.state
                        .set_status("finishing the session; wait for it to complete");
                    return;
                }
                let task = task.trim().to_string();
                if task.is_empty() {
                    self.state.set_status("name the task before recording");
                    return;
                }
                let stop = GoalToken::new();
                self.state.recorder.episode = Some(RecordingEpisode {
                    frames: 0,
                    stop: stop.clone(),
                });
                self.state.set_status(format!("recording '{task}'"));
                record::spawn(
                    self.runner.clone(),
                    self.feedback_tx.clone(),
                    self.token.clone(),
                    stop,
                    task,
                );
            }
            Command::FinishSession => {
                if !self.state.recorder.available {
                    self.state.set_status("no recorder in this deployment");
                    return;
                }
                if self.state.recorder.episode.is_some() {
                    self.state
                        .set_status("stop recording before finishing the session");
                    return;
                }
                if self.state.recorder.finishing {
                    self.state.set_status("already finishing");
                    return;
                }
                self.state.recorder.finishing = true;
                self.state
                    .set_status("finishing session (finalize + mirror)");
                record::spawn_finish(self.runner.clone(), self.feedback_tx.clone());
            }
            Command::StopRecording => match &self.state.recorder.episode {
                Some(episode) if episode.stop.is_cancelled() => {
                    self.state.set_status("still saving the episode");
                }
                Some(episode) => {
                    episode.stop.cancel();
                    self.state.set_status("stopping episode, saving");
                }
                None => self.state.set_status("not recording"),
            },
            Command::SetGovernorParams {
                d_stop,
                d_safe,
                max_ee_velocity_m_s,
                max_gripper_rate_frac_s,
            } => {
                // The backbone validates again before applying; reject a degenerate band here
                // so the UI cannot stream one.
                if !(valid_governor_band(d_stop, d_safe, max_ee_velocity_m_s)
                    && max_gripper_rate_frac_s.is_finite()
                    && max_gripper_rate_frac_s > 0.0)
                {
                    self.state.set_status(
                        "governor params ignored: require 0 < d_stop < d_safe and positive rates",
                    );
                    return;
                }
                self.state.d_stop = d_stop;
                self.state.d_safe = d_safe;
                self.state.max_ee_velocity_m_s = max_ee_velocity_m_s;
                self.state.max_gripper_rate_frac_s = max_gripper_rate_frac_s;
                self.state.set_status(format!(
                    "governor: d_stop={d_stop} d_safe={d_safe} max_ee={max_ee_velocity_m_s} m/s \
                     gripper opening={max_gripper_rate_frac_s} /s"
                ));
            }
        }
    }

    fn reduce_feedback(&mut self, fb: Feedback) {
        match fb {
            Feedback::ArmMeasured { side, joints } => {
                self.state.arms[side].last_feedback = Some(joints);
                // Initialize the streamed target from the first measured pose so the
                // panel starts where the arm is, then leave it: tracking measured while
                // disabled re-seeded the gravity-sagged pose and ratcheted the arm down.
                if !self.state.arms[side].established {
                    self.state.arms[side].joints = joints;
                    self.state.arms[side].established = true;
                }
            }
            Feedback::GripperMeasured {
                side,
                opening,
                max_effort,
            } => {
                self.state.grippers[side].last_feedback = Some(opening);
                self.state.grippers[side].effort_ceiling = Some(max_effort);
            }
            Feedback::Proximity(proximity) => {
                self.state.proximity = Some(proximity);
            }
            Feedback::GripperMotorHealth { side, health } => {
                self.state.gripper_health[side] = Some(health);
            }
            Feedback::MotorHealth { side, health } => {
                self.state.health[side] = Some(*health);
            }
            Feedback::Alert(alert) => {
                self.state.apply_alert(alert);
            }
            Feedback::ArmGoalDone { side, summary } => {
                self.state.arms[side].in_flight = false;
                self.state.arms[side].preempt = None;
                self.state.set_status(summary);
                // Fire any move queued behind the one that just finished; the backbone needs
                // a beat to release its single-flight gate first, so give it grace.
                if let Some(goal) = self.pending[side].take() {
                    self.fire_now(side, goal, true);
                }
            }
            Feedback::GripperGoalDone { side, summary } => {
                self.state.grippers[side].in_flight = false;
                self.state.set_status(summary);
            }
            Feedback::RecordProgress { frames } => {
                if let Some(episode) = &mut self.state.recorder.episode {
                    episode.frames = frames;
                }
            }
            Feedback::RecordDone { summary } => {
                self.state.recorder.episode = None;
                self.state.set_status(summary);
            }
            Feedback::SessionFinished { summary } => {
                self.state.recorder.finishing = false;
                self.state.set_status(summary);
            }
        }
    }

    // Fire a discrete arm move, or queue it behind the in-flight one it preempts. The
    // backbone is single-flight, so an overlapping fire cancels the running goal and waits
    // for its result (via `pending`) rather than racing a second goal in.
    fn fire_or_queue(&mut self, side: Side, goal: ArmGoal) {
        if self.state.arms[side].in_flight {
            if let Some(tok) = &self.state.arms[side].preempt {
                tok.cancel();
            }
            self.pending[side] = Some(goal);
            self.status(side, "preempting, move queued");
        } else {
            self.fire_now(side, goal, false);
        }
    }

    // Claim the side's move slot and spawn the goal task. `grace` makes the task wait
    // for the backbone to release its gate first (set when firing a queued preempt).
    fn fire_now(&mut self, side: Side, goal: ArmGoal, grace: bool) {
        // A preempt wait could have raced an Enable; never fire under a live deadman
        // or into a side a gesture holds.
        if self.state.enabled[side] {
            self.status(side, "enabled during preempt, move dropped");
            return;
        }
        if self.state.gesture_holds(side) {
            self.status(side, "gesture playing, queued move dropped");
            return;
        }
        let preempt = GoalToken::new();
        self.state.arms[side].in_flight = true;
        self.state.arms[side].preempt = Some(preempt.clone());
        // Retain the target so the panel mirrors where the move is going.
        self.state.arms[side].joints = goal.preview();
        let action = goal.action();
        let runner = self.runner.clone();
        let feedback = self.feedback_tx.clone();
        let token = self.token.clone();
        match goal {
            ArmGoal::Joints { joints, duration_s } => move_arm_joints::spawn(
                runner,
                feedback,
                token,
                preempt,
                move_arm_joints::Goal {
                    side,
                    joint_positions: joints,
                    duration_s,
                    grace,
                },
            ),
            ArmGoal::Pose {
                position,
                orientation,
                duration_s,
                ..
            } => move_arm::spawn(
                runner,
                feedback,
                token,
                preempt,
                move_arm::Goal {
                    side,
                    position,
                    orientation,
                    duration_s,
                    grace,
                },
            ),
        }
        self.status(side, &format!("firing {action}"));
    }

    // World-frame end-effector position (x, y, z) of a joint target.
    fn ee_position(&self, side: Side, joints: &[f64; ARM_DOF]) -> [f64; 3] {
        let p = self.models.ee_pose_world(side, joints);
        [p[0], p[1], p[2]]
    }

    // Floor a requested arm-move duration so the straight-line EE speed stays under the
    // governor cap (time >= distance / cap). With no measured pose yet, just sanitize.
    fn floored_duration(&self, side: Side, target_ee: [f64; 3], requested_s: f64) -> f64 {
        match self.state.arms[side].last_feedback {
            Some(measured) => {
                let dist = dist3(self.ee_position(side, &measured), target_ee);
                ee_speed_floored(
                    sane_duration(requested_s),
                    dist,
                    self.state.max_ee_velocity_m_s,
                )
            }
            None => sane_duration(requested_s),
        }
    }

    fn status(&mut self, side: Side, what: &str) {
        self.state
            .set_status(format!("{} arm: {what}", side.label()));
    }

    fn status_gripper(&mut self, side: Side, what: &str) {
        self.state
            .set_status(format!("{} gripper: {what}", side.label()));
    }
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "ON" } else { "OFF" }
}

/// Start a gesture: validate every involved side, then arm the lead-in blend
/// from the retained target (the setpoint the backbone already holds, so the
/// wire stays continuous). Validation runs over all sides before any mutation,
/// so a refusal on the second side cannot leave the first half-started.
fn start_gesture(
    s: &mut UiState,
    models: &ArmModels,
    gesture: Arc<BakedGesture>,
) -> Result<String, String> {
    if let Some(playing) = &s.gesture {
        return Err(format!(
            "gesture '{}' still playing, stop it first",
            playing.gesture.label
        ));
    }
    let mut from: BySide<Option<[f64; ARM_DOF]>> = BySide::new(None, None);
    let mut gripper_from: BySide<Option<f64>> = BySide::new(None, None);
    for side in SIDES {
        if !gesture.involves(side) {
            continue;
        }
        if s.enabled[side] {
            return Err(format!(
                "{}: disable streaming before a gesture",
                side.label()
            ));
        }
        if s.arms[side].in_flight || s.grippers[side].in_flight {
            return Err(format!(
                "{}: move in flight, not starting gesture",
                side.label()
            ));
        }
        if s.arms[side].last_feedback.is_none() || s.grippers[side].last_feedback.is_none() {
            return Err(format!(
                "{}: no measured pose yet, not starting gesture",
                side.label()
            ));
        }
        // Lead in from the retained target, not the measured pose: the backbone
        // holds the last commanded setpoint, so anchoring the blend there keeps
        // the wire continuous (re-seeding from the gravity-sagged measured pose
        // snapped the arm at gesture start). The gripper does not sag, so its
        // blend starts from the measured opening.
        from[side] = Some(s.arms[side].joints);
        gripper_from[side] = s.grippers[side].last_feedback;
    }
    let mut lead_s = 0.0_f64;
    for side in SIDES {
        let (Some(held), Some(first)) = (from[side], gesture.first_joints(side)) else {
            continue;
        };
        s.arms[side].jog = None;
        s.arms[side].jog_blocked = false;
        // Playback streams this side's gripper from its first frame; seed it from
        // measured so a gesture without gripper keys holds the jaw where it is.
        let gripper_measured = s.grippers[side].last_feedback.expect("validated above");
        s.grippers[side].position = gripper_measured;
        lead_s = lead_s.max(lead_in_duration(
            &held,
            &first,
            models.velocity_limits(side),
        ));
    }
    let label = gesture.label;
    s.gesture = Some(GesturePlayback {
        gesture,
        phase: GesturePhase::LeadIn {
            from,
            gripper_from,
            t: 0.0,
            duration_s: lead_s,
        },
    });
    Ok(format!("gesture '{label}': lead-in"))
}

/// A unit quaternion from the wire `[x, y, z, w]`, or `None` if it is non-finite or too
/// near zero to normalize. A degenerate orientation would normalize to NaN and poison
/// the IK solve, so both pose paths reject it here.
fn unit_quat_from_wire(q: [f64; 4]) -> Option<UnitQuaternion<f64>> {
    let quat = Quaternion::new(q[3], q[0], q[1], q[2]);
    (quat.norm() > 1e-6).then(|| UnitQuaternion::from_quaternion(quat))
}

// Reset on operator disconnect: drop the streaming deadman for both sides (each stream's
// timeout then releases to hold; the enabled gate also stops advancing their jogs) and
// restore the governor enable to its launch default, so an operator who left avoidance
// off cannot latch the backbone ungoverned, while a launch-ungoverned deployment is not
// force-armed either.
fn reset_on_disconnect(s: &mut UiState) {
    for side in SIDES {
        s.enabled[side] = false;
    }
    // No operator, no show: a gesture must not keep dancing after the panel is gone.
    s.gesture = None;
    s.collision_enabled = s.collision_enabled_default;
}

// --------------------------- per-tick jog advance ---------------------------

/// The outcome of advancing one side's jog a tick, ready for the caller to apply: the
/// reconciled joint setpoint, the jog to retain (`None` once it retires or reaches its
/// target), whether it is held at the reach boundary, and any moving <-> blocked
/// transition to announce. Pure, so the tick decision is testable without a live
/// [`UiState`] and the mutation is a straight assignment in [`apply_jog`].
#[derive(Clone, Copy, Debug)]
struct JogAdvance {
    joints: [f64; ARM_DOF],
    jog: Option<Jog>,
    blocked: bool,
    event: Option<JogEvent>,
}

/// A one-shot jog status transition. Emitted only on the edge, so a held boundary
/// reports once, not at the command rate.
#[derive(Clone, Copy, Debug)]
enum JogEvent {
    Moving { mode: JogMode },
    Blocked { mode: JogMode, desired: Pose },
}

// Advance one side's active jog by one tick. A joint jog walks the setpoint one
// acceleration-limited step toward the slider target and retires once it settles there; a
// Cartesian jog steps the joint target a capped increment toward the desired pose, holds it
// at the reach boundary, and retires once converged. Pure: `jog_tick` briefly takes the
// model lock inside it, but no UiState is held here, so the caller applies the result.
fn advance_jog(arm: &ArmTarget, side: Side, models: &ArmModels, caps: JogCaps) -> JogAdvance {
    let hold = |jog, blocked| JogAdvance {
        joints: arm.joints,
        jog,
        blocked,
        event: None,
    };
    let cartesian = match arm.jog {
        None => return hold(None, arm.jog_blocked),
        // Ramp the setpoint toward the slider under the velocity/acceleration cap; a joint
        // jog never blocks (targets are pre-clamped to limits), so no status event.
        Some(Jog::Joints { target, vel }) => {
            let (joints, jog) = match joint_jog_tick(&arm.joints, &target, &vel, caps) {
                JointJogStep::Converged(joints) => (joints, None),
                JointJogStep::Stepped { joints, vel } => {
                    (joints, Some(Jog::Joints { target, vel }))
                }
            };
            return JogAdvance {
                joints,
                jog,
                blocked: false,
                event: None,
            };
        }
        Some(Jog::Cartesian(cartesian)) => cartesian,
    };
    match jog_tick(models, side, &arm.joints, &cartesian, caps) {
        JogStep::Converged => hold(None, false),
        JogStep::Stepped(joints) => JogAdvance {
            joints,
            jog: arm.jog,
            blocked: false,
            // Announce resumption only when leaving a held boundary.
            event: arm.jog_blocked.then_some(JogEvent::Moving {
                mode: cartesian.mode,
            }),
        },
        JogStep::Blocked => JogAdvance {
            joints: arm.joints,
            jog: arm.jog,
            blocked: true,
            // Announce the boundary once, on entry.
            event: (!arm.jog_blocked).then_some(JogEvent::Blocked {
                mode: cartesian.mode,
                desired: cartesian.desired,
            }),
        },
    }
}

// Apply a computed jog advance to the side's target and emit its status transition, if
// any. The only mutation half of the pure/apply split above.
fn apply_jog(s: &mut UiState, side: Side, adv: JogAdvance) {
    s.arms[side].joints = adv.joints;
    s.arms[side].jog = adv.jog;
    s.arms[side].jog_blocked = adv.blocked;
    match adv.event {
        Some(JogEvent::Moving { mode }) => {
            s.set_status(format!("{}: pose jog moving", side.label()));
            tracing::info!(side = side.label(), ?mode, "pose jog resumed");
        }
        Some(JogEvent::Blocked { mode, desired }) => {
            s.set_status(format!("{}: pose at reach limit, holding", side.label()));
            tracing::info!(
                side = side.label(),
                ?mode,
                ?desired,
                "pose jog at reach limit"
            );
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::CartesianJog;
    use openarm_description::HardwareVersion;

    fn models() -> ArmModels {
        ArmModels::from_version(HardwareVersion::V2)
    }

    // The caps a 100 Hz tick at the sim launcher's 0.5 m/s knob derives.
    fn caps() -> JogCaps {
        JogCaps::per_tick(0.01, 0.5, 10.0)
    }

    fn arm(joints: [f64; ARM_DOF], jog: Option<Jog>, jog_blocked: bool) -> ArmTarget {
        let mut a = ArmTarget::home();
        a.joints = joints;
        a.jog = jog;
        a.jog_blocked = jog_blocked;
        a
    }

    fn position_jog(desired: Pose) -> Jog {
        Jog::Cartesian(CartesianJog {
            mode: JogMode::Position,
            desired,
            arm_angle: 0.0,
        })
    }

    fn joint_jog(target: [f64; ARM_DOF]) -> Jog {
        Jog::Joints {
            target,
            vel: [0.0; ARM_DOF],
        }
    }

    #[test]
    fn joint_jog_ramps_toward_the_target_without_snapping() {
        let target = [0.1, -0.2, 0.3, 0.9, -0.1, 0.2, 0.05];
        let a = arm([0.0; ARM_DOF], Some(joint_jog(target)), false);
        let adv = advance_jog(&a, Side::Left, &models(), caps());
        // The first tick moves only a capped increment, not the whole way (no snap), and
        // the jog stays armed to keep ramping.
        assert_ne!(
            adv.joints, target,
            "the setpoint does not snap to the target"
        );
        assert!(
            adv.joints
                .iter()
                .zip(target)
                .all(|(j, t)| j.abs() <= t.abs() + 1e-12)
        );
        assert!(adv.jog.is_some(), "the jog stays armed while ramping");
        assert!(!adv.blocked);
        assert!(adv.event.is_none());
    }

    #[test]
    fn joint_jog_converges_on_the_target_and_retires() {
        let m = models();
        let target = [0.1, -0.2, 0.3, 0.9, -0.1, 0.2, 0.05];
        let mut a = arm([0.0; ARM_DOF], Some(joint_jog(target)), false);
        // Thread each advance back into the arm exactly as the stream does.
        let mut converged = false;
        for _ in 0..2000 {
            let adv = advance_jog(&a, Side::Left, &m, caps());
            a.joints = adv.joints;
            a.jog = adv.jog;
            if adv.jog.is_none() {
                converged = true;
                assert_eq!(adv.joints, target, "it lands exactly on the target");
                break;
            }
        }
        assert!(converged, "the joint jog settles on the target and retires");
    }

    #[test]
    fn idle_arm_holds_its_setpoint() {
        let q = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let adv = advance_jog(&arm(q, None, false), Side::Left, &models(), caps());
        assert_eq!(adv.joints, q, "an idle side keeps its setpoint");
        assert!(adv.jog.is_none());
        assert!(adv.event.is_none());
    }

    #[test]
    fn cartesian_jog_on_the_current_pose_converges() {
        let m = models();
        let q = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let here = m.ee_pose_world(Side::Left, &q);
        let adv = advance_jog(
            &arm(q, Some(position_jog(here)), false),
            Side::Left,
            &m,
            caps(),
        );
        assert!(
            adv.jog.is_none(),
            "reaching the desired pose retires the jog"
        );
        assert!(!adv.blocked);
        assert!(adv.event.is_none());
    }

    #[test]
    fn boundary_is_announced_on_entry_not_while_held() {
        let m = models();
        let start = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0];
        let here = m.ee_pose_world(Side::Left, &start);
        let far = position_jog([here[0] + 2.0, here[1], here[2], here[3], here[4], here[5]]);
        // Drive the jog to the envelope, threading each advance back into the arm
        // exactly as the stream does, until it first reports the boundary.
        let mut a = arm(start, Some(far), false);
        let entry = loop_to_boundary(&m, &mut a);
        assert!(
            matches!(entry.event, Some(JogEvent::Blocked { .. })),
            "the boundary is announced when first entered"
        );
        assert!(
            a.jog.is_some(),
            "the jog stays armed so pulling back into reach resumes it"
        );
        // Held at the boundary (jog_blocked now set): the joints are pinned, so the next
        // tick blocks identically but must stay quiet.
        let held = advance_jog(&a, Side::Left, &m, caps());
        assert!(held.blocked, "the pinned arm stays at the boundary");
        assert!(held.event.is_none(), "a held boundary does not re-announce");
    }

    #[test]
    fn resuming_from_a_held_boundary_announces_moving() {
        let m = models();
        let q = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let here = m.ee_pose_world(Side::Left, &q);
        // A reachable nudge so the step advances, flagged as previously held.
        let near = position_jog([here[0] + 0.02, here[1], here[2], here[3], here[4], here[5]]);
        let adv = advance_jog(&arm(q, Some(near), true), Side::Left, &m, caps());
        assert!(!adv.blocked);
        assert!(
            matches!(adv.event, Some(JogEvent::Moving { .. })),
            "leaving the boundary announces movement"
        );
    }

    // Advance until the jog first reports the boundary, applying each result to `a`
    // like [`apply_jog`] does. Returns the entering advance; panics if it never blocks.
    fn loop_to_boundary(m: &ArmModels, a: &mut ArmTarget) -> JogAdvance {
        for _ in 0..5000 {
            let adv = advance_jog(a, Side::Left, m, caps());
            a.joints = adv.joints;
            a.jog = adv.jog;
            a.jog_blocked = adv.blocked;
            if adv.blocked {
                return adv;
            }
            assert!(
                a.jog.is_some(),
                "the jog retired before reaching the boundary"
            );
        }
        panic!("jog never reached the boundary within 5000 ticks");
    }

    // The baked roster, shared across tests (baking runs the IK rollouts once).
    fn registry() -> &'static Registry {
        static REG: std::sync::OnceLock<Registry> = std::sync::OnceLock::new();
        REG.get_or_init(|| Registry::bake(&models()))
    }

    // A state with measurements on both sides, as RunGesture requires.
    fn measured_state() -> UiState {
        let mut s = UiState::new(true, 0.005, 0.02, 0.25, 6.0, 10.0);
        for side in SIDES {
            s.arms[side].last_feedback = Some([0.1, 0.2, -0.1, 0.8, 0.0, 0.1, 0.0]);
            s.grippers[side].last_feedback = Some(0.3);
        }
        s
    }

    #[test]
    fn start_gesture_leads_in_from_the_retained_target() {
        let m = models();
        let mut s = measured_state();
        s.arms[Side::Right].joints = [0.2, 0.3, -0.2, 1.1, 0.0, 0.0, 0.1];
        s.arms[Side::Right].jog = Some(joint_jog([0.2; ARM_DOF]));
        start_gesture(&mut s, &m, registry().find("wave").unwrap()).expect("starts");
        let pb = s.gesture.as_ref().expect("playback armed");
        let GesturePhase::LeadIn {
            from, duration_s, ..
        } = pb.phase
        else {
            panic!("starts in lead-in");
        };
        // The blend anchors on the retained target (the backbone's held
        // setpoint), never the sagged measured pose.
        assert_eq!(from[Side::Right], Some(s.arms[Side::Right].joints));
        assert_ne!(from[Side::Right], s.arms[Side::Right].last_feedback);
        assert_eq!(from[Side::Left], None, "wave leaves the left arm alone");
        assert!((0.8..=4.0).contains(&duration_s));
        assert!(s.arms[Side::Right].jog.is_none(), "playback clears the jog");
        assert_eq!(
            s.grippers[Side::Right].position,
            0.3,
            "gripper seeds from measured"
        );
        assert!(s.gesture_holds(Side::Right) && !s.gesture_holds(Side::Left));
    }

    #[test]
    fn start_gesture_refusals_leave_state_untouched() {
        let m = models();
        let wave = registry().find("wave").unwrap();

        let mut enabled = measured_state();
        enabled.enabled[Side::Right] = true;
        assert!(start_gesture(&mut enabled, &m, wave.clone()).is_err());
        assert!(enabled.gesture.is_none());

        let mut in_flight = measured_state();
        in_flight.arms[Side::Right].in_flight = true;
        assert!(start_gesture(&mut in_flight, &m, wave.clone()).is_err());
        assert!(in_flight.gesture.is_none());

        let mut blind = measured_state();
        blind.arms[Side::Right].last_feedback = None;
        assert!(start_gesture(&mut blind, &m, wave.clone()).is_err());
        assert!(blind.gesture.is_none());

        let mut busy = measured_state();
        start_gesture(&mut busy, &m, wave.clone()).expect("first start");
        assert!(
            start_gesture(&mut busy, &m, wave).is_err(),
            "one gesture at a time"
        );

        // A dual-arm gesture validates the uninvolved-looking side too: a left
        // refusal must not half-start the right.
        let mut half = measured_state();
        half.enabled[Side::Left] = true;
        let shrug = registry().find("shrug").unwrap();
        assert!(start_gesture(&mut half, &m, shrug).is_err());
        assert!(half.gesture.is_none());
        assert!(half.arms[Side::Right].jog.is_none());
    }

    #[test]
    fn gesture_sides_stream_in_the_command_frame() {
        let m = models();
        let mut s = measured_state();
        start_gesture(&mut s, &m, registry().find("wave").unwrap()).expect("starts");
        assert!(!s.enabled[Side::Right], "deadman stays off under playback");
        let frame = CommandFrame::from_state(&s);
        assert!(frame.arms[Side::Right].is_some(), "gesture side streams");
        assert!(frame.grippers[Side::Right].is_some());
        assert!(
            frame.arms[Side::Left].is_none(),
            "uninvolved side stays quiet"
        );
        s.gesture = None;
        let released = CommandFrame::from_state(&s);
        assert!(
            released.arms[Side::Right].is_none(),
            "stop releases the side"
        );
    }

    #[test]
    fn disconnect_kills_gesture_playback() {
        let m = models();
        let mut s = measured_state();
        start_gesture(&mut s, &m, registry().find("wave").unwrap()).expect("starts");
        reset_on_disconnect(&mut s);
        assert!(s.gesture.is_none(), "no operator, no show");
    }

    #[test]
    fn disconnect_disarms_sides_and_restores_governor_default_on() {
        // Launched with avoidance on; operator turned it off with both sides armed.
        let mut s = UiState::new(true, 0.005, 0.02, 0.25, 6.0, 10.0);
        s.collision_enabled = false;
        s.enabled[Side::Left] = true;
        s.enabled[Side::Right] = true;
        reset_on_disconnect(&mut s);
        assert!(
            !s.enabled[Side::Left] && !s.enabled[Side::Right],
            "disconnect must drop the deadman for both sides"
        );
        assert!(
            s.collision_enabled,
            "disconnect must restore the launch governor default (on)"
        );
    }

    #[test]
    fn disconnect_restores_governor_default_off_when_launched_ungoverned() {
        // Launched deliberately ungoverned; operator turned avoidance on.
        let mut s = UiState::new(false, 0.005, 0.02, 0.25, 6.0, 10.0);
        s.collision_enabled = true;
        reset_on_disconnect(&mut s);
        assert!(
            !s.collision_enabled,
            "disconnect must restore the launch default (off), not force on"
        );
    }
}
