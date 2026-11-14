//! The state owner: a single task that owns `UiState` outright, so nothing else ever
//! locks it. Everyone talks to it over channels instead of a shared mutex.
//!
//!   - the WS server sends operator input as `UiMsg` (a wire `Command` or a
//!     disconnect) on the command channel;
//!   - the state streams (arm/gripper/collision/motor health/alerts) and the
//!     discrete goal tasks send measured state and goal outcomes as `Feedback`;
//!   - the owner publishes the browser snapshot (pre-serialized) and the per-tick
//!     `CommandFrame` (the setpoints to stream) on two watch channels.
//!
//! The owner is also the single motion authority: once per tick it advances each
//! arm's joint jog and republishes the frame. Because it is the only writer, the
//! tick decision and every reducer are ordinary synchronous functions on owned
//! state, with no lock discipline to reason about.

use std::sync::Arc;
use std::time::{Duration, Instant};

use peppygen::NodeRunner;
use peppylib::runtime::CancellationToken;
use robot_description::{ArmId, PerArm};
use srs_model::nalgebra::{Quaternion, UnitQuaternion};
use srs_model::{ARM_DOF, JointVec};
use tokio::sync::{mpsc, watch};
use tokio::time::{MissedTickBehavior, interval};
use tokio_util::sync::CancellationToken as GoalToken;

use crate::command_stream::CommandFrame;
use crate::gestures::{BakedGesture, Registry, advance_gesture_step, lead_in_duration};
use crate::limits::{clamp_to_limits, gripper_limits};
use crate::pose::{ArmModels, JogCaps, JointJog, JointJogStep, dist3, joint_jog_tick};
use crate::robot::{self, ARMS};
use crate::snapshot::build_snapshot_json;
use crate::state::{
    ArmHealth, ArmTarget, GesturePhase, GesturePlayback, GovernorSettings, GripperHealth, Limb,
    Proximity, RecordingEpisode, UiState,
};
use crate::ui::Command;
use crate::{alerts, motor_health, move_arm, move_arm_joints, move_gripper, record};

/// The browser snapshot cadence (10 Hz); the command tick runs far faster, so the
/// FK-heavy snapshot is built here rather than every tick.
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);

/// Operator input to the owner: a decoded wire command, or the WS closing (which drops
/// the deadman and restores the governor default, same as a released panel).
pub enum UiMsg {
    Command(Command),
    /// A command the panel sent that the server could not read, and why.
    Unreadable(String),
    Disconnect,
}

/// State reported back to the owner from the always-on streams and the discrete goal
/// tasks. The owner is the only writer, so these are the sole way measured state and
/// goal outcomes reach `UiState`.
pub enum Feedback {
    ArmMeasured {
        arm: ArmId,
        joints: JointVec,
    },
    GripperMeasured {
        arm: ArmId,
        opening: f64,
        // The gripper's reported effort ceiling (0 = no effort control).
        max_effort: f64,
    },
    Proximity(Proximity),
    // Boxed: seven readings dwarf every other variant, and this channel also
    // carries the high-rate measured-state stream.
    ArmMotorHealth {
        arm: ArmId,
        health: Box<ArmHealth>,
    },
    GripperMotorHealth {
        arm: ArmId,
        health: GripperHealth,
    },
    /// The set a producer published.
    AlertSet {
        producer: peppylib::messaging::ProducerRef,
        set: crate::state::AlertSet,
    },
    /// The producers the alerts slot holds, as of this delivery.
    AlertProducers(Vec<peppylib::messaging::ProducerRef>),
    ArmGoalDone {
        arm: ArmId,
        summary: String,
    },
    GripperGoalDone {
        arm: ArmId,
        summary: String,
    },
    // Live frame count from the in-flight record_episode goal's feedback.
    RecordProgress {
        frames: u64,
    },
    // The record goal reached a terminal state; the summary names the saved
    // episode, or why no episode was saved.
    RecordDone {
        summary: String,
    },
    // finish_session answered; the summary names the finished session or the
    // refusal reason.
    SessionFinished {
        summary: String,
    },
}

/// A discrete arm move, resolved to what the goal task needs. A move that preempts an
/// in-flight one is stashed as one of these until the old goal reports done.
#[derive(Clone, Copy)]
enum ArmGoal {
    Joints(move_arm_joints::Goal),
    Pose {
        goal: move_arm::Goal,
        // The pose solved to joints, for the sliders' preview while the move runs.
        target_joints: JointVec,
    },
}

impl ArmGoal {
    // The joint target to preview on the sliders while the move runs.
    fn preview(&self) -> JointVec {
        match self {
            Self::Joints(goal) => goal.joint_positions,
            Self::Pose { target_joints, .. } => *target_joints,
        }
    }

    fn action(&self) -> &'static str {
        match self {
            Self::Joints(_) => "move_arm_joints",
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
    pending: PerArm<Option<ArmGoal>, ARMS>,
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
        pending: PerArm::from_fn(|_| None),
    };
    owner.state.recorder.available = record::available(&owner.runner);
    owner.state.health_bound = motor_health::available(&owner.runner);
    // The slot's membership as it stands before the alert listener's first
    // delivery, so the opening snapshot says whether anything is wired. From
    // then on `Feedback::AlertProducers` carries it, and a producer that
    // joins later flips it.
    owner.state.alerts_bound = alerts::available(&owner.runner);

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
    // arms' targets; the frame publisher then streams them because playback keeps
    // those arms active. Runs before the jog advance, which skips gesture arms
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
        for arm in PerArm::<(), ARMS>::ids() {
            if let Some((joints, gripper)) = samples[arm] {
                self.state.arms[arm].joints = joints;
                if let Some(opening) = gripper {
                    self.state.grippers[arm].opening = opening;
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

    // Step each enabled arm's jog one tick and apply it. Only enabled arms advance
    // (a disabled arm streams nothing, so walking its target would just drift the panel
    // off the measured pose). The step is pure (see `advance_jog`).
    fn advance_jogs(&mut self, tick_dt_s: f64) {
        let caps = self.state.jog_caps;
        for arm in PerArm::<(), ARMS>::ids() {
            if !self.state.enabled[arm] {
                continue;
            }
            let JogAdvance { joints, jog } = advance_jog(&self.state.arms[arm], caps, tick_dt_s);
            self.state.arms[arm].joints = joints;
            self.state.arms[arm].jog = jog;
        }
    }

    fn reduce_ui(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Command(cmd) => self.reduce_command(cmd),
            UiMsg::Unreadable(reason) => self.state.set_status(format!(
                "the panel sent a command this node could not read: {reason}"
            )),
            UiMsg::Disconnect => self.on_disconnect(),
        }
    }

    fn on_disconnect(&mut self) {
        reset_on_disconnect(&mut self.state);
    }

    fn reduce_command(&mut self, cmd: Command) {
        match cmd {
            Command::FireArm {
                arm,
                mut joints,
                duration_s,
                settle_tolerance_rad,
                settle_timeout_s,
            } => {
                // A discrete move preempts the live stream, so refuse one while enabled.
                if self.state.enabled[arm] {
                    self.status(arm, Limb::Arm, "disable before a discrete move");
                    return;
                }
                if self.state.gesture_holds(arm) {
                    self.status(arm, Limb::Arm, "gesture playing, not firing");
                    return;
                }
                clamp_to_limits(&mut joints, arm);
                let target_ee = self.ee_position(arm, &joints);
                let duration_s = match self.floored_duration(arm, target_ee, duration_s) {
                    Ok(duration_s) => duration_s,
                    Err(refusal) => {
                        self.status(arm, Limb::Arm, &refusal);
                        return;
                    }
                };
                self.fire_or_queue(
                    arm,
                    ArmGoal::Joints(move_arm_joints::Goal {
                        arm,
                        joint_positions: joints,
                        duration_s,
                        settle_tolerance_rad,
                        settle_timeout_s,
                    }),
                );
            }
            Command::FireArmPose {
                arm,
                position,
                orientation,
                duration_s,
                settle_position_tolerance_m,
                settle_orientation_tolerance_rad,
                settle_timeout_s,
            } => {
                if self.state.enabled[arm] {
                    self.status(arm, Limb::Arm, "disable before a discrete move");
                    return;
                }
                if self.state.gesture_holds(arm) {
                    self.status(arm, Limb::Arm, "gesture playing, not firing");
                    return;
                }
                if !position.iter().all(|v| v.is_finite()) {
                    self.status(arm, Limb::Arm, "invalid position, not firing");
                    return;
                }
                let Some(rotation) = unit_quat_from_wire(orientation) else {
                    self.status(arm, Limb::Arm, "invalid orientation, not firing");
                    return;
                };
                let seed = self.state.arms[arm].joints;
                // Preview the pose as joints (seeded from the current target) so both
                // the sliders and the FK readout show where it is going. A pose with no
                // exact solution still fires: the backbone plans within the goal's
                // tolerances and can serve poses only approximately reachable, so its
                // decision is the refusal that matters. Without a solution the preview
                // holds the current joints, since there is no configuration to show.
                let target_joints = match self.models.solve_ik(arm, position, rotation, &seed) {
                    Some(mut solved) => {
                        clamp_to_limits(&mut solved, arm);
                        solved
                    }
                    None => seed,
                };
                let duration_s = match self.floored_duration(arm, position, duration_s) {
                    Ok(duration_s) => duration_s,
                    Err(refusal) => {
                        self.status(arm, Limb::Arm, &refusal);
                        return;
                    }
                };
                // Send the backbone the normalized quaternion, not the raw wire values.
                let orientation = [rotation.i, rotation.j, rotation.k, rotation.w];
                self.fire_or_queue(
                    arm,
                    ArmGoal::Pose {
                        goal: move_arm::Goal {
                            arm,
                            position,
                            orientation,
                            duration_s,
                            settle_position_tolerance_m,
                            settle_orientation_tolerance_rad,
                            settle_timeout_s,
                        },
                        target_joints,
                    },
                );
            }
            Command::SetEnabled { arm, on } => {
                if on {
                    // A discrete move owns the arm until its result lands; enabling now
                    // would fight it and snap back when it completes.
                    if self.state.arms[arm].in_flight {
                        self.status(arm, Limb::Arm, "move in flight, not enabling");
                        return;
                    }
                    if self.state.gesture_holds(arm) {
                        self.status(arm, Limb::Arm, "gesture playing, not enabling");
                        return;
                    }
                    // Refuse until measurements exist so the first emitted command holds
                    // position instead of a stale default. The arm target keeps its
                    // retained value (never re-seeded from the sagging measured pose);
                    // the gripper does not sag, so seed it from measured.
                    let (Some(_), Some(gripper_measured)) = (
                        self.state.arms[arm].last_feedback,
                        self.state.grippers[arm].last_feedback,
                    ) else {
                        self.status(arm, Limb::Arm, "no measured pose yet, not enabling");
                        return;
                    };
                    self.state.grippers[arm].opening = gripper_measured;
                }
                // A jog must not survive across a deadman edge in either direction.
                self.state.arms[arm].jog = None;
                self.state.enabled[arm] = on;
                // The arm and its gripper share the deadman, so the arm's name speaks for both.
                self.state.set_status(format!(
                    "{}: {}",
                    robot::LIMBS.arm_name(arm),
                    if on {
                        "ENABLED, streaming arm + gripper"
                    } else {
                        "disabled"
                    }
                ));
            }
            Command::SetArmTarget { arm, mut joints } => {
                clamp_to_limits(&mut joints, arm);
                if self.state.enabled[arm] {
                    // Arm (or re-target) the joint jog: the tick ramps the setpoint toward
                    // the slider under a velocity/acceleration cap. Preserve the carried
                    // jog velocity when a live drag re-targets, so a continuous drag keeps
                    // its momentum; start from rest when arming fresh.
                    let vel = self.state.arms[arm]
                        .jog
                        .map_or([0.0; ARM_DOF], |jog| jog.vel);
                    self.state.arms[arm].jog = Some(JointJog {
                        target: joints,
                        vel,
                    });
                }
            }
            Command::SetGripperTarget { arm, position } => {
                let [lo, hi] = gripper_limits();
                let position = position.clamp(lo, hi);
                if self.state.enabled[arm] {
                    self.state.grippers[arm].opening = position;
                }
            }
            Command::SetGripperEffort { arm, max_effort } => {
                if !max_effort.is_finite() || max_effort <= 0.0 {
                    return;
                }
                // Applies in both modes (streamed setpoints and discrete moves), so
                // it is not gated on the deadman the way the opening target is.
                // Clamped to the reported ceiling; without one the gripper's own
                // min() against its ceiling still bounds it.
                let ceiling = self.state.grippers[arm].effort_ceiling;
                let capped = ceiling
                    .filter(|c| *c > 0.0)
                    .map_or(max_effort, |c| max_effort.min(c));
                self.state.grippers[arm].max_effort = Some(capped);
            }
            Command::FireGripper {
                arm,
                position,
                settle_timeout_s,
            } => {
                if !position.is_finite() {
                    return;
                }
                let [lo, hi] = gripper_limits();
                let position = position.clamp(lo, hi);
                // Disabled for a discrete move, and only one gripper goal in flight;
                // refuse rather than preempt (the moves are short).
                if self.state.enabled[arm] {
                    self.status(arm, Limb::Gripper, "disable before a discrete move");
                    return;
                }
                if self.state.gesture_holds(arm) {
                    self.status(arm, Limb::Gripper, "gesture playing, not firing");
                    return;
                }
                if self.state.grippers[arm].in_flight {
                    self.status(arm, Limb::Gripper, "previous move still finishing");
                    return;
                }
                self.state.grippers[arm].in_flight = true;
                self.state.grippers[arm].opening = position;
                self.status(arm, Limb::Gripper, "firing move_gripper");
                move_gripper::spawn(
                    self.runner.clone(),
                    self.feedback_tx.clone(),
                    self.token.clone(),
                    move_gripper::Goal {
                        arm,
                        opening: position,
                        max_effort: self.state.grippers[arm].max_effort.unwrap_or(0.0),
                        settle_timeout_s,
                    },
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
                    // The involved arms go inactive next frame; the backbone holds
                    // their last governed setpoint, the same release as a jog deadman.
                    self.state
                        .set_status(format!("gesture '{}' stopped", pb.gesture.label));
                }
                None => self.state.set_status("no gesture playing"),
            },
            Command::SetCollision { enabled } => {
                self.state.governor.collision_enabled = enabled;
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
                max_stream_ee_velocity_m_s,
                max_gripper_rate_frac_s,
            } => {
                let requested = GovernorSettings {
                    d_stop,
                    d_safe,
                    max_stream_ee_velocity_m_s,
                    max_gripper_rate_frac_s,
                    ..self.state.governor
                };
                // The backbone validates again before applying; reject a degenerate band here
                // so the UI cannot stream one.
                if !requested.is_streamable() {
                    self.state.set_status(
                        "governor params ignored: require 0 < d_stop < d_safe and positive rates",
                    );
                    return;
                }
                self.state.governor = requested;
                self.state.set_status(format!(
                    "governor: d_stop={d_stop} d_safe={d_safe} max_ee={max_stream_ee_velocity_m_s} m/s \
                     gripper opening={max_gripper_rate_frac_s} /s"
                ));
            }
        }
    }

    fn reduce_feedback(&mut self, fb: Feedback) {
        match fb {
            Feedback::ArmMeasured { arm, joints } => {
                self.state.arms[arm].last_feedback = Some(joints);
                // Initialize the streamed target from the first measured pose so the
                // panel starts where the arm is, then leave it: tracking measured while
                // disabled re-seeded the gravity-sagged pose and ratcheted the arm down.
                if !self.state.arms[arm].established {
                    self.state.arms[arm].joints = joints;
                    self.state.arms[arm].established = true;
                }
            }
            Feedback::GripperMeasured {
                arm,
                opening,
                max_effort,
            } => {
                self.state.grippers[arm].last_feedback = Some(opening);
                self.state.grippers[arm].effort_ceiling = Some(max_effort);
            }
            Feedback::Proximity(proximity) => {
                self.state.proximity = Some(proximity);
            }
            Feedback::GripperMotorHealth { arm, health } => {
                self.state.gripper_health[arm] = Some(health);
            }
            Feedback::ArmMotorHealth { arm, health } => {
                self.state.arm_health[arm] = Some(*health);
            }
            Feedback::AlertSet { producer, set } => {
                self.state.apply_alert_set(producer, set);
            }
            Feedback::AlertProducers(producers) => {
                self.state.retain_alert_producers(&producers);
            }
            Feedback::ArmGoalDone { arm, summary } => {
                self.state.arms[arm].in_flight = false;
                self.state.arms[arm].preempt = None;
                self.state.set_status(summary);
                // Fire any move queued behind the one that just finished; the backbone needs
                // a beat to release its single-flight gate first, so give it grace.
                if let Some(goal) = self.pending[arm].take() {
                    self.fire_now(arm, goal, true);
                }
            }
            Feedback::GripperGoalDone { arm, summary } => {
                self.state.grippers[arm].in_flight = false;
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
    fn fire_or_queue(&mut self, arm: ArmId, goal: ArmGoal) {
        if self.state.arms[arm].in_flight {
            if let Some(tok) = &self.state.arms[arm].preempt {
                tok.cancel();
            }
            self.pending[arm] = Some(goal);
            self.status(arm, Limb::Arm, "preempting, move queued");
        } else {
            self.fire_now(arm, goal, false);
        }
    }

    // Claim the arm's move slot and spawn the goal task. `grace` makes the task wait
    // for the backbone to release its gate first (set when firing a queued preempt).
    fn fire_now(&mut self, arm: ArmId, goal: ArmGoal, grace: bool) {
        // A preempt wait could have raced an Enable; never fire under a live deadman
        // or into a arm a gesture holds.
        if self.state.enabled[arm] {
            self.status(arm, Limb::Arm, "enabled during preempt, move dropped");
            return;
        }
        if self.state.gesture_holds(arm) {
            self.status(arm, Limb::Arm, "gesture playing, queued move dropped");
            return;
        }
        let preempt = GoalToken::new();
        self.state.arms[arm].in_flight = true;
        self.state.arms[arm].preempt = Some(preempt.clone());
        // Retain the target so the panel mirrors where the move is going.
        self.state.arms[arm].joints = goal.preview();
        let action = goal.action();
        let runner = self.runner.clone();
        let feedback = self.feedback_tx.clone();
        let token = self.token.clone();
        match goal {
            ArmGoal::Joints(goal) => {
                move_arm_joints::spawn(runner, feedback, token, preempt, goal, grace)
            }
            ArmGoal::Pose { goal, .. } => {
                move_arm::spawn(runner, feedback, token, preempt, goal, grace)
            }
        }
        self.status(arm, Limb::Arm, &format!("firing {action}"));
    }

    // World-frame end-effector position (x, y, z) of a joint target.
    fn ee_position(&self, arm: ArmId, joints: &JointVec) -> [f64; 3] {
        let p = self.models.ee_pose_world(arm, joints);
        [p[0], p[1], p[2]]
    }

    // Floor a requested arm-move duration so the straight-line EE speed stays under the
    // governor cap (time >= distance / cap), or refuse a move that needs longer than a
    // move may take at that cap. With no measured pose yet, just sanitize.
    fn floored_duration(
        &self,
        arm: ArmId,
        target_ee: [f64; 3],
        requested_s: f64,
    ) -> Result<f64, String> {
        match self.state.arms[arm].last_feedback {
            Some(measured) => {
                let dist = dist3(self.ee_position(arm, &measured), target_ee);
                ee_speed_floored(
                    sane_duration(requested_s),
                    dist,
                    self.state.governor.max_stream_ee_velocity_m_s,
                )
            }
            None => Ok(sane_duration(requested_s)),
        }
    }

    fn status(&mut self, arm: ArmId, limb: Limb, what: &str) {
        self.state
            .set_status(format!("{}: {what}", limb.name_of(arm)));
    }
}

/// The longest duration a limb_motion goal may request: the contract's bound,
/// which the backbone refuses past.
const MAX_MOVE_DURATION_S: f64 = 60.0;

// Clamp a requested move duration into the contract's range (finite, 0..=60 s).
// 0 = fastest.
fn sane_duration(duration_s: f64) -> f64 {
    if duration_s.is_finite() {
        duration_s.clamp(0.0, MAX_MOVE_DURATION_S)
    } else {
        0.0
    }
}

// A discrete move's duration: the operator's request, floored so the straight-line
// EE speed never exceeds the governor cap (time >= distance / cap), or the refusal
// of a move that needs longer than the contract's bound at that cap. The backbone
// floors again at its joint-velocity limit, so this only ever slows a move.
fn ee_speed_floored(
    user_s: f64,
    ee_distance_m: f64,
    max_stream_ee_velocity_m_s: f64,
) -> Result<f64, String> {
    let floored = (ee_distance_m / max_stream_ee_velocity_m_s).max(user_s);
    if floored > MAX_MOVE_DURATION_S {
        return Err(format!(
            "this move needs {floored:.0} s at the {max_stream_ee_velocity_m_s} m/s speed cap, past the \
             {MAX_MOVE_DURATION_S:.0} s a move may take: raise the cap or move in steps"
        ));
    }
    Ok(floored)
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "ON" } else { "OFF" }
}

/// Start a gesture: validate every involved arm, then arm the lead-in blend
/// from the retained target (the setpoint the backbone already holds, so the
/// wire stays continuous). Validation runs over every arm before any mutation,
/// so a refusal on the second arm cannot leave the first half-started.
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
    for arm in PerArm::<(), ARMS>::ids().filter(|&arm| gesture.involves(arm)) {
        let name = robot::LIMBS.arm_name(arm);
        if s.enabled[arm] {
            return Err(format!("{name}: disable streaming before a gesture"));
        }
        if s.arms[arm].in_flight || s.grippers[arm].in_flight {
            return Err(format!("{name}: move in flight, not starting gesture"));
        }
        if s.arms[arm].last_feedback.is_none() || s.grippers[arm].last_feedback.is_none() {
            return Err(format!(
                "{name}: no measured pose yet, not starting gesture"
            ));
        }
    }
    // Lead in from the retained target: the backbone holds the last
    // commanded setpoint, so a blend anchored there keeps the wire
    // continuous at gesture start. The arm's measured pose sags under
    // gravity below that setpoint; the gripper does not sag, so its blend
    // starts from the measured opening.
    let from = PerArm::from_fn(|arm| gesture.involves(arm).then_some(s.arms[arm].joints));
    let gripper_from = PerArm::from_fn(|arm| {
        gesture
            .involves(arm)
            .then_some(s.grippers[arm].last_feedback)
            .flatten()
    });
    let mut lead_s = 0.0_f64;
    for arm in PerArm::<(), ARMS>::ids() {
        let (Some(held), Some(first)) = (from[arm], gesture.first_joints(arm)) else {
            continue;
        };
        s.arms[arm].jog = None;
        // Playback streams this arm's gripper from its first frame; seed it from
        // measured so a gesture without gripper keys holds the jaw where it is.
        let gripper_measured = s.grippers[arm].last_feedback.expect("validated above");
        s.grippers[arm].opening = gripper_measured;
        lead_s = lead_s.max(lead_in_duration(&held, &first, models.velocity_limits(arm)));
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
/// the IK solve, so the pose move rejects it here.
fn unit_quat_from_wire(q: [f64; 4]) -> Option<UnitQuaternion<f64>> {
    let quat = Quaternion::new(q[3], q[0], q[1], q[2]);
    (quat.norm() > 1e-6).then(|| UnitQuaternion::from_quaternion(quat))
}

// Reset on operator disconnect: drop the streaming deadman for every arm (each stream's
// timeout then releases to hold; the enabled gate also stops advancing their jogs) and
// restore the governor enable to its launch default, so an operator who left avoidance
// off cannot latch the backbone ungoverned, while a launch-ungoverned deployment is not
// force-armed either.
fn reset_on_disconnect(s: &mut UiState) {
    s.enabled = PerArm::from_fn(|_| false);
    // No operator, no show: a gesture must not keep dancing after the panel is gone.
    s.gesture = None;
    s.governor.collision_enabled = s.collision_enabled_default;
}

// --------------------------- per-tick jog advance ---------------------------

/// One arm's jog advanced a tick: the reconciled joint setpoint and the jog to
/// retain (`None` once it lands on its target and retires). Pure, so the tick
/// decision is testable without a live [`UiState`].
#[derive(Clone, Copy, Debug)]
struct JogAdvance {
    joints: JointVec,
    jog: Option<JointJog>,
}

// Advance one arm's jog by one tick: walk the setpoint one acceleration-limited step
// toward the slider target, and retire the jog once it settles there. An idle arm
// holds its setpoint.
fn advance_jog(arm: &ArmTarget, caps: JogCaps, dt_s: f64) -> JogAdvance {
    let Some(JointJog { target, vel }) = arm.jog else {
        return JogAdvance {
            joints: arm.joints,
            jog: None,
        };
    };
    match joint_jog_tick(&arm.joints, &target, &vel, caps, dt_s) {
        JointJogStep::Converged(joints) => JogAdvance { joints, jog: None },
        JointJogStep::Stepped { joints, vel } => JogAdvance {
            joints,
            jog: Some(JointJog { target, vel }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{LEFT, RIGHT, caps, governor, models, registry, ui_state};

    /// A 100 Hz command tick.
    const DT_S: f64 = 0.01;

    fn arm(joints: JointVec, jog: Option<JointJog>) -> ArmTarget {
        let mut a = ArmTarget::unmeasured();
        a.joints = joints;
        a.jog = jog;
        a
    }

    fn joint_jog(target: JointVec) -> JointJog {
        JointJog {
            target,
            vel: [0.0; ARM_DOF],
        }
    }

    #[test]
    fn a_move_too_slow_for_the_contract_bound_is_refused() {
        assert_eq!(
            ee_speed_floored(2.0, 0.8, 0.01),
            Err(
                "this move needs 80 s at the 0.01 m/s speed cap, past the 60 s a move may \
                 take: raise the cap or move in steps"
                    .to_string()
            )
        );
        assert_eq!(sane_duration(61.0), MAX_MOVE_DURATION_S);
        assert_eq!(ee_speed_floored(2.0, 0.1, 0.25), Ok(2.0));
    }

    #[test]
    fn joint_jog_ramps_toward_the_target_without_snapping() {
        let target = [0.1, -0.2, 0.3, 0.9, -0.1, 0.2, 0.05];
        let a = arm([0.0; ARM_DOF], Some(joint_jog(target)));
        let adv = advance_jog(&a, caps(), DT_S);
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
    }

    #[test]
    fn joint_jog_converges_on_the_target_and_retires() {
        let target = [0.1, -0.2, 0.3, 0.9, -0.1, 0.2, 0.05];
        let mut a = arm([0.0; ARM_DOF], Some(joint_jog(target)));
        // Thread each advance back into the arm exactly as the stream does.
        let mut converged = false;
        for _ in 0..2000 {
            let adv = advance_jog(&a, caps(), DT_S);
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
        let adv = advance_jog(&arm(q, None), caps(), DT_S);
        assert_eq!(adv.joints, q, "an idle arm keeps its setpoint");
        assert!(adv.jog.is_none());
    }

    // A state with measurements on both arms, as RunGesture requires.
    fn measured_state() -> UiState {
        let mut s = ui_state();
        for arm in PerArm::<(), ARMS>::ids() {
            s.arms[arm].last_feedback = Some([0.1, 0.2, -0.1, 0.8, 0.0, 0.1, 0.0]);
            s.grippers[arm].last_feedback = Some(0.3);
        }
        s
    }

    #[test]
    fn start_gesture_leads_in_from_the_retained_target() {
        let m = models();
        let mut s = measured_state();
        s.arms[RIGHT].joints = [0.2, 0.3, -0.2, 1.1, 0.0, 0.0, 0.1];
        s.arms[RIGHT].jog = Some(joint_jog([0.2; ARM_DOF]));
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
        assert_eq!(from[RIGHT], Some(s.arms[RIGHT].joints));
        assert_ne!(from[RIGHT], s.arms[RIGHT].last_feedback);
        assert_eq!(from[LEFT], None, "wave leaves the left arm alone");
        assert!((0.8..=4.0).contains(&duration_s));
        assert!(s.arms[RIGHT].jog.is_none(), "playback clears the jog");
        assert_eq!(
            s.grippers[RIGHT].opening, 0.3,
            "gripper seeds from measured"
        );
        assert!(s.gesture_holds(RIGHT) && !s.gesture_holds(LEFT));
    }

    #[test]
    fn start_gesture_refusals_leave_state_untouched() {
        let m = models();
        let wave = registry().find("wave").unwrap();

        let mut enabled = measured_state();
        enabled.enabled[RIGHT] = true;
        assert!(start_gesture(&mut enabled, &m, wave.clone()).is_err());
        assert!(enabled.gesture.is_none());

        let mut in_flight = measured_state();
        in_flight.arms[RIGHT].in_flight = true;
        assert!(start_gesture(&mut in_flight, &m, wave.clone()).is_err());
        assert!(in_flight.gesture.is_none());

        let mut blind = measured_state();
        blind.arms[RIGHT].last_feedback = None;
        assert!(start_gesture(&mut blind, &m, wave.clone()).is_err());
        assert!(blind.gesture.is_none());

        let mut busy = measured_state();
        start_gesture(&mut busy, &m, wave.clone()).expect("first start");
        assert!(
            start_gesture(&mut busy, &m, wave).is_err(),
            "one gesture at a time"
        );

        // A dual-arm gesture validates the uninvolved-looking arm too: a left
        // refusal must not half-start the right.
        let mut half = measured_state();
        half.enabled[LEFT] = true;
        let shrug = registry().find("shrug").unwrap();
        assert!(start_gesture(&mut half, &m, shrug).is_err());
        assert!(half.gesture.is_none());
        assert!(half.arms[RIGHT].jog.is_none());
    }

    #[test]
    fn gesture_arms_stream_in_the_command_frame() {
        let m = models();
        let mut s = measured_state();
        start_gesture(&mut s, &m, registry().find("wave").unwrap()).expect("starts");
        assert!(!s.enabled[RIGHT], "deadman stays off under playback");
        let frame = CommandFrame::from_state(&s);
        assert!(frame.arms[RIGHT].is_some(), "gesture arm streams");
        assert!(frame.grippers[RIGHT].is_some());
        assert!(frame.arms[LEFT].is_none(), "uninvolved arm stays quiet");
        s.gesture = None;
        let released = CommandFrame::from_state(&s);
        assert!(released.arms[RIGHT].is_none(), "stop releases the arm");
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
    fn disconnect_disarms_both_arms_and_restores_governor_default_on() {
        // Launched with avoidance on; operator turned it off with both arms armed.
        let mut s = ui_state();
        s.governor.collision_enabled = false;
        s.enabled[LEFT] = true;
        s.enabled[RIGHT] = true;
        reset_on_disconnect(&mut s);
        assert!(
            !s.enabled[LEFT] && !s.enabled[RIGHT],
            "disconnect must drop the deadman for both arms"
        );
        assert!(
            s.governor.collision_enabled,
            "disconnect must restore the launch governor default (on)"
        );
    }

    #[test]
    fn disconnect_restores_governor_default_off_when_launched_ungoverned() {
        // Launched deliberately ungoverned; operator turned avoidance on.
        let mut s = UiState::new(governor(false), caps());
        s.governor.collision_enabled = true;
        reset_on_disconnect(&mut s);
        assert!(
            !s.governor.collision_enabled,
            "disconnect must restore the launch default (off), not force on"
        );
    }
}
