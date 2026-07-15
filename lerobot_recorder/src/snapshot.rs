//! Latest-value cache between drain tasks and the fps sampler, and the pure
//! zero-order-hold sampling function. Sizes are set at startup from the
//! [`RecordingPlan`]; the sampler reads whatever has already arrived (never
//! waits for a fresher value, so no future leakage into the dataset).

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::watch;

use crate::config::Config;
use crate::plan::RecordingPlan;
use crate::types::FrameBuf;

/// One measured joint group. Dimension names are not on the wire (the peppy
/// Rust generator does not support string-array fields yet); the recorder
/// derives them from the producer identity and joint index.
#[derive(Debug, Clone)]
pub struct JointSample {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
}

/// One commanded joint group.
#[derive(Debug, Clone)]
pub struct CommandSample {
    pub positions: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Stamped<T> {
    pub value: T,
    pub at: Instant,
}

impl<T> Stamped<T> {
    pub fn now(value: T) -> Self {
        Stamped {
            value,
            at: Instant::now(),
        }
    }
    pub fn age(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.at)
    }
}

pub type Slot<T> = watch::Sender<Option<Stamped<T>>>;
type Rx<T> = watch::Receiver<Option<Stamped<T>>>;

/// Writer half, held by the drain tasks.
pub struct CacheWriter {
    pub states: Vec<Slot<JointSample>>,
    pub actions: Vec<Slot<CommandSample>>,
    pub cameras: Vec<Slot<Arc<FrameBuf>>>,
}

/// Reader half, held by the episode manager.
pub struct CacheReader {
    states: Vec<Rx<JointSample>>,
    actions: Vec<Rx<CommandSample>>,
    cameras: Vec<Rx<Arc<FrameBuf>>>,
}

pub fn cache(plan: &RecordingPlan) -> (CacheWriter, CacheReader) {
    fn slots<T>(n: usize) -> (Vec<Slot<T>>, Vec<Rx<T>>) {
        (0..n).map(|_| watch::channel(None)).unzip()
    }
    let (state_tx, state_rx) = slots(plan.state_keys.len());
    let (action_tx, action_rx) = slots(plan.action_keys.len());
    let (cam_tx, cam_rx) = slots(plan.camera_count());
    (
        CacheWriter {
            states: state_tx,
            actions: action_tx,
            cameras: cam_tx,
        },
        CacheReader {
            states: state_rx,
            actions: action_rx,
            cameras: cam_rx,
        },
    )
}

/// A copy of the cache at one instant.
pub struct CacheView {
    pub states: Vec<Option<Stamped<JointSample>>>,
    pub actions: Vec<Option<Stamped<CommandSample>>>,
    pub cameras: Vec<Option<Stamped<Arc<FrameBuf>>>>,
}

impl CacheReader {
    pub fn view(&self) -> CacheView {
        CacheView {
            states: self.states.iter().map(|r| r.borrow().clone()).collect(),
            actions: self.actions.iter().map(|r| r.borrow().clone()).collect(),
            cameras: self.cameras.iter().map(|r| r.borrow().clone()).collect(),
        }
    }
}

/// The dataset schema captured once every source has delivered its first
/// message: dimension names and whether a velocity feature exists.
#[derive(Debug, Clone)]
pub struct SourceSchema {
    pub state_names: Vec<String>,
    pub velocity_names: Vec<String>,
    pub action_names: Vec<String>,
    pub has_velocity: bool,
    /// True when there are no action sources: the action mirrors the state.
    pub action_is_state: bool,
}

impl SourceSchema {
    /// Builds the schema from the current cache view, deriving per-joint names
    /// from each source's producer identity plus joint index. Every state
    /// source needs a message; a velocity feature is included only when every
    /// state source reports velocities.
    pub fn from_view(view: &CacheView, plan: &RecordingPlan) -> Result<SourceSchema, String> {
        let mut state_names = Vec::new();
        let mut velocity_names = Vec::new();
        let mut has_velocity = true;
        for (i, slot) in view.states.iter().enumerate() {
            let s = slot
                .as_ref()
                .ok_or_else(|| format!("state source {i} has not produced yet"))?;
            let prefix = crate::config::sanitize_key(&plan.state_keys[i].instance_id);
            for j in 0..s.value.positions.len() {
                state_names.push(format!("{prefix}_j{j}"));
            }
            if s.value.velocities.len() == s.value.positions.len() {
                for j in 0..s.value.velocities.len() {
                    velocity_names.push(format!("{prefix}_j{j}"));
                }
            } else {
                has_velocity = false;
            }
        }
        if state_names.is_empty() {
            return Err("no state joints reported".to_string());
        }

        let action_is_state = view.actions.is_empty();
        let action_names = if action_is_state {
            state_names.clone()
        } else {
            let mut names = Vec::new();
            for (i, slot) in view.actions.iter().enumerate() {
                let a = slot
                    .as_ref()
                    .ok_or_else(|| format!("action source {i} has not produced yet"))?;
                let prefix = crate::config::sanitize_key(&plan.action_keys[i].instance_id);
                for j in 0..a.value.positions.len() {
                    names.push(format!("{prefix}_j{j}"));
                }
            }
            names
        };

        Ok(SourceSchema {
            state_names,
            velocity_names: if has_velocity {
                velocity_names
            } else {
                Vec::new()
            },
            action_names,
            has_velocity,
            action_is_state,
        })
    }
}

/// One sampled dataset frame; vectors are laid out per the [`SourceSchema`].
#[derive(Debug)]
pub struct FrameRow {
    pub state: Vec<f32>,
    pub velocity: Vec<f32>,
    pub action: Vec<f32>,
    pub images: Vec<Arc<FrameBuf>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SampleGap {
    StateMissing(usize),
    StateStale(usize),
    StateShapeChanged(usize),
    CameraMissing(String),
    CameraStale(String),
}

impl std::fmt::Display for SampleGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleGap::StateMissing(i) => write!(f, "state source {i} stopped producing"),
            SampleGap::StateStale(i) => write!(f, "state source {i} stale"),
            SampleGap::StateShapeChanged(i) => write!(f, "state source {i} changed joint count"),
            SampleGap::CameraMissing(k) => write!(f, "camera {k} stopped producing"),
            SampleGap::CameraStale(k) => write!(f, "camera {k} silent past camera_timeout_s"),
        }
    }
}

/// Zero-order-hold sample of the cache onto one dataset frame. States must be
/// present, fresh, and keep the shape the schema was built with. Commands are
/// held last (a position command means "stay here"); with no action sources
/// the action mirrors the state. Cameras hold their last frame up to timeout.
pub fn sample(
    view: &CacheView,
    schema: &SourceSchema,
    plan: &RecordingPlan,
    config: &Config,
    now: Instant,
) -> Result<FrameRow, SampleGap> {
    let mut state = Vec::with_capacity(schema.state_names.len());
    let mut velocity = Vec::new();
    for (i, slot) in view.states.iter().enumerate() {
        let s = slot.as_ref().ok_or(SampleGap::StateMissing(i))?;
        if s.age(now) > config.state_staleness {
            return Err(SampleGap::StateStale(i));
        }
        if schema.has_velocity && s.value.velocities.len() != s.value.positions.len() {
            return Err(SampleGap::StateShapeChanged(i));
        }
        state.extend(s.value.positions.iter().map(|&v| v as f32));
        if schema.has_velocity {
            velocity.extend(s.value.velocities.iter().map(|&v| v as f32));
        }
    }

    let action = if schema.action_is_state {
        state.clone()
    } else {
        let mut action = Vec::with_capacity(schema.action_names.len());
        for slot in &view.actions {
            // Hold-last: use the latest command even if old; a start gate
            // guaranteed each source produced at least once.
            match slot.as_ref() {
                Some(a) => action.extend(a.value.positions.iter().map(|&v| v as f32)),
                None => return Err(SampleGap::StateMissing(usize::MAX)),
            }
        }
        action
    };

    let mut images = Vec::with_capacity(plan.cameras.len());
    for (slot, entry) in view.cameras.iter().zip(&plan.cameras) {
        let frame = slot
            .as_ref()
            .ok_or_else(|| SampleGap::CameraMissing(entry.key.clone()))?;
        if frame.age(now) > config.camera_timeout {
            return Err(SampleGap::CameraStale(entry.key.clone()));
        }
        images.push(frame.value.clone());
    }

    Ok(FrameRow {
        state,
        velocity,
        action,
        images,
    })
}
