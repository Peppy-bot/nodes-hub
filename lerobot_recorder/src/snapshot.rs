//! Latest-value cache between drain tasks and the fps sampler, and the pure
//! zero-order-hold sampling function. Sizes are set at startup from the
//! [`RecordingPlan`]; the sampler reads whatever has already arrived (never
//! waits for a fresher value, so no future leakage into the dataset).

use std::sync::Arc;

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

/// A cached sample tagged with its producer stamp: nanoseconds since the Unix
/// epoch on the producer's peppy-synchronized clock. Staleness is measured
/// against the recorder's own synchronized clock, so it reflects true sample
/// age across hosts rather than local arrival time.
#[derive(Debug, Clone)]
pub struct Stamped<T> {
    pub value: T,
    pub stamp_ns: u64,
}

impl<T> Stamped<T> {
    pub fn new(value: T, stamp_ns: u64) -> Self {
        Stamped { value, stamp_ns }
    }
    /// Sample age in nanoseconds; a stamp slightly ahead of `now_ns` (residual
    /// cross-host sync skew) reads as zero rather than wrapping.
    pub fn age_ns(&self, now_ns: u64) -> u64 {
        now_ns.saturating_sub(self.stamp_ns)
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
    /// Expected `positions.len()` per state source, indexed like `view.states`.
    pub state_dims: Vec<usize>,
    /// Expected `(width, height)` per camera, indexed like `view.cameras`.
    pub camera_dims: Vec<(u32, u32)>,
}

impl SourceSchema {
    /// Builds the schema from the current cache view, deriving per-joint names
    /// from each source's producer identity plus joint index. Every state
    /// source needs a message; a velocity feature is included only when every
    /// state source reports velocities.
    pub fn from_view(view: &CacheView, plan: &RecordingPlan) -> Result<SourceSchema, String> {
        let mut state_names = Vec::new();
        let mut velocity_names = Vec::new();
        let mut state_dims = Vec::with_capacity(view.states.len());
        let mut has_velocity = true;
        for (i, slot) in view.states.iter().enumerate() {
            let s = slot
                .as_ref()
                .ok_or_else(|| format!("state source {i} has not produced yet"))?;
            let prefix = crate::config::sanitize_key(&plan.state_keys[i].instance_id);
            state_dims.push(s.value.positions.len());
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

        let mut camera_dims = Vec::with_capacity(view.cameras.len());
        for (i, slot) in view.cameras.iter().enumerate() {
            let c = slot
                .as_ref()
                .ok_or_else(|| format!("camera {i} has not produced yet"))?;
            camera_dims.push((c.value.width, c.value.height));
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
            state_dims,
            camera_dims,
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
    CameraShapeChanged(String),
}

impl std::fmt::Display for SampleGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleGap::StateMissing(i) => write!(f, "state source {i} stopped producing"),
            SampleGap::StateStale(i) => write!(f, "state source {i} stale"),
            SampleGap::StateShapeChanged(i) => write!(f, "state source {i} changed joint count"),
            SampleGap::CameraMissing(k) => write!(f, "camera {k} stopped producing"),
            SampleGap::CameraStale(k) => write!(f, "camera {k} silent past camera_timeout_s"),
            SampleGap::CameraShapeChanged(k) => write!(f, "camera {k} changed resolution"),
        }
    }
}

/// Zero-order-hold sample of the cache onto one dataset frame. Every state
/// source must be present, fresh, and keep the joint count captured at begin;
/// every camera must be fresh and keep the resolution captured at begin.
/// Commands are held last (a position command means "stay here"); with no
/// action sources the action mirrors the state.
pub fn sample(
    view: &CacheView,
    schema: &SourceSchema,
    plan: &RecordingPlan,
    config: &Config,
    now_ns: u64,
) -> Result<FrameRow, SampleGap> {
    let mut state = Vec::with_capacity(schema.state_names.len());
    let mut velocity = Vec::new();
    for (i, slot) in view.states.iter().enumerate() {
        let s = slot.as_ref().ok_or(SampleGap::StateMissing(i))?;
        if s.age_ns(now_ns) > config.state_staleness.as_nanos() as u64 {
            return Err(SampleGap::StateStale(i));
        }
        if s.value.positions.len() != schema.state_dims[i] {
            return Err(SampleGap::StateShapeChanged(i));
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
            // Hold-last: use the latest command even if old. The start gate
            // required every action source, and a watch slot never reverts to
            // None, so an occupied slot is an invariant here.
            let a = slot
                .as_ref()
                .expect("action source occupied: guaranteed by the start gate");
            action.extend(a.value.positions.iter().map(|&v| v as f32));
        }
        action
    };

    let mut images = Vec::with_capacity(plan.cameras.len());
    for (i, (slot, entry)) in view.cameras.iter().zip(&plan.cameras).enumerate() {
        let frame = slot
            .as_ref()
            .ok_or_else(|| SampleGap::CameraMissing(entry.key.clone()))?;
        if frame.age_ns(now_ns) > config.camera_timeout.as_nanos() as u64 {
            return Err(SampleGap::CameraStale(entry.key.clone()));
        }
        if (frame.value.width, frame.value.height) != schema.camera_dims[i] {
            return Err(SampleGap::CameraShapeChanged(entry.key.clone()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, StorageBackend};
    use crate::plan::{CameraEntry, RecordingPlan};
    use crate::types::CameraEncoding;
    use lerobot_dataset::VideoCodec;
    use std::collections::HashMap;
    use std::num::NonZeroU32;
    use std::path::PathBuf;
    use std::time::Duration;

    fn test_config() -> Config {
        Config {
            robot_type: "bot".into(),
            fps: NonZeroU32::new(30).unwrap(),
            output_root: PathBuf::from("/tmp"),
            dataset_name: "d".into(),
            default_task: "t".into(),
            codec: VideoCodec::H264Libx264,
            camera_keys: HashMap::new(),
            record_depth: false,
            depth_unit_m: 0.001,
            storage: StorageBackend::Local,
            state_staleness: Duration::from_secs(10),
            camera_start_fresh: Duration::from_secs(10),
            camera_timeout: Duration::from_secs(10),
            max_episode_frames: 1000,
            status_period: Duration::from_millis(500),
        }
    }

    fn plan_with_cameras(cameras: Vec<CameraEntry>) -> RecordingPlan {
        RecordingPlan {
            state_keys: Vec::new(),
            action_keys: Vec::new(),
            cameras,
            state_index: HashMap::new(),
            action_index: HashMap::new(),
            color_index: HashMap::new(),
            rgbd_color_index: HashMap::new(),
            rgbd_depth_index: HashMap::new(),
            depth_index: HashMap::new(),
        }
    }

    /// The instant the tests sample at; slots are stamped at this time so a
    /// fresh sample has zero age.
    const TEST_NOW_NS: u64 = 1_000_000_000_000;

    fn state_slot(positions: Vec<f64>) -> Option<Stamped<JointSample>> {
        Some(Stamped::new(
            JointSample {
                positions,
                velocities: Vec::new(),
            },
            TEST_NOW_NS,
        ))
    }

    fn camera_slot(width: u32, height: u32) -> Option<Stamped<Arc<FrameBuf>>> {
        Some(Stamped::new(
            Arc::new(FrameBuf {
                encoding: CameraEncoding::Rgb8,
                width,
                height,
                bytes: vec![0; (width * height * 3) as usize],
            }),
            TEST_NOW_NS,
        ))
    }

    fn mirror_schema(state_dims: Vec<usize>, camera_dims: Vec<(u32, u32)>) -> SourceSchema {
        let state_names: Vec<String> = state_dims
            .iter()
            .enumerate()
            .flat_map(|(s, &n)| (0..n).map(move |j| format!("s{s}_j{j}")))
            .collect();
        SourceSchema {
            action_names: state_names.clone(),
            state_names,
            velocity_names: Vec::new(),
            has_velocity: false,
            action_is_state: true,
            state_dims,
            camera_dims,
        }
    }

    #[test]
    fn sample_ok_on_matching_shapes() {
        let plan = plan_with_cameras(vec![CameraEntry { key: "cam".into() }]);
        let schema = mirror_schema(vec![7], vec![(640, 480)]);
        let view = CacheView {
            states: vec![state_slot(vec![0.0; 7])],
            actions: Vec::new(),
            cameras: vec![camera_slot(640, 480)],
        };
        let row = sample(&view, &schema, &plan, &test_config(), TEST_NOW_NS).unwrap();
        assert_eq!(row.state.len(), 7);
        assert_eq!(row.images.len(), 1);
    }

    #[test]
    fn sample_flags_stale_state_by_producer_stamp() {
        let plan = plan_with_cameras(Vec::new());
        let schema = mirror_schema(vec![1], Vec::new());
        let staleness_ns = test_config().state_staleness.as_nanos() as u64;
        let view = CacheView {
            states: vec![Some(Stamped::new(
                JointSample {
                    positions: vec![0.0],
                    velocities: Vec::new(),
                },
                TEST_NOW_NS - staleness_ns - 1,
            ))],
            actions: Vec::new(),
            cameras: Vec::new(),
        };
        let gap = sample(&view, &schema, &plan, &test_config(), TEST_NOW_NS).unwrap_err();
        assert_eq!(gap, SampleGap::StateStale(0));
    }

    #[test]
    fn sample_flags_state_shape_change() {
        let plan = plan_with_cameras(Vec::new());
        let schema = mirror_schema(vec![7], Vec::new());
        let view = CacheView {
            states: vec![state_slot(vec![0.0; 8])],
            actions: Vec::new(),
            cameras: Vec::new(),
        };
        let gap = sample(&view, &schema, &plan, &test_config(), TEST_NOW_NS).unwrap_err();
        assert_eq!(gap, SampleGap::StateShapeChanged(0));
    }

    #[test]
    fn sample_flags_camera_shape_change() {
        let plan = plan_with_cameras(vec![CameraEntry { key: "cam".into() }]);
        let schema = mirror_schema(vec![1], vec![(640, 480)]);
        let view = CacheView {
            states: vec![state_slot(vec![0.0])],
            actions: Vec::new(),
            cameras: vec![camera_slot(800, 480)],
        };
        let gap = sample(&view, &schema, &plan, &test_config(), TEST_NOW_NS).unwrap_err();
        assert_eq!(gap, SampleGap::CameraShapeChanged("cam".into()));
    }
}
