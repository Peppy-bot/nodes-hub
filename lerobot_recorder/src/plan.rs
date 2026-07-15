//! Startup discovery: enumerate the producers bound to each cardinality slot
//! and turn them into an ordered recording plan. Binding order is the order
//! values are concatenated into the dataset vectors, so the launcher defines
//! the layout with no robot-specific code here.

use std::collections::HashMap;

use peppygen::NodeRunner;
use peppygen::consumed_topics::{
    action_sources_joint_commands, color_cameras_video_stream, depth_cameras_video_stream,
    rgbd_cameras_color_stream, state_sources_joint_states,
};

use crate::config::Config;
use crate::types::ProducerKey;

/// One camera stream that becomes one dataset video feature.
#[derive(Debug, Clone)]
pub struct CameraEntry {
    /// Dataset key without the `observation.images.` prefix.
    pub key: String,
}

/// The ordered set of sources this session records, discovered from the bound
/// producers. Index positions are stable and shared with the snapshot cache.
pub struct RecordingPlan {
    /// State producers, in binding order; index = cache slot.
    pub state_keys: Vec<ProducerKey>,
    /// Action producers, in binding order; index = cache slot.
    pub action_keys: Vec<ProducerKey>,
    /// Camera features, in a stable order; index = camera cache slot.
    pub cameras: Vec<CameraEntry>,
    /// Producer -> state cache slot.
    pub state_index: HashMap<ProducerKey, usize>,
    /// Producer -> action cache slot.
    pub action_index: HashMap<ProducerKey, usize>,
    /// Producer -> color-frame camera slot, for the color_cameras slot.
    pub color_index: HashMap<ProducerKey, usize>,
    /// Producer -> color-frame camera slot, for the rgbd color_stream.
    pub rgbd_color_index: HashMap<ProducerKey, usize>,
    /// Producer -> depth-frame camera slot, for the rgbd depth_stream.
    pub rgbd_depth_index: HashMap<ProducerKey, usize>,
    /// Producer -> depth-frame camera slot, for the depth_cameras slot.
    pub depth_index: HashMap<ProducerKey, usize>,
}

impl RecordingPlan {
    pub fn discover(runner: &NodeRunner, config: &Config) -> RecordingPlan {
        let state_keys: Vec<ProducerKey> = state_sources_joint_states::bound_producers(runner)
            .iter()
            .map(ProducerKey::from_ref)
            .collect();
        let action_keys: Vec<ProducerKey> = action_sources_joint_commands::bound_producers(runner)
            .iter()
            .map(ProducerKey::from_ref)
            .collect();

        let state_index = index_of(&state_keys);
        let action_index = index_of(&action_keys);

        let mut cameras: Vec<CameraEntry> = Vec::new();
        let mut color_index = HashMap::new();
        let mut rgbd_color_index = HashMap::new();
        let mut rgbd_depth_index = HashMap::new();
        let mut depth_index = HashMap::new();

        // Color cameras: one color feature each.
        for producer in color_cameras_video_stream::bound_producers(runner) {
            let key = ProducerKey::from_ref(producer);
            let slot = push_camera(&mut cameras, config.camera_key(&key.instance_id));
            color_index.insert(key, slot);
        }
        // RGB-D cameras: a color feature, and a depth feature when enabled.
        for producer in rgbd_cameras_color_stream::bound_producers(runner) {
            let key = ProducerKey::from_ref(producer);
            let base = config.camera_key(&key.instance_id);
            let color_slot = push_camera(&mut cameras, base.clone());
            rgbd_color_index.insert(key.clone(), color_slot);
            if config.record_depth {
                let depth_slot = push_camera(&mut cameras, format!("{base}_depth"));
                rgbd_depth_index.insert(key, depth_slot);
            }
        }
        // Depth-only cameras: one depth feature each (when enabled).
        if config.record_depth {
            for producer in depth_cameras_video_stream::bound_producers(runner) {
                let key = ProducerKey::from_ref(producer);
                let slot = push_camera(&mut cameras, config.camera_key(&key.instance_id));
                depth_index.insert(key, slot);
            }
        }

        RecordingPlan {
            state_keys,
            action_keys,
            cameras,
            state_index,
            action_index,
            color_index,
            rgbd_color_index,
            rgbd_depth_index,
            depth_index,
        }
    }

    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }
}

fn index_of(keys: &[ProducerKey]) -> HashMap<ProducerKey, usize> {
    keys.iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), i))
        .collect()
}

/// Appends a camera entry with a key made unique against collisions, returning
/// its slot index.
fn push_camera(cameras: &mut Vec<CameraEntry>, mut key: String) -> usize {
    if cameras.iter().any(|c| c.key == key) {
        let mut n = 2;
        while cameras.iter().any(|c| c.key == format!("{key}_{n}")) {
            n += 1;
        }
        key = format!("{key}_{n}");
    }
    cameras.push(CameraEntry { key });
    cameras.len() - 1
}
