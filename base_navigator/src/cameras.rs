//! The sensors the grid reads: the depth cameras of the `cameras` slot, each
//! named as its copy's fragment names it (its instance id outside a copy), with
//! the producer of the `geometry` slot of the same name. sensor_mounts'
//! describe_sensors names each of them a sensor of rgbd_camera fixed to the
//! base, its carried_by empty; a camera it does not, or one a part carries, is
//! refused, naming why. The sensors are a named list, in the order of the
//! `cameras` slot, and each sensor's index is its source in the grid and in
//! the stop.
//!
//! Each camera is described in the background, again until it answers: its
//! depth lens and model, where its depth optical frame stands in its colour
//! optical frame, by camera_core's reader, which must be the frame
//! describe_sensors gives the camera, its depth unit and its farthest depth.
//! [`Sensors`] says of each what a goal waits for: a description, a refusal,
//! or a reading within the source timeout.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use camera_core::depth_camera::{DepthCamera, DepthExtrinsics, DepthIntrinsics, read_depth_camera};
use camera_core::sensor_mounts::DescribedSensors;
use control_core::asking;
use control_core::latched_log::LatchedLog;
use control_core::positive_finite::PositiveFinite;
use control_core::sync::lock_ignoring_poison;
use control_core::time::sleep_unless_cancelled;
use peppygen::consumed_services::cameras::depth_stream_info;
use peppygen::consumed_services::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::{NodeRunner, ProducerRef};
use peppylib::messaging::BoundMember;
use tracing::{info, warn};

/// The contract a sensor of the grid streams on, as describe_sensors names
/// its kind.
const DEPTH_CAMERA: &str = "rgbd_camera";

/// How long one call of a description waits for its answer.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);

/// How long the description waits between two attempts a camera did not
/// answer.
const RETRY: Duration = Duration::from_millis(500);

/// A camera's two producers, bound under one name.
#[derive(Clone, Debug)]
pub struct CameraSlot {
    pub name: String,
    pub camera: ProducerRef,
    pub geometry: ProducerRef,
}

/// The meters of one depth count: a finite number above 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthUnit(f32);

impl DepthUnit {
    /// `depth_unit` of depth_stream_info, or why it is refused.
    pub fn parse(depth_unit: f32) -> Result<Self, String> {
        if depth_unit.is_finite() && depth_unit > 0.0 {
            return Ok(Self(depth_unit));
        }
        Err(format!(
            "depth_stream_info: depth_unit {depth_unit} is not a positive number of meters"
        ))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// A described camera: its depth stream, the meters of one depth count, and
/// the farthest depth (m) it reports, beyond which a sample of 0 met nothing.
#[derive(Clone, Debug)]
pub struct Camera {
    pub depth: DepthCamera,
    pub depth_unit: DepthUnit,
    pub range: PositiveFinite,
}

/// What the navigator knows of one sensor.
#[derive(Clone, Debug)]
pub enum SensorState {
    /// Not described yet: the last reason it did not answer.
    Describing(String),
    /// Never described: why.
    Refused(String),
    Ready(Arc<Camera>),
}

/// The sensors the grid reads, by name, what the navigator knows of each,
/// and why the newest frame of each was dropped, none once a frame of it went
/// into the grid.
pub struct Sensors {
    slots: Vec<CameraSlot>,
    states: Mutex<Vec<SensorState>>,
    dropped: Mutex<Vec<Option<String>>>,
}

impl Sensors {
    /// The sensors of `slots`, none described yet; at least one.
    pub fn new(slots: Vec<CameraSlot>) -> Self {
        let states = slots
            .iter()
            .map(|_| SensorState::Describing("not asked yet".to_string()))
            .collect();
        Self {
            dropped: Mutex::new(vec![None; slots.len()]),
            slots,
            states: Mutex::new(states),
        }
    }

    pub fn count(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.slots.len()).expect("the slots pair one camera or more")
    }

    pub fn names(&self) -> Vec<&str> {
        self.slots.iter().map(|slot| slot.name.as_str()).collect()
    }

    /// The index of the sensor `producer` streams as.
    pub fn source_of(&self, producer: &ProducerRef) -> Option<usize> {
        self.slots.iter().position(|slot| slot.camera == *producer)
    }

    /// The described camera of the sensor at `source`, none before its
    /// description.
    pub fn ready(&self, source: usize) -> Option<Arc<Camera>> {
        match &lock_ignoring_poison(&self.states)[source] {
            SensorState::Ready(camera) => Some(camera.clone()),
            SensorState::Describing(_) | SensorState::Refused(_) => None,
        }
    }

    pub(crate) fn set(&self, source: usize, state: SensorState) {
        lock_ignoring_poison(&self.states)[source] = state;
    }

    /// Record that the newest frame of the sensor at `source` was dropped,
    /// for `reason`, or, with none, went into the grid.
    pub fn frame_handled(&self, source: usize, dropped: Option<String>) {
        lock_ignoring_poison(&self.dropped)[source] = dropped;
    }

    /// What a goal waits for of the sensors at `silent`, each by name: its
    /// description, its refusal, a frame it does not drop, or a reading
    /// within `source_timeout_s`.
    pub fn waits(&self, silent: &[usize], source_timeout_s: f64) -> String {
        let states = lock_ignoring_poison(&self.states);
        let dropped = lock_ignoring_poison(&self.dropped);
        silent
            .iter()
            .map(|&source| {
                let name = &self.slots[source].name;
                match (&states[source], &dropped[source]) {
                    (SensorState::Describing(reason), _) => {
                        format!("the description of the camera {name} ({reason})")
                    }
                    (SensorState::Refused(reason), _) => format!("the camera {name}: {reason}"),
                    (SensorState::Ready(_), Some(reason)) => format!(
                        "a frame of the camera {name} that goes into the grid: its frames \
                         arrive and are dropped, {reason}"
                    ),
                    (SensorState::Ready(_), None) => format!(
                        "a reading of the camera {name} within source_timeout_s \
                         {source_timeout_s} s"
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// The name of the camera `member` is: the id its copy's fragment wrote, or
/// its instance id outside a copy, which is the name describe_sensors gives
/// it.
fn camera_name(member: &BoundMember) -> String {
    member.copy.as_ref().map_or_else(
        || member.producer.instance_id.clone(),
        |copy| copy.instance_id.to_string(),
    )
}

/// The cameras of `cameras` and `geometry`, the bound members of the two
/// slots, paired by name, or why they do not pair.
pub fn pair_slots(
    cameras: &[BoundMember],
    geometry: &[BoundMember],
) -> Result<Vec<CameraSlot>, String> {
    let named = |members: &[BoundMember]| -> Vec<(String, ProducerRef)> {
        members
            .iter()
            .map(|member| (camera_name(member), member.producer.clone()))
            .collect()
    };
    let (cameras, geometry) = (named(cameras), named(geometry));
    if cameras.is_empty() {
        return Err("the cameras slot binds no camera: bind one or more depth cameras".to_string());
    }
    if let Some((_, (repeated, _))) = cameras
        .iter()
        .enumerate()
        .find(|(i, (name, _))| cameras[..*i].iter().any(|(other, _)| other == name))
    {
        return Err(format!(
            "the cameras slot binds two cameras named {repeated:?}: bind each camera once"
        ));
    }
    if geometry.len() != cameras.len() {
        return Err(format!(
            "the cameras slot binds {} cameras and the geometry slot {}: bind each camera's \
             geometry under the camera's name",
            cameras.len(),
            geometry.len()
        ));
    }
    cameras
        .into_iter()
        .map(|(name, camera)| {
            let geometry = geometry
                .iter()
                .find(|(other, _)| *other == name)
                .map(|(_, producer)| producer.clone())
                .ok_or_else(|| {
                    format!("the geometry slot binds no camera named {name:?}: bind its geometry")
                })?;
            Ok(CameraSlot {
                name,
                camera,
                geometry,
            })
        })
        .collect()
}

/// The frame `sensors`, describe_sensors' answer, gives `name` when it names
/// it a depth camera fixed to the base, or why it does not.
fn fixed_to_the_base(sensors: &DescribedSensors, name: &str) -> Result<String, String> {
    let listed = sensors.sensors().map_err(|e| e.to_string())?;
    let sensor = listed
        .iter()
        .find(|sensor| sensor.name == name)
        .ok_or_else(|| {
            format!(
                "sensor_mounts' describe_sensors names no sensor {name:?}: run the camera under \
                 the name of a sensor of the robot"
            )
        })?;
    if sensor.kind != DEPTH_CAMERA {
        return Err(format!(
            "sensor_mounts' describe_sensors names {name:?} a sensor of {:?}: the grid reads \
             depth cameras of {DEPTH_CAMERA}",
            sensor.kind
        ));
    }
    if !sensor.carried_by.is_empty() {
        return Err(format!(
            "sensor_mounts' describe_sensors names {name:?} a sensor that {:?} carries: the \
             grid reads sensors fixed to the base",
            sensor.carried_by
        ));
    }
    Ok(sensor.frame.to_owned())
}

/// Whether `colour_frame`, the colour optical frame camera_geometry gives a
/// camera, is `described_frame`, the one describe_sensors gives it, or why
/// not.
fn frames_agree(described_frame: &str, colour_frame: &str) -> Result<(), String> {
    if described_frame == colour_frame {
        return Ok(());
    }
    Err(format!(
        "get_depth_to_color_extrinsics places the depth frame in {colour_frame:?}, where \
         sensor_mounts' describe_sensors gives the camera the frame {described_frame:?}: give \
         both the camera's colour optical frame"
    ))
}

/// One description of the camera of `slot`, which describe_sensors gives
/// `frame`: its depth stream, by camera_core's reader, its depth unit and its
/// farthest depth.
async fn describe(runner: &NodeRunner, slot: &CameraSlot, frame: &str) -> Result<Camera, String> {
    let intrinsics = get_depth_intrinsics::poll(runner, &slot.geometry, ANSWER_TIMEOUT)
        .await
        .map_err(|e| format!("get_depth_intrinsics: {e}"))?
        .data;
    let extrinsics = get_depth_to_color_extrinsics::poll(runner, &slot.geometry, ANSWER_TIMEOUT)
        .await
        .map_err(|e| format!("get_depth_to_color_extrinsics: {e}"))?
        .data;
    let info = depth_stream_info::poll(runner, &slot.camera, ANSWER_TIMEOUT)
        .await
        .map_err(|e| format!("depth_stream_info: {e}"))?
        .data;
    let depth_to_color = &extrinsics.depth_to_color;
    let depth = read_depth_camera(
        &DepthIntrinsics {
            width: intrinsics.width,
            height: intrinsics.height,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            distortion_model: &intrinsics.distortion_model,
            distortion: &intrinsics.distortion,
            depth_model: &intrinsics.depth_model,
            align_mode: &intrinsics.align_mode,
        },
        &DepthExtrinsics {
            frame: &depth_to_color.frame,
            position: depth_to_color.position,
            orientation: depth_to_color.orientation,
            align_mode: &extrinsics.align_mode,
        },
    )
    .map_err(|e| e.to_string())?;
    frames_agree(frame, depth.frame.as_str())?;
    let depth_unit = DepthUnit::parse(info.depth_unit)?;
    let range = PositiveFinite::parse(intrinsics.max_depth_m).map_err(|_| {
        format!(
            "get_depth_intrinsics: max_depth_m {} is not a positive number of meters",
            intrinsics.max_depth_m
        )
    })?;
    info!(
        "camera {}: depth {}x{} in {}, align_mode {:?}, range {} m",
        slot.name,
        intrinsics.width,
        intrinsics.height,
        depth.frame.as_str(),
        depth.align_mode,
        range.get()
    );
    Ok(Camera {
        depth,
        depth_unit,
        range,
    })
}

/// Ask describe_sensors until it answers, refuse each camera it does not name
/// a depth camera fixed to the base, then describe each other camera again
/// until it answers, until the node cancels.
pub async fn run_descriptions(runner: Arc<NodeRunner>, sensors: Arc<Sensors>) {
    let token = runner.cancellation_token().clone();
    let asker = runner.clone();
    let Some(mounted) = asking::ask("describe_sensors", token.cancelled(), async move || {
        describe_sensors::poll(
            &asker,
            describe_sensors::bound_producer(&asker),
            ANSWER_TIMEOUT,
        )
        .await
        .map(|response| {
            let data = response.data;
            DescribedSensors {
                sensor_names: data.sensor_names,
                sensor_kinds: data.sensor_kinds,
                sensor_frames: data.sensor_frames,
                carried_by: data.carried_by,
            }
        })
    })
    .await
    else {
        return;
    };
    for (source, slot) in sensors.slots.iter().enumerate() {
        let frame = match fixed_to_the_base(&mounted, &slot.name) {
            Ok(frame) => frame,
            Err(reason) => {
                warn!("camera {}: {reason}", slot.name);
                sensors.set(source, SensorState::Refused(reason));
                continue;
            }
        };
        let (runner, sensors, token, slot) =
            (runner.clone(), sensors.clone(), token.clone(), slot.clone());
        tokio::spawn(async move {
            let failing = LatchedLog::warn(format!("camera {} description", slot.name));
            loop {
                match describe(&runner, &slot, &frame).await {
                    Ok(camera) => {
                        failing.success();
                        return sensors.set(source, SensorState::Ready(Arc::new(camera)));
                    }
                    Err(reason) => {
                        failing.failure(&reason);
                        sensors.set(source, SensorState::Describing(reason));
                    }
                }
                if !sleep_unless_cancelled(RETRY, token.cancelled()).await {
                    return;
                }
            }
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The slot of the camera `name`, whose depth frames and geometry the
    /// instance of that name produces.
    pub(crate) fn camera_slot(name: &str) -> CameraSlot {
        CameraSlot {
            name: name.to_string(),
            camera: ProducerRef::new("core", name),
            geometry: ProducerRef::new("core", name),
        }
    }

    fn member(instance_id: &str, copy: Option<(&str, &str)>) -> BoundMember {
        BoundMember {
            producer: ProducerRef::new("core", instance_id),
            copy: copy.map(|(name, id)| peppylib::messaging::CopyTag {
                name: name.to_string().try_into().expect("a name"),
                instance_id: id.to_string().try_into().expect("a name"),
            }),
        }
    }

    fn mounted() -> DescribedSensors {
        DescribedSensors {
            sensor_names: ["front", "wrist_left", "lidar"].map(String::from).to_vec(),
            sensor_kinds: ["rgbd_camera", "rgbd_camera", "lidar_2d"]
                .map(String::from)
                .to_vec(),
            sensor_frames: ["front_color_optical_frame", "wrist_left_link", "lidar_link"]
                .map(String::from)
                .to_vec(),
            carried_by: ["", "left_arm", ""].map(String::from).to_vec(),
        }
    }

    #[test]
    fn a_camera_is_named_by_its_copy_or_its_instance() {
        assert_eq!(camera_name(&member("front", None)), "front");
        assert_eq!(
            camera_name(&member("india_front", Some(("india", "front")))),
            "front"
        );
    }

    #[test]
    fn the_two_slots_pair_by_name_and_refuse_a_camera_with_no_geometry() {
        let cameras = [member("front", None), member("top", None)];
        let paired = pair_slots(&cameras, &[member("top", None), member("front", None)])
            .expect("both cameras have a geometry");
        assert_eq!(
            paired
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            ["front", "top"]
        );
        let refusal = pair_slots(&cameras, &[member("front", None), member("rear", None)])
            .expect_err("top has no geometry");
        assert!(refusal.contains("no camera named \"top\""), "{refusal}");
    }

    // The grid reads a depth camera fixed to the base, in the frame
    // describe_sensors gives it, and refuses one a part carries, a sensor of
    // another kind and a name describe_sensors lacks.
    #[test]
    fn a_camera_the_grid_reads_is_a_depth_camera_fixed_to_the_base() {
        assert_eq!(
            fixed_to_the_base(&mounted(), "front").as_deref(),
            Ok("front_color_optical_frame")
        );
        for (name, cause) in [
            ("wrist_left", "\"left_arm\" carries"),
            ("lidar", "\"lidar_2d\""),
            ("chest", "names no sensor"),
        ] {
            let refusal = fixed_to_the_base(&mounted(), name).expect_err(name);
            assert!(refusal.contains(cause), "{name}: {refusal}");
        }
    }

    // A camera whose geometry places its depth frame in a frame other than
    // the one describe_sensors gives it is refused, naming both; a depth unit
    // of 0 or below is refused.
    #[test]
    fn a_camera_is_described_in_its_described_frame_with_a_positive_unit() {
        assert_eq!(
            frames_agree("front_color_optical_frame", "front_color_optical_frame"),
            Ok(())
        );
        let refusal =
            frames_agree("front_color_optical_frame", "front_link").expect_err("two frames");
        assert!(
            refusal.contains("\"front_link\"") && refusal.contains("\"front_color_optical_frame\""),
            "{refusal}"
        );
        assert_eq!(DepthUnit::parse(0.001).map(DepthUnit::get), Ok(0.001));
        for refused in [0.0, -0.001, f32::NAN] {
            assert!(DepthUnit::parse(refused).is_err(), "{refused}");
        }
    }

    #[test]
    fn a_goal_waits_for_each_silent_sensor_by_name() {
        let sensors = Sensors::new(vec![camera_slot("front"), camera_slot("rear")]);
        sensors.set(1, SensorState::Refused("a part carries it".to_string()));
        assert_eq!(
            sensors.waits(&[0, 1], 0.5),
            "the description of the camera front (not asked yet); the camera rear: a part \
             carries it"
        );
        assert_eq!(
            sensors.source_of(&ProducerRef::new("core", "rear")),
            Some(1)
        );
    }
}
