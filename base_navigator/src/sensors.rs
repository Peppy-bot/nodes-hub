//! The sensors the grid reads: the depth cameras of the `cameras` slot, each with the
//! producer of the `geometry` slot of the same name, and the 2-D lidars of the `lidars`
//! slot, each named as camera_core's `sensor_name` names a sensor, its instance id in its
//! copy or its instance id outside a copy. The sensors are one named list, the cameras in
//! the `cameras` slot's order and then the lidars in the `lidars` slot's, and each sensor's
//! index is its source in the grid and in the stop. The setup refuses slots that bind no
//! sensor, a camera with no geometry of its name, and one name given to two sensors.
//!
//! sensor_mounts' describe_sensors, asked until it answers, must place each sensor fixed to
//! the base as a sensor of its kind, a depth camera of rgbd_camera and a lidar of lidar_2d,
//! through camera_core's `DescribedSensors::fixed_to_the_base`; a sensor it does not, and
//! one a part carries, are refused, naming why. A lidar is described by the frame
//! describe_sensors gives it, as each scan carries its own angles, timing and ranges; each
//! camera is then described as [`crate::cameras`] describes it. [`Sensors`] says of each
//! sensor what a goal waits for: a description, a refusal, or a reading within the source
//! timeout.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use camera_core::sensor_mounts::{
    DEPTH_CAMERA, DESCRIBE_SENSORS_TIMEOUT, DescribedSensors, LIDAR_2D, SensorClass,
    SensorMountsError, sensor_name,
};
use control_core::asking;
use control_core::sync::lock_ignoring_poison;
use frame_tree::FrameName;
use peppygen::consumed_services::cameras::depth_stream_info;
use peppygen::consumed_services::geometry::get_depth_intrinsics;
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::consumed_topics::lidars::scan_stream;
use peppygen::{NodeRunner, ProducerRef};
use tracing::warn;

use crate::cameras::{self, Camera, CameraSlot};

/// The refusal of slots that bind no sensor.
const NO_SENSOR: &str = "neither the cameras slot nor the lidars slot binds a sensor: bind one \
                         or more depth cameras (cameras and geometry) or 2-D lidars (lidars)";

/// A sensor a slot binds: the name it runs under, as sensor_mounts names it, and its
/// producer.
struct Bound {
    name: String,
    producer: ProducerRef,
}

/// The sensors the three slots bind, each in its slot's order.
#[derive(Default)]
struct BoundSlots {
    cameras: Vec<Bound>,
    geometry: Vec<Bound>,
    lidars: Vec<Bound>,
}

/// The members of the slot that `$bound_members`, a generated accessor, reads on `$runner`:
/// each a [`Bound`] named by `sensor_name` for `$class` from its copy's name and its instance
/// id, or the refusal of the first that names no sensor. The members are read by field from
/// the accessor's own type.
macro_rules! bound {
    ($bound_members:path, $runner:expr, $class:expr) => {
        $bound_members($runner)
            .into_iter()
            .map(|member| -> Result<Bound, SensorMountsError> {
                let copy = member.copy.as_ref().map(|copy| copy.name.as_str());
                Ok(Bound {
                    name: sensor_name($class, copy, &member.producer.instance_id)?,
                    producer: member.producer,
                })
            })
            .collect::<Result<Vec<Bound>, SensorMountsError>>()
    };
}

/// The sensors the node's three slots bind, or why one of them names no sensor.
fn bound_slots(runner: &NodeRunner) -> Result<BoundSlots, SensorMountsError> {
    Ok(BoundSlots {
        cameras: bound!(depth_stream_info::bound_members, runner, DEPTH_CAMERA)?,
        geometry: bound!(get_depth_intrinsics::bound_members, runner, DEPTH_CAMERA)?,
        lidars: bound!(scan_stream::bound_members, runner, LIDAR_2D)?,
    })
}

/// The sensors the node's slots bind, paired into one list, or why the setup refuses them.
pub fn read_slots(runner: &NodeRunner) -> Result<Sensors, String> {
    let bound = bound_slots(runner).map_err(|e| e.to_string())?;
    Sensors::new(pair_slots(bound)?)
}

/// What a sensor of the grid is.
#[derive(Debug, PartialEq)]
pub enum SensorKind {
    /// A depth camera of the `cameras` slot, whose lens and offset `geometry`, the producer
    /// of the `geometry` slot of its name, serves.
    DepthCamera { geometry: ProducerRef },
    /// A 2-D lidar of the `lidars` slot.
    Lidar,
}

impl SensorKind {
    /// The class describe_sensors places the sensor as.
    pub fn class(&self) -> SensorClass {
        match self {
            SensorKind::DepthCamera { .. } => DEPTH_CAMERA,
            SensorKind::Lidar => LIDAR_2D,
        }
    }

    /// What one reading of the sensor is called.
    pub fn reading(&self) -> &'static str {
        match self {
            SensorKind::DepthCamera { .. } => "frame",
            SensorKind::Lidar => "scan",
        }
    }

    /// The slot that binds the sensor.
    fn slot(&self) -> &'static str {
        match self {
            SensorKind::DepthCamera { .. } => "cameras",
            SensorKind::Lidar => "lidars",
        }
    }
}

/// A sensor of the grid: its name, the producer it streams as, and its kind.
#[derive(Debug, PartialEq)]
pub struct SensorSlot {
    pub name: String,
    pub producer: ProducerRef,
    pub kind: SensorKind,
}

/// The sensors of `slots`, the cameras paired with their geometry by name and then the
/// lidars, or why they do not pair: a geometry count other than the cameras', a camera with
/// no geometry of its name, and a name two sensors share.
fn pair_slots(slots: BoundSlots) -> Result<Vec<SensorSlot>, String> {
    let BoundSlots {
        cameras,
        geometry,
        lidars,
    } = slots;
    if geometry.len() != cameras.len() {
        return Err(format!(
            "the cameras slot binds {} cameras and the geometry slot {}: bind each camera's \
             geometry under the camera's name",
            cameras.len(),
            geometry.len()
        ));
    }
    let paired_cameras = cameras
        .into_iter()
        .map(|camera| -> Result<SensorSlot, String> {
            let geometry = geometry
                .iter()
                .find(|other| other.name == camera.name)
                .map(|other| other.producer.clone())
                .ok_or_else(|| {
                    format!(
                        "the geometry slot binds no camera named {:?}: bind its geometry",
                        camera.name
                    )
                })?;
            Ok(SensorSlot {
                name: camera.name,
                producer: camera.producer,
                kind: SensorKind::DepthCamera { geometry },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lidars = lidars.into_iter().map(|lidar| SensorSlot {
        name: lidar.name,
        producer: lidar.producer,
        kind: SensorKind::Lidar,
    });
    let sensors: Vec<SensorSlot> = paired_cameras.into_iter().chain(lidars).collect();
    match shared_name(&sensors) {
        Some(refusal) => Err(refusal),
        None => Ok(sensors),
    }
}

/// Why two of `sensors` share a name, for the first sensor whose name one before it holds;
/// none while each sensor's name is its own.
fn shared_name(sensors: &[SensorSlot]) -> Option<String> {
    sensors.iter().enumerate().find_map(|(index, later)| {
        let earlier = sensors[..index]
            .iter()
            .find(|earlier| earlier.name == later.name)?;
        let (slot, name) = (later.kind.slot(), &later.name);
        if earlier.kind.slot() == slot {
            let noun = later.kind.class().noun;
            return Some(format!(
                "the {slot} slot binds two {noun}s named {name:?}: bind each {noun} once"
            ));
        }
        Some(format!(
            "the {} slot and the {slot} slot each bind a sensor named {name:?}: bind each \
             sensor under a name of its own",
            earlier.kind.slot()
        ))
    })
}

/// A described sensor.
pub enum Described {
    DepthCamera(Arc<Camera>),
    /// A 2-D lidar, by the frame describe_sensors gives it, which each of its scans names.
    Lidar(FrameName),
}

/// What the navigator knows of one sensor.
pub enum SensorState {
    /// Not described yet: the last reason it did not answer.
    Describing(String),
    /// Never described: why.
    Refused(String),
    Ready(Described),
}

/// The sensors the grid reads, one or more, by name, what the navigator knows of each, and
/// why the newest reading of each was dropped, none once a reading of it went into the grid.
pub struct Sensors {
    slots: Vec<SensorSlot>,
    count: NonZeroUsize,
    states: Mutex<Vec<SensorState>>,
    dropped: Mutex<Vec<Option<String>>>,
}

impl Sensors {
    /// The sensors of `slots`, none described yet; refused, naming both slots and what to
    /// bind, when `slots` holds none.
    pub fn new(slots: Vec<SensorSlot>) -> Result<Self, String> {
        let count = NonZeroUsize::new(slots.len()).ok_or(NO_SENSOR)?;
        let states = slots
            .iter()
            .map(|_| SensorState::Describing("not asked yet".to_string()))
            .collect();
        Ok(Self {
            dropped: Mutex::new(vec![None; slots.len()]),
            slots,
            count,
            states: Mutex::new(states),
        })
    }

    pub fn count(&self) -> NonZeroUsize {
        self.count
    }

    /// Each sensor, its index its source.
    pub fn slots(&self) -> &[SensorSlot] {
        &self.slots
    }

    pub fn names(&self) -> Vec<&str> {
        self.slots.iter().map(|slot| slot.name.as_str()).collect()
    }

    /// The index of the sensor `producer` streams as.
    pub fn source_of(&self, producer: &ProducerRef) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.producer == *producer)
    }

    /// The described camera at `source`, none before its description and for a lidar.
    pub fn camera(&self, source: usize) -> Option<Arc<Camera>> {
        match &lock_ignoring_poison(&self.states)[source] {
            SensorState::Ready(Described::DepthCamera(camera)) => Some(camera.clone()),
            SensorState::Ready(Described::Lidar(_))
            | SensorState::Describing(_)
            | SensorState::Refused(_) => None,
        }
    }

    /// The frame of the described lidar at `source`, none before its description and for a
    /// camera.
    pub fn lidar(&self, source: usize) -> Option<FrameName> {
        match &lock_ignoring_poison(&self.states)[source] {
            SensorState::Ready(Described::Lidar(frame)) => Some(frame.clone()),
            SensorState::Ready(Described::DepthCamera(_))
            | SensorState::Describing(_)
            | SensorState::Refused(_) => None,
        }
    }

    pub(crate) fn set(&self, source: usize, state: SensorState) {
        lock_ignoring_poison(&self.states)[source] = state;
    }

    /// Record that the newest reading of the sensor at `source` was dropped, for `reason`,
    /// or, with none, went into the grid.
    pub fn reading_handled(&self, source: usize, dropped: Option<String>) {
        lock_ignoring_poison(&self.dropped)[source] = dropped;
    }

    /// What a goal waits for of the sensors at `silent`, each by its class's noun and its
    /// name: its description, its refusal, a reading it does not drop, or a reading within
    /// `source_timeout_s`.
    pub fn waits(&self, silent: &[usize], source_timeout_s: f64) -> String {
        let states = lock_ignoring_poison(&self.states);
        let dropped = lock_ignoring_poison(&self.dropped);
        silent
            .iter()
            .map(|&source| {
                let SensorSlot { name, kind, .. } = &self.slots[source];
                let (noun, reading) = (kind.class().noun, kind.reading());
                match (&states[source], &dropped[source]) {
                    (SensorState::Describing(reason), _) => {
                        format!("the description of the {noun} {name} ({reason})")
                    }
                    (SensorState::Refused(reason), _) => format!("the {noun} {name}: {reason}"),
                    (SensorState::Ready(_), Some(reason)) => format!(
                        "a {reading} of the {noun} {name} that goes into the grid: its \
                         {reading}s arrive and are dropped, {reason}"
                    ),
                    (SensorState::Ready(_), None) => format!(
                        "a reading of the {noun} {name} within source_timeout_s \
                         {source_timeout_s} s"
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// The frame that `mounted`, describe_sensors' answer, gives the sensor of `slot` when it
/// places it fixed to the base as a sensor of its kind's class, or why it does not.
fn placed(mounted: &DescribedSensors, slot: &SensorSlot) -> Result<FrameName, String> {
    mounted
        .fixed_to_the_base(slot.kind.class(), &slot.name)
        .map_err(|e| e.to_string())
}

/// Ask describe_sensors until it answers, refuse each sensor it does not place fixed to the
/// base as a sensor of its kind, describe each lidar by its frame, then describe each other
/// camera again until it answers, until the node cancels.
pub async fn run_descriptions(runner: Arc<NodeRunner>, sensors: Arc<Sensors>) {
    let token = runner.cancellation_token().clone();
    let asker = runner.clone();
    let Some(mounted) = asking::ask("describe_sensors", token.cancelled(), async move || {
        describe_sensors::poll(
            &asker,
            describe_sensors::bound_producer(&asker),
            DESCRIBE_SENSORS_TIMEOUT,
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
        let frame = match placed(&mounted, slot) {
            Ok(frame) => frame,
            Err(reason) => {
                warn!("{} {}: {reason}", slot.kind.class().noun, slot.name);
                sensors.set(source, SensorState::Refused(reason));
                continue;
            }
        };
        match &slot.kind {
            SensorKind::Lidar => sensors.set(source, SensorState::Ready(Described::Lidar(frame))),
            SensorKind::DepthCamera { geometry } => {
                let camera = CameraSlot {
                    name: slot.name.clone(),
                    camera: slot.producer.clone(),
                    geometry: geometry.clone(),
                };
                tokio::spawn(cameras::describe_until_answered(
                    runner.clone(),
                    sensors.clone(),
                    source,
                    camera,
                    frame,
                ));
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The slot of the depth camera `name`, whose depth frames and geometry the instance of
    /// that name produces.
    pub(crate) fn camera_slot(name: &str) -> SensorSlot {
        SensorSlot {
            name: name.to_string(),
            producer: ProducerRef::new("core", name),
            kind: SensorKind::DepthCamera {
                geometry: ProducerRef::new("core", name),
            },
        }
    }

    /// The slot of the 2-D lidar `name`, whose scans the instance of that name produces.
    pub(crate) fn lidar_slot(name: &str) -> SensorSlot {
        SensorSlot {
            name: name.to_string(),
            producer: ProducerRef::new("core", name),
            kind: SensorKind::Lidar,
        }
    }

    /// The sensors of `slots`, one or more.
    pub(crate) fn sensors_of(slots: Vec<SensorSlot>) -> Sensors {
        Sensors::new(slots).expect("one sensor or more")
    }

    fn bound(name: &str) -> Bound {
        Bound {
            name: name.to_string(),
            producer: ProducerRef::new("core", name),
        }
    }

    fn bound_all(names: &[&str]) -> Vec<Bound> {
        names.iter().map(|name| bound(name)).collect()
    }

    fn mounted() -> DescribedSensors {
        DescribedSensors {
            sensor_names: ["front", "wrist_left", "lidar", "arm_lidar"]
                .map(String::from)
                .to_vec(),
            sensor_kinds: ["rgbd_camera", "rgbd_camera", "lidar_2d", "lidar_2d"]
                .map(String::from)
                .to_vec(),
            sensor_frames: [
                "front_color_optical_frame",
                "wrist_left_link",
                "lidar_frame",
                "arm_lidar_frame",
            ]
            .map(String::from)
            .to_vec(),
            carried_by: ["", "left_arm", "", "left_arm"].map(String::from).to_vec(),
        }
    }

    // The sensors are the cameras in the cameras slot's order, each paired with the
    // geometry of its name, then the lidars in the lidars slot's order.
    #[test]
    fn the_cameras_pair_with_their_geometry_by_name_and_the_lidars_follow_them() {
        let paired = pair_slots(BoundSlots {
            cameras: bound_all(&["front", "top"]),
            geometry: bound_all(&["top", "front"]),
            lidars: bound_all(&["lidar", "rear_lidar"]),
        })
        .expect("each camera has a geometry");
        assert_eq!(
            paired,
            [
                camera_slot("front"),
                camera_slot("top"),
                lidar_slot("lidar"),
                lidar_slot("rear_lidar")
            ]
        );
        let lidar_alone = pair_slots(BoundSlots {
            lidars: bound_all(&["lidar"]),
            ..BoundSlots::default()
        })
        .expect("a lidar needs no geometry");
        assert_eq!(lidar_alone, [lidar_slot("lidar")]);
    }

    // A camera with no geometry of its name, a geometry count other than the cameras', a
    // name in both slots and a name twice in one slot are refused, each naming what to bind;
    // slots that bind no sensor are refused naming both slots.
    #[test]
    fn slots_that_do_not_pair_or_bind_no_sensor_are_refused_naming_what_to_bind() {
        let refusal = |cameras: &[&str], geometry: &[&str], lidars: &[&str]| {
            pair_slots(BoundSlots {
                cameras: bound_all(cameras),
                geometry: bound_all(geometry),
                lidars: bound_all(lidars),
            })
            .expect_err("slots that do not pair")
        };
        let no_geometry = refusal(&["front", "top"], &["front", "rear"], &[]);
        assert!(
            no_geometry.contains("no camera named \"top\""),
            "{no_geometry}"
        );
        let counts = refusal(&["front"], &[], &[]);
        assert!(
            counts.contains("binds 1 cameras and the geometry slot 0"),
            "{counts}"
        );
        let across = refusal(&["front"], &["front"], &["front"]);
        assert_eq!(
            across,
            "the cameras slot and the lidars slot each bind a sensor named \"front\": bind \
             each sensor under a name of its own"
        );
        let twice = refusal(&[], &[], &["lidar", "lidar"]);
        assert_eq!(
            twice,
            "the lidars slot binds two lidars named \"lidar\": bind each lidar once"
        );
        let none =
            Sensors::new(pair_slots(BoundSlots::default()).expect("no slot that does not pair"))
                .err()
                .expect("no sensor");
        assert_eq!(none, NO_SENSOR);
        assert!(
            none.contains("the cameras slot") && none.contains("the lidars slot"),
            "{none}"
        );
    }

    // Each sensor is placed by its kind's class: a depth camera fixed to the base in its
    // colour optical frame and a lidar fixed to the base in its sensor frame; a camera under
    // a lidar's name, a lidar under a camera's, one a part carries and a name
    // describe_sensors lacks are refused, each naming why.
    #[test]
    fn each_sensor_is_placed_fixed_to_the_base_as_a_sensor_of_its_kind() {
        let frame = |slot: SensorSlot| placed(&mounted(), &slot).map(|frame| frame.to_string());
        assert_eq!(
            frame(camera_slot("front")).as_deref(),
            Ok("front_color_optical_frame")
        );
        assert_eq!(frame(lidar_slot("lidar")).as_deref(), Ok("lidar_frame"));
        for (slot, cause) in [
            (camera_slot("lidar"), "a sensor of \"lidar_2d\""),
            (lidar_slot("front"), "a sensor of \"rgbd_camera\""),
            (camera_slot("wrist_left"), "\"left_arm\" carries"),
            (lidar_slot("arm_lidar"), "\"left_arm\" carries"),
            (lidar_slot("rear_lidar"), "names no sensor \"rear_lidar\""),
        ] {
            let name = slot.name.clone();
            let refusal = frame(slot).expect_err(&name);
            assert!(refusal.contains(cause), "{name}: {refusal}");
        }
    }

    // A goal waits for each silent sensor by its class's noun and its name.
    #[test]
    fn a_goal_waits_for_each_silent_sensor_by_its_noun_and_name() {
        let sensors = sensors_of(vec![
            camera_slot("front"),
            camera_slot("rear"),
            lidar_slot("lidar"),
        ]);
        sensors.set(1, SensorState::Refused("a part carries it".to_string()));
        sensors.set(
            2,
            SensorState::Ready(Described::Lidar(
                FrameName::parse("lidar_frame").expect("a frame"),
            )),
        );
        assert_eq!(
            sensors.waits(&[0, 1, 2], 0.5),
            "the description of the depth camera front (not asked yet); the depth camera \
             rear: a part carries it; a reading of the lidar lidar within source_timeout_s \
             0.5 s"
        );
        sensors.reading_handled(2, Some("it names another frame".to_string()));
        assert_eq!(
            sensors.waits(&[2], 0.5),
            "a scan of the lidar lidar that goes into the grid: its scans arrive and are \
             dropped, it names another frame"
        );
        assert_eq!(
            sensors.source_of(&ProducerRef::new("core", "lidar")),
            Some(2)
        );
        assert_eq!(
            sensors.lidar(2).map(|frame| frame.to_string()).as_deref(),
            Some("lidar_frame")
        );
        assert!(sensors.camera(2).is_none() && sensors.lidar(0).is_none());
    }
}
