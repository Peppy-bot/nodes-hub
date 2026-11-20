//! Every publisher the backbone owns, and the one path that stamps, builds,
//! publishes and reports a message. (Peppy vocabulary throughout: a *publisher*
//! sends on a topic; a pairing *slot* carries one direction of a pairing's two
//! one-way streams to its one *peer*; "wire" below means the encoded message
//! on the transport, nothing else.)
//!
//! Each pairing slot is its own generated module, so two slots carrying the
//! same schema expose two distinct `build_message` items. Holding the builder
//! as a function pointer lets one [`Publisher`] serve both arms of a pairing,
//! and the send path is written once instead of once per slot.
//!
//! Publishing on an unpaired slot is a legal no-op, so the backbone declares
//! every slot at bringup and publishes regardless; a follower simply starts
//! tracking once its pair is established. A slot that cannot be declared at
//! all is a bringup fault and takes the node down.

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::geometry::pose_arrays;
use control_core::robot_tree::RobotTree;
use control_core::throttle::Throttle;
use frame_tree::{MovingMessage, Publisher as TreePublisher};
use peppygen::NodeRunner;
use peppygen::emitted_topics::collision_status::collision_status;
use peppygen::emitted_topics::limb_state::limb_states;
use peppygen::emitted_topics::transforms::transforms;
use peppygen::paired_topics::{
    leader_left_arm, leader_left_arm_pose, leader_left_gripper, leader_right_arm,
    leader_right_arm_pose, leader_right_gripper, left_arm, left_gripper, right_arm, right_gripper,
};
use peppylib::{Payload, TopicPublisher};
use robot_description::PerArm;
use srs_model::JointVec;
use srs_model::nalgebra::Isometry3;
use tracing::{error, warn};

use crate::robot::ARMS;
use crate::streams::{GripperState, THROTTLED_WARN_PERIOD};
use crate::transforms::JointTransforms;
use crate::types::limb_names;

/// Now on this instance's bound clock: the timeline its messages are
/// stamped and aged on, and the time of each measurement a report gives.
/// Errors until the clock delivers its first tick.
pub(crate) fn clock_now() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

/// A joint vector slot: `joint_setpoints` downstream and `joint_states`
/// upstream share this schema.
type JointBuild = fn(SystemTime, Vec<f64>, Vec<f64>, Vec<f64>) -> peppygen::Result<Payload>;

/// A commanded aperture: the opening fraction and the effort cap to relay.
type OpeningBuild = fn(SystemTime, f64, f64) -> peppygen::Result<Payload>;

/// A measured aperture: the opening fraction, the measured effort, and the
/// follower's effort ceiling.
type ApertureBuild = fn(SystemTime, f64, f64, f64) -> peppygen::Result<Payload>;

/// An end-effector pose in a frame: position and scalar-last quaternion.
type PoseBuild = fn(String, SystemTime, [f64; 3], [f64; 4]) -> peppygen::Result<Payload>;

/// The governor proximity readout: nearest-pair distance and links, plus the
/// disposition of the commanded motion.
type StatusBuild = fn(SystemTime, f64, String, String, bool, bool) -> peppygen::Result<Payload>;

/// The whole-robot state snapshot: its frame, the name tables, the joint
/// vectors flattened per the limb_state contract's slicing rule, each arm's
/// grasp point and each gripper's opening.
type LimbStateBuild = fn(
    String,
    SystemTime,
    Vec<String>,
    Vec<u32>,
    Vec<f64>,
    Vec<limb_states::MessageGraspPointsItem>,
    Vec<String>,
    Vec<f64>,
) -> peppygen::Result<Payload>;

/// The transforms of one measurement, each a record of a child frame placed
/// in its parent frame, and when they all started.
type TransformsBuild =
    fn(SystemTime, Vec<transforms::MessageTransformsItem>, SystemTime) -> peppygen::Result<Payload>;

/// One whole-robot state snapshot, gathered by the coordinator at the readout
/// cadence: every limb's measured joints, grasp-point pose, and gripper
/// opening, in place order like every [`PerArm`], and the time it holds at,
/// the newest of the arms' joint-state times.
pub struct LimbStateSnapshot {
    pub joints: PerArm<JointVec, ARMS>,
    pub poses: PerArm<Isometry3<f64>, ARMS>,
    pub openings: PerArm<f64, ARMS>,
    pub measured_at: SystemTime,
}

/// One outbound slot: its declared publisher, that slot's generated
/// `build_message`, and the phrase naming it in a log line.
pub struct Publisher<Build> {
    publisher: TopicPublisher,
    build: Build,
    what: String,
    /// Throttles this slot's failure logs: a stalled clock or a persistently
    /// failing wire hits every cycle, and the log must say so once per
    /// [`crate::streams::THROTTLED_WARN_PERIOD`] per slot, not at the control
    /// rate. One state for both failure kinds (a message either fails to build
    /// or fails to publish, never both). A mutex only because the coordinator
    /// future must be `Send`; it is touched exclusively on error paths, by one
    /// task.
    error_log: Mutex<Throttle>,
}

impl<Build> Publisher<Build> {
    async fn declare(
        what: String,
        declaring: impl Future<Output = peppygen::Result<TopicPublisher>>,
        build: Build,
    ) -> peppygen::Result<Self> {
        Ok(Self {
            publisher: declare(&what, declaring).await?,
            build,
            what,
            error_log: Mutex::new(Throttle::new(THROTTLED_WARN_PERIOD)),
        })
    }
}

impl Publisher<JointBuild> {
    /// Publish one limb's joint vector. Efforts ride empty: this backbone
    /// neither commands nor measures them, which the contract spells as an
    /// empty list rather than a vector of zeros.
    pub async fn send(&self, positions: &JointVec, velocities: &JointVec) {
        self.emit(|timestamp| {
            (self.build)(
                timestamp,
                positions.to_vec(),
                velocities.to_vec(),
                Vec::new(),
            )
        })
        .await;
    }
}

impl Publisher<OpeningBuild> {
    /// Publish one gripper's governed opening fraction and the effort cap to
    /// relay (`None` rides as the wire's 0: no preference, leaving the
    /// follower's configured ceiling in charge).
    pub async fn send(&self, opening: f64, max_effort: Option<f64>) {
        self.emit(|timestamp| (self.build)(timestamp, opening, max_effort.unwrap_or(0.0)))
            .await;
    }
}

impl Publisher<PoseBuild> {
    /// Publish one arm's measured end-effector pose in `frame`, the robot
    /// frame, at `measured_at`, the time of the joints it comes from. What
    /// lets a Cartesian leader hold no kinematics of its own.
    pub async fn send(&self, frame: &str, measured_at: SystemTime, pose: &Isometry3<f64>) {
        let (position, orientation) = pose_arrays(pose);
        self.emit_at(measured_at, |timestamp| {
            (self.build)(frame.to_owned(), timestamp, position, orientation)
        })
        .await;
    }
}

impl Publisher<ApertureBuild> {
    /// Relay one gripper's measured state as its follower reported it.
    pub async fn send(&self, measured: &GripperState) {
        self.emit(|timestamp| {
            (self.build)(
                timestamp,
                measured.fraction,
                measured.effort,
                measured.max_effort,
            )
        })
        .await;
    }
}

impl Publisher<StatusBuild> {
    /// Publish the governor proximity readout.
    pub async fn send(
        &self,
        distance: f64,
        link_a: String,
        link_b: String,
        throttling: bool,
        stopped: bool,
    ) {
        self.emit(|timestamp| {
            (self.build)(timestamp, distance, link_a, link_b, throttling, stopped)
        })
        .await;
    }
}

impl Publisher<LimbStateBuild> {
    /// Publish one whole-robot snapshot in `frame`, the robot frame, at the
    /// time it holds at: joint vectors concatenated in arm order, one grasp
    /// point per arm, the name tables labelling every index.
    pub async fn send(&self, frame: &str, s: &LimbStateSnapshot) -> Option<SystemTime> {
        let grasp_points = s
            .poses
            .iter()
            .map(|pose| {
                let (position, orientation) = pose_arrays(pose);
                limb_states::MessageGraspPointsItem {
                    position,
                    orientation,
                }
            })
            .collect();
        let names = limb_names();
        self.emit_at(s.measured_at, move |timestamp| {
            (self.build)(
                frame.to_owned(),
                timestamp,
                names.arm_names,
                names.joints_per_arm,
                s.joints.into_array().concat(),
                grasp_points,
                names.gripper_names,
                s.openings.into_array().to_vec(),
            )
        })
        .await
    }
}

impl Publisher<TransformsBuild> {
    /// Publish `message`, the transforms of one measurement, under its
    /// timestamp.
    pub async fn send(&self, message: MovingMessage) {
        let records = message
            .transforms
            .into_iter()
            .map(|t| transforms::MessageTransformsItem {
                frame: t.frame,
                child_frame: t.child_frame,
                position: t.position,
                orientation: t.orientation,
            })
            .collect();
        self.emit_at(message.timestamp, |timestamp| {
            (self.build)(timestamp, records, message.start_timestamp)
        })
        .await;
    }
}

impl<Build> Publisher<Build> {
    /// Emit one failure line for this slot, at most once per throttle window.
    fn log_throttled(&self, emit: impl FnOnce()) {
        let mut error_log = self.error_log.lock().expect("no panic while logging");
        if error_log.admit() {
            emit();
        }
    }

    /// Stamp with the clock's now, build and publish ([`Self::emit_at`]). A
    /// clock that has not ticked forms no message, and the next tick tries
    /// again.
    async fn emit(
        &self,
        build: impl FnOnce(SystemTime) -> peppygen::Result<Payload>,
    ) -> Option<SystemTime> {
        match clock_now() {
            Ok(timestamp) => self.emit_at(timestamp, build).await,
            Err(e) => {
                self.log_throttled(|| error!("{} build: {e}", self.what));
                None
            }
        }
    }

    /// Build under `timestamp` and publish, naming the slot in either
    /// failure. A publish error is a transient wire condition; a build error
    /// means the message was never formed. Neither is fatal: the next tick
    /// tries again. The stamp the message was formed under, once it was.
    async fn emit_at(
        &self,
        timestamp: SystemTime,
        build: impl FnOnce(SystemTime) -> peppygen::Result<Payload>,
    ) -> Option<SystemTime> {
        let msg = match build(timestamp) {
            Ok(msg) => msg,
            Err(e) => {
                self.log_throttled(|| error!("{} build: {e}", self.what));
                return None;
            }
        };
        if let Err(e) = self.publisher.publish(msg).await {
            self.log_throttled(|| warn!("{} publish: {e}", self.what));
        }
        Some(timestamp)
    }
}

/// Every publisher the coordination loop owns, declared once at bringup.
pub struct Publishers {
    /// The governed joint setpoints, one per paired arm.
    pub arm_setpoints: PerArm<Publisher<JointBuild>, ARMS>,
    /// The governed gripper opening fractions, one per paired gripper.
    pub gripper_setpoints: PerArm<Publisher<OpeningBuild>, ARMS>,
    /// Each arm's measured state, relayed up its leader slot so the leading
    /// node sees the same back-channel a follower gives the backbone.
    pub arm_states: PerArm<Publisher<JointBuild>, ARMS>,
    /// Each gripper's measured state, relayed up its leader slot.
    pub gripper_states: PerArm<Publisher<ApertureBuild>, ARMS>,
    /// Each arm's measured end-effector pose, relayed up its Cartesian leader
    /// slot in pose mode (an unpaired slot is a no-op).
    pub arm_pose_states: PerArm<Publisher<PoseBuild>, ARMS>,
    /// The governor proximity readout, on its contract-backed topic.
    status: Publisher<StatusBuild>,
    /// The whole-robot state snapshot, on its contract-backed topic.
    limb_states: Publisher<LimbStateBuild>,
    /// The transforms of the joints each snapshot measures, on transforms.
    transforms: Publisher<TransformsBuild>,
    /// What the transforms of a snapshot are worked out from.
    joint_transforms: JointTransforms,
    /// The robot's tree, which takes each message of the transforms of a
    /// snapshot as it is published, under `measured_by`.
    tree: Arc<RobotTree>,
    /// This backbone as a publisher of the robot's tree: its core node and
    /// its instance id.
    measured_by: TreePublisher,
    /// The robot frame, which every measured pose this node publishes is in.
    robot_frame: String,
}

impl Publishers {
    pub async fn declare(
        runner: &NodeRunner,
        joint_transforms: JointTransforms,
        tree: Arc<RobotTree>,
    ) -> peppygen::Result<Self> {
        let processor = runner.processor();
        Ok(Self {
            robot_frame: joint_transforms.robot_frame().to_owned(),
            joint_transforms,
            tree,
            measured_by: TreePublisher {
                core_node: processor.bound_core_node().to_owned(),
                instance_id: processor.bound_instance_id().to_owned(),
            },
            arm_setpoints: PerArm::new([
                Publisher::declare(
                    "left_arm joint_setpoints".to_owned(),
                    left_arm::joint_setpoints::declare_publisher(runner),
                    left_arm::joint_setpoints::build_message as JointBuild,
                )
                .await?,
                Publisher::declare(
                    "right_arm joint_setpoints".to_owned(),
                    right_arm::joint_setpoints::declare_publisher(runner),
                    right_arm::joint_setpoints::build_message as JointBuild,
                )
                .await?,
            ]),
            gripper_setpoints: PerArm::new([
                Publisher::declare(
                    "left_gripper gripper_setpoints".to_owned(),
                    left_gripper::gripper_setpoints::declare_publisher(runner),
                    left_gripper::gripper_setpoints::build_message as OpeningBuild,
                )
                .await?,
                Publisher::declare(
                    "right_gripper gripper_setpoints".to_owned(),
                    right_gripper::gripper_setpoints::declare_publisher(runner),
                    right_gripper::gripper_setpoints::build_message as OpeningBuild,
                )
                .await?,
            ]),
            arm_states: PerArm::new([
                Publisher::declare(
                    "upstream left_arm joint_states".to_owned(),
                    leader_left_arm::joint_states::declare_publisher(runner),
                    leader_left_arm::joint_states::build_message as JointBuild,
                )
                .await?,
                Publisher::declare(
                    "upstream right_arm joint_states".to_owned(),
                    leader_right_arm::joint_states::declare_publisher(runner),
                    leader_right_arm::joint_states::build_message as JointBuild,
                )
                .await?,
            ]),
            gripper_states: PerArm::new([
                Publisher::declare(
                    "upstream left_gripper gripper_states".to_owned(),
                    leader_left_gripper::gripper_states::declare_publisher(runner),
                    leader_left_gripper::gripper_states::build_message as ApertureBuild,
                )
                .await?,
                Publisher::declare(
                    "upstream right_gripper gripper_states".to_owned(),
                    leader_right_gripper::gripper_states::declare_publisher(runner),
                    leader_right_gripper::gripper_states::build_message as ApertureBuild,
                )
                .await?,
            ]),
            arm_pose_states: PerArm::new([
                Publisher::declare(
                    "upstream left_arm pose_states".to_owned(),
                    leader_left_arm_pose::pose_states::declare_publisher(runner),
                    leader_left_arm_pose::pose_states::build_message as PoseBuild,
                )
                .await?,
                Publisher::declare(
                    "upstream right_arm pose_states".to_owned(),
                    leader_right_arm_pose::pose_states::declare_publisher(runner),
                    leader_right_arm_pose::pose_states::build_message as PoseBuild,
                )
                .await?,
            ]),
            status: Publisher::declare(
                "collision_status".to_owned(),
                collision_status::declare_publisher(runner),
                collision_status::build_message as StatusBuild,
            )
            .await?,
            limb_states: Publisher::declare(
                "limb_states".to_owned(),
                limb_states::declare_publisher(runner),
                limb_states::build_message as LimbStateBuild,
            )
            .await?,
            transforms: Publisher::declare(
                "transforms".to_owned(),
                transforms::declare_publisher(runner),
                transforms::build_message as TransformsBuild,
            )
            .await?,
        })
    }

    /// Publish the operator readout: the nearest checked pair's signed distance
    /// and link names, plus the governor's disposition of the commanded motion.
    pub async fn send_status(
        &self,
        distance: f64,
        link_a: String,
        link_b: String,
        throttling: bool,
        stopped: bool,
    ) {
        self.status
            .send(distance, link_a, link_b, throttling, stopped)
            .await;
    }

    /// The robot frame, which every measured pose this node publishes is in.
    pub fn robot_frame(&self) -> &str {
        &self.robot_frame
    }

    /// Publish one whole-robot state snapshot, then the transforms of the
    /// joints it measures under the same timestamp, which the robot's tree
    /// takes as they are published, so a command in a link the backbone
    /// measures is placed by them.
    pub async fn send_limb_states(&self, snapshot: &LimbStateSnapshot) {
        let Some(timestamp) = self.limb_states.send(&self.robot_frame, snapshot).await else {
            return;
        };
        let message =
            match self
                .joint_transforms
                .message_at(timestamp, &snapshot.joints, &snapshot.openings)
            {
                Ok(message) => message,
                Err(e) => {
                    return self
                        .transforms
                        .log_throttled(|| error!("transforms build: {e}"));
                }
            };
        if let Err(e) = self.tree.take_moving(&self.measured_by, &message) {
            self.transforms
                .log_throttled(|| error!("the robot's tree refused the transforms: {e}"));
        }
        self.transforms.send(message).await;
    }
}

/// Declare one publisher, naming it in the error.
async fn declare(
    what: &str,
    declaring: impl Future<Output = peppygen::Result<TopicPublisher>>,
) -> peppygen::Result<TopicPublisher> {
    declaring
        .await
        .inspect_err(|e| error!("declare {what} publisher: {e}"))
}
