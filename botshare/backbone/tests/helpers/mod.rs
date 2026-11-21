//! The rig the base's harness tests drive: the node booted on a simulated
//! clock the test ticks and told the robot's model, robot_description's
//! botshare_base, and that the robot is ready, as the robot's initializer
//! tells it, the wheels' follower
//! played by the `base` pair's mock and the leader by the `drive` pair's
//! mock. Each step publishes the wheels' measurement stamped with the step's
//! time, waits until the node took it (its transforms of the wheel joints
//! carry the stamp), ticks the clock to that time and waits for the node's
//! base_states of that measurement, which the node publishes once its clock
//! reads the stamp, after the steps up to it. The node so runs each step after
//! the measurement of its time, and the test's time is the node's, whatever
//! the host's speed. A silent step ticks the clock alone and waits for
//! nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use control_core::fault::Fault;
use frame_tree::Stamp;
use frame_tree::arithmetic::quaternion_of_yaw;
use peppygen::fixtures::emitted_topics::transforms::transforms;
use peppygen::fixtures::exposed_actions::base_motion::{drive_base, move_base};
use peppygen::fixtures::exposed_services::robot_stop::stop;

mod parameters;
pub use parameters::*;
use peppygen::fixtures::harness::{Config, Harness, Mocks, PairingMocks};
use peppygen::mock::deps::{other_transforms, robot_identity};
use peppygen::mock::pairings::{base, drive};

/// One step of the velocity limiter (ns): 20 Hz.
pub const STEP_NS: i64 = 50_000_000;

/// One step (s).
pub const DT: f64 = 0.05;

/// The time the rig starts at (ns): a whole step.
pub const START_NS: i64 = 1_000_000_000;

/// How long a leader's stream lasts after its last setpoint arrives (ns):
/// drive_link's 0.5 s.
pub const LEADER_STREAMING_WINDOW_NS: i64 = 500_000_000;

/// How many steps without a measurement the wheels still report:
/// base_state's 0.5 s of silence. The step after them finds them silent.
pub const REPORTING_SILENT_STEPS: usize = 10;

/// The model the rig names, and its wheels as its URDF gives them.
pub const MODEL: &str = "botshare_base";
pub const WHEEL_RADIUS_M: f64 = 0.1;
pub const WHEEL_SEPARATION_M: f64 = 0.4;
/// The child links of the wheel joints, in the URDF's order of joints.
pub const WHEEL_LINKS: [&str; 2] = ["left_wheel_link", "right_wheel_link"];
/// How far a moving wheel's measured angle may lie from its last setpoint
/// for the next setpoint to start from it: one step of travel at the
/// fastest wheel rate the limits allow.
pub const ANCHOR_BOUND_RAD: f64 =
    (MAX_LINEAR_SPEED_M_S + MAX_ANGULAR_SPEED_RAD_S * WHEEL_SEPARATION_M / 2.0) / WHEEL_RADIUS_M
        * DT;

/// How long a test waits for what the node answers once its time has come:
/// a bound on a test that fails, never a pass condition.
pub const ANSWER_BUDGET: Duration = Duration::from_secs(30);

/// A wire time as nanoseconds, as its stamp counts them.
pub fn ns(time: SystemTime) -> i64 {
    let stamp = Stamp::from_time(time).expect("a time a stamp holds");
    i64::try_from(stamp.as_nanos()).expect("a time of the rig's clock")
}

/// Nanoseconds of the rig's clock as a wire time.
fn wire(time_ns: i64) -> SystemTime {
    Stamp::from_nanos(u64::try_from(time_ns).expect("a time of the rig's clock")).time()
}

/// The forward speed (m/s) of a wheel setpoint's rates.
pub fn forward_speed(setpoint: &base::joint_setpoints::Message) -> f64 {
    (setpoint.velocities[0] + setpoint.velocities[1]) * WHEEL_RADIUS_M / 2.0
}

/// The turn rate (rad/s) of a wheel setpoint's rates.
pub fn turn_rate(setpoint: &base::joint_setpoints::Message) -> f64 {
    (setpoint.velocities[1] - setpoint.velocities[0]) * WHEEL_RADIUS_M / WHEEL_SEPARATION_M
}

/// A velocity in the robot frame at the latest time: `linear_x` (m/s) ahead
/// and `turn` (rad/s) about +z.
pub fn velocity(
    linear_x: f64,
    turn: f64,
) -> peppygen::exposed_actions::base_motion::drive_base::GoalVelocity {
    peppygen::exposed_actions::base_motion::drive_base::GoalVelocity {
        frame: None,
        timestamp: None,
        linear_velocity: [linear_x, 0.0, 0.0],
        angular_velocity: [0.0, 0.0, turn],
    }
}

/// A drive_base goal at `linear_x` (m/s) ahead, held for
/// `hold_duration_s`, the base given 1 s to rest.
pub fn drive_goal(linear_x: f64, hold_duration_s: f64) -> drive_base::GoalRequestData {
    drive_base::GoalRequestData {
        velocity: velocity(linear_x, 0.0),
        hold_duration_s,
        settle_timeout_s: 1.0,
    }
}

/// A target in the robot frame at the latest time, at (x, y) turned `yaw`.
pub fn target(
    x: f64,
    y: f64,
    yaw: f64,
) -> peppygen::exposed_actions::base_motion::move_base::GoalTarget {
    peppygen::exposed_actions::base_motion::move_base::GoalTarget {
        frame: None,
        timestamp: None,
        position: [x, y, 0.0],
        orientation: quaternion_of_yaw(yaw),
    }
}

/// A move_base goal to (x, y, yaw) in the robot frame as fast as the limits
/// allow, within 2 cm and 0.03 rad, settled within `settle_timeout_s`.
pub fn move_goal(x: f64, y: f64, yaw: f64, settle_timeout_s: f64) -> move_base::GoalRequestData {
    move_to(target(x, y, yaw), settle_timeout_s)
}

/// A move_base goal to `target` as fast as the limits allow, within 2 cm and
/// 0.03 rad, settled within `settle_timeout_s`.
pub fn move_to(
    target: peppygen::exposed_actions::base_motion::move_base::GoalTarget,
    settle_timeout_s: f64,
) -> move_base::GoalRequestData {
    move_base::GoalRequestData {
        target,
        duration_s: 0.0,
        settle_position_tolerance_m: 0.02,
        settle_orientation_tolerance_rad: 0.03,
        settle_timeout_s,
    }
}

/// Answers every robot_init `is_ready` poll of the node with the value of
/// `ready` at the poll, as the robot's initializer does, until the mock's
/// session closes.
fn answer_readiness(
    service: peppygen::mock::deps::robot_init::is_ready::Service,
    ready: Arc<AtomicBool>,
) {
    use peppygen::mock::deps::robot_init::is_ready::ResponseData;
    tokio::spawn(async move {
        while let Ok(responder) = service.next_request(ANSWER_BUDGET).await {
            let ready = ready.load(Ordering::SeqCst);
            if responder.respond(ResponseData { ready }).await.is_err() {
                break;
            }
        }
    });
}

/// One base_states message of the node.
pub type BaseState = peppygen::fixtures::emitted_topics::base_state::base_states::Message;

pub struct Rig {
    pub harness: Harness,
    pub pairings: PairingMocks,
    /// The robot's initializer, as the node asks it for the robot.
    pub identity: robot_identity::Mock,
    /// The robot's initializer, as a publisher of the robot's tree.
    pub initializer_transforms: other_transforms::Mock,
    /// What the robot's initializer answers to the node's polls of its
    /// readiness.
    pub ready: Arc<AtomicBool>,
    /// The node's fault, which a robot it cannot run raises.
    pub fault: Fault,
    /// The time the rig's clock reads (ns).
    pub now_ns: i64,
    /// The time the test's timeline starts at (ns): [`START_NS`], or the
    /// time at which the node took the setpoint of
    /// [`Rig::lead_until_it_drives`].
    pub origin_ns: i64,
    /// The wheel angles the follower reports (rad), [left, right].
    pub wheels: [f64; 2],
}

impl Rig {
    /// The node on a simulated clock that reads no time yet, before the
    /// wheels report, with its leader paired on drive_link, its model named,
    /// its description served and the robot ready: the wheels report
    /// `wheels` once the rig reports them.
    pub async fn boot(wheels: [f64; 2]) -> Self {
        let rig = Self::boot_unnamed(wheels).await;
        rig.name_model(MODEL)
            .await
            .expect("the node asks for the robot's model");
        rig.serve_description(None)
            .await
            .expect("the node asks for the robot's description");
        rig.ready.store(true, Ordering::SeqCst);
        rig
    }

    /// Answer the node's ask of robot_identity's `get_description` as the
    /// initializer of a robot of [`MODEL`] does, its digest `digest` when
    /// given and the bundled description's otherwise.
    pub async fn serve_description(&self, digest: Option<&str>) -> peppygen::Result<()> {
        use peppygen::mock::deps::robot_identity::get_description::ResponseData;
        let bundled = robot_description::model(MODEL).expect("the bundled base");
        self.identity
            .get_description
            .next_request(ANSWER_BUDGET)
            .await?
            .respond(ResponseData {
                urdf: bundled.urdf().to_owned(),
                srdf: bundled.srdf().to_owned(),
                meshes: Vec::new(),
                digest: digest.unwrap_or(bundled.digest()).to_owned(),
            })
            .await
    }

    /// The node on a simulated clock that reads no time yet, with its leader
    /// paired on drive_link, before the robot's initializer answers.
    pub async fn boot_unnamed(wheels: [f64; 2]) -> Self {
        let fault = Fault::default();
        let raised = fault.clone();
        let (harness, mocks) = Harness::start_with(
            Config {
                parameters: Some(params()),
                clock: peppylib::testing::HarnessClock::Consumer,
                ..Config::default()
            },
            move |params, runner| botshare_backbone::setup(params, runner, raised.clone()),
        )
        .await
        .expect("the harness starts the node");
        let Mocks { deps, pairings, .. } = mocks;
        let ready = Arc::new(AtomicBool::new(false));
        answer_readiness(deps.robot_init.is_ready, ready.clone());
        let [initializer_transforms] = <[_; 1]>::try_from(deps.other_transforms)
            .unwrap_or_else(|_| panic!("the rig binds the initializer alone on other_transforms"));
        Self {
            harness,
            pairings,
            identity: deps.robot_identity,
            initializer_transforms,
            ready,
            fault,
            now_ns: 0,
            origin_ns: START_NS,
            wheels,
        }
    }

    /// Answer the node's ask of robot_identity's `get_identity` as the
    /// initializer of a robot of `model` does.
    pub async fn name_model(&self, model: &str) -> peppygen::Result<()> {
        use peppygen::mock::deps::robot_identity::get_identity::ResponseData;
        self.identity
            .get_identity
            .next_request(ANSWER_BUDGET)
            .await?
            .respond(ResponseData {
                robot: "base".to_owned(),
                model: model.to_owned(),
                core_node: "core".to_owned(),
            })
            .await
    }

    /// The node on a simulated clock at [`START_NS`], the wheels reporting
    /// `wheels` there.
    pub async fn start(wheels: [f64; 2]) -> Self {
        let mut rig = Self::boot(wheels).await;
        rig.report_first().await;
        rig
    }

    /// Report the wheels at [`START_NS`], tick the clock there and wait for
    /// the node's state of that measurement. The wheels' first measurement
    /// and the clock's first tick each wait for the node's subscription.
    pub async fn report_first(&mut self) {
        self.now_ns = START_NS;
        self.report_wheels().await;
        self.tick().await;
        self.state_at(START_NS).await;
    }

    /// The time `seconds` after the test's origin ([`Rig::origin_ns`]) (ns).
    pub fn at(&self, seconds: f64) -> i64 {
        self.origin_ns + (seconds * 1e9).round() as i64
    }

    /// Tick the clock to the time the rig reads.
    pub async fn tick(&self) {
        self.harness
            .clock
            .tick(self.now_ns as u64)
            .await
            .expect("the clock ticks");
    }

    /// Publish the wheels' measurement, stamped with the time the rig reads.
    pub async fn publish_wheels(&self) {
        self.publish_wheels_stamped(self.now_ns).await;
    }

    /// Publish the wheels' measurement, stamped with the time the rig reads,
    /// and wait until the node took it: its transforms of the wheel joints
    /// carry the stamp. Returns those transforms.
    pub async fn report_wheels(&mut self) -> transforms::Message {
        self.publish_wheels().await;
        let stamp_ns = self.now_ns;
        self.transforms_at(stamp_ns, &WHEEL_LINKS).await
    }

    /// Publish the wheels' measurement, stamped `stamp_ns`.
    pub async fn publish_wheels_stamped(&self, stamp_ns: i64) {
        self.publish_wheels_turning(stamp_ns, Vec::new()).await;
    }

    /// Publish the wheels' measurement, stamped `stamp_ns`, the wheels
    /// turning at `rates` (rad/s), as a follower that runs a setpoint reports
    /// them.
    pub async fn publish_wheels_turning(&self, stamp_ns: i64, rates: Vec<f64>) {
        self.pairings
            .base
            .joint_states
            .publish(&base::joint_states::Message {
                timestamp: wire(stamp_ns),
                positions: self.wheels,
                velocities: rates,
                efforts: Vec::new(),
            })
            .await
            .expect("the wheels publish");
    }

    /// The node's transforms message of time `t_ns` whose child frames are
    /// `children`, skipping every other.
    pub async fn transforms_at(&mut self, t_ns: i64, children: &[&str]) -> transforms::Message {
        let messages = &mut self.harness.emitted.transforms_transforms;
        tokio::time::timeout(ANSWER_BUDGET, async {
            loop {
                let message = messages
                    .next()
                    .await
                    .expect("transforms receive")
                    .expect("transforms open");
                let message_children = message.transforms.iter().map(|t| t.child_frame.as_str());
                if ns(message.timestamp) == t_ns && message_children.eq(children.iter().copied()) {
                    return message;
                }
            }
        })
        .await
        .expect("the node publishes the transforms at the time")
    }

    /// The node's first base_states of time `t_ns` or later.
    pub async fn state_at(&mut self, t_ns: i64) -> BaseState {
        let states = &mut self.harness.emitted.base_state_base_states;
        tokio::time::timeout(ANSWER_BUDGET, async {
            loop {
                let state = states
                    .next()
                    .await
                    .expect("base_states receive")
                    .expect("base_states open");
                if ns(state.timestamp) >= t_ns {
                    return state;
                }
            }
        })
        .await
        .expect("the node publishes its state at the time")
    }

    /// The node's first drive_states of time `t_ns` or later.
    pub async fn drive_state_at(&mut self, t_ns: i64) -> drive::drive_states::Message {
        let states = &mut self.pairings.drive.drive_states;
        tokio::time::timeout(ANSWER_BUDGET, async {
            loop {
                let state = states
                    .next()
                    .await
                    .expect("drive_states receive")
                    .expect("drive_states open");
                if ns(state.timestamp) >= t_ns {
                    return state;
                }
            }
        })
        .await
        .expect("the node publishes drive_states at the time")
    }

    /// Step with the wheels reporting: their measurement of the next step,
    /// taken by the node, then the clock's tick, and the node's state of
    /// that measurement.
    pub async fn step(&mut self) -> BaseState {
        self.now_ns += STEP_NS;
        self.report_wheels().await;
        self.tick().await;
        self.state_at(self.now_ns).await
    }

    /// Tick one step with the wheels silent.
    pub async fn step_silently(&mut self) {
        self.now_ns += STEP_NS;
        self.tick().await;
    }

    /// Step until the clock reads `t_ns`.
    pub async fn step_to(&mut self, t_ns: i64) {
        while self.now_ns < t_ns {
            self.step().await;
        }
    }

    /// Step with the wheels where the node's last setpoint sent them, read
    /// the node's setpoint of the step, and move the wheels there for the
    /// next one: a follower that reaches each setpoint by its next
    /// measurement.
    pub async fn step_following(&mut self) {
        self.step().await;
        let setpoint = self.wheel_setpoint_at(self.now_ns).await;
        self.wheels = setpoint.positions;
    }

    /// Step with the wheels following, as [`Rig::step_following`] does, until
    /// `goal`'s result comes. A goal that ended leaves the limiter its last
    /// setpoint for 0.5 s, after which the node sends no setpoint, so each
    /// step waits for its setpoint or the result, whichever comes.
    pub async fn follow_until_result(
        &mut self,
        goal: &move_base::GoalHandle,
    ) -> move_base::ResultResponse {
        let result = goal.get_result(ANSWER_BUDGET);
        tokio::pin!(result);
        loop {
            self.step().await;
            let at = self.now_ns;
            let setpoint = tokio::select! {
                result = &mut result => return result.expect("a result"),
                setpoint = self.wheel_setpoint_at(at) => setpoint,
            };
            self.wheels = setpoint.positions;
        }
    }

    /// The node's wheel setpoint of the step at `t_ns`, skipping earlier
    /// ones.
    pub async fn wheel_setpoint_at(&mut self, t_ns: i64) -> base::joint_setpoints::Message {
        let setpoints = &mut self.pairings.base.joint_setpoints;
        tokio::time::timeout(ANSWER_BUDGET, async {
            loop {
                let setpoint = setpoints
                    .next()
                    .await
                    .expect("joint_setpoints receive")
                    .expect("joint_setpoints open");
                let at = ns(setpoint.timestamp);
                assert!(
                    at <= t_ns,
                    "no wheel setpoint at {t_ns} ns, the next is at {at} ns"
                );
                if at == t_ns {
                    return setpoint;
                }
            }
        })
        .await
        .expect("the node streams a wheel setpoint at the time")
    }

    /// The node's next wheel setpoint, in the order it streamed them.
    pub async fn next_wheel_setpoint(&mut self) -> base::joint_setpoints::Message {
        let setpoints = &mut self.pairings.base.joint_setpoints;
        tokio::time::timeout(ANSWER_BUDGET, setpoints.next())
            .await
            .expect("the node streams a wheel setpoint")
            .expect("joint_setpoints receive")
            .expect("joint_setpoints open")
    }

    /// Publish the leader's setpoint on drive_link.
    pub async fn lead(&self, linear_velocity: [f64; 3], yaw_rate: f64) {
        self.pairings
            .drive
            .drive_setpoints
            .publish(&drive::drive_setpoints::Message {
                frame: None,
                timestamp: Some(wire(self.now_ns)),
                linear_velocity,
                angular_velocity: [0.0, 0.0, yaw_rate],
            })
            .await
            .expect("the leader publishes");
    }

    /// Lead the base at `linear_velocity` and `yaw_rate`, the clock standing
    /// still, until the node follows the setpoint, and start the test's
    /// timeline at the time the node took it ([`Rig::origin_ns`]). A probe
    /// goal refused with `leader_streaming` proves the node took the
    /// setpoint while no goal held the base. A probe admitted before the
    /// setpoint arrived keeps it from the base, as drive_link states: the rig
    /// ends that probe and the leader's stream, then leads again.
    pub async fn lead_until_it_drives(&mut self, linear_velocity: [f64; 3], yaw_rate: f64) {
        /// How many leads the rig sends before it gives up on the node.
        const LEADS: usize = 10;
        for _ in 0..LEADS {
            self.lead(linear_velocity, yaw_rate).await;
            let Some(probe) = self.probe_until_the_stream_refuses().await else {
                self.origin_ns = self.now_ns;
                return;
            };
            let taken_ns = self.now_ns;
            self.end_probe(probe).await;
            self.step_to(taken_ns + LEADER_STREAMING_WINDOW_NS + STEP_NS)
                .await;
        }
        panic!("the node follows none of {LEADS} leads");
    }

    /// Send probe goals, the clock standing still, until one is refused with
    /// `leader_streaming`, which proves the node took a setpoint the leader
    /// published. A probe admitted first is ended once one is refused.
    pub async fn until_the_leader_stream_refuses_a_goal(&mut self) {
        if let Some(probe) = self.probe_until_the_stream_refuses().await {
            self.end_probe(probe).await;
        }
    }

    /// Send drive_base goals, the clock standing still, until one is refused
    /// with `leader_streaming`. A probe the node admits first runs, so later
    /// probes are refused with `precondition_failed` until the leader's
    /// setpoint arrives, which the running probe keeps from the base: that
    /// probe is returned.
    async fn probe_until_the_stream_refuses(&self) -> Option<drive_base::GoalHandle> {
        let probes = async {
            let mut admitted = None;
            loop {
                let probe = self.send_drive(&drive_goal(0.1, 1.0)).await;
                match probe.refusal.as_ref().map(|refusal| refusal.kind) {
                    None => {
                        assert!(admitted.is_none(), "a probe is admitted while another runs");
                        admitted = Some(probe);
                    }
                    Some(drive_base::FailureKind::LeaderStreaming) => return admitted,
                    Some(drive_base::FailureKind::PreconditionFailed) if admitted.is_some() => {}
                    Some(_) => panic!("a probe is refused: {:?}", probe.refusal),
                }
            }
        };
        tokio::time::timeout(ANSWER_BUDGET, probes)
            .await
            .expect("a probe is refused with leader_streaming")
    }

    /// Cancel `probe`, an admitted drive that has not moved the base, and
    /// step once: a cancelled goal ends at the next step, at rest.
    async fn end_probe(&mut self, probe: drive_base::GoalHandle) {
        let cancel = probe
            .cancel_goal(ANSWER_BUDGET)
            .await
            .expect("the cancel answers");
        assert_eq!(cancel, drive_base::CancelState::Signalled);
        self.step().await;
        let result = probe.get_result(ANSWER_BUDGET).await.expect("a result");
        assert!(
            matches!(result.outcome, drive_base::ResultOutcome::Cancelled { .. }),
            "{:?}",
            result.outcome
        );
    }

    pub async fn send_drive(&self, goal: &drive_base::GoalRequestData) -> drive_base::GoalHandle {
        drive_base::send_goal(
            &self.harness,
            goal,
            peppygen::DEFAULT_FEEDBACK_BOUND,
            ANSWER_BUDGET,
        )
        .await
        .expect("drive_base answers")
    }

    pub async fn send_move(&self, goal: &move_base::GoalRequestData) -> move_base::GoalHandle {
        move_base::send_goal(
            &self.harness,
            goal,
            peppygen::DEFAULT_FEEDBACK_BOUND,
            ANSWER_BUDGET,
        )
        .await
        .expect("move_base answers")
    }

    /// The robot stop, for `reason`: the parts it stopped.
    pub async fn stop(&self, reason: &str) -> Vec<String> {
        stop::poll(
            &self.harness,
            &stop::RequestData {
                reason: reason.to_string(),
            },
            ANSWER_BUDGET,
        )
        .await
        .expect("robot_stop's stop answers")
        .stopped
    }

    /// Publish the fixed joints of [`MODEL`]'s URDF on the robot's tree as the
    /// robot's initializer publishes them, on static_transforms.
    pub async fn publish_fixed_transforms(&self) {
        use peppygen::consumed_topics::other_transforms::static_transforms;
        let description = robot_description::model(MODEL)
            .expect("the bundled base")
            .description();
        let transforms = description
            .fixed_transforms()
            .iter()
            .map(|link| {
                let transform = link.frame_transform().expect("a fixed joint of the URDF");
                static_transforms::MessageTransformsItem {
                    frame: transform.parent().as_str().to_owned(),
                    child_frame: transform.child().as_str().to_owned(),
                    position: transform.placement().position(),
                    orientation: transform.placement().orientation(),
                }
            })
            .collect();
        self.initializer_transforms
            .static_transforms
            .publish(&static_transforms::Message { transforms })
            .await
            .expect("the initializer publishes its fixed joints");
    }

    /// The answer to `goal`, whose placement waits on the node's clock for the
    /// robot's tree: the clock ticks on, a step at a time with the wheels
    /// silent, until the node answers. The rig's own time stays where it was.
    pub async fn send_move_ticking(
        &self,
        goal: &move_base::GoalRequestData,
    ) -> move_base::GoalHandle {
        let sent = self.send_move(goal);
        tokio::pin!(sent);
        let mut ticked_ns = self.now_ns;
        loop {
            ticked_ns += STEP_NS;
            tokio::select! {
                handle = &mut sent => return handle,
                ticked = self.harness.clock.tick(ticked_ns as u64) => ticked.expect("the clock ticks"),
            }
        }
    }

    /// [`Rig::send_move_ticking`] for a drive_base goal.
    pub async fn send_drive_ticking(
        &self,
        goal: &drive_base::GoalRequestData,
    ) -> drive_base::GoalHandle {
        let sent = self.send_drive(goal);
        tokio::pin!(sent);
        let mut ticked_ns = self.now_ns;
        loop {
            ticked_ns += STEP_NS;
            tokio::select! {
                handle = &mut sent => return handle,
                ticked = self.harness.clock.tick(ticked_ns as u64) => ticked.expect("the clock ticks"),
            }
        }
    }

    pub async fn shutdown(self) {
        self.harness
            .shutdown()
            .await
            .expect("the node shuts down cleanly");
    }
}
