//! The streams the base receives: the wheels' measured angles on the `base`
//! pair, with the instance that sent them, and the leader's velocity
//! setpoints on the `drive` pair, each handed to the control loop in arrival
//! order. A measurement with a value that is not finite is dropped. A setpoint
//! is placed in the robot frame by the robot's tree, at its timestamp or at the
//! latest time for none, and is dropped, as drive_link counts it as no
//! setpoint, when a value is not finite, when the tree cannot place it, and
//! when, placed, its linear_velocity's z or its angular_velocity's x or y lies
//! farther than 1e-9 from 0. A setpoint's y reaches the limiter, whose range
//! along y on a differential base is 0 alone. Each stream logs at most one
//! dropped message per [`LOG_WINDOW`].

use std::ops::ControlFlow;
use std::sync::Arc;

use control_core::geometry::{Stamped, Twist};
use control_core::receiving::receive;
use control_core::robot_tree::Placer;
use control_core::throttle::{LOG_WINDOW, Throttle};
use control_core::twist::PlanarTwist;
use peppygen::NodeRunner;
use peppygen::paired_topics::base::joint_states;
use peppygen::paired_topics::drive::drive_setpoints;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::clock;
use crate::control::{Event, WheelState};
use crate::goal::velocity_off_the_floor;

/// Receive the wheels' measured angles until the node cancels.
pub async fn run_wheel_listener(runner: Arc<NodeRunner>, events: mpsc::Sender<Event>) {
    let mut subscription = match joint_states::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("wheel joint_states subscribe: {e}"),
    };
    let mut drop_log = Throttle::new(LOG_WINDOW);
    receive(
        "wheel joint_states",
        runner.cancellation_token().cancelled(),
        async move || subscription.next().await,
        async move |(peer, message): (peppygen::PeerInfo, joint_states::Message)| {
            let finite = message.positions.iter().all(|angle| angle.is_finite());
            let Some(timestamp_ns) = clock::ns_from_wire(message.timestamp).filter(|_| finite)
            else {
                if drop_log.admit() {
                    warn!("wheel joint_states: dropping a measurement that is not finite");
                }
                return ControlFlow::Continue(());
            };
            let state = WheelState {
                positions: message.positions,
                timestamp_ns,
                follower: peer.producer.instance_id,
            };
            match events.send(Event::Wheels(state)).await {
                Ok(()) => ControlFlow::Continue(()),
                Err(_) => ControlFlow::Break(()),
            }
        },
    )
    .await;
}

/// Receive the leader's setpoints until the node cancels, each stamped with
/// its arrival on the node's clock, then placed in the robot frame by
/// `placer`.
pub async fn run_setpoint_listener(
    runner: Arc<NodeRunner>,
    placer: Arc<Placer>,
    events: mpsc::Sender<Event>,
) {
    let mut subscription = match drive_setpoints::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("drive_setpoints subscribe: {e}"),
    };
    let mut drop_log = Throttle::new(LOG_WINDOW);
    receive(
        "drive_setpoints",
        runner.cancellation_token().cancelled(),
        async move || subscription.next().await,
        async move |(_peer, message): (peppygen::PeerInfo, drive_setpoints::Message)| {
            let Some(received_ns) = clock::now_ns() else {
                return ControlFlow::Continue(());
            };
            let velocity = match placed_setpoint(&placer, message).await {
                Ok(velocity) => velocity,
                Err(reason) => {
                    if drop_log.admit() {
                        warn!("drive_setpoints: dropping a setpoint: {reason}");
                    }
                    return ControlFlow::Continue(());
                }
            };
            let event = Event::Setpoint {
                velocity,
                received_ns,
            };
            match events.send(event).await {
                Ok(()) => ControlFlow::Continue(()),
                Err(_) => ControlFlow::Break(()),
            }
        },
    )
    .await;
}

/// The velocity of `message` placed in the robot frame on the floor plane, or
/// why drive_link counts the setpoint as no setpoint.
async fn placed_setpoint(
    placer: &Placer,
    message: drive_setpoints::Message,
) -> Result<PlanarTwist, String> {
    let stamped = Twist::parse(message.linear_velocity, message.angular_velocity)
        .and_then(|twist| {
            Stamped::parse(
                placer.frame_of(message.frame.as_deref()),
                message.timestamp,
                twist,
            )
        })
        .map_err(|e| e.to_string())?;
    let placed = placer
        .place_velocity(&stamped)
        .await
        .map_err(|e| format!("the robot's tree cannot place it: {e}"))?;
    placed
        .planar()
        .map_err(|refusal| velocity_off_the_floor(&refusal))
}
