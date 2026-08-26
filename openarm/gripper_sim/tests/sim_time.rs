//! Sim-time integration tests over the generated harness: the test drives
//! the daemon-clock stand-in (`Config::use_sim_time` plus `harness.clock`),
//! so the node's clock decisions are asserted at exact virtual instants.
//! Wall-mode relay behavior is covered in `tests/relay.rs`; this file covers
//! only what needs a driven clock.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::motor_health::{HEALTH_PERIOD, STATE_STALE_AFTER};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::pairings::backbone::gripper_setpoints as backbone_setpoints;
use peppygen::mock::pairings::engine::gripper_states as engine_states;

/// Motors per gripper, matching the node's health `level` vector length.
const MOTORS: usize = 1;

/// The first sim instant the test drives.
const T1_NS: u64 = 1_000_000_000;

/// Exactly one staleness window past T1: the earliest instant at which a
/// T1-stamped state stops counting as recent (the gate is a strict
/// `age < STATE_STALE_AFTER`).
const T2_NS: u64 = T1_NS + STATE_STALE_AFTER.as_nanos() as u64;

/// Liveness bound on every wire wait; no assertion depends on its value.
const RECV_TIMEOUT: Duration = Duration::from_secs(10);

fn instant(ns: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_nanos(ns)
}

fn gripper_state(timestamp: SystemTime) -> engine_states::Message {
    engine_states::Message {
        timestamp,
        opening: 0.5,
        effort: 0.25,
        max_effort: 1.0,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sim_time_stamps_the_heartbeat_and_bounds_the_recency_gate() -> peppygen::Result<()> {
    let (mut harness, mut mocks) = Harness::start_with(
        Config {
            use_sim_time: true,
            ..Default::default()
        },
        openarm_gripper_sim::setup,
    )
    .await?;

    // Before the first tick the sim clock has no time at all, yet the
    // setpoint leg relays: the passthrough takes no clock read.
    let setpoint_ts = instant(500_000_000);
    mocks
        .pairings
        .backbone
        .gripper_setpoints
        .publish(&backbone_setpoints::Message {
            timestamp: setpoint_ts,
            opening: 0.75,
            max_effort: 2.5,
        })
        .await?;
    let relayed =
        tokio::time::timeout(RECV_TIMEOUT, mocks.pairings.engine.gripper_setpoints.next())
            .await
            .expect("the engine should receive the relayed setpoint before any tick")?
            .expect("engine setpoints subscription should be open");
    assert_eq!(relayed.timestamp, setpoint_ts);

    // First driven instant: the engine speaks, stamped with the same sim
    // time the node reads.
    harness.clock.tick(T1_NS).await?;
    mocks
        .pairings
        .engine
        .gripper_states
        .publish(&gripper_state(instant(T1_NS)))
        .await?;
    let relayed = tokio::time::timeout(RECV_TIMEOUT, mocks.pairings.backbone.gripper_states.next())
        .await
        .expect("the backbone should receive the relayed state")?
        .expect("backbone states subscription should be open");
    assert_eq!(relayed.timestamp, instant(T1_NS));

    // The heartbeat stamps the exact driven instant: one clock read serves
    // both the recency gate and the timestamp, and that read is the test's
    // tick.
    let health = tokio::time::timeout(
        RECV_TIMEOUT,
        harness.emitted.motor_health_motor_health.next(),
    )
    .await
    .expect("the heartbeat should start after the first state")?
    .expect("motor_health subscription should be open");
    assert_eq!(health.timestamp, instant(T1_NS));
    assert_eq!(health.level, vec![0u8; MOTORS]);

    // The gate's boundary, exactly. At T2 the T1 state is one whole
    // STATE_STALE_AFTER old and the gate is a strict `age <
    // STATE_STALE_AFTER`, so the heartbeat must hold. Holding is an absence,
    // and an absence reads as silence: with the clock parked at T2, drain
    // until the topic stays quiet for several health periods. Every heartbeat
    // carries the very clock read that passed the gate, so a T2-stamped one
    // could only come from a gate that admitted a state exactly
    // STATE_STALE_AFTER old. Nothing newer is published until this drain
    // finishes, so no delivery race can forge that stamp.
    harness.clock.tick(T2_NS).await?;
    let drain_deadline = tokio::time::Instant::now() + RECV_TIMEOUT;
    loop {
        match tokio::time::timeout(
            HEALTH_PERIOD * 3,
            harness.emitted.motor_health_motor_health.next(),
        )
        .await
        {
            // Quiet for three-plus periods: the heartbeat is held.
            Err(_) => break,
            Ok(next) => {
                let health = next?.expect("motor_health subscription should be open");
                assert_eq!(
                    health.timestamp,
                    instant(T1_NS),
                    "the heartbeat vouched at T2, where the newest state it held \
                     was exactly STATE_STALE_AFTER old"
                );
            }
        }
        assert!(
            tokio::time::Instant::now() < drain_deadline,
            "the heartbeat never held once the only state went stale"
        );
    }

    // A fresh state reopens it. Stamped T2 against a clock still parked at
    // T2, so the gate sees age zero without a second tick: the node's clock
    // is provably already T2, having been read at it throughout the drain,
    // and no new instant can race the state's delivery.
    mocks
        .pairings
        .engine
        .gripper_states
        .publish(&gripper_state(instant(T2_NS)))
        .await?;
    let health = tokio::time::timeout(
        RECV_TIMEOUT,
        harness.emitted.motor_health_motor_health.next(),
    )
    .await
    .expect("the heartbeat should resume on the fresh state")?
    .expect("motor_health subscription should be open");
    assert_eq!(health.timestamp, instant(T2_NS));

    harness.shutdown().await
}
