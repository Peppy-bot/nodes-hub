"""Wiring: parse parameters, bring up the device thread, and run one task per
stream: measured state out, setpoints in, health and alerts out, readiness."""

from __future__ import annotations

import asyncio
import os
import time

import peppygen.clock
from control_core_py import runtime
from control_core_py.runtime import CancellationToken
from peppygen import NodeBuilder, NodeRunner
from peppygen.emitted_topics.alerts import alerts as alerts_topic
from peppygen.emitted_topics.motor_health import motor_health as motor_health_topic
from peppygen.exposed_services.ready import is_ready as is_ready_svc
from peppygen.paired_topics.arm import joint_setpoints, joint_states
from peppygen.paired_topics.gripper import gripper_setpoints, gripper_states
from peppygen.parameters import Parameters
from so101_description import units
from so101_description.setpoints import parse_gripper_setpoint, parse_joint_setpoints

from so101_follower import params as params_mod
from so101_follower.device import (
    STALE_WIRE_TIMEOUT_S,
    DeviceLoop,
    LerobotFollower,
    StateSnapshot,
)
from so101_follower.health import (
    EWMA_TAU_S,
    HEALTH_RATE_HZ,
    SILENT_AFTER_MISSED_READS,
    Alert,
    AlertTracker,
    SustainedLoads,
    assess,
)

runtime.configure("so101_follower")

# Re-emission cadence of active alerts, inside the contract's 2000 ms bound.
_ALERT_REEMIT_S = 1.6
# A health snapshot older than this many health periods means the device
# thread is wedged; the stream then goes silent rather than republishing
# stale readings under fresh timestamps.
_HEALTH_FRESH_PERIODS = 2.0
# A state read this old means the serial bus is not serving: readiness is
# revoked (an orchestrator must not gate onto an arm nobody can read), and
# past the exit horizon the node dies loudly so the daemon records the
# failure and supervision can restart it instead of a healthy-looking
# process idling on a dead bus.
_BUS_SILENT_READY_S = 2.0
_BUS_DEAD_EXIT_S = 10.0
_BUS_WATCH_PERIOD_S = 1.0


def _now_s() -> float:
    return peppygen.clock.now_ns() / 1e9


def _captured_timestamp_s(captured_monotonic: float) -> float:
    """The wire stamp naming when the bus read happened, not when this tick
    published it."""
    return runtime.wire_timestamp_s(peppygen.clock.now_ns(), captured_monotonic)


async def _stream_states(
    node_runner: NodeRunner,
    topic_module,
    build,
    device: DeviceLoop,
    period_s: float,
    token: CancellationToken,
):
    """One measured-state stream: publish build(snapshot) for every snapshot
    the hardware loop refreshed. A tick it did not refresh is skipped, so a
    wedged loop goes silent rather than streaming stale-as-fresh."""
    publisher = await topic_module.declare_publisher(node_runner)
    failing = runtime.Latch()
    last_captured = 0.0
    async for _ in runtime.ticks(period_s, token):
        snapshot = device.latest_state()
        if snapshot is None or snapshot.captured_monotonic == last_captured:
            continue
        last_captured = snapshot.captured_monotonic
        try:
            await publisher.publish(build(snapshot))
            failing.clear()
        except Exception as e:
            failing.trip(f"{topic_module.TOPIC_NAME} publish failing: {e!r}")


async def _consume_setpoints(
    node_runner: NodeRunner,
    topic_module,
    label: str,
    parse_and_submit,
    token: CancellationToken,
):
    """One setpoint stream over the shared gated consume, which drops a
    violating message with its own edge-triggered log; the alert surface
    carries motor conditions only."""
    subscription = await topic_module.subscribe(node_runner)
    await runtime.consume_gated(
        subscription, token, label, _now_s, STALE_WIRE_TIMEOUT_S, parse_and_submit,
    )


class _AlertBus:
    """One publisher for every alert this node raises."""

    def __init__(self, publisher):
        self._publisher = publisher
        self._failing = runtime.Latch()

    async def emit(self, alert: Alert) -> None:
        try:
            await self._publisher.publish(
                alerts_topic.build_message(
                    _now_s(), alert.source, alert.kind, alert.severity, alert.message
                )
            )
            self._failing.clear()
        except Exception as e:
            self._failing.trip(f"alert publish failing: {e!r}")


async def _stream_health(
    node_runner: NodeRunner,
    device: DeviceLoop,
    config: params_mod.Config,
    alert_bus: _AlertBus,
    token: CancellationToken,
):
    publisher = await motor_health_topic.declare_publisher(node_runner)
    sustained = SustainedLoads(EWMA_TAU_S)
    tracker = AlertTracker(f"so101_follower {config.robot_id}")
    period_s = 1.0 / HEALTH_RATE_HZ
    failing = runtime.Latch()
    last_readings_stamp = 0.0
    last_reemit_monotonic = time.monotonic()
    async for _ in runtime.ticks(period_s, token):
        now = time.monotonic()
        snapshot = device.latest_health()
        # A snapshot the device thread stopped refreshing must not be
        # republished under fresh timestamps: silence is what lets the
        # contract's cadence mandate expose a dead producer.
        if snapshot is None or now - snapshot.captured_monotonic > _HEALTH_FRESH_PERIODS * period_s:
            continue
        assert len(snapshot.temps_c) == len(units.MOTOR_NAMES), (
            "bringup verifies a full-length health read before the loop ticks"
        )
        bus_silent = snapshot.consecutive_missed_reads >= SILENT_AFTER_MISSED_READS
        # The EWMA advances only on new readings, spaced by their capture
        # stamps, so failed reads and phase drift never re-feed a sample or
        # misstate dt.
        if not bus_silent and snapshot.readings_captured_monotonic != last_readings_stamp:
            dt_s = snapshot.readings_captured_monotonic - last_readings_stamp
            sustained.update(snapshot.load_fractions, dt_s)
            last_readings_stamp = snapshot.readings_captured_monotonic
        report = assess(
            snapshot.temps_c,
            snapshot.load_fractions,
            snapshot.torque_enabled,
            snapshot.fault_bits,
            sustained.current(),
            bus_silent,
        )
        try:
            await publisher.publish(
                motor_health_topic.build_message(
                    _captured_timestamp_s(snapshot.readings_captured_monotonic),
                    bytes(report.levels),
                    # The STS3215 publishes no continuous rating, so the
                    # rated fractions stay empty; the trip-point-relative
                    # peak below carries the load signal instead.
                    [],
                    [],
                    list(report.peak_fractions()),
                    # No driver temperature sensor exists on this servo.
                    [],
                    list(report.winding_temp_c),
                )
            )
            failing.clear()
        except Exception as e:
            failing.trip(f"motor_health publish failing: {e!r}")

        for alert in tracker.transitions(report):
            await alert_bus.emit(alert)
        # Wall-clock gated: slipped ticks must not stretch the cadence past
        # the contract's 2000 ms ceiling.
        if now - last_reemit_monotonic >= _ALERT_REEMIT_S:
            last_reemit_monotonic = now
            for alert in tracker.active():
                await alert_bus.emit(alert)


def _bus_age_s(device: DeviceLoop) -> float:
    """Seconds since the last successful state read."""
    snapshot = device.latest_state()
    assert snapshot is not None, "bringup verifies a first read before the watcher starts"
    return time.monotonic() - snapshot.captured_monotonic


async def _watch_bus(device: DeviceLoop, token: CancellationToken, terminate=None):
    """Die loudly once the bus stays dead past the exit horizon. The torque
    release is attempted first; on a dead bus it can only fail fast."""
    terminate = terminate if terminate is not None else os._exit
    async for _ in runtime.ticks(_BUS_WATCH_PERIOD_S, token):
        if _bus_age_s(device) > _BUS_DEAD_EXIT_S:
            runtime.log(
                f"serial bus has served no state read for {_BUS_DEAD_EXIT_S:.0f}s; "
                "exiting for supervised restart"
            )
            await asyncio.to_thread(device.stop)
            terminate(1)
            return


def _lerobot_hardware(config: params_mod.Config) -> LerobotFollower:
    return LerobotFollower(
        device_path=config.device_path,
        calibration_dir=config.calibration_dir,
        robot_id=config.robot_id,
    )


def make_setup(hardware_factory=_lerobot_hardware):
    """The node's setup, with the hardware seam injectable for harness tests."""

    async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
        config = params_mod.parse(params)
        await peppygen.clock.init(node_runner)

        hardware = hardware_factory(config)
        device = DeviceLoop(hardware, control_rate_hz=config.control_rate_hz)
        # Raises on connect, calibration, or first-read failure, failing the
        # launch loudly.
        device.start()
        runtime.log(f"connected and calibrated ({config.robot_id} on {config.device_path})")

        async def release_arm():
            # Off the event loop: joining the device thread must not stall
            # the rest of shutdown inside the daemon's grace window.
            await asyncio.to_thread(device.stop)

        node_runner.on_shutdown(release_arm)

        alert_publisher = await alerts_topic.declare_publisher(node_runner)
        alert_bus = _AlertBus(alert_publisher)

        def submit_joints(message):
            positions_rad = parse_joint_setpoints(
                message.positions, message.velocities, message.efforts
            )
            device.submit_arm_target(units.joints_deg_from_rad(positions_rad))

        def submit_gripper(message):
            opening = parse_gripper_setpoint(message.opening, message.max_effort)
            device.submit_gripper_target(units.gripper_percent_from_fraction(opening))

        def build_joint_states(snapshot: StateSnapshot):
            positions = list(units.joints_rad_from_deg(snapshot.positions_deg))
            stamp = _captured_timestamp_s(snapshot.captured_monotonic)
            return joint_states.build_message(stamp, positions, [], [])

        def build_gripper_states(snapshot: StateSnapshot):
            opening = units.gripper_fraction_from_percent(snapshot.gripper_percent)
            # The gripper's load reading has no torque constant to convert it
            # to the wire's opening-frame effort, and the firmware ceiling is
            # not per-command, so effort and max_effort ride 0 per the contract.
            stamp = _captured_timestamp_s(snapshot.captured_monotonic)
            return gripper_states.build_message(stamp, opening, 0.0, 0.0)

        token = node_runner.cancellation_token()

        async def ready_handler(_request):
            # Readiness is live, not a bringup latch: a bus that stopped
            # serving reads is not ready however healthy the process looks,
            # and a node mid-shutdown is already releasing torque.
            return is_ready_svc.Response(
                ready=device.ready()
                and _bus_age_s(device) <= _BUS_SILENT_READY_S
                and not token.is_cancelled()
            )

        state_period_s = 1.0 / config.state_rate_hz
        return [
            asyncio.create_task(
                _stream_states(
                    node_runner, joint_states, build_joint_states, device,
                    state_period_s, token,
                )
            ),
            asyncio.create_task(
                _stream_states(
                    node_runner, gripper_states, build_gripper_states, device,
                    state_period_s, token,
                )
            ),
            asyncio.create_task(
                _consume_setpoints(
                    node_runner, joint_setpoints, "joint_setpoints",
                    submit_joints, token,
                )
            ),
            asyncio.create_task(
                _consume_setpoints(
                    node_runner, gripper_setpoints, "gripper_setpoints",
                    submit_gripper, token,
                )
            ),
            asyncio.create_task(
                _stream_health(node_runner, device, config, alert_bus, token)
            ),
            asyncio.create_task(
                runtime.serve(node_runner, is_ready_svc, ready_handler, "is_ready")
            ),
            asyncio.create_task(_watch_bus(device, token)),
        ]

    return setup


setup = make_setup()


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
