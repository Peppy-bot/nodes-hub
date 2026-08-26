"""Wiring: bring up the reader thread and run one publish task per pairing
slot. Silence is the deadman: a stale sample publishes nothing and every
downstream consumer's hold behavior takes over."""

from __future__ import annotations

import asyncio

import peppygen.clock
from control_core_py import runtime
from peppygen import NodeBuilder, NodeRunner
from peppygen.paired_topics.arm import joint_setpoints
from peppygen.paired_topics.gripper import gripper_setpoints
from peppygen.parameters import Parameters
from so101_description import units

from so101_leader import params as params_mod
from so101_leader.device import DeviceLoop, LeaderSample, LerobotLeader

runtime.configure("so101_leader")


def _sample_timestamp_s(sample: LeaderSample) -> float:
    return runtime.wire_timestamp_s(peppygen.clock.now_ns(), sample.captured_monotonic)


async def _stream(node_runner, topic_module, period_s, token, build):
    """Publish build(sample) every tick a fresh sample exists. The whole
    tick body is guarded: a raising build (e.g. a sim clock before its first
    tick) must log and retry, never kill the stream task."""
    publisher = await topic_module.declare_publisher(node_runner)
    failing = runtime.Latch()
    rate = runtime.RateMeter(f"{topic_module.TOPIC_NAME} publish loop", 1.0 / period_s)
    async for _ in runtime.ticks(period_s, token):
        rate.tick()
        try:
            payload = build()
            if payload is None:
                continue
            await publisher.publish(payload)
            failing.clear()
        except Exception as e:
            failing.trip(f"{topic_module.TOPIC_NAME} publish failing: {e!r}")


def _lerobot_hardware(config: params_mod.Config) -> LerobotLeader:
    return LerobotLeader(
        device_path=config.device_path,
        calibration_dir=config.calibration_dir,
        teleop_id=config.teleop_id,
    )


def make_setup(hardware_factory=_lerobot_hardware):
    """The node's setup, with the hardware seam injectable for harness tests."""

    async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
        config = params_mod.parse(params)
        await peppygen.clock.init(node_runner)

        hardware = hardware_factory(config)
        device = DeviceLoop(hardware, read_rate_hz=config.command_rate_hz)
        # Raises on connect or calibration failure, failing the launch loudly.
        device.start()
        runtime.log(f"connected and calibrated ({config.teleop_id} on {config.device_path})")

        async def release_device():
            # Off the event loop: joining the reader thread must not stall
            # the rest of shutdown inside the daemon's grace window.
            await asyncio.to_thread(device.stop)

        node_runner.on_shutdown(release_device)

        def build_arm():
            sample = device.fresh_sample(config.stale_timeout_s)
            if sample is None:
                return None
            positions = list(units.joints_rad_from_deg(sample.positions_deg))
            # Positions only: the backbone shapes its own pacing, and no
            # consumer of this stream takes velocity or effort feedforward.
            return joint_setpoints.build_message(_sample_timestamp_s(sample), positions, [], [])

        def build_gripper():
            sample = device.fresh_sample(config.stale_timeout_s)
            if sample is None:
                return None
            opening = units.gripper_fraction_from_percent(sample.gripper_percent)
            # A leader arm carries no effort intent; 0 leaves the follower's ceiling.
            return gripper_setpoints.build_message(_sample_timestamp_s(sample), opening, 0.0)

        token = node_runner.cancellation_token()
        period_s = 1.0 / config.command_rate_hz
        return [
            asyncio.create_task(
                _stream(node_runner, joint_setpoints, period_s, token, build_arm)
            ),
            asyncio.create_task(
                _stream(node_runner, gripper_setpoints, period_s, token, build_gripper)
            ),
        ]

    return setup


setup = make_setup()


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
