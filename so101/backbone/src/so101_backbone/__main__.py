"""Wiring: parse parameters, build the kinematics, limits, and coordinator,
and run the control tick, the upstream consumers, the state relays, and one
server per exposed action."""

from __future__ import annotations

import asyncio
import math
import time

import peppygen.clock
from control_core_py import runtime
from control_core_py.runtime import CancellationToken
from peppygen import NodeBuilder, NodeRunner
from peppygen.emitted_topics.limb_state import limb_states
from peppygen.exposed_actions.limb_motion import move_arm, move_arm_joints, move_gripper
from peppygen.exposed_actions.postures import move_to_home, move_to_ready
from peppygen.paired_topics.arm_link import joint_setpoints as down_joint_setpoints
from peppygen.paired_topics.arm_link import joint_states as down_joint_states
from peppygen.paired_topics.gripper_link import (
    gripper_setpoints as down_gripper_setpoints,
)
from peppygen.paired_topics.gripper_link import gripper_states as down_gripper_states
from peppygen.paired_topics.leader_arm import joint_setpoints as up_joint_setpoints
from peppygen.paired_topics.leader_arm import joint_states as up_joint_states
from peppygen.paired_topics.leader_gripper import (
    gripper_setpoints as up_gripper_setpoints,
)
from peppygen.paired_topics.leader_gripper import gripper_states as up_gripper_states
from peppygen.paired_topics.leader_pose import pose_setpoints as up_pose_setpoints
from peppygen.paired_topics.leader_pose import pose_states as up_pose_states
from peppygen.parameters import Parameters
from so101_description import limits as limits_mod
from so101_description import postures, units
from so101_description.kinematics import Kinematics
from so101_description.limbs import ARM_LIMB, GRIPPER_LIMB
from so101_description.model import KINEMATICS_URDF_PATH
from so101_description.setpoints import parse_gripper_setpoint, parse_joint_setpoints

from so101_backbone import params as params_mod
from so101_backbone.actions import ActionLayer, PendingSlot, serve
from so101_backbone.coordinator import (
    STALE_WIRE_TIMEOUT_S,
    Coordinator,
)
from so101_backbone.params import UpstreamMode
from so101_backbone.reach import fit_reach_ball
from so101_backbone.setpoints import parse_pose_setpoint

runtime.configure("so101_backbone")

# Cadence of the FK-backed human/monitor readouts, decimated off the control
# loop; their only consumers are panels and monitors.
_READOUT_PERIOD_S = 0.05


def _now_s() -> float:
    return peppygen.clock.now_ns() / 1e9


async def _run_control(
    node_runner: NodeRunner, coordinator: Coordinator, period_s: float, token: CancellationToken
):
    arm_pub = await down_joint_setpoints.declare_publisher(node_runner)
    gripper_pub = await down_gripper_setpoints.declare_publisher(node_runner)
    arm_computing = runtime.Latch()
    gripper_computing = runtime.Latch()
    arm_failing = runtime.Latch()
    gripper_failing = runtime.Latch()
    rate = runtime.RateMeter("control loop", 1.0 / period_s)

    def stamped(stamp: float | None) -> float:
        """Streamed samples keep the leader's capture stamp so the wire never
        hides this hop's latency; plan samples are authored here and stamped
        now."""
        return stamp if stamp is not None else _now_s()

    async def publish_setpoint(publisher, failing, report, command, build, label):
        """Publish one tick's setpoint and report its fate. The fate feeds the
        plan lifecycle: a goal completes only when its final sample is
        delivered, and an undelivered mid-plan sample fails the plan it came
        from. The command travels through rather than being captured, so an
        abort landing during the publish cannot change what gets reported."""
        try:
            await publisher.publish(build(command))
        except Exception as e:
            failing.trip(f"{label} publish failing: {e!r}")
            report(False, None)
            return
        failing.clear()
        report(True, command)

    async for _ in runtime.ticks(period_s, token):
        rate.tick()
        now = time.monotonic()
        # Guarded per limb: the motion authority's loop must survive anything
        # one limb's tick computation (IK included) throws, holding that limb
        # via silence while the other keeps streaming.
        arm_target = gripper_target = None
        try:
            arm_target = coordinator.arm_tick(now)
            arm_computing.clear()
        except Exception as e:
            arm_computing.trip(f"arm tick failed: {e!r}")
        try:
            gripper_target = coordinator.gripper_tick(now)
            gripper_computing.clear()
        except Exception as e:
            gripper_computing.trip(f"gripper tick failed: {e!r}")
        if arm_target is not None:
            await publish_setpoint(
                arm_pub,
                arm_failing,
                coordinator.arm_published,
                arm_target,
                lambda command: down_joint_setpoints.build_message(
                    stamped(coordinator.arm_tick_stamp_s), list(command), [], []
                ),
                "arm setpoint",
            )
        if gripper_target is not None:
            await publish_setpoint(
                gripper_pub,
                gripper_failing,
                coordinator.gripper_published,
                gripper_target,
                lambda command: down_gripper_setpoints.build_message(
                    stamped(coordinator.gripper_tick_stamp_s), command, 0.0
                ),
                "gripper setpoint",
            )


async def _consume(
    node_runner: NodeRunner,
    topic_module,
    stale_timeout_s: float,
    token: CancellationToken,
    label: str,
    handle,
):
    """One upstream leader stream over the shared gated consume."""
    subscription = await topic_module.subscribe(node_runner)
    await runtime.consume_gated(subscription, token, label, _now_s, stale_timeout_s, handle)


async def _relay_states(
    node_runner: NodeRunner,
    *,
    source_module,
    sink_module,
    label: str,
    state_timeout_s: float,
    token: CancellationToken,
    parse,
    apply,
):
    """One measured-state relay: fold the follower's state into the
    coordinator and republish it upstream.

    Measured state is a trust boundary like any other input, so `parse`
    returns None for a message that must neither anchor motion nor be
    relayed as if the follower said something coherent. A sample stamped far
    from now, past or future, gets the same backlog-replay defense the
    leader streams get. `apply` adopts the parsed value and returns the
    message to relay.
    """
    subscription = await source_module.subscribe(node_runner)
    publisher = await sink_module.declare_publisher(node_runner)
    malformed = runtime.Latch()
    stale = runtime.Latch()
    failing = runtime.Latch()
    async for _producer, m in runtime.messages(subscription, token, label):
        parsed = parse(m)
        if parsed is None:
            malformed.trip(f"malformed {label} from the follower; dropping")
            continue
        malformed.clear()
        age_s = _now_s() - m.timestamp
        if not math.isfinite(age_s) or abs(age_s) > state_timeout_s:
            stale.trip(f"{label} stale or future-stamped on arrival; not anchoring")
            continue
        stale.clear()
        try:
            await publisher.publish(apply(m, parsed))
            failing.clear()
        except Exception as e:
            failing.trip(f"{label} relay failing: {e!r}")


def _parse_arm_states(m):
    positions = tuple(m.positions)
    if (
        len(positions) != units.NUM_JOINTS
        or len(m.velocities) not in (0, units.NUM_JOINTS)
        or len(m.efforts) not in (0, units.NUM_JOINTS)
        or not all(math.isfinite(v) for v in (*positions, *m.velocities, *m.efforts))
    ):
        return None
    return positions


def _adopt_arm_states(coordinator: Coordinator):
    def apply(m, positions):
        coordinator.measured_joints.set(positions, wire_timestamp_s=m.timestamp)
        return up_joint_states.build_message(
            m.timestamp, list(m.positions), list(m.velocities), list(m.efforts)
        )

    return apply


def _adopt_gripper_states(coordinator: Coordinator):
    def apply(m, opening):
        coordinator.measured_gripper.set(opening, wire_timestamp_s=m.timestamp)
        return up_gripper_states.build_message(m.timestamp, opening, m.effort, m.max_effort)

    return apply


def _parse_gripper_states(m):
    if not all(math.isfinite(v) for v in (m.opening, m.effort, m.max_effort)):
        return None
    # One truth: the contract-range opening anchors motion and relays.
    return min(max(float(m.opening), 0.0), 1.0)


async def _publish_readouts(
    node_runner: NodeRunner,
    coordinator: Coordinator,
    kinematics: Kinematics,
    pose_mode: bool,
    state_timeout_s: float,
    token: CancellationToken,
):
    """The FK-backed readouts, one solve per snapshot: limb_states (the
    pairing-free whole-robot readout; any stale limb skips the tick, a
    partial robot is not a snapshot) and, in pose mode, the pose_states
    relay on the leader slot. Stamps are the measurement's capture time."""
    limb_publisher = await limb_states.declare_publisher(node_runner)
    pose_publisher = await up_pose_states.declare_publisher(node_runner) if pose_mode else None
    failing = runtime.Latch()
    async for _ in runtime.ticks(_READOUT_PERIOD_S, token):
        measured = coordinator.measured_joints.fresh(state_timeout_s)
        if measured is None:
            continue
        timestamp = coordinator.measured_joints.wire_timestamp_s
        assert timestamp is not None, (
            "the relay sets the capture stamp with the value it belongs to"
        )
        try:
            position, orientation = kinematics.forward_kinematics(measured)
            if pose_publisher is not None:
                await pose_publisher.publish(
                    up_pose_states.build_message(timestamp, list(position), list(orientation))
                )
            opening = coordinator.measured_gripper.fresh(state_timeout_s)
            if opening is not None:
                await limb_publisher.publish(
                    limb_states.build_message(
                        timestamp,
                        [ARM_LIMB],
                        [units.NUM_JOINTS],
                        list(measured),
                        list(position),
                        list(orientation),
                        [GRIPPER_LIMB],
                        [opening],
                    )
                )
            failing.clear()
        except Exception as e:
            failing.trip(f"readout publish failing: {e!r}")


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    config = params_mod.parse(params)
    await peppygen.clock.init(node_runner)

    # The postures and limits below are validated against the same model
    # bytes the constants were verified on. The point-to-point solver is a
    # second instance: move_arm solves run off the event loop and must not
    # share the control tick's solver state.
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    point_solver = Kinematics(KINEMATICS_URDF_PATH)
    limits = limits_mod.from_urdf(KINEMATICS_URDF_PATH)
    for name, posture in (
        ("HOME_POSITIONS_RAD", postures.HOME_POSITIONS_RAD),
        ("READY_POSITIONS_RAD", postures.READY_POSITIONS_RAD),
    ):
        if not limits.contains(posture):
            raise ValueError(f"{name} exceeds a joint position limit of the deployed URDF")

    reach = fit_reach_ball(kinematics, limits)
    runtime.log(
        f"reach ball center=({reach.center[0]:.3f}, {reach.center[1]:.3f}, "
        f"{reach.center[2]:.3f}) radius={reach.radius:.3f}"
    )
    coordinator = Coordinator(config, kinematics, limits, reach)
    layer = ActionLayer(coordinator, config, limits)
    runtime.log(f"upstream_mode={config.upstream_mode.value}")

    token = node_runner.cancellation_token()
    period_s = 1.0 / config.control_rate_hz
    effort_unsupported = runtime.Latch()

    def on_joint_setpoints(m):
        coordinator.leader_joints.set(
            parse_joint_setpoints(m.positions, m.velocities, m.efforts),
            wire_timestamp_s=m.timestamp,
        )

    def on_pose_setpoints(m):
        coordinator.leader_pose.set(
            parse_pose_setpoint(m.position, m.orientation), wire_timestamp_s=m.timestamp
        )

    def on_gripper_setpoints(m):
        opening = parse_gripper_setpoint(m.opening, m.max_effort)
        if m.max_effort > 0.0:
            # Legal leader intent this hardware cannot honor; the opening
            # still streams, and the follower's states already advertise
            # max_effort 0 (no effort control).
            effort_unsupported.trip(
                "leader max_effort ignored: gripper ceiling is fixed in firmware"
            )
        else:
            effort_unsupported.clear()
        coordinator.leader_gripper.set(opening, wire_timestamp_s=m.timestamp)

    tasks = [
        asyncio.create_task(_run_control(node_runner, coordinator, period_s, token)),
        asyncio.create_task(
            _relay_states(
                node_runner,
                source_module=down_joint_states,
                sink_module=up_joint_states,
                label="arm joint_states",
                state_timeout_s=STALE_WIRE_TIMEOUT_S,
                token=token,
                parse=_parse_arm_states,
                apply=_adopt_arm_states(coordinator),
            )
        ),
        asyncio.create_task(
            _relay_states(
                node_runner,
                source_module=down_gripper_states,
                sink_module=up_gripper_states,
                label="gripper_states",
                state_timeout_s=STALE_WIRE_TIMEOUT_S,
                token=token,
                parse=_parse_gripper_states,
                apply=_adopt_gripper_states(coordinator),
            )
        ),
        asyncio.create_task(
            _consume(
                node_runner, up_gripper_setpoints, STALE_WIRE_TIMEOUT_S, token,
                "leader gripper_setpoints", on_gripper_setpoints,
            )
        ),
    ]
    # Only the mode's slot kind is subscribed: a leader linked to the other
    # kind streams into a slot nothing reads.
    if config.upstream_mode is UpstreamMode.JOINTS:
        tasks.append(
            asyncio.create_task(
                _consume(
                    node_runner, up_joint_setpoints, STALE_WIRE_TIMEOUT_S, token,
                    "leader joint_setpoints", on_joint_setpoints,
                )
            )
        )
    else:
        tasks.append(
            asyncio.create_task(
                _consume(
                    node_runner, up_pose_setpoints, STALE_WIRE_TIMEOUT_S, token,
                    "leader pose_setpoints", on_pose_setpoints,
                )
            )
        )

    def drive_with(driver, pending):
        def spawn(ctx):
            return driver(ctx, pending.take())

        return spawn

    def server(module, make_decide, driver, label):
        # One pending slot per server: a failed delivery on one action can
        # only ever walk back its own admission.
        pending = PendingSlot()
        return (module, make_decide(pending), drive_with(driver, pending), pending, label)

    action_servers = [
        server(move_arm_joints,
               lambda p: layer.decide_arm_joints(move_arm_joints, p),
               layer.drive_arm, "move_arm_joints"),
        server(move_to_ready,
               lambda p: layer.decide_posture(move_to_ready, postures.READY_POSITIONS_RAD, p),
               layer.drive_posture, "move_to_ready"),
        server(move_to_home,
               lambda p: layer.decide_posture(move_to_home, postures.HOME_POSITIONS_RAD, p),
               layer.drive_posture, "move_to_home"),
        server(move_arm,
               lambda p: layer.decide_pose(move_arm, p),
               lambda ctx, goal: layer.drive_pose(ctx, goal, point_solver, kinematics),
               "move_arm"),
        server(move_gripper,
               lambda p: layer.decide_gripper(move_gripper, p),
               layer.drive_gripper, "move_gripper"),
    ]
    tasks.extend(
        asyncio.create_task(serve(node_runner, module, token, decide, drive, layer, pending, label))
        for module, decide, drive, pending, label in action_servers
    )
    tasks.append(
        asyncio.create_task(
            _publish_readouts(
                node_runner, coordinator, kinematics,
                config.upstream_mode is UpstreamMode.POSE,
                coordinator.follower_state_timeout_s, token,
            )
        )
    )
    return tasks


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
