"""g1_node_isaac entry point: Isaac Sim G1 behind the shared G1 contract surface.

Structurally identical to g1_node_mujoco: a physics thread owns the engine
(step + snapshot); the asyncio loop consumes base velocity, serves the
set_posture action, and publishes the state / IMU / joint / odometry telemetry.
The engine is the Isaac backend; running requires the Isaac Sim runtime.
"""

from __future__ import annotations

import asyncio
import threading
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics import commands_base_velocity_commands as base_velocity
from peppygen.emitted_topics.g1_imu.v1 import g1_imu
from peppygen.emitted_topics.g1_joint_states.v1 import g1_joint_states
from peppygen.emitted_topics.g1_odometry.v1 import g1_odometry
from peppygen.emitted_topics.g1_states.v1 import g1_state
from peppygen.exposed_actions import set_posture
from peppygen.parameters import Parameters

from .engine import MOTOR_COUNT, G1IsaacEngine, Snapshot
from .postures import UnknownPosture, sim_posture


class _LatestSlot:
    def __init__(self, initial: Snapshot) -> None:
        self._lock = threading.Lock()
        self._value = initial

    def set(self, value: Snapshot) -> None:
        with self._lock:
            self._value = value

    def get(self) -> Snapshot:
        with self._lock:
            return self._value


def _physics_loop(engine: G1IsaacEngine, slot: _LatestSlot, rate_hz: float, stop: threading.Event):
    period = 1.0 / rate_hz
    while not stop.is_set():
        start = time.perf_counter()
        engine.step()
        slot.set(engine.snapshot())
        elapsed = time.perf_counter() - start
        if elapsed < period:
            stop.wait(period - elapsed)


async def _run_velocity_subscriber(node_runner: NodeRunner, engine: G1IsaacEngine, token):
    subscription = await base_velocity.subscribe(node_runner)
    while not token.is_cancelled():
        receive = asyncio.ensure_future(subscription.next())
        cancelled = asyncio.ensure_future(token.cancelled())
        done, _ = await asyncio.wait({receive, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        cancelled.cancel()
        if cancelled in done:
            receive.cancel()
            break
        pair = receive.result()
        if pair is None:
            break
        _producer, msg = pair
        engine.set_velocity(msg.vx, msg.vy, msg.vyaw)


async def _run_posture_action(node_runner: NodeRunner, engine: G1IsaacEngine, token):
    action = await set_posture.ActionHandle.expose(node_runner)
    busy = [False]
    drive_tasks: set[asyncio.Task] = set()

    def decide(request):
        if busy[0]:
            return set_posture.GoalResponse.reject("a posture transition is in flight")
        try:
            sim_posture(request.data.posture)
        except UnknownPosture as exc:
            return set_posture.GoalResponse.reject(str(exc))
        busy[0] = True
        return set_posture.GoalResponse.accept()

    async def drive(ctx):
        try:
            fsm_id, pd_enabled = sim_posture(ctx.request().data.posture)
            engine.set_posture(fsm_id, pd_enabled)
            await ctx.complete(fsm_id)
        except Exception as error:
            print(f"[g1-isaac] set_posture failed: {error!r}")
        finally:
            busy[0] = False

    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_goal = asyncio.ensure_future(action.handle_goal_next_request(decide))
            await asyncio.wait([cancelled, next_goal], return_when=asyncio.FIRST_COMPLETED)
            if not next_goal.done():
                next_goal.cancel()
                break
            ctx = next_goal.result()
            if ctx is None:
                break
            task = asyncio.create_task(drive(ctx))
            drive_tasks.add(task)
            task.add_done_callback(drive_tasks.discard)
    finally:
        cancelled.cancel()


async def _run_telemetry(node_runner: NodeRunner, slot: _LatestSlot, rate_hz: float, token):
    state_pub = await g1_state.declare_publisher(node_runner)
    imu_pub = await g1_imu.declare_publisher(node_runner)
    joints_pub = await g1_joint_states.declare_publisher(node_runner)
    odom_pub = await g1_odometry.declare_publisher(node_runner)
    period = 1.0 / rate_hz
    temps = [25] * MOTOR_COUNT
    volts = [48.0] * MOTOR_COUNT
    modes = bytes([1] * MOTOR_COUNT)
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            s = slot.get()
            try:
                await state_pub.publish(g1_state.build_message(s.fsm_id, 0, 0, s.tick))
                await imu_pub.publish(
                    g1_imu.build_message(s.quaternion, s.gyroscope, s.accelerometer, s.rpy, 25)
                )
                await joints_pub.publish(
                    g1_joint_states.build_message(s.q, s.dq, s.tau_est, temps, volts, modes)
                )
                await odom_pub.publish(
                    g1_odometry.build_message(
                        s.position, s.orientation, s.linear_velocity, s.angular_velocity
                    )
                )
            except Exception as exc:
                print(f"[g1-isaac] telemetry publish error: {exc!r}")
                break
            await asyncio.wait([cancelled], timeout=period)
    finally:
        cancelled.cancel()


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    engine = G1IsaacEngine()
    slot = _LatestSlot(engine.snapshot())
    stop = threading.Event()
    physics = threading.Thread(
        target=_physics_loop,
        args=(engine, slot, float(params.physics_rate_hz), stop),
        name="g1-isaac-physics",
        daemon=True,
    )
    physics.start()

    async def on_shutdown():
        stop.set()

    node_runner.on_shutdown(on_shutdown)

    token = node_runner.cancellation_token()
    return [
        asyncio.create_task(_run_velocity_subscriber(node_runner, engine, token)),
        asyncio.create_task(_run_posture_action(node_runner, engine, token)),
        asyncio.create_task(_run_telemetry(node_runner, slot, float(params.state_rate_hz), token)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
