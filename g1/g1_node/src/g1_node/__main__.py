"""g1_node entry point: bridges peppy interfaces to the G1 control backend.

Three long-lived tasks, spawned by setup() and returned so the runtime owns
them (setup must return promptly, or peppylib never registers the health probe
and the daemon kills the instance): a set_posture action handler, a base
velocity subscriber with a deadman, and a g1_state telemetry publisher. The
Unitree SDK is synchronous, so every backend call runs in the default executor
and is serialized behind one lock, keeping the event loop responsive and the
robot single-writer.
"""

from __future__ import annotations

import asyncio

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics import commands_base_velocity_commands as base_velocity
from peppygen.emitted_topics.g1_states.v1 import g1_state
from peppygen.exposed_actions import set_posture
from peppygen.parameters import Parameters

from .backend import G1Backend, LocoClientBackend
from .postures import Posture, parse_posture, plan_transition


class G1Controller:
    """Serializes access to the backend and tracks the FSM state for the guard."""

    def __init__(self, backend: G1Backend) -> None:
        self._backend = backend
        self._lock = asyncio.Lock()
        self._current: Posture | None = None
        self._moving = False

    @property
    def current(self) -> Posture | None:
        return self._current

    async def apply_posture(self, posture: Posture) -> int:
        loop = asyncio.get_running_loop()
        async with self._lock:
            fsm_id = await loop.run_in_executor(None, self._backend.transition, posture)
            self._current = posture
            self._moving = posture is Posture.START
        return fsm_id

    async def apply_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self._backend.set_velocity, vx, vy, vyaw)

    async def read_state(self) -> tuple[int, int, int]:
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, self._backend.read_state)

    async def stop_base(self) -> None:
        """Zero the base velocity (deadman)."""
        await self.apply_velocity(0.0, 0.0, 0.0)


async def _run_posture_action(node_runner: NodeRunner, controller: G1Controller, token):
    action = await set_posture.ActionHandle.expose(node_runner)
    busy = [False]
    drive_tasks: set[asyncio.Task] = set()

    def decide(request):
        # Parse and order-check at the boundary; only a valid, in-order
        # transition on an idle robot is accepted. Downstream works with the enum.
        if busy[0]:
            return set_posture.GoalResponse.reject("a posture transition is in flight")
        try:
            plan_transition(controller.current, request.data.posture)
        except ValueError as exc:
            return set_posture.GoalResponse.reject(str(exc))
        busy[0] = True
        return set_posture.GoalResponse.accept()

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
            task = asyncio.create_task(_drive_posture(ctx, controller, busy))
            drive_tasks.add(task)
            task.add_done_callback(drive_tasks.discard)
    finally:
        cancelled.cancel()


async def _drive_posture(ctx, controller: G1Controller, busy: list[bool]) -> None:
    # The posture string was already validated in decide(); re-parse is total.
    target = parse_posture(ctx.request().data.posture)
    try:
        fsm_id = await controller.apply_posture(target)
        await ctx.complete(fsm_id)
    except Exception as error:
        print(f"[g1] set_posture {target.value} failed: {error!r}")
    finally:
        busy[0] = False


async def _run_velocity_subscriber(
    node_runner: NodeRunner, controller: G1Controller, timeout_s: float, token
):
    subscription = await base_velocity.subscribe(node_runner)
    moving = False
    while not token.is_cancelled():
        receive = asyncio.ensure_future(subscription.next())
        cancelled = asyncio.ensure_future(token.cancelled())
        done, _ = await asyncio.wait(
            {receive, cancelled}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        cancelled.cancel()
        if cancelled in done:
            receive.cancel()
            break
        if not receive.done():
            # Deadman: the stream stalled. Zero the base once so a dropped
            # commander cannot leave the robot walking, then keep waiting.
            receive.cancel()
            if moving:
                await controller.stop_base()
                moving = False
            continue
        try:
            pair = receive.result()
        except Exception as exc:
            print(f"[g1] base velocity consume error: {exc!r}")
            continue
        if pair is None:
            break  # subscription closed
        _producer, msg = pair
        await controller.apply_velocity(msg.vx, msg.vy, msg.vyaw)
        moving = msg.vx != 0.0 or msg.vy != 0.0 or msg.vyaw != 0.0


async def _run_state_publisher(
    node_runner: NodeRunner, controller: G1Controller, rate_hz: float, token
):
    publisher = await g1_state.declare_publisher(node_runner)
    period = 1.0 / rate_hz
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            fsm_id, mode, battery_soc = await controller.read_state()
            try:
                await publisher.publish(g1_state.build_message(fsm_id, mode, battery_soc))
            except Exception as exc:
                print(f"[g1] g1_state publish error: {exc!r}")
                break
            await asyncio.wait([cancelled], timeout=period)
    finally:
        cancelled.cancel()


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    backend = LocoClientBackend(params.network_interface, params.dds_domain_id)
    controller = G1Controller(backend)
    token = node_runner.cancellation_token()

    async def safe_shutdown():
        # Damp the robot on the node loop before the runtime tears tasks down.
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, backend.shutdown)
            print("[g1] robot damped on shutdown")
        except Exception as exc:
            print(f"[g1] shutdown damp error: {exc!r}")

    node_runner.on_shutdown(safe_shutdown)

    return [
        asyncio.create_task(_run_posture_action(node_runner, controller, token)),
        asyncio.create_task(
            _run_velocity_subscriber(
                node_runner, controller, params.velocity_timeout_ms / 1000.0, token
            )
        ),
        asyncio.create_task(
            _run_state_publisher(node_runner, controller, float(params.state_rate_hz), token)
        ),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
