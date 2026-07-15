"""g1_node entry point: bridges peppy interfaces to the G1 control backend.

setup() spawns the long-lived tasks and returns them so the runtime owns them
(setup must return promptly, or peppylib never registers the health probe and
the daemon kills the instance): the set_posture action, the base velocity
subscriber with a deadman, one loop per exposed service, and the telemetry
publisher. The Unitree SDK is synchronous and not thread-safe, so every backend
call funnels through one single-worker executor; async paths await it, the
sync service handlers submit-and-wait. That one worker is the sole SDK writer.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics import commands_base_velocity_commands as base_velocity
from peppygen.emitted_topics.g1_imu.v1 import g1_imu
from peppygen.emitted_topics.g1_joint_states.v1 import g1_joint_states
from peppygen.emitted_topics.g1_states.v1 import g1_state
from peppygen.exposed_actions import set_posture
from peppygen.exposed_services import (
    balance_stand,
    check_mode,
    get_fsm_id,
    move_timed,
    release_mode,
    select_mode,
    set_balance_mode,
    set_fsm_id,
    set_speed_mode,
    set_stand_height,
    set_task_id,
    stop_move,
    switch_to_internal_ctrl,
    switch_to_user_ctrl,
)
from peppygen.parameters import Parameters

from .backend import G1Backend, LocoClientBackend
from .postures import Posture, parse_posture, plan_transition

# Upper bound on any single blocking SDK call issued from a service handler, so a
# wedged DDS round-trip cannot hang the event loop indefinitely.
SDK_CALL_TIMEOUT = 12.0


class G1Controller:
    """Serializes all SDK access through one worker and tracks FSM state.

    Async callers await `_call`; sync service handlers use `call_sync`. Both
    submit to the same single-worker executor, so the SDK sees one writer and no
    asyncio lock is needed. `call_sync` briefly parks the event loop while its
    call runs, which is fine for the infrequent config/query services.
    """

    def __init__(self, backend: G1Backend) -> None:
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g1-sdk")
        self._current: Posture | None = None

    @property
    def backend(self) -> G1Backend:
        return self._backend

    @property
    def current(self) -> Posture | None:
        return self._current

    async def _call(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def call_sync(self, fn, *args):
        return self._executor.submit(fn, *args).result(timeout=SDK_CALL_TIMEOUT)

    async def apply_posture(self, posture: Posture) -> int:
        fsm_id = await self._call(self._backend.transition, posture)
        self._current = posture
        return fsm_id

    async def apply_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        await self._call(self._backend.set_velocity, vx, vy, vyaw)

    async def stop_base(self) -> None:
        await self._call(self._backend.set_velocity, 0.0, 0.0, 0.0)

    async def read_telemetry(self):
        return await self._call(self._backend.read_telemetry)

    async def refresh_fsm(self) -> None:
        await self._call(self._backend.get_fsm_id)

    def close(self) -> None:
        self._executor.shutdown(wait=False)


# --- set_posture action -------------------------------------------------------


async def _run_posture_action(node_runner: NodeRunner, controller: G1Controller, token):
    action = await set_posture.ActionHandle.expose(node_runner)
    busy = [False]
    drive_tasks: set[asyncio.Task] = set()

    def decide(request):
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
    target = parse_posture(ctx.request().data.posture)  # validated in decide()
    try:
        fsm_id = await controller.apply_posture(target)
        await ctx.complete(fsm_id)
    except Exception as error:
        print(f"[g1] set_posture {target.value} failed: {error!r}")
    finally:
        busy[0] = False


# --- base velocity subscriber (with deadman) ----------------------------------


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
            break
        _producer, msg = pair
        await controller.apply_velocity(msg.vx, msg.vy, msg.vyaw)
        moving = msg.vx != 0.0 or msg.vy != 0.0 or msg.vyaw != 0.0


# --- services -----------------------------------------------------------------


def _build_service_handlers(controller: G1Controller):
    """(service module, sync handler) for every exposed loco + mode-switch call.

    Each handler issues one SDK call through the controller's single writer and
    returns the generated Response. `ok=True` means the call was issued without
    raising; an SDK exception propagates and the serve loop logs it.
    """
    b = controller.backend

    def issued(service, fn, *args):
        controller.call_sync(fn, *args)
        return service.Response(ok=True)

    return [
        (stop_move, lambda _r: issued(stop_move, b.stop_move)),
        (move_timed, lambda r: issued(move_timed, b.move_timed, r.data.vx, r.data.vy, r.data.omega, r.data.duration)),
        (balance_stand, lambda r: issued(balance_stand, b.balance_stand, r.data.balance_mode)),
        (set_balance_mode, lambda r: issued(set_balance_mode, b.set_balance_mode, r.data.balance_mode)),
        (set_stand_height, lambda r: issued(set_stand_height, b.set_stand_height, r.data.stand_height)),
        (set_speed_mode, lambda r: issued(set_speed_mode, b.set_speed_mode, r.data.speed_mode)),
        (set_fsm_id, lambda r: issued(set_fsm_id, b.set_fsm_id, r.data.fsm_id)),
        (set_task_id, lambda r: issued(set_task_id, b.set_task_id, r.data.task_id)),
        (switch_to_user_ctrl, lambda _r: issued(switch_to_user_ctrl, b.switch_to_user_ctrl)),
        (switch_to_internal_ctrl, lambda r: issued(switch_to_internal_ctrl, b.switch_to_internal_ctrl, r.data.mode)),
        (select_mode, lambda r: issued(select_mode, b.select_mode, r.data.name)),
        (release_mode, lambda _r: issued(release_mode, b.release_mode)),
        (get_fsm_id, lambda _r: get_fsm_id.Response(fsm_id=controller.call_sync(b.get_fsm_id))),
        (check_mode, lambda _r: check_mode.Response(mode=controller.call_sync(b.check_mode))),
    ]


async def _serve(node_runner: NodeRunner, service, handler, name: str, token):
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            req = asyncio.ensure_future(service.handle_next_request(node_runner, handler))
            await asyncio.wait([cancelled, req], return_when=asyncio.FIRST_COMPLETED)
            if not req.done():
                req.cancel()
                break
            try:
                req.result()
            except Exception as exc:
                print(f"[g1] service {name} error: {exc!r}")
    finally:
        cancelled.cancel()


# --- telemetry ----------------------------------------------------------------


async def _run_telemetry(
    node_runner: NodeRunner, controller: G1Controller, state_hz: float, fsm_hz: float, token
):
    state_pub = await g1_state.declare_publisher(node_runner)
    imu_pub = await g1_imu.declare_publisher(node_runner)
    joints_pub = await g1_joint_states.declare_publisher(node_runner)
    period = 1.0 / state_hz
    fsm_every = max(1, round(state_hz / fsm_hz))
    tick = 0
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            if tick % fsm_every == 0:
                await controller.refresh_fsm()
            t = await controller.read_telemetry()
            try:
                await state_pub.publish(
                    g1_state.build_message(t.fsm_id, t.mode_pr, t.mode_machine, t.tick)
                )
                await imu_pub.publish(
                    g1_imu.build_message(
                        t.imu.quaternion, t.imu.gyroscope, t.imu.accelerometer,
                        t.imu.rpy, t.imu.temperature,
                    )
                )
                await joints_pub.publish(
                    g1_joint_states.build_message(
                        t.joints.q, t.joints.dq, t.joints.tau_est,
                        t.joints.temperature, t.joints.voltage,
                        bytes(t.joints.motor_mode),  # u8 array is generated as bytes
                    )
                )
            except Exception as exc:
                print(f"[g1] telemetry publish error: {exc!r}")
                break
            tick += 1
            await asyncio.wait([cancelled], timeout=period)
    finally:
        cancelled.cancel()


# --- setup --------------------------------------------------------------------


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    backend = LocoClientBackend(params.network_interface, params.dds_domain_id)
    controller = G1Controller(backend)
    token = node_runner.cancellation_token()

    async def safe_shutdown():
        try:
            await controller._call(backend.shutdown)
            print("[g1] robot damped on shutdown")
        except Exception as exc:
            print(f"[g1] shutdown damp error: {exc!r}")
        controller.close()

    node_runner.on_shutdown(safe_shutdown)

    tasks = [
        asyncio.create_task(_run_posture_action(node_runner, controller, token)),
        asyncio.create_task(
            _run_velocity_subscriber(
                node_runner, controller, params.velocity_timeout_ms / 1000.0, token
            )
        ),
        asyncio.create_task(
            _run_telemetry(
                node_runner, controller, float(params.state_rate_hz),
                float(params.fsm_poll_hz), token,
            )
        ),
    ]
    for service, handler in _build_service_handlers(controller):
        name = service.__name__.rsplit(".", 1)[-1]
        tasks.append(asyncio.create_task(_serve(node_runner, service, handler, name, token)))
    return tasks


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
