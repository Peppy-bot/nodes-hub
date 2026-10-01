#!/usr/bin/env python3
"""Web commander for a simulation scene.

The panel edits the scene through scene_manipulation and reads object state
through object_state, both of the same simulation. A launch may bind beside
them the scene's lighting and materials and the cameras looking at it, each
camera with its profile: the panel shows a card for every capability that
is bound and asks nothing of the ones left vacant.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Awaitable, Callable

import peppylib
from aiohttp import web

from peppygen import NodeBuilder, NodeRunner

from peppygen.consumed_actions.simulation import (
    apply_force,
    clear_scene,
    load_scene,
    move_object,
    move_robot,
    remove_object,
    spawn_object,
)

from peppygen.consumed_services.cameras import (
    describe_camera,
    get_cameras,
    reset_camera,
    set_camera_brightness,
    set_camera_contrast,
    set_camera_exposure,
    set_camera_gain,
    set_camera_white_balance,
)

from peppygen.consumed_services.lighting import (
    get_lighting,
    reset_lighting,
    set_light_color,
    set_light_cone,
    set_light_direction,
    set_light_intensity,
    set_light_orientation,
    set_light_position,
)

from peppygen.consumed_services.materials import (
    get_materials,
    reset_materials,
    set_material_color,
    set_material_finish,
)

from peppygen.consumed_services.objects import get_object_states

from peppygen.consumed_services.simulation import get_assets_list, get_robots_list

from scene_commander import listen

if TYPE_CHECKING:
    from peppygen.parameters import Parameters


logger = logging.getLogger(__name__)

# Every request the provider answers at once: a service read or setter, a
# cancel, and the result of a goal whose feedback stream has ended.
SERVICE_TIMEOUT_S = 10.0
# The admission of every goal (a provider may admit one goal at a time, so
# a goal can wait for the one before it), and the whole of a goal that
# reports no progress.
ACTION_TIMEOUT_S = 60.0
# The longest a load or a spawn may go without a progress message; each
# message starts the window again.
PROGRESS_TIMEOUT_S = 60.0


class SceneCatalogueUnavailable(RuntimeError):
    """The scene provider answered get_assets_list without a catalogue.

    An engine answers this way while it is still discovering its assets, and
    its message says so. The page waits it out and asks again.
    """


class ObjectStateUnavailable(RuntimeError):
    """The object state provider answered get_object_states without a snapshot.

    An engine answers this way when it does not inspect its scene or is not
    ready yet, and its message says why. An empty scene is a snapshot with
    no objects, never this.
    """


class StateUnavailable(RuntimeError):
    """A provider answered a read without the state it was asked for.

    Lighting and materials are absent until a scene is loaded, a camera's
    profile until the camera is attached to its device or simulation; the
    provider's message says why, and the page asks again.
    """


class CapabilityUnbound(RuntimeError):
    """The launch left vacant the capability a request needs.

    The page never asks for a capability /api/capabilities reports absent;
    a caller that does is told which one, and no provider is called.
    """


class ProviderRefusal(RuntimeError):
    """A provider refused a setter and left its target as it was.

    Its message says why; current holds the effective values it reported
    with the refusal, which the answer carries so a caller sees what stands.
    """

    def __init__(self, message: str, current: dict) -> None:
        super().__init__(message)
        self.current = current


class Canceller(StrEnum):
    """Who had a goal cancelled: this commander, when the goal made no
    progress for a window or did not end in its whole bound; the page, by
    the cancel route or by closing its stream; or the simulator itself, for
    the reason its message gives, as when a newer load replaces one that is
    still fetching its files."""

    COMMANDER = "commander"
    PAGE = "page"
    SIMULATOR = "simulator"


class GoalCancelled(RuntimeError):
    """A goal that ended cancelled; `by` names who had it cancelled."""

    def __init__(self, message: str, by: Canceller) -> None:
        super().__init__(message)
        self.by = by


class CancelNotDelivered(RuntimeError):
    """A cancel that did not reach the provider: nothing records it, and the
    next cancel of the goal is sent again."""

    def __init__(self, name: str, reason: Exception) -> None:
        super().__init__(f"the cancel of {name} was not delivered: {reason}")
        # Why the cancel did not reach the provider.
        self.reason = reason
        # What became of the cancel, for a message that names the goal.
        self.outcome = f"the cancel was not delivered: {reason}"


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


async def _fetch_assets(node_runner: NodeRunner) -> list[dict]:
    producer = get_assets_list.bound_producer(node_runner)

    # The whole catalogue: the panel filters it itself.
    response = await get_assets_list.poll(
        node_runner,
        producer,
        get_assets_list.Request(kind=None, category=None, query=None),
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise SceneCatalogueUnavailable(data.message)

    return json.loads(data.assets_json)


async def _fetch_objects(node_runner: NodeRunner) -> dict:
    """The provider's object state snapshot: its capture time and records."""

    producer = get_object_states.bound_producer(node_runner)

    response = await get_object_states.poll(
        node_runner,
        producer,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise ObjectStateUnavailable(data.message)

    return {
        "timestamp": data.timestamp,
        "objects": [_object_record(record) for record in data.objects],
    }


async def _fetch_robots(node_runner: NodeRunner) -> list[dict]:
    producer = get_robots_list.bound_producer(node_runner)

    response = await get_robots_list.poll(
        node_runner,
        producer,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise RuntimeError(data.message)

    return [
        {
            "robot": robot.robot,
            "model": robot.model,
            "position": list(robot.position),
            "yaw": robot.yaw,
            "attached": robot.attached,
        }
        for robot in data.robots
    ]


def _object_record(record) -> dict:
    return {
        "object_id": record.object_id,
        "asset_id": record.asset_id,
        "physics": record.physics,
        "mass": record.mass,
        "scale": record.scale,
        "position": list(record.position),
        "orientation": list(record.orientation),
        "linear_velocity": list(record.linear_velocity),
        "angular_velocity": list(record.angular_velocity),
    }


class _StateWatch:
    """Log whether a provider has one of its states when that changes, not
    on every read.

    provider names who answers, name the state (its catalogue, its object
    state, its lighting), unit what its count counts.
    """

    def __init__(self, name: str, unit: str, provider: str = "Scene provider") -> None:
        self._provider = provider
        self._name = name
        self._unit = unit
        self._state: tuple | None = None

    def unavailable(self, message: str) -> None:
        if self._state != ("unavailable", message):
            self._state = ("unavailable", message)
            logger.info("%s has no %s: %s", self._provider, self._name, message)

    def ready(self, count: int) -> None:
        if self._state != ("ready",):
            self._state = ("ready",)
            logger.info("%s %s ready: %d %s", self._provider, self._name, count, self._unit)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Capabilities:
    """What the launch bound beside the scene, fixed when the node starts.

    A vacant slot is a capability the page never asks for. The cameras are
    the simulation's: it lists the ones it renders, each named by its robot
    and camera.
    """

    lighting: peppylib.ProducerRef | None
    materials: peppylib.ProducerRef | None
    cameras: peppylib.ProducerRef | None

    def summary(self) -> str:
        parts = [
            name
            for name, producer in (
                ("lighting", self.lighting),
                ("materials", self.materials),
                ("cameras", self.cameras),
            )
            if producer is not None
        ]

        return ", ".join(parts) or "scene manipulation only"


def _probe_capabilities(node_runner: NodeRunner) -> _Capabilities:
    """Read the optional slots as the launch bound them, calling no provider."""

    return _Capabilities(
        lighting=get_lighting.bound_producer(node_runner),
        materials=get_materials.bound_producer(node_runner),
        cameras=get_cameras.bound_producer(node_runner),
    )


# ---------------------------------------------------------------------------
# Optional providers
# ---------------------------------------------------------------------------


def _endpoint_name(module: ModuleType) -> str:
    """The name of a consumed service or action: the last part of the name
    of its generated module."""

    return module.__name__.rsplit(".", 1)[-1]


def _summary(name: str, fields: dict) -> str:
    """What a call asked, as its log lines open with it."""

    return f"{name}({', '.join(f'{key}={value}' for key, value in fields.items())})"


async def _call_service(service, node_runner: NodeRunner, producer, request=None, **context):
    """Poll one service on producer and return its response data.

    Every call that succeeds leaves one log line naming what was asked and
    what the provider answered. A refusal raises with the provider's reason
    and the effective values it reported; the HTTP layer logs it and answers
    with both. context names what the request itself does not (the camera
    a control is set on).
    """

    fields = {**context, **(vars(request) if request is not None else {})}
    summary = _summary(_endpoint_name(service), fields)
    arguments = (request,) if request is not None else ()

    response = await service.poll(
        node_runner,
        producer,
        *arguments,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise ProviderRefusal(data.message, _current(data))

    logger.info("%s: %s", summary, data.message)

    return data


def _current(data) -> dict:
    """The effective values a setter reports: its current_* fields."""

    return {key: value for key, value in vars(data).items() if key.startswith("current_")}


def _result(data) -> dict:
    return {"success": True, "message": data.message, **_current(data)}


async def _fetch_json(service, node_runner: NodeRunner, producer, field: str) -> tuple[dict, str]:
    """A provider's state as the JSON it carries in `field` (a scene's
    lighting, its materials, a camera's profile), and the message it came
    with."""

    response = await service.poll(node_runner, producer, timeout=SERVICE_TIMEOUT_S)
    data = response.data

    if not data.success:
        raise StateUnavailable(data.message)

    return json.loads(getattr(data, field)), data.message


# ---------------------------------------------------------------------------
# Camera controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CameraControl:
    """One camera control: its service on the simulation, and the fields
    the page's payload gives its request beside the camera's name."""

    service: ModuleType
    fields: Callable[[dict], dict]


def _exposure(payload: dict) -> dict:
    return {"mode": _mode(payload), "value": _whole(payload, "value")}


def _white_balance(payload: dict) -> dict:
    return {"mode": _mode(payload), "temperature": _whole(payload, "temperature")}


def _level(payload: dict) -> dict:
    return {"value": _whole(payload, "value")}


# The camera controls by the route the page posts to.
_CAMERA_CONTROLS = {
    "exposure": _CameraControl(set_camera_exposure, _exposure),
    "white_balance": _CameraControl(set_camera_white_balance, _white_balance),
    "gain": _CameraControl(set_camera_gain, _level),
    "brightness": _CameraControl(set_camera_brightness, _level),
    "contrast": _CameraControl(set_camera_contrast, _level),
}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Progress:
    """One progress message of a load or a spawn.

    bytes_fetched counts the bytes of the asset files received so far,
    files_ready the files found in the simulator's cache or received whole,
    and building is true once every file is ready and the simulator builds.
    """

    bytes_fetched: int
    files_ready: int
    building: bool

    @classmethod
    def parse(cls, message) -> _Progress:
        return cls(
            bytes_fetched=message.bytes_fetched,
            files_ready=message.files_ready,
            building=message.building,
        )


class _Goal:
    """One goal the provider admitted: its action and name, the handle on
    it, and what was asked (the summary its log lines open with).

    A cancel counts once the provider answered it: cancel_answer holds that
    answer, a CancelState of the action, and cancelled_by who in this
    commander sent it, when the provider signalled it to the goal.
    found_over is set once a delivered cancel found the goal over: ended
    already, or unknown to the provider.
    """

    def __init__(self, action, name: str, handle, summary: str) -> None:
        self.action = action
        self.name = name
        self.handle = handle
        self.summary = summary
        self.cancel_answer = None
        self.cancelled_by: Canceller | None = None
        self.found_over = asyncio.Event()
        self._cancelling = asyncio.Lock()

    async def cancel(self, by: Canceller, why: str):
        """Ask the provider to cancel the goal, and return its answer.

        Once a cancel was delivered, a later call sends nothing and returns
        the same answer. A cancel that does not reach the provider is logged
        and raises CancelNotDelivered; the next call sends it again.
        """

        async with self._cancelling:
            if self.cancel_answer is not None:
                return self.cancel_answer

            try:
                reply = await self.handle.cancel_goal(timeout=SERVICE_TIMEOUT_S)

            except (TimeoutError, ConnectionError) as exc:
                not_delivered = CancelNotDelivered(self.name, exc)
                logger.warning("%s: %s; %s", self.summary, why, not_delivered.outcome)
                raise not_delivered from exc

            logger.info("%s: %s; cancel %s", self.summary, why, reply.state.name)
            self.cancel_answer = reply.state

            if reply.state == self.action.CancelState.SIGNALLED:
                self.cancelled_by = by
            else:
                self.found_over.set()

            return reply.state

    async def result_found_over(self):
        """The result of a goal a delivered cancel found over, which the
        provider answers at once; a goal it does not know has none."""

        if self.cancel_answer == self.action.CancelState.UNKNOWN:
            raise RuntimeError(f"{self.name}: the simulator does not know the goal")

        return await self.handle.get_result(timeout=SERVICE_TIMEOUT_S)

    def stall(self, outcome: str) -> str:
        """The end of a load or a spawn that made no progress for a window."""

        return f"{self.name} made no progress for {PROGRESS_TIMEOUT_S:g} s; {outcome}"

    def data(self, result):
        """The data of a goal that succeeded.

        A provider may run a cancelled goal to its end and answer CANCELLED
        with success: what it built stands, so the goal succeeded. Every
        goal that succeeds leaves one log line naming what was asked and
        what the provider answered. A goal that ended any other way raises
        with the reason; the HTTP layer logs it.
        """

        status = self.action.ResultStatus

        if result.status not in (status.COMPLETED, status.CANCELLED):
            raise RuntimeError(f"{self.name} did not complete: {result.status.name}")

        if result.data is not None and result.data.success:
            logger.info("%s: %s", self.summary, result.data.message)

            return result.data

        if result.status == status.CANCELLED:
            raise self._cancelled(result.data)

        if result.data is None:
            raise RuntimeError(f"{self.name} completed without result data")

        raise RuntimeError(result.data.message)

    def _cancelled(self, data) -> GoalCancelled:
        # This commander reads the end of a goal whose cancel it had
        # signalled only when it follows its progress: the whole-goal bound
        # raises at once.
        if self.cancelled_by == Canceller.COMMANDER:
            return GoalCancelled(self.stall("it was cancelled"), Canceller.COMMANDER)

        if self.cancelled_by == Canceller.PAGE:
            return GoalCancelled(f"{self.name} was cancelled from the page", Canceller.PAGE)

        reason = f": {data.message}" if data is not None and data.message else ""

        return GoalCancelled(f"{self.name} was cancelled by the simulator{reason}", Canceller.SIMULATOR)


async def _fire_goal(action, node_runner: NodeRunner, request=None, **fields) -> _Goal:
    """Fire one scene_manipulation goal and return it once admitted.

    A rejection raises with the provider's reason. fields name what was
    asked, for the log lines of the goal.
    """

    name = _endpoint_name(action)
    producer = action.bound_producer(node_runner)
    goal = (request,) if request is not None else ()

    handle = await action.ActionHandle.fire_goal(
        node_runner,
        producer,
        *goal,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(f"{name} rejected: {handle.reason}")

    return _Goal(action, name, handle, _summary(name, fields))


async def _run_action(action, node_runner: NodeRunner, request=None, **fields):
    """Fire one goal of an action that reports no progress and return its
    result data.

    The whole goal is bounded by ACTION_TIMEOUT_S: a goal that has not
    ended by then is cancelled, and the timeout raised. A cancel that finds
    the goal ended answers with its result instead.
    """

    goal = await _fire_goal(action, node_runner, request, **fields)

    try:
        result = await goal.handle.get_result(timeout=ACTION_TIMEOUT_S)

    except TimeoutError as exc:
        bound = f"{goal.name} did not end within {ACTION_TIMEOUT_S:g} s"

        try:
            answer = await goal.cancel(Canceller.COMMANDER, f"no result within {ACTION_TIMEOUT_S:g} s")

        except CancelNotDelivered as not_delivered:
            raise TimeoutError(f"{bound}; {not_delivered.outcome}") from exc

        if answer == goal.action.CancelState.SIGNALLED:
            raise TimeoutError(f"{bound}; a cancel was sent") from exc

        result = await goal.result_found_over()

    return goal.data(result)


async def _follow_progress(
    goal: _Goal,
    silence: Callable[[float], Awaitable[None]],
    on_progress: Callable[[_Progress], Awaitable[None]],
):
    """Follow a load or a spawn until it ends, and return its result data.

    Each progress message is handed to on_progress and starts the silence
    window again: `silence` waits out PROGRESS_TIMEOUT_S. A window without
    a message has the goal cancelled, unless a cancel was delivered
    already, and the goal is followed on under the same rule to report how
    it really ended; the next silent window ends the follow (see
    _end_silent_follow). The feedback stream ends with the goal, whose
    result the provider then answers at once. A cancel from here or from
    the page that finds the goal over asks for its result at once too. A
    provider that is gone ends the follow at once: it has no result left to
    ask for.
    """

    next_message = asyncio.ensure_future(goal.handle.on_next_feedback_message())
    found_over = asyncio.ensure_future(goal.found_over.wait())
    stalled = False

    try:
        while True:
            window = asyncio.ensure_future(silence(PROGRESS_TIMEOUT_S))

            try:
                await asyncio.wait((next_message, found_over, window), return_when=asyncio.FIRST_COMPLETED)

            finally:
                window.cancel()

            if found_over.done():
                return goal.data(await goal.result_found_over())

            if not next_message.done():
                if stalled:
                    return await _end_silent_follow(goal)

                stalled = True

                # A cancel that is not delivered is logged, and sent again
                # when the next silent window ends the follow.
                with contextlib.suppress(CancelNotDelivered):
                    await goal.cancel(Canceller.COMMANDER, f"no progress for {PROGRESS_TIMEOUT_S:g} s")

                continue

            try:
                message = next_message.result()

            except ConnectionError as exc:
                raise ConnectionError(f"{goal.name} ended: the simulator is gone") from exc

            except RuntimeError:
                # The end of the stream: the goal has ended.
                break

            next_message = asyncio.ensure_future(goal.handle.on_next_feedback_message())
            await on_progress(_Progress.parse(message))

    finally:
        next_message.cancel()
        found_over.cancel()

    return goal.data(await goal.handle.get_result(timeout=SERVICE_TIMEOUT_S))


async def _end_silent_follow(goal: _Goal):
    """End the follow of a goal that stayed silent for a second window
    since its stall cancel, and return its result data if it has ended.

    When a cancel was delivered, the simulator did not answer it, and the
    follow ends. When none was, the cancel is sent again, once: the end
    says whether the simulator took it, and a goal it finds over is
    answered with its result.
    """

    if goal.cancel_answer is not None:
        raise TimeoutError(goal.stall("a cancel was sent and the simulator did not answer"))

    try:
        await goal.cancel(Canceller.COMMANDER, f"no progress for {PROGRESS_TIMEOUT_S:g} s again")

    except CancelNotDelivered as exc:
        raise TimeoutError(goal.stall(f"the cancel was sent again and was not delivered: {exc.reason}")) from exc

    if goal.found_over.is_set():
        return goal.data(await goal.result_found_over())

    raise TimeoutError(goal.stall("the cancel was sent again and the simulator took it"))


async def _fire_load_scene(node_runner: NodeRunner, asset_id: str, scale: float) -> _Goal:
    return await _fire_goal(
        load_scene,
        node_runner,
        load_scene.GoalRequest(asset_id=asset_id, scale=scale),
        asset_id=asset_id,
        scale=scale,
    )


async def _action_clear_scene(node_runner: NodeRunner) -> dict:
    data = await _run_action(clear_scene, node_runner)

    return {"success": True, "message": data.message}


async def _fire_spawn_object(node_runner: NodeRunner, request) -> _Goal:
    return await _fire_goal(
        spawn_object,
        node_runner,
        request,
        asset_id=request.asset_id,
        position=request.position,
        yaw=request.yaw,
        physics=request.physics,
        mass=request.mass,
    )


async def _action_apply_force(node_runner: NodeRunner, payload: dict) -> dict:
    force = [float(value) for value in payload["force"]]

    if len(force) != 3:
        raise ValueError("force must contain exactly 3 values")

    request = apply_force.GoalRequest(
        object_id=str(payload["object_id"]),
        force=force,
        duration_s=float(payload.get("duration_s", 0.5)),
    )

    data = await _run_action(
        apply_force,
        node_runner,
        request,
        object_id=request.object_id,
        force=force,
        duration_s=request.duration_s,
    )

    return {"success": True, "message": data.message}


async def _action_move_object(node_runner: NodeRunner, payload: dict) -> dict:
    # The panel moves an object without turning it.
    request = move_object.GoalRequest(
        object_id=str(payload["object_id"]),
        position=[float(value) for value in payload["position"]],
        orientation=None,
    )

    data = await _run_action(
        move_object,
        node_runner,
        request,
        object_id=request.object_id,
        position=request.position,
    )

    return {"success": True, "message": data.message}


async def _action_remove_object(node_runner: NodeRunner, object_id: str) -> dict:
    data = await _run_action(
        remove_object,
        node_runner,
        remove_object.GoalRequest(object_id=object_id),
        object_id=object_id,
    )

    return {"success": True, "message": data.message}


async def _action_move_robot(
    node_runner: NodeRunner, robot: str, position: list[float], yaw: float
) -> dict:
    position = [float(value) for value in position]

    data = await _run_action(
        move_robot,
        node_runner,
        move_robot.GoalRequest(robot=robot, position=position, yaw=yaw),
        robot=robot,
        position=position,
        yaw=yaw,
    )

    return {"success": True, "message": data.message}


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------


class _StreamedGoals:
    """The goals this commander streams to a page, by the id each stream
    opens with; the cancel route finds a goal here while its stream runs."""

    def __init__(self) -> None:
        self._goals: dict[str, _Goal] = {}
        self._ids = itertools.count(1)

    def add(self, goal: _Goal) -> str:
        goal_id = str(next(self._ids))
        self._goals[goal_id] = goal

        return goal_id

    def find(self, goal_id: str) -> _Goal | None:
        return self._goals.get(goal_id)

    def remove(self, goal_id: str) -> None:
        del self._goals[goal_id]


_NODE_RUNNER = web.AppKey("node_runner", NodeRunner)
_CAPABILITIES = web.AppKey("capabilities", _Capabilities)
_CATALOGUE = web.AppKey("catalogue", _StateWatch)
_OBJECT_STATE = web.AppKey("object_state", _StateWatch)
# One profile watch per bound camera, by camera id.
_PROFILES = web.AppKey("profiles", dict)
# The wait of a load's or a spawn's silence window.
_SILENCE = web.AppKey[Callable[[float], Awaitable[None]]]("silence")
_STREAMED_GOALS = web.AppKey("streamed_goals", _StreamedGoals)


def _log_failure(request: web.Request, exc: Exception) -> None:
    # Bad input, provider refusals and goals that did not succeed are one
    # line each; anything else is a bug in this node and keeps its traceback.
    expected = isinstance(exc, (ValueError, KeyError, RuntimeError, TimeoutError, ConnectionError))

    logger.warning(
        "%s %s failed: %s",
        request.method,
        request.path,
        exc,
        exc_info=None if expected else exc,
    )


def _failure(exc: Exception) -> dict:
    body = {"success": False, "message": str(exc)}

    # A refused setter answers with what stands, as the provider reported it.
    if isinstance(exc, ProviderRefusal):
        body.update(exc.current)

    if isinstance(exc, GoalCancelled):
        body["cancelled_by"] = exc.by

    return body


def _json_error(request: web.Request, exc: Exception, status: int = 400) -> web.Response:
    _log_failure(request, exc)

    return web.json_response(_failure(exc), status=status)


class _GoalStream:
    """The answer of an admitted load or spawn: one JSON line per event.

    aiohttp leaves a handler running when its page goes away, and the
    handler learns it from a write; no line is written after one the page
    could not take. A line written while the goal is followed has the goal
    cancelled then. The result line comes once the follow has ended, so a
    page that cannot take it cancels nothing, and its close is logged.
    """

    def __init__(self, request: web.Request, goal: _Goal) -> None:
        self.response = web.StreamResponse()
        self.response.content_type = "application/x-ndjson"
        self._request = request
        self._goal = goal
        self._page_open = True

    async def write(self, line: dict) -> None:
        """Write a line of a goal this commander follows."""

        if not await self._write_finds_page_gone(line):
            return

        # A page that goes away cancels its goal, as an MCP client that
        # disconnects cancels its call: nobody follows the goal any more,
        # and a fetch nobody watches would keep the simulator's link busy. A
        # cancel that is not delivered is logged, and the goal is followed
        # to its end under the silence rule.
        with contextlib.suppress(CancelNotDelivered):
            await self._goal.cancel(Canceller.PAGE, "the page closed its stream")

    async def progress(self, progress: _Progress) -> None:
        await self.write({"progress": asdict(progress)})

    async def write_result(self, result: dict) -> None:
        """Write the last line, once the follow of the goal has ended."""

        if await self._write_finds_page_gone(result):
            logger.info("%s: the page closed its stream before its result", self._goal.summary)

    async def _write_finds_page_gone(self, line: dict) -> bool:
        """Write one line to the page, and return whether this write found
        the page gone. A page found gone before gets no line."""

        if not self._page_open:
            return False

        try:
            if not self.response.prepared:
                await self.response.prepare(self._request)

            await self.response.write(json.dumps(line).encode() + b"\n")

        except ConnectionError:
            self._page_open = False

            return True

        return False


async def _stream_goal(
    request: web.Request, goal: _Goal, answer: Callable[[object], dict]
) -> web.StreamResponse:
    """Follow an admitted load or spawn and stream it to the page.

    The page reads the goal's id first, which the cancel route takes, then
    one line per progress message, and last the result: `answer` builds it
    from the result data, and a goal that did not succeed ends the stream
    with its failure, as a JSON answer carries one.
    """

    goals = request.app[_STREAMED_GOALS]
    goal_id = goals.add(goal)
    stream = _GoalStream(request, goal)

    try:
        await stream.write({"goal": goal_id})

        try:
            data = await _follow_progress(goal, request.app[_SILENCE], stream.progress)
            result = answer(data)

        except Exception as exc:
            _log_failure(request, exc)
            result = _failure(exc)

        await stream.write_result(result)

    finally:
        goals.remove(goal_id)

    return stream.response


async def _request_json(request: web.Request) -> dict:
    try:
        data = await request.json()

    except Exception as exc:
        raise ValueError("Request body must contain valid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON request body must be an object")

    return data


def _is_finite(value) -> bool:
    # JSON true and false are numbers to Python; they are not values here.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(payload: dict, key: str) -> float:
    value = payload.get(key)

    if not _is_finite(value):
        raise ValueError(f"{key} must be a finite number")

    return float(value)


def _whole(payload: dict, key: str) -> int:
    value = _number(payload, key)

    if not value.is_integer():
        raise ValueError(f"{key} must be a whole number")

    return int(value)


def _vector(payload: dict, key: str, length: int) -> list[float]:
    values = payload.get(key)

    if not isinstance(values, list) or len(values) != length or not all(map(_is_finite, values)):
        raise ValueError(f"{key} must be {length} finite numbers")

    return [float(value) for value in values]


def _one_of(payload: dict, key: str, choices: tuple[str, ...]) -> str:
    value = payload.get(key)

    if value not in choices:
        quoted = [f'"{choice}"' for choice in choices]
        raise ValueError(f"{key} must be {', '.join(quoted[:-1])} or {quoted[-1]}")

    return value


def _mode(payload: dict) -> str:
    return _one_of(payload, "mode", ("auto", "manual"))


def _name(payload: dict, key: str) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")

    return value


def _optional(parse: Callable[[dict, str], object], payload: dict, key: str, default):
    """A field the payload may leave out: parsed by `parse` when present,
    `default` when absent."""

    return parse(payload, key) if key in payload else default


# The physics a spawned object takes, as scene_manipulation names them.
_PHYSICS = ("dynamic", "static", "none")


def _physics(payload: dict, key: str) -> str:
    return _one_of(payload, key, _PHYSICS)


def _spawn_request(payload: dict):
    """The spawn_object goal the page's payload asks for, every field
    parsed. A field it leaves out gives an object as authored, visual
    only, with the page's default mass."""

    return spawn_object.GoalRequest(
        asset_id=_name(payload, "asset_id"),
        position=_vector(payload, "position", 3),
        yaw=_optional(_number, payload, "yaw", 0.0),
        # The panel turns an object by its yaw alone.
        orientation=None,
        scale=_optional(_number, payload, "scale", 1.0),
        physics=_optional(_physics, payload, "physics", "none"),
        mass=_optional(_number, payload, "mass", 0.1),
    )


# ---------------------------------------------------------------------------
# Panels: lighting and materials
# ---------------------------------------------------------------------------
#
# The two panels share one shape, as the page's PANELS table has it: a
# getter answering the targets as JSON, setters by the route the page posts
# to, each naming the target by its id field and taking the fields its
# parser checks, and a reset. Every payload is checked in full before any
# provider is called: a value that is not a finite number, a vector of the
# wrong length or a mode the contract does not name is refused here.


def _vector3(payload: dict, key: str) -> list[float]:
    return _vector(payload, key, 3)


def _vector4(payload: dict, key: str) -> list[float]:
    return _vector(payload, key, 4)


@dataclass(frozen=True)
class _Panel:
    """One panel of the page: the capability slot it needs (a field of
    _Capabilities, and the refusal when the launch left it vacant), what a
    property belongs to in a refusal, the getter with the response field
    carrying its JSON and the key listing its targets, the id field its
    setters address a target by, the setters by route with the fields each
    takes, the reset, and the watch logging whether the provider has the
    state."""

    name: str
    unbound: str
    owner: str
    get: ModuleType
    json_field: str
    list_key: str
    id_field: str
    setters: dict[str, tuple[ModuleType, dict[str, Callable[[dict, str], object]]]]
    reset: ModuleType
    watch: web.AppKey

    def producer(self, app: web.Application):
        producer = getattr(app[_CAPABILITIES], self.name)

        if producer is None:
            raise CapabilityUnbound(self.unbound)

        return producer

    def request(self, route: str, payload: dict):
        """The setter answering `route` and its request, built from the
        page's payload: the target's id and every field the setter takes."""

        service, fields = self.setters[route]
        target = {self.id_field: _name(payload, self.id_field)}

        return service, service.Request(
            **target, **{key: parse(payload, key) for key, parse in fields.items()}
        )


_LIGHTING = web.AppKey("lighting", _StateWatch)
_MATERIALS = web.AppKey("materials", _StateWatch)

_PANELS = (
    _Panel(
        name="lighting",
        unbound="lighting is not bound in this launch",
        owner="lighting",
        get=get_lighting,
        json_field="lighting_json",
        list_key="lights",
        id_field="light_id",
        setters={
            "intensity": (set_light_intensity, {"value": _number}),
            "color": (set_light_color, {"color": _vector3}),
            "position": (set_light_position, {"position": _vector3}),
            "direction": (set_light_direction, {"direction": _vector3}),
            "cone": (set_light_cone, {"inner_angle": _number, "outer_angle": _number}),
            "orientation": (set_light_orientation, {"orientation": _vector4}),
        },
        reset=reset_lighting,
        watch=_LIGHTING,
    ),
    _Panel(
        name="materials",
        unbound="materials are not bound in this launch",
        owner="material",
        get=get_materials,
        json_field="materials_json",
        list_key="materials",
        id_field="material_id",
        setters={
            "color": (set_material_color, {"color": _vector3}),
            "finish": (set_material_finish, {"metallic": _number, "roughness": _number}),
        },
        reset=reset_materials,
        watch=_MATERIALS,
    ),
)


def _camera_producer(app: web.Application):
    producer = app[_CAPABILITIES].cameras

    if producer is None:
        raise CapabilityUnbound("cameras are not bound in this launch")

    return producer


def _robot(payload: dict) -> str:
    robot = payload.get("robot")

    if not isinstance(robot, str) or not robot:
        raise ValueError("robot must be the name of a robot, as /api/robots lists them")

    return robot


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


async def _api_health(request: web.Request) -> web.Response:
    return web.json_response({"success": True, "service": "scene_commander"})


async def _api_assets(request: web.Request) -> web.Response:
    catalogue = request.app[_CATALOGUE]

    try:
        assets = await _fetch_assets(request.app[_NODE_RUNNER])

    except SceneCatalogueUnavailable as exc:
        catalogue.unavailable(str(exc))

        return web.json_response(
            {"success": False, "message": str(exc)},
            status=503,
        )

    except Exception as exc:
        return _json_error(request, exc, status=500)

    catalogue.ready(len(assets))

    return web.json_response(
        {"success": True, "assets": assets, "count": len(assets)}
    )


async def _api_objects(request: web.Request) -> web.Response:
    object_state = request.app[_OBJECT_STATE]

    try:
        snapshot = await _fetch_objects(request.app[_NODE_RUNNER])

    except ObjectStateUnavailable as exc:
        object_state.unavailable(str(exc))

        return web.json_response(
            {"success": False, "message": str(exc)},
            status=503,
        )

    except Exception as exc:
        return _json_error(request, exc, status=500)

    objects = snapshot["objects"]
    object_state.ready(len(objects))

    return web.json_response(
        {
            "success": True,
            "objects": objects,
            "count": len(objects),
            "timestamp": snapshot["timestamp"],
        }
    )


async def _api_load_scene(request: web.Request) -> web.StreamResponse:
    try:
        payload = await _request_json(request)

        goal = await _fire_load_scene(
            request.app[_NODE_RUNNER],
            _name(payload, "asset_id"),
            _optional(_number, payload, "scale", 1.0),
        )

    except Exception as exc:
        return _json_error(request, exc)

    return await _stream_goal(request, goal, lambda data: {"success": True, "message": data.message})


async def _api_clear_scene(request: web.Request) -> web.Response:
    try:
        result = await _action_clear_scene(request.app[_NODE_RUNNER])

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


async def _api_spawn_object(request: web.Request) -> web.StreamResponse:
    try:
        goal = await _fire_spawn_object(
            request.app[_NODE_RUNNER],
            _spawn_request(await _request_json(request)),
        )

    except Exception as exc:
        return _json_error(request, exc)

    return await _stream_goal(
        request,
        goal,
        lambda data: {"success": True, "message": data.message, "object_id": data.object_id},
    )


async def _api_cancel_goal(request: web.Request) -> web.Response:
    """Cancel a load or a spawn by the id its stream opened with; a stream
    still open then ends with how the goal really ended. A cancel that does
    not reach the simulator answers 502 with the reason, and a caller may
    send it again while the stream runs."""

    goal_id = request.match_info["goal"]
    goal = request.app[_STREAMED_GOALS].find(goal_id)

    if goal is None:
        return _json_error(request, ValueError(f"no running goal {goal_id}"), status=404)

    try:
        answer = await goal.cancel(Canceller.PAGE, "the page asked")

    except CancelNotDelivered as exc:
        # The cancel logged why it was not delivered.
        return web.json_response(_failure(exc), status=502)

    return web.json_response(
        {"success": True, "message": f"the simulator answered the cancel of {goal.name}: {answer.name}"}
    )


async def _api_apply_force(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)

        force = payload.get("force", [])

        if not isinstance(force, list) or len(force) != 3:
            raise ValueError("force must contain exactly 3 values [Fx, Fy, Fz]")

        payload["force"] = [float(value) for value in force]
        payload["duration_s"] = float(payload.get("duration_s", 0.5))

        result = await _action_apply_force(request.app[_NODE_RUNNER], payload)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


async def _api_move_object(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)
        payload["position"] = _vector(payload, "position", 3)

        result = await _action_move_object(request.app[_NODE_RUNNER], payload)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


async def _api_remove_object(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)

        result = await _action_remove_object(
            request.app[_NODE_RUNNER],
            str(payload["object_id"]),
        )

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


async def _api_robots(request: web.Request) -> web.Response:
    try:
        robots = await _fetch_robots(request.app[_NODE_RUNNER])

    except Exception as exc:
        return _json_error(request, exc, status=500)

    return web.json_response(
        {"success": True, "robots": robots, "count": len(robots)}
    )


async def _api_move_robot(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)

        result = await _action_move_robot(
            request.app[_NODE_RUNNER],
            _robot(payload),
            _vector(payload, "position", 3),
            _number(payload, "yaw"),
        )

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


# ---------------------------------------------------------------------------
# HTTP API: lighting, materials, cameras
# ---------------------------------------------------------------------------


async def _api_capabilities(request: web.Request) -> web.Response:
    capabilities = request.app[_CAPABILITIES]

    return web.json_response(
        {
            "success": True,
            "lighting": capabilities.lighting is not None,
            "materials": capabilities.materials is not None,
            "cameras": capabilities.cameras is not None,
        }
    )


async def _api_panel(panel: _Panel, request: web.Request) -> web.Response:
    """The panel's targets as the provider lists them."""

    watch = request.app[panel.watch]

    try:
        state, _ = await _fetch_json(
            panel.get,
            request.app[_NODE_RUNNER],
            panel.producer(request.app),
            panel.json_field,
        )

        count = len(state[panel.list_key])

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except StateUnavailable as exc:
        watch.unavailable(str(exc))

        return web.json_response(
            {"success": False, "message": str(exc)},
            status=503,
        )

    except Exception as exc:
        return _json_error(request, exc, status=500)

    watch.ready(count)

    return web.json_response({"success": True, panel.name: state})


async def _api_set_panel(panel: _Panel, request: web.Request) -> web.Response:
    """One property of one target, posted to the setter's route."""

    route = request.match_info["property"]

    if route not in panel.setters:
        return _json_error(
            request,
            ValueError(f"unknown {panel.owner} property: {route}"),
            status=404,
        )

    try:
        producer = panel.producer(request.app)
        service, built = panel.request(route, await _request_json(request))

        data = await _call_service(service, request.app[_NODE_RUNNER], producer, built)

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _api_reset_panel(panel: _Panel, request: web.Request) -> web.Response:
    try:
        data = await _call_service(
            panel.reset,
            request.app[_NODE_RUNNER],
            panel.producer(request.app),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _describe_camera(app: web.Application, listed) -> dict:
    """One camera as the page lists it: its stream from the simulation's
    listing, and its device profile when the simulation describes one."""

    camera_id = f"{listed.robot}/{listed.camera}"
    watch = app[_PROFILES].setdefault(
        camera_id, _StateWatch("profile", "controls", provider=f"Camera {camera_id}")
    )
    response = await describe_camera.poll(
        app[_NODE_RUNNER],
        _camera_producer(app),
        describe_camera.Request(robot=listed.robot, camera=listed.camera),
        timeout=SERVICE_TIMEOUT_S,
    )
    data = response.data
    profile = json.loads(data.profile_json) if data.success else None

    if profile is None:
        watch.unavailable(data.message)
    else:
        watch.ready(sum(1 for control in profile["controls"].values() if control.get("supported")))

    return {
        "id": camera_id,
        "robot": listed.robot,
        "camera": listed.camera,
        "kind": listed.kind,
        "info": {
            "width": listed.width,
            "height": listed.height,
            "frames_per_second": listed.frames_per_second,
            "encoding": listed.encoding,
        },
        "profile": profile,
        "profile_message": data.message,
    }


async def _api_cameras(request: web.Request) -> web.Response:
    """Every camera the simulation renders, described together: one
    camera's round trips never wait on another's."""

    try:
        response = await get_cameras.poll(
            request.app[_NODE_RUNNER], _camera_producer(request.app), timeout=SERVICE_TIMEOUT_S
        )
        listing = response.data

        if not listing.success:
            raise StateUnavailable(listing.message)

        cameras = await asyncio.gather(
            *(_describe_camera(request.app, listed) for listed in listing.cameras)
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc, status=500)

    return web.json_response(
        {"success": True, "message": listing.message, "cameras": cameras, "count": len(cameras)}
    )


async def _api_set_camera(request: web.Request) -> web.Response:
    control = _CAMERA_CONTROLS.get(request.match_info["control"])

    if control is None:
        return _json_error(
            request,
            ValueError(f"unknown camera control: {request.match_info['control']}"),
            status=404,
        )

    try:
        payload = await _request_json(request)

        data = await _call_service(
            control.service,
            request.app[_NODE_RUNNER],
            _camera_producer(request.app),
            control.service.Request(
                robot=request.match_info["robot"],
                camera=request.match_info["camera"],
                **control.fields(payload),
            ),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _api_reset_camera(request: web.Request) -> web.Response:
    try:
        data = await _call_service(
            reset_camera,
            request.app[_NODE_RUNNER],
            _camera_producer(request.app),
            reset_camera.Request(
                robot=request.match_info["robot"], camera=request.match_info["camera"]
            ),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


# ---------------------------------------------------------------------------
# Browser UI
# ---------------------------------------------------------------------------


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scene Commander</title>

<style>
:root {
    font-family: Inter, system-ui, sans-serif;
    color-scheme: dark;
}

body {
    margin: 0;
    background: #101318;
    color: #eef2f7;
}

header {
    padding: 18px 24px;
    background: #171c23;
    border-bottom: 1px solid #303741;
}

header h1 {
    margin: 0;
    font-size: 22px;
}

header p {
    margin: 6px 0 0;
    color: #aeb7c4;
}

main {
    display: grid;
    grid-template-columns: repeat(auto-fit,minmax(340px,1fr));
    gap: 16px;
    padding: 16px;
}

.card {
    background: #171c23;
    border: 1px solid #303741;
    border-radius: 10px;
    padding: 16px;
}

.card h2 {
    margin-top: 0;
    font-size: 18px;
}

label {
    display: block;
    margin-top: 10px;
    color: #b9c3d0;
    font-size: 13px;
}

input,
select,
button {
    box-sizing: border-box;
    width: 100%;
    margin-top: 5px;
    padding: 9px;
    border-radius: 6px;
    border: 1px solid #3b4552;
    background: #101318;
    color: #eef2f7;
}

button {
    cursor: pointer;
    background: #273345;
}

button:hover {
    background: #35465f;
}

button.danger {
    background: #572c32;
}

.row {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 8px;
}

.two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.four {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 8px;
}

#status {
    margin: 0 16px 16px;
    padding: 12px;
    border-radius: 8px;
    background: #171c23;
    border: 1px solid #303741;
    white-space: pre-wrap;
}

.object,
.target {
    border-top: 1px solid #303741;
    margin-top: 12px;
    padding-top: 12px;
}

.object code,
.target code {
    word-break: break-all;
    color: #a9d0ff;
}

.small {
    font-size: 12px;
    color: #aeb7c4;
}

#assetCount,
#objectCount {
    font-size: 12px;
    color: #aeb7c4;
}

select:disabled {
    color: #aeb7c4;
}

#status.loading::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 8px;
    vertical-align: -2px;
    border: 2px solid #aeb7c4;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
</style>
</head>

<body>

<header>
<h1>Scene Commander</h1>
<p>Peppy-native runtime scene and object control</p>
</header>

<main>

<section class="card">
<h2>Scene</h2>

<label for="sceneSelect">Scene</label>
<select id="sceneSelect" aria-describedby="sceneDescription" onchange="selectScene()" disabled>
<option value="" disabled>Loading assets...</option>
</select>
<p id="sceneDescription" class="small" aria-live="polite" hidden></p>

<label>Scale</label>
<input id="sceneScale" type="number" step="0.1" value="1.0">

<div class="two">
<button onclick="loadScene()">Load Scene</button>
<button class="danger" onclick="clearScene()">Clear Scene</button>
</div>
<button id="cancelLoad" class="danger" onclick="cancelSceneLoad()" hidden>Cancel Load</button>
</section>


<section class="card">
<h2>Spawn Object</h2>

<label>Search assets</label>
<input id="assetSearch" placeholder="red block, table, drill..." oninput="renderAssets()">

<label>Category</label>
<select id="categorySelect" onchange="renderAssets()" disabled>
<option value="" disabled>Loading assets...</option>
</select>

<label for="assetSelect">Asset</label>
<select id="assetSelect" aria-describedby="assetDescription" onchange="selectAsset()" disabled>
<option value="" disabled>Loading assets...</option>
</select>
<p id="assetDescription" class="small" aria-live="polite" hidden></p>

<div id="assetCount">Loading assets...</div>

<label>Position</label>
<div class="row">
<input id="spawnX" type="number" step="0.05" value="0.5">
<input id="spawnY" type="number" step="0.05" value="0.0">
<input id="spawnZ" type="number" step="0.05" value="0.8">
</div>

<label>Yaw (rad, about +Z)</label>
<input id="spawnYaw" type="number" step="0.05" value="0">

<div class="two">
<div>
<label>Scale</label>
<input id="spawnScale" type="number" step="0.1" value="1.0">
</div>

<div>
<label>Mass</label>
<input id="spawnMass" type="number" step="0.1" value="0.1">
</div>
</div>

<label>Physics</label>
<select id="spawnPhysics">
<option value="none">None</option>
<option value="static">Static</option>
<option value="dynamic" selected>Dynamic</option>
</select>

<button onclick="spawnObject()">Spawn Object</button>
</section>


<section class="card">
<h2>Robot Root</h2>

<label>Robot</label>
<div class="row">
<select id="robotSelect" onchange="selectRobot()" disabled></select>
<button onclick="refreshRobots()">Refresh</button>
</div>
<div id="robotCount"></div>

<label>Position</label>
<div class="row">
<input id="robotX" type="number" step="0.05" value="0">
<input id="robotY" type="number" step="0.05" value="0">
<input id="robotZ" type="number" step="0.05" value="0">
</div>

<label>Yaw (rad, about +Z)</label>
<input id="robotYaw" type="number" step="0.05" value="0">

<button onclick="moveRobot()">Move Robot</button>
</section>


<section class="card">
<h2>Runtime Objects</h2>

<div class="two">
<button onclick="refreshObjects()">Refresh</button>
<div id="objectCount"></div>
</div>

<div id="objects"></div>
</section>


<section class="card" id="lightingCard" hidden>
<h2>Lighting</h2>

<div class="two">
<button class="danger" onclick="resetPanel('lighting')">Reset lighting</button>
<div id="lightingSummary" class="small"></div>
</div>

<div id="lightingTargets"></div>
</section>


<section class="card" id="materialsCard" hidden>
<h2>Materials</h2>

<div class="two">
<button class="danger" onclick="resetPanel('materials')">Reset materials</button>
<div id="materialsSummary" class="small"></div>
</div>

<div id="materialsTargets"></div>
</section>


<section class="card" id="camerasCard" hidden>
<h2>Cameras</h2>

<div id="cameraTargets"></div>
</section>

</main>

<div id="status" class="loading">Loading assets...</div>


<script>
let assets = [];
let objectList = [];

// The provider may still be discovering its assets (Isaac lists its props
// once the stage has loaded); ask again at this interval until it has them.
const CATALOGUE_RETRY_MS = 3000;

const el = id => document.getElementById(id);

function number(id) {
    return Number(el(id).value);
}

function position(prefix) {
    return [
        number(prefix + "X"),
        number(prefix + "Y"),
        number(prefix + "Z")
    ];
}

function status(message, error=false) {
    el("status").textContent = message;
    el("status").style.borderColor = error ? "#9b424c" : "#303741";
}

function setLoading(loading) {
    el("status").classList.toggle("loading", loading);

    for (const id of ["sceneSelect", "categorySelect", "assetSelect"]) {
        const select = el(id);
        select.disabled = loading;
        if (loading) {
            const placeholder = new Option("Loading assets...", "");
            placeholder.disabled = true;
            select.replaceChildren(placeholder);
        }
    }

    if (loading) {
        el("assetCount").textContent = "Loading assets...";
    }
}

async function api(path, options={}) {
    const response = await fetch(path, {
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        },
        ...options
    });

    const data = await response.json();

    if (!response.ok || data.success === false) {
        throw new Error(data.message || `HTTP ${response.status}`);
    }

    return data;
}

// A load or a spawn the simulator accepted answers with one JSON line per
// event: {"goal": id} first, then {"progress": {...}} each time the
// simulator reports progress, and last the result, shaped as the other
// actions answer. A refusal answers before any line, as they do. The
// stream holds its connection to the commander until its result, or until
// `signal` aborts it.
async function runGoal(path, body, { onProgress, onGoal = () => {}, signal }) {
    const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal
    });

    if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || `HTTP ${response.status}`);
    }

    for await (const line of jsonLines(response.body)) {
        if ("goal" in line) {
            onGoal(line.goal);
        }
        else if ("progress" in line) {
            onProgress(line.progress);
        }
        else {
            return line;
        }
    }

    throw new Error(`${path} ended without a result`);
}

// The JSON lines of a streamed answer, each parsed once it is whole.
async function* jsonLines(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    for (;;) {
        const { done, value } = await reader.read();
        pending += done ? decoder.decode() : decoder.decode(value, { stream: true });

        const lines = pending.split("\n");
        pending = done ? "" : lines.pop();

        for (const line of lines.filter(line => line.trim())) {
            yield JSON.parse(line);
        }

        if (done) {
            return;
        }
    }
}

// What the status says of a load in progress: the bytes and the files
// ready while the simulator fetches, then that it builds.
function progressText(progress, loading, building) {
    if (progress.building) {
        return building;
    }

    const megabytes = (progress.bytes_fetched / 1e6).toFixed(1);

    return `${loading}: ${megabytes} MB, ${progress.files_ready} files`;
}

async function refreshAssets() {
    const data = await api("/api/assets");

    assets = data.assets || [];
    setLoading(false);

    const scenes = assets.filter(a => a.kind === "scene");
    const props = assets.filter(a => a.kind === "object");

    const sceneSelect = el("sceneSelect");
    const previousScene = sceneSelect.value;
    sceneSelect.replaceChildren(...scenes.map(a =>
        new Option(a.display_name, a.asset_id)
    ));
    if (scenes.some(a => a.asset_id === previousScene)) {
        sceneSelect.value = previousScene;
    }
    selectScene();

    const categories = [...new Set(
        props.map(a => a.category).filter(Boolean)
    )].sort();

    const categorySelect = el("categorySelect");
    const previousCategory = categorySelect.value;
    categorySelect.replaceChildren(
        new Option("All categories", ""),
        ...categories.map(c => new Option(c, c))
    );
    if (categories.includes(previousCategory)) {
        categorySelect.value = previousCategory;
    }

    renderAssets();
    renderObjectDescriptions();

    status(`Loaded ${assets.length} assets`);
}

async function loadCatalogue() {
    setLoading(true);

    for (;;) {
        try {
            await refreshAssets();
            return;
        }
        catch (err) {
            status(`Waiting for the scene provider: ${err.message}`);
            await new Promise(resolve => setTimeout(resolve, CATALOGUE_RETRY_MS));
        }
    }
}

function selectScene() {
    const scene = assets.find(a =>
        a.kind === "scene" && a.asset_id === el("sceneSelect").value
    );
    showDescription("sceneDescription", scene);
}

function showDescription(id, asset) {
    const description = el(id);
    description.textContent = asset?.description || "";
    description.hidden = !description.textContent;
}

function selectAsset(resetDefaults = true) {
    const asset = assets.find(a =>
        a.kind === "object" && a.asset_id === el("assetSelect").value
    );
    showDescription("assetDescription", asset);
    if (resetDefaults) {
        const mass = Number(asset?.default_mass);
        el("spawnMass").value = Number.isFinite(mass) && mass > 0 ? mass : 0.1;
        el("spawnPhysics").value = "dynamic";
    }
}

function renderAssets() {
    const previous = el("assetSelect").value;
    const search = el("assetSearch").value.trim().toLowerCase();
    const category = el("categorySelect").value;

    const filtered = assets
        .filter(a => a.kind === "object")
        .filter(a =>
            !category || a.category === category
        )
        .filter(a => {
            if (!search) return true;

            return (
                a.display_name.toLowerCase().includes(search) ||
                a.asset_id.toLowerCase().includes(search) ||
                (a.category || "").toLowerCase().includes(search) ||
                (a.description || "").toLowerCase().includes(search)
            );
        });

    el("assetSelect").replaceChildren(...filtered.map(a =>
        new Option([a.display_name, a.category].filter(Boolean).join(" - "), a.asset_id)
    ));

    if (filtered.some(a => a.asset_id === previous)) {
        el("assetSelect").value = previous;
    }
    selectAsset(el("assetSelect").value !== previous);

    el("assetCount").textContent =
        `${filtered.length} matching assets`;
}

// The newest scene load this page started that the commander accepted,
// with its goal id and the controller that aborts the page's stream of
// it. Only that load speaks in the status and has the Cancel button: a
// load started before it ends in silence here, however it ends, a newer
// load the commander refuses leaves it running, and a load the page
// cancels leaves the status to its cancel.
let sceneLoad = null;

async function loadScene() {
    const load = { goal: null, stream: new AbortController() };
    const newest = () => sceneLoad === load;

    try {
        status("Loading scene...");

        const result = await runGoal("/api/scene/load", {
            asset_id: el("sceneSelect").value,
            scale: number("sceneScale")
        }, {
            onProgress: progress => {
                if (newest()) {
                    status(progressText(progress, "Loading scene", "Building the scene"));
                }
            },
            onGoal: goal => {
                load.goal = goal;
                sceneLoad = load;
                el("cancelLoad").hidden = false;
            },
            signal: load.stream.signal
        });

        if (!newest()) {
            return;
        }

        if (!result.success) {
            throw new Error(result.message);
        }

        await refreshObjects();
        await refreshPanels();

        status(result.message);
    }
    catch (err) {
        // A load the commander refused was never the newest, and says why.
        if (load.goal === null || newest()) {
            status(err.message, true);
        }
    }
    finally {
        if (newest()) {
            sceneLoad = null;
            el("cancelLoad").hidden = true;
        }
    }
}

// Cancel first aborts the page's own stream of the load, which frees its
// connection at once: a browser keeps six connections at most to the
// commander, and each load or spawn holds one until its result, so the
// cancel never waits behind them. The commander then cancels the load for
// the cancel route, and also when the next line of the aborted stream
// finds the page gone. With its stream aborted, the page does not read how
// the load ends: the answer to the cancel speaks in the status, unless the
// commander has accepted a newer load since.
async function cancelSceneLoad() {
    const load = sceneLoad;

    if (load === null) {
        return;
    }

    sceneLoad = null;
    el("cancelLoad").hidden = true;
    load.stream.abort();
    status("Cancelling the scene load...");

    const [message, failed] = await api(`/api/goals/${encodeURIComponent(load.goal)}/cancel`, {
        method: "POST",
        body: "{}"
    }).then(data => [data.message, false], err => [err.message, true]);

    if (sceneLoad === null) {
        status(message, failed);
    }
}

async function clearScene() {
    try {
        status("Clearing runtime scene...");

        const data = await api("/api/scene/clear", {
            method: "POST",
            body: "{}"
        });

        await refreshObjects();
        await refreshPanels();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

async function spawnObject() {
    try {
        const assetId = el("assetSelect").value;

        if (!assetId) {
            throw new Error("Select an asset first");
        }

        status(`Spawning ${assetId}...`);

        const result = await runGoal("/api/objects/spawn", {
            asset_id: assetId,
            position: position("spawn"),
            yaw: number("spawnYaw"),
            scale: number("spawnScale"),
            physics: el("spawnPhysics").value,
            mass: number("spawnMass")
        }, {
            onProgress: progress => {
                status(progressText(progress, `Loading ${assetId}`, `Building ${assetId}`));
            }
        });

        if (!result.success) {
            throw new Error(result.message);
        }

        await refreshObjects();

        status(
            `${result.message}\nobject_id=${result.object_id}`
        );
    }
    catch (err) {
        status(err.message, true);
    }
}

async function refreshObjects() {
    try {
        const data = await api("/api/objects");

        objectList = data.objects || [];

        el("objectCount").textContent =
            `${objectList.length} objects`;

        renderObjects();
    }
    catch (err) {
        // Without a snapshot the objects are unknown: none of them keeps
        // its controls, and the panel says why instead of an empty scene.
        const message = `Object state unavailable: ${err.message}`;

        objectList = [];

        el("objectCount").textContent = "unavailable";

        renderObjectsUnavailable(message);

        status(message, true);
    }
}

function renderObjectsUnavailable(message) {
    el("objects").innerHTML =
        `<p id="objectsUnavailable" class="small"></p>`;
    el("objectsUnavailable").textContent = message;
}

function renderObjectDescriptions() {
    const catalogue = new Map(
        assets.filter(a => a.kind === "object").map(a => [a.asset_id, a])
    );
    objectList.forEach((obj, index) => {
        showDescription(`objectDescription${index}`, catalogue.get(obj.asset_id));
    });
}

function renderObjects() {
    const container = el("objects");

    if (!objectList.length) {
        container.innerHTML =
            `<p class="small">No runtime objects.</p>`;
        return;
    }

    container.innerHTML = objectList.map((obj, index) => {
        const p = obj.position || [0,0,0];

        const dynamic =
            String(obj.physics || "").toLowerCase()
            === "dynamic";

        return `
        <div class="object">
            <code>${obj.object_id}</code>

            <div class="small">
                ${obj.asset_id}
                &nbsp;|&nbsp;
                physics=${obj.physics || "none"}
                &nbsp;|&nbsp;
                mass=${obj.mass ?? "-"} kg
                &nbsp;|&nbsp;
                scale=${obj.scale ?? "-"}
            </div>
            <p id="objectDescription${index}" class="small" hidden></p>

            <label>Position</label>

            <div class="row">
                <input id="ox${index}" type="number"
                    step="0.05" value="${p[0]}">
                <input id="oy${index}" type="number"
                    step="0.05" value="${p[1]}">
                <input id="oz${index}" type="number"
                    step="0.05" value="${p[2]}">
            </div>

            <div class="two">
                <button onclick="moveObject(${index})">
                    Move
                </button>

                <button class="danger"
                    onclick="removeObject(${index})">
                    Remove
                </button>
            </div>

            ${
                dynamic
                ? `
                    <label>Force magnitude (N)</label>
                    <input
                        id="forceMag${index}"
                        type="number"
                        step="1"
                        min="0"
                        value="20">

                    <label>Duration (s)</label>
                    <input
                        id="forceDuration${index}"
                        type="number"
                        step="0.1"
                        min="0.01"
                        max="30"
                        value="0.5">

                    <div class="row">
                        <button onclick=
                            "applyForce(${index},1,0,0)">
                            +X
                        </button>

                        <button onclick=
                            "applyForce(${index},-1,0,0)">
                            -X
                        </button>
                    </div>

                    <div class="row">
                        <button onclick=
                            "applyForce(${index},0,1,0)">
                            +Y
                        </button>

                        <button onclick=
                            "applyForce(${index},0,-1,0)">
                            -Y
                        </button>
                    </div>

                    <div class="row">
                        <button onclick=
                            "applyForce(${index},0,0,1)">
                            +Z
                        </button>

                        <button onclick=
                            "applyForce(${index},0,0,-1)">
                            -Z
                        </button>
                    </div>
                  `
                : `
                    <div class="small">
                        Force controls require
                        Physics = dynamic.
                    </div>
                  `
            }
        </div>`;
    }).join("");
    renderObjectDescriptions();
}

async function applyForce(
    index,
    dx,
    dy,
    dz
) {
    try {
        const obj = objectList[index];

        if (!obj) {
            throw new Error(
                "Selected object no longer exists"
            );
        }

        if (
            String(obj.physics || "").toLowerCase()
            !== "dynamic"
        ) {
            throw new Error(
                "Force requires a dynamic object"
            );
        }

        const magnitude =
            Number(
                el(`forceMag${index}`).value
            );

        const duration =
            Number(
                el(`forceDuration${index}`).value
            );

        if (
            !Number.isFinite(magnitude)
            || magnitude < 0
        ) {
            throw new Error(
                "Force magnitude must be >= 0 N"
            );
        }

        if (
            !Number.isFinite(duration)
            || duration <= 0
            || duration > 30
        ) {
            throw new Error(
                "Duration must be > 0 and <= 30 s"
            );
        }

        const force = [
            magnitude * dx,
            magnitude * dy,
            magnitude * dz
        ];

        status(
            `Applying ${JSON.stringify(force)} N `
            + `to ${obj.object_id}...`
        );

        const data = await api(
            "/api/objects/force",
            {
                method: "POST",
                body: JSON.stringify({
                    object_id: obj.object_id,
                    force: force,
                    duration_s: duration
                })
            }
        );

        status(data.message);
    }

    catch (err) {
        status(
            err.message,
            true
        );
    }
}


async function moveObject(index) {
    try {
        const obj = objectList[index];

        const pos = [
            Number(el(`ox${index}`).value),
            Number(el(`oy${index}`).value),
            Number(el(`oz${index}`).value)
        ];

        const data = await api("/api/objects/move", {
            method: "POST",
            body: JSON.stringify({
                object_id: obj.object_id,
                position: pos
            })
        });

        await refreshObjects();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

async function removeObject(index) {
    try {
        const obj = objectList[index];

        const data = await api("/api/objects/remove", {
            method: "POST",
            body: JSON.stringify({
                object_id: obj.object_id
            })
        });

        await refreshObjects();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

let robotList = [];

async function refreshRobots() {
    try {
        const data = await api("/api/robots");
        robotList = data.robots;

        const select = el("robotSelect");
        const previous = select.value;
        select.replaceChildren(...robotList.map(r =>
            new Option(`${r.robot} (${r.model})`, r.robot)
        ));
        select.disabled = robotList.length === 0;
        if (robotList.some(r => r.robot === previous)) {
            select.value = previous;
        }
        selectRobot();

        el("robotCount").textContent =
            `${robotList.length} robots standing`;
    }
    catch (err) {
        status(err.message, true);
    }
}

// The selected robot's own position and yaw fill the boxes, so a move starts
// from where that robot stands, facing as it does.
function selectRobot() {
    const robot = robotList.find(r => r.robot === el("robotSelect").value);
    if (!robot) {
        return;
    }
    const [x, y, z] = robot.position;
    el("robotX").value = x;
    el("robotY").value = y;
    el("robotZ").value = z;
    el("robotYaw").value = robot.yaw;
}

async function moveRobot() {
    try {
        const robot = el("robotSelect").value;

        if (!robot) {
            status("select a robot to move", true);
            return;
        }

        const data = await api("/api/robot/move", {
            method: "POST",
            body: JSON.stringify({
                robot: robot,
                position: position("robot"),
                yaw: number("robotYaw")
            })
        });

        await refreshRobots();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

// ---------------------------------------------------------------------
// Lighting, materials and cameras: what the launch bound beside the scene
// ---------------------------------------------------------------------

// Read once at startup; a card is shown only for a capability that is
// bound and whose provider lists something to edit.
let capabilities = { lighting: false, materials: false, cameras: false };

// A change made from elsewhere (an MCP client, another copy of this page)
// shows within this interval while the tab is visible.
const PANEL_REFRESH_MS = 3000;

// Lighting and materials share one shape: targets with an id, a label and
// properties, each with its unit, its default and its current value. A
// setter names the route a property posts to and the request field it
// fills; the properties of one route are applied together (the two angles
// of a cone, the metallic and roughness of a finish).
//
// A panel's elements: <name>Card, <name>Summary and <name>Targets are in
// the page; a control's inputs are <name><index>_<property>_<component>
// and its effective value <name><index>_<route>_current.
const LIGHTING = {
    name: "lighting",
    idField: "light_id",
    setters: {
        illuminance_lux: { route: "intensity", field: "value" },
        luminous_flux_lm: { route: "intensity", field: "value" },
        radiance_scale: { route: "intensity", field: "value" },
        color_rgb: { route: "color", field: "color" },
        position_m: { route: "position", field: "position" },
        direction: { route: "direction", field: "direction" },
        inner_angle_rad: { route: "cone", field: "inner_angle" },
        outer_angle_rad: { route: "cone", field: "outer_angle" },
        orientation_xyzw: { route: "orientation", field: "orientation" }
    },
    read: data => data.lighting.lights,
    summary: data =>
        `${data.lighting.scene} | ambient ${fmt(data.lighting.ambient_lux)} lx`
        + ` | fill ${fmt(data.lighting.fill_lux)} lx`,
    describe: light => `${light.kind} | ${light.id}`,
    list: [],
    shape: ""
};

const MATERIALS = {
    name: "materials",
    idField: "material_id",
    setters: {
        color_rgb: { route: "color", field: "color" },
        metallic: { route: "finish", field: "metallic" },
        roughness: { route: "finish", field: "roughness" }
    },
    read: data => data.materials.materials,
    summary: data => `${data.materials.materials.length} materials`,
    describe: material => {
        const scope = material.scope || { surfaces: "?", bodies: [] };
        return `shared by ${scope.surfaces} surfaces on: ${(scope.bodies || []).join(", ")}`;
    },
    list: [],
    shape: ""
};

const PANELS = { lighting: LIGHTING, materials: MATERIALS };

// Provider text goes into markup as literal text.
function text(value) {
    return String(value).replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
}

// Numbers as the cards show them.
function fmt(value) {
    if (Array.isArray(value)) {
        return `[${value.map(fmt).join(", ")}]`;
    }
    return String(Number(Number(value).toPrecision(6)));
}

// A target's properties by the route they post to, in the provider's
// order; a property without a setter here is listed but not edited.
function routesOf(target, setters) {
    const routes = new Map();
    for (const name of Object.keys(target.properties || {})) {
        const setter = setters[name];
        if (!setter) continue;
        if (!routes.has(setter.route)) routes.set(setter.route, []);
        routes.get(setter.route).push(name);
    }
    return routes;
}

const GRID = { 2: "two", 3: "row", 4: "four" };

// The min and max attributes of a scalar input, when the provider reports them.
function boundsOf(property) {
    return ["min", "max"]
        .filter(bound => Number.isFinite(property[bound]))
        .map(bound => ` ${bound}="${property[bound]}"`)
        .join("");
}

// Number inputs for one property: one per component of a vector, one for a
// scalar within its bounds.
function inputsFor(prefix, property) {
    const vector = Array.isArray(property.value);
    const values = vector ? property.value : [property.value];
    const bounds = vector ? "" : boundsOf(property);
    const inputs = values.map((value, k) =>
        `<input id="${prefix}_${k}" type="number" step="any"${bounds} value="${Number(value)}">`
    ).join("");
    return `<div class="${GRID[values.length] || ""}">${inputs}</div>`;
}

function readInputs(prefix, property) {
    if (!Array.isArray(property.value)) {
        return number(`${prefix}_0`);
    }
    return property.value.map((_, k) => number(`${prefix}_${k}`));
}

function renderTargets(panel) {
    el(`${panel.name}Targets`).innerHTML = panel.list.map((target, index) => `
    <div class="target">
        <code>${text(target.label || target.id)}</code>
        <div class="small">${text(panel.describe(target))}</div>
        ${[...routesOf(target, panel.setters)].map(([route, names]) => `
        <label>${text(names.map(name => `${name} (${target.properties[name].unit})`).join(", "))}</label>
        ${names.map(name => inputsFor(`${panel.name}${index}_${name}`, target.properties[name])).join("")}
        <div class="two">
            <button onclick="applyProperty('${panel.name}', ${index}, '${route}')">Apply ${route}</button>
            <div class="small" id="${panel.name}${index}_${route}_current"></div>
        </div>`).join("")}
    </div>`).join("");
}

function showValues(panel) {
    panel.list.forEach((target, index) => {
        for (const [route, names] of routesOf(target, panel.setters)) {
            el(`${panel.name}${index}_${route}_current`).textContent = names.map(name => {
                const property = target.properties[name];
                return `${name} ${fmt(property.value)} (default ${fmt(property.default)})`;
            }).join(" | ");
        }
    });
}

async function refreshPanel(panel) {
    let data;
    try {
        data = await api(`/api/${panel.name}`);
    }
    catch (err) {
        // Nothing to edit yet (no scene is loaded): the card waits hidden,
        // keeping what it showed.
        el(`${panel.name}Card`).hidden = true;
        return;
    }

    panel.list = panel.read(data) || [];
    el(`${panel.name}Card`).hidden = panel.list.length === 0;
    el(`${panel.name}Summary`).textContent = panel.summary(data);

    // Inputs keep their edits across a refresh; only a change in what the
    // provider lists rebuilds them.
    const shape = JSON.stringify(
        panel.list.map(target => [target.id, Object.keys(target.properties || {})])
    );
    if (shape !== panel.shape) {
        panel.shape = shape;
        renderTargets(panel);
    }
    showValues(panel);
}

async function applyProperty(name, index, route) {
    const panel = PANELS[name];
    try {
        const target = panel.list[index];
        const body = { [panel.idField]: target.id };
        for (const property of routesOf(target, panel.setters).get(route)) {
            body[panel.setters[property].field] =
                readInputs(`${name}${index}_${property}`, target.properties[property]);
        }

        const data = await api(`/api/${name}/${route}`, {
            method: "POST",
            body: JSON.stringify(body)
        });

        await refreshPanel(panel);

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

async function resetPanel(name) {
    try {
        const data = await api(`/api/${name}/reset`, {
            method: "POST",
            body: "{}"
        });

        await refreshPanel(PANELS[name]);

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

// The camera controls a profile may list, each with the request field its
// setter takes; exposure and white balance carry a mode beside it. A
// camera's elements are camera<index>_info, camera<index>_profile and, per
// supported control, camera<index>_<control>_0 (its value),
// camera<index>_<control>_mode and camera<index>_<control>_current.
const CAMERA_CONTROLS = {
    exposure: { field: "value", modes: true },
    white_balance: { field: "temperature", modes: true },
    gain: { field: "value" },
    brightness: { field: "value" },
    contrast: { field: "value" }
};

let cameras = [];
let cameraShape = "";

// A camera's route: its robot and its camera, each a path segment.
function cameraPath(camera) {
    return `${encodeURIComponent(camera.robot)}/${encodeURIComponent(camera.camera)}`;
}

function supportedControls(camera) {
    const controls = (camera.profile && camera.profile.controls) || {};
    return Object.keys(CAMERA_CONTROLS).filter(name => controls[name] && controls[name].supported);
}

function renderCameras() {
    el("cameraTargets").innerHTML = cameras.map((camera, index) => `
    <div class="target">
        <code>${text(camera.id)}</code>
        <div class="small" id="camera${index}_info"></div>
        <p class="small" id="camera${index}_profile"></p>
        ${supportedControls(camera).map(name => {
            const control = camera.profile.controls[name];
            const modes = CAMERA_CONTROLS[name].modes
                ? `<select id="camera${index}_${name}_mode">${(control.modes || []).map(mode =>
                    `<option value="${text(mode)}"${mode === control.mode ? " selected" : ""}>${text(mode)}</option>`
                ).join("")}</select>`
                : "";
            return `
        <label>${text(`${name} (${control.unit})`)}</label>
        <div class="${modes ? "two" : ""}">
            ${modes}
            <input id="camera${index}_${name}_0" type="number" step="1"${boundsOf(control)} value="${Number(control.value)}">
        </div>
        <div class="two">
            <button onclick="applyCamera(${index}, '${name}')">Apply ${name}</button>
            <div class="small" id="camera${index}_${name}_current"></div>
        </div>`;
        }).join("")}
        ${camera.profile ? `<button class="danger" onclick="resetCamera(${index})">Reset camera</button>` : ""}
    </div>`).join("");
}

function showCameraValues() {
    cameras.forEach((camera, index) => {
        const info = camera.info || {};
        el(`camera${index}_info`).textContent =
            `${camera.kind} | ${info.width}x${info.height} @ ${info.frames_per_second} fps | ${info.encoding}`;
        el(`camera${index}_profile`).textContent = camera.profile_message || "";

        for (const name of supportedControls(camera)) {
            const control = camera.profile.controls[name];
            const withMode = CAMERA_CONTROLS[name].modes;
            el(`camera${index}_${name}_current`).textContent =
                `${withMode ? control.mode + " " : ""}${fmt(control.value)} ${control.unit}`
                + ` (default ${withMode ? control.default_mode + " " : ""}${fmt(control.default_value)})`;
        }
    });
}

async function refreshCameras() {
    let data;
    try {
        data = await api("/api/cameras");
    }
    catch (err) {
        el("camerasCard").hidden = true;
        return;
    }

    cameras = data.cameras || [];
    el("camerasCard").hidden = cameras.length === 0;

    const shape = JSON.stringify(
        cameras.map(camera => [camera.id, camera.kind, supportedControls(camera)])
    );
    if (shape !== cameraShape) {
        cameraShape = shape;
        renderCameras();
    }
    showCameraValues();
}

async function applyCamera(index, name) {
    try {
        const camera = cameras[index];
        const control = CAMERA_CONTROLS[name];
        const body = { [control.field]: number(`camera${index}_${name}_0`) };
        if (control.modes) {
            body.mode = el(`camera${index}_${name}_mode`).value;
        }

        const data = await api(`/api/cameras/${cameraPath(camera)}/${name}`, {
            method: "POST",
            body: JSON.stringify(body)
        });

        await refreshCameras();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

async function resetCamera(index) {
    try {
        const camera = cameras[index];

        const data = await api(`/api/cameras/${cameraPath(camera)}/reset`, {
            method: "POST",
            body: "{}"
        });

        await refreshCameras();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
    }
}

async function loadCapabilities() {
    try {
        const data = await api("/api/capabilities");
        capabilities = {
            lighting: Boolean(data.lighting),
            materials: Boolean(data.materials),
            cameras: Boolean(data.cameras)
        };
    }
    catch (err) {
        status(err.message, true);
    }
}

function anyCapability() {
    return capabilities.lighting || capabilities.materials || capabilities.cameras;
}

async function refreshPanels() {
    const refreshes = [];
    if (capabilities.lighting) refreshes.push(refreshPanel(LIGHTING));
    if (capabilities.materials) refreshes.push(refreshPanel(MATERIALS));
    if (capabilities.cameras) refreshes.push(refreshCameras());
    await Promise.all(refreshes);
}

async function startup() {
    await loadCapabilities();
    await loadCatalogue();
    await refreshObjects();
    await refreshRobots();

    if (anyCapability()) {
        await refreshPanels();
        setInterval(() => {
            if (document.visibilityState === "visible") {
                refreshPanels();
            }
        }, PANEL_REFRESH_MS);
    }
}

startup();
</script>

</body>
</html>
"""


async def _index(_request: web.Request) -> web.Response:
    return web.Response(text=HTML, content_type="text/html")


# ---------------------------------------------------------------------------
# HTTP server lifetime
# ---------------------------------------------------------------------------


def _build_app(
    node_runner: NodeRunner,
    silence: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> web.Application:
    capabilities = _probe_capabilities(node_runner)
    app = web.Application()

    app[_NODE_RUNNER] = node_runner
    app[_SILENCE] = silence
    app[_STREAMED_GOALS] = _StreamedGoals()
    app[_CAPABILITIES] = capabilities
    app[_CATALOGUE] = _StateWatch("catalogue", "assets")
    app[_OBJECT_STATE] = _StateWatch("object state", "runtime objects")
    app[_LIGHTING] = _StateWatch("lighting", "lights")
    app[_MATERIALS] = _StateWatch("materials", "materials")
    app[_PROFILES] = {}

    app.router.add_get("/", _index)
    app.router.add_get("/api/health", _api_health)
    app.router.add_get("/api/capabilities", _api_capabilities)
    app.router.add_get("/api/assets", _api_assets)
    app.router.add_get("/api/objects", _api_objects)
    app.router.add_get("/api/robots", _api_robots)
    app.router.add_post("/api/scene/load", _api_load_scene)
    app.router.add_post("/api/scene/clear", _api_clear_scene)
    app.router.add_post("/api/objects/spawn", _api_spawn_object)
    app.router.add_post("/api/goals/{goal}/cancel", _api_cancel_goal)
    app.router.add_post("/api/objects/force", _api_apply_force)
    app.router.add_post("/api/objects/move", _api_move_object)
    app.router.add_post("/api/objects/remove", _api_remove_object)
    app.router.add_post("/api/robot/move", _api_move_robot)
    for panel in _PANELS:
        app.router.add_get(f"/api/{panel.name}", partial(_api_panel, panel))
        app.router.add_post(f"/api/{panel.name}/reset", partial(_api_reset_panel, panel))
        app.router.add_post(f"/api/{panel.name}/{{property}}", partial(_api_set_panel, panel))
    app.router.add_get("/api/cameras", _api_cameras)
    app.router.add_post("/api/cameras/{robot}/{camera}/reset", _api_reset_camera)
    app.router.add_post("/api/cameras/{robot}/{camera}/{control}", _api_set_camera)

    return app


# ---------------------------------------------------------------------------
# Peppy entry point
# ---------------------------------------------------------------------------


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    logger.info("Scene commander starting")

    # The scene panel's socket, owned before the provider is waited on: a
    # launch this node cannot serve is refused first, and which copy holds the
    # preferred port follows launch order.
    listener = listen.bind_listener(params.http_host, params.http_port)

    # The daemon renders the URLs an operator opens from this announcement,
    # one per address of the machine, so the fallback port reaches the
    # operator too.
    host, port = listen.bound_host_and_port(listener)
    node_runner.announce_endpoint("panel", "http", host, port)
    logger.info("Scene panel bound at %s", listen.bound_address(listener))

    app = _build_app(node_runner)

    # What the launch bound beside the scene is read from the slots alone;
    # each provider is called only once the page asks for it.
    logger.info("Capabilities: %s", app[_CAPABILITIES].summary())

    # The provider must answer both reads; whether it has its catalogue and
    # its object state yet is a state the page asks for again.
    try:
        assets = await _fetch_assets(node_runner)

    except SceneCatalogueUnavailable as exc:
        app[_CATALOGUE].unavailable(str(exc))

    else:
        app[_CATALOGUE].ready(len(assets))

    try:
        snapshot = await _fetch_objects(node_runner)

    except ObjectStateUnavailable as exc:
        app[_OBJECT_STATE].unavailable(str(exc))

    else:
        app[_OBJECT_STATE].ready(len(snapshot["objects"]))

    server_task = await listen.start_serving(app, listener)

    return [server_task]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
