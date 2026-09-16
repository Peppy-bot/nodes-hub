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
import json
import logging
import math
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Callable

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

from peppygen.consumed_services import color_cameras, rgbd_cameras

from peppygen.consumed_services.camera_profiles import get_camera_profile, reset_camera

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

from peppygen.consumed_services.simulation import get_assets_list

from scene_commander import listen

if TYPE_CHECKING:
    from peppygen.parameters import Parameters


logger = logging.getLogger(__name__)

SERVICE_TIMEOUT_S = 10.0
ACTION_TIMEOUT_S = 60.0


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


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


async def _fetch_assets(node_runner: NodeRunner) -> list[dict]:
    producer = get_assets_list.bound_producer(node_runner)

    response = await get_assets_list.poll(
        node_runner,
        producer,
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
class _Camera:
    """One camera relay bound in this launch, with the profile bound for the
    same instance when the launch linked both here."""

    id: str
    kind: str
    producer: peppylib.ProducerRef
    profile: peppylib.ProducerRef | None


@dataclass(frozen=True)
class _Capabilities:
    """What the launch bound beside the scene, fixed when the node starts.

    A vacant slot is a capability the page never asks for. A camera is known
    by the instance id of the relay serving it, which is also what matches
    it to its profile.
    """

    lighting: peppylib.ProducerRef | None
    materials: peppylib.ProducerRef | None
    cameras: tuple[_Camera, ...]

    def camera(self, camera_id: str) -> _Camera:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera

        raise CapabilityUnbound(f"no camera {camera_id} is bound in this launch")

    def summary(self) -> str:
        parts = []

        if self.lighting is not None:
            parts.append("lighting")

        if self.materials is not None:
            parts.append("materials")

        if self.cameras:
            parts.append(
                "cameras: "
                + ", ".join(f"{camera.id} ({camera.kind})" for camera in self.cameras)
            )

        return ", ".join(parts) or "scene manipulation only"


# The stream description of each camera kind; the instance answering it is
# the one answering the kind's control setters.
_STREAM_INFO = {
    "rgb": color_cameras.video_stream_info,
    "rgbd": rgbd_cameras.video_stream_info,
}


def _probe_capabilities(node_runner: NodeRunner) -> _Capabilities:
    """Read the optional slots as the launch bound them, calling no provider.

    A camera's profile is the camera_profile producer of the same instance:
    a camera linked here without one keeps its stream and its controls, with
    nothing describing them and no reset.
    """

    profiles = get_camera_profile.bound_producers(node_runner)
    cameras = []

    for kind, stream_info in _STREAM_INFO.items():
        for producer in stream_info.bound_producers(node_runner):
            profile = next((candidate for candidate in profiles if candidate == producer), None)
            cameras.append(_Camera(producer.instance_id, kind, producer, profile))

    return _Capabilities(
        lighting=get_lighting.bound_producer(node_runner),
        materials=get_materials.bound_producer(node_runner),
        cameras=tuple(cameras),
    )


# ---------------------------------------------------------------------------
# Optional providers
# ---------------------------------------------------------------------------


async def _call_service(service, node_runner: NodeRunner, producer, request=None, **context):
    """Poll one service on producer and return its response data.

    Every call that succeeds leaves one log line naming what was asked and
    what the provider answered. A refusal raises with the provider's reason
    and the effective values it reported; the HTTP layer logs it and answers
    with both. context names what the request itself does not (the camera
    a control is set on).
    """

    name = service.__name__.rsplit(".", 1)[-1]
    fields = {**context, **(vars(request) if request is not None else {})}
    summary = f"{name}({', '.join(f'{key}={value}' for key, value in fields.items())})"
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


async def _fetch_lighting(node_runner: NodeRunner, producer) -> dict:
    """The provider's lighting: the loaded scene, its ambient and fill
    levels, and every light with its properties."""

    response = await get_lighting.poll(node_runner, producer, timeout=SERVICE_TIMEOUT_S)
    data = response.data

    if not data.success:
        raise StateUnavailable(data.message)

    return json.loads(data.lighting_json)


async def _fetch_materials(node_runner: NodeRunner, producer) -> dict:
    """The provider's materials, each with its scope and its properties."""

    response = await get_materials.poll(node_runner, producer, timeout=SERVICE_TIMEOUT_S)
    data = response.data

    if not data.success:
        raise StateUnavailable(data.message)

    return json.loads(data.materials_json)


async def _fetch_stream_info(node_runner: NodeRunner, camera: _Camera) -> dict:
    response = await _STREAM_INFO[camera.kind].poll(
        node_runner,
        camera.producer,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    return {
        "width": data.width,
        "height": data.height,
        "frames_per_second": data.frames_per_second,
        "encoding": data.encoding,
    }


async def _fetch_profile(node_runner: NodeRunner, producer) -> tuple[dict, str]:
    """A camera's profile and the message it came with."""

    response = await get_camera_profile.poll(node_runner, producer, timeout=SERVICE_TIMEOUT_S)
    data = response.data

    if not data.success:
        raise StateUnavailable(data.message)

    return json.loads(data.profile_json), data.message


# ---------------------------------------------------------------------------
# Setter requests
# ---------------------------------------------------------------------------
#
# Each builder checks the page's payload in full before any provider is
# called: a value that is not a finite number, a vector of the wrong length
# or a mode the contract does not name is refused here.


def _light_intensity(payload: dict):
    return set_light_intensity.Request(
        light_id=_name(payload, "light_id"),
        value=_number(payload, "value"),
    )


def _light_color(payload: dict):
    return set_light_color.Request(
        light_id=_name(payload, "light_id"),
        color=_vector(payload, "color", 3),
    )


def _light_position(payload: dict):
    return set_light_position.Request(
        light_id=_name(payload, "light_id"),
        position=_vector(payload, "position", 3),
    )


def _light_direction(payload: dict):
    return set_light_direction.Request(
        light_id=_name(payload, "light_id"),
        direction=_vector(payload, "direction", 3),
    )


def _light_cone(payload: dict):
    return set_light_cone.Request(
        light_id=_name(payload, "light_id"),
        inner_angle=_number(payload, "inner_angle"),
        outer_angle=_number(payload, "outer_angle"),
    )


def _light_orientation(payload: dict):
    return set_light_orientation.Request(
        light_id=_name(payload, "light_id"),
        orientation=_vector(payload, "orientation", 4),
    )


# The lighting setters by the route the page posts to.
_LIGHT_SETTERS = {
    "intensity": (set_light_intensity, _light_intensity),
    "color": (set_light_color, _light_color),
    "position": (set_light_position, _light_position),
    "direction": (set_light_direction, _light_direction),
    "cone": (set_light_cone, _light_cone),
    "orientation": (set_light_orientation, _light_orientation),
}


def _material_color(payload: dict):
    return set_material_color.Request(
        material_id=_name(payload, "material_id"),
        color=_vector(payload, "color", 3),
    )


def _material_finish(payload: dict):
    return set_material_finish.Request(
        material_id=_name(payload, "material_id"),
        metallic=_number(payload, "metallic"),
        roughness=_number(payload, "roughness"),
    )


# The material setters by the route the page posts to.
_MATERIAL_SETTERS = {
    "color": (set_material_color, _material_color),
    "finish": (set_material_finish, _material_finish),
}


@dataclass(frozen=True)
class _CameraControl:
    """One control of the camera contracts: the setter of each camera kind
    and the request it takes, built from the page's payload."""

    rgb: ModuleType
    rgbd: ModuleType
    request: Callable[[ModuleType, dict], object]

    def service(self, kind: str) -> ModuleType:
        return self.rgb if kind == "rgb" else self.rgbd


def _exposure(service: ModuleType, payload: dict):
    return service.Request(mode=_mode(payload), value=_whole(payload, "value"))


def _white_balance(service: ModuleType, payload: dict):
    return service.Request(mode=_mode(payload), temperature=_whole(payload, "temperature"))


def _level(service: ModuleType, payload: dict):
    return service.Request(value=_whole(payload, "value"))


# The camera controls by the route the page posts to. An rgb_camera names
# its setters plainly; an rgbd_camera names them after its colour stream.
_CAMERA_CONTROLS = {
    "exposure": _CameraControl(
        color_cameras.set_exposure, rgbd_cameras.set_color_exposure, _exposure
    ),
    "white_balance": _CameraControl(
        color_cameras.set_white_balance, rgbd_cameras.set_color_white_balance, _white_balance
    ),
    "gain": _CameraControl(color_cameras.set_gain, rgbd_cameras.set_color_gain, _level),
    "brightness": _CameraControl(
        color_cameras.set_brightness, rgbd_cameras.set_color_brightness, _level
    ),
    "contrast": _CameraControl(
        color_cameras.set_contrast, rgbd_cameras.set_color_contrast, _level
    ),
}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


async def _run_action(action, node_runner: NodeRunner, request=None, **fields):
    """Fire one scene_manipulation goal and return its result data.

    Every goal that completes leaves one log line naming what was asked and
    what the provider answered. Rejections and failures raise with the
    provider's reason; the HTTP layer logs them.
    """

    name = action.__name__.rsplit(".", 1)[-1]
    summary = f"{name}({', '.join(f'{key}={value}' for key, value in fields.items())})"

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

    result = await handle.get_result(timeout=ACTION_TIMEOUT_S)

    if result.status != action.ResultStatus.COMPLETED:
        raise RuntimeError(f"{name} did not complete: {result.status.name}")

    if result.data is None:
        raise RuntimeError(f"{name} completed without result data")

    if not result.data.success:
        raise RuntimeError(result.data.message)

    logger.info("%s: %s", summary, result.data.message)

    return result.data


async def _action_load_scene(node_runner: NodeRunner, asset_id: str, scale: float) -> dict:
    data = await _run_action(
        load_scene,
        node_runner,
        load_scene.GoalRequest(asset_id=asset_id, scale=float(scale)),
        asset_id=asset_id,
        scale=float(scale),
    )

    return {"success": True, "message": data.message}


async def _action_clear_scene(node_runner: NodeRunner) -> dict:
    data = await _run_action(clear_scene, node_runner)

    return {"success": True, "message": data.message}


async def _action_spawn_object(node_runner: NodeRunner, payload: dict) -> dict:
    request = spawn_object.GoalRequest(
        asset_id=str(payload["asset_id"]),
        position=[float(value) for value in payload["position"]],
        scale=float(payload.get("scale", 1.0)),
        physics=str(payload.get("physics", "none")),
        mass=float(payload.get("mass", 0.1)),
    )

    data = await _run_action(
        spawn_object,
        node_runner,
        request,
        asset_id=request.asset_id,
        position=request.position,
        physics=request.physics,
        mass=request.mass,
    )

    return {"success": True, "message": data.message, "object_id": data.object_id}


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
    request = move_object.GoalRequest(
        object_id=str(payload["object_id"]),
        position=[float(value) for value in payload["position"]],
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


async def _action_move_robot(node_runner: NodeRunner, position: list[float]) -> dict:
    position = [float(value) for value in position]

    data = await _run_action(
        move_robot,
        node_runner,
        move_robot.GoalRequest(position=position),
        position=position,
    )

    return {"success": True, "message": data.message}


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------


_NODE_RUNNER = web.AppKey("node_runner", NodeRunner)
_CAPABILITIES = web.AppKey("capabilities", _Capabilities)
_CATALOGUE = web.AppKey("catalogue", _StateWatch)
_OBJECT_STATE = web.AppKey("object_state", _StateWatch)
_LIGHTING = web.AppKey("lighting", _StateWatch)
_MATERIALS = web.AppKey("materials", _StateWatch)
# One profile watch per bound camera, by camera id.
_PROFILES = web.AppKey("profiles", dict)


def _json_error(request: web.Request, exc: Exception, status: int = 400) -> web.Response:
    # Bad input and provider refusals are one line each; anything else is a
    # bug in this node and keeps its traceback.
    expected = isinstance(exc, (ValueError, KeyError, RuntimeError, TimeoutError))

    logger.warning(
        "%s %s failed: %s",
        request.method,
        request.path,
        exc,
        exc_info=None if expected else exc,
    )

    body = {"success": False, "message": str(exc)}

    # A refused setter answers with what stands, as the provider reported it.
    if isinstance(exc, ProviderRefusal):
        body.update(exc.current)

    return web.json_response(body, status=status)


async def _request_json(request: web.Request) -> dict:
    try:
        data = await request.json()

    except Exception as exc:
        raise ValueError("Request body must contain valid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON request body must be an object")

    return data


def _position(payload: dict) -> list[float]:
    position = payload.get("position")

    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("position must be [x, y, z]")

    return [float(value) for value in position]


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


def _mode(payload: dict) -> str:
    mode = payload.get("mode")

    if mode not in ("auto", "manual"):
        raise ValueError('mode must be "auto" or "manual"')

    return mode


def _name(payload: dict, key: str) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")

    return value


def _lighting_producer(app: web.Application):
    producer = app[_CAPABILITIES].lighting

    if producer is None:
        raise CapabilityUnbound("lighting is not bound in this launch")

    return producer


def _materials_producer(app: web.Application):
    producer = app[_CAPABILITIES].materials

    if producer is None:
        raise CapabilityUnbound("materials are not bound in this launch")

    return producer


def _profile_producer(camera: _Camera):
    if camera.profile is None:
        raise CapabilityUnbound(f"no camera profile is bound for {camera.id} in this launch")

    return camera.profile


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


async def _api_load_scene(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)

        result = await _action_load_scene(
            request.app[_NODE_RUNNER],
            str(payload["asset_id"]),
            float(payload.get("scale", 1.0)),
        )

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


async def _api_clear_scene(request: web.Request) -> web.Response:
    try:
        result = await _action_clear_scene(request.app[_NODE_RUNNER])

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


async def _api_spawn_object(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)
        payload["position"] = _position(payload)

        result = await _action_spawn_object(request.app[_NODE_RUNNER], payload)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(result)


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
        payload["position"] = _position(payload)

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


async def _api_move_robot(request: web.Request) -> web.Response:
    try:
        payload = await _request_json(request)

        result = await _action_move_robot(
            request.app[_NODE_RUNNER],
            _position(payload),
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
            "cameras": [
                {"id": camera.id, "kind": camera.kind, "profile": camera.profile is not None}
                for camera in capabilities.cameras
            ],
        }
    )


async def _api_lighting(request: web.Request) -> web.Response:
    watch = request.app[_LIGHTING]

    try:
        lighting = await _fetch_lighting(
            request.app[_NODE_RUNNER],
            _lighting_producer(request.app),
        )

        count = len(lighting["lights"])

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

    return web.json_response({"success": True, "lighting": lighting})


async def _api_set_lighting(request: web.Request) -> web.Response:
    setter = _LIGHT_SETTERS.get(request.match_info["property"])

    if setter is None:
        return _json_error(
            request,
            ValueError(f"unknown lighting property: {request.match_info['property']}"),
            status=404,
        )

    service, build = setter

    try:
        payload = await _request_json(request)

        data = await _call_service(
            service,
            request.app[_NODE_RUNNER],
            _lighting_producer(request.app),
            build(payload),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _api_reset_lighting(request: web.Request) -> web.Response:
    try:
        data = await _call_service(
            reset_lighting,
            request.app[_NODE_RUNNER],
            _lighting_producer(request.app),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _api_materials(request: web.Request) -> web.Response:
    watch = request.app[_MATERIALS]

    try:
        materials = await _fetch_materials(
            request.app[_NODE_RUNNER],
            _materials_producer(request.app),
        )

        count = len(materials["materials"])

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

    return web.json_response({"success": True, "materials": materials})


async def _api_set_material(request: web.Request) -> web.Response:
    setter = _MATERIAL_SETTERS.get(request.match_info["property"])

    if setter is None:
        return _json_error(
            request,
            ValueError(f"unknown material property: {request.match_info['property']}"),
            status=404,
        )

    service, build = setter

    try:
        payload = await _request_json(request)

        data = await _call_service(
            service,
            request.app[_NODE_RUNNER],
            _materials_producer(request.app),
            build(payload),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _api_reset_materials(request: web.Request) -> web.Response:
    try:
        data = await _call_service(
            reset_materials,
            request.app[_NODE_RUNNER],
            _materials_producer(request.app),
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _describe_camera(app: web.Application, camera: _Camera) -> dict:
    """One camera as the page lists it: its stream, and its profile when a
    profile producer is bound for it and answers."""

    node_runner = app[_NODE_RUNNER]
    info = await _fetch_stream_info(node_runner, camera)

    if camera.profile is None:
        profile = None
        message = f"no camera profile is bound for {camera.id} in this launch"

    else:
        watch = app[_PROFILES][camera.id]

        try:
            profile, message = await _fetch_profile(node_runner, camera.profile)

        except StateUnavailable as exc:
            watch.unavailable(str(exc))
            profile = None
            message = str(exc)

        else:
            watch.ready(
                sum(1 for control in profile["controls"].values() if control.get("supported"))
            )

    return {
        "id": camera.id,
        "kind": camera.kind,
        "info": info,
        "profile": profile,
        "profile_message": message,
    }


async def _api_cameras(request: web.Request) -> web.Response:
    try:
        cameras = [
            await _describe_camera(request.app, camera)
            for camera in request.app[_CAPABILITIES].cameras
        ]

    except Exception as exc:
        return _json_error(request, exc, status=500)

    return web.json_response({"success": True, "cameras": cameras, "count": len(cameras)})


async def _api_set_camera(request: web.Request) -> web.Response:
    control = _CAMERA_CONTROLS.get(request.match_info["control"])

    if control is None:
        return _json_error(
            request,
            ValueError(f"unknown camera control: {request.match_info['control']}"),
            status=404,
        )

    try:
        camera = request.app[_CAPABILITIES].camera(request.match_info["camera_id"])
        service = control.service(camera.kind)
        payload = await _request_json(request)

        data = await _call_service(
            service,
            request.app[_NODE_RUNNER],
            camera.producer,
            control.request(service, payload),
            camera=camera.id,
        )

    except CapabilityUnbound as exc:
        return _json_error(request, exc, status=404)

    except Exception as exc:
        return _json_error(request, exc)

    return web.json_response(_result(data))


async def _api_reset_camera(request: web.Request) -> web.Response:
    try:
        camera = request.app[_CAPABILITIES].camera(request.match_info["camera_id"])

        data = await _call_service(
            reset_camera,
            request.app[_NODE_RUNNER],
            _profile_producer(camera),
            camera=camera.id,
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

<label>Position</label>
<div class="row">
<input id="robotX" type="number" step="0.05" value="0">
<input id="robotY" type="number" step="0.05" value="0">
<input id="robotZ" type="number" step="0.05" value="0">
</div>

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

async function loadScene() {
    try {
        status("Loading scene...");

        const data = await api("/api/scene/load", {
            method: "POST",
            body: JSON.stringify({
                asset_id: el("sceneSelect").value,
                scale: number("sceneScale")
            })
        });

        await refreshObjects();
        await refreshPanels();

        status(data.message);
    }
    catch (err) {
        status(err.message, true);
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

        const data = await api("/api/objects/spawn", {
            method: "POST",
            body: JSON.stringify({
                asset_id: assetId,
                position: position("spawn"),
                scale: number("spawnScale"),
                physics: el("spawnPhysics").value,
                mass: number("spawnMass")
            })
        });

        await refreshObjects();

        status(
            `${data.message}\nobject_id=${data.object_id}`
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

async function moveRobot() {
    try {
        const data = await api("/api/robot/move", {
            method: "POST",
            body: JSON.stringify({
                position: position("robot")
            })
        });

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
let capabilities = { lighting: false, materials: false, cameras: [] };

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

        const data = await api(`/api/cameras/${encodeURIComponent(camera.id)}/${name}`, {
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

        const data = await api(`/api/cameras/${encodeURIComponent(camera.id)}/reset`, {
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
            cameras: data.cameras || []
        };
    }
    catch (err) {
        status(err.message, true);
    }
}

function anyCapability() {
    return capabilities.lighting || capabilities.materials || capabilities.cameras.length > 0;
}

async function refreshPanels() {
    if (capabilities.lighting) await refreshPanel(LIGHTING);
    if (capabilities.materials) await refreshPanel(MATERIALS);
    if (capabilities.cameras.length) await refreshCameras();
}

async function startup() {
    await loadCapabilities();
    await loadCatalogue();
    await refreshObjects();

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


def _build_app(node_runner: NodeRunner) -> web.Application:
    capabilities = _probe_capabilities(node_runner)
    app = web.Application()

    app[_NODE_RUNNER] = node_runner
    app[_CAPABILITIES] = capabilities
    app[_CATALOGUE] = _StateWatch("catalogue", "assets")
    app[_OBJECT_STATE] = _StateWatch("object state", "runtime objects")
    app[_LIGHTING] = _StateWatch("lighting", "lights")
    app[_MATERIALS] = _StateWatch("materials", "materials")
    app[_PROFILES] = {
        camera.id: _StateWatch("profile", "controls", provider=f"Camera {camera.id}")
        for camera in capabilities.cameras
    }

    app.router.add_get("/", _index)
    app.router.add_get("/api/health", _api_health)
    app.router.add_get("/api/capabilities", _api_capabilities)
    app.router.add_get("/api/assets", _api_assets)
    app.router.add_get("/api/objects", _api_objects)
    app.router.add_post("/api/scene/load", _api_load_scene)
    app.router.add_post("/api/scene/clear", _api_clear_scene)
    app.router.add_post("/api/objects/spawn", _api_spawn_object)
    app.router.add_post("/api/objects/force", _api_apply_force)
    app.router.add_post("/api/objects/move", _api_move_object)
    app.router.add_post("/api/objects/remove", _api_remove_object)
    app.router.add_post("/api/robot/move", _api_move_robot)
    app.router.add_get("/api/lighting", _api_lighting)
    app.router.add_post("/api/lighting/reset", _api_reset_lighting)
    app.router.add_post("/api/lighting/{property}", _api_set_lighting)
    app.router.add_get("/api/materials", _api_materials)
    app.router.add_post("/api/materials/reset", _api_reset_materials)
    app.router.add_post("/api/materials/{property}", _api_set_material)
    app.router.add_get("/api/cameras", _api_cameras)
    app.router.add_post("/api/cameras/{camera_id}/reset", _api_reset_camera)
    app.router.add_post("/api/cameras/{camera_id}/{control}", _api_set_camera)

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

    logger.info(
        "Scene panel at %s (bound %s)",
        listen.served_url(listener),
        listen.bound_address(listener),
    )

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
