#!/usr/bin/env python3
"""Web commander for OpenArm Isaac scenes through Peppy APIs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

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

from peppygen.consumed_services.simulation import (
    get_assets_list,
    get_objects_list,
)


logger = logging.getLogger(__name__)

SERVICE_TIMEOUT_S = 10.0
ACTION_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def _param(
    params: Any,
    name: str,
    default: Any,
) -> Any:
    """Read a Peppy parameter from dict-like or attribute-like params."""

    if params is None:
        return default

    if isinstance(params, dict):
        return params.get(
            name,
            default,
        )

    value = getattr(
        params,
        name,
        default,
    )

    return value


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


async def _fetch_assets(
    node_runner: NodeRunner,
) -> list[dict]:
    producer = get_assets_list.bound_producer(
        node_runner
    )

    response = await get_assets_list.poll(
        node_runner,
        producer,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise RuntimeError(
            data.message
        )

    return json.loads(
        data.assets_json
    )


async def _fetch_objects(
    node_runner: NodeRunner,
) -> list[dict]:
    producer = get_objects_list.bound_producer(
        node_runner
    )

    response = await get_objects_list.poll(
        node_runner,
        producer,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise RuntimeError(
            data.message
        )

    return json.loads(
        data.objects_json
    )


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _require_completed(
    module,
    result,
):
    if result.status != module.ResultStatus.COMPLETED:
        raise RuntimeError(
            f"Action did not complete: {result.status.name}"
        )

    if result.data is None:
        raise RuntimeError(
            "Action completed without result data"
        )

    if not result.data.success:
        raise RuntimeError(
            result.data.message
        )

    return result.data


async def _action_load_scene(
    node_runner: NodeRunner,
    asset_id: str,
    scale: float,
) -> dict:
    producer = load_scene.bound_producer(
        node_runner
    )

    request = load_scene.GoalRequest(
        asset_id=asset_id,
        scale=float(scale),
    )

    handle = await load_scene.ActionHandle.fire_goal(
        node_runner,
        producer,
        request,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"load_scene rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        load_scene,
        result,
    )

    return {
        "success": True,
        "message": data.message,
    }


async def _action_clear_scene(
    node_runner: NodeRunner,
) -> dict:
    producer = clear_scene.bound_producer(
        node_runner
    )

    handle = await clear_scene.ActionHandle.fire_goal(
        node_runner,
        producer,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"clear_scene rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        clear_scene,
        result,
    )

    return {
        "success": True,
        "message": data.message,
    }


async def _action_spawn_object(
    node_runner: NodeRunner,
    payload: dict,
) -> dict:
    producer = spawn_object.bound_producer(
        node_runner
    )

    request = spawn_object.GoalRequest(
        asset_id=str(
            payload["asset_id"]
        ),
        position=[
            float(value)
            for value in payload["position"]
        ],
        scale=float(
            payload.get(
                "scale",
                1.0,
            )
        ),
        physics=str(
            payload.get(
                "physics",
                "none",
            )
        ),
        mass=float(
            payload.get(
                "mass",
                0.1,
            )
        ),
    )

    handle = await spawn_object.ActionHandle.fire_goal(
        node_runner,
        producer,
        request,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"spawn_object rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        spawn_object,
        result,
    )

    return {
        "success": True,
        "message": data.message,
        "object_id": data.object_id,
    }


async def _action_apply_force(
    node_runner: NodeRunner,
    payload: dict,
) -> dict:
    producer = apply_force.bound_producer(
        node_runner
    )

    force = [
        float(value)
        for value in payload["force"]
    ]

    if len(force) != 3:
        raise ValueError(
            "force must contain exactly 3 values"
        )

    request = apply_force.GoalRequest(
        object_id=str(
            payload["object_id"]
        ),
        force=force,
        duration_s=float(
            payload.get(
                "duration_s",
                0.5,
            )
        ),
    )

    handle = await apply_force.ActionHandle.fire_goal(
        node_runner,
        producer,
        request,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"apply_force rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        apply_force,
        result,
    )

    return {
        "success": True,
        "message": data.message,
    }


async def _action_move_object(
    node_runner: NodeRunner,
    payload: dict,
) -> dict:
    producer = move_object.bound_producer(
        node_runner
    )

    request = move_object.GoalRequest(
        object_id=str(
            payload["object_id"]
        ),
        position=[
            float(value)
            for value in payload["position"]
        ],
    )

    handle = await move_object.ActionHandle.fire_goal(
        node_runner,
        producer,
        request,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"move_object rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        move_object,
        result,
    )

    return {
        "success": True,
        "message": data.message,
    }


async def _action_remove_object(
    node_runner: NodeRunner,
    object_id: str,
) -> dict:
    producer = remove_object.bound_producer(
        node_runner
    )

    request = remove_object.GoalRequest(
        object_id=object_id,
    )

    handle = await remove_object.ActionHandle.fire_goal(
        node_runner,
        producer,
        request,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"remove_object rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        remove_object,
        result,
    )

    return {
        "success": True,
        "message": data.message,
    }


async def _action_move_robot(
    node_runner: NodeRunner,
    position: list[float],
) -> dict:
    producer = move_robot.bound_producer(
        node_runner
    )

    request = move_robot.GoalRequest(
        position=[
            float(value)
            for value in position
        ],
    )

    handle = await move_robot.ActionHandle.fire_goal(
        node_runner,
        producer,
        request,
        timeout=ACTION_TIMEOUT_S,
        feedback_qos=peppylib.QoSProfile.Standard,
    )

    if not handle.accepted:
        raise RuntimeError(
            f"move_robot rejected: {handle.reason}"
        )

    result = await handle.get_result(
        timeout=ACTION_TIMEOUT_S
    )

    data = _require_completed(
        move_robot,
        result,
    )

    return {
        "success": True,
        "message": data.message,
    }


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------


def _json_error(
    exc: Exception,
    status: int = 400,
) -> web.Response:
    logger.exception(
        "Scene commander request failed"
    )

    return web.json_response(
        {
            "success": False,
            "message": str(exc),
        },
        status=status,
    )


async def _request_json(
    request: web.Request,
) -> dict:
    try:
        data = await request.json()

    except Exception as exc:
        raise ValueError(
            "Request body must contain valid JSON"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "JSON request body must be an object"
        )

    return data


def _position(
    payload: dict,
) -> list[float]:
    position = payload.get(
        "position"
    )

    if (
        not isinstance(position, list)
        or len(position) != 3
    ):
        raise ValueError(
            "position must be [x, y, z]"
        )

    return [
        float(value)
        for value in position
    ]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


async def _api_health(
    request: web.Request,
) -> web.Response:
    return web.json_response(
        {
            "success": True,
            "service": "openarm_scene_commander",
        }
    )


async def _api_assets(
    request: web.Request,
) -> web.Response:
    try:
        node_runner = request.app["node_runner"]

        assets = await _fetch_assets(
            node_runner
        )

        return web.json_response(
            {
                "success": True,
                "assets": assets,
                "count": len(assets),
            }
        )

    except Exception as exc:
        return _json_error(
            exc,
            status=500,
        )


async def _api_objects(
    request: web.Request,
) -> web.Response:
    try:
        node_runner = request.app["node_runner"]

        objects = await _fetch_objects(
            node_runner
        )

        return web.json_response(
            {
                "success": True,
                "objects": objects,
                "count": len(objects),
            }
        )

    except Exception as exc:
        return _json_error(
            exc,
            status=500,
        )


async def _api_load_scene(
    request: web.Request,
) -> web.Response:
    try:
        payload = await _request_json(
            request
        )

        result = await _action_load_scene(
            request.app["node_runner"],
            str(payload["asset_id"]),
            float(
                payload.get(
                    "scale",
                    1.0,
                )
            ),
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


async def _api_clear_scene(
    request: web.Request,
) -> web.Response:
    try:
        result = await _action_clear_scene(
            request.app["node_runner"]
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


async def _api_spawn_object(
    request: web.Request,
) -> web.Response:
    try:
        payload = await _request_json(
            request
        )

        payload["position"] = _position(
            payload
        )

        result = await _action_spawn_object(
            request.app["node_runner"],
            payload,
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


async def _api_apply_force(
    request: web.Request,
) -> web.Response:
    try:
        payload = await _request_json(
            request
        )

        force = payload.get(
            "force",
            [],
        )

        if (
            not isinstance(force, list)
            or len(force) != 3
        ):
            raise ValueError(
                "force must contain exactly "
                "3 values [Fx, Fy, Fz]"
            )

        payload["force"] = [
            float(value)
            for value in force
        ]

        payload["duration_s"] = float(
            payload.get(
                "duration_s",
                0.5,
            )
        )

        result = await _action_apply_force(
            request.app["node_runner"],
            payload,
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


async def _api_move_object(
    request: web.Request,
) -> web.Response:
    try:
        payload = await _request_json(
            request
        )

        payload["position"] = _position(
            payload
        )

        result = await _action_move_object(
            request.app["node_runner"],
            payload,
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


async def _api_remove_object(
    request: web.Request,
) -> web.Response:
    try:
        payload = await _request_json(
            request
        )

        result = await _action_remove_object(
            request.app["node_runner"],
            str(payload["object_id"]),
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


async def _api_move_robot(
    request: web.Request,
) -> web.Response:
    try:
        payload = await _request_json(
            request
        )

        result = await _action_move_robot(
            request.app["node_runner"],
            _position(payload),
        )

        return web.json_response(
            result
        )

    except Exception as exc:
        return _json_error(exc)


# ---------------------------------------------------------------------------
# Browser UI
# ---------------------------------------------------------------------------


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenArm Isaac Scene Commander</title>

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

#status {
    margin: 0 16px 16px;
    padding: 12px;
    border-radius: 8px;
    background: #171c23;
    border: 1px solid #303741;
    white-space: pre-wrap;
}

.object {
    border-top: 1px solid #303741;
    margin-top: 12px;
    padding-top: 12px;
}

.object code {
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
</style>
</head>

<body>

<header>
<h1>OpenArm Isaac Scene Commander</h1>
<p>Peppy-native runtime scene and object control</p>
</header>

<main>

<section class="card">
<h2>Scene</h2>

<label>Scene</label>
<select id="sceneSelect"></select>

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
<select id="categorySelect" onchange="renderAssets()"></select>

<label>Asset</label>
<select id="assetSelect"></select>

<div id="assetCount"></div>

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
<option value="dynamic">Dynamic</option>
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

</main>

<div id="status">Connecting...</div>


<script>
let assets = [];
let objectList = [];

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

    const scenes = assets.filter(a => a.kind === "scene");
    const props = assets.filter(a => a.kind === "object");

    el("sceneSelect").innerHTML = scenes.map(a =>
        `<option value="${a.asset_id}">${a.display_name}</option>`
    ).join("");

    const categories = [...new Set(
        props.map(a => a.category).filter(Boolean)
    )].sort();

    el("categorySelect").innerHTML =
        `<option value="">All categories</option>` +
        categories.map(c =>
            `<option value="${c}">${c}</option>`
        ).join("");

    renderAssets();

    status(`Loaded ${assets.length} Isaac assets`);
}

function renderAssets() {
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
                (a.category || "").toLowerCase().includes(search)
            );
        });

    el("assetSelect").innerHTML = filtered.map(a =>
        `<option value="${a.asset_id}">
            ${a.display_name} — ${a.category}
        </option>`
    ).join("");

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
        status(err.message, true);
    }
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
            </div>

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

async function startup() {
    try {
        await refreshAssets();
        await refreshObjects();
    }
    catch (err) {
        status(err.message, true);
    }
}

startup();
</script>

</body>
</html>
"""


async def _index(
    _request: web.Request,
) -> web.Response:
    return web.Response(
        text=HTML,
        content_type="text/html",
    )


# ---------------------------------------------------------------------------
# HTTP server lifetime
# ---------------------------------------------------------------------------


async def _run_http_server(
    node_runner: NodeRunner,
    host: str,
    port: int,
) -> None:
    app = web.Application()

    app["node_runner"] = node_runner

    app.router.add_get(
        "/",
        _index,
    )

    app.router.add_get(
        "/api/health",
        _api_health,
    )

    app.router.add_get(
        "/api/assets",
        _api_assets,
    )

    app.router.add_get(
        "/api/objects",
        _api_objects,
    )

    app.router.add_post(
        "/api/scene/load",
        _api_load_scene,
    )

    app.router.add_post(
        "/api/scene/clear",
        _api_clear_scene,
    )

    app.router.add_post(
        "/api/objects/spawn",
        _api_spawn_object,
    )

    app.router.add_post(
        "/api/objects/force",
        _api_apply_force,
    )

    app.router.add_post(
        "/api/objects/move",
        _api_move_object,
    )

    app.router.add_post(
        "/api/objects/remove",
        _api_remove_object,
    )

    app.router.add_post(
        "/api/robot/move",
        _api_move_robot,
    )

    runner = web.AppRunner(
        app
    )

    await runner.setup()

    site = web.TCPSite(
        runner,
        host=host,
        port=port,
    )

    await site.start()

    logger.info(
        "OpenArm scene commander web UI ready at http://%s:%d",
        host,
        port,
    )

    try:
        await asyncio.Event().wait()

    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Peppy entry point
# ---------------------------------------------------------------------------


async def setup(
    params,
    node_runner: NodeRunner,
) -> list[asyncio.Task]:
    logger.info(
        "OpenArm scene commander starting"
    )

    assets = await _fetch_assets(
        node_runner
    )

    objects = await _fetch_objects(
        node_runner
    )

    logger.info(
        "Connected to Isaac provider: %d assets, %d runtime objects",
        len(assets),
        len(objects),
    )

    host = str(
        _param(
            params,
            "http_host",
            "0.0.0.0",
        )
    )

    port = int(
        _param(
            params,
            "http_port",
            8766,
        )
    )

    server_task = asyncio.create_task(
        _run_http_server(
            node_runner,
            host,
            port,
        )
    )

    return [
        server_task,
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    NodeBuilder().run(
        setup
    )


if __name__ == "__main__":
    main()
