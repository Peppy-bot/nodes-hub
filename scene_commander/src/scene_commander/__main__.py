#!/usr/bin/env python3
"""Web commander for simulation scenes through the scene_control contract."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

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


async def _fetch_objects(node_runner: NodeRunner) -> list[dict]:
    producer = get_objects_list.bound_producer(node_runner)

    response = await get_objects_list.poll(
        node_runner,
        producer,
        timeout=SERVICE_TIMEOUT_S,
    )

    data = response.data

    if not data.success:
        raise RuntimeError(data.message)

    return json.loads(data.objects_json)


class _CatalogueWatch:
    """Log the provider's catalogue state when it changes, not on every poll."""

    def __init__(self) -> None:
        self._state: tuple | None = None

    def unavailable(self, message: str) -> None:
        if self._state != ("unavailable", message):
            self._state = ("unavailable", message)
            logger.info("Scene provider has no catalogue yet: %s", message)

    def ready(self, count: int) -> None:
        if self._state != ("ready",):
            self._state = ("ready",)
            logger.info("Scene provider catalogue ready: %d assets", count)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


async def _run_action(action, node_runner: NodeRunner, request=None, **fields):
    """Fire one scene_control goal and return its result data.

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
_CATALOGUE = web.AppKey("catalogue", _CatalogueWatch)


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

    return web.json_response(
        {"success": False, "message": str(exc)},
        status=status,
    )


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
    try:
        objects = await _fetch_objects(request.app[_NODE_RUNNER])

    except Exception as exc:
        return _json_error(request, exc, status=500)

    return web.json_response(
        {"success": True, "objects": objects, "count": len(objects)}
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

async function startup() {
    await loadCatalogue();
    await refreshObjects();
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


def _build_app(node_runner: NodeRunner, catalogue: _CatalogueWatch) -> web.Application:
    app = web.Application()

    app[_NODE_RUNNER] = node_runner
    app[_CATALOGUE] = catalogue

    app.router.add_get("/", _index)
    app.router.add_get("/api/health", _api_health)
    app.router.add_get("/api/assets", _api_assets)
    app.router.add_get("/api/objects", _api_objects)
    app.router.add_post("/api/scene/load", _api_load_scene)
    app.router.add_post("/api/scene/clear", _api_clear_scene)
    app.router.add_post("/api/objects/spawn", _api_spawn_object)
    app.router.add_post("/api/objects/force", _api_apply_force)
    app.router.add_post("/api/objects/move", _api_move_object)
    app.router.add_post("/api/objects/remove", _api_remove_object)
    app.router.add_post("/api/robot/move", _api_move_robot)

    return app


# ---------------------------------------------------------------------------
# Peppy entry point
# ---------------------------------------------------------------------------


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    logger.info("Scene commander starting")

    catalogue = _CatalogueWatch()

    # The scene panel's socket, owned before the provider is waited on: a
    # launch this node cannot serve is refused first, and which copy holds the
    # preferred port follows launch order.
    listener = listen.bind_listener(params.http_host, params.http_port)

    logger.info(
        "Scene panel at %s (bound %s)",
        listen.served_url(listener),
        listen.bound_address(listener),
    )

    # The provider must answer; whether it has its catalogue yet is a state
    # the page polls for.
    try:
        assets = await _fetch_assets(node_runner)

    except SceneCatalogueUnavailable as exc:
        catalogue.unavailable(str(exc))

    else:
        catalogue.ready(len(assets))

    objects = await _fetch_objects(node_runner)

    logger.info("Scene provider reachable: %d runtime objects", len(objects))

    server_task = await listen.start_serving(
        _build_app(node_runner, catalogue), listener
    )

    return [server_task]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
