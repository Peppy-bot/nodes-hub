#!/usr/bin/env python3
"""Peppy scene_manipulation and object_state bridge for Isaac Sim."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue

from peppygen.exposed_actions.scene import (
    apply_force,
    clear_scene,
    load_scene,
    move_object,
    move_robot,
    remove_object,
    spawn_object,
)
from peppygen.exposed_services.objects import get_object_states
from peppygen.exposed_services.scene import get_assets_list

from object_state import IsaacObjectReader, ObjectStateSnapshot

logger = logging.getLogger(__name__)

# What get_object_states answers until the first capture: not an empty scene.
_NOT_CAPTURED = (
    "Isaac has not captured its object state yet; it captures once the "
    "stage has loaded and the simulation is stepping"
)


@dataclass
class _PendingCommand:
    operation: str
    payload: dict
    future: Future


class SceneActionIO:
    """Bridge scene_manipulation and object_state to the Isaac simulation thread.

    Scene edits run on the Isaac thread in submission order. Object state is
    captured there too, from the registry of spawned objects and the engine,
    and get_object_states answers the latest capture.
    """

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
        io,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        # Stamps every object-state capture on the joint states' timeline.
        self._io = io

        self._lock = threading.Lock()

        # Empty until the launcher hands over the discovered catalogue;
        # get_assets_list says so rather than answering with nothing.
        self._assets: dict[str, dict] = {}
        self._assets_ready = False
        # The spawned objects in spawn order: what each was spawned from and
        # with. Where each one is comes from the engine at capture.
        self._objects: dict[str, dict] = {}

        self._object_reader = IsaacObjectReader()
        # The latest capture, and why there is none while it is None.
        self._snapshot: ObjectStateSnapshot | None = None
        self._unavailable: str | None = _NOT_CAPTURED

        self._pending: Queue[_PendingCommand] = Queue()

        self._tasks: list[asyncio.Task] = []

        self._action_handles = {}

    async def start(self) -> None:
        """Expose Peppy services/actions and start their request loops."""

        self._action_handles = {
            "apply_force": await apply_force.ActionHandle.expose(
                self._node_runner
            ),
            "load_scene": await load_scene.ActionHandle.expose(
                self._node_runner
            ),
            "clear_scene": await clear_scene.ActionHandle.expose(
                self._node_runner
            ),
            "spawn_object": await spawn_object.ActionHandle.expose(
                self._node_runner
            ),
            "move_object": await move_object.ActionHandle.expose(
                self._node_runner
            ),
            "remove_object": await remove_object.ActionHandle.expose(
                self._node_runner
            ),
            "move_robot": await move_robot.ActionHandle.expose(
                self._node_runner
            ),
        }

        self._tasks = [
            asyncio.create_task(
                self._serve_apply_force()
            ),
            asyncio.create_task(self._serve_assets()),
            asyncio.create_task(self._serve_object_states()),
            asyncio.create_task(self._serve_load_scene()),
            asyncio.create_task(self._serve_clear_scene()),
            asyncio.create_task(self._serve_spawn_object()),
            asyncio.create_task(self._serve_move_object()),
            asyncio.create_task(self._serve_remove_object()),
            asyncio.create_task(self._serve_move_robot()),
        ]

        logger.info(
            "SceneActionIO services and actions started"
        )

    async def stop(self) -> None:
        """Stop Peppy scene service/action loops."""

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(
                *self._tasks,
                return_exceptions=True,
            )

        self._tasks = []

        logger.info(
            "SceneActionIO services and actions stopped"
        )

    def set_assets(self, assets: dict) -> None:
        """Replace the private Isaac asset catalogue."""

        with self._lock:
            self._assets = {
                asset_id: dict(asset)
                for asset_id, asset in assets.items()
            }
            self._assets_ready = True

        logger.info(
            "SceneActionIO received %d Isaac assets",
            len(assets),
        )

    def _asset(self, asset_id: str) -> dict | None:
        with self._lock:
            asset = self._assets.get(asset_id)

            if asset is None:
                return None

            return dict(asset)

    def _object(self, object_id: str) -> dict | None:
        with self._lock:
            obj = self._objects.get(object_id)

            if obj is None:
                return None

            return dict(obj)

    def owns(self, object_id) -> bool:
        """Whether object_id names an object spawned through scene_manipulation.

        The registry and the stage only stay in step through scene_manipulation,
        so the runtime commander asks before it removes or replaces a
        runtime object.
        """

        with self._lock:
            return object_id in self._objects

    def _public_assets(self) -> list:
        """Return catalogue metadata without exposing raw Isaac paths."""

        with self._lock:
            assets = list(
                self._assets.values()
            )

        public = []

        for asset in assets:
            public.append(
                {
                    "asset_id": asset.get(
                        "asset_id",
                        "",
                    ),
                    "display_name": asset.get(
                        "display_name",
                        "",
                    ),
                    "kind": asset.get(
                        "kind",
                        "",
                    ),
                    "category": asset.get(
                        "category",
                        "",
                    ),
                }
            )

        scene_order = {
            "scene/simple_warehouse": 0,
            "scene/flat_grid": 1,
            "scene/black_grid": 2,
            "scene/curved_grid": 3,
            "scene/simple_room": 4,
            "scene/office": 5,
            "scene/hospital": 6,
            "scene/warehouse_forklifts": 7,
            "scene/warehouse_multiple_shelves": 8,
            "scene/full_warehouse": 9,
        }

        public.sort(
            key=lambda item: (
                item["kind"].lower(),
                item["category"].lower(),
                scene_order.get(
                    item["asset_id"],
                    999,
                )
                if item["kind"] == "scene"
                else 0,
                item["display_name"].lower(),
                item["asset_id"],
            )
        )

        return public

    def _handle_get_assets(
        self,
        _request,
    ) -> get_assets_list.Response:
        with self._lock:
            ready = self._assets_ready

        if not ready:
            return get_assets_list.Response(
                success=False,
                message=(
                    "Isaac is still discovering its asset catalogue; "
                    "it is available once the stage has loaded"
                ),
                assets_json="[]",
            )

        assets = self._public_assets()

        return get_assets_list.Response(
            success=True,
            message=f"{len(assets)} assets available",
            assets_json=json.dumps(
                assets,
                separators=(",", ":"),
            ),
        )

    def invalidate_physics_views(self) -> None:
        """Drop the object reader's rigid-body view after a prim removal; the
        next capture creates it again."""

        self._object_reader.invalidate()

    def capture_object_states(self) -> ObjectStateSnapshot | None:
        """Capture every spawned object on the Isaac main thread.

        The capture becomes the snapshot get_object_states answers and is
        returned for the stream to publish. None when there is no snapshot:
        a capture that cannot be stamped yet is not one, and a failed read
        leaves the state unavailable with its reason rather than answering an
        older capture that may predate an edit.
        """

        timestamp_s = self._io.capture_timestamp_s()

        if timestamp_s is None:
            return None

        with self._lock:
            spawned = [
                dict(obj)
                for obj in self._objects.values()
            ]

        try:
            records = self._object_reader.read(spawned)

        except Exception as exc:
            reason = f"Isaac could not capture its object state: {exc}"

            with self._lock:
                repeated = self._unavailable == reason
                self._snapshot = None
                self._unavailable = reason

            # One line per distinct failure, not one per capture.
            if not repeated:
                logger.warning(reason)

            return None

        snapshot = ObjectStateSnapshot(
            timestamp_s=timestamp_s,
            objects=tuple(records),
        )

        with self._lock:
            self._snapshot = snapshot
            self._unavailable = None

        return snapshot

    def _handle_get_object_states(
        self,
        _request,
    ) -> get_object_states.Response:
        with self._lock:
            snapshot = self._snapshot
            unavailable = self._unavailable

        if snapshot is None:
            # Unavailable is not an empty scene: the timestamp and objects
            # carry nothing.
            return get_object_states.Response(
                success=False,
                message=unavailable,
                timestamp=0.0,
                objects=[],
            )

        return get_object_states.Response(
            success=True,
            message=f"{len(snapshot.objects)} objects",
            timestamp=snapshot.timestamp_s,
            objects=[
                get_object_states.ResponseObjectsItem(
                    **record.fields()
                )
                for record in snapshot.objects
            ],
        )

    async def _submit(
        self,
        operation: str,
        payload: dict,
    ) -> dict:
        future = Future()

        self._pending.put(
            _PendingCommand(
                operation=operation,
                payload=payload,
                future=future,
            )
        )

        return await asyncio.wrap_future(
            future
        )

    def process_pending(
        self,
        launcher,
        max_commands: int = 32,
    ) -> None:
        """Execute queued scene commands on the Isaac main thread.

        Each command's goal completes after a fresh object-state capture, so
        a read once it has completed observes its edit. Until the first
        stamped capture, every frame tries one, so reads are answered from
        the first frame on, ahead of the stream's first tick. After that the
        state tick and the capture after each edit keep the snapshot current,
        and bring it back after a failed read.
        """

        with self._lock:
            never_captured = self._unavailable == _NOT_CAPTURED

        if never_captured:
            self.capture_object_states()

        for _ in range(max_commands):
            try:
                pending = self._pending.get_nowait()

            except Empty:
                return

            try:
                result = self._execute(
                    launcher,
                    pending.operation,
                    pending.payload,
                )

            except Exception as exc:
                logger.exception(
                    "Scene action '%s' failed",
                    pending.operation,
                )

                result = {
                    "success": False,
                    "message": str(exc),
                }

            # Succeeded or failed partway, whatever the command did to the
            # scene is in the snapshot before its goal completes.
            self.capture_object_states()

            if not pending.future.done():
                pending.future.set_result(
                    result
                )

    def _execute(
        self,
        launcher,
        operation: str,
        payload: dict,
    ) -> dict:
        """Execute one scene command from the Isaac simulation thread."""

        if operation == "load_scene":
            asset_id = payload["asset_id"]
            asset = self._asset(asset_id)

            if asset is None:
                raise ValueError(
                    f"Unknown asset_id: {asset_id}"
                )

            if asset.get("kind") != "scene":
                raise ValueError(
                    f"Asset is not a scene: {asset_id}"
                )

            scale = float(payload["scale"])

            if scale <= 0.0:
                raise ValueError(
                    "scale must be greater than zero"
                )

            # scene_manipulation: spawned objects go first, then the scene is
            # replaced.
            self._remove_spawned_objects(launcher)

            launcher._runtime_load_isaac_scene(
                {
                    "path": asset["path"],
                    "scale": [scale, scale, scale],
                }
            )

            return {
                "success": True,
                "message": f"Loaded scene {asset_id}",
            }

        if operation == "clear_scene":
            self._remove_spawned_objects(launcher)

            launcher._runtime_clear_scene()

            return {
                "success": True,
                "message": "Runtime scene cleared",
            }

        if operation == "spawn_object":
            asset_id = payload["asset_id"]
            asset = self._asset(asset_id)

            if asset is None:
                raise ValueError(
                    f"Unknown asset_id: {asset_id}"
                )

            if asset.get("kind") != "object":
                raise ValueError(
                    f"Asset is not an object: {asset_id}"
                )

            position = [
                float(value)
                for value in payload["position"]
            ]

            if len(position) != 3:
                raise ValueError(
                    "position must contain exactly 3 values"
                )

            scale = float(payload["scale"])

            if scale <= 0.0:
                raise ValueError(
                    "scale must be greater than zero"
                )

            physics = str(
                payload["physics"]
            ).lower()

            if physics not in (
                "none",
                "static",
                "dynamic",
            ):
                raise ValueError(
                    "physics must be none, static, or dynamic"
                )

            mass = float(
                payload["mass"]
            )

            if mass <= 0.0:
                raise ValueError(
                    "mass must be greater than zero"
                )

            object_id = (
                "obj_"
                + uuid.uuid4().hex[:12]
            )

            launcher._runtime_spawn_isaac_asset(
                {
                    "name": object_id,
                    "path": asset["path"],
                    "position": position,
                    "scale": [
                        scale,
                        scale,
                        scale,
                    ],
                    "physics": physics,
                    "mass": mass,
                }
            )

            with self._lock:
                self._objects[object_id] = {
                    "object_id": object_id,
                    "asset_id": asset_id,
                    "scale": scale,
                    "physics": physics,
                    "mass": mass,
                }

            return {
                "success": True,
                "message": f"Spawned {asset_id}",
                "object_id": object_id,
            }

        if operation == "move_object":
            object_id = payload["object_id"]

            if self._object(object_id) is None:
                raise ValueError(
                    f"Unknown object_id: {object_id}"
                )

            position = [
                float(value)
                for value in payload["position"]
            ]

            if len(position) != 3:
                raise ValueError(
                    "position must contain exactly 3 values"
                )

            launcher._runtime_move_object(
                {
                    "name": object_id,
                    "position": position,
                }
            )

            return {
                "success": True,
                "message": f"Moved {object_id}",
            }

        if operation == "remove_object":
            object_id = payload["object_id"]

            if self._object(object_id) is None:
                raise ValueError(
                    f"Unknown object_id: {object_id}"
                )

            launcher._runtime_remove(
                {
                    "name": object_id,
                }
            )

            with self._lock:
                self._objects.pop(
                    object_id,
                    None,
                )

            return {
                "success": True,
                "message": f"Removed {object_id}",
            }

        if operation == "apply_force":
            object_id = str(
                payload["object_id"]
            )

            obj = self._object(
                object_id
            )

            if obj is None:
                raise ValueError(
                    f"Unknown object_id: {object_id}"
                )

            if (
                str(
                    obj.get(
                        "physics",
                        "none",
                    )
                ).lower()
                != "dynamic"
            ):
                raise ValueError(
                    "Force can only be applied to "
                    f"dynamic objects: {object_id}"
                )

            force = [
                float(value)
                for value in payload["force"]
            ]

            if len(force) != 3:
                raise ValueError(
                    "force must contain exactly "
                    "3 values [Fx, Fy, Fz]"
                )

            duration_s = float(
                payload["duration_s"]
            )

            if (
                duration_s <= 0.0
                or duration_s > 30.0
            ):
                raise ValueError(
                    "duration_s must be > 0 "
                    "and <= 30 seconds"
                )

            launcher._runtime_apply_force(
                {
                    "name": object_id,
                    "force": force,
                    "duration_s": duration_s,
                }
            )

            return {
                "success": True,
                "message": (
                    f"Applied force {force} N "
                    f"to {object_id} for "
                    f"{duration_s:.3f} s"
                ),
            }

        if operation == "move_robot":
            position = [
                float(value)
                for value in payload["position"]
            ]

            if len(position) != 3:
                raise ValueError(
                    "position must contain exactly 3 values"
                )

            launcher._runtime_move_robot_root(
                {
                    "position": position,
                }
            )

            return {
                "success": True,
                "message": "Robot root moved",
            }

        raise ValueError(
            f"Unsupported scene operation: {operation}"
        )

    def _remove_spawned_objects(self, launcher) -> None:
        """Remove every spawned object, as load_scene and clear_scene promise."""

        with self._lock:
            object_ids = list(self._objects)

        for object_id in object_ids:
            launcher._runtime_remove(
                {
                    "name": object_id,
                }
            )

            with self._lock:
                self._objects.pop(object_id, None)

    async def _finish_simple(
        self,
        context,
        result: dict,
    ) -> None:
        success = bool(
            result.get("success", False)
        )

        message = str(
            result.get("message", "")
        )

        if context.is_cancelled():
            await context.complete_cancelled(
                success,
                message,
            )

        else:
            await context.complete(
                success,
                message,
            )

    async def _serve_assets(self) -> None:
        while True:
            try:
                await get_assets_list.handle_next_request(
                    self._node_runner,
                    self._handle_get_assets,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_assets_list service failed"
                )
                await asyncio.sleep(1.0)

    async def _serve_object_states(self) -> None:
        while True:
            try:
                await get_object_states.handle_next_request(
                    self._node_runner,
                    self._handle_get_object_states,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_object_states service failed"
                )
                await asyncio.sleep(1.0)

    async def _serve_load_scene(self) -> None:
        handle = self._action_handles[
            "load_scene"
        ]

        while True:
            context = await handle.handle_goal_next_request(
                lambda _request:
                load_scene.GoalDecision.accept()
            )

            if context is None:
                return

            request = context.request().data

            result = await self._submit(
                "load_scene",
                {
                    "asset_id": request.asset_id,
                    "scale": request.scale,
                },
            )

            await self._finish_simple(
                context,
                result,
            )

    async def _serve_clear_scene(self) -> None:
        handle = self._action_handles[
            "clear_scene"
        ]

        while True:
            context = await handle.handle_goal_next_request(
                lambda _request:
                clear_scene.GoalDecision.accept()
            )

            if context is None:
                return

            result = await self._submit(
                "clear_scene",
                {},
            )

            await self._finish_simple(
                context,
                result,
            )

    async def _serve_spawn_object(self) -> None:
        handle = self._action_handles[
            "spawn_object"
        ]

        while True:
            context = await handle.handle_goal_next_request(
                lambda _request:
                spawn_object.GoalDecision.accept()
            )

            if context is None:
                return

            request = context.request().data

            result = await self._submit(
                "spawn_object",
                {
                    "asset_id": request.asset_id,
                    "position": list(
                        request.position
                    ),
                    "scale": request.scale,
                    "physics": request.physics,
                    "mass": request.mass,
                },
            )

            success = bool(
                result.get("success", False)
            )
            message = str(
                result.get("message", "")
            )
            object_id = str(
                result.get("object_id", "")
            )

            if context.is_cancelled():
                await context.complete_cancelled(
                    success,
                    message,
                    object_id,
                )

            else:
                await context.complete(
                    success,
                    message,
                    object_id,
                )

    async def _serve_move_object(self) -> None:
        handle = self._action_handles[
            "move_object"
        ]

        while True:
            context = await handle.handle_goal_next_request(
                lambda _request:
                move_object.GoalDecision.accept()
            )

            if context is None:
                return

            request = context.request().data

            result = await self._submit(
                "move_object",
                {
                    "object_id": request.object_id,
                    "position": list(
                        request.position
                    ),
                },
            )

            await self._finish_simple(
                context,
                result,
            )

    async def _serve_remove_object(self) -> None:
        handle = self._action_handles[
            "remove_object"
        ]

        while True:
            context = await handle.handle_goal_next_request(
                lambda _request:
                remove_object.GoalDecision.accept()
            )

            if context is None:
                return

            request = context.request().data

            result = await self._submit(
                "remove_object",
                {
                    "object_id": request.object_id,
                },
            )

            await self._finish_simple(
                context,
                result,
            )

    async def _serve_apply_force(self) -> None:
        handle = self._action_handles[
            "apply_force"
        ]

        while True:
            context = (
                await handle.handle_goal_next_request(
                    lambda _request:
                    apply_force.GoalDecision.accept()
                )
            )

            if context is None:
                return

            request = context.request().data

            result = await self._submit(
                "apply_force",
                {
                    "object_id": request.object_id,
                    "force": [
                        float(value)
                        for value in request.force
                    ],
                    "duration_s": float(
                        request.duration_s
                    ),
                },
            )

            await self._finish_simple(
                context,
                result,
            )

    async def _serve_move_robot(self) -> None:
        handle = self._action_handles[
            "move_robot"
        ]

        while True:
            context = await handle.handle_goal_next_request(
                lambda _request:
                move_robot.GoalDecision.accept()
            )

            if context is None:
                return

            request = context.request().data

            result = await self._submit(
                "move_robot",
                {
                    "position": list(
                        request.position
                    ),
                },
            )

            await self._finish_simple(
                context,
                result,
            )
