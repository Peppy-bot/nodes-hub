#!/usr/bin/env python3
"""Peppy scene services/actions bridge for Isaac Sim."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue

from peppygen.exposed_actions import (
    apply_force,
    clear_scene,
    load_scene,
    move_object,
    move_robot,
    remove_object,
    spawn_object,
)
from peppygen.exposed_services import (
    get_assets_list,
    get_objects_list,
)

logger = logging.getLogger(__name__)


@dataclass
class _PendingCommand:
    operation: str
    payload: dict
    future: Future


class SceneActionIO:
    """Bridge Peppy scene APIs to the Isaac simulation thread."""

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop

        self._lock = threading.Lock()

        self._assets: dict[str, dict] = {}
        self._objects: dict[str, dict] = {}

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
            asyncio.create_task(self._serve_objects()),
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

    def _public_objects(self) -> list:
        """Return currently spawned runtime-object metadata."""

        with self._lock:
            objects = [
                dict(obj)
                for obj in self._objects.values()
            ]

        objects.sort(
            key=lambda item: item.get(
                "object_id",
                ""
            )
        )

        return objects

    def _handle_get_assets(
        self,
        _request,
    ) -> get_assets_list.Response:
        assets = self._public_assets()

        return get_assets_list.Response(
            success=True,
            message=f"{len(assets)} assets available",
            assets_json=json.dumps(
                assets,
                separators=(",", ":"),
            ),
        )

    def _handle_get_objects(
        self,
        _request,
    ) -> get_objects_list.Response:
        objects = self._public_objects()

        return get_objects_list.Response(
            success=True,
            message=f"{len(objects)} runtime objects",
            objects_json=json.dumps(
                objects,
                separators=(",", ":"),
            ),
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
        """Execute queued scene commands on the Isaac main thread."""

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
            with self._lock:
                object_ids = list(
                    self._objects
                )

            for object_id in object_ids:
                launcher._runtime_remove(
                    {
                        "name": object_id,
                    }
                )

            launcher._runtime_clear_scene()

            with self._lock:
                self._objects.clear()

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
                    "position": position,
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

            with self._lock:
                self._objects[
                    object_id
                ]["position"] = position

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

    async def _serve_objects(self) -> None:
        while True:
            try:
                await get_objects_list.handle_next_request(
                    self._node_runner,
                    self._handle_get_objects,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_objects_list service failed"
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
