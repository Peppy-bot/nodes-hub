#!/usr/bin/env python3
"""Peppy scene_manipulation and object_state bridge for Isaac Sim."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
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
from peppygen.exposed_services.scene import (
    get_asset_categories,
    get_assets_list,
    get_objects_list,
    get_robots_list,
    get_scene,
)

from object_state import IsaacObjectReader, ObjectStateSnapshot


logger = logging.getLogger(__name__)

# What get_object_states answers until the first capture: not an empty scene.
_NOT_CAPTURED = (
    "Isaac has not captured its object state yet; it captures once the "
    "stage has loaded and the simulation is stepping"
)

# How long a progress message of a load_scene or spawn_object goal may take to
# be sent. A message still not sent then holds neither its goal nor the goals
# its action serves after it.
_REPORT_TIMEOUT_S = 5.0

# The kinds an asset has, as get_assets_list filters them.
_ASSET_KINDS = ("scene", "object")

# The sources get_objects_list filters by. Every object of this stage was
# spawned by a caller: the stage places no object of its own when it loads
# a scene, so "scene" lists nothing.
_OBJECT_SOURCES = ("spawned", "scene")

# How far from 1 the length of an orientation may be before it is refused.
_UNIT_TOLERANCE = 1e-3


def _orientation_of(payload: dict) -> list[float] | None:
    """The orientation a goal names, a unit quaternion [x, y, z, w]
    normalised, or None when the goal names none. A quaternion that is not
    a unit one, within _UNIT_TOLERANCE, is refused: USD takes it and turns
    the prim by something else."""
    orientation = payload.get("orientation")

    if orientation is None:
        return None

    values = [float(value) for value in orientation]

    if len(values) != 4:
        raise ValueError("orientation must contain exactly 4 values [x, y, z, w]")

    norm = math.sqrt(sum(value * value for value in values))

    if not math.isfinite(norm) or abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise ValueError(
            "orientation must be a unit quaternion [x, y, z, w]"
        )

    return [value / norm for value in values]


def _yaw_of(payload: dict) -> float:
    """The heading a goal names, in radians about +z. A prim turned by a yaw
    that is not a number reports no error and simulates nothing, so one is
    refused here."""
    yaw = float(payload["yaw"])

    if not math.isfinite(yaw):
        raise ValueError(
            "yaw must be a finite number of radians"
        )

    return yaw


def _filter(value: str | None) -> str | None:
    """A request's filter: a field left out, or sent empty, is no filter."""

    return value or None


def _matches(
    asset: dict,
    kind: str | None,
    category: str | None,
    query: str | None,
) -> bool:
    """Whether a public catalogue entry passes every filter that is set."""

    if kind is not None and asset["kind"] != kind:
        return False

    if category is not None and asset["category"] != category:
        return False

    if query is None:
        return True

    needle = query.lower()

    return any(
        needle in str(asset.get(field, "")).lower()
        for field in ("asset_id", "display_name", "description")
    )


@dataclass
class _PendingCommand:
    operation: str
    payload: dict
    # Answers the goal with the command's result.
    future: Future
    # Tells the goal that the Isaac main thread starts the command.
    started: Future = field(default_factory=Future)


def _settle(future: Future, value) -> None:
    """Hands `value` to the goal awaiting `future`, from the Isaac main
    thread. A goal that stopped awaiting it, as the node stops, has cancelled
    it, and is told nothing."""

    if future.set_running_or_notify_cancel():
        future.set_result(value)


async def _report_progress(
    context,
    operation: str,
    building: bool,
) -> bool:
    """Publishes one progress message of a load_scene or spawn_object goal,
    with nothing fetched. False, and logged, when the message was not sent
    within _REPORT_TIMEOUT_S or the publish failed."""

    try:
        await asyncio.wait_for(
            context.publish_feedback(
                bytes_fetched=0,
                files_ready=0,
                building=building,
            ),
            timeout=_REPORT_TIMEOUT_S,
        )

    except asyncio.TimeoutError:
        logger.warning(
            "%s goal %s: a progress message was not sent within %g s; "
            "its progress is no longer reported",
            operation,
            context.goal_id(),
            _REPORT_TIMEOUT_S,
        )
        return False

    except Exception as error:  # pylint: disable=W0718
        logger.warning(
            "%s goal %s: its progress is no longer reported: %s",
            operation,
            context.goal_id(),
            error,
        )
        return False

    return True


class SceneActionIO:
    """Bridge scene_manipulation and object_state to the Isaac simulation thread.

    Scene edits run on the Isaac thread in submission order. Object state is
    captured there too, from the registry of spawned objects and the engine,
    and get_object_states answers the latest capture. A load_scene or
    spawn_object goal reports its progress on its feedback
    (_submit_reporting_progress).
    """

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
        io,
        world,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        # Stamps every object-state capture on the joint states' timeline.
        self._io = io

        # The stage's own account of which robots stand in it, which
        # get_robots_list reports and move_robot resolves a name against.
        self._world = world

        self._lock = threading.Lock()

        # Empty until the launcher hands over the discovered catalogue;
        # get_assets_list says so rather than answering with nothing.
        self._assets: dict[str, dict] = {}
        self._assets_ready = False
        # The spawned objects in spawn order: what each was spawned from and
        # with. Where each one is comes from the engine at capture.
        self._objects: dict[str, dict] = {}
        # The scene loaded last, as get_scene names it: no asset while the
        # stage stands empty, which is how it opens and what clear_scene
        # leaves.
        self._scene: dict = {"asset_id": "", "scale": 1.0}

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
            asyncio.create_task(self._serve_asset_categories()),
            asyncio.create_task(self._serve_objects_list()),
            asyncio.create_task(self._serve_scene()),
            asyncio.create_task(self._serve_object_states()),
            asyncio.create_task(self._serve_robots()),
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
                    "description": asset.get(
                        "description",
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

    def _catalogue_ready(self) -> str | None:
        """Why the catalogue cannot be answered yet, or None once it can."""

        with self._lock:
            ready = self._assets_ready

        if ready:
            return None

        return (
            "Isaac is still discovering its asset catalogue; "
            "it is available once the stage has loaded"
        )

    def _handle_get_assets(
        self,
        request,
    ) -> get_assets_list.Response:
        """The catalogue, narrowed by every filter the request sets: a
        kind that is neither scene nor object is refused, and the text is
        looked for in an entry's id, name and description whatever the
        case."""

        not_ready = self._catalogue_ready()

        if not_ready is not None:
            return get_assets_list.Response(
                success=False,
                message=not_ready,
                assets_json="[]",
            )

        kind = _filter(request.data.kind)
        category = _filter(request.data.category)
        query = _filter(request.data.query)

        if kind is not None and kind not in _ASSET_KINDS:
            return get_assets_list.Response(
                success=False,
                message="kind must be scene or object",
                assets_json="[]",
            )

        assets = [
            asset
            for asset in self._public_assets()
            if _matches(asset, kind, category, query)
        ]

        return get_assets_list.Response(
            success=True,
            message=f"{len(assets)} assets listed",
            assets_json=json.dumps(
                assets,
                separators=(",", ":"),
            ),
        )

    def _handle_get_asset_categories(
        self,
        _request,
    ) -> get_asset_categories.Response:
        """Every category of the catalogue with its count, in the order the
        catalogue lists them."""

        not_ready = self._catalogue_ready()

        if not_ready is not None:
            return get_asset_categories.Response(
                success=False,
                message=not_ready,
                categories=[],
            )

        counts: dict[tuple[str, str], int] = {}

        for asset in self._public_assets():
            key = (asset["kind"], asset["category"])
            counts[key] = counts.get(key, 0) + 1

        categories = [
            get_asset_categories.ResponseCategoriesItem(
                kind=kind,
                category=category,
                count=count,
            )
            for (kind, category), count in counts.items()
        ]

        return get_asset_categories.Response(
            success=True,
            message=f"{len(categories)} categories",
            categories=categories,
        )

    def _handle_get_objects_list(
        self,
        request,
    ) -> get_objects_list.Response:
        """What stands on the stage and where, from the latest capture: every
        object a caller spawned, in its asset's category, narrowed by the
        request's source and category. The stage places no object of its own
        when it loads a scene, so a request for the scene's objects lists
        nothing."""

        source = _filter(request.data.source)
        category = _filter(request.data.category)

        if source is not None and source not in _OBJECT_SOURCES:
            return get_objects_list.Response(
                success=False,
                message="source must be spawned or scene",
                timestamp=0.0,
                objects=[],
            )

        with self._lock:
            snapshot = self._snapshot
            unavailable = self._unavailable

        if snapshot is None:
            return get_objects_list.Response(
                success=False,
                message=unavailable,
                timestamp=0.0,
                objects=[],
            )

        objects = []

        if source != "scene":
            for record in snapshot.objects:
                asset = self._asset(record.asset_id) or {}
                asset_category = str(asset.get("category", ""))

                if category is not None and category != asset_category:
                    continue

                objects.append(
                    get_objects_list.ResponseObjectsItem(
                        object_id=record.object_id,
                        asset_id=record.asset_id,
                        category=asset_category,
                        source="spawned",
                        physics=record.physics,
                        mass=record.mass,
                        scale=record.scale,
                        position=list(record.position),
                        orientation=list(record.orientation),
                    )
                )

        return get_objects_list.Response(
            success=True,
            message=f"{len(objects)} objects",
            timestamp=snapshot.timestamp_s,
            objects=objects,
        )

    def _handle_get_scene(
        self,
        _request,
    ) -> get_scene.Response:
        with self._lock:
            scene = dict(self._scene)

        if not scene["asset_id"]:
            return get_scene.Response(
                success=True,
                message="no scene is loaded: the stage stands empty",
                asset_id="",
                scale=1.0,
            )

        return get_scene.Response(
            success=True,
            message=f"scene {scene['asset_id']} at scale {scene['scale']}",
            asset_id=scene["asset_id"],
            scale=scene["scale"],
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

    def _handle_get_robots(
        self,
        _request,
    ) -> get_robots_list.Response:
        standing = self._world.robots()
        robots = [
            get_robots_list.ResponseRobotsItem(
                robot=robot.instance,
                model=robot.model,
                position=list(robot.placement.position),
                yaw=robot.placement.yaw,
                # Every robot on this stage joined it by attaching.
                attached=True,
            )
            for robot in standing
        ]
        plural = "" if len(robots) == 1 else "s"
        return get_robots_list.Response(
            success=True,
            message=f"{len(robots)} robot{plural} standing",
            robots=robots,
        )

    def _queue(
        self,
        operation: str,
        payload: dict,
    ) -> _PendingCommand:
        """Hands a command to the Isaac main thread, which runs it after the
        commands queued before it."""

        command = _PendingCommand(
            operation=operation,
            payload=payload,
            future=Future(),
        )

        self._pending.put(command)

        return command

    async def _submit(
        self,
        operation: str,
        payload: dict,
    ) -> dict:
        return await asyncio.wrap_future(
            self._queue(operation, payload).future
        )

    async def _submit_reporting_progress(
        self,
        context,
        operation: str,
        payload: dict,
    ) -> dict:
        """Runs a load_scene or spawn_object command as _submit does, and
        reports its progress on the goal's feedback in two messages: accepted,
        before the command is queued, then building, once the Isaac main
        thread starts the command.

        Isaac fetches no file itself, so nothing else is reported: the goal
        is silent while its command waits for the main thread and while USD
        resolves what it names. A message that is not sent in time ends the
        reports, and the command runs on.
        """

        reporting = await _report_progress(
            context,
            operation,
            building=False,
        )

        command = self._queue(operation, payload)

        if reporting:
            await asyncio.wrap_future(command.started)

            await _report_progress(
                context,
                operation,
                building=True,
            )

        return await asyncio.wrap_future(
            command.future
        )

    def process_pending(
        self,
        launcher,
        max_commands: int = 32,
    ) -> None:
        """Execute queued scene commands on the Isaac main thread.

        Each command's goal learns that the command starts before it runs,
        and completes after a fresh object-state capture, so
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

            _settle(pending.started, None)

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

            _settle(pending.future, result)

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

            with self._lock:
                self._scene = {"asset_id": asset_id, "scale": scale}

            return {
                "success": True,
                "message": f"Loaded scene {asset_id}",
            }

        if operation == "clear_scene":
            self._remove_spawned_objects(launcher)

            launcher._runtime_clear_scene()

            with self._lock:
                self._scene = {"asset_id": "", "scale": 1.0}

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

            orientation = _orientation_of(payload)
            yaw = _yaw_of(payload)

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

            spawn = {
                "name": object_id,
                "path": asset["path"],
                "position": position,
                "yaw": yaw,
                "scale": [
                    scale,
                    scale,
                    scale,
                ],
                "physics": physics,
                "mass": mass,
            }

            if orientation is not None:
                spawn["orientation"] = orientation

            launcher._runtime_spawn_isaac_asset(spawn)

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

            move = {
                "name": object_id,
                "position": position,
            }

            orientation = _orientation_of(payload)

            if orientation is not None:
                move["orientation"] = orientation

            launcher._runtime_move_object(move)

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
            robot = payload["robot"]
            standing = self._world.robots()
            names = [
                each.instance
                for each in standing
            ]

            if robot not in names:
                raise ValueError(
                    f"no robot stands as {robot!r} in this "
                    f"simulation, which stands "
                    f"{', '.join(names) or 'none'}"
                )

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
                    "robot": robot,
                    "position": position,
                    "yaw": _yaw_of(payload),
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

    async def _serve_asset_categories(self) -> None:
        while True:
            try:
                await get_asset_categories.handle_next_request(
                    self._node_runner,
                    self._handle_get_asset_categories,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_asset_categories service failed"
                )
                await asyncio.sleep(1.0)

    async def _serve_objects_list(self) -> None:
        while True:
            try:
                await get_objects_list.handle_next_request(
                    self._node_runner,
                    self._handle_get_objects_list,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_objects_list service failed"
                )
                await asyncio.sleep(1.0)

    async def _serve_scene(self) -> None:
        while True:
            try:
                await get_scene.handle_next_request(
                    self._node_runner,
                    self._handle_get_scene,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_scene service failed"
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

    async def _serve_robots(self) -> None:
        while True:
            try:
                await get_robots_list.handle_next_request(
                    self._node_runner,
                    self._handle_get_robots,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "get_robots_list service failed"
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

            result = await self._submit_reporting_progress(
                context,
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

            result = await self._submit_reporting_progress(
                context,
                "spawn_object",
                {
                    "asset_id": request.asset_id,
                    "position": list(
                        request.position
                    ),
                    "yaw": request.yaw,
                    "orientation": request.orientation,
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
                    "orientation": request.orientation,
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
                    "robot": request.robot,
                    "position": list(
                        request.position
                    ),
                    "yaw": request.yaw,
                },
            )

            await self._finish_simple(
                context,
                result,
            )
