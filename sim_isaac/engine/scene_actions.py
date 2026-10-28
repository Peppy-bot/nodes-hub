#!/usr/bin/env python3
"""Peppy scene_manipulation and object_state bridge for Isaac Sim.

A goal this bridge cannot run is refused when it arrives, with a failure kind
of peppy: `rejected` for a request that names an asset or an object the stage
does not hold or a value it does not take, `resource_unavailable` for a robot
that does not stand on the stage, and `precondition_failed` while the asset
catalogue is not discovered yet. A goal it accepted runs its command on the
Isaac main thread, which checks the request again against the stage as it is
then: a command that cannot run fails its goal with the same kinds, and one
that Isaac does not carry out fails it with `attempt_failed`. A service gives
its response, or an error reply with the same kinds.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import re
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import partial
from queue import Empty, Queue
from types import ModuleType
from typing import Callable

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
    get_object,
    get_objects_list,
    get_robots_list,
    get_scene,
)

from object_state import IsaacObjectReader, ObjectStateSnapshot


logger = logging.getLogger(__name__)

# The failure kinds of peppy that a scene command and a service give, by
# their name on the wire. Every member of scene_manipulation and
# object_state declares them.
REJECTED = "rejected"
PRECONDITION_FAILED = "precondition_failed"
RESOURCE_UNAVAILABLE = "resource_unavailable"
ATTEMPT_FAILED = "attempt_failed"

# The actions of scene_manipulation, by the name of the command each goal
# runs on the Isaac main thread.
_ACTIONS: dict[str, ModuleType] = {
    "apply_force": apply_force,
    "load_scene": load_scene,
    "clear_scene": clear_scene,
    "spawn_object": spawn_object,
    "move_object": move_object,
    "remove_object": remove_object,
    "move_robot": move_robot,
}

# The actions whose goals report their progress on their feedback.
_REPORTING_PROGRESS = ("load_scene", "spawn_object")

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

# The longest push apply_force schedules, in seconds.
_LONGEST_PUSH_S = 30.0

# The physics a spawned object may have, as spawn_object names them.
_PHYSICS = ("none", "static", "dynamic")

# How a refusal names each kind of asset.
_ASSET_KIND_WORDS = {"scene": "a scene", "object": "an object"}


class SceneFailure(Exception):
    """Why a scene command does not run, or ran and gave no result: a
    failure kind of peppy and a message for a person. A goal is refused with
    it when it arrives, or fails with it once its command ran; a service
    gives it as its error reply."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


def _rejected(message: str) -> SceneFailure:
    return SceneFailure(REJECTED, message)


@dataclass(frozen=True)
class _Load:
    """A load_scene command, checked: the scene asset and the scale it
    stands at."""

    asset_id: str
    path: str
    scale: float


@dataclass(frozen=True)
class _Spawn:
    """A spawn_object command, checked."""

    asset_id: str
    path: str
    position: list[float]
    yaw: float
    # A unit quaternion [x, y, z, w], normalised, in place of yaw; None
    # turns the object by yaw.
    orientation: list[float] | None
    scale: float
    physics: str
    mass: float
    # The object's name: the one the spawn asked for, or the one the stage
    # gave it.
    name: str


@dataclass(frozen=True)
class _Move:
    """A move_object command, checked: the object, where it goes, and the
    orientation it takes, or None to keep its own."""

    object_id: str
    position: list[float]
    orientation: list[float] | None


@dataclass(frozen=True)
class _Push:
    """An apply_force command, checked: a dynamic object, the world-frame
    force in newtons, and how long it pushes."""

    object_id: str
    force: list[float]
    duration_s: float


@dataclass(frozen=True)
class _RobotMove:
    """A move_robot command, checked: a robot the stage stands, where its
    base goes and which way it faces."""

    robot: str
    position: list[float]
    yaw: float


def _refuse_snap(payload: dict) -> None:
    """Refuses a goal that names a snap: a snap stands an object where one of
    its belt ends, a landmark, meets a belt end of another object, and Isaac
    keeps no landmark of an object, so it refuses every snap with `rejected`,
    as scene_manipulation:v1 says."""
    if payload.get("snap_to") is not None:
        raise _rejected(
            "this simulation keeps no landmark of an object, so it places "
            "nothing by a snap: give the goal a position instead of snap_to"
        )


def _finite_vector(payload: dict, name: str, length: int, form: str) -> list[float]:
    """The vector `name` a goal carries: `length` finite numbers, as `form`
    writes them. A prim moved by a value that is not a number reports no
    error and simulates nothing, so one is refused here."""
    values = [float(value) for value in payload[name]]

    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise _rejected(f"{name} must contain exactly {length} finite values {form}")

    return values


def _position_of(payload: dict) -> list[float]:
    return _finite_vector(payload, "position", 3, "[x, y, z]")


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
        raise _rejected("orientation must contain exactly 4 values [x, y, z, w]")

    norm = math.sqrt(sum(value * value for value in values))

    if not math.isfinite(norm) or abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise _rejected(
            "orientation must be a unit quaternion [x, y, z, w]"
        )

    return [value / norm for value in values]


def _yaw_of(payload: dict) -> float:
    """The heading a goal names, in radians about +z. A prim turned by a yaw
    that is not a number reports no error and simulates nothing, so one is
    refused here."""
    yaw = float(payload["yaw"])

    if not math.isfinite(yaw):
        raise _rejected(
            "yaw must be a finite number of radians"
        )

    return yaw


def _positive_of(payload: dict, name: str) -> float:
    """The quantity `name` a goal carries, which must be a finite number
    above zero, such as a scale or a mass."""
    value = float(payload[name])

    if not math.isfinite(value) or value <= 0.0:
        raise _rejected(f"{name} must be a finite number greater than zero")

    return value


def _physics_of(payload: dict) -> str:
    physics = str(payload["physics"]).lower()

    if physics not in _PHYSICS:
        raise _rejected("physics must be none, static, or dynamic")

    return physics


def _duration_of(payload: dict) -> float:
    duration_s = float(payload["duration_s"])

    if not math.isfinite(duration_s) or duration_s <= 0.0 or duration_s > _LONGEST_PUSH_S:
        raise _rejected(f"duration_s must be > 0 and <= {_LONGEST_PUSH_S:g} seconds")

    return duration_s


# The name of an object or of a feeder: lowercase letters, digits and `_`,
# at least one of them.
_NAME = re.compile(r"[a-z0-9_]+")


def _refuse_malformed_name(kind: str, text: str) -> None:
    """Refuses with `rejected` `text`, the `kind` a spawn names, when it is
    not a name."""

    if not _NAME.fullmatch(text):
        raise _rejected(
            f"{kind} '{text}' is not a name: lowercase letters, digits and _"
        )


def _object_name(asked: str | None, asset_id: str, taken: set[str]) -> str:
    """The name a spawn gets: the one it asks for, refused with `rejected`
    when it is not a name or is taken; with none, or an empty one, a name
    made from the last segment of the asset id, with `_2`, `_3`, ... until
    it is free."""

    if asked:
        _refuse_malformed_name("object name", asked)
        if asked in taken:
            raise _rejected(
                f"object name '{asked}' is taken by a spawned object; "
                "give another name, or leave name out and the simulation "
                "names it"
            )
        return asked
    last_segment = asset_id.rsplit("/", 1)[-1].lower()
    stem = re.sub(r"[^a-z0-9_]", "_", last_segment) or "object"
    numbered = (f"{stem}_{number}" for number in itertools.count(2))
    candidates = itertools.chain([stem], numbered)
    return next(name for name in candidates if name not in taken)


def _payload_of(operation: str, request) -> dict:
    """The fields a goal of `operation` carries, by name. A clear_scene goal
    carries none."""
    if operation == "clear_scene":
        return {}

    return dict(vars(request.data))


def _answering(module: ModuleType, handler: Callable) -> Callable:
    """`handler`, which gives each SceneFailure it raises as an error reply
    of the service `module`, with the same kind."""

    def answer(request):
        try:
            return handler(request)

        except SceneFailure as failure:
            raise module.Failure(failure.kind, failure.message) from None

    return answer


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


def _public_catalogue(assets) -> list[dict]:
    """The catalogue metadata a caller sees, without the raw Isaac paths:
    scenes first in their fixed order, then every other asset by kind,
    category and name."""

    public = [
        {
            "asset_id": asset.get("asset_id", ""),
            "display_name": asset.get("display_name", ""),
            "description": asset.get("description", ""),
            "kind": asset.get("kind", ""),
            "category": asset.get("category", ""),
        }
        for asset in assets
    ]

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
            scene_order.get(item["asset_id"], 999) if item["kind"] == "scene" else 0,
            item["display_name"].lower(),
            item["asset_id"],
        )
    )

    return public


# The state fields of a get_object answer, each present only for an object
# whose asset simulates it. This stage simulates none of them.
_NO_OBJECT_STATE = {
    "bounds": None,
    "breakage": None,
    "contents": None,
    "membranes": None,
    "crush": None,
    "heat": None,
    "heaters": None,
}


@dataclass
class _PendingCommand:
    operation: str
    payload: dict
    # Answers the goal with the command's result, or with the SceneFailure
    # of a command that did not run or gave no result.
    future: Future
    # Tells the goal that the Isaac main thread starts the command.
    started: Future = field(default_factory=Future)


def _settle(future: Future, value) -> None:
    """Hands `value` to the goal awaiting `future`, from the Isaac main
    thread. A goal that stopped awaiting it, as the node stops, has cancelled
    it, and is told nothing."""

    if future.set_running_or_notify_cancel():
        future.set_result(value)


def _settle_failure(future: Future, failure: SceneFailure) -> None:
    """Hands `failure` to the goal awaiting `future`, as _settle hands a
    value."""

    if future.set_running_or_notify_cancel():
        future.set_exception(failure)


async def _report_progress(
    context,
    operation: str,
    building: bool,
) -> bool:
    """Publishes one progress message of a load_scene or spawn_object goal,
    with nothing fetched and nothing to fetch: Isaac reads every asset from
    its own install. False, and logged, when the message was not sent within
    _REPORT_TIMEOUT_S or the publish failed."""

    try:
        await asyncio.wait_for(
            context.publish_feedback(
                bytes_fetched=0,
                files_ready=0,
                building=building,
                bytes_ready=0,
                bytes_total=0,
                bytes_per_second=0,
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
    and get_object_states, get_objects_list and get_object answer the latest
    capture. A load_scene or spawn_object goal reports its progress on its
    feedback (_submit_reporting_progress).
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
        # The catalogue as a caller sees it, and each asset's category,
        # both built once when the assets come.
        self._catalogue: list[dict] = []
        self._category_by_asset: dict[str, str] = {}
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

        # How each command is checked against the stage, and how it runs
        # on the Isaac main thread once it is.
        self._commands = {
            "load_scene": (self._scene_to_load, self._load_scene),
            "clear_scene": (lambda _payload: None, self._clear_scene),
            "spawn_object": (self._spawn_of, self._spawn_object),
            "move_object": (self._move_of, self._move_object),
            "remove_object": (self._object_to_remove, self._remove_object),
            "apply_force": (self._push_of, self._apply_force),
            "move_robot": (self._robot_move_of, self._move_robot),
        }

    async def start(self) -> None:
        """Expose Peppy services/actions and start their request loops."""

        self._action_handles = {
            operation: await module.ActionHandle.expose(self._node_runner)
            for operation, module in _ACTIONS.items()
        }

        services = (
            (get_assets_list, self._handle_get_assets),
            (get_asset_categories, self._handle_get_asset_categories),
            (get_objects_list, self._handle_get_objects_list),
            (get_object, self._handle_get_object),
            (get_scene, self._handle_get_scene),
            (get_object_states, self._handle_get_object_states),
            (get_robots_list, self._handle_get_robots),
        )

        self._tasks = [
            *(
                asyncio.create_task(self._serve_service(module, handler))
                for module, handler in services
            ),
            *(
                asyncio.create_task(self._serve_action(operation))
                for operation in _ACTIONS
            ),
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
            self._catalogue = _public_catalogue(self._assets.values())
            self._category_by_asset = {
                asset["asset_id"]: asset["category"]
                for asset in self._catalogue
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

    def _public_assets(self) -> list[dict]:
        """The catalogue as a caller sees it."""

        with self._lock:
            return self._catalogue

    def _require_catalogue(self) -> None:
        """Raises `precondition_failed` until the launcher has handed over
        the discovered catalogue."""

        with self._lock:
            ready = self._assets_ready

        if ready:
            return

        raise SceneFailure(
            PRECONDITION_FAILED,
            "Isaac is still discovering its asset catalogue; "
            "it is available once the stage has loaded",
        )

    def _catalogue_asset(self, asset_id: str, kind: str) -> dict:
        """The asset `asset_id` names in the catalogue, which must be of
        `kind`. Refused while the catalogue is not discovered yet, and for an
        id that names no asset of that kind."""

        self._require_catalogue()
        asset = self._asset(asset_id)

        if asset is None:
            raise _rejected(f"Unknown asset_id: {asset_id}")

        if asset.get("kind") != kind:
            raise _rejected(f"Asset is not {_ASSET_KIND_WORDS[kind]}: {asset_id}")

        return asset

    def _latest_snapshot(self) -> ObjectStateSnapshot:
        """The latest capture. Raises `precondition_failed` with why there is
        none: Isaac has not captured yet, or its last capture failed."""

        with self._lock:
            snapshot = self._snapshot
            unavailable = self._unavailable

        if snapshot is None:
            raise SceneFailure(PRECONDITION_FAILED, unavailable)

        return snapshot

    def _handle_get_assets(
        self,
        request,
    ) -> get_assets_list.Response:
        """The catalogue, narrowed by every filter the request sets: a
        kind that is neither scene nor object is refused, and the text is
        looked for in an entry's id, name and description whatever the
        case."""

        self._require_catalogue()

        kind = _filter(request.data.kind)
        category = _filter(request.data.category)
        query = _filter(request.data.query)

        if kind is not None and kind not in _ASSET_KINDS:
            raise _rejected("kind must be scene or object")

        assets = [
            asset
            for asset in self._public_assets()
            if _matches(asset, kind, category, query)
        ]

        return get_assets_list.Response(
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

        self._require_catalogue()

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
            raise _rejected("source must be spawned or scene")

        snapshot = self._latest_snapshot()

        with self._lock:
            category_by_asset = self._category_by_asset

        objects = []

        if source != "scene":
            for record in snapshot.objects:
                asset_category = category_by_asset.get(record.asset_id, "")

                if category is not None and category != asset_category:
                    continue

                objects.append(
                    get_objects_list.ResponseObjectsItem(
                        object_id=record.object_id,
                        name=record.name,
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
            timestamp=snapshot.timestamp_s,
            objects=objects,
        )

    def _handle_get_object(
        self,
        request,
    ) -> get_object.Response:
        """One spawned object from the latest capture: its pose and its
        scale, as get_objects_list lists them. This stage keeps no landmark
        and simulates no state of an object, so landmarks is empty and no
        state field is present. An object_id that names no spawned object is
        refused with `rejected`."""

        object_id = request.data.object_id
        snapshot = self._latest_snapshot()
        record = snapshot.record_of(object_id)

        if record is None:
            raise _rejected(f"Unknown object_id: {object_id}")

        return get_object.Response(
            timestamp=snapshot.timestamp_s,
            object_id=record.object_id,
            name=record.name,
            asset_id=record.asset_id,
            position=list(record.position),
            orientation=list(record.orientation),
            scale=record.scale,
            landmarks=[],
            **_NO_OBJECT_STATE,
        )

    def _handle_get_scene(
        self,
        _request,
    ) -> get_scene.Response:
        """The scene loaded last: no asset while the stage stands empty."""

        with self._lock:
            scene = dict(self._scene)

        return get_scene.Response(
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
        """The latest capture. No capture is not an empty scene: it gives an
        error reply with `precondition_failed`, as object_state:v1 says of
        an implementer that is not ready."""

        snapshot = self._latest_snapshot()

        return get_object_states.Response(
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
        return get_robots_list.Response(
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
        """Runs a command on the Isaac main thread and returns its result.
        Raises the SceneFailure of a command that did not run or gave no
        result."""
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
        and ends after a fresh object-state capture, so a read once it has
        ended observes its edit. A command the stage cannot take now gives
        its goal the SceneFailure of its check, and a command that raised
        anything else gives `attempt_failed`. Until the first stamped
        capture, every frame tries one, so reads are answered from the first
        frame on, ahead of the stream's first tick. After that the state tick
        and the capture after each edit keep the snapshot current, and bring
        it back after a failed read.
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

            failure = None

            try:
                result = self._execute(
                    launcher,
                    pending.operation,
                    pending.payload,
                )

            except SceneFailure as refused:
                logger.warning(
                    "Scene action '%s' did not run: %s",
                    pending.operation,
                    refused.message,
                )

                failure = refused

            except Exception as exc:
                logger.exception(
                    "Scene action '%s' failed",
                    pending.operation,
                )

                failure = SceneFailure(ATTEMPT_FAILED, str(exc))

            # Succeeded or failed partway, whatever the command did to the
            # scene is in the snapshot before its goal ends.
            self.capture_object_states()

            if failure is None:
                _settle(pending.future, result)
            else:
                _settle_failure(pending.future, failure)

    def _check(self, operation: str, payload: dict):
        """The command a goal of `operation` names, checked against the
        catalogue, the spawned objects and the robots as they are now.
        Raises SceneFailure for a command the stage cannot take."""

        check, _run = self._commands[operation]

        return check(payload)

    def _execute(
        self,
        launcher,
        operation: str,
        payload: dict,
    ) -> dict:
        """Execute one scene command from the Isaac simulation thread, and
        return its result: the fields of the goal's result. The command is
        checked again first, against the stage as it is now."""

        check, run = self._commands[operation]

        return run(launcher, check(payload))

    def _scene_to_load(self, payload: dict) -> _Load:
        asset = self._catalogue_asset(payload["asset_id"], "scene")

        return _Load(
            asset_id=payload["asset_id"],
            path=asset["path"],
            scale=_positive_of(payload, "scale"),
        )

    def _load_scene(self, launcher, load: _Load) -> dict:
        # scene_manipulation: spawned objects go first, then the scene is
        # replaced.
        self._remove_spawned_objects(launcher)

        launcher._runtime_load_isaac_scene(
            {
                "path": load.path,
                "scale": [load.scale, load.scale, load.scale],
            }
        )

        with self._lock:
            self._scene = {"asset_id": load.asset_id, "scale": load.scale}

        return {}

    def _clear_scene(self, launcher, _nothing) -> dict:
        self._remove_spawned_objects(launcher)

        launcher._runtime_clear_scene()

        with self._lock:
            self._scene = {"asset_id": "", "scale": 1.0}

        return {}

    def _spawn_of(self, payload: dict) -> _Spawn:
        _refuse_snap(payload)
        asset = self._catalogue_asset(payload["asset_id"], "object")
        feeder = payload.get("feeder")

        # Absent or empty, the spawn names no feeder.
        if feeder:
            _refuse_malformed_name("feeder", feeder)

        with self._lock:
            taken = {obj["name"] for obj in self._objects.values()}

        return _Spawn(
            asset_id=payload["asset_id"],
            path=asset["path"],
            position=_position_of(payload),
            yaw=_yaw_of(payload),
            orientation=_orientation_of(payload),
            scale=_positive_of(payload, "scale"),
            physics=_physics_of(payload),
            mass=_positive_of(payload, "mass"),
            name=_object_name(payload.get("name"), payload["asset_id"], taken),
        )

    def _spawn_object(self, launcher, spawn: _Spawn) -> dict:
        object_id = (
            "obj_"
            + uuid.uuid4().hex[:12]
        )

        command = {
            "name": object_id,
            "path": spawn.path,
            "position": spawn.position,
            "yaw": spawn.yaw,
            "scale": [
                spawn.scale,
                spawn.scale,
                spawn.scale,
            ],
            "physics": spawn.physics,
            "mass": spawn.mass,
        }

        if spawn.orientation is not None:
            command["orientation"] = spawn.orientation

        launcher._runtime_spawn_isaac_asset(command)

        with self._lock:
            self._objects[object_id] = {
                "object_id": object_id,
                "name": spawn.name,
                "asset_id": spawn.asset_id,
                "scale": spawn.scale,
                "physics": spawn.physics,
                "mass": spawn.mass,
            }

        return {"object_id": object_id, "name": spawn.name}

    def _spawned_object(self, object_id: str) -> dict:
        """The spawned object `object_id` names. Refused with `rejected` for
        an id that names none."""

        obj = self._object(object_id)

        if obj is None:
            raise _rejected(f"Unknown object_id: {object_id}")

        return obj

    def _move_of(self, payload: dict) -> _Move:
        _refuse_snap(payload)
        object_id = str(payload["object_id"])
        self._spawned_object(object_id)

        return _Move(
            object_id=object_id,
            position=_position_of(payload),
            orientation=_orientation_of(payload),
        )

    def _move_object(self, launcher, move: _Move) -> dict:
        command = {
            "name": move.object_id,
            "position": move.position,
        }

        if move.orientation is not None:
            command["orientation"] = move.orientation

        launcher._runtime_move_object(command)

        return {}

    def _object_to_remove(self, payload: dict) -> str:
        object_id = str(payload["object_id"])
        self._spawned_object(object_id)

        return object_id

    def _remove_object(self, launcher, object_id: str) -> dict:
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

        return {}

    def _push_of(self, payload: dict) -> _Push:
        """A push of a dynamic object: a static or visual-only object takes
        no force, so a push of one is refused."""

        object_id = str(payload["object_id"])
        obj = self._spawned_object(object_id)

        if str(obj.get("physics", "none")).lower() != "dynamic":
            raise _rejected(
                f"Force can only be applied to dynamic objects: {object_id}"
            )

        return _Push(
            object_id=object_id,
            force=_finite_vector(payload, "force", 3, "[Fx, Fy, Fz]"),
            duration_s=_duration_of(payload),
        )

    def _apply_force(self, launcher, push: _Push) -> dict:
        launcher._runtime_apply_force(
            {
                "name": push.object_id,
                "force": push.force,
                "duration_s": push.duration_s,
            }
        )

        return {}

    def _robot_move_of(self, payload: dict) -> _RobotMove:
        """A move of a robot the stage stands. A robot that does not stand on
        the stage is refused with `resource_unavailable`, and a position or a
        yaw that is not finite with `rejected`."""

        robot = payload["robot"]
        names = [
            each.instance
            for each in self._world.robots()
        ]

        if robot not in names:
            raise SceneFailure(
                RESOURCE_UNAVAILABLE,
                f"no robot stands as {robot!r} in this "
                f"simulation, which stands "
                f"{', '.join(names) or 'none'}",
            )

        return _RobotMove(
            robot=robot,
            position=_position_of(payload),
            yaw=_yaw_of(payload),
        )

    def _move_robot(self, launcher, move: _RobotMove) -> dict:
        launcher._runtime_move_robot_root(
            {
                "robot": move.robot,
                "position": move.position,
                "yaw": move.yaw,
            }
        )

        return {}

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

    async def _serve_service(self, module, handler) -> None:
        """Answers the requests of the service `module` with `handler` for
        the life of the node, each SceneFailure as an error reply; a round
        that fails is logged and the next one starts a second later."""

        name = module.__name__.rsplit(".", 1)[-1]
        answer = _answering(module, handler)

        while True:
            try:
                await module.handle_next_request(
                    self._node_runner,
                    answer,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "%s service failed",
                    name,
                )
                await asyncio.sleep(1.0)

    def _decide(self, module: ModuleType, operation: str, request):
        """Admits a goal of `operation` whose command the stage can take
        now, and refuses any other with the failure kind of its check."""

        try:
            self._check(operation, _payload_of(operation, request))

        except SceneFailure as failure:
            return module.GoalDecision.refuse(failure.kind, failure.message)

        return module.GoalDecision.accept()

    async def _serve_action(self, operation: str) -> None:
        """Serves the goals of the action `operation` for the life of the
        node, one at a time, in the order they come."""

        module = _ACTIONS[operation]
        handle = self._action_handles[operation]
        decide = partial(self._decide, module, operation)

        while True:
            context = await handle.handle_goal_next_request(decide)

            if context is None:
                return

            await self._run_goal(operation, context)

    async def _run_goal(self, operation: str, context) -> None:
        """Runs the command of one admitted goal and ends the goal: failed
        with the kind of a command that did not run or gave no result, and
        otherwise completed with its result. A cancel does not stop a
        command once it is queued, so a goal that received one still
        completes with what its command did."""

        payload = _payload_of(operation, context.request())

        try:
            if operation in _REPORTING_PROGRESS:
                result = await self._submit_reporting_progress(
                    context,
                    operation,
                    payload,
                )
            else:
                result = await self._submit(
                    operation,
                    payload,
                )

        except SceneFailure as failure:
            await context.fail(failure.kind, failure.message)
            return

        await context.complete(**result)
