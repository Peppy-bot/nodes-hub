#!/usr/bin/env python3
"""Isaac Sim SimLauncher for openarm_initializer."""

# pylint: disable=R0903

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from bridge_extension import IsaacBridgeExtension
from camera_common import FramePacer
from object_state import RUNTIME_OBJECTS_PATH, object_prim_path
from runtime_commander_server import RuntimeCommanderServer

logger = logging.getLogger(__name__)

# A loop iteration longer than this is logged with its phase breakdown: the
# state stream rides the loop, so a long iteration is a gap the backbone sees.
_SLOW_ITERATION_S = 0.05

_WARMUP_STEPS = 100

# Frames given to the stage after a robot joins or leaves, before the
# timeline plays again and after it does: the prim has to load and physics
# has to pick it up before a view of it can read.
_STAGE_SETTLE_STEPS = 30
_MAIN_RATE_LIMIT_ENABLED = "/app/runLoops/main/rateLimitEnabled"
# The renderer and anti-aliasing Kit actually runs; both are requested through
# SimulationApp's launch config and both can be changed by Kit afterwards.
_RENDER_MODE = "/rtx/rendermode"
_ANTI_ALIASING_OP = "/rtx/post/aa/op"

# The one prim a runtime scene (an Isaac environment or any USD) is referenced
# under. Loading a scene replaces whatever is there.
_RUNTIME_SCENE_PATH = "/World/RuntimeScene"

# The runtime commands that remove or replace the prim at a runtime object's
# path. Each refuses a name scene_manipulation spawned; a move keeps the prim.
_PRIM_REPLACING_COMMANDS = frozenset({"spawn_usd", "spawn_isaac_asset", "remove"})


class SimLauncher:
    def __init__(
        self,
        sim_app,
        world: World,
        edits: Edits,
        extension: IsaacBridgeExtension,
        ready: threading.Event,
        stop: threading.Event,
        io,
        scene_actions,
        frame_rate_hz: int,
        render_mode: str,
        anti_aliasing: int,
    ) -> None:
        self._sim_app = sim_app
        self._world = world
        self._edits = edits
        self._ready = ready
        self._stop = stop
        self._io = io
        self._scene_actions = scene_actions
        self._frame_rate_hz = frame_rate_hz
        self._render_mode = render_mode
        self._anti_aliasing = anti_aliasing
        self._timeline = None
        self._extension: Optional[IsaacBridgeExtension] = extension


        # Runtime-discovered NVIDIA Isaac prop catalogue.
        self._isaac_assets = {}


        self._runtime_commander = RuntimeCommanderServer(
            host="0.0.0.0",
            port=5556,
        )

    def run(self) -> None:
        # Everything from the stage load onward shares one cleanup path. Camera
        # setup, the warmup and the extension constructor all run before the
        # loop, and a failure in any of them still has to stop the timeline and
        # close Isaac; a bare raise here would strand a live SimulationApp.
        try:
            self._load_stage()

            logger.info(
                "Re-applying Isaac render settings after stage load"
            )
            self._sim_app.reset_render_settings()

            self._setup_environment()

            # Discover the NVIDIA Isaac props once during startup. The
            # resulting catalogue is cached for the Peppy asset-list service.
            self._isaac_assets = self._discover_isaac_props()

            self._isaac_assets.update(
                {
                    "scene/simple_warehouse": {
                        "asset_id": "scene/simple_warehouse",
                        "display_name": "Simple Warehouse",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Simple_Warehouse/"
                            "warehouse.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/flat_grid": {
                        "asset_id": "scene/flat_grid",
                        "display_name": "Flat Grid",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Grid/"
                            "default_environment.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/black_grid": {
                        "asset_id": "scene/black_grid",
                        "display_name": "Black Grid",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Grid/"
                            "gridroom_black.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/curved_grid": {
                        "asset_id": "scene/curved_grid",
                        "display_name": "Curved Grid",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Grid/"
                            "gridroom_curved.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/simple_room": {
                        "asset_id": "scene/simple_room",
                        "display_name": "Simple Room",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Simple_Room/"
                            "simple_room.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/office": {
                        "asset_id": "scene/office",
                        "display_name": "Office",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Office/"
                            "office.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/hospital": {
                        "asset_id": "scene/hospital",
                        "display_name": "Hospital",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Hospital/"
                            "hospital.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/warehouse_forklifts": {
                        "asset_id": "scene/warehouse_forklifts",
                        "display_name": "Warehouse + Forklifts",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Simple_Warehouse/"
                            "warehouse_with_forklifts.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/warehouse_multiple_shelves": {
                        "asset_id": "scene/warehouse_multiple_shelves",
                        "display_name": "Warehouse Multiple Shelves",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Simple_Warehouse/"
                            "warehouse_multiple_shelves.usd"
                        ),
                        "category": "Scenes",
                    },

                    "scene/full_warehouse": {
                        "asset_id": "scene/full_warehouse",
                        "display_name": "Full Warehouse",
                        "kind": "scene",
                        "path": (
                            "Isaac/Environments/Simple_Warehouse/"
                            "full_warehouse.usd"
                        ),
                        "category": "Scenes",
                    },
                }
            )

            self._scene_actions.set_assets(
                self._isaac_assets
            )

            # The rig is rendered for each robot that pairs a camera, and
            # the settings below are what a capture needs from the renderer.
            self._configure_camera_rendering()

            self._warmup()
            self._require_render_profile()
            self._start_timeline()

            self._extension.bind()

            self._runtime_commander.start()

            logger.info(
                "Runtime commander ready on TCP port 5556"
            )

            logger.info(
                "Scene loaded — waiting for bridge setup"
            )

            self._run_loop()

        except FileNotFoundError as exc:
            logger.error(str(exc))
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        except Exception:
            logger.exception("SimLauncher.run failed")
            raise
        finally:
            self._shutdown()

    def _load_stage(self) -> None:
        self._world.open()

    def unbind(self) -> None:
        """Lets go of the stage so it can be changed: every view on an
        articulation stops reading, and the timeline stops so the prims can
        move under it."""
        self._extension.unbind()
        if self._timeline is not None:
            self._timeline.stop()

    def rebind(self) -> None:
        """Plays the stage again, takes a view of every robot on it and
        resolves the views against the live stage, so a robot the engine
        cannot drive raises out of the stand that put it there. A robot that
        was commanded servos back to its setpoint as soon as its views read."""
        for _ in range(_STAGE_SETTLE_STEPS):
            self._sim_app.update()
        if self._timeline is not None:
            self._timeline.play()
        for _ in range(_STAGE_SETTLE_STEPS):
            self._sim_app.update()
        self._extension.bind()
        self._extension.resolve(self._sim_app.update, _STAGE_SETTLE_STEPS)

    def _setup_environment(self) -> None:
        """Environment is loaded on demand by the scene commander."""
        logger.info(
            "Startup environment disabled; "
            "waiting for runtime scene selection"
        )

    def _configure_camera_rendering(self) -> None:
        import carb.settings

        # Camera annotators are read right after each update; async rendering
        # would desynchronize their data from the physics state just stepped.
        carb.settings.get_settings().set("/app/asyncRendering", False)
        logger.info("Async rendering disabled for camera capture")

    def _warmup(self) -> None:
        for _ in range(_WARMUP_STEPS):
            self._sim_app.update()

    def _require_render_profile(self) -> None:
        """Refuse a render profile Kit changed underneath the launch.

        Kit keeps the requested renderer and anti-aliasing only while it can
        run them, and it swaps them without a word on the first rendered
        frames: DLSS needs the NGX core library, and without it Kit falls
        back to TAA and streams raw path-tracing noise at full frame rate.
        The check therefore runs after the warmup, once those frames are in.
        """
        import carb.settings

        settings = carb.settings.get_settings()
        actual = (settings.get(_RENDER_MODE), settings.get(_ANTI_ALIASING_OP))
        expected = (self._render_mode, self._anti_aliasing)

        if actual != expected:
            raise RuntimeError(
                f"Isaac is rendering with mode {actual[0]!r} and anti-aliasing "
                f"{actual[1]!r} instead of the requested {expected[0]!r} and "
                f"{expected[1]!r}; DLSS runs on the NGX core library "
                "(libnvidia-ngx.so.1) that the base image carries, see "
                "scripts/Dockerfile.isaac"
            )

    def _start_timeline(self) -> None:
        import omni.timeline

        self._timeline = (
            omni.timeline.get_timeline_interface()
        )

        self._timeline.play()

    # ------------------------------------------------------------------
    # Runtime commander
    # ------------------------------------------------------------------

    def execute_runtime_command(
        self,
        command: dict,
    ) -> None:
        """Execute one runtime command inside the Isaac simulation thread."""

        cmd = command.get("command")

        logger.info(
            "Runtime command received: %s",
            command,
        )

        # Every object-state capture reads each scene_manipulation object at its
        # prim. Removing or replacing one here would leave the registry
        # naming a prim that is gone or swapped, and every capture would
        # fail until a scene_manipulation edit. scene_manipulation itself calls the
        # _runtime_* methods directly, past this check.
        if (
            cmd in _PRIM_REPLACING_COMMANDS
            and self._scene_actions.owns(command.get("name"))
        ):
            raise ValueError(
                f"Runtime command '{cmd}' refused: '{command['name']}' is "
                "a scene_manipulation object; edit it through scene_manipulation"
            )

        if cmd == "move_robot_root":
            self._runtime_move_robot_root(command)

        elif cmd == "spawn_usd":
            self._runtime_spawn_usd(command)

        elif cmd == "move_object":
            self._runtime_move_object(command)

        elif cmd == "remove":
            self._runtime_remove(command)

        elif cmd == "load_usd_scene":
            self._runtime_load_usd_scene(command)

        elif cmd == "load_isaac_scene":
            self._runtime_load_isaac_scene(command)

        elif cmd == "spawn_isaac_asset":
            self._runtime_spawn_isaac_asset(command)

        elif cmd == "clear_runtime_scene":
            self._runtime_clear_scene()

        else:
            raise ValueError(
                f"Unknown runtime command: {cmd}"
            )

    # ------------------------------------------------------------------
    # Robot control
    # ------------------------------------------------------------------

    def _runtime_move_robot_root(
        self,
        command: dict,
    ) -> None:
        """Puts one robot somewhere else in the live stage, facing the yaw
        the command names, leaving every other robot standing where it is.
        The world's record moves with it, so the scene reports where the
        robot now stands and the next robot the engine places is given room
        from there."""
        self._world.move(command["robot"], command["position"], command["yaw"])

    # ------------------------------------------------------------------
    # Runtime scene control
    # ------------------------------------------------------------------

    def _runtime_apply_physics(
        self,
        root_prim,
        mode: str,
        mass: float = 0.1,
    ) -> None:
        """Apply static or dynamic physics to a referenced USD hierarchy."""

        from pxr import Usd, UsdGeom, UsdPhysics

        mode = str(mode).lower()

        if mode == "none":
            return

        if mode not in ("static", "dynamic"):
            raise ValueError(
                "physics must be one of: none, static, dynamic"
            )

        collider_count = 0

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue

            # Apply collisions to actual geometry prims.
            if prim.IsA(UsdGeom.Gprim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI.Apply(
                        prim
                    )
                else:
                    collision_api = UsdPhysics.CollisionAPI(
                        prim
                    )

                collision_api.CreateCollisionEnabledAttr(
                    True
                )

                collider_count += 1

                # Meshes need an explicit collision representation.
                if prim.IsA(UsdGeom.Mesh):
                    if not prim.HasAPI(
                        UsdPhysics.MeshCollisionAPI
                    ):
                        mesh_api = (
                            UsdPhysics.MeshCollisionAPI.Apply(
                                prim
                            )
                        )
                    else:
                        mesh_api = (
                            UsdPhysics.MeshCollisionAPI(
                                prim
                            )
                        )

                    # Dynamic triangle meshes are generally unsuitable
                    # as rigid-body colliders, so use a convex hull.
                    # Static geometry can use its authored mesh directly.
                    approximation = (
                        "convexHull"
                        if mode == "dynamic"
                        else "none"
                    )

                    mesh_api.CreateApproximationAttr().Set(
                        approximation
                    )

        if collider_count == 0:
            logger.warning(
                "No collision-capable geometry found below %s",
                root_prim.GetPath(),
            )

        if mode == "dynamic":
            if not root_prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(
                    root_prim
                )
            else:
                rigid_api = UsdPhysics.RigidBodyAPI(
                    root_prim
                )

            rigid_api.CreateRigidBodyEnabledAttr(
                True
            )

            if not root_prim.HasAPI(
                UsdPhysics.MassAPI
            ):
                mass_api = UsdPhysics.MassAPI.Apply(
                    root_prim
                )
            else:
                mass_api = UsdPhysics.MassAPI(
                    root_prim
                )

            mass_api.CreateMassAttr(
                float(mass)
            )

        logger.info(
            "Applied runtime physics: mode=%s mass=%s "
            "colliders=%d root=%s",
            mode,
            mass,
            collider_count,
            root_prim.GetPath(),
        )

    # ------------------------------------------------------------------
    # Stage edits
    # ------------------------------------------------------------------

    def _remove_prim(self, stage, path: str) -> bool:
        """Remove a prim if it exists and take every cached physics view again.

        PhysX rebuilds its tensor views when a prim leaves the stage, and the
        Articulation handles the bridge and the runtime commander hold keep
        failing afterwards ('Articulation' object has no attribute
        '_physics_view') until they are created again, as does the
        rigid-body view the object state reads. Every removal goes through
        here: the bridge's views are dropped before the prim goes and taken
        again after, and the commander's are re-created on their next use.
        Adding prims leaves them intact.
        """

        if not stage.GetPrimAtPath(path).IsValid():
            return False

        # A view of a prim that is edited under it stops reading, so the
        # views go before the prim does and are taken again after; they read
        # on the steps that follow.
        if self._extension is not None:
            self._extension.unbind()

        stage.RemovePrim(path)

        self._scene_actions.invalidate_physics_views()

        if self._extension is not None:
            self._extension.bind()

        return True

    def _reference_runtime_scene(self, stage, usd_path: str, scale) -> None:
        """Replace the runtime scene with a reference to usd_path at scale."""

        from pxr import Gf, Sdf, UsdGeom

        if len(scale) != 3:
            raise ValueError(
                "Runtime scene scale requires exactly 3 values"
            )

        if self._remove_prim(stage, _RUNTIME_SCENE_PATH):
            logger.info(
                "Replacing runtime scene %s",
                _RUNTIME_SCENE_PATH,
            )

        prim = stage.DefinePrim(
            Sdf.Path(_RUNTIME_SCENE_PATH),
            "Xform",
        )

        prim.GetReferences().AddReference(
            usd_path
        )

        xformable = UsdGeom.Xformable(prim)

        scale_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )

        if scale_op is None:
            scale_op = xformable.AddScaleOp()

        scale_op.Set(
            Gf.Vec3f(
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
            )
        )

    def _runtime_clear_scene(self) -> None:
        """Remove the currently loaded runtime scene."""

        import omni.usd

        stage = omni.usd.get_context().get_stage()

        if self._remove_prim(stage, _RUNTIME_SCENE_PATH):
            logger.info(
                "Removed runtime scene %s",
                _RUNTIME_SCENE_PATH,
            )
        else:
            logger.info(
                "No runtime scene to remove"
            )

    def _runtime_load_usd_scene(
        self,
        command: dict,
    ) -> None:
        """Replace the current runtime scene with an arbitrary USD."""

        import omni.usd

        usd_path = str(
            Path(command["path"])
            .expanduser()
            .resolve()
        )

        if not Path(usd_path).exists():
            raise FileNotFoundError(
                f"Runtime scene USD does not exist: {usd_path}"
            )

        scale = command.get(
            "scale",
            [1.0, 1.0, 1.0],
        )

        stage = omni.usd.get_context().get_stage()

        self._reference_runtime_scene(stage, usd_path, scale)

        logger.info(
            "Loaded runtime USD scene %s at scale %s",
            usd_path,
            scale,
        )

    def _resolve_isaac_asset_path(
        self,
        relative_path: str,
    ) -> str:
        """Resolve an Isaac/... asset path against Isaac Sim's asset root."""

        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()

        if not assets_root:
            raise RuntimeError(
                "Isaac asset root could not be resolved"
            )

        relative_path = str(relative_path).lstrip("/")

        full_path = (
            assets_root.rstrip("/")
            + "/"
            + relative_path
        )

        logger.info(
            "Resolved Isaac asset: %s -> %s",
            relative_path,
            full_path,
        )

        return full_path

    def _discover_isaac_props(self) -> dict:
        """Discover USD assets recursively beneath ``Isaac/Props``.

        The catalogue is built once during startup. Public callers use the
        generated ``asset_id`` while the raw Isaac asset path remains private
        to the simulation node.

        Example asset ID::

            props/ycb/axis_aligned/003_cracker_box

        Internal Isaac path::

            Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd
        """

        import omni.client

        props_root = self._resolve_isaac_asset_path(
            "Isaac/Props"
        )

        catalogue = {}

        def walk(
            remote_dir: str,
            relative_dir: str,
        ) -> None:
            result, entries = omni.client.list(
                remote_dir
            )

            if result != omni.client.Result.OK:
                logger.debug(
                    "Could not list Isaac asset directory %s: %s",
                    remote_dir,
                    result,
                )
                return

            for entry in entries:
                relative_name = getattr(
                    entry,
                    "relative_path",
                    None,
                )

                if not relative_name:
                    relative_name = getattr(
                        entry,
                        "path",
                        None,
                    )

                if not relative_name:
                    continue

                relative_name = str(
                    relative_name
                ).strip("/")

                if not relative_name:
                    continue

                child_remote = (
                    remote_dir.rstrip("/")
                    + "/"
                    + relative_name
                )

                child_relative = (
                    relative_dir.rstrip("/")
                    + "/"
                    + relative_name
                ).strip("/")

                lower_name = relative_name.lower()

                # Ignore generated thumbnail assets.
                if (
                    "/.thumbs/" in ("/" + child_relative.lower() + "/")
                    or child_relative.lower().startswith(".thumbs/")
                ):
                    continue

                if lower_name.endswith(
                    (
                        ".usd",
                        ".usda",
                        ".usdc",
                    )
                ):
                    isaac_path = (
                        "Isaac/Props/"
                        + child_relative
                    )

                    asset_id = (
                        "props/"
                        + child_relative.rsplit(".", 1)[0]
                    ).lower()

                    asset_id = (
                        asset_id
                        .replace(" ", "_")
                        .replace("\\", "/")
                    )

                    display_name = (
                        relative_name.rsplit(".", 1)[0]
                        .replace("_", " ")
                        .replace("-", " ")
                        .strip()
                    )

                    category = (
                        child_relative.split("/", 1)[0]
                        if "/" in child_relative
                        else "Props"
                    )

                    catalogue[asset_id] = {
                        "asset_id": asset_id,
                        "display_name": display_name,
                        "kind": "object",
                        "path": isaac_path,
                        "category": category,
                    }

                    continue

                # Recurse only into entries that can contain children.
                flags = getattr(
                    entry,
                    "flags",
                    0,
                )

                if (
                    flags
                    & omni.client.ItemFlags.CAN_HAVE_CHILDREN
                ):
                    walk(
                        child_remote,
                        child_relative,
                    )

        logger.info(
            "Discovering Isaac props beneath %s",
            props_root,
        )

        walk(
            props_root,
            "",
        )

        logger.info(
            "Discovered %d Isaac prop assets",
            len(catalogue),
        )

        # Log a small sample only; the full Props tree can be large.
        for asset_id in list(sorted(catalogue))[:25]:
            asset = catalogue[asset_id]

            logger.info(
                "Isaac asset: %s -> %s",
                asset_id,
                asset["path"],
            )

        if len(catalogue) > 25:
            logger.info(
                "... %d additional Isaac assets not shown",
                len(catalogue) - 25,
            )

        return catalogue

    def get_isaac_assets(self) -> dict:
        """Return a defensive copy of the cached Isaac prop catalogue."""

        return {
            asset_id: dict(asset)
            for asset_id, asset in self._isaac_assets.items()
        }

    def _runtime_load_isaac_scene(
        self,
        command: dict,
    ) -> None:
        """Replace the runtime scene with one from Isaac's asset root."""

        import omni.usd

        usd_path = self._resolve_isaac_asset_path(
            command["path"]
        )

        scale = command.get(
            "scale",
            [1.0, 1.0, 1.0],
        )

        stage = omni.usd.get_context().get_stage()

        self._reference_runtime_scene(stage, usd_path, scale)

        logger.info(
            "Isaac scene reference added %s at scale %s",
            usd_path,
            scale,
        )

    def _runtime_spawn_isaac_asset(
        self,
        command: dict,
    ) -> None:
        """Spawn an object directly from NVIDIA's Isaac asset root."""

        resolved = dict(command)

        resolved["path"] = self._resolve_isaac_asset_path(
            command["path"]
        )

        resolved["command"] = "spawn_usd"

        self._runtime_spawn_usd(
            resolved,
            allow_remote=True,
        )

    def _runtime_spawn_usd(
        self,
        command: dict,
        allow_remote: bool = False,
    ) -> None:
        import omni.usd

        from pxr import (
            Gf,
            Sdf,
            UsdGeom,
        )

        name = command["name"]

        raw_path = str(command["path"])

        if allow_remote:
            usd_path = raw_path
        else:
            usd_path = str(
                Path(raw_path)
                .expanduser()
                .resolve()
            )

        position = command.get(
            "position",
            [0.0, 0.0, 0.0],
        )

        # Rotation about +z in radians; 0 = as authored.
        yaw = float(
            command.get(
                "yaw",
                0.0,
            )
        )

        scale = command.get(
            "scale",
            [1.0, 1.0, 1.0],
        )

        if not allow_remote and not Path(usd_path).exists():
            raise FileNotFoundError(
                f"Runtime USD does not exist: {usd_path}"
            )

        stage = (
            omni.usd
            .get_context()
            .get_stage()
        )

        if not stage.GetPrimAtPath(
            RUNTIME_OBJECTS_PATH
        ).IsValid():
            stage.DefinePrim(
                Sdf.Path(RUNTIME_OBJECTS_PATH),
                "Xform",
            )

        prim_path = object_prim_path(name)

        self._remove_prim(stage, prim_path)

        prim = stage.DefinePrim(
            Sdf.Path(prim_path),
            "Xform",
        )

        prim.GetReferences().AddReference(
            usd_path
        )

        xformable = UsdGeom.Xformable(
            prim
        )

        # Converted OBJ -> USD assets commonly already contain
        # translate/orient/scale xform ops. Reuse an existing scale op
        # instead of trying to create a duplicate.
        scale_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        ) or xformable.AddScaleOp()

        scale_op.Set(
            Gf.Vec3f(
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
            )
        )

        physics_mode = command.get(
            "physics",
            "none",
        )

        mass = float(
            command.get(
                "mass",
                0.1,
            )
        )

        # The object stands where a robot would: turned about its own
        # origin, then moved, with its scale applied before both.
        self._world.place(
            prim,
            [float(value) for value in position],
            yaw,
        )

        self._runtime_apply_physics(
            prim,
            physics_mode,
            mass,
        )

        logger.info(
            "Spawned runtime object '%s' from %s at %s, at a yaw of %s rad",
            name,
            usd_path,
            position,
            yaw,
        )

    def _runtime_apply_force(
        self,
        command: dict,
    ) -> None:
        """Apply a timed world-frame force to a dynamic runtime object."""

        import time

        import omni.usd

        from pxr import (
            Gf,
            PhysxSchema,
            UsdPhysics,
        )

        name = str(
            command["name"]
        )

        force = [
            float(value)
            for value in command["force"]
        ]

        duration_s = float(
            command["duration_s"]
        )

        if len(force) != 3:
            raise ValueError(
                "force must have exactly 3 values"
            )

        stage = (
            omni.usd
            .get_context()
            .get_stage()
        )

        prim_path = object_prim_path(name)

        prim = stage.GetPrimAtPath(
            prim_path
        )

        if not prim.IsValid():
            raise RuntimeError(
                "Runtime object does not exist: "
                f"{name}"
            )

        if not prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            raise RuntimeError(
                "Runtime object is not a rigid body: "
                f"{name}. Spawn it with "
                "Physics=dynamic."
            )

        force_api = (
            PhysxSchema.PhysxForceAPI.Get(
                stage,
                prim.GetPath(),
            )
        )

        if not force_api:
            force_api = (
                PhysxSchema.PhysxForceAPI.Apply(
                    prim
                )
            )

        if not force_api:
            raise RuntimeError(
                "Could not apply PhysxForceAPI "
                f"to {prim_path}"
            )

        # Real force in Newtons rather than acceleration.
        force_api.CreateModeAttr().Set(
            "force"
        )

        # Interpret X/Y/Z in world coordinates.
        force_api.CreateWorldFrameEnabledAttr().Set(
            True
        )

        force_api.CreateForceAttr().Set(
            Gf.Vec3f(
                float(force[0]),
                float(force[1]),
                float(force[2]),
            )
        )

        force_api.CreateTorqueAttr().Set(
            Gf.Vec3f(
                0.0,
                0.0,
                0.0,
            )
        )

        force_api.CreateForceEnabledAttr().Set(
            True
        )

        if not hasattr(
            self,
            "_runtime_force_deadlines",
        ):
            self._runtime_force_deadlines = {}

        self._runtime_force_deadlines[
            prim_path
        ] = (
            time.monotonic()
            + duration_s
        )

        logger.info(
            "Applied runtime force %s N to %s "
            "for %.3f s",
            force,
            prim_path,
            duration_s,
        )

    def _update_runtime_forces(
        self,
    ) -> None:
        """Disable runtime forces whose requested duration has elapsed."""

        import time

        deadlines = getattr(
            self,
            "_runtime_force_deadlines",
            None,
        )

        if not deadlines:
            return

        now = time.monotonic()

        expired = [
            prim_path
            for prim_path, deadline
            in list(deadlines.items())
            if now >= deadline
        ]

        if not expired:
            return

        import omni.usd

        from pxr import (
            Gf,
            PhysxSchema,
        )

        stage = (
            omni.usd
            .get_context()
            .get_stage()
        )

        for prim_path in expired:
            prim = stage.GetPrimAtPath(
                prim_path
            )

            if prim.IsValid():
                force_api = (
                    PhysxSchema
                    .PhysxForceAPI
                    .Get(
                        stage,
                        prim.GetPath(),
                    )
                )

                if force_api:
                    force_api.CreateForceAttr().Set(
                        Gf.Vec3f(
                            0.0,
                            0.0,
                            0.0,
                        )
                    )

                    force_api.CreateForceEnabledAttr().Set(
                        False
                    )

            deadlines.pop(
                prim_path,
                None,
            )

            logger.info(
                "Stopped runtime force on %s",
                prim_path,
            )

    def _runtime_move_object(
        self,
        command: dict,
    ) -> None:
        import omni.usd

        from pxr import (
            Gf,
            UsdGeom,
        )

        name = command["name"]

        position = command["position"]

        stage = (
            omni.usd
            .get_context()
            .get_stage()
        )

        prim_path = object_prim_path(name)

        prim = stage.GetPrimAtPath(
            prim_path
        )

        if not prim.IsValid():
            raise RuntimeError(
                f"Runtime object does not exist: {name}"
            )

        xformable = UsdGeom.Xformable(
            prim
        )

        translate_op = None

        for op in xformable.GetOrderedXformOps():
            if (
                op.GetOpType()
                ==
                UsdGeom.XformOp.TypeTranslate
            ):
                translate_op = op
                break

        if translate_op is None:
            translate_op = (
                xformable.AddTranslateOp()
            )

        translate_op.Set(
            Gf.Vec3d(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        )

        logger.info(
            "Moved runtime object '%s' to %s",
            name,
            position,
        )

    def _runtime_remove(
        self,
        command: dict,
    ) -> None:
        import omni.usd

        name = command["name"]

        stage = omni.usd.get_context().get_stage()

        if not self._remove_prim(stage, object_prim_path(name)):
            logger.warning(
                "Runtime object '%s' does not exist",
                name,
            )
            return

        logger.info(
            "Removed runtime object '%s'",
            name,
        )

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        import carb.settings

        # Cache the interface on Isaac's main thread, after SimulationApp exists.
        settings = carb.settings.get_settings()
        pacer = FramePacer(self._frame_rate_hz)
        while self._sim_app.is_running() and not self._stop.is_set():
            # Pace from the start of the whole iteration on the wall clock.
            # Recheck both shutdown and the deadline whenever a wait returns.
            iteration_start = time.monotonic()
            if not pacer.take_if_due(iteration_start):
                self._stop.wait(pacer.seconds_until_due(iteration_start))
                continue

            # Streaming startup and (re)connection can enable an app-only
            # limiter. In both window modes, Python owns whole-iteration pacing.
            if settings.get_as_bool(_MAIN_RATE_LIMIT_ENABLED):
                settings.set_bool(_MAIN_RATE_LIMIT_ENABLED, False)

            # Isaac advances physics inside update(); we then drive the
            # bridge step on the same thread (Articulation reads require
            # Isaac's main thread). The extension defers its own setup until
            # the stage is live, so early steps are cheap no-ops.
            self._sim_app.update()
            update_s = time.monotonic() - iteration_start

            if self._extension is not None:
                self._extension.step()

                if (
                    self._extension.is_ready
                    and not self._ready.is_set()
                ):
                    self._ready.set()

                    logger.info(
                        "Scene loaded; states will flow"
                    )

            # Execute commands received by commander.py.
            #
            # This deliberately happens in the Isaac simulation
            # thread rather than the TCP listener thread.
            self._runtime_commander.process_pending(
                self
            )

            # Execute Peppy scene actions on the Isaac thread.
            self._scene_actions.process_pending(
                self
            )

            # Stand the robots that joined and take out the ones that left,
            # on this thread, which is the only one that may touch the stage.
            self._edits.drain()

            self._update_runtime_forces()

            iteration_s = time.monotonic() - iteration_start
            if iteration_s > _SLOW_ITERATION_S:
                logger.warning(
                    "slow loop iteration %.1f ms (update %.1f ms, bridge and runtime %.1f ms)",
                    iteration_s * 1e3,
                    update_s * 1e3,
                    (iteration_s - update_s) * 1e3,
                )

    def _shutdown(self) -> None:
        self._ready.clear()

        # Nothing drains the queue once this loop is over, and a robot
        # joining or leaving waits on its edit with no deadline, so every
        # one still queued is failed here.
        self._edits.cancel_all("the simulation is shutting down")

        try:
            self._runtime_commander.stop()
        except Exception:
            logger.exception(
                "Runtime commander shutdown failed"
            )

        if self._extension is not None:
            # An extension shutdown failure must not strand the Isaac process:
            # timeline.stop + sim_app.close still need to run.
            try:
                self._extension.shutdown()
            except Exception:
                logger.exception(
                    "IsaacBridgeExtension shutdown failed"
                )

        if self._timeline is not None:
            self._timeline.stop()

        self._sim_app.close()

        logger.info(
            "Isaac Sim closed."
        )
