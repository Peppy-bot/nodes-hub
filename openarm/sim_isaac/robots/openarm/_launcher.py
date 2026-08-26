#!/usr/bin/env python3
"""Isaac Sim SimLauncher for openarm_robot_initializer."""

# pylint: disable=R0903

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from bridge_extension import IsaacBridgeExtension
from runtime_commander_server import RuntimeCommanderServer

logger = logging.getLogger(__name__)

_WARMUP_STEPS = 100


class SimLauncher:
    def __init__(
        self,
        sim_app,
        usd_path: Path,
        ready: threading.Event,
        stop: threading.Event,
        io,
        scene_actions,
        state_rate_hz: int,
        cameras_enabled: bool,
    ) -> None:
        self._sim_app = sim_app
        self._usd_path = usd_path
        self._ready = ready
        self._stop = stop
        self._io = io
        self._scene_actions = scene_actions
        self._state_rate_hz = state_rate_hz
        self._cameras_enabled = cameras_enabled
        self._timeline = None
        self._extension: Optional[IsaacBridgeExtension] = None

        self._runtime_robot = None

        # Runtime-discovered NVIDIA Isaac prop catalogue.
        self._isaac_assets = {}

        self._runtime_arm_targets = {
            "left": None,
            "right": None,
        }

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

            if self._cameras_enabled:
                self._configure_camera_rendering()

            self._warmup()
            self._start_timeline()

            self._extension = IsaacBridgeExtension(
                self._io,
                self._state_rate_hz,
                self._cameras_enabled,
            )

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
        import omni.usd

        if not self._usd_path.exists():
            raise FileNotFoundError(
                f"USD not found at {self._usd_path}"
                " — assets should be baked into the container image"
            )

        logger.info(
            "Loading stage: %s",
            self._usd_path,
        )

        omni.usd.get_context().open_stage(
            str(self._usd_path)
        )

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

        if cmd == "move_arm":
            self._runtime_move_arm(command)

        elif cmd == "release_arm":
            self._runtime_release_arm(command)

        elif cmd == "move_robot_root":
            self._runtime_move_robot_root(command)

        elif cmd == "spawn_usd":
            self._runtime_spawn_usd(command)

        elif cmd == "move_object":
            self._runtime_move_object(command)

        elif cmd == "remove":
            self._runtime_remove(command)

        elif cmd == "list_joints":
            self._runtime_list_joints()

        elif cmd == "load_tabletop_scene":
            self._runtime_load_tabletop_scene(command)

        elif cmd == "load_shelf_reach_scene":
            self._runtime_load_shelf_reach_scene(command)

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

    def _get_runtime_robot(self):
        """Return the runtime OpenArm articulation."""

        if self._runtime_robot is not None:
            return self._runtime_robot

        from isaacsim.core.prims import Articulation

        self._runtime_robot = Articulation(
            "/openarm"
        )

        self._runtime_robot.initialize()

        logger.info(
            "Runtime OpenArm articulation initialized"
        )

        logger.info(
            "Runtime OpenArm DOFs: %s",
            list(self._runtime_robot.dof_names),
        )

        return self._runtime_robot

    def _runtime_list_joints(self) -> None:
        robot = self._get_runtime_robot()

        logger.info(
            "OPENARM JOINTS: %s",
            list(robot.dof_names),
        )

    def _runtime_move_arm(
        self,
        command: dict,
    ) -> None:
        """Store a 7-DOF arm target for persistent runtime control."""

        side = str(
            command["side"]
        ).lower()

        if side not in ("left", "right"):
            raise ValueError(
                "side must be 'left' or 'right'"
            )

        targets = [
            float(value)
            for value in command["positions"]
        ]

        if len(targets) != 7:
            raise ValueError(
                "Arm command requires exactly 7 joint positions"
            )

        self._runtime_arm_targets[side] = targets

        logger.info(
            "Runtime %s arm override enabled: %s",
            side,
            targets,
        )

    def _runtime_release_arm(
        self,
        command: dict,
    ) -> None:
        """Release a runtime arm override back to the Peppy bridge."""

        side = str(
            command["side"]
        ).lower()

        if side not in ("left", "right"):
            raise ValueError(
                "side must be 'left' or 'right'"
            )

        self._runtime_arm_targets[side] = None

        logger.info(
            "Runtime %s arm override released",
            side,
        )

    def _apply_runtime_arm_targets(self) -> None:
        """Re-apply active arm targets every simulation frame."""

        import numpy as np

        if not any(
            target is not None
            for target in self._runtime_arm_targets.values()
        ):
            return

        robot = self._get_runtime_robot()
        joint_names = list(robot.dof_names)

        for side in ("left", "right"):
            targets = self._runtime_arm_targets[side]

            if targets is None:
                continue

            arm_joint_names = [
                f"openarm_{side}_joint{i}"
                for i in range(1, 8)
            ]

            indices = np.array(
                [
                    joint_names.index(name)
                    for name in arm_joint_names
                ],
                dtype=np.int32,
            )

            robot.set_joint_position_targets(
                np.array(
                    targets,
                    dtype=np.float32,
                ),
                joint_indices=indices,
            )

    def _runtime_move_robot_root(
        self,
        command: dict,
    ) -> None:
        """Translate the whole OpenArm root prim in the live stage."""

        import omni.usd

        from pxr import Gf, UsdGeom

        position = command["position"]

        if len(position) != 3:
            raise ValueError(
                "Robot root position requires exactly 3 values: x y z"
            )

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/openarm")

        if not prim.IsValid():
            raise RuntimeError(
                "OpenArm root prim /openarm does not exist"
            )

        xformable = UsdGeom.Xformable(prim)
        translate_op = None

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()

        translate_op.Set(
            Gf.Vec3d(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        )

        logger.info(
            "Moved OpenArm root to %s",
            position,
        )

    # ------------------------------------------------------------------
    # Runtime scene control
    # ------------------------------------------------------------------

    def _runtime_load_tabletop_scene(self) -> None:
        """Create an amber table, red cube, and blue tray in front of OpenArm."""

        import omni.usd

        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()

        # --------------------------------------------------------------
        # Read the current OpenArm world pose.
        # --------------------------------------------------------------

        robot_prim = stage.GetPrimAtPath("/openarm")

        if not robot_prim.IsValid():
            raise RuntimeError(
                "OpenArm root prim /openarm does not exist"
            )

        robot_xform = UsdGeom.Xformable(robot_prim)

        world_transform = robot_xform.ComputeLocalToWorldTransform(
            0.0
        )

        robot_position = world_transform.ExtractTranslation()

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])
        robot_z = float(robot_position[2])

        # Treat OpenArm local +X as forward and local +Y as its lateral
        # direction. Using the robot transform makes the placement follow
        # the current robot pose rather than fixed warehouse coordinates.
        forward = world_transform.TransformDir(
            Gf.Vec3d(1.0, 0.0, 0.0)
        )

        lateral = world_transform.TransformDir(
            Gf.Vec3d(0.0, 1.0, 0.0)
        )

        if forward.GetLength() < 1e-6:
            forward = Gf.Vec3d(1.0, 0.0, 0.0)
        else:
            forward.Normalize()

        if lateral.GetLength() < 1e-6:
            lateral = Gf.Vec3d(0.0, 1.0, 0.0)
        else:
            lateral.Normalize()

        logger.info(
            "Current OpenArm root: "
            "(%.3f, %.3f, %.3f), "
            "forward=(%.3f, %.3f, %.3f)",
            robot_x,
            robot_y,
            robot_z,
            float(forward[0]),
            float(forward[1]),
            float(forward[2]),
        )

        # --------------------------------------------------------------
        # Workcell placement.
        #
        # The table centre is 0.90 m forward from the current robot root.
        # The red cube is slightly nearer and to one side.
        # The blue tray is slightly farther and on the opposite side.
        # --------------------------------------------------------------

        forward_offset = 0.90

        table_center = (
            Gf.Vec3d(
                robot_x,
                robot_y,
                robot_z,
            )
            + forward * forward_offset
        )

        table_x = float(table_center[0])
        table_y = float(table_center[1])

        # Table geometry:
        # centre Z = root Z + 0.72
        # height    = 0.08 m
        # top Z     = root Z + 0.76
        table_center_z = robot_z + 0.72
        table_top_z = robot_z + 0.76

        # --------------------------------------------------------------
        # Scene root.
        # --------------------------------------------------------------

        root_path = "/World/RuntimeObjects/Tabletop"

        existing = stage.GetPrimAtPath(root_path)

        if existing.IsValid():
            stage.RemovePrim(root_path)

        stage.DefinePrim(
            Sdf.Path(root_path),
            "Xform",
        )

        # --------------------------------------------------------------
        # Materials.
        # --------------------------------------------------------------

        def create_material(
            path,
            color,
            roughness=0.4,
        ):
            material = UsdShade.Material.Define(
                stage,
                path,
            )

            shader = UsdShade.Shader.Define(
                stage,
                path + "/Shader",
            )

            shader.CreateIdAttr(
                "UsdPreviewSurface"
            )

            shader.CreateInput(
                "diffuseColor",
                Sdf.ValueTypeNames.Color3f,
            ).Set(
                Gf.Vec3f(
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                )
            )

            shader.CreateInput(
                "roughness",
                Sdf.ValueTypeNames.Float,
            ).Set(
                float(roughness)
            )

            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(),
                "surface",
            )

            return material

        def bind_material(
            prim,
            material,
        ):
            UsdShade.MaterialBindingAPI(
                prim
            ).Bind(
                material
            )

        amber = create_material(
            root_path + "/Materials/Amber",
            (0.95, 0.52, 0.08),
            roughness=0.35,
        )

        red = create_material(
            root_path + "/Materials/Red",
            (0.90, 0.03, 0.03),
            roughness=0.30,
        )

        blue = create_material(
            root_path + "/Materials/Blue",
            (0.03, 0.15, 0.90),
            roughness=0.30,
        )

        # --------------------------------------------------------------
        # Amber table.
        # --------------------------------------------------------------

        table = UsdGeom.Cube.Define(
            stage,
            root_path + "/Table",
        )

        table.CreateSizeAttr(1.0)

        table_xform = UsdGeom.Xformable(
            table.GetPrim()
        )

        table_xform.AddTranslateOp().Set(
            Gf.Vec3d(
                table_x,
                table_y,
                table_center_z,
            )
        )

        table_xform.AddScaleOp().Set(
            Gf.Vec3f(
                0.75,
                1.20,
                0.08,
            )
        )

        bind_material(
            table.GetPrim(),
            amber,
        )

        # --------------------------------------------------------------
        # Red pick cube.
        # --------------------------------------------------------------

        cube_center = (
            Gf.Vec3d(
                table_x,
                table_y,
                table_top_z + 0.04,
            )
            - forward * 0.18
            - lateral * 0.20
        )

        cube = UsdGeom.Cube.Define(
            stage,
            root_path + "/RedCube",
        )

        cube.CreateSizeAttr(1.0)

        cube_xform = UsdGeom.Xformable(
            cube.GetPrim()
        )

        cube_xform.AddTranslateOp().Set(
            Gf.Vec3d(
                float(cube_center[0]),
                float(cube_center[1]),
                table_top_z + 0.04,
            )
        )

        cube_xform.AddScaleOp().Set(
            Gf.Vec3f(
                0.08,
                0.08,
                0.08,
            )
        )

        bind_material(
            cube.GetPrim(),
            red,
        )

        # --------------------------------------------------------------
        # Blue tray.
        # --------------------------------------------------------------

        tray_center = (
            Gf.Vec3d(
                table_x,
                table_y,
                table_top_z + 0.0125,
            )
            + forward * 0.08
            + lateral * 0.25
        )

        tray_center_x = float(tray_center[0])
        tray_center_y = float(tray_center[1])
        tray_z = table_top_z + 0.0125

        tray_root = root_path + "/BlueTray"

        stage.DefinePrim(
            tray_root,
            "Xform",
        )

        floor = UsdGeom.Cube.Define(
            stage,
            tray_root + "/Floor",
        )

        floor.CreateSizeAttr(1.0)

        floor_xform = UsdGeom.Xformable(
            floor.GetPrim()
        )

        floor_xform.AddTranslateOp().Set(
            Gf.Vec3d(
                tray_center_x,
                tray_center_y,
                tray_z,
            )
        )

        floor_xform.AddScaleOp().Set(
            Gf.Vec3f(
                0.30,
                0.40,
                0.025,
            )
        )

        bind_material(
            floor.GetPrim(),
            blue,
        )

        wall_height = 0.07
        wall_thickness = 0.025

        # The robot is currently only translated by commander.py, so the
        # tray remains world-axis aligned. Its centre still follows the
        # current robot pose.
        walls = [
            (
                "LeftWall",
                (
                    tray_center_x,
                    tray_center_y - 0.20,
                    tray_z + wall_height / 2.0,
                ),
                (
                    0.30,
                    wall_thickness,
                    wall_height,
                ),
            ),
            (
                "RightWall",
                (
                    tray_center_x,
                    tray_center_y + 0.20,
                    tray_z + wall_height / 2.0,
                ),
                (
                    0.30,
                    wall_thickness,
                    wall_height,
                ),
            ),
            (
                "FrontWall",
                (
                    tray_center_x - 0.15,
                    tray_center_y,
                    tray_z + wall_height / 2.0,
                ),
                (
                    wall_thickness,
                    0.40,
                    wall_height,
                ),
            ),
            (
                "BackWall",
                (
                    tray_center_x + 0.15,
                    tray_center_y,
                    tray_z + wall_height / 2.0,
                ),
                (
                    wall_thickness,
                    0.40,
                    wall_height,
                ),
            ),
        ]

        for wall_name, wall_position, wall_scale in walls:
            wall = UsdGeom.Cube.Define(
                stage,
                f"{tray_root}/{wall_name}",
            )

            wall.CreateSizeAttr(1.0)

            wall_xform = UsdGeom.Xformable(
                wall.GetPrim()
            )

            wall_xform.AddTranslateOp().Set(
                Gf.Vec3d(
                    float(wall_position[0]),
                    float(wall_position[1]),
                    float(wall_position[2]),
                )
            )

            wall_xform.AddScaleOp().Set(
                Gf.Vec3f(
                    float(wall_scale[0]),
                    float(wall_scale[1]),
                    float(wall_scale[2]),
                )
            )

            bind_material(
                wall.GetPrim(),
                blue,
            )

        logger.info(
            "Runtime tabletop created in front of OpenArm: "
            "robot=(%.3f, %.3f, %.3f), "
            "table=(%.3f, %.3f, %.3f), "
            "forward_offset=%.2f m",
            robot_x,
            robot_y,
            robot_z,
            table_x,
            table_y,
            table_center_z,
            forward_offset,
        )

    def _runtime_load_shelf_reach_scene(self) -> None:
        """Create a reachability shelf with coloured cubes in front of OpenArm."""

        import omni.usd

        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()

        robot_prim = stage.GetPrimAtPath("/openarm")

        if not robot_prim.IsValid():
            raise RuntimeError(
                "OpenArm root prim /openarm does not exist"
            )

        robot_xform = UsdGeom.Xformable(robot_prim)

        world_transform = robot_xform.ComputeLocalToWorldTransform(
            0.0
        )

        robot_position = world_transform.ExtractTranslation()

        robot_x = float(robot_position[0])
        robot_y = float(robot_position[1])
        robot_z = float(robot_position[2])

        # OpenArm local +X is treated as forward and +Y as lateral.
        forward = world_transform.TransformDir(
            Gf.Vec3d(1.0, 0.0, 0.0)
        )

        lateral = world_transform.TransformDir(
            Gf.Vec3d(0.0, 1.0, 0.0)
        )

        if forward.GetLength() < 1e-6:
            forward = Gf.Vec3d(1.0, 0.0, 0.0)
        else:
            forward.Normalize()

        if lateral.GetLength() < 1e-6:
            lateral = Gf.Vec3d(0.0, 1.0, 0.0)
        else:
            lateral.Normalize()

        root_path = "/World/RuntimeObjects/ShelfReach"

        existing = stage.GetPrimAtPath(root_path)

        if existing.IsValid():
            stage.RemovePrim(root_path)

        stage.DefinePrim(
            Sdf.Path(root_path),
            "Xform",
        )

        # --------------------------------------------------------------
        # Helpers
        # --------------------------------------------------------------

        def create_material(
            path,
            color,
            roughness=0.4,
        ):
            material = UsdShade.Material.Define(
                stage,
                path,
            )

            shader = UsdShade.Shader.Define(
                stage,
                path + "/Shader",
            )

            shader.CreateIdAttr(
                "UsdPreviewSurface"
            )

            shader.CreateInput(
                "diffuseColor",
                Sdf.ValueTypeNames.Color3f,
            ).Set(
                Gf.Vec3f(
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                )
            )

            shader.CreateInput(
                "roughness",
                Sdf.ValueTypeNames.Float,
            ).Set(
                float(roughness)
            )

            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(),
                "surface",
            )

            return material

        def bind_material(
            prim,
            material,
        ):
            UsdShade.MaterialBindingAPI(
                prim
            ).Bind(
                material
            )

        def add_cube(
            path,
            center,
            size,
            material,
        ):
            cube = UsdGeom.Cube.Define(
                stage,
                path,
            )

            cube.CreateSizeAttr(1.0)

            xform = UsdGeom.Xformable(
                cube.GetPrim()
            )

            xform.AddTranslateOp().Set(
                Gf.Vec3d(
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                )
            )

            xform.AddScaleOp().Set(
                Gf.Vec3f(
                    float(size[0]),
                    float(size[1]),
                    float(size[2]),
                )
            )

            bind_material(
                cube.GetPrim(),
                material,
            )

            return cube

        # --------------------------------------------------------------
        # Materials
        # --------------------------------------------------------------

        shelf_material = create_material(
            root_path + "/Materials/Shelf",
            (0.18, 0.20, 0.24),
            roughness=0.45,
        )

        red = create_material(
            root_path + "/Materials/Red",
            (0.90, 0.03, 0.03),
            roughness=0.30,
        )

        blue = create_material(
            root_path + "/Materials/Blue",
            (0.03, 0.15, 0.90),
            roughness=0.30,
        )

        green = create_material(
            root_path + "/Materials/Green",
            (0.04, 0.65, 0.10),
            roughness=0.30,
        )

        yellow = create_material(
            root_path + "/Materials/Yellow",
            (0.95, 0.75, 0.05),
            roughness=0.30,
        )

        # --------------------------------------------------------------
        # Shelf placement relative to robot.
        # --------------------------------------------------------------

        shelf_forward_offset = 0.95

        shelf_center = (
            Gf.Vec3d(
                robot_x,
                robot_y,
                robot_z,
            )
            + forward * shelf_forward_offset
        )

        shelf_x = float(shelf_center[0])
        shelf_y = float(shelf_center[1])

        # Overall shelf dimensions.
        shelf_width = 1.00
        shelf_depth = 0.30
        shelf_height = 1.35
        board_thickness = 0.04

        # The shelf is centred around the robot's torso workspace.
        shelf_base_z = robot_z + 0.20
        shelf_top_z = shelf_base_z + shelf_height

        # Side uprights.
        left_center = (
            Gf.Vec3d(
                shelf_x,
                shelf_y,
                shelf_base_z + shelf_height / 2.0,
            )
            - lateral * (shelf_width / 2.0)
        )

        right_center = (
            Gf.Vec3d(
                shelf_x,
                shelf_y,
                shelf_base_z + shelf_height / 2.0,
            )
            + lateral * (shelf_width / 2.0)
        )

        add_cube(
            root_path + "/Shelf/LeftUpright",
            left_center,
            (
                shelf_depth,
                board_thickness,
                shelf_height,
            ),
            shelf_material,
        )

        add_cube(
            root_path + "/Shelf/RightUpright",
            right_center,
            (
                shelf_depth,
                board_thickness,
                shelf_height,
            ),
            shelf_material,
        )

        # Shelf levels.
        shelf_levels = [
            shelf_base_z + 0.25,
            shelf_base_z + 0.60,
            shelf_base_z + 0.95,
            shelf_base_z + 1.30,
        ]

        for index, level_z in enumerate(
            shelf_levels
        ):
            add_cube(
                root_path + f"/Shelf/Level{index}",
                (
                    shelf_x,
                    shelf_y,
                    level_z,
                ),
                (
                    shelf_depth,
                    shelf_width,
                    board_thickness,
                ),
                shelf_material,
            )

        # Back panel.
        back_center = (
            Gf.Vec3d(
                shelf_x,
                shelf_y,
                shelf_base_z + shelf_height / 2.0,
            )
            + forward * (shelf_depth / 2.0)
        )

        add_cube(
            root_path + "/Shelf/BackPanel",
            back_center,
            (
                board_thickness,
                shelf_width,
                shelf_height,
            ),
            shelf_material,
        )

        # --------------------------------------------------------------
        # Reach cubes.
        #
        # Cube centres are arranged at different heights and laterals
        # to create easy / medium / high reach targets.
        # --------------------------------------------------------------

        cube_size = 0.09

        cube_specs = [
            (
                "RedCube",
                red,
                shelf_levels[2] + board_thickness / 2.0 + cube_size / 2.0,
                -0.28,
                -0.04,
            ),
            (
                "BlueCube",
                blue,
                shelf_levels[2] + board_thickness / 2.0 + cube_size / 2.0,
                0.28,
                0.02,
            ),
            (
                "GreenCube",
                green,
                shelf_levels[1] + board_thickness / 2.0 + cube_size / 2.0,
                0.05,
                -0.03,
            ),
            (
                "YellowCube",
                yellow,
                shelf_levels[0] + board_thickness / 2.0 + cube_size / 2.0,
                -0.25,
                0.03,
            ),
        ]

        for (
            cube_name,
            cube_material,
            cube_z,
            lateral_offset,
            depth_offset,
        ) in cube_specs:

            cube_center = (
                Gf.Vec3d(
                    shelf_x,
                    shelf_y,
                    cube_z,
                )
                + lateral * lateral_offset
                - forward * depth_offset
            )

            add_cube(
                root_path + f"/Objects/{cube_name}",
                cube_center,
                (
                    cube_size,
                    cube_size,
                    cube_size,
                ),
                cube_material,
            )

        logger.info(
            "Shelf-reach scene created in front of OpenArm: "
            "robot=(%.3f, %.3f, %.3f), "
            "shelf=(%.3f, %.3f), "
            "forward_offset=%.2f m",
            robot_x,
            robot_y,
            robot_z,
            shelf_x,
            shelf_y,
            shelf_forward_offset,
        )

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

    def _runtime_clear_scene(self) -> None:
        """Remove only the currently loaded runtime USD scene."""

        import omni.usd

        stage = omni.usd.get_context().get_stage()

        scene_path = "/World/RuntimeScene"

        if stage.GetPrimAtPath(scene_path).IsValid():
            stage.RemovePrim(scene_path)

            logger.info(
                "Removed runtime scene %s",
                scene_path,
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

        from pxr import Gf, Sdf, UsdGeom

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

        if len(scale) != 3:
            raise ValueError(
                "Runtime scene scale requires exactly 3 values"
            )

        stage = omni.usd.get_context().get_stage()

        scene_path = "/World/RuntimeScene"

        if stage.GetPrimAtPath(scene_path).IsValid():
            stage.RemovePrim(scene_path)

        prim = stage.DefinePrim(
            Sdf.Path(scene_path),
            "Xform",
        )

        prim.GetReferences().AddReference(
            usd_path
        )

        xformable = UsdGeom.Xformable(
            prim
        )

        scale_op = None

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break

        if scale_op is None:
            scale_op = xformable.AddScaleOp()

        scale_op.Set(
            Gf.Vec3f(
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
            )
        )

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
        """Load a scene from NVIDIA's configured Isaac asset root."""

        import omni.usd

        from pxr import Gf, Sdf, UsdGeom

        usd_path = self._resolve_isaac_asset_path(
            command["path"]
        )

        scale = command.get(
            "scale",
            [1.0, 1.0, 1.0],
        )

        stage = omni.usd.get_context().get_stage()

        scene_path = "/World/RuntimeScene"

        existing_prim = stage.GetPrimAtPath(scene_path)

        # Do not destroy/recreate RuntimeScene if it already exists.
        # Replacing it while PhysX tensor views are active can invalidate
        # the OpenArm articulation physics view.
        if existing_prim.IsValid():
            logger.info(
                "Runtime scene already exists at %s; "
                "skipping duplicate scene load request",
                scene_path,
            )
            return

        prim = stage.DefinePrim(
            Sdf.Path(scene_path),
            "Xform",
        )

        prim.GetReferences().AddReference(
            usd_path
        )

        xformable = UsdGeom.Xformable(
            prim
        )

        scale_op = None

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break

        if scale_op is None:
            scale_op = xformable.AddScaleOp()

        scale_op.Set(
            Gf.Vec3f(
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
            )
        )

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

        runtime_root = (
            "/World/RuntimeObjects"
        )

        if not stage.GetPrimAtPath(
            runtime_root
        ).IsValid():
            stage.DefinePrim(
                Sdf.Path(runtime_root),
                "Xform",
            )

        prim_path = (
            f"{runtime_root}/{name}"
        )

        existing = stage.GetPrimAtPath(
            prim_path
        )

        if existing.IsValid():
            stage.RemovePrim(
                prim_path
            )

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
        # translate/orient/scale xform ops. Reuse existing ops
        # instead of trying to create duplicates.
        translate_op = None
        scale_op = None

        for op in xformable.GetOrderedXformOps():
            if (
                op.GetOpType()
                == UsdGeom.XformOp.TypeTranslate
            ):
                translate_op = op

            elif (
                op.GetOpType()
                == UsdGeom.XformOp.TypeScale
            ):
                scale_op = op

        if translate_op is None:
            translate_op = (
                xformable.AddTranslateOp()
            )

        if scale_op is None:
            scale_op = (
                xformable.AddScaleOp()
            )

        translate_op.Set(
            Gf.Vec3d(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        )

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

        self._runtime_apply_physics(
            prim,
            physics_mode,
            mass,
        )

        logger.info(
            "Spawned runtime object '%s' from %s at %s",
            name,
            usd_path,
            position,
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

        prim_path = (
            f"/World/RuntimeObjects/{name}"
        )

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

        prim_path = (
            f"/World/RuntimeObjects/{name}"
        )

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

        stage = (
            omni.usd
            .get_context()
            .get_stage()
        )

        prim_path = (
            f"/World/RuntimeObjects/{name}"
        )

        prim = stage.GetPrimAtPath(
            prim_path
        )

        if not prim.IsValid():
            logger.warning(
                "Runtime object '%s' does not exist",
                name,
            )
            return

        stage.RemovePrim(
            prim_path
        )

        logger.info(
            "Removed runtime object '%s'",
            name,
        )

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._sim_app.is_running() and not self._stop.is_set():
                # Isaac advances physics inside update(); we then drive the
                # bridge step on the same thread (Articulation reads require
                # Isaac's main thread). The extension defers its own setup until
                # the stage is live, so early steps are cheap no-ops.
                self._sim_app.update()

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

                self._update_runtime_forces()

                self._apply_runtime_arm_targets()

    def _shutdown(self) -> None:
        self._ready.clear()

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
