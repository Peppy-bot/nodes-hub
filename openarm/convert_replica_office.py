#!/usr/bin/env python3

import asyncio
from pathlib import Path

from isaacsim import SimulationApp


simulation_app = SimulationApp({
    "headless": True,
})


async def convert():
    import carb
    import omni.kit.asset_converter

    base_dir = (
        Path.home()
        / "Downloads"
        / "active-gs"
        / "data"
        / "replica_v1"
        / "office_0"
    )

    src = base_dir / "mesh.ply"
    dst = base_dir / "office0.usd"

    print(f"Input : {src}")
    print(f"Output: {dst}")

    if not src.exists():
        raise FileNotFoundError(src)

    converter = omni.kit.asset_converter.get_instance()

    context = omni.kit.asset_converter.AssetConverterContext()

    context.ignore_material = False
    context.ignore_animation = True
    context.ignore_cameras = True
    context.single_mesh = False
    context.smooth_normals = True

    task = converter.create_converter_task(
        str(src),
        str(dst),
        None,
        context,
    )

    success = await task.wait_until_finished()

    if not success:
        carb.log_error(
            f"Conversion failed: "
            f"{task.get_status()} "
            f"{task.get_error_message()}"
        )
        raise RuntimeError("Replica conversion failed")

    print(f"SUCCESS: {dst}")


asyncio.get_event_loop().run_until_complete(
    convert()
)

simulation_app.close()
