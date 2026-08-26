#!/usr/bin/env python3

import asyncio
from pathlib import Path

from isaacsim import SimulationApp


simulation_app = SimulationApp({
    "headless": True,
})


async def convert_one(input_path: Path, output_path: Path):
    import carb
    import omni.kit.asset_converter

    print("Converting:")
    print(f"  {input_path}")
    print(f"-> {output_path}")

    context = omni.kit.asset_converter.AssetConverterContext()

    # OBJ dimensions are often exported in mm.
    # Keep original scale for now; we can scale when spawning.
    context.ignore_material = False
    context.ignore_animation = True
    context.ignore_cameras = True
    context.single_mesh = False
    context.smooth_normals = True

    converter = omni.kit.asset_converter.get_instance()

    task = converter.create_converter_task(
        str(input_path),
        str(output_path),
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
        raise RuntimeError(
            f"Could not convert {input_path}"
        )

    print(f"SUCCESS: {output_path}")


async def main():
    asset_dir = (
        Path.home()
        / "Downloads"
        / "Simulation-scenes"
        / "simulation-scenes"
        / "scenes"
        / "gear_assembly"
        / "assets"
    )

    await convert_one(
        asset_dir / "large_gear.obj",
        asset_dir / "large_gear.usd",
    )

    await convert_one(
        asset_dir / "small_gear.obj",
        asset_dir / "small_gear.usd",
    )


asyncio.get_event_loop().run_until_complete(
    main()
)

simulation_app.close()
