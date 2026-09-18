"""world.py loaded for tests, without the Isaac runtime its other paths reach
for."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"


def world_module():
    """A fresh world.py, its Isaac imports standing in as empty modules and
    its siblings importable, as the engine's launch puts them."""
    for name in ("omni", "omni.usd", "isaacsim", "isaacsim.core", "isaacsim.core.utils"):
        sys.modules.setdefault(name, ModuleType(name))
    if str(_ROBOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROBOT_DIR))
    spec = importlib.util.spec_from_file_location("_world_under_test", _ROBOT_DIR / "world.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stand_in_pack(head_camera, directory: Path):
    """A head camera pack for a stage that is stood in, where nothing reads
    its meshes."""
    return head_camera.Pack(
        directory=directory,
        body_position=(0.0315, 0.0, 0.743),
        visuals=(),
        collision=(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)),
    )
