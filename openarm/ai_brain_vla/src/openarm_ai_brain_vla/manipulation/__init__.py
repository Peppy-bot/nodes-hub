"""Manipulation backends: what runs grab_item, drop_item and place_item.

The registry maps a backend name to "module:Class" and imports only the
one the launcher selected. A scripted sequence over the backbone comes
first; a learned policy can register here later with no change to the
core.
"""

from __future__ import annotations

import importlib

from ..ports import Manipulator

REGISTRY: dict[str, str] = {
    "none": "openarm_ai_brain_vla.manipulation.none:NoneManipulator",
}


def make_manipulator(backend: str) -> Manipulator:
    """The manipulator class registered under `backend`, constructed."""
    try:
        target = REGISTRY[backend]
    except KeyError:
        names = ", ".join(sorted(REGISTRY))
        raise ValueError(f"unknown manipulation_backend '{backend}'; known: {names}") from None
    module_name, class_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()
