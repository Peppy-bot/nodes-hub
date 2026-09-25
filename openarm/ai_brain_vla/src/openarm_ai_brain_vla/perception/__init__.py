"""Perception: one core `Perceiver` that turns camera frames into world
detections, and a registry of swappable `Detector` backends behind it.

The registry maps a backend name to "module:Class" and imports only the
one the launcher selected, so choosing "none" never imports a model
library, and a heavy backend costs nothing to the others.
"""

from __future__ import annotations

import importlib

from ..ports import Detector

REGISTRY: dict[str, str] = {
    "none": "openarm_ai_brain_vla.perception.none:NoneDetector",
    # SAM 3 to find and SigLIP against a gallery to name: the most accurate
    # pipeline of the perception study. Needs the sam3-siglip extra and a GPU.
    "sam3_siglip": "openarm_ai_brain_vla.perception.sam3_siglip:Sam3SiglipDetector",
    # YOLOE-11M with the gallery's crops as visual prompts: the study's
    # real-time pipeline, 21 ms a frame. Needs the yoloe extra; a GPU helps.
    "yoloe_vp": "openarm_ai_brain_vla.perception.yoloe_vp:YoloeVpDetector",
}


def make_detector(backend: str) -> Detector:
    """The detector class registered under `backend`, constructed."""
    try:
        target = REGISTRY[backend]
    except KeyError:
        names = ", ".join(sorted(REGISTRY))
        raise ValueError(f"unknown perception_backend '{backend}'; known: {names}") from None
    module_name, class_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()
