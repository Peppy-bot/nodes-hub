"""The perception models this brain runs, by the name `perception_model`
gives them. Each loader takes the `perception_gallery` value and builds its
detector, importing the model's module only when it is called, so choosing
"none" never imports a model library.
"""

from __future__ import annotations

from typing import Callable

from brain_core.perception.none import NoneDetector
from brain_core.ports import Detector


def _sam3_siglip(gallery: str) -> Detector:
    from .sam3_siglip import Sam3SiglipDetector

    detector = Sam3SiglipDetector()
    detector.start_loading(gallery)
    return detector


DETECTORS: dict[str, Callable[[str], Detector]] = {
    "none": lambda gallery: NoneDetector(),
    # SAM 3 to find and SigLIP to name, by words from a general vocabulary.
    # Needs the sam3-siglip extra and a GPU, and downloads its weights
    # (weights.py) at the first load on a machine. The load runs in the
    # background from the node's start.
    "sam3_siglip": _sam3_siglip,
}
# The perception models that read an enrolment gallery.
TAKES_A_GALLERY = frozenset({"sam3_siglip"})
