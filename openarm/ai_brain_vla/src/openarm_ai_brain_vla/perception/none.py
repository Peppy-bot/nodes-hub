"""The detector that is not there: `perception_backend: "none"`.

It loads nothing and is never available, so both searches are refused
with "no perception source", exactly what the contract asks of a brain
that cannot see. Every other member keeps working. A perception_model or
perception_gallery set beside it names something it cannot read, so the
load fails saying so rather than the setting going unread.
"""

from __future__ import annotations

from typing import Sequence

from ..ports import Box, Coverage, Deadline


class NoneDetector:
    name = "none"
    min_confidence = 0.0

    @property
    def available(self) -> bool:
        return False

    def load(self, model: str, gallery: str) -> None:
        if model.strip() or gallery.strip():
            raise ValueError("perception_backend 'none' loads no model: perception_model and perception_gallery must be empty")

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        return None

    def detect(self, image, deadline: Deadline) -> list[Box]:
        return []

    def scan_coverage(self) -> Coverage:
        return Coverage()
