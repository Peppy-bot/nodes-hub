"""The detector that is not there: `perception_backend: "none"`.

It loads nothing and is never available, so both searches are refused
with "no perception source", exactly what the contract asks of a brain
that cannot see. Every other member keeps working.
"""

from __future__ import annotations

from typing import Sequence

from ..ports import Box


class NoneDetector:
    name = "none"

    @property
    def available(self) -> bool:
        return False

    def load(self, model: str, gallery: str = "") -> None:
        return None

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        return None

    def detect(self, image) -> list[Box]:
        return []
