"""The scenes this engine can stand, one per model a robot may attach as."""

from __future__ import annotations

import os
from pathlib import Path

from head_camera import Pack

# The scene of each model, baked into the base image under
# PEPPY_ROBOT_ASSETS_DIR.
BAKED_SCENES = {
    "openarm_v1": "openarm_bimanual_v1.xml",
    "openarm_v2": "openarm_bimanual_v2.xml",
}

# The head camera assembly seats on the v2 pedestal, so a robot standing as
# that model draws it and a v1 has none (see head_camera.py).
HEAD_CAMERA_MODELS = frozenset({"openarm_v2"})

ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "assets"))
)


class Catalogue:
    """The models this engine can stand, by the id a robot attaches with,
    and the head camera pack the models carrying one draw."""

    def __init__(self, scenes: dict[str, Path], *, head_camera_pack: Pack) -> None:
        self._scenes = dict(scenes)
        self._head_camera_pack = head_camera_pack

    @staticmethod
    def baked(*, head_camera_pack: Pack) -> "Catalogue":
        """The scenes of the container image, under PEPPY_ROBOT_ASSETS_DIR."""
        return Catalogue(
            {model: ASSETS_DIR / file for model, file in BAKED_SCENES.items()},
            head_camera_pack=head_camera_pack,
        )

    def models(self) -> list[str]:
        return sorted(self._scenes)

    def head_camera_pack(self, model: str) -> Pack | None:
        """The head camera pack a robot standing as `model` draws, or None
        for a model whose robot has no head camera."""
        self._known(model)
        return self._head_camera_pack if model in HEAD_CAMERA_MODELS else None

    def scene(self, model: str) -> Path:
        """The MJCF of a model."""
        path = self._known(model)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the scenes are baked into the container image"
            )
        return path

    def _known(self, model: str) -> Path:
        """The scene file of a model this engine carries. An id it does not
        carry names what it does, so a launcher's typo says what to write
        instead."""
        if model not in self._scenes:
            raise ValueError(
                f"unknown model {model!r}: this engine stands {', '.join(self.models())}"
            )
        return self._scenes[model]
