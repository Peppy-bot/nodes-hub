"""The models this brain runs, for brain_core: a `ModelSource` over the
models the parameters name.

Each role runs the model its parameter names for the node's whole life:
`perception_model` ("none" or "sam3_siglip") and `manipulation_model`
("none"). list_models names that model for each role, and switch_model
accepts it alone and refuses any other: "this brain runs only
'sam3_siglip' for perception". The parameters are read once, at the
node's start, and a name the brain does not run stops it there, as does a
`perception_gallery` beside a model that reads none.

The start (`start`) builds each model; sam3_siglip begins its load in the
background then, so the node is healthy at once. A switch to the running
model (`prepare`) hands back that same model, with no load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from brain_core.manipulation.none import NoneManipulator
from brain_core.ports import Detector, Manipulator, Refusal, Role

from .perception import DETECTORS, TAKES_A_GALLERY
from .perception.gallery import shipped_root

MANIPULATORS: dict[str, Callable[[], Manipulator]] = {
    "none": NoneManipulator,
}


@dataclass(frozen=True)
class VlaEntry:
    """A model this brain runs: its role and its name."""

    role: Role
    name: str


class VlaModels:
    def __init__(
        self,
        perception_model: str,
        manipulation_model: str,
        perception_gallery: str,
        *,
        detectors: Mapping[str, Callable[[str], Detector]] = DETECTORS,
        manipulators: Mapping[str, Callable[[], Manipulator]] = MANIPULATORS,
    ) -> None:
        self._detectors = detectors
        self._manipulators = manipulators
        self._names = {
            Role.PERCEPTION: _known("perception_model", perception_model, detectors),
            Role.MANIPULATION: _known("manipulation_model", manipulation_model, manipulators),
        }
        shipped_root(perception_gallery)
        if perception_gallery.strip() and self._names[Role.PERCEPTION] not in TAKES_A_GALLERY:
            raise ValueError(
                f"perception_gallery '{perception_gallery.strip()}' is for a model that reads one; "
                f"perception_model is '{self._names[Role.PERCEPTION]}'"
            )
        self._gallery = perception_gallery
        self._running: dict[Role, object] = {}

    def roles(self) -> Sequence[Role]:
        return (Role.PERCEPTION, Role.MANIPULATION)

    def names(self, role: Role) -> Sequence[str]:
        return [self._names[role]]

    def offered(self, role: Role, name: str) -> VlaEntry:
        running = self._names[role]
        if name != running:
            raise Refusal(f"this brain runs only '{running}' for {role}")
        return VlaEntry(role, name)

    def start(self, entry: VlaEntry):
        if entry.role == Role.PERCEPTION:
            backend = self._detectors[entry.name](self._gallery)
        else:
            backend = self._manipulators[entry.name]()
        self._running[entry.role] = backend
        return backend

    async def prepare(self, entry: VlaEntry):
        return self._running[entry.role]


def _known(parameter: str, name: str, registry: Mapping) -> str:
    if name not in registry:
        raise ValueError(f"{parameter} '{name}' is not a model this brain runs; it runs {', '.join(sorted(registry))}")
    return name
