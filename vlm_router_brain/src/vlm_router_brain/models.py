"""The models this brain can run, for brain_core: a `ModelSource` over the
models built into the node and the catalog.

Each role accepts the models built into the node (`BUILT_IN`): "none",
the role's missing model, for both, and "scripted", the scripted
manipulator with no model call, for manipulation. Each role also accepts
every catalog model that serves it: for perception the model is the
detector; for manipulation it is the model that aligns the gripper above
the item in the scripted grab and above the spot in the scripted place
(`manipulators.vlm`). A name a role does not accept is refused with
`rejected`,
pointing to list_models. `offered` gives the entry of a name for a role, a
`BuiltIn` or a `RouterEntry` for a catalog model, so the entry carries the
role it was offered for. A catalog model runs on one route, the route the
log names and no result does, and becomes the backend `BUILDERS` makes for
its role. With an OpenAI-compatible API (openai_compatible.py) that route
is the one through the API, under the model's catalog name, and a model
always has it. Without one it is the first of the model's routes whose key
is set in the node's environment, and a model may have none.

A built-in model makes no call, so a start and a switch build it alike.
At the node's start (`start`) a catalog model makes no call either: the
first search or the first grab is its first call, and a provider failure
fails that goal alone. A switch to a catalog model (`prepare`) probes the
route once, with the probe of the role (`PROBES`), before brain_core hands
the role its new model, so a model that cannot be reached, or that does
not answer the role's question, fails the switch and the role keeps its
model: with `resource_unavailable` when the model has no route, with
`attempt_failed` when the probe fails, an API that does not serve the
model among them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Optional, Sequence, Union

from brain_core.manipulators.none import NoneManipulator
from brain_core.manipulators.scripted import CentreProposer, ScriptedManipulator
from brain_core.perception.none import NoneDetector
from brain_core.ports import FailureKind, Refusal, Role

from .catalog import NONE, SCRIPTED, Catalog, CatalogEntry, Route
from .keys import first_keyed, no_key_message
from .manipulators.vlm import probe_alignment, vlm_manipulator
from .openai_compatible import CompatibleApi, route_of
from .provider import Provider
from .vlm_detector import VlmDetector, probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltIn:
    """The entry of a model built into the node, for a role."""

    role: Role
    name: str


@dataclass(frozen=True)
class RouterEntry:
    """A catalog model offered for a role."""

    role: Role
    catalog_entry: CatalogEntry


Entry = Union[BuiltIn, RouterEntry]

# The models built into the node, by role, each with the backend it runs.
BUILT_IN: Mapping[Role, Mapping[str, Callable[[], object]]] = {
    Role.PERCEPTION: {NONE: NoneDetector},
    Role.MANIPULATION: {NONE: NoneManipulator, SCRIPTED: lambda: ScriptedManipulator(CentreProposer())},
}


# What a catalog model runs as in each role a catalog model serves
# (catalog.CATALOG_ROLES), built from the model, its route and the provider.
BUILDERS: Mapping[Role, Callable[[CatalogEntry, Route, Provider], object]] = {
    Role.PERCEPTION: VlmDetector,
    Role.MANIPULATION: vlm_manipulator,
}
# The probe a switch to a catalog model runs, by role: each asks the
# role's own question.
PROBES: Mapping[Role, Callable[[Provider, Route], Awaitable[None]]] = {
    Role.PERCEPTION: probe,
    Role.MANIPULATION: probe_alignment,
}


class RouterModels:
    def __init__(
        self, catalog: Catalog, provider: Provider, environ: Mapping[str, str] = os.environ, *, api: Optional[CompatibleApi] = None
    ) -> None:
        self._catalog = catalog
        self._provider = provider
        self._environ = environ
        self._api = api

    def roles(self) -> Sequence[Role]:
        return (Role.PERCEPTION, Role.MANIPULATION)

    def names(self, role: Role) -> Sequence[str]:
        return [*BUILT_IN[role], *self._catalog.names_for(role)]

    def offered(self, role: Role, name: str) -> Entry:
        if name in BUILT_IN[role]:
            return BuiltIn(role, name)
        catalog_entry = self._catalog.get(name)
        if catalog_entry is None or role not in catalog_entry.roles:
            raise Refusal(FailureKind.REJECTED, f"'{name}' is not a {role} model here; see list_models")
        return RouterEntry(role, catalog_entry)

    def start(self, entry: Entry):
        if isinstance(entry, BuiltIn):
            return _built_in(entry)
        model = entry.catalog_entry
        route = self._route(model)
        if route is None:
            raise ValueError(no_key_message(model.name, model.routes))
        return self._build(entry, route)

    async def prepare(self, entry: Entry):
        if isinstance(entry, BuiltIn):
            return _built_in(entry)
        model = entry.catalog_entry
        route = self._route(model)
        if route is None:
            raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, no_key_message(model.name, model.routes))
        await PROBES[entry.role](self._provider, route)
        return self._build(entry, route)

    def _route(self, model: CatalogEntry) -> Optional[Route]:
        """The route `model` runs on: the one through the API when the node
        has one, else the first of its routes whose key is set, or none."""
        if self._api is not None:
            return route_of(model)
        return first_keyed(model.routes, self._environ)

    def _build(self, entry: RouterEntry, route: Route):
        logger.info("%s runs through %s", entry.catalog_entry.name, route.spelled)
        return BUILDERS[entry.role](entry.catalog_entry, route, self._provider)


def _built_in(entry: BuiltIn):
    return BUILT_IN[entry.role][entry.name]()
