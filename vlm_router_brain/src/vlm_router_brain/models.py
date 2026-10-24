"""The models this brain can run, for brain_core: a `ModelSource` over the
catalog.

Both roles accept "none", the role's missing model; perception also
accepts every catalog model that serves it. A name a role does not accept
is refused with `rejected`, pointing to list_models. `offered` gives the entry of a name for
a role, `NoModel` for "none" and a `RouterEntry` for a catalog model, so the
entry carries the role it was offered for. A catalog model runs on the
first of its routes whose key is set in the node's environment, the route
the log names and no result does, and becomes the backend `BUILDERS` makes
for its role.

At the node's start (`start`) the starting model makes no call: the first
search is its first call, and a provider failure fails that search alone.
A switch (`prepare`) probes the route once before brain_core hands the
role its new model, so a model that cannot be reached fails the switch and
the role keeps its model: with `resource_unavailable` when no route of the
model has its key, with `attempt_failed` when the probe fails.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Union

from brain_core.manipulation.none import NoneManipulator
from brain_core.perception.none import NoneDetector
from brain_core.ports import FailureKind, Refusal, Role

from .catalog import NONE, Catalog, CatalogEntry, Route
from .keys import first_keyed, no_key_message
from .provider import Provider
from .vlm_detector import VlmDetector, probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoModel:
    """The entry of "none" for a role."""

    role: Role


@dataclass(frozen=True)
class RouterEntry:
    """A catalog model offered for a role."""

    role: Role
    catalog_entry: CatalogEntry


Entry = Union[NoModel, RouterEntry]

# What a catalog model runs as in each role a catalog model serves
# (catalog.CATALOG_ROLES), built from the model, its route and the provider.
BUILDERS: Mapping[Role, Callable[[CatalogEntry, Route, Provider], object]] = {
    Role.PERCEPTION: VlmDetector,
}


class RouterModels:
    def __init__(self, catalog: Catalog, provider: Provider, environ: Mapping[str, str] = os.environ) -> None:
        self._catalog = catalog
        self._provider = provider
        self._environ = environ

    def roles(self) -> Sequence[Role]:
        return (Role.PERCEPTION, Role.MANIPULATION)

    def names(self, role: Role) -> Sequence[str]:
        return [NONE, *self._catalog.names_for(role)]

    def offered(self, role: Role, name: str) -> Entry:
        if name == NONE:
            return NoModel(role)
        catalog_entry = self._catalog.get(name)
        if catalog_entry is None or role not in catalog_entry.roles:
            raise Refusal(FailureKind.REJECTED, f"'{name}' is not a {role} model here; see list_models")
        return RouterEntry(role, catalog_entry)

    def start(self, entry: Entry):
        if isinstance(entry, NoModel):
            return _none_backend(entry.role)
        model = entry.catalog_entry
        route = first_keyed(model.routes, self._environ)
        if route is None:
            raise ValueError(no_key_message(model.name, model.routes))
        return self._build(entry, route)

    async def prepare(self, entry: Entry):
        if isinstance(entry, NoModel):
            return _none_backend(entry.role)
        model = entry.catalog_entry
        route = first_keyed(model.routes, self._environ)
        if route is None:
            raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, no_key_message(model.name, model.routes))
        await probe(self._provider, route)
        return self._build(entry, route)

    def _build(self, entry: RouterEntry, route: Route):
        logger.info("%s runs through %s", entry.catalog_entry.name, route.spelled)
        return BUILDERS[entry.role](entry.catalog_entry, route, self._provider)


def _none_backend(role: Role):
    return NoneDetector() if role == Role.PERCEPTION else NoneManipulator()
