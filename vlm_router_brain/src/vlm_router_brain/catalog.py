"""The models this brain can run, read once at the node's start from
`catalog.json`, the file that ships inside the node.

Each entry is a model under a name of the node's own, the one get_model,
list_models and switch_model use: the roles it serves, the coordinates its
boxes come in, and its routes, in order. A route is a provider and that
provider's model id, written "provider:model" and split on the first ":"
alone, since an OpenRouter id may hold one ("openrouter:org/model:free"),
with the settings that model was tested with: the least deadline a call
gets, and the reasoning effort any-llm asks for. A model is in the catalog
once it returned correct boxes for a Waldo frame through the node's own
prompt and schema.

A catalog the node cannot read stops the node at its start, naming the
entry and the field: a model the brain could not reach is better refused
before the first goal than at it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Optional, get_args

from any_llm.types.completion import ReasoningEffort

from brain_core.ports import Role

from .keys import KEY_VARIABLES

CATALOG_FILE = "catalog.json"
# The roles a catalog model serves in this node: perception, through the
# node's detector. Manipulation runs the models built into the node alone.
CATALOG_ROLES = frozenset({Role.PERCEPTION})
# The coordinate formats the node reads, each with its prompt wording
# (prompts.py) and its converter (parsing.py): yx_0_1000 is a box as
# [y_min, x_min, y_max, x_max] normalised to 0-1000.
COORDINATES = ("yx_0_1000",)
REASONING_EFFORTS = get_args(ReasoningEffort)
# The names of the models built into the node, never a catalog model's:
# "none", a role's missing model, and "scripted", the scripted manipulator.
NONE = "none"
SCRIPTED = "scripted"
BUILT_IN_NAMES = (NONE, SCRIPTED)

_ENTRY_FIELDS = frozenset({"name", "roles", "coordinates", "routes"})
_ROUTE_FIELDS = frozenset({"route", "min_call_timeout_s", "reasoning_effort"})


@dataclass(frozen=True)
class Route:
    """One way to reach a catalog model: the provider, its model id, and
    the settings it was tested with. `model_name` is the catalog name, the
    one a failure names."""

    model_name: str
    provider: str
    model: str
    min_call_timeout_s: float
    reasoning_effort: str

    @property
    def spelled(self) -> str:
        """The route as the catalog writes it, for the log."""
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    roles: frozenset[Role]
    coordinates: str
    routes: tuple[Route, ...]


@dataclass(frozen=True)
class Catalog:
    entries: tuple[CatalogEntry, ...]

    def get(self, name: str) -> Optional[CatalogEntry]:
        return next((entry for entry in self.entries if entry.name == name), None)

    def names_for(self, role: Role) -> list[str]:
        """The catalog models that serve `role`, in the catalog's order."""
        return [entry.name for entry in self.entries if role in entry.roles]


class CatalogError(ValueError):
    """A catalog the node cannot read: the message names where and why."""


def load_catalog(text: Optional[str] = None) -> Catalog:
    """The shipped catalog, or the catalog `text` holds."""
    if text is None:
        text = resources.files(__package__).joinpath(CATALOG_FILE).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise CatalogError(f"{CATALOG_FILE} is not JSON: {error}") from None
    return parse_catalog(data)


def parse_catalog(data: Any) -> Catalog:
    if not isinstance(data, dict) or set(data) != {"models"} or not isinstance(data["models"], list):
        raise CatalogError(f'{CATALOG_FILE} must be an object with one field, "models", a list')
    entries: list[CatalogEntry] = []
    for index, raw in enumerate(data["models"]):
        entry = _parse_entry(raw, f"{CATALOG_FILE} models[{index}]")
        if any(known.name == entry.name for known in entries):
            raise CatalogError(f"{CATALOG_FILE}: the model '{entry.name}' is named twice")
        entries.append(entry)
    return Catalog(tuple(entries))


def _parse_entry(raw: Any, where: str) -> CatalogEntry:
    if not isinstance(raw, dict):
        raise CatalogError(f"{where} must be an object")
    _refuse_fields(raw, _ENTRY_FIELDS, where)
    name = _text(raw, "name", where)
    where = f"{CATALOG_FILE} model '{name}'"
    if name in BUILT_IN_NAMES:
        raise CatalogError(f"{where}: '{name}' names a model built into the node, never a catalog model")
    roles = raw["roles"]
    if not isinstance(roles, list) or not roles:
        raise CatalogError(f"{where}: roles must be a list of at least one role")
    parsed_roles: set[Role] = set()
    for role in roles:
        if role not in {r.value for r in CATALOG_ROLES}:
            served = ", ".join(sorted(r.value for r in CATALOG_ROLES))
            raise CatalogError(f"{where}: the role {role!r} is not one a catalog model serves here ({served})")
        parsed_roles.add(Role(role))
    coordinates = _text(raw, "coordinates", where)
    if coordinates not in COORDINATES:
        raise CatalogError(f"{where}: unknown coordinates '{coordinates}'; known: {', '.join(COORDINATES)}")
    routes = raw["routes"]
    if not isinstance(routes, list) or not routes:
        raise CatalogError(f"{where}: routes must be a list of at least one route")
    return CatalogEntry(
        name=name,
        roles=frozenset(parsed_roles),
        coordinates=coordinates,
        routes=tuple(_parse_route(route, name, f"{where} routes[{index}]") for index, route in enumerate(routes)),
    )


def _parse_route(raw: Any, model_name: str, where: str) -> Route:
    if not isinstance(raw, dict):
        raise CatalogError(f"{where} must be an object")
    _refuse_fields(raw, _ROUTE_FIELDS, where)
    spelled = _text(raw, "route", where)
    provider, separator, model = spelled.partition(":")
    if not separator or not provider or not model:
        raise CatalogError(f"{where}: route '{spelled}' must be written provider:model")
    if provider not in KEY_VARIABLES:
        raise CatalogError(f"{where}: unknown provider '{provider}'; known: {', '.join(sorted(KEY_VARIABLES))}")
    timeout = raw["min_call_timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        raise CatalogError(f"{where}: min_call_timeout_s must be a number of seconds above 0")
    effort = _text(raw, "reasoning_effort", where)
    if effort not in REASONING_EFFORTS:
        raise CatalogError(f"{where}: unknown reasoning_effort '{effort}'; known: {', '.join(REASONING_EFFORTS)}")
    return Route(model_name, provider, model, float(timeout), effort)


def _refuse_fields(raw: dict, fields: frozenset[str], where: str) -> None:
    missing = sorted(fields - set(raw))
    if missing:
        raise CatalogError(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise CatalogError(f"{where}: unknown field {', '.join(unknown)}")


def _text(raw: dict, field: str, where: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CatalogError(f"{where}: {field} must be a non-empty string with no spaces around it")
    return value
