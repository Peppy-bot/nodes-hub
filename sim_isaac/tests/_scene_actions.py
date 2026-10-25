"""scene_actions.py loaded for tests, over fakes of the generated modules it
imports.

peppygen exists only inside the node's image. The fakes are the action
modules of the scene slot and the service modules of the scene and objects
slots. Each generated type of a service answer builds a namespace of the
fields it gets, so a test reads an answer with vars(). Each module has the
Failure of its member, and each action module the GoalDecision a goal is
admitted or refused with.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"

# The generated modules scene_actions.py imports: each action of the scene
# slot, and each service of the scene and objects slots.
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
_SERVICES = {
    "scene": (
        "get_assets_list", "get_asset_categories", "get_objects_list", "get_object", "get_robots_list", "get_scene",
    ),
    "objects": ("get_object_states",),
}

# The generated types that scene_actions.py builds a service answer from.
_ANSWER_TYPES = ("Response", "ResponseObjectsItem", "ResponseRobotsItem", "ResponseCategoriesItem")


def _answer(**fields):
    return SimpleNamespace(**fields)


class Failure(Exception):
    """Why a member refused or failed a goal, or gave an error reply, as the
    generated Failure of a member holds it: a kind and a message."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


class GoalDecision:
    """What admitting a goal answers, as the generated GoalDecision of an
    action with no goal response holds it: the failure of a refused goal, or
    None for an accepted one."""

    def __init__(self, refusal: Failure | None) -> None:
        self.refusal = refusal

    @staticmethod
    def accept() -> "GoalDecision":
        return GoalDecision(None)

    @staticmethod
    def refuse(kind: str, message: str) -> "GoalDecision":
        return GoalDecision(Failure(kind, message))


def _member_module(name: str) -> ModuleType:
    """A generated module of one member, with a Failure of its own."""
    module = ModuleType(name)
    module.Failure = type("Failure", (Failure,), {"__module__": name})
    return module


def _service_module(name: str) -> ModuleType:
    module = _member_module(name)
    for answer_type in _ANSWER_TYPES:
        setattr(module, answer_type, _answer)
    return module


def _action_module(name: str) -> ModuleType:
    module = _member_module(name)
    module.GoalDecision = GoalDecision
    return module


def scene_actions_module(monkeypatch, name: str) -> ModuleType:
    """A fresh scene_actions.py under the module name `name`, for the life of
    one test: its generated imports faked and its siblings importable, as the
    engine's launch puts them."""
    actions = ModuleType("peppygen.exposed_actions.scene")
    for action in _ACTIONS:
        setattr(actions, action, _action_module(f"{actions.__name__}.{action}"))
    modules = {
        "peppygen": ModuleType("peppygen"),
        "peppygen.exposed_actions": ModuleType("peppygen.exposed_actions"),
        actions.__name__: actions,
        "peppygen.exposed_services": ModuleType("peppygen.exposed_services"),
    }
    for link, names in _SERVICES.items():
        services = ModuleType(f"peppygen.exposed_services.{link}")
        for service in names:
            setattr(services, service, _service_module(f"{services.__name__}.{service}"))
        modules[services.__name__] = services
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.syspath_prepend(str(_ENGINE_DIR))
    spec = importlib.util.spec_from_file_location(name, _ENGINE_DIR / "scene_actions.py")
    module = importlib.util.module_from_spec(spec)
    # Its dataclass resolves postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module
