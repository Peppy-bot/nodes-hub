"""The key of each provider, read from the node's own environment and from
nowhere else: no file, no parameter. A launch puts the key there (today the
launching shell exports it), and the daemon starts the node with that
environment alone.

A key value is never logged and never in a message: what the node says
about keys names the variables.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, TypeVar

R = TypeVar("R")

KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def key_for(provider: str, environ: Mapping[str, str] = os.environ) -> str:
    """The key of `provider` in `environ`, or empty when none is set."""
    return environ.get(KEY_VARIABLES[provider], "").strip()


def first_keyed(routes: Sequence[R], environ: Mapping[str, str] = os.environ) -> Optional[R]:
    """The first of `routes` whose provider's key is set, or none."""
    return next((route for route in routes if key_for(route.provider, environ)), None)


def no_key_message(model_name: str, routes: Sequence) -> str:
    """What a model with no keyed route is refused with: the variables that
    would reach it, in the order of its routes."""
    variables = list(dict.fromkeys(KEY_VARIABLES[route.provider] for route in routes))
    return f"no key reaches '{model_name}': set {' or '.join(variables)} in the node's environment"
