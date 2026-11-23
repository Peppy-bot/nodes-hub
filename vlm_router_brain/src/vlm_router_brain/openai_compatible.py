"""The OpenAI-compatible API the node's environment names, read from the
node's own environment and from nowhere else, as the keys are (keys.py):
OPENAI_COMPATIBLE_API_BASE, the base address of an API in the chat
completions form of OpenAI's, and OPENAI_COMPATIBLE_API_KEY, the key for
it.

Both set, `compatible_api` gives the API; neither set, it gives none. One
alone stops the node at its start, naming the one that is not set: an API
half set is refused before the first goal, never ignored. A value of
spaces counts as not set, as a key does, and each value is read without
the spaces around it. The base must be an http:// or https:// address with
a host.

Neither value is ever logged or in a message: the base can carry a user
and a password. What the node says about the API names the variables, and
neither value is in the API's repr.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional
from urllib.parse import urlsplit

API_BASE_VARIABLE = "OPENAI_COMPATIBLE_API_BASE"
API_KEY_VARIABLE = "OPENAI_COMPATIBLE_API_KEY"
# The name the node gives the API in any-llm, which any-llm's messages
# carry ("[openai_compatible] Error code: 404 - ...").
OPENAI_COMPATIBLE = "openai_compatible"
# The schemes a base address may have.
BASE_SCHEMES = ("http", "https")


@dataclass(frozen=True)
class CompatibleApi:
    """The API every model call goes to: its base address and its key."""

    base: str = field(repr=False)
    key: str = field(repr=False)


def compatible_api(environ: Mapping[str, str] = os.environ) -> Optional[CompatibleApi]:
    """The API `environ` names: both variables set, or none when neither
    is. A ValueError names what is wrong: one variable alone, or a base
    that is not an http:// or https:// address with a host."""
    base = environ.get(API_BASE_VARIABLE, "").strip()
    key = environ.get(API_KEY_VARIABLE, "").strip()
    if not base and not key:
        return None
    if not key:
        raise ValueError(f"{API_BASE_VARIABLE} is set and {API_KEY_VARIABLE} is not: set both, or neither")
    if not base:
        raise ValueError(f"{API_KEY_VARIABLE} is set and {API_BASE_VARIABLE} is not: set both, or neither")
    if not _is_http_address(base):
        raise ValueError(f"{API_BASE_VARIABLE} must be an http:// or https:// address with a host, such as http://localhost:11434/v1")
    return CompatibleApi(base, key)


def _is_http_address(base: str) -> bool:
    try:
        parts = urlsplit(base)
        return parts.scheme in BASE_SCHEMES and bool(parts.hostname)
    except ValueError:
        # urlsplit refuses a malformed address, such as an IPv6 host with
        # no closing bracket.
        return False
