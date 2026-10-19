"""The one call the brain makes to a model: a picture and a question to one
route, one answer back. `Provider` is the port the detector and the switch
probe call; `AnyLlmProvider`, any-llm behind it, is the one implementation,
and the tests pass a scripted fake.

`AnyLlmProvider` sends the picture as a base64 JPEG data URL before the
question, in an OpenAI-style user message, with the route's reasoning
effort and `parsing.Answer` as the structured output. It awaits any-llm's
`acompletion`, so a call never blocks the event loop. Each provider's
client is made once, with the key of its route from the node's environment
passed explicitly (so gemini never falls back to another variable),
`unified_exceptions=True` (so every provider raises any-llm's classes) and
`max_retries=0` (so the detector's one retry is the only one).

A failure is a `ProviderFailure`, an expected failure: "<model> failed:
<class>: <message>", and whether the detector may ask again. It may for a
429, a gateway timeout, or a provider or upstream error with a 5xx status,
the failures that come back at once; any other, a lost connection among
them, fails the search. The message never holds the route's key: each
occurrence of the key in the provider's text becomes "***".
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Mapping, Protocol

from any_llm import AnyLLM
from any_llm.exceptions import (
    AnyLLMError,
    GatewayTimeoutError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)

from brain_core.ports import ExpectedFailure

from .catalog import Route
from .keys import KEY_VARIABLES, key_for
from .parsing import Answer

# The longest provider message a failure carries.
DETAIL_CHARS = 300
# What stands for the route's key in a provider message.
KEY_MASK = "***"


@dataclass(frozen=True)
class Reply:
    """One answer: its text, the tokens it took and how long it took."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0


class ProviderFailure(ExpectedFailure):
    def __init__(self, model_name: str, error_class: str, detail: str, *, retryable: bool) -> None:
        super().__init__(f"{model_name} failed: {error_class}: {detail}")
        self.retryable = retryable


class Provider(Protocol):
    async def ask(self, route: Route, image: bytes, prompt: str, timeout_s: float) -> Reply:
        """One answer from `route` within `timeout_s`, or a ProviderFailure."""
        ...


def is_retryable(error: AnyLLMError) -> bool:
    """Whether a failure is one that comes back at once and may be asked
    again: a 429, a gateway timeout, or a 5xx."""
    if isinstance(error, (RateLimitError, GatewayTimeoutError)):
        return True
    if isinstance(error, (ProviderError, UpstreamProviderError)):
        return error.status_code is not None and 500 <= error.status_code < 600
    return False


def image_message(image: bytes, prompt: str) -> list[dict]:
    """The picture, then the question, as one OpenAI-style user message."""
    url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ],
    }]


class AnyLlmProvider:
    def __init__(self, environ: Mapping[str, str] = os.environ) -> None:
        self._environ = environ
        # Each provider's client, and the key it was made with.
        self._clients: dict[str, tuple[AnyLLM, str]] = {}

    async def ask(self, route: Route, image: bytes, prompt: str, timeout_s: float) -> Reply:
        client, key = self._client(route)
        started = time.perf_counter()
        try:
            response = await client.acompletion(
                model=route.model,
                messages=image_message(image, prompt),
                reasoning_effort=route.reasoning_effort,
                response_format=Answer,
                timeout=timeout_s,
            )
        except AnyLLMError as error:
            raise ProviderFailure(route.model_name, type(error).__name__, _detail(error, key), retryable=is_retryable(error)) from error
        latency_s = time.perf_counter() - started
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        return Reply(
            text=response.choices[0].message.content or "",
            input_tokens=(usage.prompt_tokens or 0) if usage is not None else 0,
            output_tokens=(usage.completion_tokens or 0) if usage is not None else 0,
            reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0,
            latency_s=latency_s,
        )

    def _client(self, route: Route) -> tuple[AnyLLM, str]:
        """The client of the route's provider and the key it was made with."""
        made = self._clients.get(route.provider)
        if made is not None:
            return made
        key = key_for(route.provider, self._environ)
        if not key:
            raise ProviderFailure(
                route.model_name, "MissingApiKeyError", f"{KEY_VARIABLES[route.provider]} is not set", retryable=False
            )
        made = (AnyLLM.create(route.provider, api_key=key, unified_exceptions=True, max_retries=0), key)
        self._clients[route.provider] = made
        return made


def _detail(error: AnyLLMError, key: str) -> str:
    """The provider's message on one line, at most DETAIL_CHARS long, with
    each occurrence of `key` replaced by KEY_MASK. The key is masked before
    the cut, so a key across the limit leaves none of its characters."""
    text = str(error)
    if key:
        text = text.replace(key, KEY_MASK)
    text = " ".join(text.split())
    return text if len(text) <= DETAIL_CHARS else text[: DETAIL_CHARS - 3] + "..."
