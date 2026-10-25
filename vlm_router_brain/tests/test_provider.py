"""The any-llm adapter: what each call sends, what it reads back, and how
each of any-llm's failure classes comes out, retryable or not. any-llm's
client is a fake: no network and no key."""

import base64
from types import SimpleNamespace

import pytest
from any_llm.exceptions import (
    AuthenticationError,
    ContentFilterError,
    GatewayTimeoutError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)
from conftest import BOTH_KEYS, TEST_CATALOG

from brain_core.ports import ExpectedFailure, FailureKind

from vlm_router_brain import provider as provider_module
from vlm_router_brain.parsing import Answer
from vlm_router_brain.provider import AnyLlmProvider, ProviderFailure, is_retryable

FAST = TEST_CATALOG.get("fast").routes[0]
CAREFUL = TEST_CATALOG.get("careful").routes[0]


def response(text: str, prompt_tokens=1200, completion_tokens=40, reasoning_tokens=30):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)


class FakeClient:
    """any-llm's provider client: records each acompletion and answers, or
    raises, what the test hands it."""

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class Made(list):
    """Each AnyLLM.create the adapter made, as (provider, options, client),
    and the outcome the next client's calls give."""


@pytest.fixture
def clients(monkeypatch):
    made = Made()
    made.outcome = response('{"boxes": []}')

    def create(provider, **kwargs):
        client = FakeClient(made.outcome)
        made.append((provider, kwargs, client))
        return client

    monkeypatch.setattr(provider_module.AnyLLM, "create", staticmethod(create))
    return made


async def test_a_call_sends_the_picture_then_the_question_with_the_routes_settings(clients):
    clients.outcome = response('{"boxes": []}')
    reply = await AnyLlmProvider(BOTH_KEYS).ask(FAST, b"\xff\xd8jpeg", "Find the cup.", 12.5)
    [(name, options, client)] = clients
    # The key is passed explicitly, every provider raises any-llm's
    # classes, and the SDK never retries on its own.
    assert name == "gemini"
    assert options == dict(api_key="test-gemini-key", unified_exceptions=True, max_retries=0)
    [call] = client.calls
    assert call["model"] == "gemini-fast" and call["timeout"] == 12.5
    assert call["reasoning_effort"] == "medium" and call["response_format"] is Answer
    [message] = call["messages"]
    picture, question = message["content"]
    assert picture == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()}}
    assert question == {"type": "text", "text": "Find the cup."}
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.reasoning_tokens) == ('{"boxes": []}', 1200, 40, 30)
    assert reply.latency_s >= 0.0


async def test_one_client_per_provider_with_its_own_key(clients):
    provider = AnyLlmProvider(BOTH_KEYS)
    await provider.ask(FAST, b"jpeg", "q", 10.0)
    await provider.ask(FAST, b"jpeg", "q", 10.0)
    await provider.ask(CAREFUL, b"jpeg", "q", 10.0)
    assert [(name, options["api_key"]) for name, options, _ in clients] == [
        ("gemini", "test-gemini-key"), ("openrouter", "test-openrouter-key"),
    ]


async def test_an_answer_with_no_usage_reports_no_tokens(clients):
    clients.outcome = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None)
    reply = await AnyLlmProvider(BOTH_KEYS).ask(FAST, b"jpeg", "q", 10.0)
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.reasoning_tokens) == ("", 0, 0, 0)


@pytest.mark.parametrize("error, retryable", [
    (RateLimitError("slow down", status_code=429), True),
    (GatewayTimeoutError("the gateway timed out", status_code=504), True),
    (UpstreamProviderError("bad gateway", status_code=502), True),
    (ProviderError("overloaded", status_code=503), True),
    (ProviderError("internal error", status_code=500), True),
    # A lost connection or a timeout of the call has no status.
    (ProviderError("connection reset"), False),
    (UpstreamProviderError("the upstream said no"), False),
    (ProviderError("a 2xx the SDK could not parse", status_code=200), False),
    (AuthenticationError("bad key", status_code=401), False),
    (InvalidRequestError("bad request", status_code=400), False),
    (ModelNotFoundError("no such model", status_code=404), False),
    (ContentFilterError("filtered"), False),
])
async def test_each_failure_class_comes_out_retryable_or_not(clients, error, retryable):
    assert is_retryable(error) is retryable
    clients.outcome = error
    with pytest.raises(ProviderFailure) as raised:
        await AnyLlmProvider(BOTH_KEYS).ask(FAST, b"jpeg", "q", 10.0)
    failure = raised.value
    assert isinstance(failure, ExpectedFailure) and failure.retryable is retryable
    assert failure.kind is FailureKind.ATTEMPT_FAILED
    assert failure.message == f"fast failed: {type(error).__name__}: {' '.join(str(error).split())}"


async def test_a_long_provider_message_is_one_line_cut_to_its_limit(clients):
    clients.outcome = ProviderError("line one\n" + "x" * 1000)
    with pytest.raises(ProviderFailure) as raised:
        await AnyLlmProvider(BOTH_KEYS).ask(FAST, b"jpeg", "q", 10.0)
    detail = raised.value.message.removeprefix("fast failed: ProviderError: ")
    assert "\n" not in detail and len(detail) == provider_module.DETAIL_CHARS and detail.endswith("...")


async def test_a_provider_with_no_key_fails_without_a_client(clients):
    with pytest.raises(ProviderFailure) as raised:
        await AnyLlmProvider({"OPENROUTER_API_KEY": "set"}).ask(FAST, b"jpeg", "q", 10.0)
    assert raised.value.message == "fast failed: MissingApiKeyError: GEMINI_API_KEY is not set"
    assert raised.value.retryable is False and list(clients) == []
