#!/usr/bin/env python3
"""A fake OpenAI-compatible streaming client for offline tests.

AICommander.call_llm_api() drives ``self.client.chat.completions.create(...)``
and iterates the returned stream, reading ``chunk.usage``, ``chunk.choices``
and ``chunk.choices[0].delta`` (with ``content``, ``reasoning_content``,
``thinking`` and ``tool_calls``). This module reproduces that exact surface so
tests can exercise the context/token accounting and truncation logic without a
network endpoint.

It is deliberately dependency-free (no `openai` import) so the test suite runs
in any environment that can import `aic`.
"""

from typing import Any, Dict, List, Optional


class FakeFunctionChunk:
    """Partial function payload inside a tool-call delta chunk."""

    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class FakeToolCallChunk:
    """One incremental tool-call delta chunk (mirrors OpenAI's delta.tool_calls)."""

    def __init__(self, index: int, id: str = "", name: str = "", arguments: str = ""):
        self.index = index
        self.id = id
        self.function = FakeFunctionChunk(name=name, arguments=arguments)


class FakeDelta:
    """Delta payload for a single streamed chunk."""

    def __init__(self, content: str = "", reasoning_content: str = "",
                 thinking: str = "", tool_calls: Optional[List[FakeToolCallChunk]] = None):
        self.content = content or None
        self.reasoning_content = reasoning_content or None
        self.thinking = thinking or None
        self.tool_calls = tool_calls


class FakeUsage:
    """Exact token usage reported on the final stream chunk."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeChoice:
    def __init__(self, delta: Optional[FakeDelta] = None):
        self.delta = delta


class FakeChunk:
    """One element of the streamed response."""

    def __init__(self, usage: Optional[FakeUsage] = None, choices: Optional[List[FakeChoice]] = None):
        self.usage = usage
        self.choices = choices or []


class FakeCompletions:
    """`chat.completions` namespace with a scriptable `create()`.

    `create()` returns an iterable of FakeChunk objects. If `script` is not
    provided, it synthesizes a single content chunk followed by a final chunk
    carrying exact usage (so the context counter is reconciled).
    """

    def __init__(self, script: Optional[List[FakeChunk]] = None):
        # Each call to create() pops the next script entry, so a multi-turn
        # test can script a different response per LLM call.
        self._script = list(script or [])
        self.calls: List[Dict[str, Any]] = []  # every request_params dict
        self.last_messages: Optional[List[Dict[str, Any]]] = None
        self.last_params: Optional[Dict[str, Any]] = None

    def create(self, **params):
        self.calls.append(params)
        self.last_params = params
        self.last_messages = params.get("messages")
        if self._script:
            chunks = self._script.pop(0)
        else:
            chunks = self._default_response()
        return iter(chunks)

    @staticmethod
    def _default_response() -> List[FakeChunk]:
        """A trivial one-turn response: one content token, then exact usage."""
        return [
            FakeChunk(choices=[FakeChoice(FakeDelta(content="pong"))]),
            FakeChunk(usage=FakeUsage(prompt_tokens=10, completion_tokens=1)),
        ]


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeOpenAIClient:
    """Duck-typed stand-in for `openai.OpenAI`; assign to `commander.client`."""

    def __init__(self, script: Optional[List[FakeChunk]] = None):
        self._completions = FakeCompletions(script)
        self.chat = FakeChat(self._completions)

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._completions.calls

    @property
    def last_messages(self) -> Optional[List[Dict[str, Any]]]:
        return self._completions.last_messages

    @property
    def last_params(self) -> Optional[Dict[str, Any]]:
        return self._completions.last_params


# --------------------------------------------------------------------------- #
# Convenience builders for common scripted responses                         #
# --------------------------------------------------------------------------- #

def content_chunk(text: str) -> FakeChunk:
    """A stream chunk that emits plain assistant content."""
    return FakeChunk(choices=[FakeChoice(FakeDelta(content=text))])


def thinking_chunk(text: str) -> FakeChunk:
    """A stream chunk that emits reasoning/thinking tokens."""
    return FakeChunk(choices=[FakeChoice(FakeDelta(reasoning_content=text))])


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> FakeChunk:
    """The final chunk carrying exact server-reported token usage."""
    return FakeChunk(usage=FakeUsage(prompt_tokens=prompt_tokens,
                                     completion_tokens=completion_tokens))


def tool_call_chunk(index: int, id: str = "", name: str = "", arguments: str = "") -> FakeChunk:
    """A stream chunk carrying an incremental tool-call delta."""
    return FakeChunk(choices=[FakeChoice(FakeDelta(tool_calls=[
        FakeToolCallChunk(index=index, id=id, name=name, arguments=arguments)
    ]))])
