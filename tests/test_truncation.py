#!/usr/bin/env python3
"""Test suite for the truncation feature and core functionality of aic.py.

Two truncation mechanisms are exercised:

1. OUTPUT truncation in AICommander.execute_bash_command(): command output is
   capped at max_output_bytes and the literal sentinel
   `AICommander.OUTPUT_TRUNCATION_SENTINEL` ("output too long: truncated") is
   appended when the limit is exceeded.

2. CONVERSATION-HISTORY compression in AICommander._context_compress():
   the trigger (history over the prompt budget) is computed once, then the
   configured compression algorithm (`self.compress_algorithm`, selected via
   the `--compress-alg` CLI flag) is dispatched to. The default algorithm,
   "truncate", condenses oversized tool outputs in place, then drops the
   oldest messages (from index 2 onward) until the total token estimate is
   under max_prompt_len. The system prompt (index 0) and first user
   instruction (index 1) are always preserved. More algorithms are added by
   defining `_compress_<name>` methods.

The script also tests supporting helpers (_estimate_message_tokens,
_estimate_input_tokens, _validate_messages, process_llm_response,
handle_function_call) and, unless --skip-live is given, performs live
integration tests against a real OpenAI-compatible endpoint.

Usage:
    python3 test_truncation.py \
        --api-base https://api.example.com/v1 \
        --api-key YOUR_KEY \
        --model your-model

Optional flags:
    --skip-live   skip tests that hit the network (offline unit tests always run)
    --run-loop    additionally run the full agent loop (requires network)

Exits with a non-zero status if any test fails.
"""

import argparse
import os
import queue
import sys
import threading

# This script lives in ./tests, so the repo root (where aic.py lives) is the
# parent directory. Insert it at the front of sys.path so `import aic` works
# regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also make this tests/ directory importable so `import fake_openai` works
# regardless of how the script is invoked (directly or via a runner).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aic
from aic import AICommander, TUISink
import fake_openai


class TokenUsageTracker:
    """Test-only stand-in for the TUI's live token accounting.

    Mirrors how the TUI's _apply_token_usage / _track_token accumulate the
    "Total Tokens" counter: the agent emits an estimate=True event (input-token
    guess) before each call and an estimate=False event with exact server usage
    when the call finishes; streamed output chunks count live. Only the exact
    numbers are authoritative, so estimates and streamed chunks are rolled back
    on reconcile to avoid double counting. This class is defined here (not in
    aic.py) because it is purely a test harness, not production code.
    """

    def __init__(self):
        self.context_tokens = 0
        self._pending_input_estimate = 0
        self._pending_output_estimate = 0

    def on_call_start(self, input_estimate: int) -> None:
        # Roll back stale in-flight estimates from a prior call that aborted
        # before emitting exact usage, so they aren't counted against the new
        # call's context.
        if self._pending_input_estimate or self._pending_output_estimate:
            self.context_tokens -= self._pending_input_estimate
            self.context_tokens -= self._pending_output_estimate
        self._pending_input_estimate = input_estimate
        self._pending_output_estimate = 0
        self.context_tokens += input_estimate

    def on_stream_token(self) -> None:
        self.context_tokens += 1
        self._pending_output_estimate += 1

    def on_call_finish(self, exact_input: int, exact_output: int) -> None:
        # Swap the live estimates for the exact server totals.
        self.context_tokens -= self._pending_input_estimate
        self.context_tokens -= self._pending_output_estimate
        self.context_tokens += exact_input
        self.context_tokens += exact_output
        self._pending_input_estimate = 0
        self._pending_output_estimate = 0

    def reset(self) -> None:
        self.context_tokens = 0
        self._pending_input_estimate = 0
        self._pending_output_estimate = 0


class _StubOpenAI:
    """Minimal stand-in for `openai.OpenAI`.

    The system-installed `openai` package in some environments fails to
    construct (httpx version mismatch), which breaks AICommander.__init__.
    Since offline tests replace the client with a fake before any API call, a
    no-op stub is sufficient here. Live tests can still be run against a real
    endpoint by overriding `aic.OpenAI` before constructing a commander.
    """

    def __init__(self, *args, **kwargs):
        pass


# Patch the module-level OpenAI reference so AICommander can be constructed
# without hitting the (possibly broken) real client. `make_fake_commander`
# replaces `commander.client` with the fake streaming endpoint before use.
aic.OpenAI = _StubOpenAI


class RecordingSink(TUISink):
    """A TUISink that auto-approves and records every emitted event.

    Subclassing TUISink makes AICommander.execute_bash_command() treat this as
    a TUI sink, so it skips stdin monitoring and can run headless. Every
    emitted event is appended to self.events as (kind, payload) tuples, and
    input() always returns "y" so the approval gate never blocks.
    """

    def __init__(self):
        super().__init__(queue.Queue(), threading.Event())
        self.auto_approve = True
        self.events = []

    def emit(self, kind, payload):
        self.events.append((kind, dict(payload)))

    def input(self, prompt):
        self.events.append(("INPUT", {"prompt": prompt}))
        return "y"

    def close(self):
        pass


class TestRunner:
    """Minimal pass/fail test registry with a human-readable summary."""

    def __init__(self, args):
        self.args = args
        self.results = []  # (name, passed, detail)

    def check(self, name, condition, detail=""):
        passed = bool(condition)
        self.results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not passed else ""))
        return passed

    def make_commander(self, **overrides):
        """Build an AICommander wired to a fresh RecordingSink.

        Overrides let individual tests shrink max_output_bytes / max_prompt_len
        or bound the agent loop without sharing state between tests.
        """
        # The OpenAI client requires a non-empty api_key even when it never
        # connects, so offline tests substitute a placeholder when none was
        # supplied on the command line.
        api_key = self.args.api_key or "test-key"
        params = dict(
            api_base=self.args.api_base,
            model=self.args.model or "test-model",
            api_key=api_key,
            auto_approve=True,
            show_thinking=False,
            command_timeout=30,
            max_prompt_len=20000,
            max_output_bytes=10240,
            debug=False,
            sink=RecordingSink(),
            persist_history=False,
            max_steps=500,
        )
        params.update(overrides)
        return AICommander(**params)

    def make_fake_commander(self, script=None, **overrides):
        """Build an AICommander whose LLM client is a fake streaming endpoint.

        The real OpenAI client is replaced with a FakeOpenAIClient so
        call_llm_api() runs entirely offline. `script` is a list of FakeChunk
        lists (one per LLM call); if omitted, a trivial one-turn response is
        used. Returns (commander, fake_client).
        """
        c = self.make_commander(**overrides)
        fake = fake_openai.FakeOpenAIClient(script)
        c.client = fake
        # The background task-title summarizer issues its own LLM call from a
        # daemon thread and would race the agent loop for the scripted
        # responses, nondeterministically replacing the first scripted turn
        # with the fake client's default "pong". Disable it: a scripted
        # client has exactly one consumer.
        c.task_summary_enabled = False
        return c, fake

    # ------------------------------------------------------------------ #
    # 1. Output truncation (execute_bash_command)                        #
    # ------------------------------------------------------------------ #

    def test_output_truncation(self):
        c = self.make_commander(max_output_bytes=100)
        output, code = c.execute_bash_command("python3 -c \"print('x'*1000)\"")
        self.check(
            "output truncation: sentinel appended",
            output.endswith(c.OUTPUT_TRUNCATION_SENTINEL),
            f"output did not end with sentinel; len={len(output)}",
        )
        self.check(
            "output truncation: capped at max_output_bytes",
            len(output) <= 100 + len(c.OUTPUT_TRUNCATION_SENTINEL) + 1,
            f"output too long: {len(output)}",
        )
        self.check(
            "output truncation: retains leading command output",
            output.startswith("x" * 100),
            f"unexpected output head: {output[:20]!r}",
        )
        cmd_complete = [e for e in c.sink.events if e[0] == "CMD_COMPLETE"]
        self.check("output truncation: CMD_COMPLETE emitted", len(cmd_complete) >= 1)
        if cmd_complete:
            self.check(
                "output truncation: CMD_COMPLETE carries truncated output",
                cmd_complete[-1][1]["output"].endswith(c.OUTPUT_TRUNCATION_SENTINEL),
            )

    def test_small_output(self):
        c = self.make_commander(max_output_bytes=10240)
        output, code = c.execute_bash_command("echo hello")
        self.check(
            "small output: no sentinel",
            c.OUTPUT_TRUNCATION_SENTINEL not in output,
            repr(output),
        )
        self.check("small output: contains command result", "hello" in output, repr(output))
        self.check("small output: exit code zero", code == 0, f"exit={code}")

    # ------------------------------------------------------------------ #
    # 2. Conversation-history compression (_context_compress)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_history(system, user, tools):
        hist = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for i, t in enumerate(tools):
            hist.append({"role": "tool", "tool_call_id": f"call_{i}", "content": t})
        return hist

    def test_history_condense(self):
        # Oversized tool outputs are condensed in place; nothing is dropped
        # because the condensed history fits under the limit. A "gpt" model is
        # used so the prompt budget equals max_prompt_len (no output-token
        # reservation); the reservation itself is covered by test_prompt_budget.
        c = self.make_commander(max_prompt_len=100, model="gpt-test",
                                compress_algorithm="truncate")
        big = "A" * 500
        c.conversation_history = self._build_history("sys", "hello", [big, big])
        c._context_compress()
        tools = [m for m in c.conversation_history if m["role"] == "tool"]
        self.check(
            "history condense: tool outputs condensed",
            all(m["content"] == "<condensed tool output>" for m in tools),
            repr(tools),
        )
        self.check(
            "history condense: system prompt preserved",
            c.conversation_history[0]["role"] == "system",
            repr(c.conversation_history[0]),
        )
        self.check(
            "history condense: first user instruction preserved",
            c.conversation_history[1]["role"] == "user",
            repr(c.conversation_history[1]),
        )
        self.check(
            "history condense: nothing dropped",
            len(c.conversation_history) == 4,
            f"len={len(c.conversation_history)}",
        )
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        self.check("history condense: under limit", total <= 100, f"total={total}")

    def test_history_drop(self):
        # Even after condensing, the history is over the limit, so the oldest
        # messages (from index 2 onward) are dropped while system + first user
        # instruction survive. A "gpt" model is used so the prompt budget equals
        # max_prompt_len (no output-token reservation).
        c = self.make_commander(max_prompt_len=30, model="gpt-test",
                                compress_algorithm="truncate")
        big = "A" * 500
        c.conversation_history = self._build_history("sys", "hello", [big, big, big, big])
        original_len = len(c.conversation_history)
        c._context_compress()
        self.check(
            "history drop: system prompt preserved",
            c.conversation_history[0]["role"] == "system",
            repr(c.conversation_history[0]),
        )
        self.check(
            "history drop: first user instruction preserved",
            c.conversation_history[1]["role"] == "user",
            repr(c.conversation_history[1]),
        )
        tools = [m for m in c.conversation_history if m["role"] == "tool"]
        self.check(
            "history drop: remaining tools condensed",
            all(m["content"] == "<condensed tool output>" for m in tools),
            repr(tools),
        )
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        self.check("history drop: under limit", total <= 30, f"total={total}")
        self.check(
            "history drop: oldest messages dropped",
            len(c.conversation_history) < original_len,
            f"len={len(c.conversation_history)}",
        )

    def test_history_noop(self):
        # A history already under the limit is left completely untouched.
        c = self.make_commander(max_prompt_len=1000, model="gpt-test")
        c.conversation_history = self._build_history("sys", "hello", ["small"])
        before = [dict(m) for m in c.conversation_history]
        c._context_compress()
        self.check("history no-op: unchanged when under limit", c.conversation_history == before,
                   repr(c.conversation_history))

    def test_history_truncate_leaves_headroom(self):
        # The truncate algorithm must leave ~compress_target headroom (retain
        # that fraction of the budget), not fill the full window. Truncating to
        # the FULL budget makes every following prompt exceed the limit again
        # and re-compress on each step, slowly losing more history than
        # necessary. Retaining the target fraction lets the agent run several
        # more steps before the next compression, matching the
        # context-compressor-llm algorithm.
        c = self.make_commander(max_prompt_len=100, model="gpt-test",
                                compress_algorithm="truncate")
        big = "A" * 500
        c.conversation_history = self._build_history("sys", "hello", [big, big, big, big])
        c._context_compress()
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        # Retained history must fit within the compress_target fraction of the
        # budget, leaving headroom for new turns (while still preserving system
        # + first user instruction).
        target = int(100 * c.compress_target)
        self.check("truncate headroom: under target fraction of budget",
                   total <= target,
                   f"total={total} budget=100 target={target}")
        self.check("truncate headroom: system prompt preserved",
                   c.conversation_history[0]["role"] == "system",
                   repr(c.conversation_history[0]))
        self.check("truncate headroom: first user instruction preserved",
                   len(c.conversation_history) >= 2
                   and c.conversation_history[1]["role"] == "user",
                   repr(c.conversation_history))

    def test_truncate_does_not_over_truncate_tool_outputs(self):
        # Pass 1 must condense tool outputs only as many as needed to reach the
        # target fraction (oldest first), NOT every oversized output. When tool
        # outputs dominate the history, condensing them ALL collapses the
        # context far below the target (e.g. 22505 -> 1476 tokens), wasting
        # useful detail. Here the budget can absorb most of the messages, so
        # several tool outputs must keep their full content while the total
        # still lands within the target fraction of the budget.
        c = self.make_commander(max_prompt_len=1000, model="gpt-test",
                                compress_algorithm="truncate")
        hist = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        for i in range(10):
            hist.append({"role": "tool", "tool_call_id": f"c{i}",
                         "content": "A" * 300})
        c.conversation_history = hist
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        self.check("truncate no-over-truncate: initial over budget",
                   total > 1000, f"total={total}")
        c._context_compress()
        final = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        target = int(1000 * c.compress_target)
        self.check("truncate no-over-truncate: within target fraction of budget",
                   final <= target, f"final={final}")
        # At least one tool message must keep its full content (not condensed
        # to the placeholder) because the budget can absorb it.
        full = [m for m in c.conversation_history
                if m.get("role") == "tool" and len(m.get("content", "")) > 30]
        self.check("truncate no-over-truncate: some tool outputs kept full",
                   len(full) >= 1, f"full_tools={len(full)} final={final}")

    def test_default_algorithm_is_context_compressor_llm(self):
        # The "context-compressor-llm" algorithm is selected by default, so a
        # commander built without an explicit algorithm compresses via that
        # path; the "truncate" dispatcher also still exists as a fallback.
        c = self.make_commander(max_prompt_len=100, model="gpt-test")
        self.check("default algorithm is context-compressor-llm",
                   c.compress_algorithm == "context-compressor-llm",
                   f"algorithm={c.compress_algorithm}")
        self.check("context-compressor-llm dispatcher exists",
                   callable(getattr(c, "_compress_context_compressor_llm", None)))
        self.check("truncate dispatcher still exists",
                   callable(getattr(c, "_compress_truncate", None)))

    def test_compress_target_configurable(self):
        # The retained-headroom fraction is configurable via `compress_target`
        # (default 0.4, i.e. 40%). A commander built with a custom target must
        # retain only that fraction of the budget, not the hardcoded 0.6 used
        # previously.
        c = self.make_commander(max_prompt_len=100, model="gpt-test",
                                compress_algorithm="truncate",
                                compress_target=0.3)
        self.check("compress_target default wiring",
                   c.compress_target == 0.3, f"target={c.compress_target}")
        big = "A" * 500
        c.conversation_history = self._build_history("sys", "hello", [big, big, big, big])
        c._context_compress()
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        target = int(100 * c.compress_target)
        self.check("compress_target: under custom fraction of budget",
                   total <= target,
                   f"total={total} budget=100 target={target}")

    def test_compress_target_default(self):
        # The default compress_target is 0.4 (40%), replacing the previous
        # hardcoded 0.6 (60%).
        c = self.make_commander(max_prompt_len=100, model="gpt-test")
        self.check("compress_target default is 0.4",
                   c.compress_target == 0.4, f"target={c.compress_target}")

    def test_explicit_algorithm_dispatch(self):
        # An explicit "truncate" selection is honored, and an unknown algorithm
        # falls back to the default truncate path rather than erroring.
        big = "A" * 500
        c = self.make_commander(max_prompt_len=30, model="gpt-test",
                                compress_algorithm="truncate")
        c.conversation_history = self._build_history("sys", "hello", [big, big])
        c._context_compress()
        self.check(
            "explicit truncate: under limit",
            sum(c._estimate_message_tokens(m) for m in c.conversation_history) <= 30,
            f"len={len(c.conversation_history)}",
        )
        c2 = self.make_commander(max_prompt_len=30, model="gpt-test", compress_algorithm="bogus")
        c2.conversation_history = self._build_history("sys", "hello", [big, big])
        c2._context_compress()
        self.check(
            "unknown algorithm falls back to truncate",
            sum(c2._estimate_message_tokens(m) for m in c2.conversation_history) <= 30,
            f"len={len(c2.conversation_history)}",
        )

    def test_context_compressor_llm(self):
        # The "context-compressor-llm" algorithm is dispatched to via the
        # hyphenated CLI name (mapped to _compress_context_compressor_llm) and
        # performs anchored-summary incremental compression: the oldest prefix
        # is folded into an anchored summary (a system message) while the newest
        # suffix is retained and the system prompt stays byte-identical. The
        # LLM summarizer is stubbed so the test runs fully offline; when the
        # context-compressor-llm package is missing the algorithm falls back to
        # truncate, which still must bring the history under budget.
        c = self.make_commander(max_prompt_len=60, model="gpt-test",
                                compress_algorithm="context-compressor-llm")
        big = "A" * 500
        c.conversation_history = self._build_history("sys", "hello", [big, big, big])
        # Stub the LLM summarizer so no network call happens.
        c._llm_summarize = lambda messages, previous_summary: "[anchored summary]"
        c._context_compress()
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        self.check("context-compressor-llm: under limit", total <= 60, f"total={total}")
        self.check("context-compressor-llm: system prompt preserved",
                   c.conversation_history[0]["role"] == "system",
                   repr(c.conversation_history[0]))
        # The FIRST user instruction (the original task) must survive
        # compression verbatim at index 1, just like the truncate algorithm,
        # so the LLM never forgets what it was initially asked to do.
        self.check("context-compressor-llm: first user instruction preserved",
                   len(c.conversation_history) >= 2
                   and c.conversation_history[1].get("role") == "user"
                   and c.conversation_history[1].get("content") == "hello",
                   repr(c.conversation_history))
        if aic._CONTEXT_COMPRESSOR_LLM_AVAILABLE:
            self.check("context-compressor-llm: anchored summary folded in",
                       "[Prior conversation summary]" in c.conversation_history[0]["content"],
                       repr(c.conversation_history[0]))
            self.check("context-compressor-llm: no mid-list system message",
                       not any(m["role"] == "system" for m in c.conversation_history[1:]),
                       repr(c.conversation_history))
        else:
            self.check("context-compressor-llm: falls back when lib missing", True)

    def test_prompt_budget_reserves_output(self):
        # For non-"gpt" models the prompt must stay under
        # max_prompt_len - max_tokens so the completion fits in the window.
        c = self.make_commander(max_prompt_len=1000, model="test-model")
        c.max_tokens = 400
        self.check("budget: non-gpt reserves output",
                   c._prompt_budget() == 600,
                   f"budget={c._prompt_budget()}")
        gpt = self.make_commander(max_prompt_len=1000, model="gpt-test")
        self.check("budget: gpt uses full window",
                   gpt._prompt_budget() == 1000,
                   f"budget={gpt._prompt_budget()}")

    def test_history_truncate_respects_output_reservation(self):
        # A non-"gpt" model with a tiny max_tokens leaves room for output, so a
        # history that fits the full window but not the window minus the output
        # reservation still gets truncated down to the reserved budget.
        c = self.make_commander(max_prompt_len=100, model="test-model",
                                compress_algorithm="truncate")
        c.max_tokens = 40  # budget = 60
        big = "A" * 500
        c.conversation_history = self._build_history("sys", "hello", [big, big, big])
        c._context_compress()
        total = sum(c._estimate_message_tokens(m) for m in c.conversation_history)
        self.check("reservation: truncated to reserved budget",
                   total <= c._prompt_budget(),
                   f"total={total} budget={c._prompt_budget()}")
        self.check("reservation: system prompt preserved",
                   c.conversation_history[0]["role"] == "system",
                   repr(c.conversation_history[0]))
        self.check("reservation: first user instruction preserved",
                   c.conversation_history[1]["role"] == "user",
                   repr(c.conversation_history[1]))

    # ------------------------------------------------------------------ #
    # 3. Supporting helpers                                              #
    # ------------------------------------------------------------------ #

    def test_estimate_tokens(self):
        c = self.make_commander()
        est = c._estimate_message_tokens({"role": "user", "content": "hello world this is a test"})
        self.check("estimate tokens: string message", est >= 4, f"est={est}")
        est_list = c._estimate_message_tokens(
            {"role": "user", "content": [{"type": "text", "text": "abc"}]}
        )
        self.check("estimate tokens: list content", est_list >= 4, f"est={est_list}")
        total = c._estimate_input_tokens([
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "y" * 100},
        ])
        self.check("estimate input tokens: sums", total > 0, f"total={total}")

    def test_validate_messages(self):
        c = self.make_commander()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": None},              # None -> ""
            {"role": "bogus", "content": "x"},              # invalid role dropped
            {"role": "tool", "content": "no id"},           # tool without id dropped
            {"role": "function", "content": "x"},           # function without name dropped
            {"role": "assistant", "content": "a1", "tool_calls": [{"id": "c1"}]},
            {"role": "assistant", "content": "a2", "tool_calls": [{"id": "c2"}]},  # consecutive
        ]
        out = c._validate_messages(msgs)
        self.check("validate: None content replaced", out[1]["content"] == "", repr(out))
        self.check(
            "validate: invalid roles dropped",
            all(m["role"] in ("system", "user", "assistant", "tool", "function", "developer") for m in out),
            repr(out),
        )
        self.check("validate: tool without id dropped", all(m["role"] != "tool" for m in out), repr(out))
        self.check("validate: function without name dropped", all(m["role"] != "function" for m in out), repr(out))
        assistants = [m for m in out if m["role"] == "assistant"]
        self.check("validate: consecutive assistant collapsed", len(assistants) == 1,
                   f"assistants={len(assistants)}")

    def test_validate_terminal(self):
        c = self.make_commander()
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "done"},
        ]
        out = c._validate_messages(msgs)
        self.check("validate: terminal assistant returns empty", out == [], repr(out))

    def test_process_response(self):
        c = self.make_commander()
        resp = {"choices": [{"message": {
            "role": "assistant",
            "content": "hello",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "execute_bash", "arguments": '{"command": "ls"}'}}],
            "reasoning_content": "thinking here",
        }}]}
        content, tcs, first_id, thinking, malformed = c.process_llm_response(resp)
        self.check("process response: content extracted", content == "hello", repr(content))
        self.check("process response: tool call parsed", len(tcs) == 1, repr(tcs))
        self.check("process response: command extracted", tcs[0]["command"] == "ls", repr(tcs))
        self.check("process response: first tool id", first_id == "c1", repr(first_id))
        self.check("process response: thinking extracted", thinking == "thinking here", repr(thinking))
        self.check("process response: no malformed", malformed == [], repr(malformed))

    def test_process_response_malformed(self):
        c = self.make_commander()
        resp = {"choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "execute_bash", "arguments": "{bad json"}}],
        }}]}
        content, tcs, first_id, thinking, malformed = c.process_llm_response(resp)
        self.check("process response malformed: flagged", len(malformed) == 1, repr(malformed))
        self.check("process response malformed: parse error set", bool(malformed[0]["parse_error"]),
                   repr(malformed))
        self.check("process response malformed: no valid tool calls", tcs == [], repr(tcs))
        self.check("process response malformed: empty content", content == "", repr(content))

    def test_handle_function_call(self):
        c = self.make_commander()
        r1 = c.handle_function_call({"name": "x", "arguments": ""})
        self.check("handle function call: no args", "No arguments" in r1, repr(r1))
        r2 = c.handle_function_call({"name": "x", "arguments": "{bad"})
        self.check("handle function call: bad json", "could not be parsed" in r2, repr(r2))
        r3 = c.handle_function_call({"name": "foo", "arguments": "{}"})
        self.check("handle function call: unknown function", "Unknown function: foo" in r3, repr(r3))

    # ------------------------------------------------------------------ #
    # 4. Context counter / token usage accounting                        #
    # ------------------------------------------------------------------ #

    def test_token_tracker_reconcile(self):
        # A complete call: estimate=True for input, streamed output tokens,
        # then exact usage. The cumulative context must equal the exact totals
        # with nothing double counted.
        tr = TokenUsageTracker()
        tr.on_call_start(100)          # live input estimate
        tr.on_stream_token()           # 1 output chunk
        tr.on_stream_token()           # another output chunk
        self.check("tracker: live estimate during call",
                   tr.context_tokens == 102, f"ctx={tr.context_tokens}")
        tr.on_call_finish(95, 3)       # exact server totals
        self.check("tracker: reconciled to exact totals",
                   tr.context_tokens == 98, f"ctx={tr.context_tokens}")
        self.check("tracker: pending input rolled back",
                   tr._pending_input_estimate == 0,
                   f"pending={tr._pending_input_estimate}")
        self.check("tracker: pending output rolled back",
                   tr._pending_output_estimate == 0,
                   f"pending={tr._pending_output_estimate}")

    def test_token_tracker_aborted_call(self):
        # If a call is aborted before emitting exact usage, starting the next
        # call rolls back the stale in-flight estimates so they aren't counted
        # against the new call's context.
        tr = TokenUsageTracker()
        tr.on_call_start(100)
        tr.on_stream_token()
        tr.on_stream_token()
        # Aborted: no on_call_finish. A new call starts with a fresh estimate.
        tr.on_call_start(50)
        self.check("tracker: stale estimates rolled back on new call",
                   tr.context_tokens == 50, f"ctx={tr.context_tokens}")
        tr.on_call_finish(48, 2)
        self.check("tracker: final exact after aborted call",
                   tr.context_tokens == 50, f"ctx={tr.context_tokens}")

    def test_token_tracker_reset(self):
        tr = TokenUsageTracker()
        tr.on_call_start(100)
        tr.on_stream_token()
        tr.reset()
        self.check("tracker: reset clears context", tr.context_tokens == 0,
                   f"ctx={tr.context_tokens}")
        self.check("tracker: reset clears pending input",
                   tr._pending_input_estimate == 0,
                   f"pending={tr._pending_input_estimate}")
        self.check("tracker: reset clears pending output",
                   tr._pending_output_estimate == 0,
                   f"pending={tr._pending_output_estimate}")

    def test_call_llm_api_emits_estimate_then_exact(self):
        # call_llm_api must emit a TOKEN_USAGE estimate=True event before the
        # call and an estimate=False event with exact numbers afterwards. The
        # RecordingSink captures every event, so we can assert the sequence.
        script = [
            [
                fake_openai.content_chunk("pong"),
                fake_openai.usage_chunk(prompt_tokens=12, completion_tokens=1),
            ]
        ]
        c, fake = self.make_fake_commander(script)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        resp = c.call_llm_api(msgs, use_tools=False)
        usage_events = [e for e in c.sink.events if e[0] == "TOKEN_USAGE"]
        self.check("context: two TOKEN_USAGE events emitted", len(usage_events) == 2,
                   f"count={len(usage_events)}")
        if len(usage_events) == 2:
            self.check("context: first event is estimate",
                       usage_events[0][1].get("estimate") is True,
                       repr(usage_events[0][1]))
            self.check("context: second event is exact",
                       usage_events[1][1].get("estimate") is False,
                       repr(usage_events[1][1]))
            self.check("context: exact input matches server",
                       usage_events[1][1].get("input_tokens") == 12,
                       repr(usage_events[1][1]))
            self.check("context: exact output matches server",
                       usage_events[1][1].get("output_tokens") == 1,
                       repr(usage_events[1][1]))
        # The fake client must have received the exact messages we passed.
        self.check("context: fake client got messages",
                   fake.last_messages == msgs, repr(fake.last_messages))
        self.check("context: response content extracted",
                   resp["choices"][0]["message"]["content"] == "pong",
                   repr(resp))

    def test_call_llm_api_streams_thinking_and_tool_calls(self):
        # Verify the streaming parser handles reasoning chunks and incremental
        # tool-call deltas, assembling the final tool call arguments.
        script = [
            [
                fake_openai.thinking_chunk("thinking..."),
                fake_openai.tool_call_chunk(0, id="call_1", name="execute_bash",
                                            arguments='{"command": "ls"}'),
                fake_openai.usage_chunk(prompt_tokens=5, completion_tokens=3),
            ]
        ]
        c, fake = self.make_fake_commander(script, show_thinking=True)
        msgs = [{"role": "user", "content": "hi"}]
        resp = c.call_llm_api(msgs)
        msg = resp["choices"][0]["message"]
        self.check("context: reasoning collected",
                   msg.get("reasoning_content") == "thinking...",
                   repr(msg.get("reasoning_content")))
        tcs = msg.get("tool_calls") or []
        self.check("context: tool call parsed", len(tcs) == 1, repr(tcs))
        if tcs:
            self.check("context: tool call id kept", tcs[0]["id"] == "call_1", repr(tcs[0]))
            fn = tcs[0]["function"]
            self.check("context: tool call name kept", fn["name"] == "execute_bash", repr(fn))
            self.check("context: tool call args kept",
                       fn["arguments"] == '{"command": "ls"}', repr(fn))

    def test_context_counter_after_full_loop(self):
        # Run a short agent loop against the fake endpoint and assert the
        # cumulative context counter equals the exact token totals across all
        # calls (no double counting from the live estimates).
        # Each call: content then exact usage. The first call returns plain
        # content (no completion marker), so the loop continues; the second call
        # returns the completion marker to end the loop.
        marker = AICommander.COMPLETION_MARKER
        script = [
            [
                fake_openai.content_chunk("pong"),
                fake_openai.usage_chunk(prompt_tokens=12, completion_tokens=1),
            ],
            [
                fake_openai.content_chunk(f"done {marker}"),
                fake_openai.usage_chunk(prompt_tokens=14, completion_tokens=2),
            ],
        ]
        c, fake = self.make_fake_commander(script, max_steps=3)
        c.run("do something")
        # The agent emitted TOKEN_USAGE (estimate then exact) and LLM_STREAM
        # (output chunk) events to the sink. Replay them through a fresh
        # TokenUsageTracker exactly as the TUI's _apply_token_usage / _track_token
        # would, and assert the cumulative context equals the exact totals with
        # no double counting from the live estimates.
        tracker = TokenUsageTracker()
        for kind, payload in c.sink.events:
            if kind == "TOKEN_USAGE":
                if payload.get("estimate"):
                    tracker.on_call_start(payload.get("input_tokens", 0) or 0)
                else:
                    tracker.on_call_finish(payload.get("input_tokens", 0) or 0,
                                           payload.get("output_tokens", 0) or 0)
            elif kind == "LLM_STREAM":
                tracker.on_stream_token()
        expected = 12 + 1 + 14 + 2
        self.check("context: cumulative equals exact totals",
                   tracker.context_tokens == expected,
                   f"ctx={tracker.context_tokens} expected={expected}")
        self.check("context: pending estimates cleared",
                   tracker._pending_input_estimate == 0 and
                   tracker._pending_output_estimate == 0,
                   f"pending_in={tracker._pending_input_estimate} "
                   f"pending_out={tracker._pending_output_estimate}")
        self.check("context: two LLM calls made", len(fake.calls) == 2,
                   f"calls={len(fake.calls)}")

    # ------------------------------------------------------------------ #
    # 5. Live integration tests (network)                                #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # 4b. Session persistence (.aicsession JSON file)                    #
    # ------------------------------------------------------------------ #

    def _session_commander(self, tmpdir, name=".aicsession", **overrides):
        """AICommander wired to a session file inside a throwaway directory."""
        path = os.path.join(tmpdir, name)
        c = self.make_commander(session_file=path, **overrides)
        return c, path

    def test_session_save_writes_json(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            c.conversation_history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_0", "type": "function",
                     "function": {"name": "execute_bash",
                                  "arguments": "{\"command\": \"ls\"}"}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "file.txt"},
            ]
            self.check("session: save_session reports success", c.save_session("unit_test"))
            self.check("session: file created", os.path.exists(path), path)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            try:
                data = json.loads(raw)
            except ValueError as e:
                self.check("session: file is valid JSON", False, str(e))
                return
            self.check("session: file is valid JSON", True)
            self.check("session: version recorded",
                       data.get("aic_session_version") == aic._SESSION_VERSION, repr(data.get("aic_session_version")))
            self.check("session: reason recorded", data.get("reason") == "unit_test")
            self.check("session: messages saved verbatim",
                       data.get("messages") == c.conversation_history,
                       repr(data.get("messages"))[:200])
            self.check("session: human-readable (indented) JSON", "\n  \"messages\"" in raw, raw[:120])
            self.check("session: no temp files left behind",
                       not [f for f in os.listdir(tmp) if ".tmp." in f], os.listdir(tmp))

    def test_session_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_9", "type": "function",
                     "function": {"name": "execute_bash",
                                  "arguments": "{\"command\": \"echo hi\"}"}}]},
                {"role": "tool", "tool_call_id": "call_9", "content": "hi"},
                {"role": "user", "content": "and now?"},
            ]
            c.conversation_history = [dict(m) for m in history]
            c.usage_totals = {"input_tokens": 11, "output_tokens": 22,
                              "cached_tokens": 3, "billable_tokens": 30}
            c.step_count = 7
            c.save_session("unit_test")

            c2, _ = self._session_commander(tmp)
            self.check("session reload: reports restored", c2.load_session(notify=False))
            self.check("session reload: complete history restored exactly",
                       c2.conversation_history == history, repr(c2.conversation_history)[:300])
            self.check("session reload: session id restored",
                       c2.session_id == c.session_id, f"{c2.session_id} != {c.session_id}")
            self.check("session reload: step count restored", c2.step_count == 7, str(c2.step_count))
            self.check("session reload: usage totals restored",
                       c2.usage_totals == {"input_tokens": 11, "output_tokens": 22,
                                           "cached_tokens": 3, "billable_tokens": 30},
                       repr(c2.usage_totals))

    def test_session_final_answer_survives_resume(self):
        """Regression: a session that ended with the agent's final answer must
        restore that answer verbatim instead of silently dropping it."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            c.conversation_history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
            c.save_session("final_answer")
            c2, _ = self._session_commander(tmp)
            self.check("session final answer: reload succeeds", c2.load_session(notify=False))
            roles = [m.get("role") for m in c2.conversation_history]
            self.check("session final answer: all three messages restored",
                       roles == ["system", "user", "assistant"], str(roles))
            self.check("session final answer: last message intact",
                       c2.conversation_history[-1].get("content") == "2+2 equals 4.",
                       repr(c2.conversation_history[-1])[:200])

    def test_session_final_answer_with_marker_saved(self):
        """Regression: a short real final answer ending with the completion
        marker must be stored in the session file. The early-break for the
        synthesized terminal response used to swallow it by content shape."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".aicsession")
            c, _fake = self.make_fake_commander(
                script=[[
                    fake_openai.content_chunk("All done. TASKCOMPLETE"),
                    fake_openai.usage_chunk(10, 5),
                ]],
                session_file=path,
            )
            c.run("finish it")
            data = json.load(open(path))
            saved = data["messages"]
            last = saved[-1] if saved else {}
            self.check("marker answer: assistant turn stored",
                       last.get("role") == "assistant", repr(saved)[:250])
            self.check("marker answer: content intact",
                       "TASKCOMPLETE" in (last.get("content") or ""),
                       repr(last)[:250])
            self.check("marker answer: internal tag not persisted",
                       "synthesized_terminal" not in last, repr(last)[:250])

    def test_session_picks_up_external_edits(self):
        """The messages array is re-read from disk, so hand edits take effect."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            c.conversation_history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "original"},
            ]
            c.save_session("unit_test")

            # Edit the session the way a user would: open, change messages, save.
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["messages"][1]["content"] = "edited by hand"
            data["messages"].append({"role": "user", "content": "injected turn"})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            c2, _ = self._session_commander(tmp)
            self.check("session edits: reload succeeds", c2.load_session(notify=False))
            self.check("session edits: edited content loaded",
                       c2.conversation_history[1]["content"] == "edited by hand",
                       repr(c2.conversation_history)[:300])
            self.check("session edits: appended turn loaded",
                       any(m.get("content") == "injected turn" for m in c2.conversation_history),
                       repr(c2.conversation_history)[:300])
            self.check("session edits: history ready for the API",
                       c2._validate_messages(c2.conversation_history) != [],
                       "validation returned empty")

            # Deleting a turn on disk is honoured too (fewer messages after reload).
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            del data["messages"][1]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            c3, _ = self._session_commander(tmp)
            self.check("session edits: deleted turn stays deleted",
                       c3.load_session(notify=False)
                       and not any(m.get("content") == "edited by hand"
                                   for m in c3.conversation_history),
                       repr(c3.conversation_history)[:300])

    def test_session_bare_array_and_repair(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".aicsession")
            # A bare messages array and a tool message with no matching call
            # are repaired on load. The trailing assistant message is KEPT: it
            # is the previous run's final answer and must survive a resume
            # (API safety for a prompt ending on an assistant turn is handled
            # at call time, not by deleting history).
            with open(path, "w", encoding="utf-8") as f:
                json.dump([
                    {"role": "user", "content": "hello"},
                    {"role": "tool", "tool_call_id": "nope", "content": "orphan"},
                    {"role": "assistant", "content": "dangling"},
                ], f)
            c, _ = self._session_commander(tmp)
            self.check("session repair: bare array loaded", c.load_session(notify=False))
            roles = [m.get("role") for m in c.conversation_history]
            self.check("session repair: system prompt prepended", roles[0] == "system", str(roles))
            self.check("session repair: orphan tool message dropped", "tool" not in roles, str(roles))
            self.check("session repair: trailing assistant answer kept",
                       roles[-1] == "assistant"
                       and c.conversation_history[-1].get("content") == "dangling", str(roles))

    def test_session_closes_unanswered_tool_call(self):
        """A crash right after an assistant tool call still reloads cleanly."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            c.conversation_history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "run it"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_x", "type": "function",
                     "function": {"name": "execute_bash",
                                  "arguments": "{\"command\": \"rm -rf /\"}"}}]},
            ]
            c.save_session("assistant_message")

            c2, _ = self._session_commander(tmp)
            self.check("session recovery: reload succeeds", c2.load_session(notify=False))
            last = c2.conversation_history[-1]
            self.check("session recovery: tool result synthesized",
                       last.get("role") == "tool" and last.get("tool_call_id") == "call_x",
                       repr(last)[:200])
            self.check("session recovery: notice says the command was not re-run",
                       "NOT re-executed" in (last.get("content") or ""), repr(last)[:200])
            self.check("session recovery: history still ends ready for the API",
                       c2._validate_messages(c2.conversation_history) != [], "validation returned empty")

    def test_session_corrupt_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".aicsession")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"messages": [ {"role": "user" ')  # truncated write
            c, _ = self._session_commander(tmp)
            self.check("session corrupt: load reports failure", not c.load_session(notify=False))
            self.check("session corrupt: original empty (no partial parse)",
                       c.conversation_history == [])
            backups = [f for f in os.listdir(tmp) if ".corrupt." in f]
            self.check("session corrupt: bad file preserved as backup", len(backups) == 1,
                       os.listdir(tmp))
            # A fresh save still works afterwards.
            c.conversation_history = [{"role": "system", "content": "s"},
                                      {"role": "user", "content": "u"}]
            self.check("session corrupt: usable after recovery", c.save_session("unit_test"))

    def test_session_missing_file_is_quiet(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            self.check("session missing: load returns False", not c.load_session(notify=False))
            self.check("session missing: nothing created", not os.path.exists(path))

    def test_session_disabled_without_path(self):
        c = self.make_commander()  # session_file defaults to ""
        self.check("session disabled: save is a no-op", not c.save_session("unit_test"))
        self.check("session disabled: load is a no-op", not c.load_session(notify=False))

    def test_session_written_during_run(self):
        """run() mirrors history to disk after each step, tool result included."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".aicsession")
            c, _fake = self.make_fake_commander(
                script=[
                    [
                        fake_openai.tool_call_chunk(0, id="call_1", name="execute_bash",
                                                    arguments='{"command": "echo seeded"}'),
                        fake_openai.usage_chunk(20, 5),
                    ],
                    [
                        fake_openai.content_chunk("all done TASKCOMPLETE"),
                        fake_openai.usage_chunk(30, 4),
                    ],
                ],
                session_file=path,
            )
            c.run("do the thing")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved = data.get("messages")
            self.check("session run: file matches final history exactly",
                       saved == c.conversation_history,
                       f"saved={len(saved) if saved else 0} live={len(c.conversation_history)}")
            self.check("session run: tool output captured",
                       any("seeded" in (m.get("content") or "") for m in saved if isinstance(m, dict)),
                       repr(saved)[:300])
            self.check("session run: step recorded", data.get("step_count") == c.step_count,
                       f"{data.get('step_count')} vs {c.step_count}")
            self.check("session run: usage accumulated",
                       data.get("usage", {}).get("input_tokens") == 50, repr(data.get("usage")))

            # Simulated crash: the command dies mid-flight (SystemExit is not
            # caught by the loop's error handling), so the file must end on the
            # unanswered tool call rather than losing the turn.
            path2 = os.path.join(tmp, "crash.aicsession")
            c2, _fake2 = self.make_fake_commander(
                script=[[
                    fake_openai.tool_call_chunk(0, id="call_2", name="execute_bash",
                                                arguments='{"command": "sleep 999"}'),
                    fake_openai.usage_chunk(20, 5),
                ]],
                session_file=path2,
            )

            def _die(command):
                # Hard kill while the command runs: no tool result is recorded.
                raise SystemExit(9)

            c2.execute_bash_command = _die
            try:
                c2.run("hang forever")
            except SystemExit:
                pass
            with open(path2, "r", encoding="utf-8") as f:
                crashed = json.load(f)["messages"]
            self.check("session crash: assistant tool call preserved",
                       crashed[-1].get("role") == "assistant"
                       and bool(crashed[-1].get("tool_calls")),
                       repr(crashed[-1])[:200])
            c3, _ = self._session_commander(tmp, name="crash.aicsession")
            self.check("session crash: reload succeeds", c3.load_session(notify=False))
            self.check("session crash: full pre-crash history restored",
                       len(c3.conversation_history) == len(crashed) + 1,
                       f"{len(c3.conversation_history)} vs {len(crashed)}")

    def test_session_discard(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c, path = self._session_commander(tmp)
            c.conversation_history = [{"role": "system", "content": "s"},
                                      {"role": "user", "content": "u"}]
            c.save_session("unit_test")
            c.discard_session()
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.check("session discard: file emptied", data.get("messages") == [], repr(data)[:200])
            c2, _ = self._session_commander(tmp)
            self.check("session discard: nothing to restore", not c2.load_session(notify=False))

    def test_session_instance_locking(self):
        """A second live instance in the same directory shifts to .aicsession2."""
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            first, n1 = aic.acquire_session_file(os.path.join(tmp, ".aicsession"))
            self.check("session lock: first instance takes the default name",
                       first == os.path.join(tmp, ".aicsession") and n1 == 1, f"{first} n={n1}")
            second, n2 = aic.acquire_session_file(os.path.join(tmp, ".aicsession"))
            self.check("session lock: second instance shifts to .aicsession2",
                       second == os.path.join(tmp, ".aicsession2") and n2 == 2, f"{second} n={n2}")
            third, n3 = aic.acquire_session_file(os.path.join(tmp, ".aicsession"))
            self.check("session lock: third instance shifts to .aicsession3",
                       third == os.path.join(tmp, ".aicsession3") and n3 == 3, f"{third} n={n3}")
            self.check("session lock: lock files written",
                       os.path.exists(first + ".lock"), os.listdir(tmp))

            # Cross-process: a holder in another live process pushes the next
            # instance onto the numbered fallback, and the name is free again
            # once that process is gone (the kernel drops the flock), so a
            # crashed run can never block the default name forever.
            with tempfile.TemporaryDirectory() as tmp2:
                base = os.path.join(tmp2, ".aicsession")
                holder = subprocess.Popen(
                    [sys.executable, "-c",
                     "import sys, time; sys.path.insert(0, %r); import aic;"
                     "p, _ = aic.acquire_session_file(%r);"
                     "print(p, flush=True); time.sleep(5)"
                     % (os.path.dirname(aic.__file__), base)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    held = holder.stdout.readline().strip()
                    self.check("session lock: separate process holds the default name",
                               held == base, f"holder said {held!r}")
                    blocked, nb = aic.acquire_session_file(base)
                    self.check("session lock: foreign holder shifts the next instance",
                               blocked == base + "2" and nb == 2, f"{blocked} n={nb}")
                finally:
                    holder.kill()
                    holder.wait(timeout=30)
                back, nb2 = aic.acquire_session_file(base)
                self.check("session lock: name reusable after holder exits",
                           back == base and nb2 == 1, f"{back} n={nb2}")

            # An explicit --session path bypasses the default-name dance.
            custom, nc = aic.acquire_session_file(os.path.join(tmp, "my-session.json"))
            self.check("session lock: explicit path honoured",
                       custom == os.path.join(tmp, "my-session.json") and nc == 1, custom)

    # ------------------------------------------------------------------ #
    # 5. Live integration tests (network)                                #
    # ------------------------------------------------------------------ #

    def test_live_call_llm(self):
        c = self.make_commander()
        msgs = [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Reply with the single word: pong"},
        ]
        try:
            resp = c.call_llm_api(msgs, use_tools=False)
        except Exception as e:  # noqa: BLE001
            self.check("live call_llm_api succeeded", False, f"exception: {e}")
            return
        msg = resp.get("choices", [{}])[0].get("message", {})
        self.check("live call_llm_api returned message", bool(msg), repr(resp)[:200])
        content = msg.get("content")
        self.check(
            "live call_llm_api returned content",
            isinstance(content, str) and len(content) > 0,
            repr(content)[:200],
        )

    def test_run_loop(self):
        c = self.make_commander(max_steps=8)
        try:
            c.run("Reply with the single word: pong, then stop.")
        except Exception as e:  # noqa: BLE001
            self.check("run() agent loop completed", False, f"exception: {e}")
            return
        self.check("run() agent loop completed", True)


def main():
    parser = argparse.ArgumentParser(
        description="Test the truncation feature and core functionality of aic.py"
    )
    # Credentials are only needed for the live (network) tests. The offline
    # unit tests construct an AICommander but never connect, so placeholders
    # are fine unless live tests are requested.
    parser.add_argument("--api-base", default="http://localhost:8001/v1",
                        help="OpenAI-compatible API base URL (only needed for live tests)")
    parser.add_argument("--api-key", default="",
                        help="API key (only needed for live tests)")
    parser.add_argument("--model", default="",
                        help="Model name (only needed for live tests)")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip tests that hit the network (offline unit tests always run)")
    parser.add_argument("--run-loop", action="store_true",
                        help="Also run the full agent loop (requires network)")
    args = parser.parse_args()

    if not args.skip_live and not (args.api_key and args.model):
        print("[ERROR] Live tests require --api-key and --model. "
              "Provide them, or pass --skip-live to run offline unit tests only.",
              file=sys.stderr)
        sys.exit(2)

    runner = TestRunner(args)

    print("=== Offline unit tests ===")
    runner.test_output_truncation()
    runner.test_small_output()
    runner.test_history_condense()
    runner.test_history_drop()
    runner.test_history_noop()
    runner.test_history_truncate_leaves_headroom()
    runner.test_truncate_does_not_over_truncate_tool_outputs()
    runner.test_default_algorithm_is_context_compressor_llm()
    runner.test_compress_target_configurable()
    runner.test_compress_target_default()
    runner.test_explicit_algorithm_dispatch()
    runner.test_context_compressor_llm()
    runner.test_prompt_budget_reserves_output()
    runner.test_history_truncate_respects_output_reservation()
    runner.test_estimate_tokens()
    runner.test_validate_messages()
    runner.test_validate_terminal()
    runner.test_process_response()
    runner.test_process_response_malformed()
    runner.test_handle_function_call()
    runner.test_token_tracker_reconcile()
    runner.test_token_tracker_aborted_call()
    runner.test_token_tracker_reset()
    runner.test_call_llm_api_emits_estimate_then_exact()
    runner.test_call_llm_api_streams_thinking_and_tool_calls()
    runner.test_context_counter_after_full_loop()
    runner.test_session_save_writes_json()
    runner.test_session_round_trip()
    runner.test_session_final_answer_survives_resume()
    runner.test_session_final_answer_with_marker_saved()
    runner.test_session_picks_up_external_edits()
    runner.test_session_bare_array_and_repair()
    runner.test_session_closes_unanswered_tool_call()
    runner.test_session_corrupt_file()
    runner.test_session_missing_file_is_quiet()
    runner.test_session_disabled_without_path()
    runner.test_session_written_during_run()
    runner.test_session_discard()
    runner.test_session_instance_locking()

    if not args.skip_live:
        print("\n=== Live integration tests (network) ===")
        runner.test_live_call_llm()
        if args.run_loop:
            runner.test_run_loop()
    else:
        print("\n=== Skipping live tests (--skip-live) ===")

    passed = sum(1 for _, p, _ in runner.results if p)
    failed = len(runner.results) - passed
    print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
    for name, p, detail in runner.results:
        if not p:
            print(f"  FAILED: {name}  -- {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
