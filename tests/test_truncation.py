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
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
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
            # Matches the production default (80000). The prompt budget is
            # max_prompt_len - max_tokens, and max_tokens defaults to 32000,
            # so this leaves ~48k of real headroom -- the same relationship
            # the old 20000/8000 pair had. Tests that exercise compression
            # override both values explicitly.
            max_prompt_len=80000,
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
        output, code = c.execute_bash_command("python3 -c \"print('a'*500 + 'z'*500)\"")
        self.check(
            "output truncation: sentinel appended",
            output.endswith(c.OUTPUT_TRUNCATION_SENTINEL),
            f"output did not end with sentinel; len={len(output)}",
        )
        head, marker, tail = (output.split("\n") + ["", "", ""])[:3]
        self.check("output truncation: keeps 30% head and 70% tail", head == "a" * 30 and tail == "z" * 69,  # 70 bytes with the newline
                   repr(output))
        path = re.search(r"is in (\S+) \.\.\.\]", marker)
        self.check("output truncation: the marker names the saved full output",
                   path and open(path.group(1)).read() == "a" * 500 + "z" * 500 + "\n", repr(marker))
        out, _ = c.execute_bash_command("seq 1 100")
        lines = out.splitlines()
        self.check("output truncation: cut at whole lines", lines[0] == "1" and lines[-2] == "100"
                   and all(ln.isdigit() for ln in lines[:-1] if not ln.startswith("[...")), repr(out))
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
            {"role": "tool", "tool_call_id": "c2", "content": "result"},  # answers c2
            {"role": "user", "content": "go on"},           # keep the prompt non-terminal
        ]
        out = c._validate_messages(msgs)
        self.check("validate: None content replaced", out[1]["content"] == "", repr(out))
        self.check(
            "validate: invalid roles dropped",
            all(m["role"] in ("system", "user", "assistant", "tool", "function", "developer") for m in out),
            repr(out),
        )
        self.check("validate: tool without id dropped",
                   all(m.get("tool_call_id") for m in out if m["role"] == "tool"), repr(out))
        self.check("validate: function without name dropped", all(m["role"] != "function" for m in out), repr(out))
        # The consecutive-assistant collapse only fires for a trailing pair;
        # here a tool result follows, so both assistants survive, each with
        # its pairing state intact: a1's unanswered call was stripped by the
        # pairing sanitizer, a2's answered call kept.
        assistants = [m for m in out if m["role"] == "assistant"]
        self.check("validate: unanswered tool_calls stripped",
                   all(not m.get("tool_calls") for m in assistants if m["content"] == "a1"), repr(out))
        self.check("validate: answered tool_calls kept",
                   any(m.get("tool_calls") == [{"id": "c2"}] for m in assistants), repr(out))
        self.check("validate: answered tool result kept",
                   any(m.get("role") == "tool" and m.get("tool_call_id") == "c2" for m in out), repr(out))
        # A trailing consecutive pair IS collapsed: the older assistant goes.
        # The survivor is text-only, so the prompt would end on a plain
        # assistant turn -- the terminal state, reported as an empty list.
        msgs2 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "old"},
            {"role": "assistant", "content": "new"},
        ]
        out2 = c._validate_messages(msgs2)
        self.check("validate: trailing consecutive assistant collapsed",
                   out2 == [], repr(out2))
        # Same shape but ending on an answered tool result: the collapse does
        # not fire (it only applies to a trailing pair), and the intact
        # assistant/tool pair must survive untouched.
        msgs3 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "old"},
            {"role": "assistant", "content": "new", "tool_calls": [{"id": "c9"}]},
            {"role": "tool", "tool_call_id": "c9", "content": "result"},
        ]
        out3 = c._validate_messages(msgs3)
        self.check("validate: answered pair kept before tool result",
                   any(m.get("role") == "tool" and m.get("tool_call_id") == "c9" for m in out3), repr(out3))
        self.check("validate: answered tool_calls kept",
                   any(m.get("tool_calls") == [{"id": "c9"}] for m in out3), repr(out3))

    def test_sanitize_tool_pairing(self):
        c = self.make_commander()
        # Orphan tool result (assistant evicted by compression) is dropped.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "orphan"},
            {"role": "user", "content": "go on"},
        ]
        out = c._sanitize_tool_pairing(msgs)
        self.check("pairing: orphan tool dropped", all(m["role"] != "tool" for m in out), repr(out))
        # Intact pair survives untouched.
        msgs = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_2", "type": "function",
                             "function": {"name": "execute_bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "result"},
        ]
        out = c._sanitize_tool_pairing(msgs)
        self.check("pairing: intact pair kept",
                   [m["role"] for m in out] == ["assistant", "tool"], repr(out))
        # Assistant whose result was lost gets tool_calls stripped.
        msgs = [
            {"role": "assistant", "content": "doing stuff",
             "tool_calls": [{"id": "call_3", "type": "function",
                             "function": {"name": "execute_bash", "arguments": "{}"}}]},
        ]
        out = c._sanitize_tool_pairing(msgs)
        self.check("pairing: unanswered tool_calls stripped",
                   all(not m.get("tool_calls") for m in out), repr(out))
        # Input list is not mutated.
        snapshot = [dict(m) for m in msgs]
        c._sanitize_tool_pairing(msgs)
        self.check("pairing: input not mutated", msgs == snapshot, repr(msgs))

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

    def test_tui_tabs(self):
        """Agent tabs: independent agents, panes, approvals and session files per
        tab; background runs; closing a tab; saved tabs reopen on restart."""
        if not aic._TEXTUAL_AVAILABLE:
            self.check("tabs: skipped (textual missing)", True)
            return
        import asyncio
        import tempfile
        import time
        tmp = tempfile.mkdtemp()
        base = os.path.join(tmp, ".aicsession")

        def make_args():
            session_file, number = aic.acquire_session_file(base)
            return argparse.Namespace(
                api_base="http://localhost:1/v1", model="test-model", api_key="test-key", auto_approve=False,
                no_thinking=False, timeout=30, max_prompt_len=80000, max_output_bytes=10240, max_steps=10,
                debug=False, compress_alg="truncate", compress_target=0.4, fast=False, reasoning_effort=None,
                request=[], session=base, session_file=session_file, instance_number=number,
                session_msg="[SESSION] test", sandbox_enabled=False, sandbox_msg="[SANDBOX] off",
                sandbox_write_dir=os.getcwd(), loop="", max_iterations=10, verify="", confirm_reads=True)

        def release_all():  # what the process exit does to the locks
            for path in [p for p in aic._SESSION_LOCK_FDS if p.startswith(tmp)]:
                aic.release_session_lock(path)

        def answer(text):
            return [fake_openai.content_chunk(text + " TASKCOMPLETE"), fake_openai.usage_chunk(10, 3)]

        def command(cid, cmd):
            return [fake_openai.tool_call_chunk(0, id=cid, name="execute_bash",
                                                arguments=json.dumps({"command": cmd})),
                    fake_openai.usage_chunk(10, 3)]

        def fake(tab, script):
            tab.agent.client = fake_openai.FakeOpenAIClient(script)
            tab.agent.task_summary_enabled = False

        async def until(pilot, condition, timeout=10):
            end = time.monotonic() + timeout
            while not condition() and time.monotonic() < end:
                await pilot.pause(0.05)
            return condition()

        def bar(app):
            widget = app.query_one("#agent-tabs")
            return widget.build(widget.size.width or 80).plain

        async def first_run():
            app = aic.CommanderApp(make_args(), aic.PromptHistory(None))
            r = {}
            async with app.run_test(size=(140, 36)) as pilot:
                await pilot.pause(0.2)
                tab1 = app.tab
                r["one tab at start"] = len(app.tabs) == 1 and tab1.session_file == base
                fake(tab1, [answer("first-answer")])
                await pilot.press(*"hi", "enter")
                await until(pilot, lambda: not tab1.busy())
                await pilot.press("ctrl+t")
                await until(pilot, lambda: len(app.tabs) == 2)
                await pilot.pause(0.1)
                tab2 = app.tab
                r["ctrl+t opens and selects a tab"] = tab2 is not tab1 and tab2.agent is not tab1.agent
                r["new tab: own session file and lock"] = (tab2.session_file == base + "2"
                                                          and tab2.session_file in aic._SESSION_LOCK_FDS)
                r["new tab: own panes"] = (tab2.agent_log.display and not tab1.agent_log.display
                                           and "first-answer" not in tab2.agent_log.plain_text)
                fake(tab2, [command("c1", "echo tab2-output"), answer("second-done")])
                await pilot.press(*"run it", "enter")
                r["approval shown for the current tab"] = await until(
                    pilot, lambda: type(app.screen).__name__ == "ApprovalScreen")
                await pilot.press("ctrl+g")
                r["ctrl+g opens the tab switcher"] = await until(
                    pilot, lambda: type(app.screen).__name__ == "TabSwitcher", 3)
                await pilot.press("1")
                await pilot.pause(0.2)
                r["ctrl+g 1 switches back, approval stays pending"] = (
                    app.tab is tab1 and type(app.screen).__name__ not in ("ApprovalScreen", "TabSwitcher")
                    and tab2.pending_approval == "echo tab2-output" and "◆" in bar(app))
                await pilot.press("ctrl+g", "down", "enter")
                r["the pending approval returns with its tab"] = await until(
                    pilot, lambda: type(app.screen).__name__ == "ApprovalScreen" and app.tab is tab2)
                await pilot.press("ctrl+g", "1")  # select tab 1 from a switcher opened over the dialog
                await pilot.pause(0.3)
                r["switching from over the dialog leaves no stale dialog"] = (
                    app.tab is tab1 and len(app.screen_stack) == 1 and tab2.pending_approval is not None)
                await pilot.press("ctrl+g", "2")
                await until(pilot, lambda: type(app.screen).__name__ == "ApprovalScreen")
                await pilot.pause(0.5)
                await pilot.press("y")
                await until(pilot, lambda: not tab2.busy())
                await pilot.pause(0.1)
                r["command output lands in its own console"] = (
                    "tab2-output" in tab2.console_log.plain_text and "tab2-output" not in tab1.console_log.plain_text)
                await pilot.press("f2")
                r["auto-approve is per tab"] = tab2.auto_approve and not tab1.auto_approve
                fake(tab2, [command("c2", "sleep 0.5; echo bg-output"), answer("bg-done")])
                await pilot.press(*"again", "enter")
                await pilot.press("ctrl+g", "ctrl+g")
                await pilot.pause(0.1)
                r["ctrl+g twice selects the next tab"] = app.tab is tab1
                await until(pilot, lambda: not tab2.busy())
                await pilot.pause(0.1)
                r["a background tab keeps working"] = ("bg-output" in tab2.console_log.plain_text
                                                       and "bg-done" in tab2.agent_log.plain_text
                                                       and "bg-output" not in tab1.console_log.plain_text)
                r["finished background tab is marked"] = tab2.unseen and "●" in bar(app)
                await pilot.press("f3")
                await until(pilot, lambda: tab1.shell._pid is not None, 5)
                await pilot.press("ctrl+g", "2")  # from inside the shell
                await pilot.pause(0.1)
                r["each tab remembers its right pane"] = (app.tab is tab2 and tab2.pane == "console"
                                                          and tab2.console_log.display and not tab1.shell.display)
                await pilot.press("f3")
                shells = await until(pilot, lambda: tab2.shell._pid is not None, 5)
                r["each tab has its own shell"] = shells and tab2.shell._pid != tab1.shell._pid
                r["both sessions saved"] = all(aic.read_session_file(p)[0].get("messages")
                                               for p in (base, base + "2"))
                # A run ending under another dialog: the help screen stays, the
                # approval dialog goes once it is uncovered.
                fake(tab2, [command("c3", "echo never-run"), answer("x")])
                tab2.auto_approve = False
                await pilot.click("#prompt-input")
                await pilot.press(*"needs approval", "enter")
                asked = await until(pilot, lambda: type(app.screen).__name__ == "ApprovalScreen")
                await pilot.press("f1")
                await pilot.pause(0.2)
                await pilot.press("f9")
                await until(pilot, lambda: not tab2.busy())
                await pilot.pause(0.2)
                help_kept = type(app.screen).__name__ == "HelpScreen"
                await pilot.press("escape")
                await pilot.pause(0.3)
                r["a run ending under the help screen closes its approval dialog"] = (
                    asked and help_kept and len(app.screen_stack) == 1 and app._approval_screen is None)
                # A steering message the agent never read comes back to the prompt.
                tab2.agent.suggestion_queue.put("late steering")
                done = threading.Thread(target=lambda: None)
                done.start()
                done.join()
                tab2.thread = done
                r["an unread steering message returns to the prompt"] = await until(
                    pilot, lambda: app.prompt.text == "late steering", 3)
                app.prompt.clear()
                shell_pid = tab2.shell._pid
                await pilot.click("#prompt-input")
                await pilot.press(*"/closetab", "enter")
                await pilot.pause(0.1)
                r["closing a tab with a conversation asks twice"] = len(app.tabs) == 2
                await pilot.press(*"/closetab", "enter")
                await until(pilot, lambda: len(app.tabs) == 1)
                await pilot.pause(0.1)
                r["closed tab: session dropped, lock released"] = (
                    not os.path.exists(base + "2") and base + "2" not in aic._SESSION_LOCK_FDS)
                r["closed tab: its shell exited and was reaped"] = tab2.shell._pid is None and await until(
                    pilot, lambda: not os.path.exists(f"/proc/{shell_pid}"), 5)
                r["closed tab: the other one is shown"] = app.tab is tab1 and tab1.agent_log.display
                await pilot.press("ctrl+t")
                await until(pilot, lambda: len(app.tabs) == 2)
                await pilot.pause(0.1)
                tab3 = app.tab
                r["a new tab reuses the free session file"] = tab3.session_file == base + "2"
                fake(tab3, [answer("third-answer")])
                await pilot.press(*"hello", "enter")
                await until(pilot, lambda: not tab3.busy())
                await pilot.press("f10")
                await pilot.pause(0.3)
            r["quit"] = not app.is_running
            return r

        async def second_run():
            app = aic.CommanderApp(make_args(), aic.PromptHistory(None))
            r = {}
            async with app.run_test(size=(140, 36)) as pilot:
                await pilot.pause(0.3)
                r["restart reopens the saved tabs"] = (len(app.tabs) == 2 and app.tab is app.tabs[0]
                                                       and app.tabs[1].session_file == base + "2")
                if len(app.tabs) == 2:
                    r["reopened tab shows its conversation"] = "third-answer" in app.tabs[1].agent_log.plain_text
                    r["first tab restored too"] = "first-answer" in app.tabs[0].agent_log.plain_text
                await pilot.press("f10")
                await pilot.pause(0.3)
            return r

        native = aic.clipboard_write_native, aic.clipboard_read_native
        aic.clipboard_write_native, aic.clipboard_read_native = (lambda text: False), (lambda: None)
        try:
            results = asyncio.run(first_run())
            release_all()
            results.update(asyncio.run(second_run()))
        finally:
            release_all()
            aic.clipboard_write_native, aic.clipboard_read_native = native
            shutil.rmtree(tmp, ignore_errors=True)
        for name, ok in results.items():
            self.check(f"tabs: {name}", ok)

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

    def test_session_list_content_survives_reload(self):
        """Multi-part (list) content must not abort the session restore.

        _repair_loaded_history used to call .strip() on message content, which
        raised AttributeError for list-of-parts content and aborted the whole
        restore -- the agent panel then came up empty (or partial) instead of
        showing the restored conversation.
        """
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".aicsession")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": "part one"},
                            {"type": "text", "text": "part two"},
                        ]},
                        {"role": "user", "content": [
                            {"type": "text", "text": "second prompt"},
                        ]},
                        {"role": "assistant", "content": "final answer"},
                    ],
                    "task_summary": "list content",
                }, f)
            c, _ = self._session_commander(tmp)
            self.check("session list-content: reload succeeds",
                       c.load_session(notify=False))
            roles = [m.get("role") for m in c.conversation_history]
            self.check("session list-content: all messages kept",
                       roles == ["system", "user", "assistant", "user", "assistant"],
                       str(roles))
            self.check("session list-content: assistant list content preserved",
                       isinstance(c.conversation_history[2].get("content"), list)
                       and c.conversation_history[2]["content"][0]["text"] == "part one",
                       repr(c.conversation_history[2])[:200])
            self.check("session list-content: user list content preserved",
                       c.conversation_history[3]["content"][0]["text"] == "second prompt",
                       repr(c.conversation_history[3])[:200])

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
    # 4c. Regression tests for agent-loop fixes                          #
    # ------------------------------------------------------------------ #

    def test_validate_merges_consecutive_assistants(self):
        # Mid-list consecutive assistants are merged, not dropped: both texts
        # and every answered tool call survive.
        c = self.make_commander()
        out = c._validate_messages([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "r"},
        ])
        merged = [m for m in out if m["role"] == "assistant"]
        self.check("merge: one assistant turn", len(merged) == 1, repr(out))
        self.check("merge: both texts kept", merged and "first" in merged[0]["content"]
                   and "second" in merged[0]["content"], repr(merged))
        self.check("merge: tool call kept", merged and merged[0].get("tool_calls") == [{"id": "t1"}], repr(merged))
        # An assistant turn left empty by the pairing repair is dropped.
        out = c._validate_messages([
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "lost"}]},
            {"role": "user", "content": "again"},
        ])
        self.check("validate: emptied assistant turn dropped",
                   [m["role"] for m in out] == ["system", "user", "user"], repr(out))

    def test_command_output_cleaning(self):
        clean = aic._clean_command_output
        self.check("clean: ANSI colors stripped", clean("a\x1b[31mb\x1b[0m\x1b[?25l") == "ab")
        self.check("clean: progress bar keeps final state",
                   clean("10%\r50%\r100%\ndone\n") == "100%\ndone\n", repr(clean("10%\r50%\r100%\ndone\n")))
        self.check("clean: CRLF normalized", clean("x\r\ny\r\n") == "x\ny\n")
        self.check("clean: overstrike removed", clean("b\x08bo\x08ol\x08ld") == "bold")
        c = self.make_commander()
        out, code = c.execute_bash_command("printf 'a\\033[32mgreen\\033[0m\\n'; printf '1\\r2\\r3\\n'")
        self.check("execute: output is model-friendly", out == "agreen\n3\n", repr(out))

    def test_output_truncation_multibyte(self):
        c = self.make_commander(max_output_bytes=100)
        out, _ = c.execute_bash_command("python3 -c \"print('é' * 400)\"")
        lines = out.split("\n")
        body = lines[0] + lines[2]
        self.check("truncation: capped in bytes, not characters", len(body.encode("utf-8")) <= 100
                   and "\ufffd" not in body and lines[1].startswith("[..."), f"{len(body.encode('utf-8'))} bytes")

    def test_command_timeout_keeps_partial_output(self):
        c = self.make_commander(command_timeout=1)
        try:
            c.execute_bash_command("echo before-timeout; sleep 30")
            self.check("timeout: raises CommandTimeoutError", False)
        except aic.CommandTimeoutError as exc:
            self.check("timeout: partial output kept", "before-timeout" in exc.output, repr(exc.output))

    def test_process_response_argument_shapes(self):
        c = self.make_commander()

        def first_command(arguments):
            resp = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "execute_bash", "arguments": arguments}}]}}]}
            return c.process_llm_response(resp)[1][0]["command"]
        self.check("args: dict accepted", first_command({"command": "ls"}) == "ls")
        self.check("args: bare JSON string accepted", first_command('"pwd"') == "pwd")
        self.check("args: JSON list does not crash", first_command("[1, 2]") is None)

    def test_stream_tool_call_without_index(self):
        # Providers that omit "index" stream one call over several chunks.
        chunk = fake_openai.tool_call_chunk(0, id="call_x", name="execute_bash", arguments='{"comm')
        chunk.choices[0].delta.tool_calls[0].index = None
        rest = fake_openai.tool_call_chunk(0, arguments='and": "ls"}')
        rest.choices[0].delta.tool_calls[0].index = None
        rest.choices[0].delta.tool_calls[0].id = ""
        c, _ = self.make_fake_commander([[chunk, rest, fake_openai.usage_chunk(1, 1)]])
        calls = c.call_llm_api([{"role": "user", "content": "hi"}])["choices"][0]["message"]["tool_calls"] or []
        self.check("no index: one call assembled",
                   len(calls) == 1 and calls[0]["function"]["arguments"] == '{"command": "ls"}', repr(calls))

    def test_empty_responses_end_the_run(self):
        empty = [fake_openai.usage_chunk(1, 0)]
        c, fake = self.make_fake_commander([empty] * 5, max_steps=10)
        c.run("say something")
        self.check("empty: run stops after three empty turns", len(fake.calls) == 3, f"calls={len(fake.calls)}")
        self.check("empty: no empty assistant turn stored",
                   all(m.get("role") != "assistant" for m in c.conversation_history), repr(c.conversation_history))

    def test_stop_answers_remaining_tool_calls(self):
        script = [[fake_openai.tool_call_chunk(0, id="a", name="execute_bash", arguments='{"command": "true"}'),
                   fake_openai.tool_call_chunk(1, id="b", name="execute_bash", arguments='{"command": "true"}'),
                   fake_openai.usage_chunk(1, 1)]]
        c, _ = self.make_fake_commander(script)

        def stop_while_running(command):
            c.stop_event.set()  # the user stops the agent during the first command
            return "ok", 0
        c.execute_bash_command = stop_while_running
        c.run("two commands")
        results = {m["tool_call_id"]: m["content"] for m in c.conversation_history if m.get("role") == "tool"}
        self.check("stop: every tool call answered", set(results) == {"a", "b"}, repr(results))
        self.check("stop: unexecuted call says so", "Not executed" in results.get("b", ""), repr(results))

    def test_suggestion_follows_all_tool_results(self):
        script = [[fake_openai.tool_call_chunk(0, id="a", name="execute_bash", arguments='{"command": "rm x"}'),
                   fake_openai.tool_call_chunk(1, id="b", name="execute_bash", arguments='{"command": "true"}'),
                   fake_openai.usage_chunk(1, 1)],
                  [fake_openai.content_chunk("ok TASKCOMPLETE"), fake_openai.usage_chunk(1, 1)]]
        c, _ = self.make_fake_commander(script, auto_approve=False)
        answers = iter(["Use trash instead", "y"])
        c.sink.confirm = lambda command: next(answers)
        c.execute_bash_command = lambda command: ("ok", 0)
        c.run("clean up")
        roles = [m["role"] for m in c.conversation_history]
        i = roles.index("tool")
        self.check("suggestion: tool results stay adjacent", roles[i:i + 3] == ["tool", "tool", "user"], str(roles))
        self.check("suggestion: original case kept",
                   c.conversation_history[i + 2]["content"] == "I suggest you Use trash instead",
                   repr(c.conversation_history[i + 2]))

    def test_truncate_keeps_current_request(self):
        # In a long session the request that started the current run must
        # survive compression, not just the very first instruction.
        c = self.make_commander(max_prompt_len=200, model="gpt-test", compress_algorithm="truncate")
        c.conversation_history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "old task"}]
        c.conversation_history += [{"role": "assistant", "content": "x" * 300} for _ in range(3)]
        request = {"role": "user", "content": "the current task"}
        c.conversation_history.append(request)
        c.conversation_history += [{"role": "assistant", "content": "y" * 300}]
        c._current_request = request
        c._context_compress()
        self.check("truncate: current request kept",
                   any(m is request for m in c.conversation_history), repr(c.conversation_history))

    def test_ccllm_summary_is_folded_not_stacked(self):
        if not aic._CONTEXT_COMPRESSOR_LLM_AVAILABLE:
            self.check("summary folding: skipped (library missing)", True)
            return
        c = self.make_commander(max_prompt_len=300, model="gpt-test")
        seen = []

        def summarize(messages, previous):
            seen.append(previous)
            return f"summary-{len(seen)}"
        c._llm_summarize = summarize
        c.conversation_history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for round_ in range(2):
            c.conversation_history += [{"role": "assistant", "content": "z" * 600} for _ in range(4)]
            c._context_compress()
        system = c.conversation_history[0]["content"]
        self.check("summary folding: one summary block", system.count("[Prior conversation summary]") == 1, system)
        self.check("summary folding: previous summary handed over", seen[-1] == "summary-1", repr(seen))

    def test_summarizer_failures_fall_back(self):
        if not aic._CONTEXT_COMPRESSOR_LLM_AVAILABLE:
            self.check("summarizer fallback: skipped (library missing)", True)
            return

        def compress(summarize):
            c = self.make_commander(max_prompt_len=300, model="gpt-test")
            c._llm_summarize = summarize
            c.conversation_history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
            c.conversation_history += [{"role": "assistant", "content": f"step {i} " + "z" * 600} for i in range(4)]
            c._context_compress()
            return c

        def broken(messages, previous):
            raise RuntimeError("server down")
        c = compress(broken)
        self.check("summarizer fallback: a failed summary truncates instead",
                   "[Prior conversation summary]" not in c.conversation_history[0]["content"]
                   and any("falling back to 'truncate'" in p.get("text", "") for k, p in c.sink.events),
                   repr([p.get("text") for k, p in c.sink.events][-3:]))
        # A reply cut off inside the <analysis> scratchpad is not a summary.
        c2 = self.make_commander()
        c2.call_llm_api = lambda *a, **k: {"choices": [{"message": {"content": "<analysis>\\nstill thinking"}}]}
        try:
            c2._llm_summarize([{"role": "user", "content": "hi"}], None)
            self.check("summarizer fallback: cut-off analysis rejected", False)
        except RuntimeError:
            self.check("summarizer fallback: cut-off analysis rejected", True)

    def test_internal_calls_use_low_effort(self):
        done = [fake_openai.content_chunk("<summary>ok</summary>"), fake_openai.usage_chunk(1, 1)]
        c, fake = self.make_fake_commander([done])
        self.check("effort: summarizer returns the summary only",
                   c._llm_summarize([{"role": "user", "content": "x"}], None) == "ok")
        params = fake.calls[0]
        self.check("effort: summarizer asks for low reasoning effort and caps its output",
                   params.get("reasoning_effort") == "low" and params.get("max_tokens") == 8192
                   and params["extra_body"]["chat_template_kwargs"].get("reasoning_effort") == "low", repr(params))

        class Rejected(Exception):
            status_code = 400
        c2, fake2 = self.make_fake_commander([Rejected("Unrecognized request argument supplied: reasoning_effort"),
                                              done, done])
        c2._llm_summarize([{"role": "user", "content": "x"}], None)
        c2._llm_summarize([{"role": "user", "content": "x"}], None)
        self.check("effort: a rejected reasoning_effort is dropped for good",
                   len(fake2.calls) == 3 and "reasoning_effort" not in fake2.calls[1]
                   and "reasoning_effort" not in fake2.calls[2], f"calls={len(fake2.calls)}")

    def test_output_limit_cut_off(self):
        # A reply that hits max_tokens loses its half-written tool call (servers drop
        # it and may still say "tool_calls"): the model is told to split the work
        # instead of being retried blindly.
        cut = [fake_openai.thinking_chunk("planning a huge file..."), fake_openai.usage_chunk(10, 50)]
        done = [fake_openai.content_chunk("done TASKCOMPLETE"), fake_openai.usage_chunk(10, 3)]
        c, fake = self.make_fake_commander([cut, done], show_thinking=True)
        c.max_tokens = 50
        status = c.run("write a big file")
        logs = [p.get("text", "") for k, p in c.sink.events if k == "LOG"]
        note = [m for m in fake.calls[1]["messages"] if "output limit" in str(m.get("content"))]
        self.check("cut-off: reported as hitting the output limit, not as an empty reply",
                   any("output limit" in t for t in logs) and not any("Empty assistant" in t for t in logs), repr(logs))
        self.check("cut-off: the model is told to split the work", len(note) == 1 and "split" in note[0]["content"])
        self.check("cut-off: the run continues and completes", status == "complete" and len(fake.calls) == 2)

    def test_context_overflow_is_retried(self):
        class ContextError(Exception):
            status_code = 400
        script = [ContextError("This model's maximum context length is 8192 tokens"),
                  [fake_openai.content_chunk("done TASKCOMPLETE"), fake_openai.usage_chunk(1, 1)]]
        c, fake = self.make_fake_commander(script)
        c.stop_event.wait = lambda timeout=None: False  # a regression must not sleep through backoff
        c.run("hello")
        self.check("overflow: retried after compressing", len(fake.calls) == 2
                   and c.conversation_history[-1].get("content") == "done TASKCOMPLETE",
                   f"calls={len(fake.calls)}")
        self.check("overflow: compression ran before the retry",
                   any(k == "LOG" and "[COMPRESS]" in p.get("text", "") for k, p in c.sink.events))

    def test_transient_error_is_retried(self):
        script = [ConnectionError("connection reset"),
                  [fake_openai.content_chunk("pong"), fake_openai.usage_chunk(1, 1)]]
        c, fake = self.make_fake_commander(script)
        c.stop_event.wait = lambda timeout=None: False  # no real backoff sleep in tests
        resp = c.call_llm_api([{"role": "user", "content": "ping"}], use_tools=False)
        self.check("retry: transient error retried", resp["choices"][0]["message"]["content"] == "pong")

    def test_prompt_history_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hist")
            h = aic.PromptHistory(path)
            h.load()
            for item in ("first", "multi\nline \\ prompt", "multi\nline \\ prompt", "last"):
                h.add(item)
            h.close()
            h2 = aic.PromptHistory(path)
            h2.load()
            h2.close()
            self.check("history: round trip with newlines and dedupe",
                       h2.items == ["first", "multi\nline \\ prompt", "last"], repr(h2.items))

    def test_tui_log_wraps_and_copies(self):
        if not aic._TEXTUAL_AVAILABLE:
            self.check("tui log: skipped (textual missing)", True)
            return
        import asyncio
        spans = aic._wrap_spans("alpha beta gamma delta", 11)
        self.check("wrap: words kept whole", spans == [(0, 11), (11, 22)], repr(spans))
        self.check("wrap: long words folded", aic._wrap_spans("x" * 25, 10) == [(0, 10), (10, 20), (20, 25)])

        class LogApp(aic.App):
            def compose(self):
                yield aic.OutputLog(id="log")

        async def scenario():
            app = LogApp()
            async with app.run_test(size=(40, 10)) as pilot:
                log = app.query_one(aic.OutputLog)
                log.write("streamed ")
                log.write("text that is long enough to wrap across rows\nsecond line")
                log.write_line("third", "bold")
                await pilot.pause(0.1)
                rows = len(log._rows)
                text, _ = log.get_selection(aic.Selection(None, None))
                return rows, text
        rows, text = asyncio.run(scenario())
        self.check("tui log: soft wraps", rows > 3, f"rows={rows}")
        self.check("tui log: copy re-joins soft-wrapped rows",
                   text == "streamed text that is long enough to wrap across rows\nsecond line\nthird", repr(text))

    def test_preserved_thinking(self):
        # The model's reasoning goes back with its turns for the rest of the
        # run (models shown empty thinking blocks stop thinking), is cleared
        # by the next request, and is dropped when an endpoint refuses it.
        step1 = [fake_openai.thinking_chunk("I should list files."),
                 fake_openai.tool_call_chunk(0, id="r1", name="execute_bash", arguments='{"command": "true"}'),
                 fake_openai.usage_chunk(1, 1)]
        done = [fake_openai.content_chunk("done TASKCOMPLETE"), fake_openai.usage_chunk(1, 1)]
        c, fake = self.make_fake_commander([step1, done, done], show_thinking=True, persist_history=True)
        c.execute_bash_command = lambda command: ("ok", 0)
        c.run("first task")
        sent = [m for m in fake.calls[1]["messages"] if m.get("role") == "assistant"]
        self.check("thinking: reasoning sent back within the run",
                   sent and sent[0].get("reasoning_content") == "I should list files.", repr(sent))
        c.run("second task")
        self.check("thinking: a new request clears old reasoning",
                   not any("reasoning_content" in m for m in fake.calls[2]["messages"]))

        class Refused(Exception):
            status_code = 400
        c2, fake2 = self.make_fake_commander(
            [step1, Refused("reasoning_content is not allowed in input messages"), done], show_thinking=True)
        c2.execute_bash_command = lambda command: ("ok", 0)
        c2.run("task")
        self.check("thinking: refused reasoning is dropped and the call retried",
                   len(fake2.calls) == 3 and not c2._send_reasoning
                   and not any("reasoning_content" in m for m in fake2.calls[2]["messages"]),
                   f"calls={len(fake2.calls)}")

    def test_x11_clipboard(self):
        """Copy/paste through libX11, on a private selection so the user's
        clipboard is never touched. Skipped without an X display."""
        import ctypes.util
        import subprocess
        if not os.environ.get("DISPLAY") or not ctypes.util.find_library("X11"):
            self.check("x11 clipboard: skipped (no X display)", True)
            return
        selection = f"AIC_TEST_SELECTION_{os.getpid()}"
        text = "héllo — line 1\nline 2 界"
        self.check("x11 clipboard: copy owns the selection", aic._x11_copy(text, (selection,)))
        self.check("x11 clipboard: text reads back intact", aic._x11_paste(selection) == text)
        self.check("x11 clipboard: a new copy replaces the old one",
                   aic._x11_copy("second", (selection,)) and aic._x11_paste(selection) == "second")
        self.check("x11 clipboard: unowned selection reads as None",
                   aic._x11_paste(f"{selection}_NOBODY") is None)
        # the helper serving the private selection is no longer needed
        subprocess.run(["pkill", "-f", selection], capture_output=True)

    def test_tui_end_to_end(self):
        """Drive the real TUI headless (Textual pilot) against the fake client:
        streaming, approval dialog, console output, clipboard, keys."""
        if not aic._TEXTUAL_AVAILABLE:
            self.check("tui: skipped (textual missing)", True)
            return
        import asyncio
        import time
        args = argparse.Namespace(
            api_base="http://localhost:1/v1", model="test-model", api_key="test-key", auto_approve=False,
            no_thinking=False, timeout=30, max_prompt_len=80000, max_output_bytes=10240, max_steps=10,
            debug=False, compress_alg="truncate", compress_target=0.4, fast=False, reasoning_effort=None,
            request=[], session_file="", instance_number=1, session_msg="[SESSION] off",
            sandbox_enabled=False, sandbox_msg="[SANDBOX] off", sandbox_write_dir=os.getcwd(),
            loop="", max_iterations=10, verify="", confirm_reads=True)
        script = [
            [fake_openai.content_chunk("Hello there. TASK"), fake_openai.content_chunk("COMPLETE"),
             fake_openai.usage_chunk(10, 3)],
            [fake_openai.tool_call_chunk(0, id="c1", name="execute_bash",
                                         arguments='{"command": "echo tui-output"}'),
             fake_openai.usage_chunk(10, 3)],
            [fake_openai.content_chunk("Done. TASKCOMPLETE"), fake_openai.usage_chunk(10, 3)],
        ]

        async def until(pilot, condition, timeout=10):
            end = time.monotonic() + timeout
            while not condition() and time.monotonic() < end:
                await pilot.pause(0.05)
            return condition()

        async def scenario():
            app = aic.CommanderApp(args, aic.PromptHistory(None))
            results = {}
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.2)
                app.agent.client = fake_openai.FakeOpenAIClient(script)
                app.agent.task_summary_enabled = False
                await pilot.press(*"hi", "enter")
                await until(pilot, lambda: not app._agent_busy())
                await pilot.pause(0.2)
                log = app.agent_log.plain_text
                results["answer streamed"] = "Hello there." in log
                results["split marker hidden"] = "TASKCOMPLETE" not in log and "TASK" not in log.split("Hello there.")[1]
                await pilot.press(*"run it", "enter")
                results["approval dialog"] = await until(
                    pilot, lambda: type(app.screen).__name__ == "ApprovalScreen")
                await pilot.pause(0.5)  # keys are ignored right after the dialog opens
                await pilot.press("y")
                await until(pilot, lambda: not app._agent_busy())
                await pilot.pause(0.2)
                results["command output shown"] = "tui-output" in app.console_log.plain_text
                results["exit summary"] = "✓ exit 0" in app.console_log.plain_text
                app.copy_to_clipboard("pasted text")
                await pilot.click("#prompt-input", offset=(4, 1), button=3)
                results["right-click paste"] = await until(pilot, lambda: app.prompt.text == "pasted text", 3)
                app.prompt.clear()
                await pilot.press("f3")
                await pilot.pause(0.2)
                shell = app.tab.shell
                results["F3 shows the shell"] = app.tab.pane == "shell" and shell.display and not app.console_log.display
                if await until(pilot, lambda: shell._pid is not None, 5) and os.path.exists("/proc/self/cwd"):
                    # A cd in the shell must never move the agent's working directory.
                    before = os.getcwd()
                    await pilot.press("c", "d", "space", "slash", "enter")
                    moved = await until(pilot, lambda: os.readlink(f"/proc/{shell._pid}/cwd") == "/", 5)
                    await pilot.pause(1.0)
                    results["shell cd leaves the agent's directory alone"] = moved and os.getcwd() == before
                await pilot.press("f6")
                await pilot.press("ctrl+c")
                await pilot.pause(0.1)
                results["first ctrl+c keeps running"] = app.is_running
                await pilot.press("ctrl+c")
                await pilot.pause(0.3)
            results["second ctrl+c quits"] = not app.is_running
            return results

        # Never touch the desktop clipboard of whoever runs the tests.
        native = aic.clipboard_write_native, aic.clipboard_read_native
        aic.clipboard_write_native, aic.clipboard_read_native = (lambda text: False), (lambda: None)
        try:
            results = asyncio.run(scenario())
        finally:
            aic.clipboard_write_native, aic.clipboard_read_native = native
        for name, ok in results.items():
            self.check(f"tui: {name}", ok)

    # ------------------------------------------------------------------ #
    # 5. Live integration tests (network)                                #
    # ------------------------------------------------------------------ #

    def test_tui_todo(self):
        """The task list in the TUI: the agent panel's bottom border, one folded
        log line per update, /todo, history replay and /clear."""
        if not aic._TEXTUAL_AVAILABLE:
            self.check("tui todo: skipped (textual missing)", True)
            return
        import asyncio
        import time
        from textual.markup import to_content
        args = argparse.Namespace(
            api_base="http://localhost:1/v1", model="test-model", api_key="test-key", auto_approve=False,
            no_thinking=False, timeout=30, max_prompt_len=80000, max_output_bytes=10240, max_steps=10,
            debug=False, compress_alg="truncate", compress_target=0.4, fast=False, reasoning_effort=None,
            request=[], session_file="", instance_number=1, session_msg="[SESSION] off",
            sandbox_enabled=False, sandbox_msg="[SANDBOX] off", sandbox_write_dir=os.getcwd(),
            loop="", max_iterations=10, verify="", confirm_reads=False)
        items = [{"text": "one", "status": "done"}, {"text": "fix [b]two[/b]", "status": "in_progress"},
                 {"text": "three", "status": "pending"}]
        script = [self._call("todo", {"items": items}), self._done()]

        async def scenario():
            app = aic.CommanderApp(args, aic.PromptHistory(None))
            results = {}
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.2)
                app.agent.client = fake_openai.FakeOpenAIClient(script)
                app.agent.task_summary_enabled = False
                await pilot.press(*"plan", "enter")
                end = time.monotonic() + 10
                while app._agent_busy() and time.monotonic() < end:
                    await pilot.pause(0.05)
                await pilot.pause(0.2)
                subtitle = to_content(app.agent_log.border_subtitle or "").plain
                results["border shows progress and the current item"] = subtitle == "☑ 1/3 · ▸ fix [b]two[/b]"
                log = app.agent_log.plain_text
                results["one folded log line"] = "▸ todo · 1/3 done · 3 lines" in log and "exit 0" not in log
                await pilot.press(*"/todo", "enter")
                await pilot.pause(0.2)
                results["/todo prints the list"] = "[>] fix [b]two[/b]" in app.agent_log.plain_text
                app.agent_log.clear()
                app.console_log.clear()
                app.tab.render_history()
                results["replay folds it again"] = ("▸ todo · 1/3 done" in app.agent_log.plain_text
                                                    and "Task list updated" not in app.console_log.plain_text)
                await pilot.press(*"/clear", "enter")
                await pilot.pause(0.5)
                results["/clear empties the border"] = not app.agent_log.border_subtitle
            return results

        native = aic.clipboard_write_native, aic.clipboard_read_native
        aic.clipboard_write_native, aic.clipboard_read_native = (lambda text: False), (lambda: None)
        try:
            results = asyncio.run(scenario())
        finally:
            aic.clipboard_write_native, aic.clipboard_read_native = native
        for name, ok in results.items():
            self.check(f"tui todo: {name}", ok, repr(results) if not ok else "")

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


    # ------------------------------------------------------------------ #
    # Ralph loop, verify gate, project instructions, file tools, headless  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _done(text="done\nTASKCOMPLETE"):
        return [fake_openai.content_chunk(text), fake_openai.usage_chunk(1, 1)]

    @staticmethod
    def _call(tool, args, id="c1"):
        return [fake_openai.tool_call_chunk(0, id=id, name=tool, arguments=json.dumps(args)),
                fake_openai.usage_chunk(1, 1)]

    def test_completion_marker(self):
        has = aic._has_marker
        self.check("marker: own line", has("All good.\nTASKCOMPLETE\n", "TASKCOMPLETE"))
        self.check("marker: end of message", has("All done. TASKCOMPLETE", "TASKCOMPLETE"))
        self.check("marker: markup and promise tag", has("**TASKCOMPLETE**", "TASKCOMPLETE")
                   and has("<promise>TASKCOMPLETE</promise>", "TASKCOMPLETE"))
        self.check("marker: a mention does not count",
                   not has("I will emit TASKCOMPLETE once the tests pass.", "TASKCOMPLETE")
                   and not has("NOTTASKCOMPLETE", "TASKCOMPLETE"))
        c, fake = self.make_fake_commander([self._done("I'll say TASKCOMPLETE when the tests pass."), self._done()])
        status = c.run("task")
        self.check("marker: a mention keeps the run going", status == "complete" and len(fake.calls) == 2,
                   f"status={status} calls={len(fake.calls)}")

    def test_file_tools(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            c = self.make_commander()
            path = os.path.join(tmp, "sub", "a.py")

            def call(name, **args):
                return c._run_file_tool(name, args, [])
            r = call("write_file", path=path, content="one\ntwo\nthree\n")
            self.check("files: write_file creates parents", r.startswith("OK: created") and os.path.isfile(path), r)
            r = call("read_file", path=path)
            self.check("files: read_file numbers lines", r == "1\tone\n2\ttwo\n3\tthree\n", repr(r))
            r = call("read_file", path=path, offset=2, limit=1)
            self.check("files: read_file offset/limit", r.startswith("2\ttwo\n") and "offset=3" in r, repr(r))
            r = call("edit_file", path=path, old_string="two", new_string="TWO")
            self.check("files: edit_file replaces a unique string",
                       r.startswith("OK") and open(path).read() == "one\nTWO\nthree\n", r)
            r = call("edit_file", path=path, old_string="missing", new_string="x")
            self.check("files: edit_file reports a missing string", r.startswith("ERROR") and "not found" in r, r)
            call("write_file", path=path, content="x\nx\n")
            r = call("edit_file", path=path, old_string="x", new_string="y")
            self.check("files: edit_file refuses an ambiguous string", "occurs 2 times" in r, r)
            r = call("edit_file", path=path, old_string="x", new_string="y", replace_all=True)
            self.check("files: edit_file replace_all", open(path).read() == "y\ny\n", r)
            with open(path, "w", newline="") as f:
                f.write("a\r\nb\r\n")
            call("edit_file", path=path, old_string="a\nb", new_string="a\nc")
            self.check("files: edit_file keeps CRLF line endings", open(path, newline="").read() == "a\r\nc\r\n")
            r = call("edit_file", path=os.path.join(tmp, "nope.py"), old_string="a", new_string="b")
            self.check("files: edit_file on a missing file", "does not exist" in r, r)
            c.max_output_bytes = 30
            call("write_file", path=path, content="".join(f"line {i}\n" for i in range(1, 21)))
            r = call("read_file", path=path)
            self.check("files: read_file cut at whole lines", r.startswith("1\tline 1\n") and "of 20" in r, repr(r))
            c.max_output_bytes = 10240
            # Models often send a JSON file's data as an object instead of text.
            data_path = os.path.join(tmp, "data.json")
            data = {"name": 'He said "hi"', "path": "C:\\Users", "tags": ["a b", 'e"f']}
            r = call("write_file", path=data_path, content=data)
            self.check("files: write_file accepts JSON object content",
                       r.startswith("OK") and json.load(open(data_path)) == data, r)
            r = call("write_file", path=data_path, content=5)
            self.check("files: write_file says what content it got", "it was a JSON int" in r, r)

            c2 = self.make_commander(auto_approve=False)
            c2.sink.auto_approve = False
            c2.sink.confirm = lambda command: "keep it as is"
            suggestions = []
            r = c2._run_file_tool("write_file", {"path": path, "content": "changed"}, suggestions)
            self.check("files: rejected edit leaves the file and keeps the suggestion",
                       "skipped" in r and "changed" not in open(path).read() and suggestions == ["keep it as is"], r)

            target = os.path.join(tmp, "b.txt")
            c3, fake = self.make_fake_commander([self._call("write_file", {"path": target, "content": "hi\n"}),
                                                 self._done()])
            c3.run("write it")
            tool = next(m for m in c3.conversation_history if m.get("role") == "tool")
            self.check("files: the agent loop runs file tools",
                       open(target).read() == "hi\n" and tool["content"].startswith("OK: created"), tool["content"])
            self.check("files: tool schemas offered", {t["function"]["name"] for t in fake.calls[0]["tools"]}
                       == {"execute_bash", "read_file", "edit_file", "write_file", "todo", "task_complete"})

    def test_verify_gate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            flag = os.path.join(tmp, "flag")
            c, fake = self.make_fake_commander([self._done(), self._done()],
                                               verify_command=f"test -f {flag} || {{ touch {flag}; exit 1; }}")
            status = c.run("task")
            feedback = [m for m in c.conversation_history if "[VERIFY FAILED]" in str(m.get("content"))]
            self.check("verify: a failing check sends the model back to work",
                       status == "complete" and len(fake.calls) == 2 and len(feedback) == 1,
                       f"status={status} calls={len(fake.calls)}")
            self.check("verify: the system prompt mentions the check", "test -f" in c.get_system_prompt())

    def test_ralph_loop(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            prompt = os.path.join(tmp, "PROMPT.md")
            with open(prompt, "w") as f:
                f.write("Build the thing.\n")
            c, fake = self.make_fake_commander([self._done("did item 1\nTASKCOMPLETE"),
                                                self._done("everything done\nLOOPCOMPLETE")])
            status = c.run_loop(prompt, 5)
            second = fake.calls[1]["messages"] if len(fake.calls) > 1 else []
            self.check("loop: LOOPCOMPLETE ends the loop", status == "loop_complete" and len(fake.calls) == 2,
                       f"status={status} calls={len(fake.calls)}")
            self.check("loop: each iteration starts from a fresh context",
                       [m["role"] for m in second] == ["system", "user"]
                       and second[1]["content"] == "Build the thing."
                       and "Loop mode (iteration 2)" in second[0]["content"], repr(second)[:300])
            self.check("loop: loop mode is off afterwards", c.loop_iteration == 0
                       and "Loop mode" not in c.get_system_prompt())
            c2, fake2 = self.make_fake_commander([self._done()] * 3)
            status = c2.run_loop(prompt, 2)
            self.check("loop: max_iterations", status == "max_iterations" and len(fake2.calls) == 2,
                       f"status={status} calls={len(fake2.calls)}")
            c3, _ = self.make_fake_commander()
            self.check("loop: a missing prompt file is an error",
                       c3.run_loop(os.path.join(tmp, "missing.md")) == "error")

    def test_shell_and_cwd(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "sub")
            c = self.make_commander()
            out, code = c.execute_bash_command("[[ 1 == 1 ]] && arr=(a b) && echo ${arr[1]}")
            self.check("shell: bash syntax works", code == 0 and out.strip() == "b", repr(out))
            c.execute_bash_command(f"cd {shlex.quote(tmp)} && mkdir sub && cd sub")
            out, _ = c.execute_bash_command("pwd")
            self.check("cwd: a cd carries over", c.cwd == sub and out.strip() == sub, f"{c.cwd} {out!r}")
            _, code = c.execute_bash_command("cd /; exit 3")
            self.check("cwd: followed on failure, exit code intact", code == 3 and c.cwd == "/", f"{code} {c.cwd}")
            c.execute_bash_command(f"cd {shlex.quote(sub)}")
            r = c._run_file_tool("write_file", {"path": "f.txt", "content": "x"}, [])
            self.check("cwd: file tools resolve relative paths against it",
                       os.path.isfile(os.path.join(sub, "f.txt")), r)
            out, _ = c.execute_bash_command("pwd", cwd=tmp)
            self.check("cwd: an explicit cwd is not followed", out.strip() == tmp and c.cwd == sub, repr(out))
            c.execute_bash_command(f"rm -rf {shlex.quote(sub)}")
            out, _ = c.execute_bash_command("pwd")
            self.check("cwd: a vanished directory falls back to the start",
                       c.cwd == c.start_cwd and out.strip() == c.start_cwd, f"{c.cwd} {out!r}")
            c2, _ = self.make_fake_commander([self._call("execute_bash", {"command": f"cd {shlex.quote(tmp)}"}),
                                              self._done()])
            c2.run("go")
            tool = next(m for m in c2.conversation_history if m.get("role") == "tool")
            self.check("cwd: the model is told the directory changed",
                       tool["content"].endswith(f"[working directory is now {tmp}]"), repr(tool["content"]))
            self.check("cwd: saved in the session", c2.session_snapshot()["cwd"] == tmp)
            c2.persist_history = False
            c2.client = fake_openai.FakeOpenAIClient([self._done()])
            c2.run("fresh")
            self.check("cwd: a fresh conversation starts where aic started", c2.cwd == c2.start_cwd, c2.cwd)

    def test_read_only_commands(self):
        ro = aic._is_read_only
        safe = ["ls -la", "git status", "grep -rn 'a|b' src | head -5", "cat x 2>/dev/null || echo none",
                "cd sub && ls", "find . -name '*.py'", "git log --oneline -5", "sort a | uniq -c",
                "wc -l a b 2>&1", "git branch -a"]
        unsafe = ["rm x", "ls > out", "ls >| out", "echo $(rm x)", "cat `x`", "ls; rm -rf x", "ls\nrm x",
                  "find . -delete", "find . -exec rm {} +", "git push", "git branch -D x", "sort -o out x",
                  "./ls", "FOO=1 ls", "ls &", "uniq a b", "cat a#b; rm x", "ls | sh", "tee x", "python3 -c 1",
                  "git -c core.pager=x log", "git diff --output=x", "cat <(rm x)", "'unterminated", "",
                  "sort -no out x", "git grep -iO x", "tree -ao out"]
        self.check("read-only: plain reads qualify", all(ro(cmd) for cmd in safe), [cmd for cmd in safe if not ro(cmd)])
        self.check("read-only: writes, substitutions and unknowns do not", not any(ro(cmd) for cmd in unsafe),
                   [cmd for cmd in unsafe if ro(cmd)])
        two_calls = [[fake_openai.tool_call_chunk(0, id="a", name="execute_bash", arguments='{"command": "ls"}'),
                      fake_openai.tool_call_chunk(1, id="b", name="execute_bash",
                                                  arguments='{"command": "rm -f no-such-file"}'),
                      fake_openai.usage_chunk(1, 1)], self._done()]
        for reads, expected in ((True, ["rm -f no-such-file"]), (False, ["ls", "rm -f no-such-file"])):
            c, _ = self.make_fake_commander([list(t) for t in two_calls], auto_approve=False,
                                            auto_approve_reads=reads)
            asked = []
            c.sink.confirm = lambda command, asked=asked: asked.append(command) or "n"
            c.run("look")
            self.check(f"read-only: {'only writes ask' if reads else '--confirm-reads asks for all'}",
                       asked == expected, repr(asked))

    def test_todo_and_task_complete(self):
        todo = {"items": [{"text": "one", "status": "done"}, {"text": "two", "status": "in_progress"}, "three"]}
        c, fake = self.make_fake_commander([
            self._call("todo", todo),
            [fake_openai.tool_call_chunk(0, id="x", name="execute_bash", arguments='{"command": "true"}'),
             fake_openai.tool_call_chunk(1, id="y", name="task_complete", arguments='{"summary": "early"}'),
             fake_openai.usage_chunk(1, 1)],
            self._call("task_complete", {"summary": "All done."}, id="c3")])
        status = c.run("task")
        results = {m["tool_call_id"]: m["content"] for m in c.conversation_history if m.get("role") == "tool"}
        self.check("todo: the list is stored and shown", c.todos[2] == {"text": "three", "status": "pending"}
                   and results.get("c1", "").endswith("(1/3 done):\n[x] one\n[>] two\n[ ] three"), repr(results))
        self.check("finish: refused alongside other calls", results.get("y", "").startswith("Not accepted"))
        self.check("finish: task_complete ends the run", status == "complete" and len(fake.calls) == 3,
                   f"status={status} calls={len(fake.calls)}")
        self.check("finish: the summary is shown and replayable",
                   any(e[0] == "LOG" and e[1]["text"] == "All done." for e in c.sink.events)
                   and AICommander.finish_summary(c.conversation_history[-2]) == "All done.")
        self.check("todo: saved in the session", c.session_snapshot()["todos"] == c.todos)
        r = c._todo({"items": json.dumps([{"text": "x", "status": "done"}])})
        self.check("todo: a list sent as JSON text is accepted", c.todos == [{"text": "x", "status": "done"}], r)
        c.todos = AICommander.normalize_todos([{"text": "a", "status": "completed"},
                                               {"text": "b", "status": "In-Progress"}, {"text": "c", "status": ["odd"]}])
        self.check("todo: status spellings normalized", [t["status"] for t in c.todos]
                   == ["done", "in_progress", "pending"], repr(c.todos))
        c.cwd = "/"
        c.discard_session("cleared")
        self.check("todo: /clear forgets the task list and the cwd", c.todos == [] and c.cwd == c.start_cwd)
        self.check("prompt: no claim of unlimited thinking", "unlimited" not in c.get_system_prompt())

        # After a compaction the task list is shown to the model again.
        c2, fake2 = self.make_fake_commander([self._done()], model="gpt-test", compress_algorithm="truncate")
        c2.todos = [{"text": "keep me", "status": "in_progress"}]
        history = [{"role": "system", "content": c2.get_system_prompt()}, {"role": "user", "content": "start"}]
        for i in range(4):
            history += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}", "type": "function",
                         "function": {"name": "execute_bash", "arguments": '{"command": "ls"}'}}]},
                        {"role": "tool", "tool_call_id": f"t{i}", "content": "A" * 4000}]
        c2.conversation_history, c2.persist_history = history, True
        c2.max_prompt_len = c2._estimate_input_tokens(history[:2]) + 1500
        c2.run("next")
        sent = fake2.calls[0]["messages"] if fake2.calls else []
        self.check("todo: shown again after compaction",
                   sent and sent[-1]["content"].startswith("[CONTEXT COMPACTED]") and "[>] keep me" in sent[-1]["content"],
                   repr(sent[-1:])[:300])

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            flag = os.path.join(tmp, "flag")
            c3, fake3 = self.make_fake_commander([self._call("task_complete", {"summary": "a"}),
                                                  self._call("task_complete", {"summary": "b"}, id="c2")],
                                                 verify_command=f"test -f {flag} || {{ touch {flag}; exit 1; }}")
            status = c3.run("task")
            first = next(m for m in c3.conversation_history if m.get("role") == "tool")
            self.check("finish: --verify gates task_complete", status == "complete" and len(fake3.calls) == 2
                       and first["content"].startswith("[VERIFY FAILED]"), f"status={status}")
            prompt = os.path.join(tmp, "PROMPT.md")
            with open(prompt, "w") as f:
                f.write("Build the thing.\n")
            c4, fake4 = self.make_fake_commander([
                self._call("task_complete", {"summary": "item 1", "objective_complete": False}),
                self._call("task_complete", {"summary": "all", "objective_complete": True})])
            status = c4.run_loop(prompt, 5)
            tools = {t["function"]["name"]: t for t in fake4.calls[0]["tools"]} if fake4.calls else {}
            self.check("finish: objective_complete ends a --loop", status == "loop_complete"
                       and len(fake4.calls) == 2 and "objective_complete"
                       in tools.get("task_complete", {}).get("function", {}).get("parameters", {}).get("properties", {}),
                       f"status={status} calls={len(fake4.calls)}")

    def test_background_jobs(self):
        c, fake = self.make_fake_commander([
            self._call("execute_bash", {"command": "sleep 0.3; echo job-done; exit 4", "background": True}),
            self._call("execute_bash", {"command": "sleep 1"}, id="c2"),
            self._done()])
        c.run("bg")
        started = next(m for m in c.conversation_history if m.get("role") == "tool")
        self.check("jobs: a background call returns at once", started["content"].startswith("Started background job 1"),
                   started["content"])
        notes = [m for m in fake.calls[2]["messages"] if "[Background job 1 exited with code 4" in str(m.get("content"))] \
            if len(fake.calls) > 2 else []
        self.check("jobs: the exit is reported with the last output", notes and "job-done" in notes[0]["content"],
                   repr(notes))
        c2 = self.make_commander()
        c2._start_job("sleep 30")
        proc = c2.jobs[1][0]
        c2._cleanup()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.check("jobs: killed and cleaned up at exit", proc.returncode is not None
                   and not os.path.isdir(c2._scratch))

    def test_project_instructions(self):
        import tempfile
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                c = self.make_commander()
                self.check("project: no file, no section", "Project instructions" not in c.get_system_prompt())
                with open("CLAUDE.md", "w") as f:
                    f.write("Use tabs.")
                self.check("project: CLAUDE.md is a fallback", "Use tabs." in c.get_system_prompt())
                with open("AGENTS.md", "w") as f:
                    f.write("Run make test.")
                prompt = c.get_system_prompt()
                self.check("project: AGENTS.md wins", "Run make test." in prompt and "Use tabs." not in prompt)
                c.fast = True
                self.check("project: fast prompt too", "Run make test." in c.get_system_prompt())
            finally:
                os.chdir(cwd)

    def test_headless_cli(self):
        """aic.py --nogui --json against a local fake streaming server: env-var
        configuration, a prompt piped on stdin, exit codes, AIC_API_KEY scrub."""
        import http.server
        import subprocess
        import tempfile
        requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                last = body["messages"][-1]
                if "tools" not in body:
                    delta = {"content": "a title"}
                elif last["role"] == "tool":
                    delta = {"content": "done\nTASKCOMPLETE"}
                else:
                    delta = {"tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {
                        "name": "execute_bash", "arguments": json.dumps({"command": "env | grep -c AIC_API_KEY"})}}]}
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "AIC_"))}
        env.update(AIC_API_BASE=f"http://127.0.0.1:{server.server_port}/v1", AIC_MODEL="m", AIC_API_KEY="secret")
        script = os.path.join(os.path.dirname(os.path.abspath(aic.__file__)), "aic.py")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                def run(*extra):
                    return subprocess.run([sys.executable, script, "--nogui", "--auto-approve", "--disable-sandbox",
                                           "--no-session", *extra], input="say hi", capture_output=True, text=True,
                                          env=env, cwd=tmp, timeout=60)
                proc = run("--json")
                events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
                shutdown = [e for e in events if e["type"] == "SHUTDOWN"]
                self.check("headless: exit 0 on completion", proc.returncode == 0,
                           f"rc={proc.returncode} {proc.stderr[-300:]}")
                self.check("headless: --json emits JSON lines",
                           events and len(events) == len(proc.stdout.splitlines())
                           and shutdown and shutdown[0].get("status") == "complete", proc.stdout[-300:])
                tool_requests = [r for r in requests if "tools" in r]
                self.check("headless: prompt read from stdin and env config used",
                           tool_requests and tool_requests[0]["messages"][1]["content"] == "say hi")
                result = [m for r in tool_requests for m in r["messages"] if m["role"] == "tool"]
                self.check("headless: AIC_API_KEY hidden from commands",
                           result and "Output:\n0" in result[0]["content"], repr(result)[:200])
                proc = run("--max-steps", "1")
                self.check("headless: exit 2 at the step limit", proc.returncode == 2, f"rc={proc.returncode}")
                env.pop("AIC_MODEL")
                proc = run()
                self.check("headless: a missing model is a usage error",
                           proc.returncode == 2 and "AIC_MODEL" in proc.stderr, proc.stderr[-200:])
        finally:
            server.shutdown()


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
    runner.test_sanitize_tool_pairing()
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
    runner.test_session_list_content_survives_reload()
    runner.test_session_closes_unanswered_tool_call()
    runner.test_session_corrupt_file()
    runner.test_session_missing_file_is_quiet()
    runner.test_session_disabled_without_path()
    runner.test_session_written_during_run()
    runner.test_session_discard()
    runner.test_session_instance_locking()
    runner.test_validate_merges_consecutive_assistants()
    runner.test_command_output_cleaning()
    runner.test_output_truncation_multibyte()
    runner.test_command_timeout_keeps_partial_output()
    runner.test_process_response_argument_shapes()
    runner.test_stream_tool_call_without_index()
    runner.test_empty_responses_end_the_run()
    runner.test_stop_answers_remaining_tool_calls()
    runner.test_suggestion_follows_all_tool_results()
    runner.test_truncate_keeps_current_request()
    runner.test_ccllm_summary_is_folded_not_stacked()
    runner.test_summarizer_failures_fall_back()
    runner.test_internal_calls_use_low_effort()
    runner.test_output_limit_cut_off()
    runner.test_context_overflow_is_retried()
    runner.test_transient_error_is_retried()
    runner.test_prompt_history_file()
    runner.test_tui_log_wraps_and_copies()
    runner.test_preserved_thinking()
    runner.test_x11_clipboard()
    runner.test_tui_end_to_end()
    runner.test_tui_tabs()
    runner.test_tui_todo()
    runner.test_completion_marker()
    runner.test_file_tools()
    runner.test_verify_gate()
    runner.test_ralph_loop()
    runner.test_project_instructions()
    runner.test_headless_cli()
    runner.test_shell_and_cwd()
    runner.test_read_only_commands()
    runner.test_todo_and_task_complete()
    runner.test_background_jobs()

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
