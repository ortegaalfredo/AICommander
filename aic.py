#!/usr/bin/env python3
"""AI-Commander: a ralph-loop AI agent giving LLMs shell access via function calling.

Modes: --nogui (direct CLI) or the default Textual TUI. All agent I/O goes
through an EventSink (ConsoleSink or TUISink) so the same agent code drives
either presentation.
"""

import argparse
import json
import sys
import time
import threading
import queue
import re
import signal
import os
import pty
import select
import fcntl
import termios
import struct
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI

try:
    # context-compressor-llm provides the Factory.ai-style anchored-summary
    # incremental compression algorithm used by the "context-compressor-llm"
    # algorithm. Optional: when absent, that algorithm falls back to truncate.
    from context_compressor import ContextCompressor as _ContextCompressorLib
    _CONTEXT_COMPRESSOR_LLM_AVAILABLE = True
except ImportError:
    _ContextCompressorLib = None
    _CONTEXT_COMPRESSOR_LLM_AVAILABLE = False

# Browser-like User-Agent for all outbound HTTP (OpenAI API calls plus
# curl/wget run through execute_bash) so Cloudflare-fronted endpoints do not
# flag the script as a bot and block the IP.
USER_AGENT = 'Mozilla/5.0 (compatible; OpenAI-Client/1.0)'

try:
    # Textual is only required for TUI mode; --nogui runs without it.
    from textual.app import ComposeResult
    from textual.widgets import Static, Button
    from textual.screen import ModalScreen
    from textual.containers import Horizontal, Vertical
    from textual.binding import Binding
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

# --- Readline-style prompt history --------------------------------------
# The standard `readline` module provides an in-memory history plus file
# persistence. We wrap it so --nogui / platforms without readline still run.
try:
    import readline as _rl
    _READLINE_AVAILABLE = True
except ImportError:
    _rl = None
    _READLINE_AVAILABLE = False

_HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".aic_history")
_HISTORY_MAXLEN = 1000


def _rl_add_history(line: str) -> None:
    """Append a line to the readline history (dedupes consecutive repeats)."""
    if not _READLINE_AVAILABLE or not line:
        return
    try:
        if _rl.get_current_history_length() and \
                _rl.get_history_item(_rl.get_current_history_length()) == line:
            return
        _rl.add_history(line)
    except Exception:
        pass


def _rl_get_history_item(index: int) -> str:
    """Return the history item at 1-based *index*, or '' if unavailable."""
    if _READLINE_AVAILABLE:
        try:
            return _rl.get_history_item(index) or ""
        except Exception:
            return ""
    return ""


def _rl_history_length() -> int:
    """Return the current history length (0 when readline is unavailable)."""
    if _READLINE_AVAILABLE:
        try:
            return _rl.get_current_history_length()
        except Exception:
            return 0
    return 0


def _rl_load_history() -> None:
    """Load persisted history into readline at startup."""
    if _READLINE_AVAILABLE:
        try:
            _rl.set_history_length(_HISTORY_MAXLEN)
            if os.path.exists(_HISTORY_FILE):
                _rl.read_history_file(_HISTORY_FILE)
        except Exception:
            pass


def _rl_save_history() -> None:
    """Persist the current readline history to disk."""
    if _READLINE_AVAILABLE:
        try:
            _rl.set_history_length(_HISTORY_MAXLEN)
            _rl.write_history_file(_HISTORY_FILE)
        except Exception:
            pass


class colors:
    """ANSI codes used by ConsoleSink (stripped again by TUISink)."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'


class CommandTimeoutError(Exception):
    """Raised when command execution times out."""
    pass


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub('', text)


class EventSink:
    """Abstract presentation-layer I/O for AICommander.

    Event kinds: LOG, ERROR, LLM_STREAM, THINKING_STREAM, CONSOLE_STREAM,
    CMD_EXEC, CMD_OUTPUT, CMD_COMPLETE, SYSTEM, STATUS_UPDATE,
    APPROVAL_REQUEST, SHUTDOWN, TOKEN_USAGE. Common payload keys: text, end,
    flush. CONSOLE_STREAM streams text to the console output pane (used by
    internal calls like the context-compressor summarizer). TOKEN_USAGE
    carries input_tokens/output_tokens (estimate flag set while streaming;
    exact numbers once the API call finishes).
    """

    def emit(self, kind: str, payload: dict):
        raise NotImplementedError

    def input(self, prompt: str) -> str:
        """Request a line of user input (approval gate)."""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class ConsoleSink(EventSink):
    """Direct terminal I/O for --nogui mode (no queueing, no TUI)."""

    _STYLES = {
        "ERROR": colors.RED,
        "LLM_STREAM": colors.CYAN,
        "THINKING_STREAM": colors.YELLOW,
        "CMD_OUTPUT": colors.GREEN,
        "CMD_EXEC": colors.YELLOW,
        "CMD_COMPLETE": colors.BOLD,
        "SYSTEM": colors.BOLD + colors.BLUE,
    }
    _LOG_STYLES = {"yellow": colors.YELLOW, "green": colors.GREEN, "red": colors.RED}

    def emit(self, kind: str, payload: dict):
        text = payload.get("text", "")
        if kind in ("STATUS_UPDATE", "TOKEN_USAGE"):
            return  # status/token footer is meaningless in CLI mode
        # Streaming kinds must not append a newline after each token.
        end = "" if kind in ("LLM_STREAM", "THINKING_STREAM", "CONSOLE_STREAM") else payload.get("end", "\n")
        flush = payload.get("flush", True)
        if kind == "LOG":
            style = self._LOG_STYLES.get(payload.get("style", ""), "")
        else:
            style = self._STYLES.get(kind, "")
        out = sys.stderr if kind == "ERROR" else sys.stdout
        print(f"{style}{text}{colors.END}" if style else text, file=out, end=end, flush=flush)

    def input(self, prompt: str) -> str:
        import builtins
        return builtins.input(prompt)

    def close(self):
        pass


class TUISink(EventSink):
    """Queue bridge to the Textual app. The agent thread only writes to the
    queue; the main thread polls it and renders events to widgets.

    input() implements the approval gate: returns "y" immediately when
    auto_approve is on, otherwise pushes APPROVAL_REQUEST and blocks on a
    threading.Event until the TUI calls resolve_approval().
    """

    def __init__(self, event_queue: queue.Queue, stop_event: threading.Event):
        self.queue = event_queue
        self.stop_event = stop_event
        self.auto_approve = False
        self._approval_event: Optional[threading.Event] = None
        self._approval_response: Optional[str] = None

    def emit(self, kind: str, payload: dict):
        if self.stop_event.is_set():
            return
        # Strip ANSI codes; the TUI applies its own styling via Rich markup.
        clean = {k: _strip_ansi(v) if isinstance(v, str) else v for k, v in payload.items()}
        self.queue.put({"kind": kind, "payload": clean})

    def input(self, prompt: str) -> str:
        if self.auto_approve:
            return "y"

        # Create the event BEFORE publishing the request so the TUI can never
        # resolve it before we start waiting on it (race condition).
        self._approval_event = threading.Event()
        self._approval_response = None
        self.emit("APPROVAL_REQUEST", {"command": prompt})

        # Poll so we can react to shutdown while blocked.
        while not self._approval_event.is_set():
            if self.stop_event.is_set():
                self._approval_event = None
                self._approval_response = None
                return "n"  # reject on shutdown
            self._approval_event.wait(0.1)

        response = self._approval_response or "n"
        self._approval_event = None
        self._approval_response = None
        return response

    def resolve_approval(self, approved: bool, suggestion: str = ""):
        """Called by the TUI when the user answers a pending approval request."""
        self._approval_response = "y" if approved else (suggestion or "n")
        if self._approval_event:
            self._approval_event.set()

    def close(self):
        self.stop_event.set()
        if self._approval_event:
            self._approval_event.set()  # unblock any waiting approval


class _AICommanderTokenCounter:
    """Token-counter adapter that plugs aic.py's token estimator into the
    context-compressor-llm library.

    The library's ContextCompressor calls ``tokenizer.count_tokens(text)``
    (and ``count_message_tokens``) to drive its thresholds. Reusing aic.py's
    own estimator here keeps the compressor's accounting consistent with the
    live "Context" counter and the prompt-budget trigger, so there is no drift
    between the threshold check and the actual prompt size.
    """

    def __init__(self, commander):
        self._commander = commander

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return self._commander._estimate_message_tokens({"role": "user", "content": text})

    def count_message_tokens(self, messages: list) -> int:
        return sum(self.count_tokens(m.get("content", "")) for m in messages)


class AICommander:
    """Main class for AI-Commander functionality"""

    COMPLETION_MARKER = "TASKCOMPLETE"
    OUTPUT_TRUNCATION_SENTINEL = "output too long: truncated"

    def __init__(self, api_base: str, model: str, api_key: str, auto_approve: bool = False,
                  show_thinking: bool = True, command_timeout: int = 120,
                  max_prompt_len: int = 20000, max_output_bytes: int = 10240, debug: bool = False,
                  sink: EventSink = None, persist_history: bool = False, max_steps: int = 500,
                  compress_algorithm: str = "context-compressor-llm"):
        self.base_url = api_base.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.auto_approve = auto_approve
        self.conversation_history = []
        # When True, run() extends the existing conversation history instead
        # of resetting it, so a fresh prompt keeps prior chat context.
        self.persist_history = persist_history
        self.max_steps = max_steps
        self.max_tokens = 8000
        self.command_timeout = command_timeout
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.show_thinking = show_thinking
        self.max_prompt_len = max_prompt_len
        self.max_output_bytes = max_output_bytes
        self.debug = debug
        self.sink: EventSink = sink if sink is not None else ConsoleSink()
        # Context-compression algorithm applied when the conversation history
        # exceeds the prompt budget. "context-compressor-llm" (the default)
        # uses the anchored-summary incremental compressor from the
        # context-compressor-llm package; "truncate" drops the oldest messages
        # and condenses tool outputs. Selected via the command line.
        self.compress_algorithm = compress_algorithm

        # Checked between steps and during command execution so the TUI can
        # halt the agent cleanly.
        self.stop_event = threading.Event()

        # Mid-run user suggestions queued by the TUI; drained at the start of
        # each step and injected as user messages.
        self.suggestion_queue: queue.Queue = queue.Queue()

        # run() is re-entered for every new prompt but the startup banner
        # should only appear once per session.
        self._started_banner_shown = False

        # default_headers sends the User-Agent on every API call (Cloudflare).
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={'User-Agent': USER_AGENT}
        )

        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "description": "Execute a bash command in the terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]

    def _log(self, message: str, end: str = '\n', flush: bool = True, style: str = ""):
        self.sink.emit("LOG", {"text": message, "end": end, "flush": flush, "style": style})

    def _log_error(self, message: str):
        self.sink.emit("ERROR", {"text": message})

    def _estimate_message_tokens(self, msg: Dict[str, Any]) -> int:
        """Rough token estimate for a single message.

        Approximates ~3 characters per token plus a fixed per-message and
        per-tool-call overhead. No tiktoken dependency; exact numbers come
        from the API usage stats reported once the call finishes.

        The 3-char heuristic is deliberately conservative (it tends to
        over-estimate vs. the server's tokenizer), so the live "Context"
        counter and the truncation threshold trigger *before* the request
        exceeds the model's window rather than after.
        """
        total = 4  # structural overhead (role, separators)
        content = msg.get("content")
        if isinstance(content, str):
            total += max(1, len(content) // 3)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    total += max(1, len(part["text"]) // 3)
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += max(1, len(fn.get("name") or "") // 3)
            total += max(1, len(fn.get("arguments") or "") // 3)
        return total

    def _estimate_input_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Rough input-token estimate for the live "context" counter."""
        return sum(self._estimate_message_tokens(msg) for msg in messages)

    def _dump_conversation_history(self):
        """Dump the conversation history to commander-debug.txt for debugging."""
        with open("commander-debug.txt", "w", encoding="utf-8") as f:
            f.write(f"Conversation history dump at {datetime.now().isoformat()}\n")
            f.write(f"Total messages: {len(self.conversation_history)}\n")
            total_tokens = sum(self._estimate_message_tokens(msg) for msg in self.conversation_history)
            f.write(f"Total content length: {total_tokens} tokens (est)\n")
            f.write(f"Max prompt length: {self.max_prompt_len} tokens\n")
            f.write("=" * 80 + "\n\n")
            for i, msg in enumerate(self.conversation_history):
                content = msg.get('content', '') or ''
                f.write(f"--- Message {i} (role: {msg.get('role', 'unknown')}) ---\n")
                if msg.get('tool_call_id'):
                    f.write(f"tool_call_id: {msg['tool_call_id']}\n")
                if msg.get('tool_calls'):
                    f.write(f"tool_calls: {json.dumps(msg['tool_calls'], indent=2)}\n")
                f.write(f"content ({len(content)} chars):\n{content}\n\n")
            f.write("=" * 80 + "\nEnd of conversation history\n")
        self._log(f"[DEBUG DUMP] Conversation history dumped to commander-debug.txt")

    def _prompt_budget(self) -> int:
        """Max tokens the prompt may occupy, leaving room for model output.

        The request reserves ``max_tokens`` tokens for the completion (sent as
        the ``max_tokens`` request param for non-"gpt" models). The server's
        context window must hold prompt + output, so the input prompt must be
        kept under ``max_prompt_len - max_tokens`` or the server rejects the
        call with an "exceed context size" error. For models that don't send
        ``max_tokens`` the full window is available to the prompt.
        """
        if "gpt" not in self.model:
            return max(0, self.max_prompt_len - self.max_tokens)
        return self.max_prompt_len

    def _context_compress(self):
        """Compress conversation history below the prompt budget (in tokens).

        The budget is ``max_prompt_len`` minus the output reservation
        (``max_tokens``) when the request carries one, so the prompt leaves
        room for the completion inside the model's context window.

        This is the single entry point for context compression. It computes the
        trigger (is the history over budget?) and then dispatches to the
        configured compression algorithm, ``self.compress_algorithm``. Each
        algorithm is implemented as a ``_compress_<name>`` method; new
        algorithms are added in later versions and selected via the command
        line. Unknown algorithms fall back to the default, "truncate".
        """
        budget = self._prompt_budget()
        total_tokens = sum(self._estimate_message_tokens(msg) for msg in self.conversation_history)

        if total_tokens <= budget:
            return

        # Hyphenated algorithm names (e.g. "context-compressor-llm") map to
        # `_compress_<name>` methods with the hyphens replaced by underscores.
        method_name = f"_compress_{self.compress_algorithm.replace('-', '_')}"
        algorithm = getattr(self, method_name, None)
        if algorithm is None:
            self._log(f"[COMPRESS] Unknown context-compression algorithm '{self.compress_algorithm}'. Falling back to 'truncate'.", style="yellow")
            algorithm = self._compress_truncate
        algorithm(budget, total_tokens)

    def _compress_truncate(self, budget: int, total_tokens: int):
        """Truncate algorithm: shrink conversation history below the prompt budget.

        Keeps the system prompt (index 0) and first user instruction (index 1).
        Pass 1 condenses oversized tool outputs in-place, but only as many as
        needed; pass 2 drops the oldest messages from index 2 onwards. Both
        stop once the retained history fits within ~60% of the budget, leaving
        headroom so several new turns fit before the next compression.
        """
        # Leave ~60% headroom (matching the context-compressor-llm algorithm)
        # so the agent can run several more steps before re-triggering
        # compression. Truncating down to the FULL budget makes every following
        # prompt exceed the limit again and re-compress on each step, slowly
        # losing more history than necessary. Conversely, condensing EVERY tool
        # output to a placeholder collapses the context far below the target
        # (e.g. 22505 -> 1476 tokens) when tool outputs dominate, so pass 1
        # must condense only as many as needed to reach the target.
        retained_target = max(1, int(budget * 0.6))
        self._log(f"[COMPRESS] Using 'truncate' algorithm. Prompt length ({total_tokens} tokens) exceeds budget ({budget}). Retaining up to ~{retained_target} tokens ({int(budget * 0.6 / budget * 100)}% of budget). Compressing conversation history.", style="yellow")

        # Pass 1: condense oversized tool outputs, largest first (most tokens
        # saved per operation, so the fewest messages lose detail), stopping as
        # soon as the history fits the retained target.
        oversized = [
            (i, msg) for i, msg in enumerate(self.conversation_history)
            if msg.get('role') == 'tool'
            and isinstance(msg.get('content'), str)
            and len(msg['content']) > 30
        ]
        oversized.sort(key=lambda t: len(t[1]['content']), reverse=True)
        for i, msg in oversized:
            if total_tokens <= retained_target:
                break
            removed_tokens = self._estimate_message_tokens(msg)
            msg['content'] = '<condensed tool output>'
            total_tokens -= (removed_tokens - self._estimate_message_tokens(msg))
            self._log(f"[TRUNCATED] Condensed tool message (removed {removed_tokens} tokens, new length: {total_tokens})", style="yellow")

        # Pass 2: drop the oldest messages (from index 2 onwards) until the
        # retained history fits the target. This is a fallback for when
        # condensation alone can't reach the target (e.g. few small messages).
        i = 2
        while i < len(self.conversation_history) and total_tokens > retained_target:
            removed_msg = self.conversation_history.pop(i)
            removed_tokens = self._estimate_message_tokens(removed_msg)
            total_tokens -= removed_tokens
            self._log(f"[TRUNCATED] Removed {removed_msg.get('role')} message (removed {removed_tokens} tokens, new length: {total_tokens})", style="yellow")

        final_tokens = sum(self._estimate_message_tokens(msg) for msg in self.conversation_history)
        self._log(f"[TRUNCATING COMPLETE] Final prompt length: {final_tokens} tokens", style="yellow")

    def _llm_summarize(self, messages: List[dict], previous_summary: Optional[str]) -> str:
        """LLM summarizer callable for the context-compressor-llm library.

        Receives only the evicted message segment (the library isolates
        ``messages_to_summarize`` before calling) plus any prior anchored
        summary, so the summarizer call stays short on slow-prefill systems.
        Returns a concise summary string that is folded into the persistent
        ``AnchoredSummary``.
        """

        prompt = """
CRITICAL: This summarization request is a SYSTEM OPERATION, not a user message.
When analyzing "user requests" and "user intent", completely EXCLUDE this summarization message.
The "most recent user request" and "Optional Next Step" must be based on what the user was doing BEFORE this system message appeared.
The goal is for work to continue seamlessly after condensation - as if it never happened.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, key patterns, and important decisions that would be essential for continuing the work without losing context.
This summary should not be over 5000 characters in lenght.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.

If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
   - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
   - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
   - [Detailed non tool use user message]
   - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.
Remember to not exceed 5000 characters, its very important that the summary to be concise.
"""



        if previous_summary:
            prompt += "\n\nPrior summary (fold new information into it):\n" + previous_summary
        prompt += "\n\nMessages to summarize:\n"
        for m in messages:
            prompt += f"\n[{m.get('role', 'unknown')}] {m.get('content', '')}"

        try:
            # Some OpenAI-compatible endpoints reject a lone system message with
            # "no user query found". Always include a user message carrying the
            # summarization instruction so the summarizer call is accepted.
            response = self.call_llm_api(
                [
                    {"role": "system", "content": "You are a precise conversation summarizer."},
                    {"role": "user", "content": prompt},
                ],
                use_tools=False,
                stream_display=False,
                stream_label="[SESSION SUMMARY]",
            )
            msg = response.get("choices", [{}])[0].get("message", {})
            summary = msg.get("content") or ""
            return summary.strip() or "[summary unavailable]"
        except Exception as e:
            self._log_error(f"[COMPRESS] LLM summarizer failed: {e}. Using placeholder summary.")
            return "[summary unavailable]"

    def _compress_context_compressor_llm(self, budget: int, total_tokens: int):
        """context-compressor-llm algorithm: anchored-summary incremental compression.

        Delegates to the Factory.ai-style ``ContextCompressor`` from the
        context-compressor-llm package. When the non-system log exceeds the
        budget it evicts the oldest prefix, folds it into a persistent
        ``AnchoredSummary`` via an LLM call on the evicted segment only, and
        retains the newest suffix -- an append-only, KV-cache-friendly layout.
        The system prompt is kept byte-identical at the front of the retained
        context so unchanged prefixes are never re-prefilled.
        """
        if not _CONTEXT_COMPRESSOR_LLM_AVAILABLE:
            self._log("[COMPRESS] 'context-compressor-llm' selected but the "
                      "context-compressor-llm package is not installed. Falling back to 'truncate'.",
                      style="yellow")
            self._compress_truncate(budget, total_tokens)
            return

        self._log(f"[COMPRESS] Using 'context-compressor-llm' algorithm. Prompt length ({total_tokens} tokens) exceeds budget ({budget}). Compressing conversation history.",
                  style="yellow")

        # Keep the system prompt out of the compressor so it stays byte-identical.
        system_prompt = None
        if self.conversation_history and self.conversation_history[0].get("role") == "system":
            system_prompt = self.conversation_history[0].get("content", "")
        # Preserve the FIRST user instruction (the original task) so the LLM
        # never forgets what it was initially asked to do. The "truncate"
        # algorithm always keeps the system prompt (index 0) and the first user
        # instruction (index 1); the context-compressor-llm algorithm must do
        # the same. Otherwise the initial request gets evicted into the anchored
        # summary (or dropped entirely on the "no user query" fallback) and the
        # agent loses track of its primary objective. The first user message is
        # kept verbatim, outside the summarizable log.
        first_user = None
        for i, m in enumerate(self.conversation_history):
            if m.get("role") == "user":
                first_user = m
                log = [x for x in self.conversation_history[i + 1:] if x.get("role") != "system"]
                break
        else:
            log = [m for m in self.conversation_history if m.get("role") != "system"]

        # The compressor counts only the non-system log, so give it thresholds
        # derived from the ACTUAL log size (in the library's token scheme) rather
        # than from the overhead-inclusive budget. The trigger in _context_compress
        # counts per-message overhead and tool_calls, which the library does not,
        # so a log that fits the library's t_max can still trip the trigger and
        # report "0 compressions". Forcing t_max below the real log size makes the
        # library actually fold an oldest prefix into the anchored summary.
        system_tokens = 0
        if system_prompt is not None:
            system_tokens = self._estimate_message_tokens({"role": "system", "content": system_prompt})
        log_budget = max(1, budget - system_tokens)

        token_counter = _AICommanderTokenCounter(self)
        log_tokens_lib = sum(token_counter.count_tokens(m.get("content", "")) for m in log)
        log_tokens_lib = max(1, log_tokens_lib)

        # Summary budget: modest, but leave room for a retained suffix.
        t_summary = max(20, min(budget // 20, log_budget // 3))
        t_summary = max(1, min(t_summary, log_budget - 1))
        # Retained target: keep the summary + suffix at a FRACTION of the log
        # budget (plus the summary) so several new turns fit before the next
        # compression. Retaining up to ~100% of the budget makes every following
        # prompt exceed the limit again and re-compress on each step, which is
        # wasteful. Headroom here keeps the agent running many steps before it
        # must summarize again. The library also counts only content tokens, so
        # keeping clear of the budget absorbs the per-message/tool_calls overhead
        # that the trigger counts and the library does not.
        t_retained = max(t_summary + 1, min(int(log_budget * 0.6),
                                            log_tokens_lib - 1))
        t_max = max(t_summary + 1, t_retained)

        cc = _ContextCompressorLib(
            summarizer=self._llm_summarize,
            t_max=t_max,
            t_retained=t_retained,
            t_summary=t_summary,
            tokenizer=token_counter,
        )
        for i, msg in enumerate(log):
            content = msg.get("content")
            if not isinstance(content, str):
                content = ""
            cc.add_message(content, role=msg.get("role", "user"),
                           metadata={"original": msg, "index": i})

        # Auto-compresses when over t_max; returns anchored summary + suffix.
        context = cc.get_current_context(auto_compress=True)

        new_history = []
        summary_text = None
        for m in context:
            original = m.metadata.get("original")
            if original is not None:
                # Preserve the full original message (tool_calls, tool_call_id,
                # role, content) so validation and the API stay consistent --
                # stripping them would make an assistant-with-tool_calls look
                # like a terminal text-only message and trigger a false
                # [SAFETY STOP].
                new_history.append(dict(original))
            elif m.metadata.get("type") == "summary":
                summary_text = m.content
            else:
                new_history.append({"role": m.role, "content": m.content})

        # Merge the anchored summary into the single leading system prompt.
        # OpenAI-compatible endpoints expect system messages at the very start
        # (and usually exactly one); emitting the summary as a second system
        # message is rejected with "System message must be at the beginning."
        if system_prompt is not None:
            if summary_text:
                system_prompt = system_prompt + "\n\n[Prior conversation summary]\n" + summary_text
            new_history.insert(0, {"role": "system", "content": system_prompt})
        elif summary_text:
            new_history.insert(0, {"role": "system",
                                   "content": "[Prior conversation summary]\n" + summary_text})

        # Re-insert the preserved FIRST user instruction (the original task)
        # verbatim, immediately after the system prompt, mirroring the "truncate"
        # algorithm. This guarantees the LLM always remembers its initial request
        # even when the rest of the oldest history was folded into the summary.
        if first_user is not None:
            new_history.insert(1, dict(first_user))

        # Ensure a user query exists. After aggressive compression the original
        # user request can be folded into the summary, leaving only system/tool/
        # assistant turns -- which some OpenAI-compatible endpoints reject with
        # "error code 400: no user query found in messages".
        if not any(m.get("role") == "user" for m in new_history):
            new_history.append({"role": "user", "content": "Continue with the current task."})
        self.conversation_history = new_history

        stats = cc.get_stats()
        final_tokens = sum(self._estimate_message_tokens(m) for m in self.conversation_history)
        self._log(f"[COMPRESS] context-compressor-llm: {stats['compression_count']} compression(s), "
                  f"tokens saved: {stats['total_tokens_saved']}. Final prompt length: {final_tokens} tokens.",
                  style="yellow")

    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return f"""You are an expert planning and execution assistant. Fulfill the user's request by breaking it into manageable steps and executing bash commands via the execute_bash tool (one command at a time, waiting for each result).

**Workflow:**
1. Analyze the request; if complex, break it into a numbered task list.
2. Restate your plan every ~5 steps and update the task list as you progress.
3. Execute one step with execute_bash, then evaluate the result.
4. If a step fails, adjust your approach and explain. If all steps are done and the goal is met, emit the exact phrase '{self.COMPLETION_MARKER}'.
5. Only emit '{self.COMPLETION_MARKER}' once you are certain every necessary action is complete and verified. Never say it prematurely.

**Runtime Constraints (execute_bash):**
- Output is capped at {self.max_output_bytes} bytes. If truncated, the literal sentinel `{self.OUTPUT_TRUNCATION_SENTINEL}` is appended. If you see it, you are missing data — do NOT assume success or failure. Re-run with output redirected to a file and read in chunks via `sed -n 'start,end p' file`, or use `head -c N`/`tail -c N`. Prefer targeted commands (grep, wc, stat) over dumping large outputs.
- Commands are killed after {self.command_timeout}s. For long operations use `nohup ... &` and check later, split into smaller steps, or set your own `timeout`.
- Commands run in a PTY; use non-interactive flags (`-y`, `--no-interactive`) where available.

**Command Results:**
- Never echo command execution metadata (e.g. "[EXECUTING]", "exit code X") into your response content; the tool system supplies that.
- Only describe your plan, your reasoning, and conclusions drawn from the REAL tool output. Never fabricate or guess command output — wait for the actual result.

**Internet Access:**
- Web search by printing JSON results to stdout: `python3 -c "import json; from ddgs import DDGS; print(json.dumps(list(DDGS().text('QUERY', max_results=10)), ensure_ascii=False, indent=2))"` (replace QUERY). Use wget/curl and the w3m browser to fetch page contents.

**Persistence Policy (CRITICAL):**
- Keep working until the task is genuinely complete. Do NOT stop early or produce a summary-only final response — take concrete action with tools.
- If you respond without calling tools, the system injects a continuation prompt; you must keep making progress.
- Troubleshoot errors and try alternative approaches until the objective is met.
- The ONLY way to finish is to emit '{self.COMPLETION_MARKER}' AFTER verifying all objectives via actual command execution and result inspection. Partial completion is not completion.
- Do not ask the user for clarification unless you have exhausted autonomous options.
- Before emitting '{self.COMPLETION_MARKER}', verify your last actions (check file contents, run tests, confirm services).

Think carefully; response quality is the highest priority. You have unlimited thinking tokens."""

    def execute_bash_command(self, command: str) -> Tuple[str, int]:
        """Execute a bash command in a PTY with timeout; return (output, exit_code)."""
        output_buffer = bytearray()
        pid, master_fd = pty.fork()

        if pid == 0:
            # Child: start a new process group so the parent's killpg() targets
            # the whole tree. Sandboxes may forbid setsid (EPERM); the parent's
            # killpg calls are guarded by try/except OSError.
            try:
                os.setsid()
            except OSError:
                pass
            # Force curl/wget (and tools honoring HTTP_USER_AGENT) to send the
            # same browser-like User-Agent as the OpenAI client.
            wrapper = (
                'export HTTP_USER_AGENT="%s"; '
                'curl() { command curl -A "$HTTP_USER_AGENT" "$@"; }; '
                'wget() { command wget --user-agent="$HTTP_USER_AGENT" "$@"; }; '
                '%s'
            ) % (USER_AGENT, command)
            try:
                os.execvp("/bin/sh", ["/bin/sh", "-c", wrapper])
            except OSError as e:
                os.write(2, f"Child process error (os.execvp failed): {e}\n".encode('utf-8'))
                os._exit(1)

        exit_code = -1
        timed_out = False

        master_fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, master_fl | os.O_NONBLOCK)

        # In TUI mode Textual owns the terminal; touching stdin flags or
        # select()ing on it would freeze the UI. Only monitor stdin in CLI mode.
        _monitor_stdin = not isinstance(self.sink, TUISink)
        stdin_fl = None
        if _monitor_stdin:
            stdin_fl = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, stdin_fl | os.O_NONBLOCK)

        # Kill the command if its raw output grows past 5x the return limit.
        output_kill_threshold = 5 * self.max_output_bytes

        start_time = time.time()
        try:
            while True:
                if self.stop_event.is_set():
                    self._log_error("[STOP] Agent shutdown requested. Terminating command.")
                    try:
                        os.killpg(pid, signal.SIGTERM)
                        time.sleep(0.5)
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        pass
                    break

                if time.time() - start_time > self.command_timeout:
                    self._log_error(f"[TIMEOUT] Command timed out after {self.command_timeout} seconds. Sending SIGTERM.")
                    os.killpg(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except OSError:
                        pass
                    timed_out = True
                    break

                _read_fds = [master_fd, sys.stdin.fileno()] if _monitor_stdin else [master_fd]
                rlist, _, _ = select.select(_read_fds, [], [], 0.1)

                if master_fd in rlist:
                    try:
                        data = os.read(master_fd, 1024)
                        if data:
                            output_buffer.extend(data)
                            # Stream output to the sink in real-time
                            try:
                                self.sink.emit("CMD_OUTPUT", {
                                    "text": data.decode('utf-8', errors='replace'),
                                    "command": command,
                                })
                            except Exception:
                                pass
                            if len(output_buffer) > output_kill_threshold:
                                self._log_error(f"[OUTPUT LIMIT] Command output exceeded {output_kill_threshold} bytes. Stopping command.")
                                os.killpg(pid, signal.SIGTERM)
                                time.sleep(0.3)
                                try:
                                    os.waitpid(pid, os.WNOHANG)
                                except OSError:
                                    pass
                                break
                        else:
                            break
                    except OSError:
                        break

                if _monitor_stdin and sys.stdin.fileno() in rlist:
                    try:
                        user_input = os.read(sys.stdin.fileno(), 1024)
                        if user_input:
                            os.write(master_fd, user_input)
                    except OSError:
                        pass

                try:
                    wpid, status = os.waitpid(pid, os.WNOHANG)
                    if wpid == pid:
                        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else \
                                    (os.WTERMSIG(status) + 128)
                        break
                except OSError:
                    break

        finally:
            if master_fd is not None:
                os.close(master_fd)

            if stdin_fl is not None:
                try:
                    fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, stdin_fl)
                except (OSError, AttributeError):
                    pass

            if exit_code == -1:
                # The loop exited without observing the child's status (e.g.
                # PTY EOF arrived first). The child is already dead, so reap it
                # with a blocking waitpid to obtain the TRUE exit status.
                try:
                    _, status = os.waitpid(pid, 0)
                    if os.WIFEXITED(status):
                        exit_code = os.WEXITSTATUS(status)
                    elif os.WIFSIGNALED(status):
                        exit_code = os.WTERMSIG(status) + 128
                    else:
                        exit_code = 1
                except OSError:
                    pass  # ESRCH/ECHILD: already reaped or not our child

        final_output = output_buffer.decode('utf-8', errors='replace')

        # Normalize line endings (strips ^M from PTY output).
        final_output = final_output.replace('\r\n', '\n').replace('\r', '\n')

        if len(final_output.encode('utf-8')) > self.max_output_bytes:
            final_output = final_output[:self.max_output_bytes] + "\n" + self.OUTPUT_TRUNCATION_SENTINEL

        self.sink.emit("CMD_COMPLETE", {"command": command, "exit_code": exit_code, "output": final_output})

        if timed_out:
            raise CommandTimeoutError(f"Command timed out after {self.command_timeout} seconds")

        return final_output, exit_code

    def get_user_confirmation(self, command: str) -> Tuple[bool, Optional[str]]:
        """Get user confirmation for command execution via the sink.

        Returns (approved, suggestion): suggestion is a non-empty steering
        string when the user rejected with guidance instead of a plain no.
        """
        if self.auto_approve:
            return True, None

        prompt = f"Approve command? (y/n/suggestion): {command}"
        response_input = self.sink.input(prompt).strip().lower()

        if response_input in ['y', 'yes', '']:
            return True, None
        elif response_input in ['n', 'no']:
            return False, None
        else:
            return False, response_input

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize messages before sending: no None content, valid roles, tool
        messages need tool_call_id, and no trailing assistant message without
        tool calls (a terminal state APIs reject; signaled by an empty list)."""
        validated = []
        for msg in messages:
            clean_msg = dict(msg)
            if clean_msg.get("content") is None:
                clean_msg["content"] = ""
            if clean_msg.get("role") not in ("system", "user", "assistant", "tool", "developer", "function"):
                continue
            if clean_msg.get("role") == "tool" and not clean_msg.get("tool_call_id"):
                continue
            if clean_msg.get("role") == "function" and not clean_msg.get("name"):
                continue
            validated.append(clean_msg)

        # APIs reject consecutive assistant messages; drop the older one.
        while len(validated) >= 2 and validated[-1].get("role") == "assistant" and validated[-2].get("role") == "assistant":
            validated.pop(-2)

        if validated and validated[-1].get("role") == "assistant" and not validated[-1].get("tool_calls"):
            return []

        return validated

    def call_llm_api(self, messages: List[Dict[str, str]], use_tools: bool = True,
                     stream_display: bool = True,
                     stream_label: Optional[str] = None) -> Dict[str, Any]:
        """Call the LLM API using the OpenAI client with streaming support.

        When ``stream_display`` is True (default) the streamed content is echoed
        to the agent output pane via ``LLM_STREAM``/``THINKING_STREAM`` events.
        Pass False for internal calls (e.g. the context-compressor summarizer) so
        their output is routed to the console output pane via ``CONSOLE_STREAM``
        instead of appearing as if the agent were speaking. ``stream_label``, when
        given, is prepended once at the start of the stream (used to tag internal
        output such as ``[SESSION SUMMARY]``).
        """
        messages = self._validate_messages(messages)

        # Empty list = terminal assistant state (see _validate_messages);
        # calling the API would produce consecutive assistant messages.
        if not messages:
            self._log(f"\n[SAFETY STOP] Conversation history validation returned empty - terminal assistant state detected. Ending interaction.")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"I have completed my response. No further actions are needed.\n\n{self.COMPLETION_MARKER}",
                        "tool_calls": None,
                        "reasoning_content": None
                    }
                }]
            }

        # Low temperature for tool use: high randomness yields malformed JSON
        # tool arguments. Creative text-only turns keep the high temperature.
        temperature = 0.2 if use_tools else 1.0

        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        # Ask the server to include usage stats in the final stream chunk so we
        # can correct the live estimate with exact token counts.
        request_params["stream_options"] = {"include_usage": True}
        if "gpt" not in self.model:  # OpenAI rejects these extra params
            request_params["max_tokens"] = self.max_tokens
            request_params['extra_body'] = {"chat_template_kwargs": {"enable_thinking": True}}

        if use_tools:
            request_params["tools"] = self.tool_schemas
            request_params["tool_choice"] = "auto"

        # Live input-token estimate before the call starts; the TUI shows it in
        # the status bar until the exact usage arrives with the final chunk.
        self.sink.emit("TOKEN_USAGE", {
            "input_tokens": self._estimate_input_tokens(messages),
            "output_tokens": 0,
            "estimate": True,
        })

        max_retries = 5
        wait_seconds = 60

        for attempt in range(1, max_retries + 1):
            try:
                collected_content = ""
                collected_thinking = ""
                collected_tool_calls = []
                in_thinking = False
                thinking_buffer = ""
                usage = None
                label_sent = False

                stream = self.client.chat.completions.create(**request_params)

                for chunk in stream:
                    # The final chunk carries exact usage when include_usage is set.
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if self.stop_event.is_set():
                        self._log("\n[STOP] Agent shutdown requested. Aborting LLM stream.")
                        return {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": f"Agent stopped by user.\n\n{self.COMPLETION_MARKER}",
                                    "tool_calls": None,
                                    "reasoning_content": None
                                }
                            }]
                        }
                    delta = chunk.choices[0].delta if chunk.choices else None

                    if delta:
                        # Reasoning tokens: 'reasoning_content' (DeepSeek/OpenAI)
                        # or 'thinking' (some other providers).
                        reasoning_chunk = None
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                            reasoning_chunk = delta.reasoning_content
                        elif hasattr(delta, 'thinking') and delta.thinking:
                            reasoning_chunk = delta.thinking

                        if self.show_thinking and reasoning_chunk:
                            collected_thinking += reasoning_chunk
                            thinking_buffer += reasoning_chunk
                            stream_kind = "THINKING_STREAM" if stream_display else "CONSOLE_STREAM"
                            self.sink.emit(stream_kind, {"text": reasoning_chunk})
                            in_thinking = True

                        if delta.content:
                            collected_content += delta.content
                            if self.show_thinking and in_thinking and thinking_buffer:
                                self._log("[THINKING COMPLETE]", style="yellow")
                                in_thinking = False
                            stream_kind = "LLM_STREAM" if stream_display else "CONSOLE_STREAM"
                            if stream_kind == "CONSOLE_STREAM" and stream_label and not label_sent:
                                label_sent = True
                                self.sink.emit(stream_kind, {"text": delta.content, "label": stream_label})
                            else:
                                self.sink.emit(stream_kind, {"text": delta.content})

                        if delta.tool_calls:
                            for tool_call_chunk in delta.tool_calls:
                                index = tool_call_chunk.index

                                while len(collected_tool_calls) <= index:
                                    collected_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })

                                current_tool_call = collected_tool_calls[index]

                                if tool_call_chunk.id:
                                    current_tool_call["id"] = tool_call_chunk.id

                                if tool_call_chunk.function:
                                    if tool_call_chunk.function.name:
                                        current_tool_call["function"]["name"] += tool_call_chunk.function.name
                                    if tool_call_chunk.function.arguments:
                                        current_tool_call["function"]["arguments"] += tool_call_chunk.function.arguments
                break
            except Exception as e:
                if attempt < max_retries:
                    self._log(f"[API ERROR] Connection error: {str(e)}. Retrying in {wait_seconds}s... (attempt {attempt}/{max_retries})")
                    time.sleep(wait_seconds)
                else:
                    raise

        self._log("")
        final_tool_calls = [tc for tc in collected_tool_calls if tc.get("id")]

        # The API call finished: report the exact token counts so the TUI can
        # replace the live estimate in the "context" status-bar counter.
        if usage is not None:
            # Prompt-cached tokens are served from cache (not re-processed) and
            # are typically not billed, so report them separately so the total
            # can reflect only the tokens actually processed/charged.
            #
            # Cached-token field names differ per provider:
            #   - OpenAI:      usage.prompt_tokens_details.cached_tokens
            #   - DeepSeek:    usage.prompt_cache_hit_tokens
            #   - Anthropic:   usage.cache_read_input_tokens
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            self.sink.emit("TOKEN_USAGE", {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "cached_tokens": cached,
                "estimate": False,
            })

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": collected_content if collected_content else None,
                    "tool_calls": final_tool_calls if final_tool_calls else None,
                    # Sent back as reasoning_content so reasoning models receive
                    # their prior reasoning on the next request.
                    "reasoning_content": collected_thinking if collected_thinking else None
                }
            }]
        }

    def process_llm_response(self, response: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Optional[str], Optional[str], List[Dict[str, Any]]]:
        """Extract content, tool calls, thinking, and malformed tool calls from
        an LLM response.

        Returns (content, tool_calls_info, first_tool_call_id, thinking,
        malformed_tool_calls). Each tool_calls_info entry has tool_call_id,
        function_name, function_arguments, and command (for execute_bash).
        Malformed entries have tool_call_id, function_name, raw_arguments,
        parse_error.
        """
        message = response["choices"][0]["message"]
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        thinking = message.get("reasoning_content") or message.get("thinking")

        tool_calls_info: List[Dict[str, Any]] = []
        first_tool_call_id = None
        malformed_tool_calls: List[Dict[str, Any]] = []

        if tool_calls:
            first_tool_call_id = tool_calls[0].get("id")
            for tool_call in tool_calls:
                tc_id = tool_call.get("id")
                function_name = tool_call.get("function", {}).get("name")
                arguments_str = tool_call.get("function", {}).get("arguments")

                command = None
                parsed_args = None
                parse_error = None

                if arguments_str:
                    try:
                        parsed_args = json.loads(arguments_str)
                    except json.JSONDecodeError as e:
                        parse_error = str(e)
                        self._log_error(f"Tool call arguments could not be parsed as JSON: {parse_error}")

                if parse_error is not None:
                    malformed_tool_calls.append({
                        "tool_call_id": tc_id,
                        "function_name": function_name,
                        "raw_arguments": arguments_str,
                        "parse_error": parse_error,
                    })
                    continue

                if function_name == "execute_bash" and parsed_args is not None:
                    command = parsed_args.get("command")

                tool_calls_info.append({
                    "tool_call_id": tc_id,
                    "function_name": function_name,
                    "function_arguments": arguments_str,
                    "command": command,
                })

        return content or "", tool_calls_info, first_tool_call_id, thinking, malformed_tool_calls

    def handle_function_call(self, function_info: Dict[str, Any]) -> str:
        """Produce a tool result for tool calls that carry no bash command."""
        function_name = function_info.get("name")
        arguments_str = function_info.get("arguments")

        if not arguments_str:
            return "No arguments provided for function call"

        try:
            json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return (f"ERROR: The JSON arguments provided for function '{function_name}' "
                    f"could not be parsed. Parse error: {e}. "
                    f"Please re-issue this tool call with valid, properly formatted JSON "
                    f"arguments (double-quoted strings, newlines escaped as \\n, "
                    f"no trailing commas, a single JSON object).")

        return f"Unknown function: {function_name}"

    def inject_suggestion(self, suggestion: str):
        """Queue a mid-run user suggestion; drained at the start of the next
        step and appended to the conversation as a user message."""
        if suggestion and suggestion.strip():
            self.suggestion_queue.put(suggestion.strip())

    def _drain_suggestions(self):
        """Append any queued user suggestions to the conversation history."""
        while not self.suggestion_queue.empty():
            try:
                s = self.suggestion_queue.get_nowait()
            except queue.Empty:
                break
            self.conversation_history.append({
                "role": "user",
                "content": f"[Suggestion from user] {s}"
            })
            self._log(f"[USER SUGGESTION] {s}")

    def run(self, user_request: str):
        """Main interactive loop"""
        if not self._started_banner_shown:
            self._started_banner_shown = True
            self.sink.emit("SYSTEM", {
                "text": "[AI-COMMANDER STARTED]",
                "api_base": self.base_url,
                "model": self.model,
                "auto_approve": self.auto_approve,
                "max_prompt_len": self.max_prompt_len,
                "compress_algorithm": self.compress_algorithm,
            })
        self._log(f"[USER REQUEST] {user_request}", style="yellow")
        self._log(f"{'='*60}")

        # With persist_history, refresh the system prompt in place and append
        # the new request so prior chat context carries over; otherwise reset.
        if self.persist_history and self.conversation_history:
            self.conversation_history[0] = {
                "role": "system",
                "content": self.get_system_prompt()
            }
            self.conversation_history.append({
                "role": "user",
                "content": user_request
            })
        else:
            self.conversation_history = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_request}
            ]

        step = 0

        try:
            while step < self.max_steps and not self.stop_event.is_set():

                if self.debug:
                    self._dump_conversation_history()

                step += 1
                self.sink.emit("STATUS_UPDATE", {"step": step, "max_steps": self.max_steps, "phase": "llm_call"})

                self._drain_suggestions()
                self._context_compress()

                response = self.call_llm_api(list(self.conversation_history))

                if not response.get("choices") or not response.get("choices", [{}])[0].get("message"):
                    self._log(f"[SAFETY STOP] API call returned no valid response. Ending interaction.")
                    break

                assistant_message = response["choices"][0]["message"]

                # Terminal safety response synthesized by call_llm_api.
                _content_raw = assistant_message.get("content") or ""
                if (_content_raw.strip().endswith(self.COMPLETION_MARKER) and
                    not assistant_message.get("tool_calls") and
                    not assistant_message.get("reasoning_content") and
                    len(_content_raw) < 200):
                    break

                # Don't store reasoning_content; it confuses APIs that don't
                # expect it in subsequent requests.
                clean_message = dict(assistant_message)
                clean_message.pop("reasoning_content", None)

                # Never create consecutive assistant messages.
                if self.conversation_history and self.conversation_history[-1].get("role") == "assistant":
                    self.conversation_history[-1] = clean_message
                else:
                    self.conversation_history.append(clean_message)
                content, tool_calls_info, first_tool_call_id, thinking, malformed_tool_calls = self.process_llm_response(response)

                # LLM content was already streamed via LLM_STREAM events; do
                # not re-print it here or it would appear twice.

                # Malformed tool calls: feed the parse errors back to the LLM
                # and let it re-issue corrected calls on the next step.
                if malformed_tool_calls:
                    self._log(f"[INFO] {len(malformed_tool_calls)} tool call(s) had malformed JSON arguments. Requesting correction from LLM.")
                    for mal in malformed_tool_calls:
                        err_msg = (
                            f"ERROR: The tool call with id '{mal['tool_call_id']}' (function: '{mal['function_name']}') "
                            f"contained malformed JSON arguments that could not be parsed.\n\n"
                            f"Parse error: {mal['parse_error']}\n\n"
                            f"Your function arguments MUST be a valid JSON object. Ensure:\n"
                            f"1. All strings are properly quoted with double quotes\n"
                            f"2. No unescaped newline characters inside strings (use \\n instead of literal newlines)\n"
                            f"3. No trailing commas\n"
                            f"4. The entire argument string is a single valid JSON object\n\n"
                            f"Please re-issue the tool call with corrected, valid JSON arguments. "
                            f"If you cannot produce valid JSON for this tool call, respond with a normal text message instead."
                        )
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": mal["tool_call_id"],
                            "content": err_msg,
                        })
                    self._log(f"[INFO] Correction messages appended. Continuing to next step for LLM to fix the tool calls.")
                    continue

                if tool_calls_info:
                    for tc_info in tool_calls_info:
                        if self.stop_event.is_set():
                            self._log("[STOP] Agent shutdown requested. Aborting tool execution.")
                            break
                        tool_call_id = tc_info["tool_call_id"]
                        command = tc_info.get("command")
                        function_name = tc_info.get("function_name")

                        if not command:
                            if tool_call_id and function_name:
                                result = self.handle_function_call({"name": function_name, "arguments": tc_info["function_arguments"]})
                                self.conversation_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": result
                                })
                            continue

                        self.sink.emit("CMD_EXEC", {"command": command, "tool_call_id": tool_call_id, "function_name": function_name})

                        approved, user_suggestion = self.get_user_confirmation(command)

                        if approved:
                            self._log(f"[Executing{' (AUTO)' if self.auto_approve else ''}] {command}", end="")
                            try:
                                output, exit_code = self.execute_bash_command(command)

                                if tool_call_id:
                                    self.conversation_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": f"Command executed with exit code {exit_code}.\nOutput:\n{output}"
                                    })

                            except CommandTimeoutError as e:
                                error_msg = str(e)
                                self._log_error(f"[TIMEOUT] {error_msg}")

                                if tool_call_id:
                                    self.conversation_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": error_msg
                                    })
                            except Exception as e:
                                error_msg = f"Command execution failed: {str(e)}"
                                self._log_error(error_msg)

                                if tool_call_id:
                                    self.conversation_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": error_msg
                                    })
                        else:
                            self._log(f"[INFO] Command not approved by user.")

                            if tool_call_id:
                                self.conversation_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": "Command execution skipped by user."
                                })

                                if user_suggestion:
                                    self.conversation_history.append({
                                        "role": "user",
                                        "content": f"I suggest you {user_suggestion}"
                                    })
                else:
                    _content_check = content if isinstance(content, str) else ""

                    if _content_check.strip().endswith(self.COMPLETION_MARKER) or self.COMPLETION_MARKER in _content_check:
                        break

                    # The assistant stopped with neither tool calls nor the
                    # completion marker: a premature stop. Inject a
                    # continuation message so it keeps working.
                    continuation_msg = (
                        f"Continue working on the task. You previously responded without using any tools. "
                        f"If the task is truly complete, you MUST emit the completion signal \"{self.COMPLETION_MARKER}\" "
                        f"on its own. Otherwise, continue by using the appropriate tools to make progress. "
                        f"Do NOT simply describe what needs to be done - take the next concrete step using "
                        f"execute_bash or other available tools. Remember: you must keep going until the "
                        f"task is fully finished, then emit \"{self.COMPLETION_MARKER}\"."
                    )
                    self.conversation_history.append({"role": "user", "content": continuation_msg})

                if self.COMPLETION_MARKER in content:
                    self._log(f"[{self.COMPLETION_MARKER} DETECTED - TASK COMPLETED SUCCESSFULLY]")
                    break

                if step >= self.max_steps:
                    self._log(f"[LIMIT REACHED] Maximum steps ({self.max_steps}) exceeded")

        finally:
            self.sink.emit("SHUTDOWN", {"reason": "agent_loop_terminated"})

        self.sink.close()


if _TEXTUAL_AVAILABLE:
    class ApprovalScreen(ModalScreen):
        """Blocking modal that asks the user to approve or reject a command.

        The agent thread is blocked in TUISink.input(); the decision callback
        resolves the sink's event and unblocks it. Rendered as a small
        centered dialog over a dimmed full-screen overlay.
        """

        CSS = """
        ApprovalScreen {
            align: center middle;
            background: $surface 50%;
        }
        #approval-dialog {
            width: 72;
            max-width: 85%;
            height: auto;
            max-height: 60%;
            border: thick $accent;
            background: $surface;
            padding: 1 2;
        }
        #approval-message {
            width: 100%;
            height: auto;
            padding: 0 1;
        }
        #approval-buttons {
            width: 100%;
            height: auto;
            align: center middle;
            padding-top: 1;
        }
        #approval-buttons Button {
            margin: 0 1;
        }
        """

        # Screen-level bindings fire regardless of widget focus.
        BINDINGS = [
            Binding("y", "approve", "Approve", show=False),
            Binding("n", "reject", "Reject", show=False),
            Binding("enter", "approve", "Approve", show=False),
            Binding("escape", "reject", "Reject", show=False),
        ]

        def __init__(self, command: str, on_decision):
            super().__init__()
            self.command = command
            self._on_decision = on_decision

        def compose(self) -> ComposeResult:
            with Vertical(id="approval-dialog"):
                yield Static(
                    "⚠ APPROVAL REQUIRED ⚠\n\n"
                    "The agent wants to execute the following command:\n\n"
                    f"  {self.command}\n\n"
                    "Press  Y  to approve and run it, or  N  to reject it.",
                    id="approval-message",
                )
                with Horizontal(id="approval-buttons"):
                    yield Button("Yes (Y)", id="approve-yes", variant="success")
                    yield Button("No (N)", id="approve-no", variant="error")

        def on_button_pressed(self, event: Button.Pressed):
            self._resolve(event.button.id == "approve-yes")

        def _resolve(self, approved: bool):
            self._on_decision(approved)
            self.dismiss()

        def action_approve(self):
            self._resolve(True)

        def action_reject(self):
            self._resolve(False)


def enable_sandbox() -> Tuple[bool, str]:
    """Enable a Linux OS-level sandbox via py-landlock.

    Writes are restricted to the current working directory (recursively),
    /tmp, /dev and /dev/pts; reads and execution are allowed anywhere and
    network access is preserved. /dev and /dev/pts (plus the IOCTL_DEV right,
    Landlock ABI v5) are required for pty allocation. strict=False makes
    everything best-effort on older kernels.

    Prints nothing; the caller routes the returned [SANDBOX] status message
    through the active sink. Returns (enabled, message).
    """
    if sys.platform != 'linux':
        return (False, "[SANDBOX] Not running on Linux; no OS-level sandbox applied")

    try:
        from py_landlock import Landlock, AccessFs
    except ImportError:
        return (False, "[SANDBOX] py-landlock not installed; running unsandboxed")

    try:
        ll = Landlock(strict=False) \
            .allow_read("/") \
            .allow_execute("/") \
            .allow_write(os.getcwd(), "/tmp", "/dev", "/dev/pts") \
            .allow_all_network() \
            .allow_all_scope()
        ll.add_path_rule(
            "/dev", "/dev/pts", "/dev/tty",
            access=AccessFs.IOCTL_DEV,
        )
        ll.apply()
        return (True,
                "[SANDBOX] OS-level Landlock sandbox enabled "
                "(writes restricted to current directory, /tmp and /dev)")
    except Exception as e:
        return (False, f"[SANDBOX] Failed to enable Landlock sandbox: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI-Commander - a ralph-loop AI agent")
    parser.add_argument("--api-base", required=True, help="API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Auto-approve command execution (for testing)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Hide thinking tokens from output")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Command timeout in seconds (default: 120)")
    parser.add_argument("--max-prompt-len", type=int, default=80000,
                        help="Maximum prompt length in tokens (default: 80000)")
    parser.add_argument("--max-output-bytes", type=int, default=10240,
                        help="Maximum output bytes to return from commands (default: 10240)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum number of agent loop steps before stopping (default: 500)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (dump conversation history on truncation)")
    parser.add_argument("--nogui", action="store_true",
                        help="Run in direct CLI mode without TUI (original behaviour)")
    parser.add_argument("--disable-sandbox", action="store_true",
                        help="Disable the OS-level Landlock sandbox (Linux only)")
    parser.add_argument("--compress-alg", default="context-compressor-llm",
                        help="Context-compression algorithm for shrinking the "
                             "conversation history when it exceeds the prompt "
                             "budget. Options: 'context-compressor-llm' (default) "
                             "uses the anchored-summary incremental compressor "
                             "from the context-compressor-llm package (LLM "
                             "summarizer; falls back to truncate if the package "
                             "is missing); 'truncate' drops the oldest messages "
                             "and condenses tool outputs.")
    parser.add_argument("request", nargs="*", help="Task request")

    args = parser.parse_args()

    # The TUI can start with no request (user types one in the prompt input);
    # --nogui is a one-shot runner and still requires one.
    if args.nogui and not args.request:
        print("[ERROR] Please provide a task request", file=sys.stderr)
        sys.exit(1)

    # Sandbox must be enabled before the agent thread starts so the Landlock
    # domain covers the whole process. The status message is routed through
    # the active sink below.
    if args.disable_sandbox:
        args.sandbox_enabled = False
        args.sandbox_msg = "[SANDBOX] Sandbox disabled via --disable-sandbox"
    else:
        args.sandbox_enabled, args.sandbox_msg = enable_sandbox()

    if args.nogui:
        sink = ConsoleSink()
        sink.emit("LOG", {
            "text": args.sandbox_msg,
            "style": "green" if args.sandbox_enabled else "red",
        })
        commander = AICommander(
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            auto_approve=args.auto_approve,
            show_thinking=not args.no_thinking,
            command_timeout=args.timeout,
            max_prompt_len=args.max_prompt_len,
            max_output_bytes=args.max_output_bytes,
            debug=args.debug,
            sink=sink,
            max_steps=args.max_steps,
            compress_algorithm=args.compress_alg
        )
        try:
            user_request = " ".join(args.request)
            commander.run(user_request)
        except KeyboardInterrupt:
            commander._log("[INTERRUPTED] AI-Commander stopped by user")
            sys.exit(0)
        except Exception:
            commander._log_error(f"\n[FATAL ERROR] An unexpected error occurred:")
            commander._log_error(traceback.format_exc())
            sys.exit(1)
        return

    # TUI mode
    if not _TEXTUAL_AVAILABLE:
        print("Textual library is not installed. Install it with: pip install textual")
        print("Alternatively, run with --nogui for direct CLI mode.")
        sys.exit(1)

    from textual.app import App
    from textual.widgets import RichLog, Input, ContentSwitcher
    from textual.widget import Widget
    from textual import events
    from rich.text import Text as RichText
    import pyte

    class PromptInput(Input):
        """Textual Input with readline-style history navigation.

        Up/Down (and Ctrl+P/Ctrl+N) walk the shared readline history while
        preserving the in-progress line (readline semantics); Ctrl+R opens a
        reverse incremental search modal.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 1-based readline history index currently shown, or None when the
            # cursor sits at the bottom (editing a fresh line).
            self._hist_pos: Optional[int] = None
            # The in-progress line stashed when the user first navigates up.
            self._pending: str = ""

        def on_key(self, event):
            key = event.key
            if key in ("up", "ctrl+p"):
                self._hist_previous()
                event.stop()
            elif key in ("down", "ctrl+n"):
                self._hist_next()
                event.stop()
            elif key == "ctrl+r":
                self._open_reverse_search()
                event.stop()

        def reset_history_nav(self):
            """Return to the bottom of history after a prompt is submitted."""
            self._hist_pos = None
            self._pending = ""

        def _hist_previous(self):
            length = _rl_history_length()
            if length == 0:
                return
            if self._hist_pos is None:
                self._pending = self.value
                self._hist_pos = length
            elif self._hist_pos > 1:
                self._hist_pos -= 1
            self._show_hist_item()

        def _hist_next(self):
            if self._hist_pos is None:
                return
            length = _rl_history_length()
            if self._hist_pos < length:
                self._hist_pos += 1
                self._show_hist_item()
            else:
                self._hist_pos = None
                self.value = self._pending
                self.cursor_position = len(self.value)

        def _show_hist_item(self):
            text = _rl_get_history_item(self._hist_pos)
            self.value = text
            self.cursor_position = len(text)

        def _open_reverse_search(self):
            def on_result(match: Optional[str]):
                if match is not None:
                    self.value = match
                    self.cursor_position = len(match)
                try:
                    self.focus()
                except Exception:
                    pass
            try:
                self.app.push_screen(ReverseSearchScreen(on_result))
            except Exception:
                pass

    class ReverseSearchScreen(ModalScreen):
        """Modal incremental reverse search over the readline history.

        Typing filters history (newest-first) for a case-insensitive substring;
        Ctrl+R cycles to the next older match; Enter accepts the current match
        back into the prompt; Esc/Ctrl+G cancels (restoring the prior text).
        """

        BINDINGS = [
            Binding("ctrl+r", "next_match", "Next match", show=False),
            Binding("enter", "accept", "Accept", show=False),
            Binding("escape", "cancel", "Cancel", show=False),
            Binding("ctrl+g", "cancel", "Cancel", show=False),
        ]

        CSS = """
        ReverseSearchScreen {
            align: center top;
            /* Transparent so the underlying TUI stays visible behind the
            search dialog (ModalScreen's default opaque background would
            otherwise blank the whole screen). */
            background: transparent;
        }
        #reverse-search-dialog {
            width: 80;
            height: auto;
            background: #0b2f5e;
            border: round #06989a;
            padding: 1 2;
            margin-top: 3;
        }
        #reverse-search-input {
            width: 1fr;
        }
        #reverse-search-status {
            color: #f2f2f2;
            margin-top: 1;
        }
        """

        def __init__(self, on_result):
            super().__init__()
            self._on_result = on_result
            # Matches as (1-based_index, text) newest-first; _match_pos is the
            # currently highlighted match.
            self._matches: list = []
            self._match_pos = 0

        def compose(self) -> ComposeResult:
            with Vertical(id="reverse-search-dialog"):
                yield Static("(reverse-i-search)`': ", id="reverse-search-label", markup=False)
                yield Input(id="reverse-search-input", placeholder="Search history")
                yield Static("", id="reverse-search-status", markup=False)

        def on_mount(self):
            try:
                self.query_one("#reverse-search-input").focus()
            except Exception:
                pass
            self._refresh()

        def on_input_changed(self, event):
            if getattr(event.input, "id", None) == "reverse-search-input":
                self._refresh()

        def on_input_submitted(self, event):
            # The search Input has focus, so Enter is consumed by the widget
            # and never reaches the screen-level "enter" binding. Accept (or
            # cancel when empty) here so Enter closes the modal.
            if getattr(event.input, "id", None) == "reverse-search-input":
                if self._matches:
                    self.action_accept()
                else:
                    self.action_cancel()

        def _refresh(self):
            query = ""
            try:
                query = self.query_one("#reverse-search-input").value
            except Exception:
                pass
            self._matches = []
            self._match_pos = 0
            length = _rl_history_length()
            if query:
                q = query.lower()
                for i in range(length, 0, -1):
                    text = _rl_get_history_item(i)
                    if q in text.lower():
                        self._matches.append((i, text))
            label = f"(reverse-i-search)`{query}': "
            if query and not self._matches:
                label += "failing"
            try:
                self.query_one("#reverse-search-label", Static).update(label)
            except Exception:
                pass
            self._update_status()

        def _update_status(self):
            try:
                status = self.query_one("#reverse-search-status", Static)
            except Exception:
                return
            if self._matches:
                _, text = self._matches[self._match_pos]
                status.update(
                    f"{text}\n\n"
                    f"[{self._match_pos + 1}/{len(self._matches)}] "
                    "Ctrl+R next, Enter accept, Esc cancel"
                )
            else:
                status.update("No matches. Esc to cancel.")

        def action_next_match(self):
            if self._matches:
                self._match_pos = (self._match_pos + 1) % len(self._matches)
                self._update_status()

        def action_accept(self):
            if self._matches:
                _, text = self._matches[self._match_pos]
                self._on_result(text)
            else:
                self._on_result(None)
            self.dismiss()

        def action_cancel(self):
            self._on_result(None)
            self.dismiss()

    class ShellWidget(Widget, can_focus=True):
        """An interactive shell embedded in a Textual widget.

        Spawns the user's shell in a pty, emulates VT100 output with pyte,
        and forwards all keyboard input while focused (Tab autocomplete,
        Ctrl+C, arrows, etc.).
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._master_fd: Optional[int] = None
            self._pid: Optional[int] = None
            self._screen = None  # pyte.Screen
            self._stream = None  # pyte.ByteStream
            self._cols = 80
            self._rows = 24
            self._history: list = []
            self._last_render = ""

        def check_consume_key(self, key: str, character: Optional[str]) -> bool:
            # Return False so keys reach _on_key (which forwards them to the
            # pty and stops propagation) while App-level bindings still work.
            return False

        def _spawn(self) -> None:
            """Fork a shell in a pty and set up the pyte emulator."""
            shell = os.environ.get("SHELL", "") or "/bin/sh"
            if not shell or not os.path.exists(shell):
                shell = "/bin/sh"
            try:
                pid, master_fd = pty.fork()
            except OSError:
                return
            if pid == 0:
                os.environ["TERM"] = "xterm-256color"
                try:
                    os.execv(shell, [shell])
                except Exception:
                    try:
                        os.execv("/bin/sh", ["/bin/sh"])
                    except Exception:
                        os._exit(127)
            self._pid = pid
            self._master_fd = master_fd
            try:
                fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
                fcntl.fcntl(master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            except OSError:
                pass
            self._resize_pty()
            self._screen = pyte.Screen(self._cols, self._rows)
            self._stream = pyte.ByteStream(self._screen)
            self._stream.attach(self._screen)

        def _resize_pty(self) -> None:
            """Tell the kernel the pty window size (so full-screen apps work)."""
            if self._master_fd is None:
                return
            try:
                winsize = struct.pack("HHHH", self._rows, self._cols, 0, 0)
                fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)
            except OSError:
                pass

        def on_mount(self) -> None:
            # Don't spawn here: the widget starts hidden inside the
            # ContentSwitcher (size 0x0), which would give the pty a broken
            # window size. Spawn lazily in on_resize() instead.
            self.set_interval(0.05, self._poll)

        def on_resize(self, event: events.Resize) -> None:
            # Ignore degenerate (hidden) sizes reported by the ContentSwitcher.
            if event.size.width < 4 or event.size.height < 3:
                return
            self._cols = max(2, event.size.width)
            self._rows = max(2, event.size.height)
            if self._master_fd is None and self._pid is None:
                self._spawn()
            if self._screen is not None:
                try:
                    self._screen.resize(self._rows, self._cols)
                except Exception:
                    pass
            self._resize_pty()
            self.refresh()

        def _poll(self) -> None:
            """Read pending pty output, feed it to pyte, and refresh."""
            if self._master_fd is None:
                return
            try:
                data = os.read(self._master_fd, 65536)
            except (BlockingIOError, OSError):
                data = b""
            if data:
                try:
                    self._stream.feed(data)
                except Exception:
                    pass
                self._sync_history()
                self.refresh()
            # Reap the child if it exited; resetting both fields lets the next
            # on_resize spawn a fresh shell.
            if self._pid is not None:
                try:
                    wpid, _status = os.waitpid(self._pid, os.WNOHANG)
                except ChildProcessError:
                    wpid = self._pid
                if wpid == self._pid:
                    try:
                        os.close(self._master_fd)
                    except OSError:
                        pass
                    self._master_fd = None
                    self._pid = None

        def _sync_history(self) -> None:
            """Copy the pyte screen contents into self._history for rendering."""
            if self._screen is None:
                return
            try:
                self._history = [line.rstrip() for line in self._screen.display]
            except Exception:
                pass

        def render(self) -> RichText:
            """Render the last *height* history lines with a block cursor at
            the pyte cursor position."""
            if not self._history:
                return RichText("")
            height = self.size.height
            if height <= 0:
                height = 24
            visible = self._history[-height:] if height else []
            # Pad short buffers so the prompt sits at the bottom.
            if len(visible) < height:
                visible = [""] * (height - len(visible)) + visible

            # Map the absolute pyte cursor row into the visible window.
            cursor_row = -1
            cursor_col = 0
            if self._screen is not None:
                try:
                    cy = self._screen.cursor.y
                    cx = self._screen.cursor.x
                    pad_top = height - len(self._history[-height:])
                    row_in_hist = cy - (len(self._history) - height)
                    cursor_row = pad_top + row_in_hist
                    cursor_col = cx
                except Exception:
                    cursor_row = -1

            text = RichText()
            for i, line in enumerate(visible):
                if i:
                    text.append("\n")
                if i == cursor_row and 0 <= cursor_col:
                    # History lines were rstrip()ed; pad up to the cursor
                    # column so the block advances over whitespace.
                    if len(line) < cursor_col:
                        line = line + " " * (cursor_col - len(line))
                    text.append(line[:cursor_col], style="default on #000000")
                    text.append(line[cursor_col:cursor_col + 1] or " ", style="reverse on #000000")
                    text.append(line[cursor_col + 1:], style="default on #000000")
                else:
                    text.append(line, style="default on #000000")
            return text

        async def _on_key(self, event: events.Key) -> None:
            """Forward every key press to the shell and stop propagation."""
            data = self._key_to_bytes(event)
            if data and self._master_fd is not None:
                try:
                    os.write(self._master_fd, data)
                except OSError:
                    pass
            event.stop()

        def _key_to_bytes(self, event: events.Key) -> bytes:
            """Convert a Textual Key event into the byte sequence for the pty."""
            key = event.key
            if key.startswith("ctrl+"):
                ch = key[5:]
                if len(ch) == 1 and ch.isalpha():
                    return bytes([ord(ch.lower()) & 0x1f])
            if event.is_printable and event.character:
                return event.character.encode("utf-8")
            special = {
                "enter": b"\r",
                "backspace": b"\x7f",
                "tab": b"\t",
                "escape": b"\x1b",
                "up": b"\x1b[A",
                "down": b"\x1b[B",
                "right": b"\x1b[C",
                "left": b"\x1b[D",
                "home": b"\x1b[H",
                "end": b"\x1b[F",
                "delete": b"\x1b[3~",
                "pageup": b"\x1b[5~",
                "pagedown": b"\x1b[6~",
            }
            if key in special:
                return special[key]
            if key.startswith("f") and key[1:].isdigit():
                n = int(key[1:])
                if 1 <= n <= 4:
                    return b"\x1bOP" if n == 1 else b"\x1bO" + bytes([ord("P") + n - 1])
                if 5 <= n <= 12:
                    return b"\x1b[" + str(n + 9).encode() + b"~"
            return b""

    class RightPanel(Vertical):
        """Right-hand tab panel (Console Output + Shell) built from a button
        strip and a ContentSwitcher (TabbedContent breaks this layout)."""

        BINDINGS = [
            Binding("tab", "switch_tab", "Switch panel", show=False),
        ]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._switcher = None

        def compose(self) -> ComposeResult:
            with Horizontal(id="tab-strip"):
                yield Button("Console Output", id="btn-console")
                yield Button("Shell", id="btn-shell")
            with ContentSwitcher(initial="console-log") as switcher:
                self._switcher = switcher
                yield RichLog(id="console-log", auto_scroll=True, wrap=True, max_lines=1000)
                yield ShellWidget(id="shell-widget")

        def on_mount(self) -> None:
            self._update_tab_buttons()

        def _update_tab_buttons(self, active=None) -> None:
            """Reflect the active pane on the tab buttons via the active-tab
            class. Callers that just switched panes pass *active* explicitly
            because ContentSwitcher applies `current` asynchronously."""
            if active is None:
                active = self._switcher.current
            console_active = active == "console-log"

            def _apply():
                try:
                    bc = self.query_one("#btn-console", Button)
                    bs = self.query_one("#btn-shell", Button)
                    if console_active:
                        bc.add_class("active-tab")
                        bs.remove_class("active-tab")
                    else:
                        bs.add_class("active-tab")
                        bc.remove_class("active-tab")
                except Exception:
                    pass
            # Defer so the class change isn't clobbered by the switcher/focus
            # updates that immediately follow a tab switch.
            self.call_after_refresh(_apply)

        def action_switch_tab(self) -> None:
            """Cycle between the Console Output and Shell tabs."""
            new = "console-log" if self._switcher.current == "shell-widget" else "shell-widget"
            self._switcher.current = new
            self._update_tab_buttons(new)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            # Stop the event so CommanderApp's handler doesn't double-handle it.
            event.stop()
            if event.button.id == "btn-console":
                self._switcher.current = "console-log"
                self._update_tab_buttons("console-log")
            elif event.button.id == "btn-shell":
                self._switcher.current = "shell-widget"
                self._update_tab_buttons("shell-widget")
                # Focus the shell after Textual's own click->focus logic runs.
                def _focus_shell():
                    try:
                        self.query_one("#shell-widget").focus()
                    except Exception:
                        pass
                self.set_timer(0.01, _focus_shell)

    class CommanderApp(App):
        """Textual TUI for AI-Commander.

        Norton-Commander style layout: agent output (left), console/shell tabs
        (right), prompt input (bottom), status footer. The agent runs in a
        background thread and pushes events to a queue; the main thread polls
        it and renders to widgets. The agent thread never touches widgets.
        """

        CSS = """
        Screen {
            background: #12488b;
        }
        #warning-banner {
            display: none;
            background: #06989a;
            color: #000000;
            padding: 1 2;
            border: solid #0000A0;
            margin-bottom: 1;
        }
        #prompt-container {
            background: #000000;
            border: none;
            height: 5;
            margin-top: 1;
        }
        #prompt-container:focus-within {
            border: none;
        }
        #prompt-marker {
            color: #06989a;
            width: 1;
            height: 3;
        }
        #prompt-input {
            background: transparent;
            border: none;
            color: #f2f2f2;
            padding: 0 1;
            width: 1fr;
            height: 3;
        }
        #prompt-input:focus {
            border: none;
        }
        #prompt-input > .input--cursor {
            background: #06989a;
            color: #12488b;
        }
        #status-bar {
            background: #06989a;
            color: #000000;
            height: 1;
            padding: 0 1;
        }
        #panels {
            height: 1fr;
            width: 1fr;
        }
        RichLog {
            border: solid #f2f2f2;
            padding: 0;
            height: 1fr;
        }
        #console-log {
            background: #12488b;
            color: #f2f2f2;
            border: solid #f2f2f2;
            width: 1fr;
        }
        #agent-log {
            background: #12488b;
            color: #f2f2f2;
            border: solid #f2f2f2;
            width: 1fr;
        }
        /* Explicit min-heights keep the switcher body from collapsing to
        zero in real terminals (tab strip rendered, body blank). */
        #right-panel {
            height: 1fr;
            width: 1fr;
            min-height: 5;
            layout: vertical;
        }
        #right-panel ContentSwitcher {
            height: 1fr;
            width: 1fr;
            min-height: 3;
        }
        #tab-strip {
            height: 3;
            background: #0b2f5e;
        }
        /* Flat tab strip: suppress Textual's default Button focus chrome. */
        #tab-strip Button {
            width: 1fr;
            height: 3;
            background: #0b2f5e;
            color: #f2f2f2;
            border: none;
            border-top: none;
            border-bottom: none;
        }
        #tab-strip Button:focus {
            background: #0b2f5e;
            color: #f2f2f2;
            text-style: none;
        }
        /* Target button IDs directly (higher specificity than Textual's
        internal component classes) so the teal highlight wins. */
        #btn-console.active-tab, #btn-shell.active-tab,
        #btn-console.active-tab:focus, #btn-shell.active-tab:focus {
            background: #06989a;
            color: #000000;
            text-style: bold;
        }
        #shell-widget {
            background: #000000;
            color: #f2f2f2;
            height: 1fr;
            width: 1fr;
            padding: 0;
        }
        """

        # Keep mouse support for buttons/tabs but disable Textual's own text
        # selection so the terminal's native select/copy works.
        ALLOW_SELECT = False

        def __init__(self, args):
            super().__init__()
            self.args = args
            self.event_queue: queue.Queue = queue.Queue()
            self.stop_event = threading.Event()
            self.agent_thread: Optional[threading.Thread] = None
            self.auto_approve = args.auto_approve
            self.step_count = 0
            self.max_steps = 500
            self.tokens_per_second = 0.0
            # Rolling ~1s window for the live tok/s display in the status bar.
            self._tok_window_start = time.monotonic()
            self._tok_window_tokens = 0
            # Cumulative session token total. Only exact server-reported usage
            # is added (see _apply_token_usage), so it never over-counts.
            self.context_tokens = 0
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_name = args.model
            self.api_base = args.api_base
            self.sink: TUISink = None
            self.agent = None
            self.pending_approval = None  # dict with 'command' awaiting decision
            # Per-widget buffers so streamed tokens are written as whole lines
            # instead of one RichLog line per token.
            self._stream_buffer: Dict[str, str] = {}
            # Stream labels (e.g. "[THINKING]") shown once per stream.
            self._stream_prefix_done: set = set()
            # Show the [SANDBOX] status once per session, not on every re-run.
            self._sandbox_notified = False

        def compose(self) -> ComposeResult:
            yield Static(
                "⚠ AGENT WILL EXECUTE REAL BASH COMMANDS ON THIS SYSTEM ⚠\n"
                "Auto-approve is OFF. Every command requires /approve or /reject.\n"
                "Type /autoapprove (or /aa) to toggle auto-approve for this session.\n"
                "───────────────────────────────────────────────────────────────────",
                id="warning-banner",
                markup=False
            )
            agent = RichLog(id="agent-log", auto_scroll=True, wrap=True, max_lines=20000)
            agent.border_title = "Agent Output"
            yield Horizontal(agent, RightPanel(id="right-panel"), id="panels")
            yield Horizontal(
                Static(">", id="prompt-marker", markup=False),
                PromptInput(id="prompt-input", placeholder="Type a prompt or /command, then Enter"),
                id="prompt-container",
            )
            yield Static(id="status-bar")

        def on_mount(self):
            self.title = "AI-Commander TUI"
            self.sub_title = f"Model: {self.model_name} | Session: {self.session_id}"
            # Load any persisted prompt history into readline so Up/Down and
            # Ctrl+R see prior sessions' prompts.
            _rl_load_history()
            self.update_status_bar()
            # The banner is informational only: show it for a grace period,
            # then auto-hide (it is shown again while an approval is pending).
            # It is hidden by default in CSS so a remount never flashes it.
            self._show_warning_banner()
            self.set_timer(20.0, self._auto_hide_warning_banner)
            try:
                self.query_one("#right-panel ContentSwitcher").current = "console-log"
            except Exception:
                pass
            try:
                self.query_one("#prompt-input").focus()
            except Exception:
                pass
            self.set_interval(0.05, self._drain_queue)

            # Auto-start the agent with a CLI-supplied request, mirroring
            # --nogui. Deferred so widgets exist before the thread starts;
            # the request is consumed so a remount doesn't re-launch.
            cli_request = " ".join(self.args.request) if self.args.request else ""
            self.args.request = []
            if cli_request.strip():
                self._cli_request = cli_request
                self.set_timer(0.5, self._autostart_once)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Focus the shell when its tab button is clicked."""
            if event.button.id == "btn-shell":
                try:
                    switcher = self.query_one("#right-panel ContentSwitcher")
                    if switcher.current == "shell-widget":
                        self.query_one("#shell-widget").focus()
                except Exception:
                    pass

        def update_status_bar(self):
            sb = self.query_one("#status-bar")
            ap_status = "ON" if self.auto_approve else "OFF"
            sandbox_on = getattr(self.args, "sandbox_enabled", False)
            sandbox_txt = (
                "[green]Sandbox: on[/green]"
                if sandbox_on
                else "[red]Sandbox: off[/red]"
            )
            # "Context" = the current context window: the token estimate of the
            # live conversation history (the prompt currently being sent). The
            # cumulative session total (sum of every call's input+output) is
            # shown separately as "Total Tokens".
            history = self.agent.conversation_history if self.agent else []
            context_window = (
                sum(self.agent._estimate_message_tokens(m) for m in history)
                if self.agent
                else 0
            )
            sb.update(
                f"Model: {self.model_name} | "
                f"Step: {self.step_count}/{self.max_steps} | "
                f"Context: {context_window} | "
                f"Total Tokens: {self.context_tokens} | "
                f"tok/s: {self.tokens_per_second:.1f} | "
                f"Session: {self.session_id} | "
                f"Auto-approve: {ap_status} | "
                f"Pending approval: {'YES' if self.pending_approval else 'no'} | "
                f"{sandbox_txt}"
            )

        def _drain_queue(self):
            """Poll the event queue and dispatch events to widgets (main
            thread only; the agent thread only writes to the queue)."""
            # Cap events per tick to avoid starving the UI.
            for _ in range(200):
                if self.event_queue.empty():
                    break
                try:
                    event = self.event_queue.get_nowait()
                except queue.Empty:
                    break
                self._dispatch_event(event)

            if self.agent_thread and not self.agent_thread.is_alive():
                self._handle_agent_exit()

            self.update_status_bar()

        def _dispatch_event(self, event: dict):
            kind = event.get("kind", "")
            payload = event.get("payload", {})

            if kind == "SYSTEM":
                self._write_agent("[SYSTEM] " + payload.get('text',''), style="bold blue")
                self._write_agent(f"  API: {payload.get('api_base','')}", style="blue")
                self._write_agent(f"  Model: {payload.get('model','')}", style="blue")
                self._write_agent(f"  Auto-approve: {payload.get('auto_approve', False)}", style="blue")
                self._write_agent(f"  Max prompt len: {payload.get('max_prompt_len', 0)}", style="blue")
                self._write_agent(f"  Context compression: {payload.get('compress_algorithm', 'truncate')}", style="blue")
                self._write_agent("  [!] Agent starting. Commands will require approval unless /autoapprove is toggled.", style="yellow")
            elif kind == "LOG":
                self._write_agent(payload.get("text", ""), end=payload.get("end", "\n"), style=payload.get("style", ""))
            elif kind == "ERROR":
                self._write_agent(f"[ERROR] {payload.get('text', '')}", style="bold red")
            elif kind == "LLM_STREAM":
                self._write_agent(payload.get("text", ""), streaming=True, style="cyan")
                self._track_token()
            elif kind == "THINKING_STREAM":
                self._write_agent(payload.get("text", ""), streaming=True, prefix="[THINKING] ", style="yellow")
                self._track_token()
            elif kind == "CONSOLE_STREAM":
                self._write_console(payload.get("text", ""), style="yellow", streaming=True)
            elif kind == "CMD_EXEC":
                cmd = payload.get("command", "")
                tcid = payload.get("tool_call_id", "")
                fn = payload.get("function_name", "")
                self._write_console(f"[COMMAND REQUESTED] {cmd}", style="bold yellow")
                self._write_console(f"  tool_call_id: {tcid}  function: {fn}", style="yellow")
                self._write_console(f"  ⚠ This command will execute when approved.", style="yellow")
                self.pending_approval = {"command": cmd, "tool_call_id": tcid}
                self._refresh_approval_prompt()
            elif kind == "APPROVAL_REQUEST":
                cmd = payload.get("command", "")
                self.pending_approval = {"command": cmd}
                self._write_console(f"\n[APPROVAL REQUIRED] Command: {cmd}", style="bold yellow")
                self._write_console(f"  Respond to the dialog to approve or reject.", style="yellow")
                self._refresh_approval_prompt()
                # The agent thread is blocked on the sink's approval event;
                # the modal resolves it via the on_decision callback.
                self._push_approval_dialog(cmd)
            elif kind == "CMD_OUTPUT":
                self._write_console(payload.get("text", ""), style="green")
            elif kind == "CMD_COMPLETE":
                self._write_console(f"[COMMAND COMPLETE] {payload.get('command', '')}", style="bold green")
                self._write_console(f"  Exit code: {payload.get('exit_code', -1)}", style="green")
                # Raw output was already written live via CMD_OUTPUT chunks.
                self.pending_approval = None
                self._refresh_approval_prompt()
            elif kind == "STATUS_UPDATE":
                self.step_count = payload.get("step", 0)
                self.max_steps = payload.get("max_steps", 500)
                self._decay_token_rate()
            elif kind == "TOKEN_USAGE":
                self._apply_token_usage(payload)
            elif kind == "SHUTDOWN":
                self._write_agent("[READY] Agent loop terminated, waiting for next command.", style="bold green")
                self._handle_agent_exit()

            self.update_status_bar()

        def _apply_token_usage(self, payload: dict):
            """Accumulate exact server-reported token usage.

            The agent emits an estimate=True event (input-token guess) before
            each call and an estimate=False event with exact numbers when the
            call finishes. Only the exact numbers count toward the cumulative
            total; the estimates and live-streamed chunks are ignored so the
            counter never over-counts.
            """
            inp = payload.get("input_tokens", 0) or 0
            out = payload.get("output_tokens", 0) or 0
            if payload.get("estimate"):
                return
            # Prompt-cached tokens are served from cache and typically not
            # billed, so exclude them from the cumulative total: only the
            # non-cached (newly processed) tokens count.
            cached = payload.get("cached_tokens", 0) or 0
            self.context_tokens += max(0, inp - cached) + out

        def _track_token(self):
            """Update the rolling tok/s rate for a streamed chunk (does not
            affect the cumulative token total, which comes from exact usage)."""
            now = time.monotonic()
            self._tok_window_tokens += 1
            if now - self._tok_window_start >= 1.0:
                elapsed = now - self._tok_window_start
                if elapsed > 0:
                    self.tokens_per_second = self._tok_window_tokens / elapsed
                self._tok_window_start = now
                self._tok_window_tokens = 0

        def _decay_token_rate(self):
            """Reset the rate to 0 once streaming has been idle for 2s."""
            if self._tok_window_tokens == 0 and time.monotonic() - self._tok_window_start >= 2.0:
                self.tokens_per_second = 0.0

        def _sync_autoscroll(self, widget):
            """Pin the view to the bottom only when the user is already there,
            so new messages don't yank the scroll bar away from history being
            read. If the user has scrolled up, auto-scroll is disabled."""
            try:
                widget.auto_scroll = widget.scroll_y >= widget.max_scroll_y
            except Exception:
                pass

        def _anchor_scroll(self, widget, start_line_before):
            """Keep the viewport anchored to the same content after a write.

            When the log hits its max_lines cap, the RichLog trims the oldest
            lines off the top (incrementing its internal ``_start_line``). If the
            user is scrolled up reading history, that trimming shifts the content
            up and would yank the scroll bar to different, newer text. This
            compensates by scrolling up by the same number of trimmed lines so
            the lines being read stay put.
            """
            try:
                trimmed = widget._start_line - start_line_before
                if trimmed > 0 and not widget.auto_scroll:
                    widget.scroll_to(
                        y=max(0, widget.scroll_y - trimmed),
                        animate=False,
                    )
            except Exception:
                pass

        def _write_agent(self, text: str, streaming: bool = False, prefix: str = "", style: str = "", end: str = "\n"):
            """Write text to the agent output RichLog.

            Streaming text is buffered and flushed as complete lines so tokens
            flow inline. The prefix label is shown once per stream, not per
            chunk. *style* is a Rich style string.
            """
            try:
                from rich.text import Text
                widget = self.query_one("#agent-log")
                self._sync_autoscroll(widget)
                clean = _strip_ansi(text)
                # Hide the internal completion marker from the panel; the
                # agent loop still uses it to detect task completion.
                clean = clean.replace("TASKCOMPLETE", "")
                if not clean:
                    return
                start_line_before = widget._start_line
                buf_key = "#agent-log"
                if streaming:
                    if prefix and buf_key not in self._stream_prefix_done:
                        self._stream_prefix_done.add(buf_key)
                        widget.write(Text(_strip_ansi(prefix).strip(), style=style or "yellow"))

                    # Flush complete lines; keep the partial line buffered.
                    buf = self._stream_buffer.get(buf_key, "") + clean
                    parts = buf.split("\n")
                    self._stream_buffer[buf_key] = parts[-1]
                    for part in parts[:-1]:
                        widget.write(Text(part, style=style) if style else Text(part))
                else:
                    # A non-streaming write ends any active stream: flush the
                    # buffer and reset stream-tracking state.
                    pending = self._stream_buffer.pop(buf_key, "")
                    if pending:
                        widget.write(Text(pending, style=style) if style else Text(pending))
                    self._stream_prefix_done.discard(buf_key)

                    full = _strip_ansi(prefix) + clean
                    widget.write(Text(full, style=style) if style else Text(full))
                # If the log hit its max_lines cap and trimmed old lines off the
                # top, re-anchor the viewport so the lines being read stay put.
                self._anchor_scroll(widget, start_line_before)
            except Exception:
                pass

        def _write_console(self, text: str, style: str = "", streaming: bool = False):
            """Write text to the console output RichLog.

            When *streaming* is True, tokens are buffered and flushed as complete
            lines so the partial line stays inline (used by CONSOLE_STREAM).
            """
            try:
                from rich.text import Text
                widget = self.query_one("#console-log")
                self._sync_autoscroll(widget)
                clean = _strip_ansi(text)
                if not clean:
                    return
                start_line_before = widget._start_line
                buf_key = "#console-log"
                if streaming:
                    buf = self._stream_buffer.get(buf_key, "") + clean
                    parts = buf.split("\n")
                    self._stream_buffer[buf_key] = parts[-1]
                    for part in parts[:-1]:
                        widget.write(Text(part, style=style) if style else Text(part))
                else:
                    pending = self._stream_buffer.pop(buf_key, "")
                    if pending:
                        widget.write(Text(pending, style=style) if style else Text(pending))
                    widget.write(Text(clean, style=style) if style else Text(clean))
                # If the log hit its max_lines cap and trimmed old lines off the
                # top, re-anchor the viewport so the lines being read stay put.
                self._anchor_scroll(widget, start_line_before)
            except Exception:
                pass

        def _auto_hide_warning_banner(self):
            """Hide the banner after the grace period unless an approval is
            pending (the blocked-agent state must stay visible)."""
            if not self.pending_approval:
                self._hide_warning_banner()

        def _hide_warning_banner(self):
            try:
                self.query_one("#warning-banner").display = False
            except Exception:
                pass

        def _show_warning_banner(self):
            # With auto-approve on, commands never require approval, so the
            # banner is irrelevant; keep it hidden entirely.
            if self.auto_approve:
                return
            try:
                self.query_one("#warning-banner").display = True
            except Exception:
                pass

        def _refresh_approval_prompt(self):
            """Update the warning banner to show pending approval status."""
            try:
                banner = self.query_one("#warning-banner")
                if self.pending_approval:
                    banner.update(
                        "⚠ PENDING APPROVAL ⚠\n"
                        f"  Command: {self.pending_approval.get('command','')}\n"
                        "  Type /approve to execute, /reject to deny, or /suggest <text> for guidance\n"
                        "  Agent is BLOCKED waiting for your decision."
                    )
                    self._show_warning_banner()
                else:
                    self._hide_warning_banner()
            except Exception:
                pass

        def _push_approval_dialog(self, command: str):
            """Push the blocking Y/N approval modal."""
            def on_decision(approved: bool):
                if self.sink:
                    self.sink.resolve_approval(approved)
                self.pending_approval = None
                self._refresh_approval_prompt()

            self.push_screen(ApprovalScreen(command, on_decision))

        def _handle_agent_exit(self):
            """Called when the agent thread finishes (normally or via stop)."""
            self.stop_event.set()
            self.agent_thread = None
            # Keep self.agent so its conversation_history persists across commands.
            self.pending_approval = None
            # Stay open so the user can review output.
            self._refresh_approval_prompt()
            self.update_status_bar()

        def _autostart_once(self):
            """One-shot timer callback launching the agent with the CLI
            request; returning False cancels the timer."""
            try:
                prompt = getattr(self, "_cli_request", "")
                if prompt:
                    # The initial CLI-supplied prompt is a real task, so make
                    # it recallable via Up/Down and Ctrl+R like typed prompts.
                    _rl_add_history(prompt)
                    _rl_save_history()
                    self._start_agent(prompt)
            except Exception as exc:
                self._write_agent(f"[ERROR] Failed to auto-start from CLI request: {exc}")
            finally:
                return False

        def _start_agent(self, prompt: str):
            """Start the agent thread with the given prompt. If already
            running, queue the text as a mid-run steering suggestion."""
            if self.agent_thread and self.agent_thread.is_alive():
                if self.agent:
                    self.agent.inject_suggestion(prompt)
                    self._write_agent(f"[SUGGESTION QUEUED] Steering agent: {prompt}")
                else:
                    self._write_agent("[ERROR] Agent is running but not available for suggestions.")
                return
            self.stop_event.clear()
            self.sink = TUISink(self.event_queue, self.stop_event)
            self.sink.auto_approve = self.auto_approve
            if not self._sandbox_notified:
                self.sink.emit("LOG", {
                    "text": self.args.sandbox_msg,
                    "style": "green" if self.args.sandbox_enabled else "red",
                })
                self._sandbox_notified = True
            if self.agent is None:
                # persist_history lets later commands reuse this instance and
                # its conversation history.
                self.agent = AICommander(
                    api_base=self.api_base,
                    model=self.model_name,
                    api_key=self.args.api_key,
                    auto_approve=self.auto_approve,
                    show_thinking=not self.args.no_thinking,
                    command_timeout=self.args.timeout,
                    max_prompt_len=self.args.max_prompt_len,
                    max_output_bytes=self.args.max_output_bytes,
                    debug=self.args.debug,
                    sink=self.sink,
                    persist_history=True,
                    max_steps=self.args.max_steps,
                    compress_algorithm=self.args.compress_alg
                )
            else:
                # Reuse the agent (and its history); re-sync mutable state in
                # case the user toggled /aa since the last run.
                self.agent.persist_history = True
                self.agent.stop_event.clear()
                self.agent.sink = self.sink
                self.agent.auto_approve = self.auto_approve
            self.agent_thread = threading.Thread(
                target=self.agent.run,
                args=(prompt,),
                daemon=True
            )
            self.agent_thread.start()

        def _stop_agent(self):
            """Stop the agent thread cleanly."""
            if self.agent:
                self.agent.stop_event.set()
            self.stop_event.set()
            if self.agent_thread and self.agent_thread.is_alive():
                self._write_agent("[STOP] Signal sent. Waiting for agent to exit (up to 5s)...")
                self.agent_thread.join(timeout=5)
            self.pending_approval = None
            self._refresh_approval_prompt()

        def on_input_submitted(self, event):
            """Handle Enter key in the prompt input field."""
            text = event.value.strip()
            try:
                prompt_input = self.query_one("#prompt-input")
                prompt_input.value = ""
                # Return to the bottom of history for the next prompt.
                prompt_input.reset_history_nav()
            except Exception:
                pass

            if not text:
                return

            # Only plain prompts go into history; slash commands are UI
            # controls, not tasks worth recalling.
            if not text.startswith("/"):
                _rl_add_history(text)
                _rl_save_history()

            if text.startswith("/"):
                self._handle_slash_command(text)
                return

            self._start_agent(text)

        def _handle_slash_command(self, cmd: str):
            """Handle slash commands typed in the prompt panel."""
            parts = cmd.lower().split(" ", 1)
            command = parts[0]
            arg = parts[1] if len(parts) > 1 else ""

            if command in ("/autoapprove", "/aa"):
                self.auto_approve = not self.auto_approve
                if self.sink:
                    self.sink.auto_approve = self.auto_approve
                # The gate lives in the agent (get_user_confirmation), so it
                # must be kept in sync with the UI toggle.
                if self.agent:
                    self.agent.auto_approve = self.auto_approve
                self._write_agent(
                    f"[AUTO-APPROVE TOGGLED] auto_approve is now "
                    f"{'ON (commands will run without asking)' if self.auto_approve else 'OFF (commands will require /approve)'}"
                )
                if self.auto_approve:
                    self._write_agent(
                        "  ⚠ WARNING: All commands will execute automatically without approval.\n"
                        "  ⚠ Ensure you are monitoring the console panel and can /stop if needed."
                    )
                self.update_status_bar()

            elif command == "/approve":
                if not self.pending_approval:
                    self._write_agent("[ERROR] No pending approval to approve.")
                    return
                if self.sink and self.sink._approval_event:
                    self.sink.resolve_approval(approved=True)
                    self._write_console(f"[APPROVED] {self.pending_approval.get('command','')}")
                else:
                    self._write_agent("[ERROR] No active approval gate to resolve.")
                self.pending_approval = None
                self._refresh_approval_prompt()

            elif command == "/reject":
                if not self.pending_approval:
                    self._write_agent("[ERROR] No pending approval to reject.")
                    return
                if self.sink and self.sink._approval_event:
                    self.sink.resolve_approval(approved=False)
                    self._write_console(f"[REJECTED] {self.pending_approval.get('command','')}")
                else:
                    self._write_agent("[ERROR] No active approval gate to resolve.")
                self.pending_approval = None
                self._refresh_approval_prompt()

            elif command == "/suggest":
                if not self.pending_approval:
                    self._write_agent("[ERROR] No pending approval to suggest for.")
                    return
                if self.sink and self.sink._approval_event:
                    self.sink.resolve_approval(approved=False, suggestion=arg)
                    self._write_console(f"[REJECTED WITH SUGGESTION] {self.pending_approval.get('command','')}")
                else:
                    self._write_agent("[ERROR] No active approval gate to resolve.")
                self.pending_approval = None
                self._refresh_approval_prompt()

            elif command in ("/clear", "/new"):
                # Clear the conversation history and start a fresh session,
                # reusing the same agent instance (like /new).
                if self.agent_thread and self.agent_thread.is_alive():
                    self._stop_agent()
                if self.agent:
                    self.agent.conversation_history = []
                    self.agent.suggestion_queue = queue.Queue()
                # Clear the on-screen output panels so old messages disappear.
                for widget_id in ("#agent-log", "#console-log"):
                    try:
                        self.query_one(widget_id).clear()
                    except Exception:
                        pass
                # Reset the cumulative token total so it doesn't accumulate
                # across /clear and /new.
                self.context_tokens = 0
                self.step_count = 0
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._write_agent("[CLEAR] New conversation started. Ready for your next prompt.")
                self.update_status_bar()

            elif command == "/stop":
                self._write_agent("[STOP] Stopping agent...")
                self._stop_agent()

            elif command in ("/quit", "/exit"):
                self._write_agent("[QUIT] Exiting agent...")
                self.shutdown()

            else:
                self._write_agent(f"[UNKNOWN COMMAND] {cmd}")
                self._write_agent("Available: /autoapprove /aa /approve /reject /suggest /clear /new /stop /quit /exit")

        def key_press(self, event):
            """Ctrl+C / Escape: stop the agent and quit."""
            if event.key in ("ctrl+c", "escape"):
                self._stop_agent()
                self.shutdown()

        def shutdown(self):
            """Clean shutdown: stop agent, close sink, exit app."""
            self._stop_agent()
            # Persist prompt history so it survives this session.
            _rl_save_history()
            if self.sink:
                self.sink.close()
            try:
                self.query_one("#prompt-input").remove()
            except Exception:
                pass
            self.exit()

    app = CommanderApp(args)
    try:
        app.run()
    except KeyboardInterrupt:
        app._stop_agent()
        sys.exit(0)
    except Exception as e:
        print(f"[FATAL ERROR] TUI crashed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
