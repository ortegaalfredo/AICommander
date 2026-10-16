#!/usr/bin/env python3
"""AI-Commander: a ralph-loop AI agent that gives an LLM shell access through
function calling.

One agent core (AICommander) drives two front-ends through an EventSink: the
default Textual TUI (TUISink) and a plain CLI (--nogui, ConsoleSink).
"""
from __future__ import annotations

__version__ = "2.0.0"

import argparse
import asyncio
import atexit
import base64
import bisect
import codecs
import contextlib
import json
import os
import queue
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from importlib.util import find_spec
from typing import Any

import openai
from openai import OpenAI

_IS_WINDOWS = os.name == "nt"
if not _IS_WINDOWS:  # the PTY command runner and the embedded shell are POSIX-only
    import fcntl
    import pty
    import termios

try:  # optional: anchored-summary context compression
    from context_compressor import ContextCompressor as _ContextCompressorLib
    _CONTEXT_COMPRESSOR_LLM_AVAILABLE = True
except ImportError:
    _ContextCompressorLib = None
    _CONTEXT_COMPRESSOR_LLM_AVAILABLE = False

try:  # optional: the TUI (--nogui runs without it)
    from rich.cells import cell_len
    from rich.segment import Segment
    from rich.style import Style
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.geometry import Offset, Size
    from textual.message import Message
    from textual.screen import ModalScreen
    from textual.scroll_view import ScrollView
    from textual.selection import Selection
    from textual.strip import Strip
    from textual.widget import Widget
    from textual.widgets import Button, ContentSwitcher, Input, Static, TextArea
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

try:  # optional: the embedded interactive shell tab
    import pyte
except ImportError:
    pyte = None

try:
    _DDGS_AVAILABLE = find_spec("ddgs") is not None
except (ImportError, ValueError):
    _DDGS_AVAILABLE = False

DEFAULT_MAX_PROMPT_LEN = 80000
# Browser-like User-Agent for API calls and for curl/wget run by the agent, so
# Cloudflare-fronted endpoints do not block the requests as bot traffic.
USER_AGENT = "Mozilla/5.0 (compatible; OpenAI-Client/1.0)"
_SUMMARY_MARKER = "\n\n[Prior conversation summary]\n"
_CONTINUE_PROMPT = "Continue with the current task."
_CONTINUATION_PREFIX = "Continue working on the task. You previously responded without using any tools."
_VALID_ROLES = ("system", "user", "assistant", "tool", "developer", "function")
# Agent commands run on a PTY nobody reads interactively: a pager (git log,
# man, systemctl) would wait for a keypress until the command times out.
_COMMAND_ENV = {"PAGER": "cat", "GIT_PAGER": "cat", "MANPAGER": "cat",
                "SYSTEMD_PAGER": "", "HTTP_USER_AGENT": USER_AGENT}
_SHELL_PRELUDE = ('curl() { command curl -A "$HTTP_USER_AGENT" "$@"; }; '
                  'wget() { command wget --user-agent="$HTTP_USER_AGENT" "$@"; }; ')

# Windows: PowerShell must speak UTF-8 on the redirected pipe (it would use the
# OEM code page otherwise), child Pythons too, and the exit code of the last
# native command must propagate like it did under cmd /c.
_PS_UTF8_PREFIX = (
    "try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch {}\n"
    "try { $OutputEncoding = [Text.UTF8Encoding]::new($false) } catch {}\n"
    "$env:PYTHONIOENCODING = 'utf-8'\n"
    "$env:PYTHONUTF8 = '1'\n"
)
_PS_UTF8_SUFFIX = "\nif ($LASTEXITCODE -ne $null) { exit $LASTEXITCODE }\n"


def _powershell_argv(command: str) -> list:
    """powershell.exe argv for a command; -EncodedCommand passes it verbatim."""
    script = _PS_UTF8_PREFIX + command + _PS_UTF8_SUFFIX
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


def _check_openai_version() -> None:
    """Exit with a clear message when the installed openai SDK is older than 2.0."""
    try:
        major = int(openai.__version__.split(".")[0])
    except (AttributeError, ValueError):
        return
    if major < 2:
        sys.exit(f"ERROR: AI-Commander requires openai>=2.0.0 but found {openai.__version__}. "
                 "Upgrade with: pip install --upgrade 'openai>=2.0.0'")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on", "y")


class colors:
    """ANSI codes used by ConsoleSink."""
    RED, GREEN, YELLOW, BLUE = "\033[91m", "\033[92m", "\033[93m", "\033[94m"
    MAGENTA, CYAN, BOLD, DIM, END = "\033[95m", "\033[96m", "\033[1m", "\033[2m", "\033[0m"


# CSI (incl. private "?" modes), OSC (BEL or ST terminated), DCS/PM/APC strings
# and two-byte escapes.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)"
                      r"|[PX^_][^\x1b]*\x1b\\|[ -/]*[0-~])")
# An escape sequence cut off at the end of a chunk (completed by the next one).
_ESC_TAIL_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*|\][^\x07\x1b]*|[ -/]*)?\Z")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)


def _apply_cr(line: str) -> str:
    """Resolve carriage returns like a terminal: later text overwrites the line."""
    out = ""
    for seg in line.split("\r"):
        out = seg + out[len(seg):]
    return out


def _clean_command_output(text: str) -> str:
    """Make raw PTY output model-friendly: no escape codes, progress-bar
    rewrites collapsed to their final state, no stray control characters."""
    text = _strip_ansi(text).replace("\r\n", "\n")
    text = re.sub(".\x08", "", text)  # overstrike bold/underline (man pages)
    if "\r" in text:
        text = "\n".join(_apply_cr(line) if "\r" in line else line for line in text.split("\n"))
    return _CTRL_RE.sub("", text)


def _content_text(content) -> str:
    """Message content as text; list-of-parts content is flattened, other shapes give ''."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p if isinstance(p, str) else p.get("text", "") for p in content
                         if isinstance(p, str) or (isinstance(p, dict) and isinstance(p.get("text"), str)))
    return ""


def _is_continuation(msg) -> bool:
    return (isinstance(msg, dict) and msg.get("role") == "user"
            and isinstance(msg.get("content"), str) and msg["content"].startswith(_CONTINUATION_PREFIX))


def _is_empty_assistant(msg) -> bool:
    return (msg.get("role") == "assistant" and not msg.get("tool_calls")
            and not _content_text(msg.get("content")).strip())


def _exit_code(status: int) -> int:
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code  # killed by signal N -> 128+N, like a shell


def _safe_cwd() -> str:
    try:
        return os.getcwd()
    except OSError:
        return ""


def _describe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    text = str(exc) or type(exc).__name__
    return f"HTTP {status}: {text}" if status else text


def _is_retryable(exc: Exception) -> bool:
    """Transient failures worth retrying (network, rate limit, server errors)."""
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 425, 429) or exc.status_code >= 500
    if isinstance(exc, (openai.APIConnectionError, openai.APIError, ConnectionError, TimeoutError)):
        return True
    return type(exc).__module__.split(".")[0] in ("httpx", "httpcore")


def _is_param_rejection(exc: Exception) -> bool:
    """A 4xx complaining about the optional stream_options/extra_body extensions."""
    if getattr(exc, "status_code", 400) not in (400, 422):
        return False
    text = str(exc).lower()
    return any(k in text for k in ("stream_options", "extra_body", "chat_template_kwargs",
                                   "enable_thinking", "unrecognized", "unexpected keyword",
                                   "extra fields", "extra inputs"))


def _is_context_overflow(exc: Exception) -> bool:
    if getattr(exc, "status_code", 400) not in (400, 413, 422):
        return False
    text = str(exc).lower()
    return any(k in text for k in ("context length", "context_length", "maximum context",
                                   "context window", "too many tokens", "prompt is too long",
                                   "exceed context", "exceeds the context", "context size", "reduce the length"))


# --- Clipboard ---------------------------------------------------------------
# The TUI copies through OSC 52 (Textual) and an in-app clipboard. Terminals
# without OSC 52 (VTE: GNOME Terminal, Tilix...) need the desktop clipboard set
# directly: clipboard tools when installed, otherwise X11 through libX11.

def _clipboard_commands(read: bool) -> list[list[str]]:
    cmds = []
    if os.environ.get("WAYLAND_DISPLAY"):
        cmds.append(["wl-paste", "--no-newline"] if read else ["wl-copy"])
    if os.environ.get("DISPLAY"):
        cmds.append(["xclip", "-selection", "clipboard", "-o"] if read else ["xclip", "-selection", "clipboard"])
        cmds.append(["xsel", "--clipboard", "--output"] if read else ["xsel", "--clipboard", "--input"])
    if sys.platform == "darwin":
        cmds.append(["pbpaste"] if read else ["pbcopy"])
    if _IS_WINDOWS or os.environ.get("WSL_DISTRO_NAME"):
        cmds.append(["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"] if read else ["clip.exe"])
    return [c for c in cmds if shutil.which(c[0])]


def _x11_clipboard(data: bytes | None = None, selections: tuple = ("CLIPBOARD", "PRIMARY"),
                   timeout: float = 2.0) -> str | None:
    """The X11 clipboard through libX11 (Xwayland bridges it on Wayland desktops).

    Without *data*: return the text of the first selection, or None. With
    *data*: own the selections, write "1" to stdout, then answer paste requests
    until other programs have taken every selection over (blocking, like xclip).
    It always runs in a helper interpreter (see _x11_helper): Xlib may exit()
    on fatal errors, and the owner must outlive the app, so it is self-contained.
    """
    import ctypes
    import ctypes.util
    import os
    import select
    import sys
    import time
    library = ctypes.util.find_library("X11")
    if not library or not os.environ.get("DISPLAY"):
        return None
    x, ul, vp, cint = ctypes.CDLL(library), ctypes.c_ulong, ctypes.c_void_p, ctypes.c_int
    for name, restype, argtypes in (
            ("XOpenDisplay", vp, [ctypes.c_char_p]), ("XDefaultRootWindow", ul, [vp]),
            ("XCreateSimpleWindow", ul, [vp, ul, cint, cint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ul, ul]),
            ("XInternAtom", ul, [vp, ctypes.c_char_p, cint]), ("XSetSelectionOwner", cint, [vp, ul, ul, ul]),
            ("XGetSelectionOwner", ul, [vp, ul]), ("XConvertSelection", cint, [vp, ul, ul, ul, ul, ul]),
            ("XChangeProperty", cint, [vp, ul, ul, ul, cint, cint, vp, cint]),
            ("XSendEvent", cint, [vp, ul, cint, ctypes.c_long, vp]), ("XNextEvent", cint, [vp, vp]),
            ("XPending", cint, [vp]), ("XConnectionNumber", cint, [vp]), ("XFlush", cint, [vp]),
            ("XFree", cint, [vp]),
            ("XGetWindowProperty", cint, [vp, ul, ul, ctypes.c_long, ctypes.c_long, cint, ul, ctypes.POINTER(ul),
                                          ctypes.POINTER(cint), ctypes.POINTER(ul), ctypes.POINTER(ul),
                                          ctypes.POINTER(vp)])):
        getattr(x, name).restype, getattr(x, name).argtypes = restype, argtypes
    ignore_errors = ctypes.CFUNCTYPE(cint, vp, vp)(lambda display, error: 0)  # e.g. a requestor vanished
    x.XSetErrorHandler(ignore_errors)
    display = x.XOpenDisplay(None)
    if not display:
        return None
    window = x.XCreateSimpleWindow(display, x.XDefaultRootWindow(display), 0, 0, 1, 1, 0, 0, 0)

    def atom(name):
        return x.XInternAtom(display, name.encode(), 0)
    utf8, targets, wanted = atom("UTF8_STRING"), atom("TARGETS"), [atom(name) for name in selections]
    event = (ul * 24)()  # an XEvent: past type/serial/send_event/display, the fields are longs
    if data is None:  # read
        prop = atom("AIC_CLIPBOARD")
        x.XConvertSelection(display, wanted[0], utf8, prop, window, 0)
        x.XFlush(display)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not x.XPending(display):
                select.select([x.XConnectionNumber(display)], [], [], max(0.0, deadline - time.monotonic()))
                continue
            x.XNextEvent(display, event)
            if ctypes.c_int.from_buffer(event).value != 31:  # SelectionNotify
                continue
            if not event[7]:  # no owner, or it cannot provide text
                return None
            kind, fmt, count, rest, buf = ul(), cint(), ul(), ul(), vp()
            x.XGetWindowProperty(display, window, prop, 0, 1 << 24, 1, 0, ctypes.byref(kind), ctypes.byref(fmt),
                                 ctypes.byref(count), ctypes.byref(rest), ctypes.byref(buf))
            try:  # an INCR (chunked) transfer of a huge clipboard is not supported
                if fmt.value != 8 or not buf.value or kind.value == atom("INCR"):
                    return None
                return ctypes.string_at(buf.value, count.value).decode("utf-8", "replace")
            finally:
                if buf.value:
                    x.XFree(buf)
        return None
    owned = set()
    for selection in wanted:
        x.XSetSelectionOwner(display, selection, window, 0)
        if x.XGetSelectionOwner(display, selection) == window:
            owned.add(selection)
    sys.stdout.write("1" if owned else "0")
    sys.stdout.flush()
    latin1 = data.decode("utf-8", "replace").encode("latin-1", "replace")
    formats = {atom("STRING"): (31, latin1), atom("TEXT"): (utf8, data), utf8: (utf8, data),
               atom("text/plain"): (utf8, data), atom("text/plain;charset=utf-8"): (utf8, data)}
    offered = (ul * (len(formats) + 1))(targets, *formats)
    while owned:
        x.XNextEvent(display, event)
        kind = ctypes.c_int.from_buffer(event).value
        if kind == 29:  # SelectionClear: another program owns this selection now
            owned.discard(event[5])
        elif kind == 30:  # SelectionRequest: owner, requestor, selection, target, property, time
            requestor, selection, target, prop, when = event[5], event[6], event[7], event[8] or event[7], event[9]
            if target == targets:
                x.XChangeProperty(display, requestor, prop, 4, 32, 0, offered, len(offered))  # XA_ATOM list
            elif target in formats:
                kind_atom, payload = formats[target]
                x.XChangeProperty(display, requestor, prop, kind_atom, 8, 0, payload, len(payload))
            else:
                prop = 0  # refuse other targets
            reply = (ul * 24)()
            ctypes.c_int.from_buffer(reply).value = 31  # SelectionNotify
            reply[3:9] = [display, requestor, selection, target, prop, when]
            x.XSendEvent(display, requestor, 0, 0, reply)
            x.XFlush(display)
    return None


_CLIPBOARD_LOCK = threading.Lock()


def _x11_helper(tail: str) -> list[str] | None:
    """argv running _x11_clipboard's own source plus *tail* in a fresh interpreter."""
    if _IS_WINDOWS or sys.platform == "darwin" or not os.environ.get("DISPLAY"):
        return None
    import inspect
    try:
        source = inspect.getsource(_x11_clipboard)
    except (OSError, TypeError):
        return None
    return [sys.executable, "-c", f"from __future__ import annotations\nimport sys\n{source}\n{tail}"]


def _x11_copy(text: str, selections: tuple = ("CLIPBOARD", "PRIMARY")) -> bool:
    argv = _x11_helper(f"_x11_clipboard(sys.stdin.buffer.read(), {selections!r})")
    if not argv:
        return False
    with _CLIPBOARD_LOCK:  # one owner at a time, so copies land in order
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
            ready = select.select([proc.stdout], [], [], 5)[0]
            ok = bool(ready) and proc.stdout.read(1) == b"1"
            proc.stdout.close()
            return ok
        except (OSError, ValueError):
            return False


def _x11_paste(selection: str = "CLIPBOARD") -> str | None:
    argv = _x11_helper(f"t = _x11_clipboard(None, ({selection!r},))\n"
                       f"sys.stdout.buffer.write(t.encode()) if t is not None else sys.exit(1)")
    if not argv:
        return None
    try:
        result = subprocess.run(argv, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.decode("utf-8", "replace") if result.returncode == 0 else None


def clipboard_write_native(text: str) -> bool:
    for cmd in _clipboard_commands(read=False):
        try:  # DEVNULL: xclip forks a server that would otherwise hold our pipes open
            subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3, check=True)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return _x11_copy(text)


def clipboard_read_native() -> str | None:
    for cmd in _clipboard_commands(read=True):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3, check=True)
            return result.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
        except (OSError, subprocess.SubprocessError):
            continue
    return _x11_paste()


# --- Prompt history ----------------------------------------------------------

class PromptHistory:
    """Prompt history persisted in ~/.aic_history, one entry per line with
    backslashes and newlines escaped (so multi-line prompts survive).

    The append descriptor is opened before the Landlock sandbox is applied, so
    new prompts are still saved although the sandbox blocks writes to $HOME.
    """

    def __init__(self, path: str | None = None, maxlen: int = 1000):
        self.path = path
        self.maxlen = maxlen
        self.items: list[str] = []
        self._fd: int | None = None

    @staticmethod
    def _escape(line: str) -> str:
        return line.replace("\\", "\\\\").replace("\n", "\\n")

    @staticmethod
    def _unescape(line: str) -> str:
        return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), line)

    def load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                raw = [ln.rstrip("\n") for ln in f if ln.strip() and ln.strip() != "_HiStOrY_V2_"]
        except OSError:
            raw = []
        self.items = [self._unescape(ln) for ln in raw[-self.maxlen:]]
        if len(raw) > 2 * self.maxlen:  # compact the file now, before the sandbox applies
            with contextlib.suppress(OSError), open(self.path, "w", encoding="utf-8") as f:
                f.writelines(self._escape(item) + "\n" for item in self.items)
        with contextlib.suppress(OSError):
            self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            os.set_inheritable(self._fd, False)

    def add(self, line: str) -> None:
        if not line.strip() or (self.items and self.items[-1] == line):
            return
        self.items.append(line)
        del self.items[:-self.maxlen]
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.write(self._fd, (self._escape(line) + "\n").encode("utf-8"))

    def close(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None


# --- Session persistence -----------------------------------------------------
# The conversation is mirrored to a hand-editable JSON file after every step, so
# a crash loses at most the in-flight turn. Each process holds an flock on
# "<file>.lock"; a second instance in the same directory takes .aicsession2, ...
_SESSION_BASENAME = ".aicsession"
_SESSION_LOCK_SUFFIX = ".lock"
_SESSION_VERSION = 1
_SESSION_LOCK_FDS: list[int] = []  # kept open (and locked) for the process lifetime


def _acquire_instance_lock(session_path: str) -> int | None:
    """Non-blocking exclusive lock on ``<session_path>.lock``; None if held elsewhere."""
    try:
        fd = os.open(session_path + _SESSION_LOCK_SUFFIX, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    with contextlib.suppress(OSError):
        os.set_inheritable(fd, False)  # agent commands must not inherit (and keep) the lock
    try:
        if _IS_WINDOWS:
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    with contextlib.suppress(OSError):  # diagnostics only; the lock is the truth
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\nstarted={datetime.now().isoformat()}\n".encode())
    _SESSION_LOCK_FDS.append(fd)
    return fd


def acquire_session_file(session_path: str | None = None) -> tuple[str, int]:
    """Reserve this process's session file; returns ``(path, instance_number)``.

    ``<path>2``, ``<path>3`` ... are tried while the preferred file is locked by
    another live instance. ``("", 0)`` means persistence could not be set up.
    """
    base = (os.path.abspath(os.path.expanduser(session_path))
            if session_path else os.path.join(os.getcwd(), _SESSION_BASENAME))
    for instance in range(1, 1000):
        candidate = base if instance == 1 else f"{base}{instance}"
        if _acquire_instance_lock(candidate) is not None:
            return candidate, instance
    return "", 0


def read_session_file(path: str) -> tuple[dict | None, str | None]:
    """Load a session file as ``(data, error)``; a missing file is ``(None, None)``.
    A bare JSON array is accepted as the messages list."""
    if not path:
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"invalid JSON: {exc}"
    if isinstance(data, list):
        return {"messages": data}, None
    if not isinstance(data, dict):
        return None, "session file is neither a JSON object nor an array"
    return data, None


# --- Event sinks -------------------------------------------------------------

class EventSink:
    """Presentation-layer I/O for AICommander.

    Event kinds: LOG, ERROR, USER_REQUEST, LLM_STREAM, THINKING_STREAM,
    CONSOLE_STREAM (internal calls such as the summarizer), CMD_EXEC,
    CMD_OUTPUT, CMD_COMPLETE, APPROVAL_REQUEST, STATUS_UPDATE, TOKEN_USAGE,
    TASK_TITLE, SYSTEM and SHUTDOWN.
    """

    def emit(self, kind: str, payload: dict):
        raise NotImplementedError

    def input(self, prompt: str) -> str:
        raise NotImplementedError

    def confirm(self, command: str) -> str:
        """Ask whether to run *command*; returns y / n / a free-text suggestion."""
        return self.input(f"Approve command? (y/n/suggestion): {command}")

    def close(self):
        pass


class ConsoleSink(EventSink):
    """Direct terminal output for --nogui mode."""

    _STREAMS = {"LLM_STREAM": colors.CYAN, "THINKING_STREAM": colors.YELLOW,
                "CONSOLE_STREAM": colors.YELLOW, "CMD_OUTPUT": colors.GREEN}
    _LOG_STYLES = {"yellow": colors.YELLOW, "green": colors.GREEN, "red": colors.RED,
                   "cyan": colors.CYAN, "dim": colors.DIM, "bright_yellow": colors.BOLD + colors.YELLOW,
                   "bold bright_yellow": colors.BOLD + colors.YELLOW}

    def __init__(self):
        self._at_line_start = True
        self._stream = None
        self._lock = threading.RLock()
        self._tty = sys.stdout.isatty()
        self._color = self._tty and not os.environ.get("NO_COLOR")

    def _write(self, text: str, style: str = "", out=None) -> None:
        if not text:
            return
        out = out or sys.stdout
        if not self._tty:  # piped: PTY line endings would show up as ^M
            text = text.replace("\r\n", "\n")
        out.write(f"{style}{text}{colors.END}" if style and self._color else text)
        out.flush()
        self._at_line_start = text.endswith("\n")

    def _line(self, text: str, style: str = "", out=None) -> None:
        if not self._at_line_start:
            self._write("\n")
        self._write(text + "\n", style, out)
        self._stream = None

    def emit(self, kind: str, payload: dict):
        text = payload.get("text", "")
        with self._lock:
            if kind in self._STREAMS:
                if self._stream not in (None, kind) and not self._at_line_start:
                    self._write("\n")
                if payload.get("label"):
                    self._line(payload["label"], colors.BOLD + colors.YELLOW)
                self._stream = kind
                self._write(text, self._STREAMS[kind])
            elif kind == "LOG":
                self._line(text, self._LOG_STYLES.get(payload.get("style", ""), ""))
            elif kind == "ERROR":
                self._line(text, colors.RED, sys.stderr)
            elif kind == "USER_REQUEST":
                self._line(f"[USER REQUEST] {text}", colors.BOLD + colors.YELLOW)
                self._line("=" * 60)
            elif kind == "CMD_EXEC":
                self._line(f"$ {payload.get('command', '')}", colors.BOLD + colors.YELLOW)
            elif kind == "CMD_COMPLETE":
                self._line(f"[exit {payload.get('exit_code')}]", colors.BOLD)
            elif kind == "SYSTEM":
                self._line(f"[AI-COMMANDER STARTED] model={payload.get('model')} "
                           f"api={payload.get('api_base')}", colors.BOLD + colors.BLUE)
            elif kind == "SHUTDOWN" and payload.get("completed"):
                self._line("[TASK COMPLETE]", colors.BOLD + colors.GREEN)
            # STATUS_UPDATE, TOKEN_USAGE, TASK_TITLE, APPROVAL_REQUEST: TUI only

    def input(self, prompt: str) -> str:
        with self._lock:
            if not self._at_line_start:
                self._write("\n")
        with contextlib.suppress(ImportError):
            import readline  # noqa: F401  (line editing for the answer)
        answer = input(prompt)
        if not sys.stdin.isatty():  # nothing echoed the answer's newline
            print(answer)
        self._at_line_start = True
        return answer

    def confirm(self, command: str) -> str:
        question = "Run it? [y]es / [n]o / or type a suggestion: "
        return self.input(f"{colors.BOLD}{colors.YELLOW}{question}{colors.END}" if self._color else question)


class TUISink(EventSink):
    """Queue bridge to the Textual app: the agent thread only enqueues events and
    the UI thread renders them. confirm() blocks until resolve_approval()."""

    def __init__(self, event_queue: queue.Queue, stop_event: threading.Event):
        self.queue = event_queue
        self.stop_event = stop_event
        self.auto_approve = False
        self._approval_event: threading.Event | None = None
        self._approval_response: str | None = None

    def emit(self, kind: str, payload: dict):
        # The TUI styles text itself, so terminal escape codes are removed here.
        self.queue.put({"kind": kind, "payload": {
            k: _strip_ansi(v) if isinstance(v, str) else v for k, v in payload.items()}})

    def input(self, prompt: str) -> str:
        return self.confirm(prompt)

    def confirm(self, command: str) -> str:
        if self.auto_approve:
            return "y"
        # Create the event before publishing the request so the UI can never
        # answer before we wait on it.
        event = self._approval_event = threading.Event()
        self._approval_response = None
        self.emit("APPROVAL_REQUEST", {"command": command})
        while not event.wait(0.1):
            if self.stop_event.is_set():
                break
        response = self._approval_response if event.is_set() else None
        self._approval_event = None
        return response or "n"

    def resolve_approval(self, approved: bool, suggestion: str = ""):
        self._approval_response = "y" if approved else (suggestion or "n")
        if self._approval_event:
            self._approval_event.set()

    def close(self):
        if self._approval_event:
            self._approval_event.set()


class _AICommanderTokenCounter:
    """Lets the context-compressor-llm library count tokens with aic's estimator,
    so its thresholds use the same units as the compression trigger."""

    def __init__(self, commander):
        self._commander = commander

    def count_tokens(self, text: str) -> int:
        return self._commander._estimate_message_tokens({"role": "user", "content": text}) if text else 0

    def count_message_tokens(self, messages: list) -> int:
        return sum(self.count_tokens(m.get("content", "")) for m in messages)


class CommandTimeoutError(Exception):
    """A command exceeded the timeout; ``output`` holds what it printed so far."""

    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


# --- The agent ----------------------------------------------------------------

class AICommander:
    """The agent: conversation state, streaming LLM calls, command execution,
    context compression and session persistence. All I/O goes to ``self.sink``."""

    COMPLETION_MARKER = "TASKCOMPLETE"
    OUTPUT_TRUNCATION_SENTINEL = "output too long: truncated"

    def __init__(self, api_base: str, model: str, api_key: str, auto_approve: bool = False,
                 show_thinking: bool = True, command_timeout: int = 120,
                 max_prompt_len: int = DEFAULT_MAX_PROMPT_LEN, max_output_bytes: int = 10240,
                 debug: bool = False, sink: EventSink | None = None, persist_history: bool = False,
                 max_steps: int = 500, compress_algorithm: str = "context-compressor-llm",
                 compress_target: float = 0.4, fast: bool = False, reasoning_effort: str | None = None,
                 sandbox_enabled: bool = False, sandbox_write_dir: str = "",
                 session_file: str = "", instance_number: int = 1):
        self.base_url = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.auto_approve = auto_approve
        self.show_thinking = show_thinking
        self.command_timeout = command_timeout
        self.max_prompt_len = max_prompt_len
        self.max_output_bytes = max_output_bytes
        self.max_tokens = 16000  # completion reservation, sent as max_tokens to non-"gpt" models
        self.debug = debug
        self.sink: EventSink = sink if sink is not None else ConsoleSink()
        self.persist_history = persist_history  # run() extends the history instead of resetting it
        self.max_steps = max_steps
        self.compress_algorithm = compress_algorithm  # "context-compressor-llm" or "truncate"
        self.compress_target = compress_target  # fraction of the budget kept after compressing
        self.fast = fast
        self.reasoning_effort = reasoning_effort
        self.sandbox_enabled = sandbox_enabled
        self.sandbox_write_dir = sandbox_write_dir
        self.session_file = session_file  # "" disables persistence
        self.instance_number = instance_number

        self.conversation_history: list[dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_count = 0
        self.usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "billable_tokens": 0}
        self.initial_prompt = ""
        self.task_summary = ""
        # The background title summarizer makes its own LLM call; tests driving a
        # scripted fake client disable it so it cannot consume scripted turns.
        self.task_summary_enabled = True
        self.stop_event = threading.Event()
        self.suggestion_queue: queue.Queue = queue.Queue()
        self._session_write_lock = threading.RLock()
        self._active_streams: set = set()
        self._summary_generation = 0
        self._current_request: dict | None = None
        self._started_banner_shown = False
        self._command_started = 0.0
        self._send_reasoning = True  # off once an endpoint rejects reasoning_content in requests
        # Retries are done here (stop-aware, with backoff), not inside the SDK.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             default_headers={"User-Agent": USER_AGENT}, max_retries=0)
        self.tool_schemas = [{
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a shell command in the terminal (bash on Linux/macOS, PowerShell on Windows)",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                    "required": ["command"],
                },
            },
        }]

    def _log(self, message: str, style: str = ""):
        self.sink.emit("LOG", {"text": message, "style": style})

    def _log_error(self, message: str):
        self.sink.emit("ERROR", {"text": message})

    def request_stop(self):
        """Stop the run: commands are killed and a pending LLM stream is closed."""
        self.stop_event.set()
        for stream in list(self._active_streams):
            with contextlib.suppress(Exception):
                stream.close()

    # --- Token accounting ---------------------------------------------------

    def _estimate_message_tokens(self, msg: dict[str, Any]) -> int:
        """~3 characters per token plus per-message and per-tool-call overhead.
        Deliberately conservative so compression triggers before the server's
        window overflows; exact numbers come from the API usage stats."""
        total = 4
        content = msg.get("content")
        if isinstance(content, str):
            total += max(1, len(content) // 3)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    total += max(1, len(part["text"]) // 3)
        if isinstance(msg.get("reasoning_content"), str):
            total += len(msg["reasoning_content"]) // 3
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += max(1, len(fn.get("name") or "") // 3)
            total += max(1, len(str(fn.get("arguments") or "")) // 3)
        return total

    def _estimate_input_tokens(self, messages: list[dict[str, Any]]) -> int:
        return sum(self._estimate_message_tokens(m) for m in messages)

    def _emit_token_usage(self, input_tokens: int, output_tokens: int, cached: int = 0,
                          estimate: bool = False, tokens_per_sec: float | None = None) -> None:
        """Report usage; exact (non-estimate) numbers also accumulate the
        persisted totals. Prompt-cached tokens are not counted as billable."""
        self.sink.emit("TOKEN_USAGE", {"input_tokens": input_tokens, "output_tokens": output_tokens,
                                       "cached_tokens": cached, "estimate": estimate,
                                       "tokens_per_sec": tokens_per_sec})
        if estimate:
            return
        with self._session_write_lock:  # the title summarizer reports from another thread
            self.usage_totals["input_tokens"] += input_tokens
            self.usage_totals["output_tokens"] += output_tokens
            self.usage_totals["cached_tokens"] += cached
            self.usage_totals["billable_tokens"] += max(0, input_tokens - cached) + output_tokens

    @staticmethod
    def _extract_cached_tokens(usage) -> int:
        """Prompt-cache hits: OpenAI, DeepSeek and Anthropic name the field differently."""
        details = getattr(usage, "prompt_tokens_details", None)
        return (getattr(details, "cached_tokens", 0) or getattr(usage, "prompt_cache_hit_tokens", 0)
                or getattr(usage, "cache_read_input_tokens", 0) or 0)

    # --- Session persistence -------------------------------------------------

    def session_snapshot(self, reason: str = "") -> dict[str, Any]:
        """The session file document; ``messages`` is the live history verbatim."""
        return {
            "aic_session_version": _SESSION_VERSION,
            "session_id": self.session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": reason,
            "model": self.model,
            "api_base": self.base_url,
            "cwd": _safe_cwd(),
            "instance_number": self.instance_number,
            "step_count": self.step_count,
            "usage": dict(self.usage_totals),
            "initial_prompt": self.initial_prompt,
            "task_summary": self.task_summary,
            "messages": list(self.conversation_history),
        }

    def save_session(self, reason: str = "") -> bool:
        """Atomically mirror the history to ``self.session_file`` (temp file,
        fsync, rename). Never raises: persistence must not break the agent."""
        if not self.session_file:
            return False
        tmp = f"{self.session_file}.tmp.{os.getpid()}"
        try:
            with self._session_write_lock:
                data = json.dumps(self.session_snapshot(reason), indent=2, ensure_ascii=False)
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(data + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.session_file)
            return True
        except Exception as exc:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            if self.debug:
                self._log(f"[SESSION] Could not save {self.session_file}: {exc}", style="yellow")
            return False

    @staticmethod
    def _merge_assistant_runs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive assistant messages (APIs reject them) without losing
        the text or tool calls of either."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            if out and msg.get("role") == "assistant" and out[-1].get("role") == "assistant":
                prev = out[-1]
                texts = [t for t in (_content_text(prev.get("content")), _content_text(msg.get("content")))
                         if t.strip()]
                merged = {**prev, **msg, "content": "\n\n".join(texts)}
                calls = list(prev.get("tool_calls") or []) + list(msg.get("tool_calls") or [])
                if calls:
                    merged["tool_calls"] = calls
                else:
                    merged.pop("tool_calls", None)
                out[-1] = merged
            else:
                out.append(msg)
        return out

    def _repair_loaded_history(self, messages: list) -> tuple[list[dict[str, Any]], list[str]]:
        """Make a loaded (possibly hand-edited) message list safe to send.

        Drops malformed entries and orphan tool results, guarantees a leading
        system prompt, collapses stacked continuation prompts and empty turns,
        merges consecutive assistants and answers tool calls left dangling by a
        crash with a "not re-executed" notice. A trailing assistant answer is
        kept: it is the previous run's final reply.
        """
        notes: list[str] = []
        cleaned: list[dict[str, Any]] = []
        known_ids: set = set()
        for msg in messages:
            if not isinstance(msg, dict):
                notes.append("dropped a non-object message entry")
                continue
            role = msg.get("role")
            if role not in _VALID_ROLES:
                notes.append(f"dropped a message with invalid role {role!r}")
                continue
            if role == "tool" and msg.get("tool_call_id") not in known_ids:
                notes.append("dropped a tool message without a matching tool call")
                continue
            if role == "assistant":
                known_ids.update(tc["id"] for tc in msg.get("tool_calls") or []
                                 if isinstance(tc, dict) and tc.get("id"))
                if _is_empty_assistant(msg):
                    notes.append("dropped an empty assistant message")
                    continue
            if _is_continuation(msg) and cleaned and _is_continuation(cleaned[-1]):
                notes.append("collapsed a duplicate continuation prompt")
                continue
            cleaned.append(msg)
        if not cleaned or cleaned[0].get("role") != "system":
            cleaned.insert(0, {"role": "system", "content": self.get_system_prompt()})
            notes.append("prepended a fresh system prompt")
        cleaned = self._merge_assistant_runs(cleaned)
        if cleaned[-1].get("role") == "assistant":
            answered = {m.get("tool_call_id") for m in cleaned if m.get("role") == "tool"}
            for tc in cleaned[-1].get("tool_calls") or []:
                tcid = tc.get("id") if isinstance(tc, dict) else None
                if not tcid or tcid in answered:
                    continue
                fn = (tc.get("function") or {}).get("name") or "unknown function"
                cleaned.append(self._tool_result(
                    tcid, f"[SESSION RECOVERED] This tool call ({fn}) was issued in a previous "
                          f"AI-Commander session that ended before its result was recorded, so it was "
                          f"NOT re-executed. Do not assume it had any effect; verify the current state "
                          f"before retrying it."))
                notes.append(f"closed an unanswered tool call ({fn})")
        return cleaned, notes

    def load_session(self, notify: bool = True) -> bool:
        """Restore the history, id, counters and title from ``self.session_file``.
        Returns True when messages were restored; a missing file is a quiet fresh start."""
        if not self.session_file:
            return False
        data, err = read_session_file(self.session_file)
        if err:  # keep the unreadable file aside instead of destroying it
            backup = f"{self.session_file}.corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.replace(self.session_file, backup)
                moved = f"; original moved to {backup}"
            except OSError:
                moved = ""
            self._log(f"[SESSION] Ignoring unreadable session file {self.session_file} ({err}){moved}", style="yellow")
            return False
        messages = (data or {}).get("messages")
        if not isinstance(messages, list) or not messages:
            if data and notify:
                self._log(f"[SESSION] {self.session_file} has no messages to restore", style="yellow")
            return False
        restored, notes = self._repair_loaded_history(messages)
        if notes:
            self._log(f"[SESSION] Repaired the saved history: {'; '.join(dict.fromkeys(notes))}.", style="yellow")
        self.conversation_history = restored
        self.persist_history = True
        if isinstance(data.get("session_id"), str) and data["session_id"]:
            self.session_id = data["session_id"]
        if isinstance(data.get("step_count"), int) and data["step_count"] >= 0:
            self.step_count = data["step_count"]
        usage = data.get("usage")
        if isinstance(usage, dict):
            for key in self.usage_totals:
                if isinstance(usage.get(key), int) and usage[key] >= 0:
                    self.usage_totals[key] = usage[key]
        self.initial_prompt = data.get("initial_prompt") or ""
        self.task_summary = data.get("task_summary") or ""
        if notify:
            self._log(f"[SESSION] Restored {len(restored)} message(s) from {os.path.basename(self.session_file)} "
                      f"(session {self.session_id}, step {self.step_count})", style="green")
            last = restored[-1]
            if last.get("role") == "assistant":
                text = _content_text(last.get("content")).strip()
                if last.get("tool_calls"):
                    text = "tool call: " + ", ".join((tc.get("function") or {}).get("name") or "?"
                                                     for tc in last["tool_calls"] if isinstance(tc, dict))
                if text:
                    self._log(f"[SESSION] Last assistant message: {text[:197] + '...' if len(text) > 200 else text}",
                              style="cyan")
        return True

    def discard_session(self, reason: str = "discarded"):
        """Forget the conversation (/clear, /new) and overwrite the session file,
        so the next startup does not resurrect it."""
        self.conversation_history = []
        self.step_count = 0
        self.initial_prompt = self.task_summary = ""
        self.usage_totals = dict.fromkeys(self.usage_totals, 0)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_session(reason)

    def _dump_conversation_history(self):
        """--debug: write the full history to ~/.aic (or AIC_DEBUG_DUMP_DIR, or
        the temp dir when the sandbox forbids both) under a unique name."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        lines = [f"Conversation history dump at {datetime.now().isoformat()}",
                 f"Total messages: {len(self.conversation_history)}",
                 f"Estimated tokens: {self._estimate_input_tokens(self.conversation_history)} "
                 f"(max prompt length {self.max_prompt_len})", "=" * 80, ""]
        for i, msg in enumerate(self.conversation_history):
            lines.append(f"--- Message {i} (role: {msg.get('role', 'unknown')}) ---")
            if msg.get("tool_call_id"):
                lines.append(f"tool_call_id: {msg['tool_call_id']}")
            if msg.get("tool_calls"):
                lines.append(f"tool_calls: {json.dumps(msg['tool_calls'], indent=2)}")
            content = _content_text(msg.get("content"))
            lines += [f"content ({len(content)} chars):", content, ""]
        dirs = [os.environ.get("AIC_DEBUG_DUMP_DIR") or os.path.expanduser("~/.aic"),
                os.path.dirname(self.session_file) if self.session_file else "", "/tmp"]
        for base in filter(None, dirs):
            path = os.path.join(base, f"commander-debug-{self.session_id}-{stamp}.txt")
            try:
                os.makedirs(base, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                self._log(f"[DEBUG DUMP] Conversation history dumped to {path}", style="dim")
                return
            except OSError:
                continue

    # --- Context compression -------------------------------------------------

    def _prompt_budget(self) -> int:
        """Tokens the prompt may use: the window minus the completion reservation
        (only models that are sent max_tokens reserve it)."""
        if "gpt" not in self.model:
            return max(0, self.max_prompt_len - self.max_tokens)
        return self.max_prompt_len

    def _protected_messages(self) -> list[dict[str, Any]]:
        """Messages compression must keep verbatim: the system prompt, the first
        user instruction and the request that started the current run."""
        hist = self.conversation_history
        keep = [hist[0]] if hist and hist[0].get("role") == "system" else []
        first_user = next((m for m in hist if m.get("role") == "user"), None)
        for msg in (first_user, self._current_request):
            if msg is not None and any(msg is m for m in hist) and not any(msg is k for k in keep):
                keep.append(msg)
        return keep

    def _context_compress(self, force: bool = False):
        """Compress the history when it exceeds the prompt budget, dispatching to
        ``_compress_<algorithm>``. *force* compresses even under budget (used
        after the server rejected a prompt as too long)."""
        budget = self._prompt_budget()
        total_tokens = self._estimate_input_tokens(self.conversation_history)
        if force:
            budget = max(1, min(budget, int(total_tokens * 0.75)))
        elif total_tokens <= budget:
            return
        self.sink.emit("STATUS_UPDATE", {"step": self.step_count, "max_steps": self.max_steps,
                                         "phase": "compressing"})
        algorithm = getattr(self, f"_compress_{self.compress_algorithm.replace('-', '_')}", None)
        if algorithm is None:
            self._log(f"[COMPRESS] Unknown context-compression algorithm '{self.compress_algorithm}'. "
                      f"Falling back to 'truncate'.", style="yellow")
            algorithm = self._compress_truncate
        algorithm(budget, total_tokens)

    def _compress_truncate(self, budget: int, total_tokens: int):
        """Condense old tool outputs, then drop the oldest unprotected messages,
        each pass stopping once the history fits ``compress_target`` of the
        budget (headroom, so the next steps do not re-trigger compression)."""
        target = max(1, int(budget * self.compress_target))
        self._log(f"[COMPRESS] Using 'truncate': ~{total_tokens} tokens exceed the budget ({budget}); "
                  f"keeping ~{target} ({int(self.compress_target * 100)}% of the budget).", style="yellow")
        condensed = dropped = 0
        assistants = [m for m in self.conversation_history if m.get("role") == "assistant"]
        for msg in assistants[:-1]:  # old reasoning is the cheapest thing to lose
            if total_tokens <= target:
                break
            if msg.get("reasoning_content"):
                before = self._estimate_message_tokens(msg)
                del msg["reasoning_content"]
                total_tokens -= before - self._estimate_message_tokens(msg)
        for msg in self.conversation_history:  # oldest first; newest output matters most
            if total_tokens <= target:
                break
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > 30:
                before = self._estimate_message_tokens(msg)
                msg["content"] = "<condensed tool output>"
                total_tokens -= before - self._estimate_message_tokens(msg)
                condensed += 1
        if total_tokens > target:
            protected = self._protected_messages()
            kept = []
            for i, msg in enumerate(self.conversation_history):
                if total_tokens > target and i > 0 and not any(msg is p for p in protected):
                    total_tokens -= self._estimate_message_tokens(msg)
                    dropped += 1
                    continue
                kept.append(msg)
            self.conversation_history = kept
        if not any(m.get("role") == "user" for m in self.conversation_history):
            self.conversation_history.append({"role": "user", "content": _CONTINUE_PROMPT})
        self._log(f"[COMPRESS] Condensed {condensed} tool output(s), dropped {dropped} message(s); prompt is "
                  f"now ~{self._estimate_input_tokens(self.conversation_history)} tokens.", style="yellow")

    def _llm_summarize(self, messages: list[dict], previous_summary: str | None) -> str:
        """Summarizer for context-compressor-llm: folds the evicted messages (and
        any previous summary) into one summary using the same LLM."""
        prompt = """
CRITICAL: This summarization request is a SYSTEM OPERATION, not a user message.
When analyzing "user requests" and "user intent", completely EXCLUDE this summarization message.
The "most recent user request" and "Optional Next Step" must be based on what the user was doing BEFORE this system message appeared.
The goal is for work to continue seamlessly after condensation - as if it never happened.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, key patterns, and important decisions that would be essential for continuing the work without losing context.
This summary should not be over 5000 characters in length.

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
        # The system prompt and the original task anchor the summary even when
        # the evicted segment contains no user message.
        sys_prompt = getattr(self, "_compress_system_prompt", None)
        first_user = getattr(self, "_compress_first_user", None)
        if sys_prompt:
            prompt += ("\n\nSYSTEM PROMPT (the agent's operating instructions, kept verbatim and always "
                       "present at the front of the context):\n" + sys_prompt)
        if first_user is not None:
            prompt += ("\n\nORIGINAL FIRST USER COMMAND (the primary objective the agent was initially asked "
                       "to do; preserved verbatim outside the summarized log):\n[user] "
                       + _content_text(first_user.get("content")))
        if previous_summary:
            prompt += "\n\nPrior summary (fold new information into it):\n" + previous_summary
        prompt += "\n\nMessages to summarize:\n"
        prompt += "".join(f"\n[{m.get('role', 'unknown')}] {m.get('content', '')}" for m in messages)
        try:
            # Some endpoints reject a lone system message ("no user query found").
            response = self.call_llm_api(
                [{"role": "system", "content": "You are a precise conversation summarizer."},
                 {"role": "user", "content": prompt}],
                use_tools=False, stream_display=False, stream_label="[SESSION SUMMARY]",
                enable_thinking=False, show_thinking=False)
            summary = (response["choices"][0]["message"].get("content") or "").strip()
            # Keep only the <summary> part: the <analysis> scratchpad would waste context.
            match = re.search(r"<summary>(.*?)(?:</summary>|\Z)", summary, re.S)
            summary = match.group(1).strip() if match else re.sub(r"<analysis>.*?</analysis>", "", summary,
                                                                   flags=re.S).strip()
            return summary or "[summary unavailable]"
        except Exception as exc:
            self._log_error(f"[COMPRESS] LLM summarizer failed: {exc}. Using a placeholder summary.")
            return "[summary unavailable]"

    def _compress_context_compressor_llm(self, budget: int, total_tokens: int):
        """Anchored-summary compression (context-compressor-llm package): the
        oldest messages are folded into a summary kept in the system prompt and
        the newest suffix is retained verbatim, an append-only, KV-cache-friendly
        layout. The previous summary is folded into the new one, so summaries
        never pile up across compressions."""
        if not _CONTEXT_COMPRESSOR_LLM_AVAILABLE:
            self._log("[COMPRESS] The context-compressor-llm package is not installed. "
                      "Falling back to 'truncate'.", style="yellow")
            return self._compress_truncate(budget, total_tokens)
        self._log(f"[COMPRESS] Using 'context-compressor-llm': ~{total_tokens} tokens exceed the budget "
                  f"({budget}). Summarizing the oldest messages.", style="yellow")
        hist = self.conversation_history
        base_prompt, old_summary = None, None
        if hist and hist[0].get("role") == "system":
            base_prompt, _, old_summary = _content_text(hist[0].get("content")).partition(_SUMMARY_MARKER)
        protected = self._protected_messages()
        first_user = next((m for m in protected if m.get("role") == "user"), None)
        request = self._current_request if any(self._current_request is p for p in protected) else None
        if request is first_user:
            request = None
        log = [(i, m) for i, m in enumerate(hist)
               if m.get("role") != "system" and not any(m is p for p in protected)]
        self._compress_system_prompt = base_prompt
        self._compress_first_user = first_user

        # Size everything in the trigger's units: content + tool-call arguments +
        # per-message overhead (the library alone would count content only and
        # retain far too much, re-triggering compression on the next step).
        fixed = self._estimate_input_tokens([m for m in protected if m is not hist[0]])
        if base_prompt is not None:
            fixed += self._estimate_message_tokens({"role": "system", "content": base_prompt})
        log_budget = max(1, budget - fixed)
        log_tokens = max(1, sum(self._estimate_message_tokens(m) for _, m in log))
        t_summary = max(1, min(max(20, min(budget // 20, log_budget // 3)), log_budget - 1))
        t_retained = max(t_summary + 1, min(int(budget * self.compress_target) - fixed, log_tokens - 1))
        cc = _ContextCompressorLib(
            summarizer=lambda msgs, prev: self._llm_summarize(msgs, prev or old_summary or None),
            t_max=t_retained, t_retained=t_retained, t_summary=t_summary,
            tokenizer=_AICommanderTokenCounter(self))
        for index, msg in log:
            content = _content_text(msg.get("content"))
            calls = msg.get("tool_calls") or []
            if calls:  # tool-call turns usually have empty content: give the summarizer a short preview
                previews = "; ".join(f"{(tc.get('function') or {}).get('name', '?')}"
                                     f"({str((tc.get('function') or {}).get('arguments') or '')[:200]})"
                                     for tc in calls[:4])
                content = (content + f"\n[tool_calls: {previews}]").strip()
            cc.add_message(content, role=msg.get("role", "user"), metadata={"original": msg, "index": index})
        for lib_msg in cc.state.messages:
            lib_msg.token_count = self._estimate_message_tokens(lib_msg.metadata["original"])
        context = cc.get_current_context(auto_compress=True)
        stats = cc.get_stats()
        if not stats["compression_count"]:
            self._log("[COMPRESS] Nothing could be summarized; falling back to 'truncate'.", style="yellow")
            return self._compress_truncate(budget, self._estimate_input_tokens(hist))

        summary = next((m.content for m in context if m.metadata.get("type") == "summary"), None) or old_summary
        retained = [(m.metadata["index"], m.metadata["original"]) for m in context if "original" in m.metadata]
        if request is not None:  # keep the current task at its chronological place
            retained.append((next(i for i, m in enumerate(hist) if m is request), request))
            retained.sort(key=lambda pair: pair[0])
        new_history: list[dict[str, Any]] = []
        if base_prompt is not None or summary:
            content = (base_prompt or "") + (_SUMMARY_MARKER + summary if summary else "")
            new_history.append({"role": "system", "content": content if base_prompt is not None else content.lstrip()})
        if first_user is not None:
            new_history.append(first_user)
        new_history.extend(m for _, m in retained)
        if not any(m.get("role") == "user" for m in new_history):
            new_history.append({"role": "user", "content": _CONTINUE_PROMPT})
        self.conversation_history = new_history
        self._log(f"[COMPRESS] Summarized {len(log) - len(retained) + (request is not None)} message(s); "
                  f"prompt is now ~{self._estimate_input_tokens(new_history)} tokens.", style="yellow")

    def _refresh_system_prompt(self):
        """Refresh the system prompt for a new run, keeping any anchored summary."""
        first = self.conversation_history[0]
        if first.get("role") != "system":
            self.conversation_history.insert(0, {"role": "system", "content": self.get_system_prompt()})
            return
        _, marker, summary = _content_text(first.get("content")).partition(_SUMMARY_MARKER)
        self.conversation_history[0] = {"role": "system", "content": self.get_system_prompt() + marker + summary}

    # --- System prompts ------------------------------------------------------

    def get_system_prompt(self) -> str:
        """The system prompt (the shorter variant in --fast mode)."""
        if self.fast:
            return self.get_fast_system_prompt()
        sandbox = (f"You are running inside an OS-level sandbox. You may WRITE only to {self.sandbox_write_dir}, "
                   f"/tmp and /dev; reads and execution are allowed anywhere, and network access is preserved. "
                   f"Write files within these directories or they may fail."
                   if self.sandbox_enabled else
                   f"You are running inside an OS-level sandbox (currently disabled). When the sandbox is active, "
                   f"writes are restricted to {self.sandbox_write_dir}, /tmp and /dev.")
        return f"""You are an expert planning and execution assistant. Fulfill the user's request by breaking it into manageable steps and executing bash commands via the execute_bash tool (one command at a time, waiting for each result).

**Workflow:**
1. Analyze the request; if complex, break it into a numbered task list.
2. Restate your plan every ~5 steps and update the task list as you progress.
3. Execute one step with execute_bash, then evaluate the result.
4. If a step fails, adjust your approach and explain. If all steps are done and the goal is met, emit the exact phrase '{self.COMPLETION_MARKER}'.
5. Only emit '{self.COMPLETION_MARKER}' once you are certain every necessary action is complete and verified. Never say it prematurely.

**Working Directory:**
- Create a working directory named after the task (e.g. `work/<short-task-slug>/`, under the current directory or /tmp — wherever writes are allowed) and keep ALL temporary files there: scripts, patches, downloads, test output, generated artifacts. Never scatter temporary files in the project root; only the task's real deliverables belong outside the working directory.

**Runtime Constraints (execute_bash):**
- Output is capped at {self.max_output_bytes} bytes. If truncated, the literal sentinel `{self.OUTPUT_TRUNCATION_SENTINEL}` is appended. If you see it, you are missing data — do NOT assume success or failure. Re-run with output redirected to a file and read in chunks via `sed -n 'start,end p' file`, or use `head -c N`/`tail -c N`. Prefer targeted commands (grep, wc, stat) over dumping large outputs.
- Commands are killed after {self.command_timeout}s. For long operations use `nohup ... &` and check later, split into smaller steps, or set your own `timeout`.
- Commands run in a PTY (a plain pipe on Windows); use non-interactive flags (`-y`, `--no-interactive`) where available. Pagers are disabled (PAGER=cat).
- {sandbox}

**Command Results:**
- Never echo command execution metadata (e.g. "[EXECUTING]", "exit code X") into your response content; the tool system supplies that.
- Only describe your plan, your reasoning, and conclusions drawn from the REAL tool output. Never fabricate or guess command output — wait for the actual result.

**Internet Access:**
- Web search by printing JSON results to stdout: `python3 -c "import json; from ddgs import DDGS; print(json.dumps(list(DDGS().text('QUERY', max_results=10)), ensure_ascii=False, indent=2))"` (replace QUERY). Use wget/curl and the w3m browser to fetch page contents.
{self._ddgs_note()}

**Persistence Policy (CRITICAL):**
- Keep working until the task is genuinely complete. Do NOT stop early or produce a summary-only final response — take concrete action with tools.
- If you respond without calling tools, the system injects a continuation prompt; you must keep making progress.
- Troubleshoot errors and try alternative approaches until the objective is met.
- The ONLY way to finish is to emit '{self.COMPLETION_MARKER}' AFTER verifying all objectives via actual command execution and result inspection. Partial completion is not completion.
- Do not ask the user for clarification unless you have exhausted autonomous options.
- Before emitting '{self.COMPLETION_MARKER}', verify your last actions (check file contents, run tests, confirm services).

Think carefully; response quality is the highest priority. You have unlimited thinking tokens."""

    def get_fast_system_prompt(self) -> str:
        """A smaller, faster system prompt for --fast mode."""
        return f"""You are an expert bash assistant. Complete the user's task by running commands via execute_bash, one at a time, and acting on each result.

Workflow:
- Break complex tasks into steps and track them.
- If a step fails, fix it and retry.
- When the task is fully done and verified, emit '{self.COMPLETION_MARKER}' and stop.

Rules:
- Create a working directory named after the task (e.g. `work/<short-task-slug>/`) and keep temporary files there, not in the project root.
- Output is capped at {self.max_output_bytes} bytes; if truncated the sentinel `{self.OUTPUT_TRUNCATION_SENTINEL}` is appended. If you see it, you are missing data — do not assume success or failure; re-run a more targeted command.
- Commands are killed after {self.command_timeout}s.
- Only report conclusions from real tool output; never fabricate results.
- Web search by printing JSON results to stdout: `python3 -c "import json; from ddgs import DDGS; print(json.dumps(list(DDGS().text('QUERY', max_results=10)), ensure_ascii=False, indent=2))"` (replace QUERY). Use wget/curl and the w3m browser to fetch page contents.
{self._ddgs_note()}
- Keep working until the task is genuinely complete; partial completion is not completion."""

    @staticmethod
    def _ddgs_note() -> str:
        if _DDGS_AVAILABLE:
            return "(ddgs is installed and available.)"
        return ("(NOTE: the ddgs package is NOT installed in this environment: that one-liner will fail with "
                "ModuleNotFoundError. Install it with `pip install ddgs`, or rely on curl/wget/w3m only.)")

    # --- Command execution ---------------------------------------------------

    def execute_bash_command(self, command: str) -> tuple[str, int]:
        """Run *command* with a timeout and return (output, exit_code).

        POSIX runs it in a PTY of its own session, so the whole process tree
        can be killed; Windows uses PowerShell over pipes. Output streams to the
        sink as CMD_OUTPUT and the returned text is cleaned and truncated.
        """
        self._command_started = time.monotonic()
        if _IS_WINDOWS:
            return self._execute_bash_command_windows(command)
        try:
            pid, fd = pty.fork()
        except OSError as exc:
            return self._finish_command(bytearray(f"Failed to start a shell: {exc}\n".encode()), command, 1, False)
        if pid == 0:  # child (a session leader on the PTY)
            try:
                os.environ.update(_COMMAND_ENV)
                with contextlib.suppress(OSError):  # a wide window: fewer wrapped lines
                    fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))
                os.execv("/bin/sh", ["/bin/sh", "-c", _SHELL_PRELUDE + command])
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    os.write(2, f"cannot run /bin/sh: {exc}\n".encode())
            finally:
                os._exit(127)

        output = bytearray()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        held = ""  # escape sequence split across reads, completed by the next chunk
        limit = 5 * self.max_output_bytes  # raw bytes after which the command is killed
        deadline = time.monotonic() + self.command_timeout
        # --nogui: forward the user's keystrokes to the command (only from a real terminal).
        watch_stdin = not isinstance(self.sink, TUISink) and sys.stdin is not None and sys.stdin.isatty()
        exit_code: int | None = None
        timed_out = eof = False
        os.set_blocking(fd, False)
        try:
            while True:
                if self.stop_event.is_set():
                    self._log_error("[STOP] Agent shutdown requested. Terminating command.")
                    exit_code = self._kill_process_group(pid)
                    break
                if time.monotonic() > deadline:
                    self._log_error(f"[TIMEOUT] Command timed out after {self.command_timeout} seconds.")
                    exit_code = self._kill_process_group(pid)
                    timed_out = True
                    break
                watched = ([] if eof else [fd]) + ([sys.stdin.fileno()] if watch_stdin else [])
                ready = select.select(watched, [], [], 0.1)[0] if watched else []
                if not watched:
                    time.sleep(0.1)
                if fd in ready:
                    data = self._read_available(fd)
                    if data is None:
                        eof = True  # every slave fd closed; the process may still be running
                    elif data:
                        output += data
                        held = self._emit_output(decoder, held, data, command)
                        if len(output) > limit:
                            self._log_error(f"[OUTPUT LIMIT] Command output exceeded {limit} bytes. Stopping command.")
                            exit_code = self._kill_process_group(pid)
                            break
                if watch_stdin and sys.stdin.fileno() in ready:
                    try:
                        data = os.read(sys.stdin.fileno(), 1024)
                    except OSError:
                        data = b""
                    if data:
                        with contextlib.suppress(OSError):
                            os.write(fd, data)
                    else:
                        watch_stdin = False
                try:
                    wpid, status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    exit_code = -1
                    break
                if wpid:
                    exit_code = _exit_code(status)
                    data = self._read_available(fd)  # what the child left in the PTY buffer
                    if data:
                        output += data
                        held = self._emit_output(decoder, held, data, command)
                    break
        finally:
            tail = held + decoder.decode(b"", final=True)
            if tail:
                self.sink.emit("CMD_OUTPUT", {"text": tail, "command": command})
            os.close(fd)
            if exit_code is None:  # an exception escaped the loop: do not leak the process
                exit_code = self._kill_process_group(pid)
        return self._finish_command(output, command, exit_code, timed_out)

    @staticmethod
    def _read_available(fd: int, cap: int = 1 << 20) -> bytes | None:
        """Read what is buffered on a non-blocking fd; None at EOF/EIO."""
        chunks, size = [], 0
        while size < cap:
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                break
            except OSError:  # EIO: the slave side is closed
                return b"".join(chunks) or None
            if not data:
                return b"".join(chunks) or None
            chunks.append(data)
            size += len(data)
        return b"".join(chunks)

    def _emit_output(self, decoder, held: str, data: bytes, command: str) -> str:
        """Stream decoded output, holding back a trailing partial escape sequence."""
        text = held + decoder.decode(data)
        match = _ESC_TAIL_RE.search(text, max(0, len(text) - 256))
        held = text[match.start():] if match else ""
        text = text[:match.start()] if match else text
        if text:
            self.sink.emit("CMD_OUTPUT", {"text": text, "command": command})
        return held

    @staticmethod
    def _kill_process_group(pid: int) -> int:
        """SIGTERM the command's process group, SIGKILL it if needed; reap and
        return the exit code (-1 if it could not be determined)."""
        for sig, grace in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 2.0)):
            with contextlib.suppress(OSError):
                os.killpg(pid, sig)
            end = time.monotonic() + grace
            while time.monotonic() < end:
                try:
                    wpid, status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    return -1
                if wpid:
                    return _exit_code(status)
                time.sleep(0.02)
        return -1

    def _finalize_command_output(self, output_buffer: bytearray) -> str:
        """Decode, clean and truncate (by bytes) the output returned to the model."""
        text = _clean_command_output(bytes(output_buffer).decode("utf-8", errors="replace"))
        data = text.encode("utf-8")
        if len(data) > self.max_output_bytes:
            text = (data[:self.max_output_bytes].decode("utf-8", errors="ignore")
                    + "\n" + self.OUTPUT_TRUNCATION_SENTINEL)
        return text

    def _finish_command(self, output_buffer: bytearray, command: str, exit_code: int,
                        timed_out: bool) -> tuple[str, int]:
        final_output = self._finalize_command_output(output_buffer)
        self.sink.emit("CMD_COMPLETE", {"command": command, "exit_code": exit_code, "output": final_output,
                                        "duration": time.monotonic() - self._command_started})
        if timed_out:
            raise CommandTimeoutError(f"Command timed out after {self.command_timeout} seconds "
                                      f"and was killed.", final_output)
        return final_output, exit_code

    def _execute_bash_command_windows(self, command: str) -> tuple[str, int]:
        """Windows runner: PowerShell over pipes (there is no PTY)."""
        try:
            proc = subprocess.Popen(_powershell_argv(command), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        except OSError as exc:
            return self._finish_command(bytearray(f"Failed to start shell: {exc}\n".encode()), command, 1, False)
        output = bytearray()
        lock = threading.Lock()
        limit = 5 * self.max_output_bytes
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def reader():
            with contextlib.suppress(Exception):
                for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                    with lock:
                        output.extend(chunk)
                        text = decoder.decode(chunk)
                    if text:
                        self.sink.emit("CMD_OUTPUT", {"text": text, "command": command})
                    if len(output) > limit:
                        return

        def terminate():
            """taskkill /T walks the tree: killing PowerShell alone would orphan the command."""
            with contextlib.suppress(Exception):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                               timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with contextlib.suppress(Exception):
                proc.kill()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        timed_out = False
        deadline = time.monotonic() + self.command_timeout
        while proc.poll() is None:
            if self.stop_event.is_set():
                self._log_error("[STOP] Agent shutdown requested. Terminating command.")
                terminate()
                break
            if time.monotonic() > deadline:
                self._log_error(f"[TIMEOUT] Command timed out after {self.command_timeout} seconds.")
                terminate()
                timed_out = True
                break
            with lock:
                over = len(output) > limit
            if over:
                self._log_error(f"[OUTPUT LIMIT] Command output exceeded {limit} bytes. Stopping command.")
                terminate()
                break
            time.sleep(0.05)
        thread.join(timeout=2)
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        return self._finish_command(output, command, proc.returncode if proc.returncode is not None else 1,
                                    timed_out)

    def get_user_confirmation(self, command: str) -> tuple[bool, str | None]:
        """Ask the sink to approve *command*: (approved, suggestion-or-None).
        A blank answer, EOF or Ctrl+C never approves."""
        if self.auto_approve:
            return True, None
        try:
            answer = (self.sink.confirm(command) or "").strip()
        except (EOFError, KeyboardInterrupt):
            return False, None
        if answer.lower() in ("y", "yes"):
            return True, None
        if answer.lower() in ("", "n", "no"):
            return False, None
        return False, answer

    # --- Messages and the LLM call -----------------------------------------------

    def _sanitize_tool_pairing(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enforce assistant tool_calls / tool-result pairing (strict endpoints
        such as DeepSeek 400 otherwise): orphan tool results are dropped and tool
        calls that never got a result are removed. The input is not mutated."""
        cleaned: list[dict[str, Any]] = []
        pending: dict[str, bool] = {}  # tool_call_id -> still unanswered
        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                msg = dict(msg)
                calls = []
                for tc in msg["tool_calls"]:
                    tcid = tc.get("id") if isinstance(tc, dict) else None
                    if tcid and pending.get(tcid):
                        continue  # duplicate id inside the same list
                    calls.append(tc)
                    if tcid:
                        pending[tcid] = True
                msg["tool_calls"] = calls
            elif role == "tool":
                tcid = msg.get("tool_call_id")
                if not tcid or not pending.pop(tcid, False):
                    continue  # its assistant turn was evicted or condensed
            cleaned.append(msg)
        if pending:
            for i, msg in enumerate(cleaned):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    kept = [tc for tc in msg["tool_calls"]
                            if not (isinstance(tc, dict) and tc.get("id") in pending)]
                    if len(kept) != len(msg["tool_calls"]):
                        msg = dict(msg)
                        if kept:
                            msg["tool_calls"] = kept
                        else:
                            msg.pop("tool_calls")
                        cleaned[i] = msg
        return cleaned

    def _validate_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize a prompt before sending: valid roles, no None content,
        consecutive assistants merged, tool pairing enforced, empty assistant
        turns dropped. An empty list signals a terminal state (the prompt would
        end on a plain assistant turn)."""
        validated = []
        for msg in messages:
            role = msg.get("role")
            if (role not in _VALID_ROLES or (role == "tool" and not msg.get("tool_call_id"))
                    or (role == "function" and not msg.get("name"))):
                continue
            clean = dict(msg)
            if clean.get("content") is None:
                clean["content"] = ""
            if "reasoning_content" in clean and not (self._send_reasoning and role == "assistant"
                                                     and isinstance(clean["reasoning_content"], str)):
                del clean["reasoning_content"]
            validated.append(clean)
        for _ in range(3):  # each repair can expose another (e.g. an orphan between two assistants)
            before = len(validated)
            validated = self._merge_assistant_runs(validated)
            validated = self._sanitize_tool_pairing(validated)
            validated = [m for m in validated if not _is_empty_assistant(m)]
            if len(validated) == before:
                break
        if validated and validated[-1].get("role") == "assistant" and not validated[-1].get("tool_calls"):
            return []
        return validated

    @staticmethod
    def _assistant_response(content, tool_calls=None, reasoning=None, terminal=False, stopped=False):
        """An OpenAI-shaped response dict as consumed by the agent loop."""
        message = {"role": "assistant", "content": content, "tool_calls": tool_calls,
                   "reasoning_content": reasoning}
        if terminal:
            message["synthesized_terminal"] = True
        if stopped:
            message["stopped"] = True
        return {"choices": [{"message": message}]}

    def call_llm_api(self, messages: list[dict[str, Any]], use_tools: bool = True,
                     stream_display: bool = True, stream_label: str | None = None,
                     enable_thinking: bool = True, show_thinking: bool | None = None) -> dict[str, Any]:
        """Call the chat-completions API with streaming and return the assembled
        response.

        With ``stream_display`` the output streams as LLM_STREAM/THINKING_STREAM;
        internal calls pass False and either stream to the console pane
        (CONSOLE_STREAM, tagged once with ``stream_label``) or, without a label,
        stay silent. ``enable_thinking`` toggles the model's reasoning mode;
        ``show_thinking`` overrides the instance flag for this call. Transient
        errors are retried with backoff; a stop request returns immediately.
        """
        if show_thinking is None:
            show_thinking = self.show_thinking
        messages = self._validate_messages(messages)
        if not messages:
            self._log("[SAFETY STOP] The conversation already ends on an assistant reply; nothing to send.")
            return self._assistant_response(
                f"I have completed my response. No further actions are needed.\n\n{self.COMPLETION_MARKER}",
                terminal=True)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # AIC_TEMPERATURE_TOOL / AIC_TEMPERATURE_TEXT retune these per deployment.
            "temperature": (_env_float("AIC_TEMPERATURE_TOOL", 0.7) if use_tools
                            else _env_float("AIC_TEMPERATURE_TEXT", 1.0)),
            "stream": True,
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        # vLLM/SGLang extensions; servers that reject them get a retry without
        # (or disable them up front with AIC_STREAM_OPTIONS=0 / AIC_EXTRA_BODY=0).
        if _env_bool("AIC_STREAM_OPTIONS", True):
            params["stream_options"] = {"include_usage": True}
        if "gpt" not in self.model:  # OpenAI rejects these
            params["max_tokens"] = self.max_tokens
            if _env_bool("AIC_EXTRA_BODY", True):
                params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if use_tools:
            params["tools"] = self.tool_schemas
            params["tool_choice"] = "auto"
        self._emit_token_usage(self._estimate_input_tokens(messages), 0, estimate=True)

        max_attempts, extras_dropped = 5, False
        for attempt in range(1, max_attempts + 1):
            try:
                return self._stream_completion(params, stream_display, stream_label, show_thinking)
            except Exception as exc:
                if self.stop_event.is_set():
                    return self._assistant_response(None, stopped=True)
                if (self._send_reasoning and getattr(exc, "status_code", None) in (400, 422)
                        and "reasoning" in str(exc).lower()
                        and any("reasoning_content" in m for m in params["messages"])):
                    # e.g. DeepSeek's older reasoner API refuses its reasoning back as input
                    self._send_reasoning = False
                    params["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"}
                                          for m in params["messages"]]
                    self._log("[API] The endpoint rejected earlier reasoning in the request; "
                              "continuing without sending it back.", style="yellow")
                    continue
                if not extras_dropped and ("stream_options" in params or "extra_body" in params) \
                        and _is_param_rejection(exc):
                    extras_dropped = True
                    params.pop("stream_options", None)
                    params.pop("extra_body", None)
                    self._log("[API] The endpoint rejected stream_options/extra_body; retrying without them.",
                              style="yellow")
                    continue
                if attempt == max_attempts or not _is_retryable(exc):
                    raise
                wait = min(60, 2 ** attempt)
                self._log(f"[API ERROR] {_describe_error(exc)}. Retrying in {wait}s "
                          f"(attempt {attempt}/{max_attempts}).", style="yellow")
                self.sink.emit("STATUS_UPDATE", {"step": self.step_count, "max_steps": self.max_steps,
                                                 "phase": "api_retry",
                                                 "retry": f"retry {attempt}/{max_attempts} in {wait}s"})
                if self.stop_event.wait(wait):
                    return self._assistant_response(None, stopped=True)
        raise RuntimeError("unreachable")

    def _stream_completion(self, params: dict, stream_display: bool, stream_label: str | None,
                           show_thinking: bool) -> dict[str, Any]:
        silent = not stream_display and not stream_label
        content, thinking, calls = [], [], []
        usage = first_token = None
        started = time.monotonic()
        label = stream_label
        stream = self.client.chat.completions.create(**params)
        self._active_streams.add(stream)
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage  # the final chunk carries exact usage
                if self.stop_event.is_set():
                    self._log("[STOP] Agent shutdown requested. Aborting the LLM stream.")
                    return self._assistant_response(None, stopped=True)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                reasoning = (getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                             or getattr(delta, "thinking", None))
                if reasoning:
                    first_token = first_token or time.monotonic()
                    thinking.append(reasoning)
                    if show_thinking and not silent:
                        self.sink.emit("THINKING_STREAM" if stream_display else "CONSOLE_STREAM", {"text": reasoning})
                if delta.content:
                    first_token = first_token or time.monotonic()
                    content.append(delta.content)
                    if stream_display:
                        self.sink.emit("LLM_STREAM", {"text": delta.content})
                    elif not silent:
                        self.sink.emit("CONSOLE_STREAM", {"text": delta.content, **({"label": label} if label else {})})
                        label = None
                for tc in delta.tool_calls or []:
                    first_token = first_token or time.monotonic()
                    # Providers that omit "index" stream one call at a time: a
                    # chunk with an id starts a call, one without continues it.
                    if tc.index is not None:
                        index = tc.index
                    else:
                        index = len(calls) if (tc.id or not calls) else len(calls) - 1
                    while len(calls) <= index:
                        calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    entry = calls[index]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        entry["function"]["name"] += tc.function.name or ""
                        entry["function"]["arguments"] += tc.function.arguments or ""
        finally:
            self._active_streams.discard(stream)
            with contextlib.suppress(Exception):
                stream.close()

        tool_calls = []
        for i, call in enumerate(calls):
            if not call["function"]["name"] and not call["function"]["arguments"]:
                continue
            call["id"] = call["id"] or f"call_{int(time.time() * 1000)}_{i}"  # some servers omit ids
            tool_calls.append(call)
        if usage is not None:
            completion = getattr(usage, "completion_tokens", 0) or 0
            tps = getattr(usage, "completion_tokens_per_sec", None)
            if not tps and completion >= 8:  # too few tokens give a meaningless rate
                elapsed = time.monotonic() - (first_token or started)  # generation time, not prefill
                tps = completion / elapsed if elapsed > 0.2 else None
            self._emit_token_usage(getattr(usage, "prompt_tokens", 0) or 0, completion,
                                   cached=self._extract_cached_tokens(usage), tokens_per_sec=tps)
        # reasoning_content lets the loop detect reasoning-only turns; it is not stored.
        return self._assistant_response("".join(content) or None, tool_calls or None, "".join(thinking) or None)

    @staticmethod
    def _tool_result(tool_call_id, content: str) -> dict[str, str]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def process_llm_response(self, response: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None,
                                                                       str | None, list[dict[str, Any]]]:
        """Split a response into (content, tool_calls_info, first_tool_call_id,
        thinking, malformed_tool_calls). tool_calls_info entries carry
        tool_call_id, function_name, function_arguments and command; malformed
        entries carry tool_call_id, function_name, raw_arguments, parse_error."""
        message = response["choices"][0]["message"]
        content = message.get("content")
        thinking = message.get("reasoning_content") or message.get("thinking")
        tool_calls_info: list[dict[str, Any]] = []
        malformed: list[dict[str, Any]] = []
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function") or {}
            name, raw = fn.get("name"), fn.get("arguments")
            parsed = None
            if isinstance(raw, dict):  # some gateways return already-parsed arguments
                parsed = raw
            elif raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._log_error(f"Tool call arguments could not be parsed as JSON: {exc}")
                    malformed.append({"tool_call_id": tool_call.get("id"), "function_name": name,
                                      "raw_arguments": raw, "parse_error": str(exc)})
                    continue
            command = None
            if name == "execute_bash":
                command = parsed.get("command") if isinstance(parsed, dict) else parsed
                command = command if isinstance(command, str) else None
            tool_calls_info.append({"tool_call_id": tool_call.get("id"), "function_name": name,
                                    "function_arguments": raw, "command": command})
        first_id = tool_calls_info[0]["tool_call_id"] if tool_calls_info else None
        return (content if isinstance(content, str) else ""), tool_calls_info, first_id, thinking, malformed

    def handle_function_call(self, function_info: dict[str, Any]) -> str:
        """Tool result for calls that carry no runnable bash command."""
        name, raw = function_info.get("name"), function_info.get("arguments")
        if isinstance(raw, dict):
            return f"Unknown function: {name}"
        if not raw:
            return "No arguments provided for function call"
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            return (f"ERROR: The JSON arguments provided for function '{name}' could not be parsed. "
                    f"Parse error: {exc}. Please re-issue this tool call with valid, properly formatted JSON "
                    f"arguments (double-quoted strings, newlines escaped as \\n, no trailing commas, a "
                    f"single JSON object).")
        return f"Unknown function: {name}"

    # --- The agent loop ----------------------------------------------------------

    def inject_suggestion(self, suggestion: str):
        """Queue a mid-run steering message, delivered at the start of the next step."""
        if suggestion and suggestion.strip():
            self.suggestion_queue.put(suggestion.strip())

    def _drain_suggestions(self):
        while True:
            try:
                suggestion = self.suggestion_queue.get_nowait()
            except queue.Empty:
                return
            self.conversation_history.append({"role": "user", "content": f"[Suggestion from user] {suggestion}"})
            self._log(f"[USER SUGGESTION] {suggestion}", style="bold bright_yellow")

    def generate_task_summary(self, user_request: str, generation: int = 0):
        """Background: summarize the request in 10-15 words for the panel title
        (falls back to the truncated prompt). An outdated summary is discarded."""
        prompt = (user_request or "").strip()
        if not prompt:
            return
        try:
            response = self.call_llm_api(
                [{"role": "system", "content": "You are a precise summarizer."},
                 {"role": "user", "content": "Summarize the following task in 10 to 15 words. Reply with the "
                                             "summary only, no quotes, no punctuation at the end.\n\n" + prompt}],
                use_tools=False, stream_display=False, enable_thinking=False, show_thinking=False)
            summary = " ".join((response["choices"][0]["message"].get("content") or "").split())
        except Exception as exc:
            if self.debug:
                self._log(f"[TASK TITLE] LLM summary failed: {exc}", style="yellow")
            summary = ""
        summary = summary.replace(self.COMPLETION_MARKER, "").strip() or " ".join(prompt.split())[:80]
        if generation != self._summary_generation:
            return  # a newer request already took over the title
        self.task_summary = summary
        self.sink.emit("TASK_TITLE", {"summary": summary})
        self.save_session("task_summary")

    def _continuation_prompt(self) -> str:
        return (f"{_CONTINUATION_PREFIX} If the task is truly complete, you MUST emit the completion signal "
                f"\"{self.COMPLETION_MARKER}\" on its own. Otherwise, continue by using the appropriate tools to "
                f"make progress. Do NOT simply describe what needs to be done - take the next concrete step "
                f"using execute_bash or other available tools. Remember: you must keep going until the task is "
                f"fully finished, then emit \"{self.COMPLETION_MARKER}\".")

    def _run_tool_calls(self, calls: list[dict[str, Any]], malformed: list[dict[str, Any]]):
        """Answer every tool call of one assistant turn, in order. User
        suggestions given while rejecting go after all results, keeping the
        tool results adjacent to their assistant turn."""
        if malformed:
            self._log(f"[INFO] {len(malformed)} tool call(s) had malformed JSON arguments; "
                      f"asking the model to re-issue them.")
        for bad in malformed:
            self.conversation_history.append(self._tool_result(bad["tool_call_id"], (
                f"ERROR: The tool call with id '{bad['tool_call_id']}' (function: '{bad['function_name']}') "
                f"contained malformed JSON arguments that could not be parsed.\n\nParse error: "
                f"{bad['parse_error']}\n\nYour function arguments MUST be a valid JSON object. Ensure:\n"
                f"1. All strings are properly quoted with double quotes\n2. No unescaped newline characters "
                f"inside strings (use \\n instead of literal newlines)\n3. No trailing commas\n4. The entire "
                f"argument string is a single valid JSON object\n\nPlease re-issue the tool call with "
                f"corrected, valid JSON arguments. If you cannot produce valid JSON for this tool call, "
                f"respond with a normal text message instead.")))
        suggestions = []
        for index, call in enumerate(calls):
            if self.stop_event.is_set():
                for rest in calls[index:]:
                    self.conversation_history.append(self._tool_result(
                        rest["tool_call_id"], "Not executed: the user stopped the agent before this command ran."))
                break
            tcid, command, name = call["tool_call_id"], call.get("command"), call.get("function_name")
            if not command:
                if name == "execute_bash":
                    result = ("ERROR: execute_bash requires a 'command' string argument. Re-issue the tool "
                              "call with a JSON object like {\"command\": \"...\"}.")
                else:
                    result = self.handle_function_call({"name": name, "arguments": call["function_arguments"]})
                self.conversation_history.append(self._tool_result(tcid, result))
                self.save_session("tool_result")
                continue
            self.sink.emit("CMD_EXEC", {"command": command, "tool_call_id": tcid, "function_name": name})
            approved, suggestion = self.get_user_confirmation(command)
            if approved:
                try:
                    output, code = self.execute_bash_command(command)
                    result = f"Command executed with exit code {code}.\nOutput:\n{output}"
                except CommandTimeoutError as exc:
                    result = f"{exc}" + (f"\nOutput before the timeout:\n{exc.output}" if exc.output else "")
                except Exception as exc:
                    result = f"Command execution failed: {exc}"
                    self._log_error(result)
            else:
                self._log("[INFO] Command not approved by user.")
                result = "Command execution skipped by user."
                if suggestion:
                    suggestions.append(suggestion)
            self.conversation_history.append(self._tool_result(tcid, result))
            self.save_session("tool_result")  # a crash keeps every finished result
        for suggestion in suggestions:
            self.conversation_history.append({"role": "user", "content": f"I suggest you {suggestion}"})
        if suggestions:
            self.save_session("tool_result")

    def run(self, user_request: str):
        """Run the agent loop for one request until it completes, is stopped or
        reaches max_steps. With persist_history the request extends the
        existing conversation."""
        if not self._started_banner_shown:
            self._started_banner_shown = True
            self.sink.emit("SYSTEM", {"text": "[AI-COMMANDER STARTED]", "api_base": self.base_url,
                                      "model": self.model, "auto_approve": self.auto_approve,
                                      "max_prompt_len": self.max_prompt_len,
                                      "compress_algorithm": self.compress_algorithm, "fast": self.fast})
        self.sink.emit("USER_REQUEST", {"text": user_request})
        self.initial_prompt = user_request
        self._summary_generation += 1
        generation = self._summary_generation
        request = {"role": "user", "content": user_request}
        if self.persist_history and self.conversation_history:
            self._refresh_system_prompt()
            for msg in self.conversation_history:  # earlier runs' reasoning is no longer needed
                msg.pop("reasoning_content", None)
            self.conversation_history.append(request)
        else:
            self.conversation_history = [{"role": "system", "content": self.get_system_prompt()}, request]
        self._current_request = request
        self.save_session("user_request")  # the request survives even if the first call dies

        step = empty_turns = overflow_retries = 0
        completed = False
        try:
            while step < self.max_steps and not self.stop_event.is_set():
                step += 1
                self.step_count = step
                self.sink.emit("STATUS_UPDATE", {"step": step, "max_steps": self.max_steps, "phase": "llm_call"})
                if self.debug:
                    self._dump_conversation_history()
                self._drain_suggestions()
                self._context_compress(force=overflow_retries > 0)
                # Compression can split an assistant/tool pair; repair before saving/sending.
                self.conversation_history = self._sanitize_tool_pairing(self.conversation_history)
                self.save_session("before_llm_call")
                try:
                    response = self.call_llm_api(list(self.conversation_history))
                except Exception as exc:
                    if self.stop_event.is_set():
                        break
                    if _is_context_overflow(exc) and overflow_retries < 2:
                        overflow_retries += 1
                        self._log("[API] The prompt exceeded the model's context window; compressing harder "
                                  "and retrying.", style="yellow")
                        continue
                    self._log_error(f"[API ERROR] {_describe_error(exc)}")
                    break
                overflow_retries = 0
                if step == 1 and self.task_summary_enabled:  # after the first reply: never delays the task
                    threading.Thread(target=self.generate_task_summary, args=(user_request, generation),
                                     daemon=True).start()
                message = response["choices"][0]["message"]
                if message.get("synthesized_terminal") or message.get("stopped"):
                    break
                content, calls, _first_id, thinking, malformed = self.process_llm_response(response)
                if not content.strip() and not message.get("tool_calls"):
                    # Nothing to store. A reasoning-only turn gets a nudge; a truly
                    # empty one is retried as is. Three in a row end the run.
                    empty_turns += 1
                    if empty_turns >= 3:
                        self._log_error("[STOP] The model returned three empty responses in a row.")
                        break
                    self._log("[INFO] Empty assistant response; retrying this step.")
                    if thinking and not _is_continuation(self.conversation_history[-1]):
                        self.conversation_history.append({"role": "user", "content": self._continuation_prompt()})
                    continue
                empty_turns = 0
                # The reasoning stays with the turn for the rest of this run ("preserved
                # thinking"): models that see their earlier turns without it learn to
                # skip thinking (see _validate_messages).
                stored = {k: v for k, v in message.items()
                          if k not in ("synthesized_terminal", "stopped") and v is not None}
                stored.setdefault("content", None)
                if self.conversation_history[-1].get("role") == "assistant":  # never two assistants in a row
                    self.conversation_history.append({"role": "user", "content": _CONTINUE_PROMPT})
                self.conversation_history.append(stored)
                # Saved before any command runs: a crash mid-command leaves this
                # (unanswered) call on disk, closed as "not re-executed" on reload.
                self.save_session("assistant_message")
                if calls or malformed:
                    self._run_tool_calls(calls, malformed)
                elif self.COMPLETION_MARKER not in content:
                    self.conversation_history.append({"role": "user", "content": self._continuation_prompt()})
                    self.save_session("continuation_prompt")
                if self.COMPLETION_MARKER in content:
                    completed = True
                    break
            else:
                if step >= self.max_steps:
                    self._log(f"[LIMIT REACHED] Maximum steps ({self.max_steps}) reached.", style="yellow")
        finally:
            self.save_session("loop_terminated")
            self.sink.emit("SHUTDOWN", {"reason": "agent_loop_terminated", "steps": step, "completed": completed})
        self.sink.close()


# --- The TUI --------------------------------------------------------------------

if _TEXTUAL_AVAILABLE:
    _WRAP_TOKEN_RE = re.compile(r"\S+\s*|\s+")
    _NEWLINE_RE = re.compile(r"(\r\n|\n|\r)")
    _WORD_CHARS = re.compile(r"[\w\-./~:@%+=?&#,]")
    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _SLASH_COMMANDS = ("/help", "/autoapprove", "/aa", "/approve", "/reject", "/suggest ", "/clear", "/new",
                       "/session", "/reload", "/copy", "/stop", "/quit", "/exit")
    # Palette (Norton Commander blues).
    _C_USER, _C_ANSWER, _C_THINK = "bold #ffff5f", "#87d7ff", "italic #b2b28a"
    _C_CMD, _C_OK, _C_FAIL, _C_DIM = "bold #ffd75f", "#5fd787", "#ff8787", "#8fa8cc"
    _SELECTION = Style(color="#000000", bgcolor="#87d7ff")
    _LOG_STYLES = {"yellow": "#ffd75f", "green": "#5fd787", "red": "#ff8787", "cyan": "#87d7ff",
                   "dim": _C_DIM, "bright_yellow": _C_USER, "bold bright_yellow": _C_USER}

    def _fmt_tokens(n: float) -> str:
        return f"{n:.0f}" if n < 1000 else f"{n / 1000:.1f}k" if n < 1_000_000 else f"{n / 1e6:.1f}M"

    def _fmt_secs(s: float) -> str:
        return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"

    def _fit_chars(text: str, start: int, cells: int) -> int:
        """How many characters from *start* fit in *cells* columns (at least 1)."""
        used = 0
        for count, ch in enumerate(text[start:]):
            used += cell_len(ch)
            if used > cells:
                return max(1, count)
        return max(1, len(text) - start)

    def _wrap_spans(text: str, width: int) -> list[tuple[int, int]]:
        """Greedy word wrap of *text* to *width* cells: (start, end) per row.
        Over-long words are folded; trailing spaces may overhang the edge."""
        width = max(1, width)
        if len(text) <= width and (text.isascii() or cell_len(text) <= width):
            return [(0, len(text))]
        measure = len if text.isascii() else cell_len
        rows, row_start, col = [], 0, 0
        for match in _WRAP_TOKEN_RE.finditer(text):
            start, end = match.span()
            visible = measure(text[start:end].rstrip() or text[start:end])
            if col and col + visible > width:
                rows.append((row_start, start))
                row_start, col = start, 0
            while visible > width - col:
                cut = start + (width - col if measure is len else _fit_chars(text, start, width - col))
                rows.append((row_start, cut))
                row_start, start, col = cut, cut, 0
                visible = measure(text[start:end].rstrip() or text[start:end])
            col += measure(text[start:end])
        rows.append((row_start, len(text)))
        return rows

    def _tool_command(tool_call) -> str | None:
        try:
            args = json.loads((tool_call.get("function") or {}).get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        command = args.get("command") if isinstance(args, dict) else None
        return command if isinstance(command, str) else None

    class _Line:
        """A logical log line: its text plus style runs [(end_offset, style)]."""
        __slots__ = ("plain", "runs", "rule")

        def __init__(self, rule: bool = False):
            self.plain, self.runs, self.rule = "", [], rule

        def append(self, text: str, style) -> None:
            self.plain += text
            if self.runs and self.runs[-1][1] == style:
                self.runs[-1] = (len(self.plain), style)
            else:
                self.runs.append((len(self.plain), style))

        def text(self, start: int, end: int) -> Text:
            out = Text(self.plain[start:end])
            pos = 0
            for run_end, style in self.runs:
                a, b = max(pos, start), min(run_end, end)
                if a < b and style:
                    out.stylize(style, a - start, b - start)
                pos = run_end
                if pos >= end:
                    break
            return out

    class OutputLog(ScrollView, can_focus=True):
        """Scrolling text log with live streaming, re-wrapping on resize and
        mouse selection (copies re-join soft-wrapped lines)."""

        ALLOW_SELECT = True
        DEFAULT_CSS = "OutputLog { overflow-x: hidden; overflow-y: scroll; }"

        def __init__(self, max_lines: int = 20000, **kwargs):
            super().__init__(**kwargs)
            self.max_lines = max_lines
            self._lines: list[_Line] = [_Line()]  # the last line is the open (streaming) one
            self._rows: list[tuple[int, int, int]] = []  # (line index, start, end) per visual row
            self._tail = 0  # rows belonging to the open line (always at the end of _rows)
            self._tail_line: _Line | None = None  # the line those rows were computed for
            self._width = 0
            self._cache: dict[int, Strip] = {}
            self._cr = False  # a '\r' ended the last chunk: decided by what follows

        # Writing -------------------------------------------------------------

        def write(self, text: str, style="") -> None:
            """Append streamed text: '\\n' ends a line, a lone '\\r' rewinds it."""
            if not text:
                return
            follow = self._at_bottom()
            for part in _NEWLINE_RE.split(_CTRL_RE.sub("", text.expandtabs(4))):
                if not part:
                    continue
                if part == "\r":
                    self._cr = True
                    continue
                if self._cr:
                    self._cr = False
                    if part not in ("\n", "\r\n"):
                        self._lines[-1] = _Line()
                if part in ("\n", "\r\n"):
                    self._newline()
                else:
                    self._lines[-1].append(part, style)
            self._settle(follow)

        def write_line(self, text: str, style="") -> None:
            """Write complete line(s), ending any open streamed line first."""
            self.close_line()
            self.write(text + "\n", style)

        def write_rule(self, title: str = "", style="") -> None:
            """A full-width horizontal rule with an optional title."""
            follow = self._at_bottom()
            self.close_line(settle=False)
            rule = _Line(rule=True)
            rule.append(" ".join(title.split()), style)
            self._lines[-1] = rule
            self._newline()
            self._settle(follow)

        def close_line(self, settle: bool = True) -> None:
            self._cr = False
            if self._lines[-1].plain:
                follow = self._at_bottom()
                self._newline()
                if settle:
                    self._settle(follow)

        def clear(self) -> None:
            self._lines, self._rows, self._tail, self._cr = [_Line()], [], 0, False
            self._cache.clear()
            self.virtual_size = Size(self._width, 0)
            self.scroll_to(y=0, animate=False, immediate=True)
            self.refresh()

        @property
        def plain_text(self) -> str:
            return "\n".join(line.plain for line in self._lines).rstrip("\n")

        def _newline(self) -> None:
            self._retail(final=True)
            self._lines.append(_Line())
            self._tail = 0

        def _line_rows(self, i: int) -> list[tuple[int, int, int]]:
            line = self._lines[i]
            if line.rule:
                return [(i, 0, len(line.plain))]
            return [(i, s, e) for s, e in _wrap_spans(line.plain, self._width or 1_000_000)]

        def _retail(self, final: bool = False) -> None:
            """Recompute the rows of the open (last) line. Appending text can only
            change its last row (greedy wrapping), so only that row is redone."""
            i = len(self._lines) - 1
            line = self._lines[i]
            keep = self._tail - 1 if self._tail and self._tail_line is line and not line.rule else 0
            first = len(self._rows) - self._tail + keep
            start = self._rows[first - 1][2] if keep else 0
            del self._rows[first:]
            if line.rule:
                rows = [(i, 0, len(line.plain))]
            elif final or line.plain:
                rows = [(i, start + a, start + b)
                        for a, b in _wrap_spans(line.plain[start:], self._width or 1_000_000)]
            else:
                rows = []
            self._rows.extend(rows)
            self._tail, self._tail_line = keep + len(rows), line
            for key in [k for k in self._cache if k >= first]:
                del self._cache[key]

        def _settle(self, follow: bool) -> None:
            self._retail()
            excess = len(self._lines) - self.max_lines
            if excess > max(50, self.max_lines // 10):
                self._trim(excess)
            self.virtual_size = Size(self._width, len(self._rows))
            if follow:
                self.scroll_end(animate=False, immediate=True, x_axis=False)
            self.refresh()

        def _trim(self, count: int) -> None:
            """Drop the oldest *count* lines, keeping the viewport on the same text."""
            removed = bisect.bisect_left(self._rows, (count, -1, -1))
            del self._lines[:count]
            self._rows = [(i - count, s, e) for i, s, e in self._rows[removed:]]
            self._cache.clear()
            with contextlib.suppress(Exception):
                if self.text_selection is not None:
                    self.screen.selections = {w: s for w, s in self.screen.selections.items() if w is not self}
            self.scroll_to(y=max(0, self.scroll_y - removed), animate=False, immediate=True)

        def _at_bottom(self) -> bool:
            try:
                if self.screen._selecting:  # never scroll under a drag selection
                    return False
            except Exception:
                pass
            return self.scroll_y >= self.max_scroll_y - 1

        def on_resize(self, event: events.Resize) -> None:
            width = self.scrollable_content_region.width
            if width <= 0 or width == self._width:
                return
            follow = self._at_bottom()
            top = self._rows[min(int(self.scroll_y), len(self._rows) - 1)] if self._rows else None
            self._width = width
            self._rows, self._tail = [], 0
            for i in range(len(self._lines) - 1):
                self._rows.extend(self._line_rows(i))
            self._retail()
            self._cache.clear()
            self.virtual_size = Size(self._width, len(self._rows))
            if follow:
                self.scroll_end(animate=False, immediate=True, x_axis=False)
            elif top:
                row = bisect.bisect_right(self._rows, (top[0], top[1], 1 << 60)) - 1
                self.scroll_to(y=max(0, row), animate=False, immediate=True)
            self.refresh()

        # Rendering and selection ---------------------------------------------

        @staticmethod
        def _rule_text(line: _Line, width: int) -> Text:
            style = line.runs[0][1] if line.runs else ""
            title = line.plain
            if not title:
                return Text("─" * width, style)
            if cell_len(title) > width - 6:
                title = title[:max(1, width - 7)] + "…"
            out = Text("── ", style)
            out.append(title, style)
            out.append(" " + "─" * max(0, width - cell_len(title) - 4), style)
            return out

        def render_line(self, y: int) -> Strip:
            row = int(self.scroll_offset.y) + y
            width = self.scrollable_content_region.width
            base = self.rich_style
            if row >= len(self._rows) or width <= 0:
                return Strip.blank(max(0, width), base)
            selection = self.text_selection
            span = selection.get_span(row) if selection is not None else None
            if span is None:
                cached = self._cache.get(row)
                if cached is not None and cached.cell_length == width:
                    return cached
            i, start, end = self._rows[row]
            line = self._lines[i]
            text = self._rule_text(line, width) if line.rule else line.text(start, end)
            if span is not None:
                a, b = span
                text.stylize(_SELECTION, a, len(text) if b == -1 else b)
            strip = (Strip(list(text.render(self.app.console))).apply_style(base)
                     .crop_extend(0, width, base).apply_offsets(0, row))
            if span is None:
                if len(self._cache) > 2048:
                    self._cache.clear()
                self._cache[row] = strip
            return strip

        def get_selection(self, selection: Selection) -> tuple[str, str] | None:
            if not self._rows:
                return None
            start = selection.start or Offset(0, 0)
            end = selection.end or Offset(1 << 30, len(self._rows) - 1)
            last = min(end.y, len(self._rows) - 1)
            parts = []
            for row in range(max(0, start.y), last + 1):
                i, s, e = self._rows[row]
                line = self._lines[i]
                if line.rule:
                    parts.append(line.plain)
                else:
                    a = min(e, s + (start.x if row == start.y else 0))
                    b = min(e, s + end.x) if row == end.y else e
                    parts.append(line.plain[a:max(a, b)])
                if row < last and self._rows[row + 1][0] != i:
                    parts.append("\n")  # soft-wrapped rows of one line are re-joined
            return "".join(parts), "\n"

        def selection_updated(self, selection: Selection | None) -> None:
            self.refresh()

        def _offset_at(self, event: events.MouseEvent) -> tuple[int, int] | None:
            """(character index in its logical line, line index) under the mouse."""
            offset = event.get_content_offset(self)
            if offset is None:
                return None
            row = int(self.scroll_offset.y) + offset.y
            if not 0 <= row < len(self._rows):
                return None
            i, s, e = self._rows[row]
            cells = 0
            for k, ch in enumerate(self._lines[i].plain[s:e]):
                cells += cell_len(ch)
                if cells > offset.x:
                    return s + k, i
            return e, i

        def _position(self, i: int, char: int) -> Offset:
            """Selection offset (x within row, row) of a character of line *i*."""
            first = bisect.bisect_left(self._rows, (i, -1, -1))
            row = first
            while row + 1 < len(self._rows) and self._rows[row + 1][0] == i and self._rows[row + 1][1] <= char:
                row += 1
            return Offset(char - self._rows[row][1], row)

        async def _on_click(self, event: events.Click) -> None:
            if event.chain < 2:
                return
            event.prevent_default()  # instead of Textual's select-the-whole-widget
            hit = self._offset_at(event)
            if hit is None:
                return
            char, i = hit
            plain = self._lines[i].plain
            if event.chain == 2:
                a = b = char
                while a > 0 and _WORD_CHARS.match(plain[a - 1]):
                    a -= 1
                while b < len(plain) and _WORD_CHARS.match(plain[b]):
                    b += 1
                if a == b:
                    return
            else:
                a, b = 0, len(plain)
            self.screen.selections = {self: Selection(self._position(i, a), self._position(i, b))}
            self.app.copy_selection()

        async def _on_mouse_down(self, event: events.MouseDown) -> None:
            if event.button in (2, 3):
                event.prevent_default()
                self.app.mouse_paste_or_copy(event.button, target=None)

        def on_paste(self, event: events.Paste) -> None:
            """Logs are read-only: pasting or typing here goes to the prompt."""
            event.stop()
            self.app.prompt.focus()
            self.app.prompt.paste_text(event.text)

        def on_key(self, event: events.Key) -> None:
            if event.is_printable and event.character:
                event.stop()
                self.app.prompt.focus()
                self.app.prompt.paste_text(event.character)

    class PromptArea(TextArea):
        """Multi-line prompt: Enter sends, Ctrl+J / Shift+Enter add a line,
        Up/Down walk the history at the first/last line, Ctrl+R searches it,
        Tab completes /commands. Multi-line pastes are kept intact."""

        class Submitted(Message):
            def __init__(self, text: str):
                super().__init__()
                self.text = text

        BINDINGS = [
            Binding("ctrl+j,shift+enter,alt+enter", "newline", "New line", show=False),
            Binding("ctrl+r", "search_history", "Search history", show=False),
        ]
        HINT = "Enter send · Ctrl+J newline · ↑↓ history · Ctrl+R search · F1 help"

        def __init__(self, history: PromptHistory, **kwargs):
            super().__init__(soft_wrap=True, tab_behavior="focus", show_line_numbers=False, **kwargs)
            self.prompt_history = history
            self._hist_pos: int | None = None
            self._pending = ""

        def on_mount(self) -> None:
            self.border_subtitle = self.HINT

        def set_text(self, text: str) -> None:
            self.text = text
            self.move_cursor(self.document.end)

        def paste_text(self, text: str) -> None:
            result = self.replace(text, *self.selection)
            self.move_cursor(result.end_location)

        def action_newline(self) -> None:
            self.paste_text("\n")

        def action_search_history(self) -> None:
            def done(match: str | None) -> None:
                if match is not None:
                    self.set_text(match)
                self.focus()
            self.app.push_screen(ReverseSearchScreen(self.prompt_history.items), done)

        def _at_top(self) -> bool:
            return self.cursor_at_first_line and self.navigator.is_first_wrapped_line(self.cursor_location)

        def _at_bottom(self) -> bool:
            return self.cursor_at_last_line and self.navigator.is_last_wrapped_line(self.cursor_location)

        def _browse(self, step: int) -> None:
            items = self.prompt_history.items
            if step < 0:
                if not items or self._hist_pos == 0:
                    return
                if self._hist_pos is None:
                    self._pending, self._hist_pos = self.text, len(items)
                self._hist_pos -= 1
                self.set_text(items[self._hist_pos])
            elif self._hist_pos is not None:
                self._hist_pos += 1
                if self._hist_pos >= len(items):
                    self._hist_pos = None
                    self.set_text(self._pending)
                else:
                    self.set_text(items[self._hist_pos])

        def _complete_command(self) -> bool:
            prefix = self.text.lower()
            matches = [c for c in _SLASH_COMMANDS if c.startswith(prefix)]
            if not matches:
                return False
            common = os.path.commonprefix(matches)
            if len(matches) == 1:
                self.set_text(common if common.endswith(" ") else common + " ")
            elif common != prefix:
                self.set_text(common)
            else:
                self.app.notify("  ".join(m.strip() for m in matches), timeout=3)
            return True

        async def _on_key(self, event: events.Key) -> None:
            key = event.key
            if key == "enter":
                event.stop()
                event.prevent_default()
                if self.text.strip():
                    self.post_message(self.Submitted(self.text))
                self.clear()
                self._hist_pos = None
            elif (key in ("up", "ctrl+p") and self._at_top()) or (key in ("down", "ctrl+n") and self._at_bottom()
                                                                   and self._hist_pos is not None):
                event.stop()
                event.prevent_default()
                self._browse(-1 if key in ("up", "ctrl+p") else 1)
            elif key == "tab" and self.text.startswith("/") and "\n" not in self.text and self._complete_command():
                event.stop()
                event.prevent_default()
            # anything else: TextArea's own handler runs next (Textual walks the MRO)

        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            text = self.text
            if text.startswith("/") and "\n" not in text and " " not in text:
                matches = [c.strip() for c in _SLASH_COMMANDS if c.startswith(text.lower())]
                self.border_subtitle = ("Tab completes: " + " ".join(matches)) if matches else "unknown command"
            else:
                self.border_subtitle = self.HINT

        async def _on_mouse_down(self, event: events.MouseDown) -> None:
            if event.button in (2, 3):
                event.stop()
                event.prevent_default()
                self.app.mouse_paste_or_copy(event.button, target=self)

    class ShellWidget(Widget, can_focus=True):
        """An interactive shell in a PTY, emulated with pyte: colors, scrollback
        (mouse wheel, Shift+PgUp/PgDn), mouse selection and bracketed paste.
        Every key except the app's F-keys goes to the shell."""

        ALLOW_SELECT = True
        _KEYS = {"enter": b"\r", "backspace": b"\x7f", "tab": b"\t", "shift+tab": b"\x1b[Z", "escape": b"\x1b",
                 "up": b"\x1b[A", "down": b"\x1b[B", "right": b"\x1b[C", "left": b"\x1b[D",
                 "home": b"\x1b[H", "end": b"\x1b[F", "insert": b"\x1b[2~", "delete": b"\x1b[3~",
                 "pageup": b"\x1b[5~", "pagedown": b"\x1b[6~", "ctrl+left": b"\x1b[1;5D",
                 "ctrl+right": b"\x1b[1;5C", "ctrl+up": b"\x1b[1;5A", "ctrl+down": b"\x1b[1;5B",
                 "ctrl+space": b"\x00", "ctrl+@": b"\x00", "f11": b"\x1b[23~", "f12": b"\x1b[24~"}
        _COLOR_NAMES = {"brown": "yellow", "brightbrown": "bright_yellow", "brightblack": "bright_black",
                        "brightred": "bright_red", "brightgreen": "bright_green", "brightblue": "bright_blue",
                        "brightmagenta": "bright_magenta", "brightcyan": "bright_cyan",
                        "brightwhite": "bright_white"}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._fd: int | None = None
            self._pid: int | None = None
            self._screen = None
            self._stream = None
            self._scroll = 0  # rows scrolled back into history
            self._status = ""
            self._cwd: str | None = None
            self._outbuf = bytearray()
            self._styles: dict = {}

        def on_mount(self) -> None:
            self.set_interval(0.5, self._sync_cwd)

        def on_resize(self, event: events.Resize) -> None:
            cols, rows = self.size.width, self.size.height
            if cols < 4 or rows < 2:  # hidden in the ContentSwitcher
                return
            if self._screen is None and not self._status:
                self._spawn(cols, rows)
            elif self._screen is not None and (cols, rows) != (self._screen.columns, self._screen.lines):
                self._screen.resize(rows, cols)
                self._set_winsize()
            self.refresh()

        def _spawn(self, cols: int, rows: int) -> None:
            if _IS_WINDOWS or pyte is None:
                self._status = "The interactive shell needs a POSIX system and the 'pyte' package."
                return
            shell = os.environ.get("SHELL") or "/bin/sh"
            try:
                pid, fd = pty.fork()
            except OSError as exc:
                self._status = f"Could not start a shell: {exc}"
                return
            if pid == 0:
                try:
                    os.environ["TERM"] = "xterm-256color"
                    with contextlib.suppress(OSError):
                        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                    for candidate in (shell, "/bin/sh"):
                        with contextlib.suppress(OSError):
                            os.execv(candidate, [candidate])
                finally:
                    os._exit(127)
            self._pid, self._fd = pid, fd
            os.set_blocking(fd, False)
            self._screen = pyte.HistoryScreen(cols, rows, history=5000, ratio=0.5)
            self._screen.write_process_input = lambda data: self._send(data.encode())  # answers DSR queries
            self._stream = pyte.ByteStream(self._screen)
            self._status, self._scroll, self._cwd = "", 0, _safe_cwd()
            asyncio.get_running_loop().add_reader(fd, self._on_readable)

        def _set_winsize(self) -> None:
            if self._fd is not None:
                with contextlib.suppress(OSError):
                    fcntl.ioctl(self._fd, termios.TIOCSWINSZ,
                                struct.pack("HHHH", self._screen.lines, self._screen.columns, 0, 0))

        def _on_readable(self) -> None:
            try:
                data = os.read(self._fd, 1 << 18)
            except BlockingIOError:
                return
            except OSError:
                data = b""
            if not data:
                self._exited()
                return
            history = len(self._screen.history.top)
            self._stream.feed(data)
            self._scroll = 0
            if self.text_selection is not None and len(self._screen.history.top) != history:
                self.screen.selections = {w: s for w, s in self.screen.selections.items() if w is not self}
            self.refresh()

        def _exited(self) -> None:
            loop = asyncio.get_running_loop()
            loop.remove_reader(self._fd)
            with contextlib.suppress(Exception):
                loop.remove_writer(self._fd)
            with contextlib.suppress(OSError):
                os.close(self._fd)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(self._pid, os.WNOHANG)
            self._fd = self._pid = None
            self._outbuf.clear()
            self._status = "Shell exited — press Enter to start a new one."
            self.refresh()

        def _send(self, data: bytes) -> None:
            """Queue bytes for the PTY; asyncio writes them when it can (large pastes)."""
            if self._fd is None or not data:
                return
            self._outbuf += data
            self._flush_out()

        def _flush_out(self) -> None:
            try:
                while self._outbuf:
                    written = os.write(self._fd, self._outbuf[:4096])
                    del self._outbuf[:written]
            except BlockingIOError:
                pass
            except OSError:
                self._outbuf.clear()
            loop = asyncio.get_running_loop()
            if self._outbuf:
                loop.add_writer(self._fd, self._flush_out)
            else:
                with contextlib.suppress(Exception):
                    loop.remove_writer(self._fd)

        def _sync_cwd(self) -> None:
            """Follow the shell's cd: agent commands inherit the process cwd."""
            if self._pid is None:
                return
            try:
                cwd = os.readlink(f"/proc/{self._pid}/cwd")
            except OSError:
                return
            if cwd != self._cwd:
                with contextlib.suppress(OSError):
                    os.chdir(cwd)
                    self._cwd = cwd

        def paste_text(self, text: str) -> None:
            if self._fd is None:
                return
            if (2004 << 5) in self._screen.mode:  # the program asked for bracketed paste
                self._send(b"\x1b[200~" + text.encode() + b"\x1b[201~")
            else:
                self._send(text.replace("\r\n", "\r").replace("\n", "\r").encode())
            self._scroll = 0

        def on_paste(self, event: events.Paste) -> None:
            event.stop()
            self.paste_text(event.text)

        def _key_bytes(self, event: events.Key) -> bytes:
            key = event.key
            if key in ("up", "down", "right", "left", "home", "end") and (1 << 5) in self._screen.mode:
                return b"\x1bO" + {"up": b"A", "down": b"B", "right": b"C", "left": b"D",
                                   "home": b"H", "end": b"F"}[key]  # application cursor mode
            if key in self._KEYS:
                return self._KEYS[key]
            if key.startswith("ctrl+") and len(key) == 6 and key[5].isalpha():
                return bytes([ord(key[5]) & 0x1F])
            if key.startswith("alt+") and len(key) == 5:
                return b"\x1b" + key[4].encode()
            if event.character and event.is_printable:
                return event.character.encode()
            return b""

        async def _on_key(self, event: events.Key) -> None:
            event.stop()
            event.prevent_default()
            if self._fd is None:
                if event.key == "enter" and not _IS_WINDOWS and pyte is not None:
                    self._screen, self._status = None, ""
                    self._spawn(self.size.width, self.size.height)
                    self.refresh()
                return
            if event.key == "ctrl+c" and self.text_selection is not None and self.app.copy_selection():
                return
            if event.key in ("shift+pageup", "shift+pagedown"):
                self._scroll_by(self.size.height - 1 if event.key == "shift+pageup" else 1 - self.size.height)
                return
            data = self._key_bytes(event)
            if data:
                self._scroll = 0
                self._send(data)

        def _scroll_by(self, rows: int) -> None:
            if self._screen is not None:
                self._scroll = max(0, min(len(self._screen.history.top), self._scroll + rows))
                self.refresh()

        def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
            event.stop()
            self._scroll_by(3)

        def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
            event.stop()
            self._scroll_by(-3)

        async def _on_mouse_down(self, event: events.MouseDown) -> None:
            if event.button in (2, 3):
                event.prevent_default()
                self.focus()
                self.app.mouse_paste_or_copy(event.button, target=self)

        def _color(self, name: str) -> str | None:
            if name == "default":
                return None
            if len(name) == 6 and all(c in "0123456789abcdefABCDEF" for c in name):
                return "#" + name
            return self._COLOR_NAMES.get(name, name)

        def _style(self, char, cursor: bool, selected: bool) -> Style:
            key = (char.fg, char.bg, char.bold, char.italics, char.underscore, char.reverse,
                   char.strikethrough, cursor, selected)
            style = self._styles.get(key)
            if style is None:
                flags = {"bold": char.bold or None, "italic": char.italics or None,
                         "underline": char.underscore or None, "strike": char.strikethrough or None,
                         "reverse": (char.reverse != cursor) or None}
                try:
                    style = Style(color=self._color(char.fg), bgcolor=self._color(char.bg), **flags)
                except Exception:  # an exotic color name: keep the attributes only
                    style = Style(**flags)
                if selected:
                    style += _SELECTION
                self._styles[key] = style
            return style

        def selection_updated(self, selection: Selection | None) -> None:
            self.refresh()

        def render_line(self, y: int) -> Strip:
            width, base = self.size.width, self.rich_style
            scr = self._screen
            if scr is None:
                text = Text(self._status if y == 0 else "", "#e5e5e5")
                return Strip(list(text.render(self.app.console))).apply_style(base).crop_extend(0, width, base)
            top = scr.history.top
            index = len(top) - self._scroll + y
            if index < len(top):
                row, screen_y = top[index], None
            else:
                screen_y = index - len(top)
                if screen_y >= scr.lines:
                    return Strip.blank(width, base)
                row = scr.buffer[screen_y]
            if self._status and y == self.size.height - 1:
                text = Text(f" {self._status} ".ljust(width), "reverse")
                return Strip(list(text.render(self.app.console))).crop_extend(0, width, base)
            cursor_x = (scr.cursor.x if screen_y == scr.cursor.y and self.has_focus and not scr.cursor.hidden
                        and self._scroll == 0 else -1)
            selection = self.text_selection
            span = selection.get_span(index) if selection is not None else None
            segments, run, run_style, char_x = [], [], None, 0
            for x in range(scr.columns):
                char = row[x]
                if char.data == "":  # right half of a wide character
                    continue
                selected = span is not None and span[0] <= char_x and (span[1] == -1 or char_x < span[1])
                style = self._style(char, x == cursor_x, selected)
                if style is not run_style and run:
                    segments.append(Segment("".join(run), run_style))
                    run = []
                run_style = style
                run.append(char.data)
                char_x += 1
            if run:
                segments.append(Segment("".join(run), run_style))
            return Strip(segments).apply_style(base).crop_extend(0, width, base).apply_offsets(0, index)

        def get_selection(self, selection: Selection) -> tuple[str, str] | None:
            scr = self._screen
            if scr is None:
                return None
            top = scr.history.top
            total = len(top) + scr.lines
            start = selection.start or Offset(0, 0)
            end = selection.end or Offset(1 << 30, total - 1)
            lines = []
            for index in range(max(0, start.y), min(end.y, total - 1) + 1):
                row = top[index] if index < len(top) else scr.buffer[index - len(top)]
                text = "".join(row[x].data for x in range(scr.columns))
                a = start.x if index == start.y else 0
                b = end.x if index == end.y else len(text)
                lines.append(text[a:b].rstrip())
            return "\n".join(lines), "\n"

        def on_unmount(self) -> None:
            if self._pid is not None:
                with contextlib.suppress(OSError):
                    os.killpg(self._pid, signal.SIGHUP)

    class _ClickBar(Static):
        """A one-row bar of clickable items: subclasses build (text, items)."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._spans: list[tuple[int, int, str]] = []

        def build(self, width: int) -> Text:
            raise NotImplementedError

        def render(self) -> Text:
            return self.build(self.size.width or 80)

        def _add(self, text: Text, label: str, style: str, action: str) -> None:
            start = text.cell_len
            text.append(label, style)
            self._spans.append((start, text.cell_len, action))

        async def on_click(self, event: events.Click) -> None:
            for start, end, action in self._spans:
                if start <= event.x < end:
                    await self.app.run_action(action)
                    break

    class FKeyBar(_ClickBar):
        """Norton Commander style function-key bar."""
        ITEMS = (("1", "Help", "help"), ("2", "Auto", "toggle_auto"), ("3", "Shell", "toggle_pane"),
                 ("4", "Copy", "copy"), ("5", "Paste", "paste"), ("6", "Focus", "cycle_focus"),
                 ("7", "Zoom", "zoom"), ("8", "Clear", "clear"), ("9", "Stop", "stop"), ("10", "Quit", "quit"))

        def build(self, width: int) -> Text:
            self._spans = []
            text = Text(no_wrap=True, overflow="crop")
            slot = max(5, width // len(self.ITEMS))
            for number, label, action in self.ITEMS:
                start = text.cell_len
                text.append(number, "bold #e5e5e5 on #000000")
                text.append(label.ljust(slot - len(number) - 1)[:slot - len(number) - 1], "#000000 on #06989a")
                text.append(" ", "on #000000")
                self._spans.append((start, text.cell_len, action))
            return text

    class PaneTabs(_ClickBar):
        """The Console / Shell tab strip of the right panel."""
        active = "console-log"

        def build(self, width: int) -> Text:
            self._spans = []
            text = Text(no_wrap=True, overflow="crop")
            for pane, label in (("console-log", " Console "), ("shell-widget", " Shell ")):
                style = "bold #000000 on #06989a" if pane == self.active else "#e5e5e5 on #0b2f5e"
                self._add(text, label, style, f"show_pane('{pane}')")
                text.append(" ")
            text.append("F3 switches", "#6f8fc0")
            return text

    class ApprovalScreen(ModalScreen):
        """Run this command? Yes / No / Always, or reject with a suggestion.
        Keys are ignored for a moment after it opens, so typing that was meant
        for the prompt cannot approve a command by accident."""

        CSS = """
        ApprovalScreen { align: center middle; background: #000000 50%; }
        #approval { width: 96; max-width: 96%; height: auto; max-height: 90%; border: double #ffd75f;
                    background: #0b2f5e; padding: 0 1; }
        #approval-cmd { height: auto; max-height: 16; margin: 1 0; background: #000000; }
        #approval-buttons { height: auto; align: center middle; }
        #approval-buttons Button { margin: 0 1; min-width: 16; }
        #approval-suggest { margin: 1 0 0 0; }
        """
        BINDINGS = [Binding("y", "decide('yes')", show=False), Binding("n", "decide('no')", show=False),
                    Binding("a", "decide('always')", show=False),
                    Binding("escape,ctrl+c", "decide('no')", show=False)]

        def __init__(self, command: str):
            super().__init__()
            self.command = command
            self._opened = 0.0

        def compose(self) -> ComposeResult:
            with Vertical(id="approval"):
                yield Static(Text("The agent wants to run this command:", style="bold #ffd75f"))
                with VerticalScroll(id="approval-cmd"):
                    yield Static(Syntax(self.command, "bash", theme="ansi_dark", word_wrap=True,
                                        background_color="#000000"))
                with Horizontal(id="approval-buttons"):
                    yield Button("Yes (y)", id="yes", variant="success")
                    yield Button("No (n)", id="no", variant="error")
                    yield Button("Always (a)", id="always", variant="warning")
                yield Input(placeholder="…or type a suggestion for the agent and press Enter", id="approval-suggest")

        def on_mount(self) -> None:
            self._opened = time.monotonic()
            self.query_one("#approval").border_title = "Approve command"
            self.query_one("#yes").focus()

        def action_decide(self, verdict: str) -> None:
            if verdict != "no" and time.monotonic() - self._opened < 0.4:
                return
            self.dismiss((verdict, ""))

        def on_button_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            self.action_decide(event.button.id)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            event.stop()
            if event.value.strip():
                self.dismiss(("suggest", event.value.strip()))

    class ReverseSearchScreen(ModalScreen):
        """Incremental reverse search over the prompt history (Ctrl+R)."""

        CSS = """
        ReverseSearchScreen { align: center top; background: transparent; }
        #search { width: 96; max-width: 96%; height: auto; background: #0b2f5e; border: round #06989a;
                  padding: 0 1; margin-top: 3; }
        #search-status { color: #e5e5e5; margin-top: 1; }
        """
        BINDINGS = [Binding("ctrl+r", "next_match", show=False), Binding("escape,ctrl+g", "cancel", show=False)]

        def __init__(self, items: list[str]):
            super().__init__()
            self._items = items
            self._matches: list[str] = []
            self._pos = 0

        def compose(self) -> ComposeResult:
            with Vertical(id="search"):
                yield Input(placeholder="Search prompt history", id="search-input")
                yield Static("", id="search-status", markup=False)

        def on_mount(self) -> None:
            self.query_one("#search").border_title = "reverse-i-search"
            self.query_one("#search-input").focus()
            self._update()

        def on_input_changed(self, event: Input.Changed) -> None:
            query = event.value.lower()
            seen, self._matches = set(), []
            for item in reversed(self._items):  # newest first, without duplicates
                if query and query in item.lower() and item not in seen:
                    seen.add(item)
                    self._matches.append(item)
            self._pos = 0
            self._update()

        def _update(self) -> None:
            status = self.query_one("#search-status", Static)
            if self._matches:
                status.update(f"{self._matches[self._pos]}\n\n[{self._pos + 1}/{len(self._matches)}] "
                              "Ctrl+R older · Enter accept · Esc cancel")
            else:
                status.update("Type to search. Esc cancels.")

        def action_next_match(self) -> None:
            if self._matches:
                self._pos = (self._pos + 1) % len(self._matches)
                self._update()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            event.stop()
            self.dismiss(self._matches[self._pos] if self._matches else None)

        def action_cancel(self) -> None:
            self.dismiss(None)

    _HELP = (
        ("Prompt", "Enter sends · Ctrl+J or Shift+Enter adds a line · ↑/↓ browse history at the first/last line · "
                   "Ctrl+R searches history · Tab completes /commands · typing while the agent works steers it"),
        ("Mouse", "drag selects text (copied automatically) · double-click selects a word, triple-click a line · "
                  "right-click copies the selection, or pastes · middle-click pastes · the wheel scrolls"),
        ("Clipboard", "Ctrl+C copies the selection · Ctrl+V pastes in the prompt · F4 copies the selection or the "
                      "last answer · F5 pastes · copies reach the desktop clipboard directly on X11/Xwayland, "
                      "through OSC 52 in terminals that support it, and wl-copy or pbcopy when present"),
        ("Agent", "F2 toggles auto-approve · F9 or Ctrl+C stops the agent · F8 twice clears the conversation"),
        ("Panes", "F3 switches Console ⇄ Shell · F6 or Tab moves to the next pane · F7 zooms the focused pane "
                  "(Esc restores) · in the Shell every key goes to the shell: leave it with F6 or the mouse"),
        ("Quit", "F10 · Ctrl+Q · Ctrl+C twice while idle"),
        ("Commands", "/help  /aa (/autoapprove)  /clear (/new)  /session  /reload  /copy  /stop  /quit (/exit)  "
                     "·  /approve  /reject  /suggest <text> answer a pending approval"),
    )

    class HelpScreen(ModalScreen):
        CSS = """
        HelpScreen { align: center middle; background: #000000 50%; }
        #help { width: 100; max-width: 96%; height: auto; max-height: 90%; border: double #5fd7ff;
                background: #0b2f5e; padding: 1 2; }
        """
        BINDINGS = [Binding("escape,q,enter", "close", show=False)]

        def compose(self) -> ComposeResult:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold #87d7ff", no_wrap=True)
            table.add_column(style="#e5e5e5")
            for section, text in _HELP:
                table.add_row(section, text)
            with VerticalScroll(id="help"):
                yield Static(Text("AI-Commander — keys, mouse and commands\n", style="bold #ffff5f"))
                yield Static(table)
                yield Static(Text("\nEsc closes this help.", style=_C_DIM))

        def action_close(self) -> None:
            self.dismiss()

    class CommanderApp(App):
        """Norton-Commander style TUI: agent output (left), console/shell tabs
        (right), prompt, status line and function-key bar. The agent runs in a
        thread and only enqueues events; the UI thread drains and renders them."""

        CSS = """
        Screen { background: #12488b; }
        #panels { height: 1fr; }
        * { scrollbar-size-vertical: 1; scrollbar-background: #0b2f5e; scrollbar-background-hover: #0b2f5e;
            scrollbar-background-active: #0b2f5e; scrollbar-color: #4f74b3; scrollbar-color-hover: #8fb3e0;
            scrollbar-color-active: #87d7ff; }
        OutputLog, ShellWidget { border: solid #6f8fc0; }
        OutputLog:focus, ShellWidget:focus { border: double #87d7ff; }
        OutputLog { background: #12488b; color: #e5e5e5; }
        #agent-log { width: 1fr; border-title-color: #ffff5f; border-title-style: bold; }
        #right-panel { width: 1fr; }
        #tabs { height: 1; background: #0b2f5e; }
        #switcher { height: 1fr; }
        ShellWidget { background: #000000; color: #e5e5e5; }
        PromptArea { height: auto; min-height: 3; max-height: 12; background: #000000; color: #f2f2f2;
                     border: round #06989a; padding: 0 1; border-title-color: #5fd7ff;
                     border-subtitle-color: #6f8fc0; }
        PromptArea:focus { border: round #5fd7ff; }
        PromptArea .text-area--cursor-line { background: #000000; }
        #status-bar { height: 1; background: #06989a; color: #000000; }
        #fkeys { height: 1; background: #000000; }
        """
        BINDINGS = [
            Binding("f1", "help", "Help", priority=True),
            Binding("f2", "toggle_auto", "Auto-approve", priority=True),
            Binding("f3", "toggle_pane", "Console/Shell", priority=True),
            Binding("f4", "copy", "Copy", priority=True),
            Binding("f5", "paste", "Paste", priority=True),
            Binding("f6", "cycle_focus", "Next pane", priority=True),
            Binding("f7", "zoom", "Zoom", priority=True),
            Binding("f8", "clear", "Clear", priority=True),
            Binding("f9", "stop", "Stop", priority=True),
            Binding("f10,ctrl+q", "quit", "Quit", priority=True),
            # Not a priority binding: the focused widget and the screen copy a
            # selection first; only an unhandled Ctrl+C stops or quits.
            Binding("ctrl+c", "interrupt", "Stop / quit", show=False),
        ]
        ENABLE_COMMAND_PALETTE = False  # it would take Ctrl+P (history) from the prompt

        def __init__(self, args, history: PromptHistory | None = None):
            super().__init__()
            self.args = args
            self.history = history or PromptHistory()
            self.event_queue: queue.Queue = queue.Queue()
            self.sink = TUISink(self.event_queue, threading.Event())
            self.auto_approve = self.sink.auto_approve = args.auto_approve
            self.agent: AICommander | None = None
            self.agent_thread: threading.Thread | None = None
            self._idle_callbacks: list = []
            self.phase, self._phase_since = "idle", time.monotonic()
            self.step, self.max_steps = 0, args.max_steps
            self.total_tokens, self.tok_rate = 0, 0.0
            self.retry_text = self.current_command = ""
            self._run_started = self._cmd_started = time.monotonic()
            self._approval_screen: ApprovalScreen | None = None
            self._agent_stream: str | None = None  # "think" / "text" while the agent pane streams
            self._console_stream = False
            self._marker_hold = ""
            self._stream_chunks, self._stream_t0 = 0, 0.0
            self._last_ctrl_c = self._last_clear = 0.0
            self._last_status = None
            self._ctx_cache = (None, 0)
            self._last_autocopy = None
            self._shutting_down = False

        # Layout -----------------------------------------------------------------

        def compose(self) -> ComposeResult:
            with Horizontal(id="panels"):
                yield OutputLog(id="agent-log")
                with Vertical(id="right-panel"):
                    yield PaneTabs(id="tabs")
                    with ContentSwitcher(initial="console-log", id="switcher"):
                        yield OutputLog(id="console-log", max_lines=5000)
                        yield ShellWidget(id="shell-widget")
            yield PromptArea(self.history, id="prompt-input", placeholder="Type a task or /command…")
            yield Static(id="status-bar")
            yield FKeyBar(id="fkeys")

        @property
        def agent_log(self) -> OutputLog:
            return self.query_one("#agent-log", OutputLog)

        @property
        def console_log(self) -> OutputLog:
            return self.query_one("#console-log", OutputLog)

        @property
        def prompt(self) -> PromptArea:
            return self.query_one("#prompt-input", PromptArea)

        def on_mount(self) -> None:
            self.title = "AI-Commander"
            self.agent_log.border_title = "Agent"
            self.prompt.border_title = "Prompt"
            self._init_agent()
            self.prompt.focus()
            self.set_interval(1 / 20, self._tick)
            request = " ".join(self.args.request or [])
            self.args.request = []
            if request.strip():
                self.call_after_refresh(self._submit, request)

        def _init_agent(self) -> None:
            a = self.args
            self.agent = AICommander(
                api_base=a.api_base, model=a.model, api_key=a.api_key, auto_approve=self.auto_approve,
                show_thinking=not a.no_thinking, command_timeout=a.timeout, max_prompt_len=a.max_prompt_len,
                max_output_bytes=a.max_output_bytes, debug=a.debug, sink=self.sink, persist_history=True,
                max_steps=a.max_steps, compress_algorithm="truncate" if a.fast else a.compress_alg,
                compress_target=a.compress_target, fast=a.fast, reasoning_effort=a.reasoning_effort,
                sandbox_enabled=a.sandbox_enabled, sandbox_write_dir=a.sandbox_write_dir,
                session_file=a.session_file, instance_number=a.instance_number)
            self.sink.stop_event = self.agent.stop_event
            atexit.register(self.agent.save_session, "process_exit")
            log = self.agent_log
            log.write_line(f"AI-Commander {__version__} · model {a.model} · {a.api_base}", _C_DIM)
            log.write_line(a.sandbox_msg, _C_OK if a.sandbox_enabled else _C_FAIL)
            log.write_line(a.session_msg, _C_OK if a.instance_number == 1 and a.session_file else "#ffd75f")
            try:
                restored = self.agent.load_session(notify=False)
            except Exception as exc:
                restored = False
                log.write_line(f"[SESSION] Could not restore the session: {exc}", _C_FAIL)
            if restored:
                self.step = self.agent.step_count
                self.total_tokens = self.agent.usage_totals.get("billable_tokens", 0)
                self._render_history()
            self._set_title(self.agent.task_summary)
            log.write_line("Commands run automatically (auto-approve is ON; F2 toggles)." if self.auto_approve
                           else "Every command asks for approval (F2 or /aa turns on auto-approve).",
                           "#ffd75f" if self.auto_approve else _C_DIM)
            log.write_line("Type a task below and press Enter · F1 shows keys, mouse and commands.", _C_DIM)

        def _render_history(self) -> None:
            """Replay a restored conversation into both panes."""
            log, console = self.agent_log, self.console_log
            commands: dict = {}
            for msg in self.agent.conversation_history:
                role, text = msg.get("role"), _content_text(msg.get("content")).strip()
                if role == "user" and text and not _is_continuation(msg) and text != _CONTINUE_PROMPT:
                    log.write_rule("You", _C_DIM)
                    log.write_line(text, _C_USER)
                elif role == "assistant":
                    text = text.replace(AICommander.COMPLETION_MARKER, "").rstrip()
                    if text:
                        log.write_line(text, _C_ANSWER)
                    for tc in msg.get("tool_calls") or []:
                        command = _tool_command(tc)
                        if command:
                            commands[tc.get("id")] = command
                            log.write_line(f"$ {command}", _C_CMD)
                elif role == "tool":
                    command = commands.pop(msg.get("tool_call_id"), None)
                    console.write_rule(f"$ {command}" if command else "tool result", _C_CMD)
                    match = re.match(r"Command executed with exit code (-?\d+)\.\nOutput:\n?(.*)\Z", text, re.S)
                    output = match.group(2) if match else text
                    console.write_line(output[:4000] + ("\n… (truncated)" if len(output) > 4000 else ""), "#d0d0d0")
                    if match:
                        ok = match.group(1) == "0"
                        console.write_line(f"{'✓' if ok else '✗'} exit {match.group(1)}", f"bold {_C_OK if ok else _C_FAIL}")
            log.write_rule("restored session", _C_DIM)

        def _set_title(self, summary: str) -> None:
            self.agent_log.border_title = f"Agent · {summary}" if summary else "Agent"

        # Agent lifecycle --------------------------------------------------------

        def _agent_busy(self) -> bool:
            return self.agent_thread is not None and self.agent_thread.is_alive()

        def _when_idle(self, callback) -> None:
            if self._agent_busy():
                self._idle_callbacks.append(callback)
            else:
                callback()

        def _start_agent(self, prompt: str) -> None:
            agent = self.agent
            agent.stop_event.clear()
            agent.auto_approve = self.sink.auto_approve = self.auto_approve
            agent.persist_history = True
            self.step = 0
            self._run_started = time.monotonic()
            self._set_phase("thinking")
            self.agent_thread = threading.Thread(target=self._agent_main, args=(prompt,), daemon=True,
                                                 name="aic-agent")
            self.agent_thread.start()

        def _agent_main(self, prompt: str) -> None:
            try:
                self.agent.run(prompt)
            except BaseException as exc:  # never let the thread die silently
                self.sink.emit("ERROR", {"text": f"[AGENT CRASHED] {exc!r}"})
                if self.args.debug:
                    self.sink.emit("ERROR", {"text": traceback.format_exc()})
                self.sink.emit("SHUTDOWN", {"reason": "crashed", "steps": self.step})

        def _stop_agent(self) -> None:
            if self._agent_busy():
                self.agent.request_stop()
                self._set_phase("stopping")

        def _on_agent_finished(self) -> None:
            self.agent_thread = None
            self._end_streams()
            if self._approval_screen is not None:
                with contextlib.suppress(Exception):
                    self._approval_screen.dismiss(None)
            self._set_phase("idle")
            callbacks, self._idle_callbacks = self._idle_callbacks, []
            for callback in callbacks:
                callback()

        def _tick(self) -> None:
            self._drain_events()
            if self.agent_thread is not None and not self.agent_thread.is_alive():
                self._drain_events()
                self._on_agent_finished()
            self._refresh_status()

        # Events from the agent ---------------------------------------------------

        def _drain_events(self) -> None:
            batch = []
            with contextlib.suppress(queue.Empty):
                for _ in range(5000):
                    batch.append(self.event_queue.get_nowait())
            merged: list[dict] = []
            for event in batch:  # coalesce consecutive stream chunks
                kind, payload = event["kind"], event["payload"]
                if kind in ("LLM_STREAM", "THINKING_STREAM"):  # ~1 token per chunk: live tok/s
                    if not self._stream_chunks:
                        self._stream_t0 = time.monotonic()
                    self._stream_chunks += 1
                if (merged and merged[-1]["kind"] == kind and set(payload) <= {"text", "command"}
                        and kind in ("LLM_STREAM", "THINKING_STREAM", "CMD_OUTPUT", "CONSOLE_STREAM")):
                    merged[-1] = {"kind": kind, "payload": {**merged[-1]["payload"],
                                                            "text": merged[-1]["payload"]["text"] + payload["text"]}}
                else:
                    merged.append(event)
            for event in merged:
                try:
                    self._dispatch(event["kind"], event["payload"])
                except Exception as exc:  # a rendering bug must not kill the UI
                    self.agent_log.write_line(f"[UI ERROR] {exc!r}", _C_FAIL)

        def _dispatch(self, kind: str, p: dict) -> None:
            text = p.get("text", "")
            if kind in ("LLM_STREAM", "THINKING_STREAM"):
                self._agent_write(text, "think" if kind == "THINKING_STREAM" else "text")
            elif kind == "CMD_OUTPUT":
                self.console_log.write(text, "#d0d0d0")
            elif kind == "CONSOLE_STREAM":
                if p.get("label") and not self._console_stream:
                    self.console_log.write_rule(p["label"].strip("[]").title(), "#ffd75f")
                self._console_stream = True
                self.console_log.write(text, "#ffd75f")
            elif kind == "STATUS_UPDATE":
                self.step, self.max_steps = p.get("step", self.step), p.get("max_steps", self.max_steps)
                phase = p.get("phase")
                self.retry_text = p.get("retry", "") if phase == "api_retry" else ""
                self._set_phase({"llm_call": "thinking", "api_retry": "retry"}.get(phase, phase or "thinking"))
            elif kind == "TOKEN_USAGE":
                if not p.get("estimate"):
                    self.total_tokens += (max(0, (p.get("input_tokens") or 0) - (p.get("cached_tokens") or 0))
                                          + (p.get("output_tokens") or 0))
                    if p.get("tokens_per_sec"):
                        self.tok_rate = float(p["tokens_per_sec"])
                else:
                    self._stream_chunks = 0
            elif kind == "TASK_TITLE":
                self._set_title(p.get("summary", ""))
            elif kind == "SYSTEM":
                pass  # the startup lines already show this
            else:
                self._end_streams()
                self._dispatch_line(kind, p, text)

        def _dispatch_line(self, kind: str, p: dict, text: str) -> None:
            log, console = self.agent_log, self.console_log
            if kind == "LOG":
                log.write_line(text, _LOG_STYLES.get(p.get("style", ""), "#c8d0e0"))
            elif kind == "ERROR":
                log.write_line(text, "bold #ff5f5f")
            elif kind == "USER_REQUEST":
                log.write_rule(f"You · {datetime.now().strftime('%H:%M:%S')}", _C_DIM)
                log.write_line(text, _C_USER)
            elif kind == "CMD_EXEC":
                self.current_command, self._cmd_started = p.get("command", ""), time.monotonic()
                log.write_line(f"$ {self.current_command}", _C_CMD)
                console.write_rule(f"$ {self.current_command}", _C_CMD)
                self._set_phase("running")
            elif kind == "APPROVAL_REQUEST":
                self._set_phase("approval")
                self._ask_approval(p.get("command", ""))
            elif kind == "CMD_COMPLETE":
                code, output = p.get("exit_code"), p.get("output") or ""
                ok = code == 0
                lines = output.count("\n") + bool(output and not output.endswith("\n"))
                summary = (f"{'✓' if ok else '✗'} exit {code} · {_fmt_secs(p.get('duration') or 0.0)}"
                           f" · {lines} line{'s' * (lines != 1)}")
                console.write_line(summary, f"bold {_C_OK if ok else _C_FAIL}")
                log.write_line(f"  {summary}", _C_OK if ok else _C_FAIL)
                self._set_phase("thinking")
            elif kind == "SHUTDOWN":
                log.write_rule(f"{'✓ task complete' if p.get('completed') else 'ready'} · "
                               f"{p.get('steps', self.step)} step(s) · "
                               f"{_fmt_secs(time.monotonic() - self._run_started)}", _C_OK)

        def _agent_write(self, text: str, kind: str) -> None:
            log = self.agent_log
            if kind != self._agent_stream:
                self._flush_marker()
                log.close_line()
                if kind == "think":
                    log.write_line("✻ thinking", _C_DIM)
                self._agent_stream = kind
                self._set_phase("thinking" if kind == "think" else "writing")
            if kind == "text":
                text = self._filter_marker(text)
            log.write(text, _C_THINK if kind == "think" else _C_ANSWER)

        def _filter_marker(self, text: str) -> str:
            """Hide the completion marker, even when it arrives split across chunks."""
            marker = AICommander.COMPLETION_MARKER
            buf = (self._marker_hold + text).replace(marker, "")
            hold = next((k for k in range(min(len(marker) - 1, len(buf)), 0, -1) if marker.startswith(buf[-k:])), 0)
            self._marker_hold = buf[len(buf) - hold:] if hold else ""
            return buf[:len(buf) - hold] if hold else buf

        def _flush_marker(self) -> None:
            if self._marker_hold:
                self.agent_log.write(self._marker_hold, _C_ANSWER)
                self._marker_hold = ""

        def _end_streams(self) -> None:
            if self._agent_stream:
                self._flush_marker()
                self.agent_log.close_line()
                self._agent_stream = None
            if self._console_stream:
                self.console_log.close_line()
                self._console_stream = False

        def _ask_approval(self, command: str) -> None:
            def done(result) -> None:
                self._approval_screen = None
                if result is None:  # dismissed because the run ended
                    return
                verdict, suggestion = result
                if verdict == "always":
                    self._set_auto_approve(True)
                approved = verdict in ("yes", "always")
                self.sink.resolve_approval(approved, suggestion)
                self.agent_log.write_line(
                    "  ✓ approved" if approved else f"  ↳ rejected with suggestion: {suggestion}" if suggestion
                    else "  ✗ rejected", _C_OK if approved else "#ffd75f")
                self._set_phase("running" if approved else "thinking")
            self._approval_screen = ApprovalScreen(command)
            self.push_screen(self._approval_screen, done)

        # Status line --------------------------------------------------------------

        def _set_phase(self, phase: str) -> None:
            if phase != self.phase:
                self.phase, self._phase_since = phase, time.monotonic()

        def _context_tokens(self) -> int:
            hist = self.agent.conversation_history if self.agent else []
            key = (id(hist), len(hist), id(hist[-1]) if hist else 0, int(time.monotonic()))
            if self._ctx_cache[0] != key:
                self._ctx_cache = (key, self.agent._estimate_input_tokens(hist) if self.agent else 0)
            return self._ctx_cache[1]

        def _refresh_status(self) -> None:
            now = time.monotonic()
            busy = self._agent_busy()
            elapsed = _fmt_secs(now - self._phase_since)
            spin = _SPINNER[int(now * 10) % len(_SPINNER)]
            if not busy:
                phase = ("●", "ready")
            elif self.phase == "approval":
                phase = ("◆", "waiting for your approval")
            elif self.phase == "retry":
                phase = ("↻", self.retry_text or "retrying")
            elif self.phase == "running":
                command = " ".join(self.current_command.split())
                phase = (spin, f"running {elapsed}: {command[:40] + '…' if len(command) > 40 else command}")
            else:
                phase = (spin, f"{self.phase} {elapsed}")
            if busy and self.phase == "writing" and self._stream_chunks > 4:
                rate = self._stream_chunks / max(0.1, now - self._stream_t0)
            else:
                rate = self.tok_rate
            ctx, budget = self._context_tokens(), max(1, self.agent._prompt_budget() if self.agent else 1)
            parts = [
                (f"{phase[0]} {phase[1]}", "bold #000000 on #5fd7ff" if busy else "bold #000000 on #5fd787"),
                (f"step {self.step}/{self.max_steps}", ""),
                (f"ctx {_fmt_tokens(ctx)}/{_fmt_tokens(budget)} {min(999, ctx * 100 // budget)}%",
                 "bold #800000" if ctx > budget * 0.9 else ""),
                (f"Σ {_fmt_tokens(self.total_tokens)} tok", ""),
                (f"{rate:.0f} tok/s", ""),
                ("AUTO-APPROVE" if self.auto_approve else "approve: ask",
                 "bold #ffffff on #af0000" if self.auto_approve else ""),
                ("sandbox ✓" if self.args.sandbox_enabled else "sandbox ✗", ""),
                (f"{self.args.model} · {_safe_cwd().replace(os.path.expanduser('~'), '~', 1)} · "
                 f"{os.path.basename(self.args.session_file) if self.args.session_file else 'no session'}", ""),
            ]
            status = Text(no_wrap=True, overflow="ellipsis")
            for i, (chunk, style) in enumerate(parts):
                if i > 1:
                    status.append("│", "#0b2f5e")
                status.append(f" {chunk} ", style)
            if status.plain != self._last_status:
                self._last_status = status.plain
                self.query_one("#status-bar", Static).update(status)

        # Prompt and slash commands -------------------------------------------------

        def on_prompt_area_submitted(self, message: PromptArea.Submitted) -> None:
            self._submit(message.text)

        def _submit(self, text: str) -> None:
            text = text.strip()
            if not text:
                return
            if text.startswith("/"):
                self._slash_command(text)
                return
            self.history.add(text)
            if self._agent_busy():
                self.agent.inject_suggestion(text)
                self.agent_log.write_line(f"↳ steering message queued for the agent: {text}", _C_USER)
                return
            self._start_agent(text)

        def _slash_command(self, text: str) -> None:
            command, _, arg = text.partition(" ")
            command, arg = command.lower(), arg.strip()
            actions = {
                "/help": self.action_help, "/autoapprove": self.action_toggle_auto, "/aa": self.action_toggle_auto,
                "/clear": self._clear_conversation, "/new": self._clear_conversation,
                "/session": self._report_session, "/reload": self._reload_session,
                "/copy": lambda: self._copy_answer(), "/stop": self.action_stop,
                "/quit": self.action_quit, "/exit": self.action_quit,
                "/approve": lambda: self._answer_approval("yes"), "/reject": lambda: self._answer_approval("no"),
                "/suggest": lambda: self._answer_approval("suggest", arg),
            }
            if command in actions:
                actions[command]()
            else:
                self.agent_log.write_line(f"Unknown command {command}. Available: "
                                          + " ".join(c.strip() for c in _SLASH_COMMANDS), _C_FAIL)

        def _answer_approval(self, verdict: str, suggestion: str = "") -> None:
            if self._approval_screen is None:
                self.agent_log.write_line("There is no pending approval.", _C_FAIL)
            else:
                self._approval_screen.dismiss((verdict, suggestion))

        def _clear_conversation(self) -> None:
            def clear():
                self.agent.suggestion_queue = queue.Queue()
                self.agent.discard_session("cleared")
                self.agent_log.clear()
                self.console_log.clear()
                self.step, self.total_tokens, self.tok_rate = 0, 0, 0.0
                self._set_title("")
                self.agent_log.write_line("New conversation started. Ready for your next prompt.", _C_OK)
            self._stop_agent()
            self._when_idle(clear)

        def _reload_session(self) -> None:
            if not self.agent.session_file:
                self.agent_log.write_line("Session persistence is disabled.", _C_FAIL)
                return
            if self._agent_busy():
                self.agent_log.write_line("Stop the agent (F9) before reloading the session.", _C_FAIL)
                return
            if self.agent.load_session(notify=False):
                self.agent_log.clear()
                self.console_log.clear()
                self.step = self.agent.step_count
                self.total_tokens = self.agent.usage_totals.get("billable_tokens", 0)
                self._render_history()
                self._set_title(self.agent.task_summary)
                self.agent_log.write_line("Session reloaded from disk.", _C_OK)
            else:
                self.agent_log.write_line(f"Nothing reloaded from {self.agent.session_file}.", "#ffd75f")

        def _report_session(self) -> None:
            log, path = self.agent_log, self.args.session_file
            if not path:
                log.write_line("[SESSION] Persistence is disabled.", "#ffd75f")
                return
            data, err = read_session_file(path)
            log.write_line(f"[SESSION] {path} (instance {self.args.instance_number}) · id {self.agent.session_id} "
                           f"· {len(self.agent.conversation_history)} message(s) in memory", "#87d7ff")
            if err:
                log.write_line(f"  The saved file is unreadable: {err}", _C_FAIL)
            elif data:
                saved = data.get("messages") if isinstance(data.get("messages"), list) else []
                log.write_line(f"  Last saved {data.get('updated_at', '?')} (reason: {data.get('reason', '?')}, "
                               f"step {data.get('step_count', '?')}, {len(saved)} message(s))", "#87d7ff")
            log.write_line('  Edit its "messages" array and run /reload to reshape the session.', _C_DIM)

        # Clipboard -------------------------------------------------------------------

        def copy_to_clipboard(self, text: str) -> None:
            """OSC 52 + in-app clipboard (Textual), plus the native clipboard tool."""
            super().copy_to_clipboard(text)
            if text:
                threading.Thread(target=clipboard_write_native, args=(text,), daemon=True).start()
                self.notify(f"Copied {len(text)} character{'s' * (len(text) != 1)}.", timeout=2)

        def copy_selection(self) -> bool:
            """Copy the mouse selection (if any); remembered so it is copied once."""
            text = self.screen.get_selected_text()
            if not text:
                return False
            self._last_autocopy = dict(self.screen.selections)
            self.copy_to_clipboard(text)
            return True

        def on_text_selected(self, event: events.TextSelected) -> None:
            """Copy on select, like a terminal."""
            if self.screen.selections and dict(self.screen.selections) != self._last_autocopy:
                self.copy_selection()

        def _last_answer(self) -> str:
            for msg in reversed(self.agent.conversation_history if self.agent else []):
                text = _content_text(msg.get("content")).replace(AICommander.COMPLETION_MARKER, "").strip()
                if msg.get("role") == "assistant" and text:
                    return text
            return ""

        def _copy_answer(self) -> None:
            text = self._last_answer()
            if text:
                self.copy_to_clipboard(text)
            else:
                self.notify("There is no answer to copy yet.", timeout=2)

        def mouse_paste_or_copy(self, button: int, target) -> None:
            """Right-click copies the selection of the clicked pane when it has one;
            otherwise (and on middle-click) the clipboard is pasted into the
            clicked input, or into the prompt when an output pane was clicked."""
            if button == 3:
                if isinstance(target, PromptArea):
                    if target.selected_text:
                        self.copy_to_clipboard(target.selected_text)
                        return
                elif target is None or target.text_selection is not None:
                    if self.copy_selection():
                        return
            self._paste(target if isinstance(target, (PromptArea, ShellWidget)) else self.prompt)

        def _paste(self, target) -> None:
            def work():
                text = clipboard_read_native()
                self.call_from_thread(deliver, text)

            def deliver(text):
                text = text or self.clipboard
                if not text:
                    self.notify("The clipboard is empty.", timeout=2)
                    return
                target.paste_text(text)
                target.focus()
            if _clipboard_commands(read=True) or os.environ.get("DISPLAY"):
                threading.Thread(target=work, daemon=True).start()
            else:
                deliver(None)

        # Actions (F-keys) --------------------------------------------------------------

        def action_help(self) -> None:
            if isinstance(self.screen, HelpScreen):
                self.screen.dismiss()
            else:
                self.push_screen(HelpScreen())

        def _set_auto_approve(self, value: bool) -> None:
            self.auto_approve = self.sink.auto_approve = value
            if self.agent:
                self.agent.auto_approve = value
            self.agent_log.write_line(
                "Auto-approve is ON: commands run without asking. F9 stops the agent." if value
                else "Auto-approve is OFF: every command asks for approval.", "#ffd75f" if value else _C_OK)

        def action_toggle_auto(self) -> None:
            self._set_auto_approve(not self.auto_approve)

        def show_pane(self, pane: str, focus: bool = True) -> None:
            self.query_one("#switcher", ContentSwitcher).current = pane
            tabs = self.query_one("#tabs", PaneTabs)
            tabs.active = pane
            tabs.refresh()
            if focus and pane == "shell-widget":
                self.call_after_refresh(self.query_one("#shell-widget").focus)
            elif isinstance(self.focused, ShellWidget):
                self.prompt.focus()

        def action_show_pane(self, pane: str) -> None:
            self.show_pane(pane)

        def action_toggle_pane(self) -> None:
            current = self.query_one("#switcher", ContentSwitcher).current
            self.show_pane("console-log" if current == "shell-widget" else "shell-widget")

        def action_copy(self) -> None:
            text = self.screen.get_selected_text()
            if not text and isinstance(self.focused, TextArea):
                text = self.focused.selected_text
            if text:
                self.copy_to_clipboard(text)
            else:
                self._copy_answer()

        def action_paste(self) -> None:
            target = self.focused if isinstance(self.focused, (PromptArea, ShellWidget)) else self.prompt
            self._paste(target)

        def action_cycle_focus(self) -> None:
            right = self.query_one("#switcher", ContentSwitcher).current
            order = [self.prompt, self.agent_log, self.query_one(f"#{right}")]
            current = next((i for i, w in enumerate(order) if w is self.focused), -1)
            order[(current + 1) % len(order)].focus()

        def action_zoom(self) -> None:
            if self.screen.maximized is not None:
                self.screen.minimize()
                return
            widget = self.focused if isinstance(self.focused, (OutputLog, ShellWidget)) else self.agent_log
            self.screen.maximize(widget, container=False)

        def action_clear(self) -> None:
            now = time.monotonic()
            if now - self._last_clear < 2:
                self._last_clear = 0.0
                self._clear_conversation()
            else:
                self._last_clear = now
                self.notify("Press F8 again to clear the conversation.", timeout=2)

        def action_stop(self) -> None:
            if self._agent_busy():
                self.agent_log.write_line("Stopping the agent…", "#ffd75f")
                self._stop_agent()
            else:
                self.notify("The agent is not running.", timeout=2)

        def action_interrupt(self) -> None:
            if self._agent_busy():
                self.action_stop()
                return
            now = time.monotonic()
            if now - self._last_ctrl_c < 2:
                self.action_quit()
            else:
                self._last_ctrl_c = now
                self.notify("Press Ctrl+C again to quit (or F10).", timeout=2)

        def action_quit(self) -> None:
            """Stop the agent, save the session and history, exit."""
            if self._shutting_down:
                return
            self._shutting_down = True
            if self.agent:
                self.agent.request_stop()
                if self.agent_thread is not None:
                    self.agent_thread.join(timeout=3)
                self.agent.save_session("tui_shutdown")
            self.history.close()
            self.exit()


# --- Startup ----------------------------------------------------------------------

def enable_sandbox() -> tuple[bool, str]:
    """Apply a Landlock sandbox (Linux, py-landlock): writes only to the cwd,
    /tmp and /dev (PTYs need /dev and /dev/pts); reads, execution and network
    stay allowed. Best effort on older kernels. Returns (enabled, message)."""
    if sys.platform != "linux":
        return False, "[SANDBOX] Not running on Linux; no OS-level sandbox applied"
    try:
        from py_landlock import AccessFs, Landlock
    except ImportError:
        return False, "[SANDBOX] py-landlock not installed; running unsandboxed"
    try:
        ll = (Landlock(strict=False).allow_read("/").allow_execute("/")
              .allow_write(os.getcwd(), "/tmp", "/dev", "/dev/pts").allow_all_network().allow_all_scope())
        ll.add_path_rule("/dev", "/dev/pts", "/dev/tty", access=AccessFs.IOCTL_DEV)
        ll.apply()
        return True, "[SANDBOX] OS-level Landlock sandbox enabled (writes restricted to current directory, /tmp and /dev)"
    except Exception as exc:
        return False, f"[SANDBOX] Failed to enable Landlock sandbox: {exc}"


def _enable_windows_console_vt() -> None:
    """Turn on ANSI escape processing in a classic Windows console (best effort)."""
    if os.name != "nt":
        return
    import ctypes
    with contextlib.suppress(Exception):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def main():
    _enable_windows_console_vt()
    _check_openai_version()
    parser = argparse.ArgumentParser(description="AI-Commander - a ralph-loop AI agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--api-base", required=True, help="API base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve command execution")
    parser.add_argument("--no-thinking", action="store_true", help="Hide thinking tokens from output")
    parser.add_argument("--timeout", type=int, default=120, help="Command timeout in seconds (default: 120)")
    parser.add_argument("--max-prompt-len", type=int, default=DEFAULT_MAX_PROMPT_LEN,
                        help=f"Maximum prompt length in tokens (default: {DEFAULT_MAX_PROMPT_LEN})")
    parser.add_argument("--max-output-bytes", type=int, default=10240,
                        help="Maximum output bytes returned from a command (default: 10240); the command "
                             "is killed after printing 5x as much")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum agent loop steps per request (default: 500)")
    parser.add_argument("--debug", action="store_true", help="Dump the conversation history at every step")
    parser.add_argument("--nogui", action="store_true",
                        help="Plain CLI mode without the TUI (no session file unless --session is given)")
    parser.add_argument("--no-mouse", action="store_true",
                        help="TUI without mouse support, leaving selection and paste to the terminal")
    parser.add_argument("--disable-sandbox", action="store_true",
                        help="Disable the OS-level Landlock sandbox (Linux only)")
    parser.add_argument("--compress-alg", default="context-compressor-llm",
                        help="Context compression: 'context-compressor-llm' (default; LLM anchored summaries, "
                             "falls back to truncate without the package) or 'truncate' (condense tool outputs, "
                             "drop the oldest messages)")
    parser.add_argument("--compress-target", type=float, default=0.4,
                        help="Fraction of the prompt budget kept after compressing (default: 0.4)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 'truncate' compression and a smaller system prompt")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort (e.g. low, medium, high)")
    parser.add_argument("--session", default=None,
                        help=f"Session file mirroring the conversation (default: ./{_SESSION_BASENAME}); "
                             f"rewritten after every step and reloaded on startup. Plain JSON: its "
                             f"\"messages\" array can be edited by hand. A second instance in the same "
                             f"directory uses {_SESSION_BASENAME}2, {_SESSION_BASENAME}3, ...")
    parser.add_argument("--no-session", action="store_true", help="Disable session persistence")
    parser.add_argument("request", nargs="*", help="Task request")
    args = parser.parse_args()
    if args.nogui and not args.request:
        parser.error("a task request is required with --nogui")
    if not args.nogui and not _TEXTUAL_AVAILABLE:
        sys.exit("The Textual library is not installed (pip install textual). Use --nogui for the plain CLI.")

    # Reserve the session file before anything can fail; the lock lives as
    # long as the process (the kernel drops it if the process dies).
    args.instance_number = 1
    if args.no_session or (args.nogui and not args.session):
        args.session_file = ""
        args.session_msg = ("[SESSION] Persistence disabled (--no-session)" if args.no_session
                            else "[SESSION] Persistence disabled (--nogui without --session)")
    else:
        args.session_file, args.instance_number = acquire_session_file(args.session)
        if not args.session_file:
            args.session_msg = "[SESSION] WARNING: could not lock a session file; running without persistence"
        elif args.instance_number > 1:
            args.session_msg = (f"[SESSION] WARNING: another AI-Commander instance is already using "
                                f"{_SESSION_BASENAME} in this directory; this instance uses "
                                f"{os.path.basename(args.session_file)} (instance {args.instance_number})")
        else:
            args.session_msg = f"[SESSION] Persisting session to {args.session_file}"

    history = None
    if not args.nogui:  # opened before the sandbox, which would block writes to $HOME
        history = PromptHistory(os.path.join(os.path.expanduser("~"), ".aic_history"))
        history.load()

    # The sandbox must cover the whole process before any agent thread starts.
    args.sandbox_write_dir = os.getcwd()
    if args.disable_sandbox:
        args.sandbox_enabled, args.sandbox_msg = False, "[SANDBOX] Sandbox disabled via --disable-sandbox"
    else:
        args.sandbox_enabled, args.sandbox_msg = enable_sandbox()

    if args.nogui:
        sink = ConsoleSink()
        sink.emit("LOG", {"text": args.sandbox_msg, "style": "green" if args.sandbox_enabled else "red"})
        sink.emit("LOG", {"text": args.session_msg,
                          "style": "green" if args.session_file and args.instance_number == 1 else "yellow"})
        commander = AICommander(
            api_base=args.api_base, model=args.model, api_key=args.api_key, auto_approve=args.auto_approve,
            show_thinking=not args.no_thinking, command_timeout=args.timeout,
            max_prompt_len=args.max_prompt_len, max_output_bytes=args.max_output_bytes, debug=args.debug,
            sink=sink, max_steps=args.max_steps,
            compress_algorithm="truncate" if args.fast else args.compress_alg,
            compress_target=args.compress_target, fast=args.fast, reasoning_effort=args.reasoning_effort,
            sandbox_enabled=args.sandbox_enabled, sandbox_write_dir=args.sandbox_write_dir,
            session_file=args.session_file, instance_number=args.instance_number)
        commander.persist_history = commander.load_session()  # resume where the last run stopped
        atexit.register(commander.save_session, "process_exit")

        def save_then_reraise(signum, _frame):
            """SIGTERM/SIGHUP skip atexit: save, then die with the default action."""
            commander.save_session(f"signal_{signum}")
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
            if sig is not None:
                with contextlib.suppress(ValueError, OSError, RuntimeError):
                    signal.signal(sig, save_then_reraise)
        try:
            commander.run(" ".join(args.request))
        except KeyboardInterrupt:
            commander._log("[INTERRUPTED] AI-Commander stopped by user")
            commander.save_session("interrupted")
            sys.exit(130)
        except Exception:
            commander._log_error("[FATAL ERROR] An unexpected error occurred:\n" + traceback.format_exc())
            sys.exit(1)
        return

    app = CommanderApp(args, history)

    def on_signal(signum, _frame):
        if app.agent:
            app.agent.save_session(f"signal_{signum}")
        if signum == getattr(signal, "SIGHUP", None):
            os._exit(128 + signum)  # the terminal is gone: nothing left to restore
        app.call_later(app.action_quit)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
        if sig is not None:
            with contextlib.suppress(ValueError, OSError, RuntimeError):
                signal.signal(sig, on_signal)
    try:
        app.run(mouse=not args.no_mouse)
    except KeyboardInterrupt:
        if app.agent:
            app.agent.request_stop()
            app.agent.save_session("interrupted")
        sys.exit(130)
    finally:
        history.close()


if __name__ == "__main__":
    main()
