# AI-Commander

AI-Commander is a **ralph-loop AI agent** that provides shell access and task
planning capabilities to Large Language Models through function calling. It
runs a tight agent loop: it plans, executes shell commands, observes the
results, and iterates until the requested task is complete — all while you stay
in control through an approval gate.

It supports two presentation modes:

- **Default (TUI)** — a [Textual](https://textual.textualize.io/)-based terminal
  UI with split panels (prompt, console, agent output, status footer) and an
  `/autoapprove` toggle for the approval gate.
- **`--nogui`** — the original direct-CLI behaviour (stdout/stderr, no TUI).

The agent logic (`AICommander`) is presentation-agnostic: all I/O is funnelled
through an `EventSink` interface. `ConsoleSink` replicates the original CLI
behaviour, while `TUISink` bridges events to the Textual app via a thread-safe
queue.

## Features

- **Ralph-loop agent** — repeatedly plans, executes, and observes until the
  task is done.
- **Shell access** — runs arbitrary commands and captures their output.
- **Approval gate** — review each command before it runs, or auto-approve.
- **Thinking tokens** — optional display/hiding of reasoning output.
- **Command timeouts** — prevent runaway commands (configurable).
- **OS-level sandbox** — optional [Landlock](https://landlock.io/) sandbox on
  Linux to restrict writes to the working directory, `/tmp`, `/dev`, and
  `/dev/pts`.
- **Two presentation modes** — full TUI or plain CLI (`--nogui`).

## Requirements

- Python 3.9+
- An OpenAI-compatible API endpoint (any provider that exposes the
  `/chat/completions` interface).

## Installation

Clone the repository and install the dependencies:

```bash
git clone <your-repo-url>
cd AICommander
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `textual` (used by the TUI) pulls in `rich` transitively, which the
> TUI code imports directly. The `py-landlock` package is only used on Linux and
> is optional — the agent runs unsandboxed if it is not installed.

## Usage

AI-Commander is a single-file script. Run it from the repository root:

```bash
python3 aic.py \
  --api-base https://api.example.com/v1 \
  --model your-model-name \
  --api-key YOUR_API_KEY \
  "your task request"
```

### Command-line options

| Option | Description |
| ------ | ----------- |
| `--api-base` | API base URL (required). |
| `--model` | Model name (required). |
| `--api-key` | API key (required). |
| `--auto-approve` | Auto-approve command execution (for testing). |
| `--no-thinking` | Hide thinking tokens from output. |
| `--timeout` | Command timeout in seconds (default: `30`). |
| `--max-prompt-len` | Maximum prompt length in characters (default: `80000`). |
| `--max-output-bytes` | Maximum output bytes returned from commands (default: `10240`). |
| `--debug` | Enable debug mode (dump conversation history on truncation). |
| `--nogui` | Run in direct CLI mode without TUI (original behaviour). |
| `--disable-sandbox` | Disable the OS-level Landlock sandbox (Linux only). |
| `request` | The task request (positional, required). |

### Example

```bash
# Run in the TUI (default)
python3 aic.py --api-base https://api.example.com/v1 \
  --model gpt-4o --api-key sk-... "List all files in this directory"

# Run in plain CLI mode
python3 aic.py --nogui --api-base https://api.example.com/v1 \
  --model gpt-4o --api-key sk-... "Summarize the contents of aic.py"

# Auto-approve all commands (useful for testing / scripting)
python3 aic.py --auto-approve --api-base https://api.example.com/v1 \
  --model gpt-4o --api-key sk-... "Set up a new Python project"
```

## How it works

1. The agent receives your task request.
2. It plans the next step and may emit *thinking* tokens.
3. It proposes shell commands, which you approve or auto-approve.
4. Commands run (optionally sandboxed and time-limited), and their output is
   fed back to the model.
5. The loop repeats until the task is considered complete.

All I/O is routed through the `EventSink` abstraction, so the same agent logic
runs unchanged in both the TUI and `--nogui` modes.

## Sandboxing (Linux)

By default, on Linux, the agent attempts to enable an OS-level
[Landlock](https://landlock.io/) sandbox using `py-landlock`. When active,
writes are restricted to the current working directory, `/tmp`, `/dev`, and
`/dev/pts`. The status is shown in the agent output panel (green when active,
red when unsandboxed).

- Disable it with `--disable-sandbox`.
- If `py-landlock` is not installed, the agent simply runs unsandboxed.

## Project structure

```
├── aic.py            # The entire agent (single-file implementation)
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── LICENSE           # Apache 2.0
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
