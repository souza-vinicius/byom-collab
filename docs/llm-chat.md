---
type: Feature
title: Isolated llm chat Integration
description: Idempotent setup, isolated `llm chat` execution via `LLM_USER_PATH`, and marker-guarded uninstall under `byom_home()`.
tags: [cli, llm, chat, isolation, setup, uninstall]
status: stable
generated: { by: hermes/hy4-flash-free, at: 2026-11-28T03:50:00Z }
sources:
  - id: llm-chat-module
    resource: /src/byom/llm_chat.py
    title: Isolated setup, chat runner, and safe uninstall implementation
  - id: config-module
    resource: /src/byom/config.py
    title: Centralized `byom_home()` path resolver and atomic 0600 file writes
  - id: cli-module
    resource: /src/byom/cli.py
    title: Typer commands `setup`, `chat`, and `uninstall`
---

# Isolated llm chat Integration

`byom` integrates Simon Willison's `llm chat` CLI for talking to a remote OpenAI-compatible model served on Google Colab while keeping every persisted artifact inside the exclusive `byom` directory.[^llm-chat-module]

## Directory Resolution (`byom_home()`)

Path resolution is centralized in `byom_home()` in [src/byom/config.py](/src/byom/config.py) using only the Python standard library, with `config_dir()` delegating to it.[^config-module]

1. `BYOM_HOME`: explicit override (empty/whitespace raises `ConfigError`).
2. `BYOM_CONFIG_DIR`: legacy override preserved for test isolation.
3. OS-standard per-user directory named `byom`:
   - Linux/POSIX: `$XDG_CONFIG_HOME/byom` or `~/.config/byom`
   - macOS (`darwin`): `~/Library/Application Support/byom`
   - Windows (`win32`): `%APPDATA%\byom` or `~/AppData/Roaming/byom`

## Commands (`setup`, `chat`, `uninstall`)

All three commands are exposed via [src/byom/cli.py](/src/byom/cli.py) and implemented in [src/byom/llm_chat.py](/src/byom/llm_chat.py).[^cli-module]

- **`byom setup`**: Idempotently creates `byom_home()` (`0700` on POSIX) and atomically writes (`0600`) `.byom-managed`, `llm-chat.json`, `extra-openai-models.yaml` (rendered with stdlib `json.dumps`), `default_model.txt`, and optionally `keys.json`. Passing `--url` updates the remote endpoint while preserving previously saved model preferences. `--model-id` sets the local `llm` id (default `byom`) and `--api-key` stores the endpoint key in `keys.json` (`0600`).
- **`byom chat`**: Requires `.byom-managed`, `llm-chat.json`, and `extra-openai-models.yaml` to exist before running `[sys.executable, "-m", "llm", "chat", ...]` in a subprocess with `LLM_USER_PATH` set exclusively in the child environment. Supports `--system` (`-s`), `--continue` (`-c`), `--model` (`-m`), and `--no-history` / `--no-log` (`-n`). Because `llm 0.38`'s `llm chat` subcommand reads `logs_on()` via `(user_dir() / "logs-off").exists()` rather than accepting a CLI `--no-log` flag, `byom chat --no-history` temporarily creates `logs-off` inside `byom_home()` during the subprocess run and restores its prior state in a `finally` block. History persists inside `byom_home() / "logs.db"`. Normal cancellation (`Ctrl+D` or `Ctrl+C`, reported by click as `Aborted!` with exit code 1) is treated as a clean end of the conversation and exits 0. A Ctrl+C caught by `run_chat` itself returns exit code 130, also classified as `aborted`, so the CLI and the TUI still report a clean end without a traceback; real failures keep the child stderr and return its exit code.
- **Chat errors**: `run_chat()` captures the child's stderr in `ChatResult`; the CLI prints it on a real failure and the TUI includes it in the `Detalhes técnicos` panel, so an unavailable model API is visible instead of a generic message.
- **`byom uninstall`**: Refuses empty paths, `.`/`..`, symlinks, filesystem root, user `HOME`, and any directory lacking the `.byom-managed` marker file. Lists entries inside `byom_home()`, prompts for confirmation unless `--yes` (`-y`) is given, removes only `byom_home()`, and prints `pip uninstall byom`.

[^llm-chat-module]: `src/byom/llm_chat.py`
[^config-module]: `src/byom/config.py`
[^cli-module]: `src/byom/cli.py`
