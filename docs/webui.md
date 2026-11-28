---
type: Feature
title: Open WebUI Frontend
description: How `byom web` runs an isolated Open WebUI instance on 127.0.0.1:8080 against the local OpenAI-compatible proxy, where chats and uploads live, and why image generation still needs two runtimes.
tags: [frontend, open-webui, webui, chat, ui, uploads]
status: draft
generated: { by: hermes/hy4-flash-free, at: 2026-11-28T12:30:00Z }
sources:
  - id: webui-module
    resource: /src/byom/webui.py
    title: Command, environment and process runner for the local Open WebUI instance
  - id: cli-module
    resource: /src/byom/cli.py
    title: The `web` command and its options
  - id: llm-chat
    resource: /src/byom/llm_chat.py
    title: Shared `byom_home()` marker used by `byom uninstall`
  - id: proxy
    resource: /src/byom/proxy.py
    title: Local OpenAI-compatible proxy at 127.0.0.1:4000/v1
  - id: openwebui-quickstart
    resource: https://docs.openwebui.com/getting-started/quick-start/
    title: Open WebUI quick start (Python install, single-user mode, data volume)
  - id: openwebui-images
    resource: https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/usage
    title: Image generation requires a chat model selected plus an image connection
  - id: openwebui-env
    resource: https://docs.openwebui.com/reference/env-configuration/
    title: OpenAI connection variables, DEFAULT_MODELS and ENABLE_PERSISTENT_CONFIG (consulted 2026-11-28)
---

# Open WebUI Frontend

`byom web` starts a local [Open WebUI](https://openwebui.com) instance and points
it at the byom proxy. The browser talks to Open WebUI on
`http://127.0.0.1:8080`, and Open WebUI talks to the model on
`http://127.0.0.1:<proxy_port>/v1`. Both bindings are loopback only.

## What it starts

The command builds and runs:

```text
uv tool run --python 3.12 --from open-webui==0.13.5 open-webui serve \
  --host 127.0.0.1 --port 8080
```

* Open WebUI is pinned to `0.13.5` in `src/byom/webui.py`.
* Python 3.12 is requested because Open WebUI declares `>=3.11,<3.13`. The byom
  virtualenv may be 3.13, so the frontend runs in its own `uv tool` environment.
* The first run downloads Open WebUI and Python 3.12 into the uv cache. Later
  runs reuse them. `uv` must be on `PATH`.
* The frontend runs in the foreground. `Ctrl+C` stops it; the Colab session
  started by `byom up` is independent and keeps running.

## Isolation and data

| Path | Contents |
| :--- | :--- |
| `byom_home()/webui` | SQLite `webui.db`, uploads and cache |
| `byom_home()/webui-secret` | Session secret (`0600`), exported as `WEBUI_SECRET_KEY` |
| `byom_home()/.byom-managed` | Marker that lets `byom uninstall` remove the directory |

`DATA_DIR` points Open WebUI at that directory, so the instance does not read or
write a global Open WebUI installation. The secret is generated once; keeping it
stable preserves sessions across restarts and prevents Open WebUI from writing
`.webui_secret_key` into the current working directory.

`WEBUI_AUTH=false` runs Open WebUI in single-user mode. That is safe because the
server listens only on `127.0.0.1`. Open WebUI does not support switching an
existing data directory between single-user and multi-account mode, so a
multi-user setup needs a fresh `byom_home()`/`webui` directory.

## How it reaches the model

`byom web` exports:

* `OPENAI_API_BASE_URL` and `OPENAI_API_BASE_URLS` as
  `http://127.0.0.1:<proxy_port>/v1`, with `proxy_port` from `config.toml`
  (default 4000).
* `OPENAI_API_KEY` and `OPENAI_API_KEYS` as `byom`. The proxy strips the
  `Authorization` header and does not authenticate local clients.
* `DEFAULT_MODELS` as `config.served_model_name` — the same id the gateway
  publishes in `GET /v1/models` (the runtime alias, or the synthetic id on the
  `tts` runtime). Open WebUI pre-selects it on a new chat.[^env]
* `ENABLE_OLLAMA_API=false` and `ENABLE_VERSION_UPDATE_CHECK=false`.
* `ENABLE_PERSISTENT_CONFIG=false`, so a URL previously saved in `webui.db`
  through the Admin UI cannot override the environment on the next start. The
  trade-off is documented by Open WebUI: admin-panel changes still apply to the
  running session but are lost on restart; byom's environment becomes the
  source of truth.[^persist]

`build_env` **fills only what is missing**: any variable already exported by
the user wins. `WebUIPlan.overrides` records which of the byom defaults were
supplied by the environment, and `byom web` prints them as `Ambiente: ...`.
That makes the documented override path real — export `OPENAI_API_BASE_URLS`
and `OPENAI_API_KEYS` before `byom web` to point the frontend at another
provider, without editing any file. Variables owned by the isolation
(`DATA_DIR`, `WEBUI_SECRET_KEY`, `WEBUI_URL`) are also preserved if exported;
the default secret remains generated once with mode `0600`.

[^env]: `DEFAULT_MODELS` accepts comma-separated model ids that are
pre-selected in a new chat; if a stored model no longer exists it is replaced
by these defaults — Open WebUI environment reference, 2026-11-28.

[^persist]: `ENABLE_PERSISTENT_CONFIG` defaults to `true`, which makes values
stored in `webui.db` take precedence over environment variables. With `false`,
environment variables always win and UI changes are not kept across restarts —
Open WebUI environment reference and connection troubleshooting, 2026-11-28.

Open WebUI makes these calls from its backend, not from the browser. That
matters because the proxy only accepts `Host` equal to `127.0.0.1` or
`localhost` and does not send CORS headers (`src/byom/proxy.py`). A server-side
client satisfies both rules without any change to the proxy.

The model list comes from `GET /v1/models`, so it always reflects the runtime
that is currently active. Streaming chat uses SSE and passes through the proxy.
Image routes are non-stream and the proxy allows up to 600 s for them.

## Features

* Conversations, history, search and per-chat settings stored in the isolated
  SQLite database.
* System prompt per conversation and per model.
* File uploads. Images reach vision-capable profiles that load an `mmproj`; the
  profile must declare it. Document RAG uses Open WebUI's own embedding settings,
  which `byom web` does not configure, so it may need a one-time setup in the
  admin panel.
* Model parameters and presets.
* Generated images are displayed inline when an image tool returns them.

## Limits

* **One runtime per session.** A profile switch requires `byom down`,
  `byom configure --profile ...` and `byom up`. Open WebUI's model list changes
  with the active runtime.
* **Image models do not chat.** The gateway answers `/v1/chat/completions` with
  501 for the `image` runtime and exposes only `/v1/images/generations` and
  `/v1/images/edits`.
* **Image generation needs two runtimes.** Open WebUI generates images from a
  chat with a normal chat model selected, using a separate image connection. A
  single byom session cannot serve a chat model and the image model at the same
  time, so the combination is not available today. Viewing images requires
  either a second byom session or a dedicated image panel.
* **Idle shutdown.** Browsing the UI does not count as activity; only model
  requests do. The default `idle_timeout_min` is 20 minutes.
* **Concurrency.** The gateway accepts up to 4 concurrent jobs
  (`DEFAULT_MAX_CONCURRENT_JOBS`) and cancels jobs past `max_job_min` (40 min).

## Commands

```bash
uv run byom web                 # http://127.0.0.1:8080
uv run byom web --port 8099     # another local port
uv run byom web --no-browser    # do not open the browser automatically
```

Start the session first with `byom up`; otherwise Open WebUI starts with an empty
model list. `byom uninstall` removes the whole `byom_home()` directory, including
the frontend data. To update Open WebUI, bump `OPEN_WEBUI_VERSION` in
`src/byom/webui.py`; the next run resolves the new pin in the uv cache.

## Verification

The command, environment, `DEFAULT_MODELS`, preserved user overrides, port
validation, occupied-port refusal, browser open, `Ctrl+C` handling and CLI
output are covered by 13 mock-based cases in `tests/test_webui.py`. None of them
starts Colab, a worker or an Open WebUI instance. The first real run still
needs a machine with `uv` and network access to download the pinned package.
