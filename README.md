# byom-collab

Run a coding model on a Google Colab GPU and use it with local coding agents
through an OpenAI-compatible API at `http://127.0.0.1:4000/v1`. The VM shuts
down automatically when idle to avoid wasting compute units.

> **Status:** Experimental MVP. The tunnel, supervision, and automatic
> shutdown have been validated on Colab; inference with the Qwen 32k/256k
> profiles is still being validated.

## Requirements

* macOS or Linux with `ssh` and [`uv`](https://docs.astral.sh/uv/).
* The [Colab CLI](https://github.com/googlecolab/google-colab-cli), installed
  and authenticated (`byom` does not provide its own login):

  ```bash
  uv tool install google-colab-cli
  colab sessions    # completes the login flow
  ```

* A Colab account with available compute units and access to the GPU required
  by the selected profile (A100 for the Qwen profiles; high-memory runtime for
  the 256k profile).

## Quick start

```bash
uv sync
uv run byom init                                  # checks dependencies, creates an SSH key and config
uv run byom configure --profile qwen3.8-27b-32k   # selects a model profile
uv run byom up                                    # allocates a GPU and starts the endpoint
uv run byom status                                # shows session, worker, tunnel, and account usage
uv run byom down                                  # stops the session
```

* `byom profiles` lists the available profiles: `qwen3.8-27b-32k` and
  `qwen3.8-27b-256k`. Stop the session before switching profiles.
* `byom up --force` stops the previous session before creating a new one (the
  model is loaded from scratch).

## Using coding agents

| Agent | Configuration |
| :--- | :--- |
| OpenCode | Automatic: `byom up` adds the provider and sets the default model |
| dsh (DeepSeek harness) | Automatic: `byom up` adds the `byom` provider and sets the default model |
| Aider | `OPENAI_API_BASE=http://127.0.0.1:4000/v1` and `OPENAI_API_KEY=dummy` |
| Cline | OpenAI-compatible provider with base URL `http://127.0.0.1:4000/v1` |

**OpenCode:** `byom up` merges the provider, discovery plugin, and 32k/256k
aliases into the global config (`$XDG_CONFIG_HOME/opencode` or
`~/.config/opencode`) while preserving the rest. Use
`--opencode-config <file>` to specify another path, and restart OpenCode after
changes. JSONC files (with comments or trailing commas) are not rewritten, so
`up` aborts before allocating a GPU.

**dsh:** if `~/.dsh/profiles/desktop/cordis.patch.yml` exists (or
`--dsh-config <file>` is given), `byom up` adds `providers.byom` to the
`llm-pi-ai` entry (OpenAI-compatible, `baseURL` pointing at the local proxy, and
the model with the active profile's `contextWindow`, and
`apiKeyEnv: BYOM_API_KEY` unless you already set one) and sets
`agent-default-model` to `byom`. Nothing else is changed or removed: other
providers, models, API key variables, entries and comments stay as they were.
Export any value, e.g. `export BYOM_API_KEY=dummy` (the proxy does not check it)
before starting dsh. The file is edited as text; a non-list, tab-indented, or
non-standard `byom` block file aborts `up` before allocating a GPU (the file is
left untouched). Restart dsh after changes.

**Not supported:** Cursor (requests are sent from Cursor's backend and cannot
reach your localhost) and Claude Code (it uses the Anthropic Messages API at
`/v1/messages`).

**Reasoning:** If the client does not send `reasoning_effort`, the gateway
injects `medium` (the model's default, `xhigh`, generates many more tokens).
Supported values: `low`, `medium`, `xhigh`.

## How it works

```text
Agent --HTTP/SSE--> Proxy :4000 --> Tunnel :4001 ==SSH (via colab ssh)==> Colab VM
                                                                        Gateway :8001 --> vLLM :8000
                                                                           ^
                                                   Watchdog (cell) --------+--> runtime.unassign()
```

* **Gateway:** All inference requests pass through it, making it the source of
  truth for activity (`/health` and `/v1/models` do not count as work).
* **Watchdog:** A long-running kernel cell that checks the gateway every 10
  seconds and calls `unassign()` after the idle timeout. The local daemon
  restarts it if the connection drops.
* **Safe shutdown:** The VM shuts down only after the gateway confirms there
  is no active work. If the watchdog misses heartbeats for more than 60
  seconds, the gateway stops vLLM.
* **SSH:** A single OpenSSH master connection (`ControlMaster`, with
  `colab ssh` as the `ProxyCommand`) carries the tunnel and remote commands.

## Configuration

Configuration file: `~/.config/byom/config.toml` (created by `init`; profiles
set the model-related fields):

| Key | Default | Description |
| :--- | :--- | :--- |
| `profile` | empty | Active profile |
| `model` | `philbert440/Qwen3.8-27B-W4A16-AWQ` | Checkpoint served by vLLM |
| `served_model_name` | `qwen3.8-27b` | Model ID exposed by the API |
| `max_model_len` | `16384` | Maximum context length |
| `reasoning_effort` | `medium` | Value injected when the client omits it |
| `gpu` | `L4` | Requested accelerator |
| `gpu_high_mem` | `false` | Requests a high-memory runtime (A100; eligible plan required) |
| `idle_timeout_min` | `20` | Idle time before shutdown |
| `max_job_min` | `40` | Timeout for a stuck job |
| `bootstrap_timeout_min` | `30` | Time allowed for the worker to become ready |
| `watchdog_timeout_s` | `60` | Watchdog timeout that stops vLLM |
| `proxy_port` / `tunnel_port` | `4000` / `4001` | Local ports |
| `gateway_port` / `vllm_port` | `8001` / `8000` | VM ports |
| `gpu_memory_utilization` | `0.92` | Fraction of GPU memory used by vLLM |
| `kv_cache_dtype` | `fp8` | KV cache data type |
| `vllm_spec` | `vllm` | vLLM package to install (version not pinned) |
| `vllm_enforce_eager` | `false` | Disables CUDA graphs if memory is insufficient |
| `vllm_language_model_only` | `true` | Does not reserve memory for the vision tower |
| `ssh_key` | `~/.ssh/byom_ed25519` | Tunnel key |

Session state is stored in `~/.config/byom/state.json` (no secrets). `byom`
sessions use a separate store (`~/.config/byom/colab-sessions.json`) from your
other Colab sessions.

## Known issues

**One SSH connection per session.** The runtime accepts only one `colab ssh`
connection at a time. Running another `colab ssh` disconnects `byom` (HTTP
429). Use `byom status` to inspect the worker.

**Orphaned sessions.** If the Colab CLI allocates a VM but fails to save its
local state, the session appears as `[?]` in `colab sessions`, continues
consuming compute units, and cannot be stopped with `colab stop`. `byom status`
and `byom up` warn when this happens. Stop the session from the Colab website
or run:

```bash
~/.local/share/uv/tools/google-colab-cli/bin/python -c \
  "from colab_cli.common import state; state.client.unassign('<endpoint>')"
```

where `<endpoint>` is the second field in the `colab sessions` output.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests do not access Colab: they use a fake vLLM server and a stub Colab CLI.

To validate against real Colab (the script creates and stops its own sessions;
use `--keep` to retain them for inspection):

```bash
uv run python scripts/spike.py --phase cheap   # CPU: tunnel, tmux, watchdog, unassign()
uv run python scripts/spike.py --phase model   # GPU (T4 by default; --gpu selects another): vLLM, decode, tool calling
```

The `model` phase downloads about 20 GB of model weights and uses compute
units. Use `--enforce-eager` if you run out of memory.

### Project structure

```text
src/byom/
  cli.py                  init, profiles, configure, up, status, and down commands
  profiles/               Qwen3.8 profiles and OpenCode aliases
  opencode.py             merges OpenCode configuration
  dsh.py                  merges dsh (DeepSeek harness) configuration
  colab.py                Colab CLI calls and kernel cells
  tunnel.py, remote.py    SSH master connection, tunnel, and VM commands
  daemon.py, proxy.py     local proxy process and tunnel supervision
  watchdog_supervisor.py  keeps the Colab watchdog cell alive
worker/                   files uploaded to the VM
  bootstrap.sh            installs dependencies and starts vLLM and gateway in tmux
  build_vllm_command.py   builds vLLM arguments from the active profile
  gateway.py              inference proxy, activity tracking, and shutdown coordination
  watchdog.py             heartbeat and unassign()
scripts/spike.py          manual validation against real Colab
```
