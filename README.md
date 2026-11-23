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
uv run byom gpus                                  # static GPU catalog and profiles that fit
uv run byom models                                # asks for a GPU, then lists compatible models
uv run byom models --gpu L4                       # lists models compatible with L4, no prompt
uv run byom configure --profile qwen3.8-27b-32k   # selects a model profile
uv run byom up                                    # allocates a GPU and starts the endpoint
uv run byom status                                # shows session, worker, tunnel, and account usage
uv run byom down                                  # stops the session
```

* `byom profiles` lists all profiles; `byom gpus` shows the static GPU catalog.
  Stop the session before switching profiles.
* `byom models --gpu L4` lists only the catalog variants whose served context
  fits that GPU, with no prompt. Without `--gpu`, `byom models` prints the
  numbered GPU list, asks for one number or ID (case-insensitive), and then
  prints only the compatible variants. An invalid entry stops with the
  accepted IDs; no GPU is chosen automatically.
* The GPU list is static and its VRAM values are approximate, from the Google
  Cloud source recorded in `src/byom/catalog.py` — not from the Colab account.
  Selecting a GPU for `byom models` only filters the listing: it does not query
  the account, Colab or Hugging Face, allocate a VM, download a model, or write
  `config.toml`, `state.json` or any other configuration file. Use
  `byom configure --profile ...` when you actually want to persist a choice.
* `byom gpus` does **not** query the account. Pick the GPU in `config.toml`;
  `byom up` refuses to start if the declared GPU cannot fit the selected
  variant (it never allocates a different GPU).
* `byom up --force` stops the previous session before creating a new one (the
  model is loaded from scratch).

## GPU and model catalogs

`src/byom/catalog.py` is the single source of GPU/VRAM and model/quantization
numbers; the commands render it. `byom up` checks
`weights + 2 GiB runtime + 2 GiB per 32K of context` against the GPU declared
in `config.toml`. Image models do not reserve autoregressive KV cache.

GPUs (VRAM is **approximate for Colab**: Google's Colab FAQ states GPU types
vary over time and publishes no VRAM table; the numbers below are from
`docs.cloud.google.com/compute/docs/gpus`, "General comparison chart", checked
2026-11-23):

| GPU (catalog ID) | VRAM | bf16 | Colab CLI value |
| :--- | ---: | :--- | :--- |
| T4 | 16 GiB | no | `T4` |
| L4 | 24 GiB | yes | `L4` |
| A100 40 GB | 40 GiB | yes | `A100`, `gpu_high_mem=false` |
| A100 80 GB | 80 GiB | yes | `A100`, `gpu_high_mem=true` |
| H100 80 GB | 80 GiB | yes | `H100` |

First batch (docs checked 2026-11-23). "Context" below is the maximum in the
official page; the served context is smaller on most profiles to fit the GPU
declared in `config.toml` (`byom models` prints both). `byom models --gpu <ID>`
uses the same served-context budget as `byom up` to filter the listing; it does
not change the declared GPU:

| Model | Runtime | Variant | Memory | Context doc | Context served | Minimum GPU |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| Inkling 2 Mini | llama.cpp | 2 bit / 4 bit (17.6 GB) / 8 bit | 11 / 18 / 24 GiB | 1,000,000 | 32,768 / 65,536 / 131,072 | T4 / L4 / A100 |
| Qwen3.8 27B | vLLM (AWQ) | W4A16-AWQ | 17 GiB | 262,144 | 32,768 or 262,144 | L4 (A100 for 256K) |
| Qwen3.8 27B | llama.cpp | 3 bit / 4 bit | 12 / 17 GiB | 262,144 | 32,768 | T4 / L4 |
| Gemma 4 | llama.cpp | E4B / 12B / 26B-A4B / 31B, 4 bit | 6 / 8 / 18 / 20 GiB | 131,072 (E2B/E4B), 262,144 (others) | 65,536 / 32,768 / 32,768 / 131,072 | T4 to A100 |
| Muse Glimmer 30B | llama.cpp | 2 bit / 4 bit / 6 bit | 12 / 17 / 22 GiB | 131,072 (up to 262,144) | 32,768 / 65,536 / 131,072 | T4 to A100 |
| Qwen-Image-2.1 | image | INT8-ConvRot / GGUF Q4_K_M | 20 / 12 GiB | n/a | n/a | L4 / T4 |

The reduction is data, not prose: `src/byom/catalog.py` holds
`Variant.context_doc`, `Variant.context` and `PROFILE_CONTEXTS`, one entry per
profile with the GPU and the reason (KV cache grows with context; 2 GiB per
32K in the budget). `byom models` shows both numbers; `byom profiles` and
`byom configure` print the doc/served pair with the reason. Compatibility
checks keep using the SERVED context — the documented maximum is never used
to allocate a GPU that cannot hold the KV cache.

`reasoning_effort` is also a model property in the catalog, with the source
recorded in `ModelEntry.reasoning_source` (all checked 2026-11-23):

| Model | Documented default | Source |
| :--- | :--- | :--- |
| Inkling 2 Mini | `high` | huggingface.co/blog/inkling2mini and `thinkingmachines/Inkling-2-Mini/chat_template.jinja` — template default 0.9; scale none/minimal/low/medium/high/xhigh/max. The briefing also mentions `xhigh` in Thinking Machines' evaluation, but that exact line was **not** located on the consulted pages. |
| Qwen3.8 27B | `xhigh` | unsloth.ai/docs/models/qwen3.8.md — "xhigh (default)" and "xhigh reasoning" in the benchmark settings |
| Muse Glimmer 30B | `high` | unsloth.ai/docs/models/muse-glimmer.md — "Thinking Settings" lists low/medium/high/xhigh and the benchmark table says "High Reasoning" |
| Gemma 4 | none | unsloth.ai/docs/models/gemma-4.md — the page does **not** define `reasoning_effort`; it defines `<|think|>` and `enable_thinking`. No value was invented. |
| Qwen-Image-2.1 | none | diffusion model; no chat reasoning field |

The whitelist in `config.toml`, `session.json` and the gateway is exactly the
union documented by those pages: `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` (plus `null`/absence for Gemma and image). Numbers such as
`0.9` are accepted by Inkling's template but are intentionally rejected at
our boundary: no free-form string or number.

Profile notes: Inkling 2 Mini uses the mainline llama.cpp `b12611` build with
the `Mainline_Support` GGUFs (the repo's `.gguf_file` extension is renamed to
`.gguf` on download). Older mainline builds lose the per-expert NVFP4 scales
on the second and third looped passes and produce wrong output. DSpark is off:
mainline support is still an open pull request. Muse Glimmer uses pinned
upstream `b12649`.
Qwen3.8 keeps the existing AWQ profile for vLLM and adds GGUF options. Gemma 4
profiles carry an explicit `thinking_policy` key from the catalog: the gateway
adds `<|think|>` at the start of the system prompt (creating the system message
when absent) and removes `<|channel>thought … <channel|>` blocks from prior
assistant turns before forwarding. A client that already placed the token,
cleaned the history, or sent `enable_thinking: false` keeps its intent.
All binaries are
downloaded from pinned URLs with SHA-256 from the release assets; model repo,
quantization and revision pattern are fixed in the profile — no `@latest`.

Out of scope (no executable profile): MiniMax-M3.1-Flash (1 bit 129.4 GB,
2 bit 143.9 GB, 4 bit 249.4 GB / 256 GB+ budget), GLM-5.3-Flash (1 bit 93.09 GB,
3 bit 120.37–147.54 GB, 4 bit 199.71 GB) and DeepSeek-V4.1-Flash (1 bit
128–145 GB, 4 bit download 500.7 GB, table 305 GB + Engram on SSD). None fits a
single Colab GPU.

### Runtimes and endpoints

* `vllm` and `llama-cpp` serve `POST /v1/chat/completions` and `GET /v1/models`
  through the same proxy and gateway. `llama-server` is started with the pinned
  GGUF, the `mmproj` vision projector when present, and DSpark draft when the
  profile requests it.
* `image` serves `POST /v1/images/generations` and `POST /v1/images/edits` via
  `sd-server`'s OpenAI-compatible image API, plus `GET /v1/models` for model
  metadata. It is a diffusion model, **not** a chat model:
  `POST /v1/chat/completions` returns HTTP 501 for this runtime. Requests take
  `prompt` and image options, not `messages`; responses contain image data
  `data[].b64_json`), not tokens or finish reasons. The Unsloth page documents
  defaults of **25 steps and guidance 1** (checked 2026-11-23); both live only
  in `ModelEntry.generation_defaults`. sd-server does **not** accept a
  top-level `guidance` field: the gateway maps the pair into the documented
  native extension inside `prompt` —
  `<sd_cpp_extra_args>{"sample_params":{"sample_steps":25,"guidance":{"txt_cfg":1.0}}}</sd_cpp_extra_args>`
  (`examples/server/api.md`, checked 2026-11-23). A client can override
  `steps`/`guidance` in the body or supply its own `sd_cpp_extra_args` block.
  Divergence recorded: the same Unsloth page suggests `--steps 20 --cfg-scale
  6.0` for the sd-cli workflow; this project uses the documented API defaults
  (25/1) from the briefing, not a value invented for sd.cpp.
  `POST /v1/images/edits` is multipart and keeps the server's defaults. On a 15 GB
  Colab T4, one 1024x1024 image at 25 steps takes about 120 s in fp16; up to
  ~14 GB of GPU memory and 24 GB of system RAM are used. T4 has no bf16, so
  training is refused by Unsloth; bf16 training needs Ampere or newer. The
  pinned Linux asset's name does not state its backend; verify GPU use on the
  first real run. If it is a CPU build, build from the pinned tag with
  `-DSD_CUDA=ON` and point `BYOM_SD_SERVER` at the resulting `sd-server`.
* Hugging Face repositories and the Colab CLI are **verified only on the first
  real run**: this repository does not call them during local tests. Check the
  model files resolved from the pinned patterns and the GPU actually assigned
  by `colab new` before relying on a profile.

Sources consulted 2026-11-23: Unsloth model pages
(`inkling-2-mini.md`, `qwen3.8.md`, `gemma-4.md`, `muse-glimmer.md`,
`qwen-image-2.1.md`, plus the three "Flash" pages), Hugging Face's
`blog/inkling2mini` and `thinkingmachines/Inkling-2-Mini/chat_template.jinja`,
`leejet/stable-diffusion.cpp` `examples/server/api.md`, Google Cloud GPU machine
types and Colab FAQ, Hugging Face `unsloth/Inkling-2-Mini-GGUF`
(`Mainline_Support/README.md`, `HOW_TO_RUN.md` and discussion #8),
GitHub release asset lists for `ggml-org/llama.cpp` `b12611` and `b12649`,
llama.cpp PRs #32694 and #32704, and
`leejet/stable-diffusion.cpp` `master-1106-532d2c1`.

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

**Reasoning:** each model carries its documented default in the catalog; the
profile inherits it and the client can override it per request. If the client
omits `chat_template_kwargs.reasoning_effort`, the gateway injects that value —
`high` for Inkling 2 Mini, `xhigh` for Qwen3.8, `high` for Muse Glimmer. For
Gemma 4 the catalog value is `null` (the page does not define the field) and
**nothing is injected**. Valid values are exactly `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, `max`, validated in `config.toml`, `session.json`
and at the gateway boundary; an invalid value aborts startup. A client that
sends a non-object `chat_template_kwargs` keeps its original payload. Sources
and the 2026-11-23 check date are in the table above and in
`ModelEntry.reasoning_source`.

## How it works

```text
Agent --HTTP/SSE--> Proxy :4000 --> Tunnel :4001 ==SSH (via colab ssh)==> Colab VM
                                                                        Gateway :8001 --> runtime :8000
                                                                          (vLLM | llama-server | sd-server)
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
| `runtime` | `vllm` | Runtime selected by the profile: `vllm`, `llama-cpp` or `image` |
| `model` | `philbert440/Qwen3.8-27B-W4A16-AWQ` | Checkpoint served by vLLM |
| `served_model_name` | `qwen3.8-27b` | Model ID exposed by the API |
| `max_model_len` | `16384` | Maximum context length |
| `reasoning_effort` | inherited | Documented model default from the catalog (Qwen3.8 `xhigh`, Inkling `high`, Muse `high`; Gemma/image omitted). A profile or the client can override it. |
| `gpu` | `L4` | Requested accelerator |
| `gpu_high_mem` | `false` | Requests a high-memory runtime (A100; eligible plan required) |
| `idle_timeout_min` | `20` | Idle time before shutdown |
| `max_job_min` | `40` | Timeout for a stuck job |
| `bootstrap_timeout_min` | `30` | Time allowed for the worker to become ready |
| `watchdog_timeout_s` | `60` | Watchdog timeout that stops vLLM |
| `proxy_port` / `tunnel_port` | `4000` / `4001` | Local ports |
| `gateway_port` / `vllm_port` | `8001` / `8000` | VM ports |
| `gpu_memory_utilization` | `0.92` | Fraction of GPU memory used by vLLM (must be in `(0, 1]`) |
| `kv_cache_dtype` | `fp8` | KV cache data type |
| `vllm_spec` | `vllm` | vLLM package to install (version not pinned) |
| `vllm_enforce_eager` | `false` | Disables CUDA graphs if memory is insufficient |
| `vllm_language_model_only` | `true` | Does not reserve memory for the vision tower |
| `ssh_key` | `~/.ssh/byom_ed25519` | Tunnel key |

Session state is stored in `~/.config/byom/state.json` (written with mode 0600,
serialized by a lock file, and containing the worker's identity nonce: it is not
a user credential, but it should not be shared). `byom` sessions use a separate
store (`~/.config/byom/colab-sessions.json`) from your other Colab sessions.
`session.json` additionally carries `thinking_policy` (explicit Gemma 4 key
from the catalog) and `generation_defaults` (`steps`/`guidance` for Qwen-Image),
both validated by the gateway at startup.

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

## Design decisions and unverified external behavior

Recorded on 2026-11-23 against the local clone at commit `cba79b8`.

* **Documented values are catalog data.** `reasoning_effort`, the Gemma 4
  `thinking_policy`, Qwen-Image `generation_defaults`, and both context numbers
  live in `src/byom/catalog.py`; profiles inherit them and the gateway receives
  them in `session.json`. Sources and dates for every value are recorded in
  the catalog and in the tables above. No profile repeats `medium`, `25` or a
  documented context maximum.
* **Context is not silently reduced.** `PROFILE_CONTEXTS` lists all 15 profiles
  with doc context, served context, GPU and reason. `byom models` prints both;
  `check_fit` still uses the served value from `config.toml`. Raising a profile
  to the documented maximum without a larger GPU is explicitly out of scope.
* **Thinking is keyed by behavior, not model name.** Only a model with
  `ModelEntry.thinking` gets the Gemma 4 prompt transformation. The client's
  own token, clean history, or `enable_thinking: false` wins.
* **Image defaults use the server's names.** Unsloth says "steps/guidance";
  sd-server documents `sample_params.sample_steps` and
  `sample_params.guidance.txt_cfg` via `sd_cpp_extra_args`. The equivalence is
  recorded in `ModelEntry.generation_source`; no top-level `guidance` is sent.

* **Corrupted `state.json` blocks `up`.** An absent file means "no session"; an
  unreadable or invalid file is an error. `up` also checks `orphan_assignments()`
  on both the normal and `--force` paths and **blocks** (without allocating)
  while orphans exist. Rationale: an orphan keeps consuming compute units and
  cannot be stopped by `colab stop`; asking for confirmation would leave the
  decision to a user who may not see the backend state. Fix it in Colab, then
  retry.
* **Gateway fallback deallocation.** The watchdog cell remains the primary
  caller of `runtime.unassign()`. If it (or its local supervisor) fails, the
  gateway schedules one fallback via the same `google.colab.runtime` module,
  30 seconds after an accepted idle shutdown or after vLLM is stopped. It skips
  deallocation while `active_jobs > 0` and never calls external endpoints.
* **Job ceiling.** The gateway accepts at most
  `BYOM_MAX_CONCURRENT_JOBS` concurrent jobs (default `4`, invalid values fall
  back to `4`); excess requests get HTTP 503. Set the environment variable in
  the VM environment to change it; it is not part of `session.json`.
* **Internal endpoints.** `/internal/state` and `/internal/begin-shutdown`
  accept loopback clients only (403 otherwise). No client authentication was
  added because the gateway binds `127.0.0.1` and the local proxy does not
  expose those paths.
* **SSH.** `init` sets the private key to mode 0600 after generation.
  `StrictHostKeyChecking=no` plus `UserKnownHostsFile=/dev/null` is deliberate:
  the channel is an ephemeral `colab ssh --proxy-mode` tunnel with a new host
  key per VM, and the Colab CLI exposes no known_hosts option. Verifying host
  keys would require a CLI change; re-evaluate if that option appears.
* **OpenCode V1/V2.** The generated config stays in V1 form (`plugin`,
  `provider`, `npm`, `options`, `tool_call`), which the V2 migration guide says
  is normalized in memory. The discovery plugin is pinned to
  `opencode-models-discovery@2.2.1` (npm `latest` on 2026-11-23) instead of
  `@latest`. The plugin's own README describes a separate V1 adapter, but
  V1 plugin execution under V2 was **not verified** in this environment.
  Revalidate with the installed OpenCode version before relying on automatic
  model discovery.
* **Colab CLI divergence.** The upstream README documents one-shot commands as
  `colab ssh -s NAME -- CMD`; `src/byom/remote.py` states that CLI 0.7.5 does
  not accept that form and uses a multiplexed `colab ssh --proxy-mode` master
  instead. The command was **not changed** and the real CLI version/behavior
  was **not verified** here. Revalidate against `colab version` and the
  matching upstream README before porting this command.

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
  cli.py                  init, profiles, gpus, models, configure, up, status, down
  catalog.py              validated GPU/model catalog and compatibility check
  profiles/               runtime profiles (vLLM, GGUF, image) and OpenCode aliases
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
