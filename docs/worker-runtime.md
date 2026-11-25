---
type: Architecture
title: Worker Runtime, Binaries and CUDA Libraries
description: How the Colab worker goes from an uploaded bundle to a serving runtime: bootstrap steps, pinned binaries, CUDA 12 library resolution, and the gateway that owns readiness on the session's vllm_port.
tags: [worker, bootstrap, cuda, llama-cpp, gateway, tmux, readiness]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-24T23:32:00Z }
sources:
  - id: bootstrap
    resource: /worker/bootstrap.sh
    title: Bundle extraction, venv, runtime start and startup diagnostics
  - id: prepare
    resource: /worker/prepare_runtime.py
    title: Pinned binary download, checksum and HF model files
  - id: runtime-sh
    resource: /worker/run_runtime.sh
    title: Runtime launcher, PID file and CUDA 12 library path
  - id: build-command
    resource: /worker/build_vllm_command.py
    title: Runtime argv from session.json and the manifest
  - id: gateway
    resource: /worker/gateway.py
    title: Gateway config, readiness poll and session vllm_port
  - id: cli-up
    resource: /src/byom/cli.py
    title: Bundle upload, bootstrap call and readiness wait
  - id: ggml-cuda-128
    resource: https://github.com/ggml-org/llama.cpp/releases/tag/b12649
    title: Upstream CUDA 12.8 and 13.5 assets compared in this concept
  - id: prism-cuda-133
    resource: https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b12128-d7b2896
    title: PrismML fork assets (CUDA 12.8 and 13.3)
---

# Worker Runtime, Binaries and CUDA Libraries

Every `byom up` sends a bundle of worker scripts to `/content/byom` and starts
the runtime inside a tmux session on the Colab VM. The model is served only
after the gateway, which owns readiness, reaches it on loopback.

## From upload to serving

1. `byom up` builds `worker.tar.gz` with the worker files and `session.json`
   and uploads it to `/content/byom-worker.tar.gz` (`cli._build_worker_tar`).
2. `bootstrap.sh` extracts the bundle to `/content/byom`, creates a Python 3.12
   venv with `uv`, installs the gateway dependencies from
   `requirements.lock` (pinned with hashes), and runs `prepare_runtime.py`.
3. `prepare_runtime.py` downloads the binary pinned in the active profile
   (`[worker] binary_url` plus `binary_sha256`) and the model files matched by
   the profile patterns, then writes `runtime_manifest.json`.
4. `build_vllm_command.py` turns `session.json` and the manifest into
   `runtime.cmd`, a single `exec` line for vLLM, llama-server or sd-server.
5. `bootstrap.sh` starts two tmux windows: the runtime (`run_runtime.sh` or
   `run_vllm.sh`) and the gateway (`run_gateway.sh`). It then verifies that the
   runtime PID written to `vllm.pid` is alive and prints the end of the runtime
   and gateway logs if it is not.
6. `byom up` replaces the command master with the local daemon and waits until
   the gateway reports `model_ready` (the `bootstrap_timeout_min` ceiling).

## Pinned binaries

Nothing uses `latest`. Each `llama-cpp` or `image` profile declares the release
asset, its SHA-256 and the directory to extract (`binary_dir`). The runtime is
exposed as a stable path by a symlink created in `prepare_runtime.find_binary`:
`/content/byom/llama-server` or `/content/byom/sd-server`.

## CUDA 12.8 pins and why not 13

The pins are the `cuda-12.8` assets of upstream `b12611`/`b12649` and of the
PrismML fork. That is a deliberate choice, not an old leftover. The Colab image
used on 2026-11-24 publishes CUDA 13.0 (`/usr/local/cuda`, driver 580.82.07),
and the CUDA 13 assets only carry SASS for a narrow set of architectures.

| Asset family | SASS (runs without JIT) | PTX |
| :--- | :--- | :--- |
| `cuda-12.8` (upstream and PrismML) | sm_86, sm_89, sm_120a | sm_75, sm_80, sm_90 |
| `cuda-13.3` / `cuda-13.5` | sm_86, sm_89, sm_103 (13.5), sm_120a, sm_121a | Not usable on the Colab driver, measured on T4 |

Consequences for the catalog GPUs:

- **L4** (sm_89) runs the 12.8 SASS and loads in seconds.
- **T4** (sm_75) and **A100** (sm_80) have no SASS in any of these assets; they
  compile the 12.8 PTX on first load. On the T4 measured here the model took
  2 min 36 s to load, inside the 30 min bootstrap ceiling.
- A CUDA 13 asset on the T4 fails before loading the model with `CUDA error:
  the provided PTX was compiled with an unsupported toolchain`, so re-pinning
  everything to 13 would trade the library error for an architecture error.

The architecture lists were read from the shipped `libggml-cuda.so` with
`cuobjdump --list-elf` and `--list-ptx` on the Colab VMs, not inferred from the
release notes.

## Where the libraries come from

The 12.8 build declares `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`
and `libcuda.so.1`. On the current image:

- `libcudart.so.13` and `libcublas.so.13` are in `ldconfig` through
  `/usr/local/cuda`.
- The CUDA 12 stack exists only inside the kernel Python packages, under
  `/usr/local/lib/python3*/dist-packages/nvidia/*/lib`
  (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-nvjitlink-cu12`).
- `libcuda.so.1` comes from the driver mount `/usr/lib64-nvidia`, which the
  VM's `sshd` already exports in `LD_LIBRARY_PATH`.

`run_runtime.sh` prepends the `nvidia/*/lib` directories to `LD_LIBRARY_PATH`
before executing `runtime.cmd`. Without that, the server dies in the loader with
`error while loading shared libraries: libcudart.so.12`, which is what used to
look like a generic `server exited unexpectedly`.

## The gateway

`gateway.py` listens on `127.0.0.1:<gateway_port>` and is the only path to the
runtime, which also listens on loopback only. It:

- forwards `POST /v1/chat/completions`, `GET /v1/models` and the image routes,
- polls `GET /v1/models` every 2 s to set `model_ready`,
- keeps the idle clock, cancels jobs past `max_job_min`, and stops the runtime
  when the watchdog heartbeat is lost,

The runtime URL is derived from `vllm_port`, the field `session.json` carries.
There is no fixed URL for the runtime port: a fixed 8000 made the worker stay in
`initializing` forever whenever the configured port was different. `vllm_port`
and `gateway_port` are validated as ports at the session boundary.

Internal endpoints (`/health`, `/internal/state`, `/internal/begin-shutdown`)
require the session nonce in `X-BYOM-Nonce`, because every client that arrives
through the tunnel appears as loopback. See
[Connection Path](./connection-path.md).

## Logs

On the VM, under `/content/byom/logs/`: `<runtime>.log` (`llama-cpp`, `image`
or `vllm`), `gateway.log`, and the tmux session named `byom`. Locally, under the
byom config directory (`~/.config/byom/logs/` by default): `daemon.log`,
`tunnel.log`, `watchdog.log` and `ssh-<session>.log`.

`bootstrap.sh` prints the tail of the runtime log when the process dies in the
first 10 s, which is when a missing library, a rejected argument or a missing
model file shows up. A crash after that window is not detected by the gateway,
which keeps polling until the bootstrap timeout; see
[Troubleshooting](./troubleshooting.md).
