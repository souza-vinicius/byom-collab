---
type: Runbook
title: Troubleshooting the Worker Session
description: Symptom to cause to action for the failures seen while bringing a session up: silent runtime exit, missing CUDA libraries, unsupported PTX, a worker stuck initializing and a tunnel that drops right after up.
tags: [troubleshooting, runbook, bootstrap, cuda, gateway, tunnel]
status: stable
generated: { by: hermes/hy4-flash-free, at: 2026-11-28T03:50:00Z }
sources:
  - id: bootstrap
    resource: /worker/bootstrap.sh
    title: Startup diagnostics and the 10 s runtime liveness check
  - id: runtime-sh
    resource: /worker/run_runtime.sh
    title: CUDA 12 library path
  - id: gateway
    resource: /worker/gateway.py
    title: vllm_port derivation and the readiness poll
  - id: tunnel
    resource: /src/byom/tunnel.py
    title: Tunnel state and the separate control socket
  - id: cli
    resource: /src/byom/cli.py
    title: byom up wait and failure messages
---

# Troubleshooting the Worker Session

Start with `byom status`. It separates the three independent layers: the worker
state the proxy got from the gateway, the tunnel, and the watchdog.

## Reading the VM logs without taking the SSH slot

Do not run a fresh `colab ssh`: it takes the single connection slot and can kill
the tunnel master. Reuse the daemon socket instead.

```bash
CONFIG_DIR="${BYOM_HOME:-${XDG_CONFIG_HOME:-$HOME/.config}/byom}"
SESSION=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["session"])' "$CONFIG_DIR/state.json")
ssh -S "$CONFIG_DIR/cm-$SESSION-tunnel.sock" -o BatchMode=yes -l root "$SESSION" \
  'tmux ls; tail -n 40 /content/byom/logs/llama-cpp.log'
```

Replace `llama-cpp.log` with `image.log` or `vllm.log` according to the runtime,
and check `gateway.log` in the same directory.

## Symptom map

| Symptom | Cause | Action |
| :--- | :--- | :--- |
| `server exited unexpectedly`, no cause | The tmux session died with its only process and the bootstrap did not surface the log | On current code the bootstrap prints `--- tail of ...` first; read that block. If it is absent, the bundle predates the fix; rerun `byom up` with the current code. |
| `error while loading shared libraries: libcudart.so.12` | CUDA 12 libraries outside the loader path | The bootstrap adds them; confirm with `ldd /content/byom/llama-server` and check that no line says `not found`. If the image changed, extend the `nvidia/*/lib` lookup in `run_runtime.sh`. |
| `CUDA error: the provided PTX was compiled with an unsupported toolchain` | A CUDA 13 asset running on a GPU with no SASS for it (T4 sm_75, A100 sm_80) | Use the CUDA 12.8 pins. The 13.x assets only cover sm_86/89/120/121, and the Colab driver rejects their PTX. |
| `byom up` stuck at `Worker initializing` for minutes | The gateway cannot reach the runtime on `vllm_port` | With current code the gateway derives the URL from `session.json`; if it repeats, compare the runtime log line `listening on http://127.0.0.1:<port>` with `vllm_port` in `session.json` and look for `readiness poll failed` in `gateway.log`. |
| `Tunnel unavailable` or `SSH connection dropped` right after `Worker ready` | A previous master was stopped while the tunnel was using it | Current code gives the tunnel its own socket, so `byom status` should show `connected` with `reconnects: 0`. A persistent `failed` needs `byom down` and a new `byom up`. |
| `Worker unavailable` and `Gateway is inaccessible through the tunnel` during `up` | Normal transient while the daemon opens the tunnel | Wait; `up` keeps polling. If it lasts, check the tunnel state in `byom status`. |
| `Colab CLI could not list sessions` with `Last CLI output` | Authentication failure, network error, or a CLI error unrelated to login | For login text, run `colab sessions`; otherwise fix the reported CLI error and run `byom init` again |
| `a porta 4000 do proxy já está em uso` or `a porta 4001 do túnel já está em uso` | A local process holds a port needed by the daemon | Stop that process or run `byom down` before allocating a GPU; `up` now checks both ports before allocation |
| `Worker did not become ready within ... (last state: ...)` | Model load failed, tunnel dropped, or the GPU quota ran out | Use the reported last state as the lead, run `byom status`, then read the VM logs with the command above |

## Known limitation: a crash after the bootstrap window

The bootstrap only proves the runtime lived for the first 10 s. If it dies
later, the gateway keeps polling `/v1/models` and `byom up` waits up to
`bootstrap_timeout_min` (30 min) with the worker in `initializing`. There is no
process watchdog for the runtime yet, and `byom status` shows the same
`initializing` whether the model is still loading or the process is gone: the
gateway only knows the HTTP result.

Workarounds:

- `byom down` and `byom up`: a crash during the first 10 s now prints the
  runtime log tail, which names the cause.
- While the session is up, read the logs with the command above and look for
  `CUDA error`, `failed to load model` or an out-of-memory line.

## Recovering a session

`byom down` stops the VM and restores the harness config, including when the
deallocation cannot be confirmed. It is idempotent: a session recorded in
`state.json` but already stopped is reported as such, and the state is cleared
only after confirmation. If the daemon is not responding, `byom status` reports
`Worker: unavailable (local daemon is not responding)` and `byom down` still
stops the allocation by the session name recorded in `state.json`.

Related: [Worker Runtime](./worker-runtime.md) and
[Connection Path](./connection-path.md).
