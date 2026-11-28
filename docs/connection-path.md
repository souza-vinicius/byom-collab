---
type: Architecture
title: Connection Path: Local Proxy, SSH Tunnel and Control Sockets
description: How a request reaches the Colab gateway through the local proxy and a single OpenSSH ControlMaster, why the command master and the tunnel use separate control sockets, and how the watchdog fits in.
tags: [ssh, tunnel, proxy, controlmaster, colab-ssh, watchdog, nonce]
status: stable
generated: { by: hermes/hy4-flash-free, at: 2026-11-28T03:50:00Z }
sources:
  - id: tunnel
    resource: /src/byom/tunnel.py
    title: Tunnel supervisor, session liveness check and health probe
  - id: remote
    resource: /src/byom/remote.py
    title: Remote master, control sockets, host key pinning
  - id: daemon
    resource: /src/byom/daemon.py
    title: Local process with proxy, tunnel and watchdog supervisor
  - id: proxy
    resource: /src/byom/proxy.py
    title: Local OpenAI-compatible proxy and worker state
  - id: gateway
    resource: /worker/gateway.py
    title: Loopback gateway, nonce and internal endpoints
  - id: watchdog
    resource: /worker/watchdog.py
    title: Kernel cell that polls the gateway and deallocates the VM
---

# Connection Path: Local Proxy, SSH Tunnel and Control Sockets

The agent talks to `127.0.0.1:<proxy_port>` (4000 by default). The proxy process
forwards to the tunnel port, the tunnel forwards over SSH to the gateway on the
VM, and the gateway forwards to the runtime. Every hop after the proxy is
loopback only.

```text
agent -> proxy :4000 -> tunnel :4001 ==(colab ssh)==> gateway :8001 -> runtime :vllm_port
```

## One connection at a time

`colab ssh` accepts a single connection per runtime; a second one gets HTTP 429.
byom therefore keeps one OpenSSH `ControlMaster` and multiplexes everything
through it: remote commands, uploads and port forwarding. The `ProxyCommand` is
`colab ssh -s <session> -i <key> --proxy-mode`.

Opening a fresh `colab ssh` outside the master takes that slot and can kill the
master, which is why diagnostics reuse the existing socket (see
[Troubleshooting](./troubleshooting.md)).

## Two masters, two sockets

The command master used by `byom up` and the daemon's tunnel are different
actors and now have different control sockets:

| Ator | Socket | Lifetime |
| :--- | :--- | :--- |
| `byom up` (commands, upload) | `cm-<session>.sock` | the `up` transaction |
| daemon tunnel (`-L tunnel:gateway`) | `cm-<session>-tunnel.sock` | until `byom down` or daemon exit |

They do not overlap: `up` stops its master before the daemon starts, so the
single `colab ssh` slot is never disputed. Sharing one socket was a bug: the
final `ssh -O exit` of `up` killed the master the tunnel had just opened, and the
API stayed unavailable for a few seconds right after `Worker ready`.

`byom down` stops the daemon by the PID and start ticks recorded in
`state.json`; the daemon's shutdown closes the tunnel.

## What the tunnel checks

Before connecting, `TunnelSupervisor.session_allocated` asks the backend
(`colab sessions`) whether the session still exists. This matters because
`colab ssh --proxy-mode` creates a session when the name is not in the local
store; connecting blindly could allocate a new VM.

Once connected, the supervisor probes `GET /health` through the tunnel every
2 s, always with the session nonce (see below), and records `latency_ms`.
Fatal failures (local port in use, `cannot listen`, `permission denied`, host
key changed) end in `state = failed` with a readable reason instead of an
endless retry. Transient failures reconnect with backoff and give up after ten
consecutive cycles without a health response.

## Why loopback is not identity

The tunnel turns every client into a loopback address, so `127.0.0.1` proves
nothing about who is calling. The gateway requires `X-BYOM-Nonce` on
`/health`, `/internal/state` and `/internal/begin-shutdown`, comparing it in
constant time and never echoing it. Public inference routes do not need the
nonce; the proxy is their only entrance.

## Host keys

Each session pins its host key in `known_hosts_<session>` on the first
connection (`accept-new`) and requires the same key afterwards (`yes`). A
changed key aborts with an explicit message instead of falling back to `no`,
because the fallback would let a man-in-the-middle read prompts and the nonce.
`byom down` removes the file when deallocation is confirmed; an unconfirmed shutdown keeps the state for a retry.

## The watchdog

The watchdog is not an SSH client. It runs as a long kernel cell
(`colab exec -f watchdog.py`) that polls `/internal/state` every 10 s. When the
model is ready and the worker stays idle for `idle_timeout_min`, it asks the
gateway to begin shutdown and calls `runtime.unassign()`. The gateway, in turn,
stops the runtime if the watchdog heartbeat is silent for
`watchdog_timeout_s`; the local daemon relaunches the cell when its connection
drops.

## What `byom status` shows

- `Worker`: the state the proxy derived from the gateway (`ready`,
  `initializing`, `unavailable`).
- `Tunnel`: `connected`, `reconnecting`, `failed` or `stopped`, with the
  reconnect count, the last latency and the last error.
- `Watchdog`: `running`, `gave_up` or `stopped`, with the consecutive failure
  count and the last error.

Related: [Worker Runtime](./worker-runtime.md) for what the gateway serves, and
[Troubleshooting](./troubleshooting.md) for the failures seen so far.
