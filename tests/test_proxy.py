"""Tests for the local proxy, with the tunnel faked by an in-memory transport."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from byom import config
from byom.proxy import Runtime, WorkerStatus, create_app
from byom.tunnel import TunnelState, TunnelSupervisor

NONCE = "session-nonce"


def make_state() -> config.State:
    return config.State(session="byom-test", endpoint="/tun/m/x", nonce=NONCE, created_at="t")


def make_runtime(*, verified: bool = True, model_ready: bool = True) -> Runtime:
    cfg = config.Config()
    tunnel = TunnelSupervisor(session="byom-test", endpoint="/tun/m/x", cfg=cfg)
    tunnel.status = TunnelState(state="connected", reconnects=0, latency_ms=12.5)
    runtime = Runtime(state=make_state(), cfg=cfg, tunnel=tunnel)
    runtime.worker = WorkerStatus(
        verified=verified,
        model_ready=model_ready,
        idle_s=42.0,
        idle_timeout_min=20.0,
        active_jobs=0,
        watchdog_silence_s=3.0,
        error=None if verified else "Gateway nonce does not match the registered session",
    )
    return runtime


def sse_transport(chunks: list[bytes], *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "rejected by the gateway"}})

        async def body():
            for chunk in chunks:
                yield chunk

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


async def _idle_verify_loop(stop: asyncio.Event) -> None:
    """Replace the verification loop: the test sets state directly."""
    await stop.wait()


def client_for(runtime: Runtime, transport: httpx.MockTransport) -> TestClient:
    # A real `verify_loop` would make a real request to the tunnel, which does
    # not exist in tests. The worker state is set in `make_runtime`.
    runtime.verify_loop = _idle_verify_loop  # type: ignore[method-assign]
    app = create_app(runtime, transport=transport)
    return TestClient(app)


def test_local_status_reports_tunnel_and_worker():
    runtime = make_runtime()
    with client_for(runtime, sse_transport([])) as client:
        payload = client.get("/__byom/status").json()
    assert payload["session"] == "byom-test"
    assert payload["tunnel"]["state"] == "connected"
    assert payload["tunnel"]["latency_ms"] == 12.5
    assert payload["worker"]["state"] == "ready"


def test_forwards_sse_event_by_event():
    runtime = make_runtime()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    with (
        client_for(runtime, sse_transport(chunks)) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        ) as response,
    ):
        assert response.status_code == 200
        received = [chunk for chunk in response.iter_bytes() if chunk]

    # Events arrive in the order sent and without buffering.
    assert b"".join(received) == b"".join(chunks)


def test_503_when_nonce_mismatches():
    runtime = make_runtime(verified=False)
    with client_for(runtime, sse_transport([])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 503
    assert "nonce" in response.json()["error"]["message"]
    assert response.json()["error"]["type"] == "byom_unavailable"


def test_503_when_worker_not_verified():
    runtime = make_runtime()
    runtime.worker.verified = False
    runtime.worker.error = None
    with client_for(runtime, sse_transport([])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 503
    assert "identity" in response.json()["error"]["message"]


def test_gateway_error_is_forwarded():
    runtime = make_runtime()
    with client_for(runtime, sse_transport([], status=429)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 429
    assert response.json()["error"]["message"] == "rejected by the gateway"


def test_network_error_becomes_503():
    runtime = make_runtime()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("tunnel dropped")

    with client_for(runtime, httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "byom_tunnel"


async def test_verify_marks_unavailable_when_health_fails(monkeypatch):
    runtime = make_runtime()

    async def failing_health() -> None:
        return None

    monkeypatch.setattr(runtime.tunnel, "probe_health", failing_health)
    await runtime.verify()
    assert runtime.worker.verified is False
    assert runtime.worker.error is not None


async def test_verify_accepts_correct_nonce(monkeypatch):
    runtime = make_runtime(verified=False)

    async def ok_health():
        return {
            "nonce": NONCE,
            "model_ready": True,
            "idle_s": 1.0,
            "idle_timeout_min": 20,
            "active_jobs": 0,
            "watchdog_silence_s": 2.0,
            "shutting_down": False,
        }

    monkeypatch.setattr(runtime.tunnel, "probe_health", ok_health)
    await runtime.verify()
    assert runtime.worker.verified is True
    assert runtime.worker.model_ready is True
    assert runtime.refuse_reason() is None


async def test_verify_marks_stale_watchdog_unavailable(monkeypatch):
    runtime = make_runtime()

    async def stale_health():
        return {
            "nonce": NONCE,
            "model_ready": True,
            "watchdog_silence_s": 61.0,
            "vllm_down": False,
        }

    monkeypatch.setattr(runtime.tunnel, "probe_health", stale_health)
    await runtime.verify()

    assert runtime.worker.verified is True
    assert runtime.status_payload()["worker"]["state"] == "unavailable"
    assert "No watchdog heartbeat" in (runtime.refuse_reason() or "")


async def test_verify_does_not_report_ready_with_vllm_down(monkeypatch):
    runtime = make_runtime()

    async def failed_health():
        return {
            "nonce": NONCE,
            "model_ready": True,
            "watchdog_silence_s": 1.0,
            "vllm_down": True,
            "shutdown_reason": "silent watchdog",
        }

    monkeypatch.setattr(runtime.tunnel, "probe_health", failed_health)
    await runtime.verify()

    assert runtime.status_payload()["worker"]["state"] == "unavailable"
    assert runtime.worker.model_ready is False
    assert "vLLM unavailable" in (runtime.refuse_reason() or "")


@pytest.mark.parametrize("nonce", ["", "outro-nonce"])
async def test_verify_rejects_mismatched_nonce(monkeypatch, nonce):
    runtime = make_runtime()

    async def bad_health():
        return {"nonce": nonce, "model_ready": True}

    monkeypatch.setattr(runtime.tunnel, "probe_health", bad_health)
    await runtime.verify()
    assert runtime.worker.verified is False
    assert runtime.refuse_reason() is not None


def test_tunnel_failure_shows_ssh_output(tmp_path):
    """An authentication failure must surface, not just "port did not open"."""
    log = tmp_path / "tunnel.log"
    log.write_text(
        "Warning: Permanently added 'x' to the list of known hosts.\n"
        "root@x: Permission denied (publickey).\n",
        encoding="utf-8",
    )
    cfg = config.Config()
    tunnel = TunnelSupervisor(session="byom-abc", endpoint="e", cfg=cfg, log_path=log)

    assert "Permission denied (publickey)" in tunnel.remote.log_tail()


def test_missing_log_does_not_break(tmp_path):
    cfg = config.Config()
    tunnel = TunnelSupervisor(
        session="byom-abc", endpoint="e", cfg=cfg, log_path=tmp_path / "missing.log"
    )
    assert tunnel.remote.log_tail() == ""


def test_tunnel_command_targets_the_session(monkeypatch):
    monkeypatch.setenv("BYOM_COLAB_BIN", "/usr/local/bin/colab")
    cfg = config.Config()
    tunnel = TunnelSupervisor(session="byom-abc", endpoint="/tun/m/x", cfg=cfg)
    command = tunnel.remote.master_command(tunnel.forwards)

    proxy_command = command[command.index("-o") + 1]
    assert proxy_command.startswith("ProxyCommand=/usr/local/bin/colab")
    assert "-s byom-abc" in proxy_command
    assert "--proxy-mode" in proxy_command
    assert "colab-sessions.json" in proxy_command
    # The identity must also go to the CLI: it authorizes its public key on
    # the server, and the outer `ssh` offers the key passed with `-i`.
    assert "-i" in proxy_command
    assert str(cfg.ssh_key_path) in proxy_command

    # The master is the only `colab ssh` connection: the forward lives on it.
    assert "-M" in command
    forward = command[command.index("-L") + 1]
    assert forward == "127.0.0.1:4001:127.0.0.1:8001"
    assert command[-1] == "byom-abc"


async def test_tunnel_health_feeds_status(monkeypatch):
    runtime = make_runtime()

    async def ok_health():
        return {
            "nonce": NONCE,
            "model_ready": False,
            "idle_s": 5.0,
            "idle_timeout_min": 20,
            "active_jobs": 1,
            "watchdog_silence_s": 1.0,
            "shutting_down": False,
        }

    monkeypatch.setattr(runtime.tunnel, "probe_health", ok_health)
    await runtime.verify()
    assert runtime.worker.active_jobs == 1
    # The model is still loading, so the state is "initializing", not "unavailable".
    assert runtime.status_payload()["worker"]["state"] == "initializing"
    assert runtime.refuse_reason() is None
