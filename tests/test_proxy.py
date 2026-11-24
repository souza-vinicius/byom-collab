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
PROXY_BASE_URL = "http://127.0.0.1:4000"


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
    # O proxy só aceita `Host` de loopback (DNS rebinding); o endereço do
    # cliente precisa ser o mesmo que o daemon usa de verdade.
    return TestClient(app, base_url=PROXY_BASE_URL)


def recording_transport(seen: list[httpx.Request], *, status: int = 200) -> httpx.MockTransport:
    """Upstream que registra a requisição recebida e responde o mínimo."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "rejected"}})

        async def body():
            yield b'data: {"ok": true}\n\n'

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


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


# ---------------------------------------------------------------------------
# Superfície local: caminho, Host e headers repassados
# ---------------------------------------------------------------------------
#: Variantes que chegam ao roteador como `..`: o cliente HTTP entrega o
#: caminho decodificado, e o Starlette casa `{path:path}` com ele.
TRAVERSAL_PATHS = (
    "/v1/%2e%2e/internal/state",
    "/v1/..%2Finternal%2Fstate",
    "/v1/%2e/internal/state",
)


@pytest.mark.parametrize("path", TRAVERSAL_PATHS)
def test_traversal_recusado_antes_de_montar_a_url(path):
    seen: list[httpx.Request] = []
    runtime = make_runtime()

    with client_for(runtime, recording_transport(seen)) as client:
        response = client.get(path)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "byom_bad_path"
    assert seen == [], "nenhuma chamada pode sair para o upstream"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/%2e%2e/internal/begin-shutdown",
        "/v1/a%5c..%5cinternal/state",
        "/v1/models%00",
        "/v1/%252e%252e/internal/state",
    ],
)
def test_post_e_encoding_recusados_sem_upstream(path):
    seen: list[httpx.Request] = []
    runtime = make_runtime()

    with client_for(runtime, recording_transport(seen)) as client:
        response = client.post(path, content=b"{}")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "byom_bad_path"
    assert seen == []


def test_query_string_chega_ao_upstream():
    seen: list[httpx.Request] = []
    runtime = make_runtime()

    with client_for(runtime, recording_transport(seen)) as client:
        response = client.get("/v1/models?x=1")

    assert response.status_code == 200
    assert len(seen) == 1
    assert seen[0].url.path == "/v1/models"
    assert seen[0].url.query == b"x=1"


@pytest.mark.parametrize("host", ["evil.com", "evil.com:4000", "127.0.0.1.evil.com", "[::1]:4000"])
def test_host_nao_loopback_e_recusado(host):
    seen: list[httpx.Request] = []
    runtime = make_runtime()

    with client_for(runtime, recording_transport(seen)) as client:
        response = client.get("/v1/models", headers={"host": host})

    assert response.status_code == 403
    assert response.json()["error"]["type"] == "byom_bad_host"
    assert seen == []


def test_host_loopback_com_porta_e_aceito():
    seen: list[httpx.Request] = []
    runtime = make_runtime()

    with client_for(runtime, recording_transport(seen)) as client:
        assert client.get("/v1/models", headers={"host": "localhost:4000"}).status_code == 200

    assert len(seen) == 1


def test_headers_repassados_seguem_allowlist():
    seen: list[httpx.Request] = []
    runtime = make_runtime()

    with client_for(runtime, recording_transport(seen)) as client:
        client.get(
            "/v1/models",
            headers={
                "accept": "text/event-stream",
                "openai-beta": "assistants=v2",
                "user-agent": "opencode/1.0",
                "authorization": "Bearer sk-cliente",
                "accept-encoding": "gzip, br",
            },
        )

    headers = seen[0].headers
    assert headers["accept"] == "text/event-stream"
    assert headers["openai-beta"] == "assistants=v2"
    assert headers["user-agent"] == "opencode/1.0"
    # Comprimir o stream mudaria o corpo SSE e o repasse byte a byte.
    assert headers["accept-encoding"] == "identity"
    assert "authorization" not in headers


def _raw_get(app, path: str, *, host: bytes = b"127.0.0.1:4000") -> tuple[int, bytes]:
    """Chame o app ASGI com o `path` exatamente como o uvicorn o entrega.

    O cliente HTTP normaliza `..` antes de enviar; o uvicorn não. Esta sonda
    reproduz o pedido que realmente chega ao roteador.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", host)],
        "server": ("127.0.0.1", 4000),
        "client": ("127.0.0.1", 54321),
    }
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    status = messages[0]["status"]
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, body


def test_traversal_literal_no_nivel_asgi():
    """`/v1/../internal/state` cru, como o uvicorn repassa ao roteador."""
    seen: list[httpx.Request] = []
    runtime = make_runtime()
    runtime.verify_loop = _idle_verify_loop  # type: ignore[method-assign]
    app = create_app(runtime, transport=recording_transport(seen))

    for path in ("/v1/../internal/state", "/v1/../internal/begin-shutdown", "/v1/./models"):
        status, body = _raw_get(app, path)
        assert status == 400, path
        assert b"byom_bad_path" in body, path

    assert seen == []
