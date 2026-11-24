"""Tests for the worker gateway, with an in-memory fake vLLM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types

import httpx
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("gateway_module")

#: O gateway exige o nonce da sessão nos endpoints internos: todo cliente do
#: túnel chega como loopback, então o header é a única prova de identidade.
NONCE_HEADERS = {"X-BYOM-Nonce": "test-nonce"}


class FakeVLLM:
    """Fake vLLM: answers /v1/models and streams /v1/chat/completions."""

    def __init__(
        self,
        *,
        events: int = 3,
        delay: float = 0.0,
        status: int = 200,
        image_response: dict | None = None,
    ) -> None:
        self.events = events
        self.delay = delay
        self.status = status
        self.image_response = image_response
        self.requests: list[dict] = []
        self.closed = asyncio.Event()

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3.8-27b"}]})

        payload = json.loads(request.content or b"{}")
        self.requests.append(payload)
        if self.image_response is not None:
            return httpx.Response(200, json=self.image_response)

        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "vLLM failure"}})

        async def body():
            try:
                for index in range(self.events):
                    if self.delay:
                        await asyncio.sleep(self.delay)
                    chunk = {"choices": [{"delta": {"content": f"t{index}"}}]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                self.closed.set()

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def make_config(gateway_module, **overrides):
    base = {
        "nonce": "test-nonce",
        "idle_timeout_min": 20.0,
        "max_job_min": 40.0,
        "watchdog_timeout_s": 60.0,
        "reasoning_effort": "medium",
    }
    base.update(overrides)
    return gateway_module.GatewayConfig(**base)


@pytest.fixture
def shutdowns() -> list[str]:
    return []


def build_app(gateway_module, fake: FakeVLLM, shutdowns: list[str], **cfg_overrides):
    async def shutdown() -> None:
        shutdowns.append("called")

    cfg = make_config(gateway_module, **cfg_overrides)
    app = gateway_module.create_app(cfg, transport=fake.transport(), shutdown_vllm=shutdown)
    return app, app.state.gateway


async def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


@contextlib.asynccontextmanager
async def with_loops(gateway):
    """Start the gateway's background loops.

    httpx's `ASGITransport` does not run the app lifespan, so the readiness,
    reaper, and supervision loops are created here.
    """
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(gateway.readiness_loop(stop)),
        asyncio.create_task(gateway.reaper_loop(stop)),
        asyncio.create_task(gateway.watchdog_guard_loop(stop)),
    ]
    try:
        yield
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await gateway.aclose()


def test_injects_reasoning_effort_when_missing(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, _ = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "oi"}]},
        )

    assert response.status_code == 200
    assert fake.requests[0]["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_respects_client_reasoning_effort(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, _ = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "oi"}],
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
        )

    assert fake.requests[0]["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_health_does_not_count_as_activity(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        before = gateway.idle_s
        for _ in range(5):
            client.get("/health", headers=NONCE_HEADERS)
            client.get("/v1/models")
        assert gateway.idle_s >= before
        assert gateway.active_jobs == 0


def test_begin_shutdown_refuses_with_active_jobs(gateway_module, shutdowns):
    fake = FakeVLLM(events=5, delay=0.05)
    app, gateway = build_app(gateway_module, fake, shutdowns, idle_timeout_min=0.0)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        # Open a job and keep it open while requesting shutdown.
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        ) as stream:
            assert stream.status_code == 200
            body = client.post("/internal/begin-shutdown", headers=NONCE_HEADERS).json()
            assert body["accepted"] is False
            assert body["reason"] == "active jobs"
            for _ in stream.iter_bytes():
                pass

        # With no jobs, shutdown is accepted.
        body = client.post("/internal/begin-shutdown", headers=NONCE_HEADERS).json()
        assert body["accepted"] is True


def test_shutdown_refuses_inference_accepted_before(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns, idle_timeout_min=0.0)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        assert (
            client.post("/internal/begin-shutdown", headers=NONCE_HEADERS).json()["accepted"]
            is True
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert response.status_code == 503
        assert "shutting down" in response.json()["error"]["message"]


def test_insufficient_idle_time_does_not_shutdown(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, _ = build_app(gateway_module, fake, shutdowns, idle_timeout_min=60.0)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        body = client.post("/internal/begin-shutdown", headers=NONCE_HEADERS).json()
        assert body["accepted"] is False
        assert "idle time" in body["reason"]


def test_state_endpoint_registers_heartbeat(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert gateway._watchdog_seen is False
        payload = client.get("/internal/state", headers=NONCE_HEADERS).json()
        assert gateway._watchdog_seen is True
        assert payload["watchdog_silence_s"] < 1.0


async def test_silent_watchdog_stops_vllm(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns, watchdog_timeout_s=0.0)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://gateway",
        ) as client,
    ):
        # Let the model become ready and never register a heartbeat.
        assert await wait_for(lambda: gateway.model_ready)
        assert await wait_for(lambda: gateway._vllm_down)
        assert shutdowns == ["called"]

        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert response.status_code == 503


async def test_client_disconnect_ends_the_job(gateway_module, shutdowns):
    fake = FakeVLLM(events=50, delay=0.05)
    app, gateway = build_app(gateway_module, fake, shutdowns)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://gateway",
        ) as client,
    ):
        assert await wait_for(lambda: gateway.model_ready)

        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
            )
        )
        assert await wait_for(lambda: gateway.active_jobs == 1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # The job leaves the count and the vLLM stream is closed.
        assert await wait_for(lambda: gateway.active_jobs == 0)
        assert await wait_for(lambda: fake.closed.is_set())


async def test_stuck_job_is_cancelled(gateway_module, shutdowns, monkeypatch):
    fake = FakeVLLM(events=1000, delay=0.01)
    app, gateway = build_app(gateway_module, fake, shutdowns, max_job_min=0.0)
    monkeypatch.setattr(gateway_module, "REAPER_INTERVAL_S", 0.05)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://gateway",
        ) as client,
    ):
        assert await wait_for(lambda: gateway.model_ready)
        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
            )
        )
        assert await wait_for(lambda: gateway.active_jobs == 1)
        # The reaper cancels the job and it leaves the count.
        assert await wait_for(lambda: gateway.active_jobs == 0, timeout=5.0)
        with pytest.raises((asyncio.CancelledError, httpx.HTTPError)):
            await task


def install_fake_google_colab(monkeypatch, calls: list[str]) -> None:
    """Fake `google.colab.runtime` in `sys.modules`, without importing Colab."""
    google = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")
    colab_module.runtime = types.SimpleNamespace(unassign=lambda: calls.append("unassign"))
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)


async def test_fallback_unassigns_when_watchdog_does_not(gateway_module, shutdowns, monkeypatch):
    monkeypatch.setattr(gateway_module, "UNASSIGN_FALLBACK_DELAY_S", 0.01)
    monkeypatch.setattr(gateway_module, "UNASSIGN_FALLBACK_CHECK_S", 0.01)
    calls: list[str] = []
    install_fake_google_colab(monkeypatch, calls)
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns, idle_timeout_min=0.0)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://gateway",
        ) as client,
    ):
        assert await wait_for(lambda: gateway.model_ready)
        response = await client.post("/internal/begin-shutdown", headers=NONCE_HEADERS)
        assert response.json()["accepted"] is True
        assert await wait_for(lambda: len(calls) == 1, timeout=2.0)


async def test_fallback_waits_for_active_jobs(gateway_module, shutdowns, monkeypatch):
    monkeypatch.setattr(gateway_module, "UNASSIGN_FALLBACK_DELAY_S", 0.01)
    monkeypatch.setattr(gateway_module, "UNASSIGN_FALLBACK_CHECK_S", 0.01)
    calls: list[str] = []
    install_fake_google_colab(monkeypatch, calls)
    fake = FakeVLLM(events=50, delay=0.05)
    app, gateway = build_app(gateway_module, fake, shutdowns)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://gateway",
        ) as client,
    ):
        assert await wait_for(lambda: gateway.model_ready)
        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
            )
        )
        assert await wait_for(lambda: gateway.active_jobs == 1)
        await gateway.mark_failed("watchdog lost")
        await asyncio.sleep(0.2)
        assert calls == []
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert await wait_for(lambda: gateway.active_jobs == 0)
        assert await wait_for(lambda: len(calls) == 1, timeout=2.0)


def test_internal_endpoints_reject_non_loopback(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("10.1.2.3", 50000)) as client:
        assert client.get("/internal/state").status_code == 403
        assert client.post("/internal/begin-shutdown").status_code == 403

    assert gateway._watchdog_seen is False
    assert gateway.shutting_down is False


async def test_job_limit_returns_503(gateway_module, shutdowns, monkeypatch):
    monkeypatch.setenv(gateway_module.MAX_CONCURRENT_JOBS_ENV, "1")
    fake = FakeVLLM(events=50, delay=0.05)
    app, gateway = build_app(gateway_module, fake, shutdowns)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://gateway",
        ) as client,
    ):
        assert await wait_for(lambda: gateway.model_ready)
        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
            )
        )
        assert await wait_for(lambda: gateway.active_jobs == 1)
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert response.status_code == 503
        assert "Too many active jobs" in response.json()["error"]["message"]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_gateway_config_load_validates_boundaries(gateway_module, tmp_path):
    path = tmp_path / "session.json"
    for payload in (
        {"nonce": "n", "idle_timeout_min": "20"},
        {"nonce": "n", "gateway_port": 70000},
        {"nonce": "n", "reasoning_effort": "bogus"},
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            gateway_module.GatewayConfig.load(path)


def test_invalid_client_kwargs_are_preserved(gateway_module):
    body = b'{"chat_template_kwargs": "invalido"}'
    assert gateway_module._inject_reasoning_effort(body, "medium") == body


def gemma_policy():
    """Mesmo mapa enviado por `byom up` a partir de catalog.ThinkingPolicy."""
    return {
        "system_token": "<|think|>",
        "thought_open": "<|channel>thought",
        "thought_close": "<channel|>",
        "strip_history_thought": True,
        "source": "https://unsloth.ai/docs/models/gemma-4.md (2026-11-23)",
    }


def post_chat(app, messages, **extra):
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        return client.post(
            "/v1/chat/completions",
            json={"model": "gemma-4", "messages": messages, **extra},
        )


def test_none_reasoning_is_not_injected(gateway_module):
    fake = FakeVLLM()
    app, _ = build_app(
        gateway_module, fake, [], reasoning_effort=None, runtime="llama-cpp"
    )
    response = post_chat(app, [{"role": "user", "content": "oi"}])
    assert response.status_code == 200
    assert "chat_template_kwargs" not in fake.requests[0]


def test_all_documented_reasoning_values_are_valid(gateway_module, tmp_path):
    path = tmp_path / "session.json"
    for value in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        payload = {"nonce": "n", "reasoning_effort": value}
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert gateway_module.GatewayConfig.load(path).reasoning_effort == value
    for value in ("turbo", 0.9):
        payload = {"nonce": "n", "reasoning_effort": value}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            gateway_module.GatewayConfig.load(path)


def test_thinking_adds_token_and_strips_prior_thought(gateway_module):
    fake = FakeVLLM()
    app, _ = build_app(
        gateway_module,
        fake,
        [],
        runtime="llama-cpp",
        reasoning_effort=None,
        thinking_policy=gemma_policy(),
    )
    messages = [
        {"role": "system", "content": "Voce e util."},
        {
            "role": "assistant",
            "content": "<|channel>thought\nsegredo interno\n<channel|>\nresposta final",
        },
        {"role": "user", "content": "oi"},
    ]
    assert post_chat(app, messages).status_code == 200
    sent = fake.requests[0]["messages"]
    assert sent[0]["content"] == "<|think|>Voce e util."
    assert "segredo interno" not in sent[1]["content"]
    assert sent[1]["content"] == "resposta final"


def test_thinking_creates_system_message_when_absent(gateway_module):
    fake = FakeVLLM()
    app, _ = build_app(
        gateway_module,
        fake,
        [],
        runtime="llama-cpp",
        reasoning_effort=None,
        thinking_policy=gemma_policy(),
    )
    assert post_chat(app, [{"role": "user", "content": "oi"}]).status_code == 200
    sent = fake.requests[0]["messages"]
    assert sent[0] == {"role": "system", "content": "<|think|>"}


def test_thinking_preserves_client_managed_prompt_and_history(gateway_module):
    fake = FakeVLLM()
    app, _ = build_app(
        gateway_module,
        fake,
        [],
        runtime="llama-cpp",
        reasoning_effort=None,
        thinking_policy=gemma_policy(),
    )
    messages = [
        {"role": "system", "content": "<|think|>ja gerenciado"},
        {"role": "assistant", "content": "resposta limpa"},
        {"role": "user", "content": "oi"},
    ]
    assert post_chat(app, messages).status_code == 200
    assert fake.requests[0]["messages"] == messages


def test_thinking_respects_client_disable(gateway_module):
    fake = FakeVLLM()
    app, _ = build_app(
        gateway_module,
        fake,
        [],
        runtime="llama-cpp",
        reasoning_effort=None,
        thinking_policy=gemma_policy(),
    )
    response = post_chat(
        app,
        [{"role": "user", "content": "oi"}],
        chat_template_kwargs={"enable_thinking": False},
    )
    assert response.status_code == 200
    sent = fake.requests[0]
    assert sent["messages"] == [{"role": "user", "content": "oi"}]
    assert sent["chat_template_kwargs"]["enable_thinking"] is False


def test_no_thinking_policy_leaves_payload_untouched(gateway_module):
    fake = FakeVLLM()
    app, _ = build_app(
        gateway_module, fake, [], runtime="llama-cpp", reasoning_effort=None
    )
    messages = [
        {"role": "assistant", "content": "<|channel>thought x<channel|>final"},
        {"role": "user", "content": "oi"},
    ]
    assert post_chat(app, messages).status_code == 200
    assert fake.requests[0]["messages"] == messages


def test_image_defaults_reach_backend(gateway_module):
    fake = FakeVLLM(image_response={"created": 1, "data": [{"b64_json": "QUJD"}]})
    app, _ = build_app(
        gateway_module,
        fake,
        [],
        runtime="image",
        reasoning_effort=None,
        generation_defaults={"steps": 25, "guidance": 1.0},
    )
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "um cachorro", "size": "1024x1024"},
        )
    assert response.status_code == 200
    sent = fake.requests[0]
    # Nomes nativos do sd-server, não os nomes da API da Unsloth.
    assert "sample_params" not in sent
    assert "guidance" not in sent
    assert "steps" not in sent
    assert sent["prompt"].startswith("um cachorro <sd_cpp_extra_args>")
    assert '"sample_steps":25' in sent["prompt"]
    assert '"guidance":{"txt_cfg":1.0}' in sent["prompt"]


def test_image_client_values_override_defaults(gateway_module):
    fake = FakeVLLM(image_response={"created": 1, "data": [{"b64_json": "QUJD"}]})
    app, _ = build_app(
        gateway_module,
        fake,
        [],
        runtime="image",
        reasoning_effort=None,
        generation_defaults={"steps": 25, "guidance": 1.0},
    )
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for _ in range(100):
            if client.get("/health", headers=NONCE_HEADERS).json()["model_ready"]:
                break
        client.post(
            "/v1/images/generations",
            json={"prompt": "gato", "steps": 8, "guidance": 3.5},
        )
        already = (
            "gato <sd_cpp_extra_args>"
            '{"sample_params":{"sample_steps":4}}</sd_cpp_extra_args>'
        )
        client.post("/v1/images/generations", json={"prompt": already})
    assert '"sample_steps":8' in fake.requests[0]["prompt"]
    assert '"guidance":{"txt_cfg":3.5}' in fake.requests[0]["prompt"]
    # Segunda requisição: o cliente já injetou o bloco; gateway não acrescenta outro.
    assert fake.requests[1]["prompt"].count("<sd_cpp_extra_args>") == 1
    assert '"sample_steps":4' in fake.requests[1]["prompt"]


def test_gateway_config_validates_thinking_and_generation(tmp_path, gateway_module):
    path = tmp_path / "session.json"
    valid = {
        "nonce": "n",
        "thinking_policy": gemma_policy(),
        "generation_defaults": {"steps": 25, "guidance": 1.0},
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    cfg = gateway_module.GatewayConfig.load(path)
    assert cfg.thinking_policy["system_token"] == "<|think|>"
    assert cfg.generation_defaults["steps"] == 25

    for bad in (
        {"thinking_policy": {"system_token": 123}},
        {"thinking_policy": {"campo_extra": "x"}},
        {"generation_defaults": {"steps": 25}},
        {"generation_defaults": {"steps": "25", "guidance": 1}},
        {"generation_defaults": {"steps": 25, "guidance": "1"}},
    ):
        payload = {"nonce": "n", **bad}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            gateway_module.GatewayConfig.load(path)


# ---------------------------------------------------------------------------
# Nonce obrigatório nos endpoints internos
# ---------------------------------------------------------------------------
def test_endpoints_internos_exigem_nonce(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        # Sem header: 403 nos três endpoints, e o nonce nunca aparece no corpo.
        for response in (
            client.get("/health"),
            client.get("/internal/state"),
            client.post("/internal/begin-shutdown"),
        ):
            assert response.status_code == 403
            assert "test-nonce" not in response.text
            assert gateway_module.NONCE_HEADER not in response.text

        # Header divergente também é recusado.
        wrong = {"X-BYOM-Nonce": "outra-sessao"}
        assert client.get("/health", headers=wrong).status_code == 403
        assert client.get("/internal/state", headers=wrong).status_code == 403

        # Com o nonce da sessão, os três respondem.
        assert client.get("/health", headers=NONCE_HEADERS).status_code == 200
        assert client.get("/internal/state", headers=NONCE_HEADERS).status_code == 200
        assert (
            client.post("/internal/begin-shutdown", headers=NONCE_HEADERS).status_code == 200
        )

    # O 403 não pode registrar heartbeat nem aceitar shutdown.
    assert gateway._watchdog_seen is True  # a chamada válida acima registrou


def test_nonce_nao_aparece_em_nenhum_payload_de_status(gateway_module, shutdowns):
    """O nonce vai só na requisição: nenhuma resposta de status o devolve."""
    fake = FakeVLLM()
    app, _ = build_app(gateway_module, fake, shutdowns)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        for path in ("/health", "/internal/state"):
            body = client.get(path, headers=NONCE_HEADERS).text
            assert "nonce" not in body
            assert "test-nonce" not in body
