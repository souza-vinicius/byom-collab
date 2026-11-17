"""Tests for the worker gateway, with an in-memory fake vLLM."""

from __future__ import annotations

import asyncio
import contextlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("gateway_module")


class FakeVLLM:
    """Fake vLLM: answers /v1/models and streams /v1/chat/completions."""

    def __init__(self, *, events: int = 3, delay: float = 0.0, status: int = 200) -> None:
        self.events = events
        self.delay = delay
        self.status = status
        self.requests: list[dict] = []
        self.closed = asyncio.Event()

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3.8-27b"}]})

        payload = json.loads(request.content or b"{}")
        self.requests.append(payload)

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

    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["model_ready"]:
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

    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["model_ready"]:
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

    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["model_ready"]:
                break
        before = gateway.idle_s
        for _ in range(5):
            client.get("/health")
            client.get("/v1/models")
        assert gateway.idle_s >= before
        assert gateway.active_jobs == 0


def test_begin_shutdown_refuses_with_active_jobs(gateway_module, shutdowns):
    fake = FakeVLLM(events=5, delay=0.05)
    app, gateway = build_app(gateway_module, fake, shutdowns, idle_timeout_min=0.0)

    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["model_ready"]:
                break
        # Open a job and keep it open while requesting shutdown.
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        ) as stream:
            assert stream.status_code == 200
            body = client.post("/internal/begin-shutdown").json()
            assert body["accepted"] is False
            assert body["reason"] == "active jobs"
            for _ in stream.iter_bytes():
                pass

        # With no jobs, shutdown is accepted.
        body = client.post("/internal/begin-shutdown").json()
        assert body["accepted"] is True


def test_shutdown_refuses_inference_accepted_before(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns, idle_timeout_min=0.0)

    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["model_ready"]:
                break
        assert client.post("/internal/begin-shutdown").json()["accepted"] is True
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "oi"}]},
        )
        assert response.status_code == 503
        assert "shutting down" in response.json()["error"]["message"]


def test_insufficient_idle_time_does_not_shutdown(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, _ = build_app(gateway_module, fake, shutdowns, idle_timeout_min=60.0)

    with TestClient(app) as client:
        for _ in range(100):
            if client.get("/health").json()["model_ready"]:
                break
        body = client.post("/internal/begin-shutdown").json()
        assert body["accepted"] is False
        assert "idle time" in body["reason"]


def test_state_endpoint_registers_heartbeat(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns)

    with TestClient(app) as client:
        assert gateway._watchdog_seen is False
        payload = client.get("/internal/state").json()
        assert gateway._watchdog_seen is True
        assert payload["watchdog_silence_s"] < 1.0


async def test_silent_watchdog_stops_vllm(gateway_module, shutdowns):
    fake = FakeVLLM()
    app, gateway = build_app(gateway_module, fake, shutdowns, watchdog_timeout_s=0.0)

    async with (
        with_loops(gateway),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
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
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
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
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
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
