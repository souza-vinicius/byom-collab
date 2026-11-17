#!/usr/bin/env python3
"""Worker gateway: the authoritative activity point for the session.

Responsibilities:
* Forward `/v1/chat/completions` and `/v1/models` to vLLM, with streaming.
* Inject `reasoning_effort` when the client does not send a value.
* Maintain the idle clock, which starts when the model becomes ready and
  restarts when the last job ends.
* Refuse inference while the watchdog is silent, and stop vLLM.
* Coordinate shutdown with the watchdog via `/internal/begin-shutdown`.

No inference bypasses this process: vLLM listens on loopback only. No prompt
content is logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULT_SESSION_FILE = "/content/byom/session.json"
READINESS_POLL_S = 2.0
REAPER_INTERVAL_S = 5.0
CANCEL_GRACE_S = 30.0

log = logging.getLogger("byom.gateway")


@dataclass
class GatewayConfig:
    """Configuration delivered by `byom` in `session.json`."""

    nonce: str
    model: str = ""
    served_model_name: str = "qwen3.8-27b"
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_pid_file: str = "/content/byom/vllm.pid"
    gateway_port: int = 8001
    reasoning_effort: str = "medium"
    idle_timeout_min: float = 20.0
    max_job_min: float = 40.0
    watchdog_timeout_s: float = 60.0

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_SESSION_FILE) -> GatewayConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Job:
    """An inference job accepted by the gateway."""

    id: str
    started: float
    task: asyncio.Task[Any] | None = None
    cancel_requested_at: float | None = None
    finished: bool = False


@dataclass
class Gateway:
    cfg: GatewayConfig
    transport: httpx.AsyncBaseTransport | None = None
    shutdown_vllm: Callable[[], Awaitable[None]] | None = None
    jobs: dict[str, Job] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _next_id: int = 0
    _shutting_down: bool = False
    _shutdown_reason: str | None = None
    _model_ready: bool = False
    _watchdog_seen: bool = False
    _last_watchdog_poll: float = field(default_factory=time.monotonic)
    _idle_since: float = field(default_factory=time.monotonic)
    _vllm_down: bool = False
    _client: httpx.AsyncClient | None = None

    # -- infrastructure ---------------------------------------------------
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.cfg.vllm_base_url,
                transport=self.transport,
                timeout=None,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- state ----------------------------------------------------------
    @property
    def model_ready(self) -> bool:
        return self._model_ready

    @property
    def idle_s(self) -> float:
        return time.monotonic() - self._idle_since

    @property
    def watchdog_silence_s(self) -> float:
        return time.monotonic() - self._last_watchdog_poll

    @property
    def watchdog_stale(self) -> bool:
        """The watchdog is only required once the model is ready."""
        return self._model_ready and self.watchdog_silence_s > self.cfg.watchdog_timeout_s

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def active_jobs(self) -> int:
        return sum(1 for job in self.jobs.values() if not job.finished)

    def register_watchdog_poll(self) -> None:
        self._last_watchdog_poll = time.monotonic()
        self._watchdog_seen = True

    def refuse_reason(self) -> str | None:
        """Reason to refuse a new inference, or None if it can be accepted."""
        if self._shutting_down:
            return "Worker is shutting down due to inactivity"
        if self._vllm_down:
            return "vLLM stopped after watchdog failure"
        if self.watchdog_stale:
            return "Watchdog heartbeat lost; worker is unsupervised"
        if not self._model_ready:
            return "Model is not ready yet"
        return None

    # -- job lifecycle ----------------------------------------------------
    async def begin_job(self) -> Job | None:
        async with self._lock:
            if self.refuse_reason() is not None:
                return None
            self._next_id += 1
            job = Job(id=f"job-{self._next_id}", started=time.monotonic())
            self.jobs[job.id] = job
            return job

    async def attach_task(self, job: Job, task: asyncio.Task[Any]) -> None:
        job.task = task

    async def end_job(self, job: Job) -> None:
        async with self._lock:
            job.finished = True
            self.jobs.pop(job.id, None)
            if not self.jobs:
                self._idle_since = time.monotonic()

    async def try_begin_shutdown(self) -> tuple[bool, str]:
        """Accept shutdown only with no active jobs and enough idle time."""
        async with self._lock:
            if self._shutting_down:
                return True, "already shutting down"
            if self.active_jobs:
                return False, "active jobs"
            needed = self.cfg.idle_timeout_min * 60
            if self.idle_s < needed:
                return False, f"insufficient idle time ({int(self.idle_s)}s of {int(needed)}s)"
            self._shutting_down = True
            self._shutdown_reason = "idle"
            return True, "idle"

    # -- readiness and supervision ----------------------------------------
    async def poll_readiness(self) -> bool:
        try:
            response = await self.client().get("/v1/models")
            ready = response.status_code == 200
        except Exception:
            ready = False
        if ready and not self._model_ready:
            self._model_ready = True
            self._idle_since = time.monotonic()
            log.info("Model ready")
        return ready

    async def readiness_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.poll_readiness()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=READINESS_POLL_S)

    async def reaper_loop(self, stop: asyncio.Event) -> None:
        """Cancel jobs that exceed `max_job_min`."""
        while not stop.is_set():
            await asyncio.sleep(REAPER_INTERVAL_S)
            limit = self.cfg.max_job_min * 60
            for job in list(self.jobs.values()):
                elapsed = time.monotonic() - job.started
                if job.cancel_requested_at is None and elapsed > limit:
                    log.warning("Job %s exceeded %.0fs; cancelling", job.id, limit)
                    job.cancel_requested_at = time.monotonic()
                    if job.task is not None:
                        job.task.cancel()
                elif (
                    job.cancel_requested_at is not None
                    and time.monotonic() - job.cancel_requested_at > CANCEL_GRACE_S
                ):
                    # If the job does not stop, treat it as a failure and stop
                    # vLLM to avoid leaving the VM consuming resources.
                    log.error("Job %s did not stop after cancellation", job.id)
                    await self.mark_failed("Stuck job did not stop after cancellation")

    async def watchdog_guard_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            if self.watchdog_stale:
                log.error("Watchdog heartbeat lost; stopping vLLM")
                await self.mark_failed("Watchdog heartbeat lost")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=READINESS_POLL_S)

    async def mark_failed(self, reason: str) -> None:
        if self._vllm_down:
            return
        self._vllm_down = True
        self._shutdown_reason = reason
        await self.shutdown_vllm_process()

    async def shutdown_vllm_process(self) -> None:
        if self.shutdown_vllm is not None:
            await self.shutdown_vllm()
            return
        await asyncio.to_thread(_signal_vllm, self.cfg.vllm_pid_file)

    # -- endpoints -------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        return {
            "nonce": self.cfg.nonce,
            "model_ready": self._model_ready,
            "idle_s": round(self.idle_s, 3),
            "idle_timeout_min": self.cfg.idle_timeout_min,
            "active_jobs": self.active_jobs,
            "watchdog_silence_s": round(self.watchdog_silence_s, 3),
            "watchdog_seen": self._watchdog_seen,
            "shutting_down": self._shutting_down,
            "shutdown_reason": self._shutdown_reason,
            "vllm_down": self._vllm_down,
        }


def _signal_vllm(pid_file: str) -> None:
    """Stop vLLM using the PID written by `run_vllm.sh`."""
    try:
        pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        log.error("Could not read vLLM PID from %s", pid_file)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:  # pragma: no cover - environment-dependent
        log.error("Failed to stop vLLM: %s", exc)


def _inject_reasoning_effort(body: bytes, effort: str) -> bytes:
    """Add `reasoning_effort` only when the client did not send a value."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
        payload["chat_template_kwargs"] = kwargs
    if "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = effort
        return json.dumps(payload).encode()
    return body


def create_app(
    cfg: GatewayConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    shutdown_vllm: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Build the gateway app. `transport` and `shutdown_vllm` are injectable."""
    gateway = Gateway(cfg, transport=transport, shutdown_vllm=shutdown_vllm)
    stop = asyncio.Event()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        tasks = [
            asyncio.create_task(gateway.readiness_loop(stop), name="readiness"),
            asyncio.create_task(gateway.reaper_loop(stop), name="reaper"),
            asyncio.create_task(gateway.watchdog_guard_loop(stop), name="watchdog-guard"),
        ]
        try:
            yield
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await gateway.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.gateway = gateway

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Does not count as activity and does not register a watchdog heartbeat."""
        payload = gateway.status_payload()
        payload.pop("watchdog_seen", None)
        return payload

    @app.get("/internal/state")
    async def internal_state() -> dict[str, Any]:
        """Called by the watchdog; each call refreshes the heartbeat."""
        gateway.register_watchdog_poll()
        return gateway.status_payload()

    @app.post("/internal/begin-shutdown")
    async def begin_shutdown() -> dict[str, Any]:
        accepted, reason = await gateway.try_begin_shutdown()
        return {"accepted": accepted, "reason": reason}

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        reason = gateway.refuse_reason()
        if reason is not None:
            return JSONResponse(status_code=503, content={"error": {"message": reason}})
        return await _forward_simple(gateway, request, "/v1/models")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        reason = gateway.refuse_reason()
        if reason is not None:
            return JSONResponse(status_code=503, content={"error": {"message": reason}})

        body = await request.body()
        body = _inject_reasoning_effort(body, cfg.reasoning_effort)

        job = await gateway.begin_job()
        if job is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": gateway.refuse_reason() or "Request rejected"}},
            )
        task = asyncio.current_task()
        if task is not None:
            await gateway.attach_task(job, task)
        return await _forward_stream(gateway, "/v1/chat/completions", body, job)

    return app


async def _forward_simple(gateway: Gateway, request: Request, path: str) -> Any:
    try:
        response = await gateway.client().request(
            request.method,
            path,
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=503, content={"error": {"message": str(exc)}})
    return JSONResponse(status_code=response.status_code, content=_safe_json(response))


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text[:2000]}}


async def _forward_stream(gateway: Gateway, path: str, body: bytes, job: Job) -> Any:
    """Forward the vLLM response event by event, without buffering.

    The job only ends when the stream ends. Ending it when the handler
    returns would mark the worker idle during generation, because Starlette
    only iterates the body after the handler returns the response.
    """
    client = gateway.client()
    request = client.build_request(
        "POST",
        path,
        content=body,
        headers={"content-type": "application/json"},
    )
    try:
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        await gateway.end_job(job)
        return JSONResponse(status_code=503, content={"error": {"message": str(exc)}})

    if response.status_code != 200:
        payload = await response.aread()
        await response.aclose()
        await gateway.end_job(job)
        try:
            content = json.loads(payload)
        except ValueError:
            content = {"error": {"message": payload.decode(errors="replace")[:2000]}}
        return JSONResponse(status_code=response.status_code, content=content)

    media_type = response.headers.get("content-type", "text/event-stream")

    async def stream() -> AsyncIterator[bytes]:
        # Closing the stream on any exit (end, error, or cancellation from
        # client disconnect) aborts the request on vLLM and ends the job,
        # returning the worker to the idle clock.
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await gateway.end_job(job)

    return StreamingResponse(stream(), status_code=response.status_code, media_type=media_type)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    session_file = os.environ.get("BYOM_SESSION_FILE", DEFAULT_SESSION_FILE)
    cfg = GatewayConfig.load(session_file)
    app = create_app(cfg)
    uvicorn.run(app, host="127.0.0.1", port=cfg.gateway_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
