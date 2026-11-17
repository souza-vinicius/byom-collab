"""Local proxy: stable endpoint used by coding agents.

It listens only on `127.0.0.1:4000` and forwards `/v1/*` through the tunnel
to the worker gateway. Before forwarding anything, it checks that the
gateway's `/health` returns the nonce of the session registered in
`state.json`: this ensures the destination is that session's worker, not some
other machine.

It also exposes `GET /__byom/status`, used by `byom status`. That endpoint is
local, does not cross the tunnel, and does not count as activity.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config
from .tunnel import TunnelSupervisor

log = logging.getLogger("byom.proxy")

VERIFY_INTERVAL_S = 5.0


@dataclass
class WorkerStatus:
    """Result of the latest gateway check."""

    verified: bool = False
    model_ready: bool = False
    idle_s: float | None = None
    idle_timeout_min: float | None = None
    active_jobs: int | None = None
    watchdog_silence_s: float | None = None
    vllm_down: bool = False
    shutting_down: bool = False
    error: str | None = None
    checked_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state(),
            "verified": self.verified,
            "model_ready": self.model_ready,
            "idle_s": self.idle_s,
            "idle_timeout_min": self.idle_timeout_min,
            "active_jobs": self.active_jobs,
            "watchdog_silence_s": self.watchdog_silence_s,
            "vllm_down": self.vllm_down,
            "shutting_down": self.shutting_down,
            "error": self.error,
            "checked_ago_s": None
            if self.checked_at is None
            else round(time.monotonic() - self.checked_at, 1),
        }

    def _state(self) -> str:
        if self.error is not None or self.vllm_down:
            return "unavailable"
        if not self.verified:
            return "initializing"
        return "ready" if self.model_ready else "initializing"


@dataclass
class Runtime:
    """State shared between the proxy and the tunnel supervisor."""

    state: config.State
    cfg: config.Config
    tunnel: TunnelSupervisor
    worker: WorkerStatus = field(default_factory=WorkerStatus)

    @property
    def base_url(self) -> str:
        return self.tunnel.local_base_url

    async def verify(self) -> None:
        """Read `/health` through the tunnel and validate the session nonce."""
        payload = await self.tunnel.probe_health()
        self.worker.checked_at = time.monotonic()
        if payload is None:
            self.worker.verified = False
            self.worker.model_ready = False
            self.worker.error = (
                self.tunnel.status.last_error or "Gateway is inaccessible through the tunnel"
            )
            return
        nonce = str(payload.get("nonce", ""))
        if not nonce or nonce != self.state.nonce:
            # Never forward to a destination of unknown identity.
            self.worker.verified = False
            self.worker.model_ready = False
            self.worker.error = "Gateway nonce does not match the registered session"
            return
        self.worker.verified = True
        self.worker.vllm_down = bool(payload.get("vllm_down"))
        self.worker.model_ready = bool(payload.get("model_ready")) and not self.worker.vllm_down
        self.worker.idle_s = payload.get("idle_s")
        self.worker.idle_timeout_min = payload.get("idle_timeout_min")
        self.worker.active_jobs = payload.get("active_jobs")
        try:
            silence = payload.get("watchdog_silence_s")
            self.worker.watchdog_silence_s = None if silence is None else float(silence)
        except (TypeError, ValueError):
            self.worker.watchdog_silence_s = None
        self.worker.shutting_down = bool(payload.get("shutting_down"))
        if self.worker.vllm_down:
            reason = payload.get("shutdown_reason") or "vLLM stopped"
            self.worker.error = f"vLLM unavailable: {reason}"
        elif (
            self.worker.watchdog_silence_s is not None
            and self.worker.watchdog_silence_s >= self.cfg.watchdog_timeout_s
        ):
            self.worker.error = f"No watchdog heartbeat for {self.worker.watchdog_silence_s:.0f}s"
        else:
            self.worker.error = None

    async def verify_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            with contextlib.suppress(Exception):
                await self.verify()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=VERIFY_INTERVAL_S)

    def status_payload(self) -> dict[str, Any]:
        return {
            "session": self.state.session,
            "endpoint": self.state.endpoint,
            "created_at": self.state.created_at,
            "tunnel": self.tunnel.status.to_dict(),
            "worker": self.worker.to_dict(),
        }

    def refuse_reason(self) -> str | None:
        if self.worker.error is not None:
            return self.worker.error
        if self.worker.vllm_down:
            return "vLLM unavailable"
        if not self.worker.verified:
            return "Worker identity has not been verified yet"
        return None


def create_app(
    runtime: Runtime,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> FastAPI:
    """Build the proxy app. The HTTP transport is injectable for tests."""
    stop = asyncio.Event()

    def make_client() -> httpx.AsyncClient:
        if client_factory is not None:
            return client_factory()
        return httpx.AsyncClient(transport=transport, timeout=None)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(runtime.verify_loop(stop), name="verify")
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/__byom/status")
    async def status() -> dict[str, Any]:
        return runtime.status_payload()

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> Any:
        reason = runtime.refuse_reason()
        if reason is not None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": reason, "type": "byom_unavailable"}},
            )
        return await _forward(runtime, make_client(), path, request)

    return app


async def _forward(
    runtime: Runtime,
    client: httpx.AsyncClient,
    path: str,
    request: Request,
) -> Any:
    url = f"{runtime.base_url}/v1/{path}"
    body = await request.body()
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    upstream = client.build_request(request.method, url, content=body, headers=headers)
    try:
        response = await client.send(upstream, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"Tunnel unavailable: {exc}", "type": "byom_tunnel"}},
        )

    if response.status_code != 200:
        payload = await response.aread()
        await response.aclose()
        await client.aclose()
        try:
            content = json.loads(payload)
        except ValueError:
            content = {"error": {"message": payload.decode(errors="replace")[:2000]}}
        return JSONResponse(status_code=response.status_code, content=content)

    media_type = response.headers.get("content-type", "text/event-stream")

    async def stream() -> AsyncIterator[bytes]:
        # Forward each event as it arrives instead of waiting for the end of
        # the response. Closing the stream aborts the request on the worker.
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(stream(), status_code=response.status_code, media_type=media_type)
