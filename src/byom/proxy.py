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
import re
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

#: Cabeçalhos do cliente repassados ao gateway. `host` e `authorization` nunca
#: entram: o destino é loopback e o gateway não autentica o cliente, então
#: repassá-los só aumentaria a superfície de ataque.
FORWARDED_REQUEST_HEADERS = ("accept", "openai-beta", "user-agent")

#: Segmentos de caminho aceitos. Qualquer outro caractere (barra invertida,
#: `%`, NUL) é recusado antes de montar a URL.
ALLOWED_PATH_CHARS = re.compile(r"[A-Za-z0-9._~/-]+")

#: Únicos valores aceitos em `Host`. Bloqueia DNS rebinding: um navegador que
#: resolva um domínio público para 127.0.0.1 envia o domínio nesse header.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")

#: Timeout do cliente que fala com o gateway. Sem `read`: um SSE legítimo fica
#: quieto entre eventos, e um read timeout mataria a geração no meio.
PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)

#: Deadline do primeiro byte. Cobre a VM pausada pelo Colab: o `send` espera os
#: headers e, sem prazo, o request penduraria para sempre.
FIRST_BYTE_DEADLINE_S = 60.0


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
        if not config.nonce_matches(payload, self.state):
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
            try:
                await self.verify()
            except Exception as exc:
                # Never log the nonce or any response body, only the cause.
                log.warning("Worker verification failed: %s", exc)
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
        return httpx.AsyncClient(transport=transport, timeout=PROXY_TIMEOUT)

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
    async def status(request: Request) -> Any:
        if not _host_allowed(request):
            return _bad_host()
        return runtime.status_payload()

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> Any:
        if not _host_allowed(request):
            return _bad_host()
        # A recusa acontece antes de qualquer coisa sair: o Starlette casa
        # `{path:path}` com `../internal/state`, e o httpx normaliza `..` ao
        # montar a URL, entregando o endpoint interno do gateway.
        url, path_error = _forward_url(runtime.base_url, path, request.url.query)
        if url is None:
            return _bad_path(path_error or "Invalid request path")
        reason = runtime.refuse_reason()
        if reason is not None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": reason, "type": "byom_unavailable"}},
            )
        return await _forward(runtime, make_client(), url, request)

    return app


def _bad_path(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": {"message": message, "type": "byom_bad_path"}},
    )


def _bad_host() -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content={
            "error": {
                "message": "Host not allowed: the proxy only accepts 127.0.0.1 or localhost",
                "type": "byom_bad_host",
            }
        },
    )


def _host_allowed(request: Request) -> bool:
    """Aceite apenas loopback no `Host`, com porta opcional."""
    raw = request.headers.get("host", "").strip().lower()
    if not raw or raw.startswith("["):
        return False
    name, _, port = raw.partition(":")
    if name not in LOOPBACK_HOSTS:
        return False
    if not port:
        return True
    return port.isdigit() and 1 <= int(port) <= 65535


def _path_error(path: str) -> str | None:
    """Motivo para recusar `path`, ou None quando ele é seguro."""
    if not path:
        return "Empty request path"
    if "\\" in path:
        return "Backslash is not allowed in the request path"
    if "\x00" in path:
        return "NUL byte is not allowed in the request path"
    if "%" in path:
        # O caminho chega decodificado; um `%` restante é dupla codificação.
        return "Percent-encoding is not allowed in the request path"
    if not ALLOWED_PATH_CHARS.fullmatch(path):
        return "Character outside [A-Za-z0-9._~/-] in the request path"
    if any(segment in {".", ".."} for segment in path.split("/")):
        return "Dot segments are not allowed in the request path"
    return None


def _forward_url(
    base_url: str, path: str, query: str
) -> tuple[httpx.URL | None, str | None]:
    """URL de destino validada, preservando a query string do cliente."""
    error = _path_error(path)
    if error is not None:
        return None, error
    raw = f"{base_url}/v1/{path}"
    if query:
        raw = f"{raw}?{query}"
    try:
        url = httpx.URL(raw)
    except (httpx.InvalidURL, ValueError) as exc:
        return None, f"Invalid request URL: {exc}"
    # Defesa em profundidade: depois da normalização do httpx, o caminho
    # encaminhado continua obrigatoriamente dentro de /v1/.
    if not url.path.startswith("/v1/") or "/../" in url.path:
        return None, "Request path escapes /v1/"
    return url, None


def _forward_headers(request: Request) -> dict[str, str]:
    """Allowlist de headers de entrada; o stream nunca é comprimido."""
    headers = {
        name: request.headers[name]
        for name in FORWARDED_REQUEST_HEADERS
        if name in request.headers
    }
    headers["content-type"] = request.headers.get("content-type", "application/json")
    headers["accept-encoding"] = "identity"
    return headers


async def _forward(
    runtime: Runtime,
    client: httpx.AsyncClient,
    url: httpx.URL,
    request: Request,
) -> Any:
    body = await request.body()
    upstream = client.build_request(
        request.method, url, content=body, headers=_forward_headers(request)
    )
    # O gerador do stream passa a ser o dono do cliente: fechá-lo aqui
    # abortaria uma resposta que ainda está sendo lida.
    streamed = False
    try:
        try:
            response = await asyncio.wait_for(
                client.send(upstream, stream=True), FIRST_BYTE_DEADLINE_S
            )
        except TimeoutError:
            # `asyncio.TimeoutError` é `TimeoutError` desde o Python 3.11.
            return _tunnel_unavailable(
                f"Gateway did not send a first byte within {int(FIRST_BYTE_DEADLINE_S)}s"
            )
        except httpx.HTTPError as exc:
            return _tunnel_unavailable(f"Tunnel unavailable: {exc}")

        if response.status_code != 200:
            payload = await response.aread()
            await response.aclose()
            try:
                content = json.loads(payload)
            except ValueError:
                content = {"error": {"message": payload.decode(errors="replace")[:2000]}}
            return JSONResponse(status_code=response.status_code, content=content)

        media_type = response.headers.get("content-type", "text/event-stream")

        async def stream() -> AsyncIterator[bytes]:
            # Forward each event as it arrives instead of waiting for the end
            # of the response. Closing the stream aborts the request on the
            # worker.
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        streamed = True
        return StreamingResponse(stream(), status_code=response.status_code, media_type=media_type)
    finally:
        # Qualquer saída — erro, deadline, `CancelledError` durante o `send` —
        # fecha o cliente. `aclose` é idempotente, então o `finally` do gerador
        # de stream pode chamá-lo de novo.
        if not streamed:
            await client.aclose()


def _tunnel_unavailable(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": {"message": message, "type": "byom_tunnel"}},
    )
