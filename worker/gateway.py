#!/usr/bin/env python3
"""Worker gateway: the authoritative activity point for the session.

Responsibilities:
* Forward `/v1/chat/completions` and `/v1/models` to vLLM, with streaming.
* Inject the model's documented `reasoning_effort` when the client omits it;
  never invent a value for a model whose docs do not define one.
* Apply the Gemma 4 thinking policy (explicit catalog key): `<|think|>` at the
  start of the system prompt and no thought blocks from prior assistant turns.
* Fill documented image generation defaults through sd-server's
  `sd_cpp_extra_args`, without overwriting a client's own choice.
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
import re
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

#: Fallback delay after shutdown is accepted. The watchdog normally
#: deallocates within seconds; this covers a dead watchdog cell or supervisor.
UNASSIGN_FALLBACK_DELAY_S = 30.0
UNASSIGN_FALLBACK_CHECK_S = 5.0

#: Conservative concurrency ceiling for the in-process inference queue.
DEFAULT_MAX_CONCURRENT_JOBS = 4
MAX_CONCURRENT_JOBS_ENV = "BYOM_MAX_CONCURRENT_JOBS"
#: União documentada, conferida em 2026-11-23. Origens por modelo no catálogo.
REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
RUNTIME_IDS = frozenset({"vllm", "llama-cpp", "image"})

THINKING_POLICY_FIELDS = frozenset(
    {"system_token", "thought_open", "thought_close", "strip_history_thought", "source"}
)
GENERATION_DEFAULT_FIELDS = frozenset({"steps", "guidance"})
SD_EXTRA_TAG = "sd_cpp_extra_args"

log = logging.getLogger("byom.gateway")


def _max_concurrent_jobs() -> int:
    raw = os.environ.get(MAX_CONCURRENT_JOBS_ENV)
    if not raw:
        return DEFAULT_MAX_CONCURRENT_JOBS
    try:
        value = int(raw)
    except ValueError:
        log.warning(
            "Invalid %s=%r; using %d", MAX_CONCURRENT_JOBS_ENV, raw, DEFAULT_MAX_CONCURRENT_JOBS
        )
        return DEFAULT_MAX_CONCURRENT_JOBS
    if value < 1:
        log.warning(
            "Invalid %s=%r; using %d", MAX_CONCURRENT_JOBS_ENV, raw, DEFAULT_MAX_CONCURRENT_JOBS
        )
        return DEFAULT_MAX_CONCURRENT_JOBS
    return value


@dataclass
class GatewayConfig:
    """Configuration delivered by `byom` in `session.json`."""

    nonce: str
    runtime: str = "vllm"
    model: str = ""
    served_model_name: str = "qwen3.8-27b"
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_pid_file: str = "/content/byom/vllm.pid"
    gateway_port: int = 8001
    #: None = a página do modelo não documenta reasoning_effort (Gemma 4).
    reasoning_effort: str | None = None
    #: Chave explícita do catálogo; None = sem transformação de prompt.
    thinking_policy: dict[str, Any] | None = None
    #: Padrões de geração do modelo de imagem, com nomes da documentação.
    generation_defaults: dict[str, int | float] | None = None
    idle_timeout_min: float = 20.0
    max_job_min: float = 40.0
    watchdog_timeout_s: float = 60.0

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_SESSION_FILE) -> GatewayConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("session.json must contain a JSON object")
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate the `session.json` boundary before the gateway starts."""
        if not isinstance(self.nonce, str) or not self.nonce:
            raise ValueError(f"Invalid nonce: {self.nonce!r}; expected a non-empty string")
        if self.runtime not in RUNTIME_IDS:
            choices = ", ".join(sorted(RUNTIME_IDS))
            raise ValueError(f"Invalid runtime: {self.runtime!r}; expected one of {choices}")
        for key in ("gateway_port",):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"Invalid {key}: {value!r}; expected a port between 1 and 65535")
        for key in ("idle_timeout_min", "max_job_min", "watchdog_timeout_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Invalid {key}: {value!r}; expected a positive timeout")
        if self.reasoning_effort is not None and (
            not isinstance(self.reasoning_effort, str)
            or self.reasoning_effort not in REASONING_EFFORTS
        ):
            raise ValueError(
                f"Invalid reasoning_effort: {self.reasoning_effort!r}; "
                f"expected one of {', '.join(sorted(REASONING_EFFORTS))} or null"
            )
        if self.thinking_policy is not None:
            policy = self.thinking_policy
            if not isinstance(policy, dict) or set(policy) - THINKING_POLICY_FIELDS:
                raise ValueError(f"Invalid thinking_policy: {policy!r}")
            for key in ("system_token", "thought_open", "thought_close"):
                if not isinstance(policy.get(key), str) or not policy[key]:
                    raise ValueError(f"Invalid thinking_policy.{key}: {policy.get(key)!r}")
            if not isinstance(policy.get("strip_history_thought", True), bool):
                raise ValueError("thinking_policy.strip_history_thought must be boolean")
        if self.generation_defaults is not None:
            defaults = self.generation_defaults
            if not isinstance(defaults, dict) or set(defaults) - GENERATION_DEFAULT_FIELDS:
                raise ValueError(f"Invalid generation_defaults: {defaults!r}")
            if set(defaults) != GENERATION_DEFAULT_FIELDS:
                raise ValueError("generation_defaults needs steps and guidance")
            steps = defaults["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
                raise ValueError(f"Invalid generation_defaults.steps: {steps!r}")
            guidance = defaults["guidance"]
            if (
                isinstance(guidance, bool)
                or not isinstance(guidance, (int, float))
                or not 0 < guidance <= 1000
            ):
                raise ValueError(f"Invalid generation_defaults.guidance: {guidance!r}")


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
    unassign_runtime: Callable[[], Awaitable[bool]] | None = None
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
    _unassign_task: asyncio.Task[Any] | None = None

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
        if self._unassign_task is not None:
            self._unassign_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._unassign_task
            self._unassign_task = None
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
        limit = _max_concurrent_jobs()
        if self.active_jobs >= limit:
            return f"Too many active jobs (limit {limit})"
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
            self._schedule_unassign_fallback("idle shutdown accepted")
            return True, "idle"

    # -- readiness and supervision ----------------------------------------
    async def poll_readiness(self) -> bool:
        try:
            response = await self.client().get("/v1/models")
            ready = response.status_code == 200
        except Exception as exc:
            # Diagnostics only: never log a prompt or a response body here.
            log.warning("vLLM readiness poll failed: %s", exc)
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
        self._schedule_unassign_fallback(reason)

    def _schedule_unassign_fallback(self, reason: str) -> None:
        """Start one deallocation task if the watchdog has not done its job."""
        if self._unassign_task is not None and not self._unassign_task.done():
            return
        self._unassign_task = asyncio.create_task(
            self._unassign_fallback(reason), name="unassign-fallback"
        )

    async def _unassign_fallback(self, reason: str) -> None:
        """Deallocate from inside the VM after the watchdog's grace period.

        This uses the same `google.colab.runtime` module as the watchdog cell.
        It never runs while a job is active: shutdown/refusal blocks new jobs,
        and the reaper stops stuck ones before this deadline.
        """
        try:
            await asyncio.sleep(UNASSIGN_FALLBACK_DELAY_S)
            deadline = time.monotonic() + self.cfg.max_job_min * 60 + CANCEL_GRACE_S
            while self.active_jobs:
                if time.monotonic() >= deadline:
                    log.error("Not deallocating: %d job(s) still active", self.active_jobs)
                    return
                await asyncio.sleep(UNASSIGN_FALLBACK_CHECK_S)
            if not (self._shutting_down or self._vllm_down):
                return
            log.warning("Watchdog did not deallocate the VM; gateway fallback: %s", reason)
            callback = self.unassign_runtime or _unassign_runtime
            await callback()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Gateway unassign fallback failed")

    async def shutdown_vllm_process(self) -> None:
        if self.shutdown_vllm is not None:
            await self.shutdown_vllm()
            return
        await asyncio.to_thread(_signal_vllm, self.cfg.vllm_pid_file)

    # -- endpoints -------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        return {
            "nonce": self.cfg.nonce,
            "runtime": self.cfg.runtime,
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


async def _unassign_runtime() -> bool:
    """Deallocate the VM from inside the worker, via the kernel runtime API.

    Same module as `worker/watchdog.py`; imported lazily because it only exists
    in the Colab kernel environment.
    """
    try:
        from google.colab import runtime  # type: ignore[import-not-found]
    except ImportError as exc:
        log.error("google.colab.runtime is unavailable; VM not deallocated: %s", exc)
        return False
    try:
        await asyncio.to_thread(runtime.unassign)
    except Exception as exc:  # pragma: no cover - runtime-dependent
        log.error("runtime.unassign() failed: %s", exc)
        return False
    log.warning("Gateway fallback deallocated the runtime")
    return True


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _inject_reasoning_effort(body: bytes, effort: str | None) -> bytes:
    """Add the model's documented `reasoning_effort` only when absent.

    `effort=None` means the model's documentation does not define the field
    (Gemma 4): nothing is injected and Gemma's `enable_thinking` is untouched.
    """
    if effort is None:
        return body
    payload = _json_object(body)
    if payload is None:
        return body
    kwargs = payload.get("chat_template_kwargs")
    if "chat_template_kwargs" in payload and not isinstance(kwargs, dict):
        # Preserve the client's value: silently replacing it discards intent.
        log.warning("Preserving non-object chat_template_kwargs from client")
        return body
    if kwargs is None:
        kwargs = {}
        payload["chat_template_kwargs"] = kwargs
    if "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = effort
        return json.dumps(payload).encode()
    return body


def _thinking_disabled_by_client(payload: dict[str, Any]) -> bool:
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return False
    value = kwargs.get("enable_thinking")
    if isinstance(value, bool):
        return value is False
    return isinstance(value, str) and value.strip().lower() == "false"


def _strip_thought_blocks(text: str, policy: dict[str, Any]) -> str:
    pattern = re.compile(
        re.escape(str(policy["thought_open"]))
        + r".*?"
        + re.escape(str(policy["thought_close"])),
        re.DOTALL,
    )
    return pattern.sub("", text).strip()


def _apply_thinking_policy(body: bytes, policy: dict[str, Any] | None) -> bytes:
    """Gemma 4: `<|think|>` in system; no thought blocks in assistant history.

    Driven by the explicit catalog key, never by comparing a model-name string.
    A client that already placed the token or cleaned the history keeps its
    payload; `enable_thinking: false` disables both transformations.
    """
    if not policy or not policy.get("strip_history_thought", True):
        return body
    payload = _json_object(body)
    if payload is None or _thinking_disabled_by_client(payload):
        return body
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return body
    token = str(policy["system_token"])
    changed = False

    system = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None
    )
    if system is None:
        messages.insert(0, {"role": "system", "content": token})
        changed = True
    else:
        content = system.get("content")
        if isinstance(content, str):
            if not content.lstrip().startswith(token):
                system["content"] = token + content
                changed = True
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    if not part["text"].lstrip().startswith(token):
                        part["text"] = token + part["text"]
                        changed = True
                    break
            else:
                content.insert(0, {"type": "text", "text": token})
                changed = True

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            cleaned = _strip_thought_blocks(content, policy)
            if cleaned != content:
                message["content"] = cleaned
                changed = True
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    cleaned = _strip_thought_blocks(part["text"], policy)
                    if cleaned != part["text"]:
                        part["text"] = cleaned
                        changed = True
    return json.dumps(payload).encode() if changed else body


def _inject_image_defaults(
    body: bytes, defaults: dict[str, int | float] | None
) -> bytes:
    """Map Unsloth's `steps`/`guidance` to sd-server's documented schema.

    sd-server documents `sd_cpp_extra_args` embedded in `prompt`; its native
    fields are `sample_params.sample_steps` and `sample_params.guidance.txt_cfg`.
    Top-level `guidance` is NOT a documented sd-server field.
    """
    if not defaults:
        return body
    payload = _json_object(body)
    if payload is None:
        return body
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or f"<{SD_EXTRA_TAG}>" in prompt:
        return body
    steps = payload.pop("steps", defaults["steps"])
    guidance = payload.pop("guidance", defaults["guidance"])
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
        log.warning("Preserving image request with invalid client steps=%r", steps)
        return body
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not 0 < guidance <= 1000
    ):
        log.warning("Preserving image request with invalid client guidance=%r", guidance)
        return body
    extra = {
        "sample_params": {
            "sample_steps": steps,
            "guidance": {"txt_cfg": guidance},
        }
    }
    block = json.dumps(extra, separators=(",", ":"))
    payload["prompt"] = f"{prompt} <{SD_EXTRA_TAG}>{block}</{SD_EXTRA_TAG}>"
    return json.dumps(payload).encode()


def _is_loopback(request: Request) -> bool:
    """Only processes inside the VM may call the internal control endpoints."""
    host = request.client.host if request.client is not None else ""
    return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _forbidden() -> JSONResponse:
    return JSONResponse(
        status_code=403, content={"error": {"message": "Internal endpoints are loopback-only"}}
    )


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
    async def internal_state(request: Request) -> Any:
        """Called by the watchdog; each call refreshes the heartbeat."""
        if not _is_loopback(request):
            return _forbidden()
        gateway.register_watchdog_poll()
        return gateway.status_payload()

    @app.post("/internal/begin-shutdown")
    async def begin_shutdown(request: Request) -> Any:
        if not _is_loopback(request):
            return _forbidden()
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
        if cfg.runtime == "image":
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "message": "Image model. Use POST /v1/images/generations; "
                        "there is no /v1/chat/completions for this runtime."
                    }
                },
            )
        reason = gateway.refuse_reason()
        if reason is not None:
            return JSONResponse(status_code=503, content={"error": {"message": reason}})

        body = await request.body()
        body = _apply_thinking_policy(body, cfg.thinking_policy)
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

    @app.post("/v1/images/generations")
    async def images_generations(request: Request) -> Any:
        """Qwen-Image-2.1: geração de imagem, não conclusão de chat."""
        if cfg.runtime != "image":
            return JSONResponse(
                status_code=501,
                content={"error": {"message": "Image generation is unavailable in this runtime."}},
            )
        return await _forward_image(gateway, request, "/v1/images/generations")

    @app.post("/v1/images/edits")
    async def images_edits(request: Request) -> Any:
        if cfg.runtime != "image":
            return JSONResponse(
                status_code=501,
                content={"error": {"message": "Image editing is unavailable in this runtime."}},
            )
        return await _forward_image(gateway, request, "/v1/images/edits")

    return app


async def _forward_image(gateway: Gateway, request: Request, path: str) -> Any:
    """Encaminhe a chamada de imagem contando o tempo como trabalho ativo."""
    reason = gateway.refuse_reason()
    if reason is not None:
        return JSONResponse(status_code=503, content={"error": {"message": reason}})
    job = await gateway.begin_job()
    if job is None:
        return JSONResponse(status_code=503, content={"error": {"message": "Request rejected"}})
    body = await request.body()
    if path == "/v1/images/generations":
        body = _inject_image_defaults(body, gateway.cfg.generation_defaults)
    try:
        response = await gateway.client().post(
            path,
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
    except httpx.HTTPError as exc:
        await gateway.end_job(job)
        return JSONResponse(status_code=503, content={"error": {"message": str(exc)}})
    await gateway.end_job(job)
    return JSONResponse(status_code=response.status_code, content=_safe_json(response))


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
