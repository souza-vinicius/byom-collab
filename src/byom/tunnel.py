"""SSH tunnel to the worker gateway.

The OpenSSH master connection (`ControlMaster`) is the tunnel: a single
`colab ssh --proxy-mode` connection serves byom's port forwarding and remote
commands. This matters because the runtime accepts **one** `colab ssh`
connection at a time (HTTP 429): opening a separate `ssh -L` per use would
take the master down.

Critical detail: `colab ssh --proxy-mode -s <name>` **creates** the session
when the name is not in the local store. That is why the supervisor checks
`colab sessions` and only connects while the endpoint registered in
`state.json` remains allocated.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import colab, config
from .remote import Remote, RemoteError

log = logging.getLogger("byom.tunnel")

BACKOFF_MIN_S = 1.0
BACKOFF_MAX_S = 30.0
HEALTH_TIMEOUT_S = 3.0
CONNECT_TIMEOUT_S = 30.0
MONITOR_INTERVAL_S = 2.0
ENDPOINT_RECHECK_S = 5.0


@dataclass
class TunnelState:
    """Observable tunnel state, read by `byom status`."""

    state: str = "stopped"
    reconnects: int = 0
    latency_ms: float | None = None
    last_error: str | None = None
    since: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reconnects": self.reconnects,
            "latency_ms": self.latency_ms,
            "last_error": self.last_error,
        }


class TunnelSupervisor:
    """Keep the master connection alive while the session exists."""

    def __init__(
        self,
        *,
        session: str,
        endpoint: str,
        cfg: config.Config,
        ssh_bin: str = "ssh",
        log_path: Path | None = None,
    ) -> None:
        self.session = session
        self.endpoint = endpoint
        self.cfg = cfg
        self.remote = Remote(
            session=session,
            cfg=cfg,
            ssh_bin=ssh_bin,
            log_path=log_path,
        )
        self.status = TunnelState()

    @property
    def forwards(self) -> list[tuple[int, int]]:
        """Local proxy port to the gateway, which listens on loopback only."""
        return [(self.cfg.tunnel_port, self.cfg.gateway_port)]

    @property
    def local_base_url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.tunnel_port}"

    # -- verification -----------------------------------------------------
    async def session_allocated(self) -> bool:
        """Confirm with the backend that the session still exists."""
        try:
            return await asyncio.to_thread(colab.is_allocated, self.session, self.endpoint)
        except colab.ColabCliError as exc:
            log.warning("Failed to check sessions: %s", exc)
            return False

    async def probe_health(self) -> dict[str, object] | None:
        """Read `/health` through the tunnel and measure latency."""
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
                response = await client.get(f"{self.local_base_url}/health")
        except httpx.HTTPError as exc:
            log.debug("Gateway health probe failed: %s", exc)
            return None
        self.status.latency_ms = round((time.monotonic() - started) * 1000, 1)
        if response.status_code != 200:
            log.debug("Gateway health probe returned HTTP %s", response.status_code)
            return None
        try:
            return response.json()
        except ValueError as exc:
            log.warning("Gateway health probe returned invalid JSON: %s", exc)
            return None

    # -- lifecycle --------------------------------------------------------
    async def run(self, stop: asyncio.Event) -> None:
        """Supervise the tunnel until `stop` is signaled."""
        backoff = BACKOFF_MIN_S
        while not stop.is_set():
            if not await self.session_allocated():
                self.status.state = "stopped"
                self.status.last_error = "Session is no longer allocated by the backend"
                self.status.since = time.monotonic()
                await config.sleep_or_stop(stop, ENDPOINT_RECHECK_S)
                continue

            self.status.state = "connecting" if self.status.reconnects == 0 else "reconnecting"
            self.status.since = time.monotonic()

            try:
                await asyncio.to_thread(self.remote.start, forwards=self.forwards)
            except RemoteError as exc:
                self.status.last_error = str(exc)
            else:
                if await self._wait_healthy(stop):
                    self.status.state = "connected"
                    self.status.last_error = None
                    self.status.since = time.monotonic()
                    backoff = BACKOFF_MIN_S
                else:
                    detail = self.remote.log_tail()
                    self.status.last_error = (
                        f"Tunnel failed to connect: {detail}"
                        if detail
                        else "Gateway did not respond through the tunnel"
                    )
                await self._monitor(stop)

            await asyncio.to_thread(self.remote.stop)

            if stop.is_set():
                break
            self.status.reconnects += 1
            self.status.state = "reconnecting"
            self.status.since = time.monotonic()
            await config.sleep_or_stop(stop, backoff)
            backoff = min(BACKOFF_MAX_S, backoff * 2)

        await asyncio.to_thread(self.remote.stop)
        self.status.state = "stopped"

    async def _wait_healthy(self, stop: asyncio.Event) -> bool:
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while time.monotonic() < deadline and not stop.is_set():
            if not await asyncio.to_thread(self.remote.alive):
                return False
            if await self.probe_health() is not None:
                return True
            await config.sleep_or_stop(stop, 0.5)
        return False

    async def _monitor(self, stop: asyncio.Event) -> None:
        """Follow the master, measuring latency periodically."""
        while not stop.is_set():
            if not await asyncio.to_thread(self.remote.alive):
                self.status.last_error = "SSH connection dropped"
                return
            await config.sleep_or_stop(stop, MONITOR_INTERVAL_S)
            if stop.is_set():
                return
            if not await asyncio.to_thread(self.remote.alive):
                self.status.last_error = "SSH connection dropped"
                return
            if await self.probe_health() is None:
                self.status.last_error = "Gateway did not respond through the tunnel"
                self.status.state = "reconnecting"
