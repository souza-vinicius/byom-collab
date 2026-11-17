"""Supervision of the long-running Colab kernel watchdog."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from . import colab, config

log = logging.getLogger("byom.watchdog")

WATCHDOG_EXEC_TIMEOUT_S = 86400
CHECK_INTERVAL_S = 5.0
RESTART_BACKOFF_MIN_S = 1.0
RESTART_BACKOFF_MAX_S = 30.0
HEALTH_TIMEOUT_S = 2.0

ProcessFactory = Callable[..., subprocess.Popen[bytes]]
HeartbeatProbe = Callable[[], Awaitable[float | None]]


def watchdog_file() -> Path:
    override = os.environ.get("BYOM_WORKER_DIR")
    worker_dir = (
        Path(override) if override else Path(__file__).resolve().parent.parent.parent / "worker"
    )
    return worker_dir / "watchdog.py"


class WatchdogSupervisor:
    """Keeps the Colab watchdog execution alive while the local daemon runs.

    `colab exec` streams a long-running kernel cell over a WebSocket. If that
    local connection exits or stops delivering heartbeats, restart the
    execution before the gateway's watchdog deadline expires.
    """

    def __init__(
        self,
        *,
        state: config.State,
        cfg: config.Config,
        process_factory: ProcessFactory | None = None,
        heartbeat_probe: HeartbeatProbe | None = None,
        check_interval_s: float = CHECK_INTERVAL_S,
    ) -> None:
        self.state = state
        self.cfg = cfg
        self.process_factory = process_factory or colab.exec_popen
        self.heartbeat_probe = heartbeat_probe or self._probe_heartbeat
        self.restart_after_s = max(1.0, min(30.0, cfg.watchdog_timeout_s / 2))
        self.check_interval_s = min(check_interval_s, max(0.5, self.restart_after_s / 2))
        self.log_path = config.config_dir() / "logs" / "watchdog.log"
        self.process: subprocess.Popen[bytes] | None = None

    async def run(self, stop: asyncio.Event) -> None:
        """Start and supervise the watchdog until `stop` is signaled."""
        backoff = RESTART_BACKOFF_MIN_S
        try:
            while not stop.is_set():
                if self.process is None:
                    try:
                        self.process = await asyncio.to_thread(self._start_process)
                    except Exception:
                        log.exception("Could not start watchdog")
                        await _sleep_or_stop(stop, backoff)
                        backoff = min(RESTART_BACKOFF_MAX_S, backoff * 2)
                        continue
                    self._save_pid(self.process.pid)
                    log.info("Watchdog started (pid %s)", self.process.pid)

                await _sleep_or_stop(stop, self.check_interval_s)
                if stop.is_set():
                    break

                reason = await self._restart_reason()
                if reason is None:
                    backoff = RESTART_BACKOFF_MIN_S
                    continue

                log.warning("Restarting watchdog: %s", reason)
                await self._stop_process()
                self._save_pid(None)
                await _sleep_or_stop(stop, backoff)
                backoff = min(RESTART_BACKOFF_MAX_S, backoff * 2)
        finally:
            await self._stop_process()
            self._save_pid(None)

    def _start_process(self) -> subprocess.Popen[bytes]:
        return self.process_factory(
            self.state.session,
            watchdog_file(),
            timeout=WATCHDOG_EXEC_TIMEOUT_S,
            log_path=self.log_path,
        )

    async def _restart_reason(self) -> str | None:
        if self.process is None:
            return None
        returncode = self.process.poll()
        if returncode is not None:
            return f"`colab exec` exited with code {returncode}"

        silence_s = await self.heartbeat_probe()
        if silence_s is not None and silence_s >= self.restart_after_s:
            return f"No heartbeat for {silence_s:.0f}s"
        return None

    async def _probe_heartbeat(self) -> float | None:
        url = f"http://127.0.0.1:{self.cfg.tunnel_port}/health"
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
                response = await client.get(url)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
            if payload.get("nonce") != self.state.nonce:
                return None
            value = payload.get("watchdog_silence_s")
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    async def _stop_process(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            return
        try:
            await asyncio.to_thread(process.wait, timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                return
            await asyncio.to_thread(process.wait)

    def _save_pid(self, pid: int | None) -> None:
        current = config.load_state()
        if current is None or current.session != self.state.session:
            return
        current.watchdog_pid = pid
        try:
            config.save_state(current)
        except OSError:
            log.exception("Could not save watchdog PID")


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)
