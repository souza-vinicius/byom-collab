"""Supervision of the long-running Colab kernel watchdog."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import colab, config

log = logging.getLogger("byom.watchdog")

WATCHDOG_EXEC_TIMEOUT_S = 86400
CHECK_INTERVAL_S = 5.0
RESTART_BACKOFF_MIN_S = 1.0
RESTART_BACKOFF_MAX_S = 30.0
HEALTH_TIMEOUT_S = 2.0

#: Falhas consecutivas de execução sem nenhum heartbeat válido antes de parar
#: de relançar. Depois de uma reciclagem da VM o `session.json` some e a célula
#: sai com código 1: relançar para sempre só gasta o slot do `colab exec`.
MAX_CONSECUTIVE_FAILURES = 5

ProcessFactory = Callable[..., subprocess.Popen[bytes]]
HeartbeatProbe = Callable[[], Awaitable[float | None]]


@dataclass
class WatchdogState:
    """Estado observável do supervisor, lido por `byom status`."""

    state: str = "stopped"
    failures: int = 0
    last_error: str | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "failures": self.failures,
            "last_error": self.last_error,
            "pid": self.pid,
        }


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
        self.status = WatchdogState()

    async def run(self, stop: asyncio.Event) -> None:
        """Start and supervise the watchdog until `stop` is signaled.

        After `MAX_CONSECUTIVE_FAILURES` executions that die without a single
        valid heartbeat — the case of a recycled VM whose `session.json`
        disappeared — the supervisor stops relaunching and records why, so
        `byom status` shows the cause instead of an endless restart loop.
        """
        backoff = RESTART_BACKOFF_MIN_S
        self._reset_log()
        try:
            while not stop.is_set():
                if self.process is None:
                    try:
                        self.process = await asyncio.to_thread(self._start_process)
                    except Exception as exc:
                        log.exception("Could not start watchdog")
                        self._record_failure(f"falha ao iniciar: {exc}")
                        if self.status.state == "gave_up":
                            return
                        await config.sleep_or_stop(stop, backoff)
                        backoff = min(RESTART_BACKOFF_MAX_S, backoff * 2)
                        continue
                    self._save_pid(self.process.pid)
                    self.status.pid = self.process.pid
                    self.status.state = "running"
                    log.info("Watchdog started (pid %s)", self.process.pid)

                await config.sleep_or_stop(stop, self.check_interval_s)
                if stop.is_set():
                    break

                reason = await self._restart_reason()
                if reason is None:
                    backoff = RESTART_BACKOFF_MIN_S
                    continue

                log.warning("Restarting watchdog: %s", reason)
                await self._stop_process()
                self._save_pid(None)
                self.status.pid = None
                self._record_failure(reason)
                if self.status.state == "gave_up":
                    return
                await config.sleep_or_stop(stop, backoff)
                backoff = min(RESTART_BACKOFF_MAX_S, backoff * 2)
        finally:
            await self._stop_process()
            self._save_pid(None)
            self.status.pid = None
            if self.status.state != "gave_up":
                self.status.state = "stopped"

    def _reset_log(self) -> None:
        """Cada sessão começa com `watchdog.log` vazio.

        Cada relançamento do `colab exec` fazia append no mesmo arquivo e ele
        crescia sem limite; truncar no início da sessão mantém só a atual.
        """
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
        except OSError as exc:
            log.warning("Could not reset %s: %s", self.log_path, exc)

    def _record_failure(self, reason: str) -> None:
        self.status.failures += 1
        self.status.last_error = reason
        if self.status.failures < MAX_CONSECUTIVE_FAILURES:
            return
        self.status.state = "gave_up"
        self.status.last_error = (
            f"watchdog desistiu após {self.status.failures} reinícios: {reason}. "
            "Rode `byom down` e `byom up` para recriar a sessão."
        )
        log.error("%s", self.status.last_error)

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
        if silence_s is None:
            return None
        if silence_s >= self.restart_after_s:
            return f"No heartbeat for {silence_s:.0f}s"
        # Heartbeat válido: zera o contador de falhas consecutivas.
        self.status.failures = 0
        self.status.last_error = None
        return None

    async def _probe_heartbeat(self) -> float | None:
        url = f"http://127.0.0.1:{self.cfg.tunnel_port}/health"
        headers = {config.NONCE_HEADER: self.state.nonce}
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
                response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            log.warning("Watchdog heartbeat probe failed: %s", exc)
            return None
        if response.status_code != 200:
            log.warning("Watchdog heartbeat probe returned HTTP %s", response.status_code)
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            log.warning("Watchdog heartbeat probe returned invalid JSON: %s", exc)
            return None
        if "nonce" in payload and not config.nonce_matches(payload, self.state):
            log.warning("Watchdog heartbeat probe returned a different session nonce")
            return None
        try:
            value = payload.get("watchdog_silence_s")
            return None if value is None else float(value)
        except (TypeError, ValueError) as exc:
            log.warning("Watchdog heartbeat probe returned an invalid silence value: %s", exc)
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
        try:
            if pid is None:
                current = config.update_state(watchdog_pid=None, watchdog_start_ticks=None)
            else:
                # O start time identifica o PID mesmo depois de reciclado.
                current = config.update_state(
                    watchdog_pid=pid,
                    watchdog_start_ticks=config.process_start_ticks(pid),
                )
        except (config.StateFileError, OSError) as exc:
            log.warning("Could not save watchdog PID: %s", exc)
            return
        if current.session != self.state.session:
            log.warning("State session changed while saving watchdog PID")
