"""Local process that runs the proxy and tunnel supervisor.

A single process (`byom _daemon`) avoids coordination between two PIDs and
simplifies `byom down`. The PID is stored in `state.json`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path

import uvicorn

from . import config
from .proxy import Runtime, create_app
from .tunnel import TunnelSupervisor
from .watchdog_supervisor import WatchdogSupervisor

log = logging.getLogger("byom.daemon")


def _log_path(name: str) -> Path:
    return config.config_dir() / "logs" / f"{name}.log"


async def run() -> int:
    state = await _load_registered_state()
    if state is None:
        print("No active session. Run `byom up` first.", file=sys.stderr)
        return 1

    cfg = config.load_config()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    tunnel = TunnelSupervisor(
        session=state.session,
        endpoint=state.endpoint,
        cfg=cfg,
        nonce=state.nonce,
        log_path=_log_path("tunnel"),
    )
    watchdog = WatchdogSupervisor(state=state, cfg=cfg)
    runtime = Runtime(state=state, cfg=cfg, tunnel=tunnel, watchdog=watchdog.status)
    app = create_app(runtime)

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=cfg.proxy_port, log_level="warning")
    )

    # The proxy app starts the worker verification loop during startup, so do
    # not create another one here.
    tasks = [
        asyncio.create_task(tunnel.run(stop), name="tunnel"),
        asyncio.create_task(server.serve(), name="proxy"),
        asyncio.create_task(watchdog.run(stop), name="watchdog"),
    ]
    log.info("Daemon started: proxy listening on 127.0.0.1:%s", cfg.proxy_port)

    await stop.wait()
    server.should_exit = True
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Daemon stopped")
    return 0


async def _load_registered_state(timeout_s: float = 5.0) -> config.State | None:
    """Wait for state, accepting the PID published by the child before exec."""
    expected = os.environ.get("BYOM_DAEMON_EXPECTED_PID", "")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            state = config.load_state()
        except config.StateFileError as exc:
            log.error("Cannot read state.json: %s", exc)
            return None
        if state is None:
            return None
        if expected.isdigit() and int(expected) == os.getpid():
            return state
        if state.daemon_pid == os.getpid():
            return state
        await asyncio.sleep(0.05)
    log.error("Daemon PID %s was not registered in state.json", os.getpid())
    return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=str(_log_path("daemon")),
    )
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
