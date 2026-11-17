#!/usr/bin/env python3
"""Long-running kernel cell: keeps the session alive and deallocates when idle.

Run by `byom` with `colab exec -s <name> -f watchdog.py --timeout 86400` in a
background local process. The timeout must be long: the 30s `colab exec`
default would kill the cell midway, and `--timeout 0` does not work because in
this CLI version the value is a whole-call deadline, and zero makes the
polling loop spin instead of waiting for messages.

Consequences of this design:
* While this cell runs, the kernel is busy, so Colab keeps the session alive
  even with vLLM idle.
* New `colab exec` calls in the same session are blocked. Status and
  diagnostics go through the gateway over the tunnel, `colab status`, and
  `colab sessions`.

Uses only the standard library because it runs on the kernel, not in the
worker venv. No prompt content is read or logged.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SESSION_FILE = "/content/byom/session.json"
POLL_S = 10.0


def _log(message: str) -> None:
    print(f"[byom-watchdog] {message}", flush=True)


def _get(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback
        return json.loads(response.read().decode())


def _post(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(url, data=b"", method="POST")  # noqa: S310 - loopback
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback
        return json.loads(response.read().decode())


def _unassign() -> bool:
    """Deallocate the VM from the kernel context."""
    try:
        from google.colab import runtime  # type: ignore[import-not-found]
    except ImportError:
        _log("google.colab is unavailable; could not deallocate the runtime")
        return False
    try:
        runtime.unassign()
    except Exception as exc:  # pragma: no cover - runtime-dependent
        _log(f"unassign failed: {exc}")
        return False
    return True


def main() -> int:
    cfg = json.loads(Path(SESSION_FILE).read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{cfg.get('gateway_port', 8001)}"
    idle_timeout_s = float(cfg.get("idle_timeout_min", 20)) * 60
    watchdog_timeout_s = float(cfg.get("watchdog_timeout_s", 60))

    ready_seen = False
    silent_since: float | None = None

    _log(f"Started; idle timeout is {idle_timeout_s:.0f}s")

    while True:
        try:
            state = _get(f"{base}/internal/state")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # A failed poll is not idleness: the gateway may still be starting.
            if silent_since is None:
                silent_since = time.monotonic()
                _log(f"Gateway is not responding ({exc})")
            elif ready_seen and time.monotonic() - silent_since > watchdog_timeout_s:
                _log("Gateway heartbeat lost after becoming ready; deallocating runtime")
                if _unassign():
                    return 0
                return 1
            time.sleep(POLL_S)
            continue

        silent_since = None
        if state.get("model_ready"):
            ready_seen = True

        idle_s = float(state.get("idle_s", 0.0))
        _log(f"Idle={idle_s:.0f}s jobs={state.get('active_jobs', 0)}")

        if ready_seen and idle_s >= idle_timeout_s:
            try:
                decision = _post(f"{base}/internal/begin-shutdown")
            except (urllib.error.URLError, OSError, ValueError) as exc:
                _log(f"begin-shutdown failed: {exc}")
            else:
                if decision.get("accepted"):
                    _log(f"Idle for {idle_s:.0f}s; deallocating runtime")
                    if _unassign():
                        return 0
                    return 1

        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
