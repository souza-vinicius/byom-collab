"""Tests for the VM-side watchdog cell, with a fake `google.colab` module."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parent.parent / "worker"


@pytest.fixture
def watchdog_module():
    spec = importlib.util.spec_from_file_location("byom_worker_watchdog", WORKER / "watchdog.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StopLoop(Exception):
    pass


def install_fake_runtime(monkeypatch, calls: list[str], *, fail: bool = False) -> None:
    google = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")

    def unassign() -> None:
        if fail:
            raise RuntimeError("simulated unassign failure")
        calls.append("unassign")

    setattr(colab_module, "runtime", types.SimpleNamespace(unassign=unassign))  # noqa: B010
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)


def make_session_file(watchdog_module, tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "gateway_port": 8001,
                "idle_timeout_min": 1,
                "watchdog_timeout_s": 60,
                "nonce": "watchdog-nonce",
            }
        ),
        encoding="utf-8",
    )
    watchdog_module.SESSION_FILE = str(path)


def test_unassign_uses_google_colab_runtime(watchdog_module, monkeypatch):
    calls: list[str] = []
    install_fake_runtime(monkeypatch, calls)
    assert watchdog_module._unassign() is True
    assert calls == ["unassign"]


def test_unassign_reports_failure(watchdog_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert watchdog_module._unassign() is False


def test_idle_calls_unassign(watchdog_module, monkeypatch, tmp_path):
    calls: list[str] = []
    install_fake_runtime(monkeypatch, calls)
    make_session_file(watchdog_module, tmp_path)
    monkeypatch.setattr(
        watchdog_module,
        "_get",
        lambda url, nonce: {"model_ready": True, "idle_s": 100.0, "active_jobs": 0},
    )
    monkeypatch.setattr(
        watchdog_module, "_post", lambda url, nonce: {"accepted": True}
    )
    monkeypatch.setattr(watchdog_module, "_unassign", lambda: calls.append("called") or True)

    def stop(*args, **kwargs):
        raise StopLoop

    monkeypatch.setattr(
        watchdog_module, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=stop)
    )

    # A célula encerra com 0 logo após desalocar; não há novo sleep.
    assert watchdog_module.main() == 0
    assert calls == ["called"]


def test_active_jobs_block_unassign(watchdog_module, monkeypatch, tmp_path):
    calls: list[str] = []
    install_fake_runtime(monkeypatch, calls)
    make_session_file(watchdog_module, tmp_path)
    monkeypatch.setattr(
        watchdog_module,
        "_get",
        lambda url, nonce: {"model_ready": True, "idle_s": 100.0, "active_jobs": 1},
    )
    monkeypatch.setattr(watchdog_module, "_post", lambda url, nonce: {"accepted": True})

    def stop(*args, **kwargs):
        raise StopLoop

    monkeypatch.setattr(
        watchdog_module, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=stop)
    )

    with pytest.raises(StopLoop):
        watchdog_module.main()
    assert calls == []


def test_watchdog_envia_nonce_no_header(watchdog_module, monkeypatch):
    """O gateway recusa sem `X-BYOM-Nonce`; a célula precisa enviá-lo."""
    captured: list = []

    class FakeResponse:
        def read(self) -> bytes:
            return b'{"model_ready": false, "idle_s": 0, "active_jobs": 0}'

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return FakeResponse()

    monkeypatch.setattr(watchdog_module.urllib.request, "urlopen", fake_urlopen)

    watchdog_module._get("http://127.0.0.1:8001/internal/state", "n-123")
    watchdog_module._post("http://127.0.0.1:8001/internal/begin-shutdown", "n-123")

    sent = [{k.lower(): v for k, v in request.header_items()} for request in captured]
    assert [item["x-byom-nonce"] for item in sent] == ["n-123", "n-123"]
    assert [request.get_method() for request in captured] == ["GET", "POST"]


def test_session_json_ausente_encerra_sem_traceback(watchdog_module, tmp_path, capsys):
    """VM reciclada: a célula informa e sai com código 1, sem stack trace."""
    watchdog_module.SESSION_FILE = str(tmp_path / "nao-existe.json")

    assert watchdog_module.main() == 1

    saida = capsys.readouterr().out
    assert "VM reciclada: session.json ausente" in saida
    assert "byom up" in saida
