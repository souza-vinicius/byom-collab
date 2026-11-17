from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "worker"


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    """No test touches ~/.config/byom."""
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    return tmp_path / "byom"


@pytest.fixture(scope="session")
def gateway_module():
    """Load `worker/gateway.py` as a module, since it is not a package."""
    spec = importlib.util.spec_from_file_location("byom_worker_gateway", WORKER / "gateway.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["byom_worker_gateway"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """No test touches the real ~/.deepseek."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home
