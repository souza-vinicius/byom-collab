"""Tests for allocation-blocked handling in the spike's model phase."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from byom import colab, config

SPIKE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "spike.py"
SPEC = importlib.util.spec_from_file_location("byom_spike", SPIKE_PATH)
assert SPEC is not None and SPEC.loader is not None
spike = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spike)


def test_model_phase_defaults_to_t4():
    """The account used by the spike has no L4 quota, so the default is a T4."""
    assert spike.DEFAULT_GPU == "T4"


def test_gpu_without_quota_is_reported_as_blocked(monkeypatch, capsys):
    def reject_gpu(name: str, gpu: str) -> None:
        assert gpu == "T4"
        raise colab.ColabCliError(
            "backend rejected T4: account has no quota or entitlement for this accelerator"
        )

    monkeypatch.setattr(spike.colab, "new", reject_gpu)
    monkeypatch.setattr(
        spike.remote,
        "Remote",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not open SSH")),
    )
    spike.RESULTS.clear()

    spike.phase_model(config.Config(gpu="T4"), keep=False)

    assert spike.RESULTS == [
        (
            "H6",
            "BLOCKED",
            "T4 allocation blocked: backend rejected T4: account has no quota "
            "or entitlement for this accelerator",
        )
    ]
    assert "H6: BLOCKED" in capsys.readouterr().out
