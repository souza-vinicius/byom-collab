"""Regressões de resiliência e clareza dos fluxos `init`, `up` e `down`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from byom import cli, config, opencode

runner = CliRunner()


@pytest.mark.parametrize(
    ("requested", "hardware", "esperado"),
    [
        ("l4", "L4", True),
        ("T4", "NVIDIA T4", True),
        ("g4", "RTX PRO 6000", True),
        ("T4", "A100 40GB", False),
        ("", "L4", False),
    ],
)
def test_hardware_matches_ignora_caixa(requested, hardware, esperado):
    assert cli._hardware_matches(requested, hardware) is esperado


def test_porta_do_tunel_ocupada_aborta_antes_de_alocar(monkeypatch, capsys):
    cfg = config.Config(proxy_port=49100, tunnel_port=49101)
    monkeypatch.setattr(cli, "_port_free", lambda port: port != cfg.tunnel_port)

    with pytest.raises(typer.Exit):
        cli._check_proxy_port(cfg)

    saida = capsys.readouterr().err
    assert str(cfg.tunnel_port) in saida
    assert "túnel" in saida


def test_timeout_do_worker_inclui_o_ultimo_motivo(monkeypatch):
    monkeypatch.setattr(cli, "READY_POLL_S", 0)
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "worker": {"state": "initializing", "error": "downloading weights"}
        },
    )

    with pytest.raises(TimeoutError, match="downloading weights"):
        cli._wait_ready(config.Config(), 0.01)


def test_up_sem_alocacao_devolve_o_harness_configurado(tmp_path, monkeypatch):
    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-256k")
    config.save_config(cfg)
    monkeypatch.setattr(cli.colab, "unregistered_sessions", lambda known=None: [])
    monkeypatch.setattr(
        cli.opencode,
        "sync_config",
        lambda cfg, explicit_path=None: opencode.SyncResult(
            path=tmp_path / "opencode.json", model="byom/qwen", changed=True
        ),
    )
    monkeypatch.setattr(
        cli.colab,
        "new",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("GPU quota exceeded")),
    )
    restaurados: list[str] = []
    monkeypatch.setattr(
        cli.harness_state,
        "restore",
        lambda: (restaurados.append("OpenCode") or ["OpenCode"], None),
    )

    result = runner.invoke(cli.app, ["up"])

    assert result.exit_code == 1
    assert "GPU quota exceeded" in result.output
    assert "[1/5]" in result.output
    assert "[3/5]" in result.output
    assert restaurados == ["OpenCode"]


def test_down_sem_confirmacao_devolve_o_harness(monkeypatch):
    config.save_state(
        config.State(
            session="byom-x",
            endpoint="old-endpoint",
            nonce="n" * 32,
            created_at="now",
        )
    )
    monkeypatch.setattr(cli.colab, "stop", lambda name: None)
    monkeypatch.setattr(cli.colab, "is_allocated", lambda name, endpoint="": True)
    monkeypatch.setattr(cli, "_terminate_local", lambda state: None)

    class FakeRemote:
        def __init__(self, **kwargs):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(cli.remote, "Remote", FakeRemote)
    monkeypatch.setattr(cli.harness_state, "restore", lambda: (["dsh"], None))

    result = runner.invoke(cli.app, ["down"])

    assert result.exit_code == 1
    assert "Harness config restored: dsh" in result.output
    assert "Unknown: could not confirm deallocation" in result.output
    assert config.load_state() is not None


def test_init_com_erro_de_colo_diz_o_que_aconteceu_e_como_corrop_dir(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/ssh")
    monkeypatch.setattr(cli.colab, "version", lambda: "Version: 0.7.5")
    monkeypatch.setattr(
        cli.colab,
        "run",
        lambda args, check=False: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"Error: GPU quota exceeded"
        ),
    )

    result = runner.invoke(cli.app, ["init"])

    assert result.exit_code == 1
    assert "GPU quota exceeded" in result.output
    assert "What to do:" in result.output
