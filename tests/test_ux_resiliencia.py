"""Regressões de resiliência e clareza dos fluxos `init`, `up` e `down`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from byom import cli, colab, config, opencode, remote, tui

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


@pytest.mark.parametrize(
    ("exc_cls", "msg", "esperado_acao"),
    [
        (
            colab.ColabBackendUnavailable,
            "Service Unavailable",
            "not a compute-unit balance issue",
        ),
        (
            colab.ColabAllocationRefused,
            "Allocation refused",
            "active session or usage limits",
        ),
        (
            colab.ColabAcceleratorRejected,
            "Backend rejected accelerator",
            "lacks quota or entitlement",
        ),
    ],
)
def test_up_cli_exibe_orientacao_acionavel_em_falhas_do_colab(
    tmp_path, monkeypatch, exc_cls, msg, esperado_acao
):
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
        lambda *args, **kwargs: (_ for _ in ()).throw(exc_cls(msg)),
    )
    restaurados: list[str] = []
    monkeypatch.setattr(
        cli.harness_state,
        "restore",
        lambda: (restaurados.append("OpenCode") or ["OpenCode"], None),
    )

    result = runner.invoke(cli.app, ["up"])

    assert result.exit_code == 1
    assert "Failed:" in result.output
    assert "What to do:" in result.output
    assert esperado_acao in result.output
    assert restaurados == ["OpenCode"]


def test_up_cli_interrompido_informa_limpeza_e_resumo_inclui_webui(monkeypatch):
    cfg = config.Config()
    config.save_config(cfg)

    class FakeRemote:
        def __init__(self, session, cfg):
            self.session = session

        def start(self, forwards=(), on_progress=None):
            return None

        def stop(self):
            return None

        def gpu_name(self):
            return "NVIDIA L4"

        def upload(self, *args, **kwargs):
            return None

        def run(self, *args, **kwargs):
            return remote.RemoteResult(returncode=0, output="ok")

    monkeypatch.setattr(cli, "_sync_external_configs", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_orphans", lambda: None)
    monkeypatch.setattr(cli.remote, "Remote", FakeRemote)
    monkeypatch.setattr(colab, "new", lambda *a, **k: None)
    monkeypatch.setattr(
        colab,
        "status_line",
        lambda name: colab.SessionLine(
            name=name, endpoint="m-s-abc", hardware="L4", shape="STANDARD", variant="GPU"
        ),
    )
    monkeypatch.setattr(cli, "_bootstrap_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_register_daemon", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_wait_ready", lambda *a, **k: {"worker": {"state": "ready"}})

    ok_res = runner.invoke(cli.app, ["up"])
    assert ok_res.exit_code == 0
    assert "Worker ready" in ok_res.output
    assert "WebUI:    http://127.0.0.1:8080 (`byom web`)" in ok_res.output

    config.clear_state()
    paradas: list[str] = []
    monkeypatch.setattr(colab, "stop", lambda name: paradas.append(name))
    monkeypatch.setattr(colab, "is_allocated", lambda name, endpoint="": False)
    monkeypatch.setattr(
        cli,
        "_bootstrap_worker",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    int_res = runner.invoke(cli.app, ["up"])
    assert int_res.exit_code != 0
    assert "Interrupted (Ctrl+C): stopping session and cleaning up local processes..." in (
        int_res.output
    )
    assert len(paradas) == 1


def test_cli_help_organiza_paineis_e_exemplos_sem_vazar_markup():
    root_res = runner.invoke(cli.app, ["--help"])
    assert root_res.exit_code == 0
    for painel in (
        "Configuração e Catálogo",
        "Sessão Colab e Ciclo de Vida",
        "Chat e Frontend Web",
    ):
        assert painel in root_res.output
    assert "[bold]" not in root_res.output
    assert "[cyan]" not in root_res.output

    for cmd in (
        "init",
        "profiles",
        "gpus",
        "models",
        "configure",
        "up",
        "status",
        "down",
        "setup",
        "chat",
        "web",
        "uninstall",
    ):
        cmd_res = runner.invoke(cli.app, [cmd, "--help"])
        assert cmd_res.exit_code == 0
        assert "Ex.:" in cmd_res.output
        assert "[bold]" not in cmd_res.output


def test_status_tui_exibe_modelo_api_byom_e_open_webui(monkeypatch):
    cfg = config.Config()
    config.save_config(cfg)
    config.save_state(
        config.State(
            session="byom-tui1",
            endpoint="m-s-tui",
            nonce="n" * 32,
            created_at="2026-11-28T10:00:00+00:00",
        )
    )
    monkeypatch.setattr(cli.colab, "orphan_assignments", lambda: [])
    monkeypatch.setattr(
        cli.colab,
        "status_line",
        lambda name: colab.SessionLine(
            name=name, endpoint="m-s-tui", hardware="L4", shape="STANDARD", variant="GPU"
        ),
    )
    monkeypatch.setattr(cli.colab, "usage", lambda: "")
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda c, expected_session=None: {
            "session": expected_session,
            "worker": {"state": "ready"},
            "tunnel": {"state": "connected", "reconnects": 0, "latency_ms": 15},
        },
    )
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")

    monkeypatch.setenv("NO_COLOR", "1")
    rec_estreito = tui.criar_console(width=60, height=24, record=True)
    assert rec_estreito.no_color is True
    assert tui._executar_status(rec_estreito, cli, tui.ler_estado_local()) == "menu"

    saida = rec_estreito.export_text(clear=False)
    assert "Modelo ativo" in saida
    assert "API Byom" in saida
    assert "http://127.0.0.1:4000/v1" in saida
    assert "Open WebUI" in saida
    assert "http://127.0.0.1:8080" in saida
