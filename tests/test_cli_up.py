from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from byom import cli, config, opencode


def _save_active_state():
    config.save_state(
        config.State(
            session="byom-old",
            endpoint="old-endpoint",
            nonce="old-nonce",
            created_at="now",
        )
    )


def test_up_force_downs_before_sync_and_new_allocation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-256k")
    config.save_config(cfg)
    _save_active_state()
    events = []

    def down(_cfg, *, quiet=False):
        events.append("down")
        config.clear_state()
        return True

    def orphan_assignments():
        events.append("orphans")
        return []

    def sync(_cfg, explicit_path=None):
        events.append("sync")
        return opencode.SyncResult(
            path=explicit_path or tmp_path / "opencode.json",
            model="byom/qwen3.8-27b-256k",
            changed=False,
        )

    def allocate(*args, **kwargs):
        events.append("allocate")
        raise RuntimeError("mocked stop before allocation")

    class FakeRemote:
        def __init__(self, **kwargs):
            events.append("remote")

        def stop(self):
            events.append("remote-stop")

    monkeypatch.setattr(cli, "_down", down)
    monkeypatch.setattr(cli.colab, "orphan_assignments", orphan_assignments)
    monkeypatch.setattr(cli.opencode, "sync_config", sync)
    monkeypatch.setattr(cli.colab, "new", allocate)
    monkeypatch.setattr(cli.remote, "Remote", FakeRemote)

    result = CliRunner().invoke(cli.app, ["up", "--force"])

    assert result.exit_code == 1
    assert (
        events.index("down")
        < events.index("orphans")
        < events.index("sync")
        < events.index("allocate")
    )


def test_up_force_aborts_if_shutdown_cannot_be_confirmed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    _save_active_state()
    sync_called = False
    allocated = False

    def fail_down(_cfg, *, quiet=False):
        return False

    def sync(*args, **kwargs):
        nonlocal sync_called
        sync_called = True
        return opencode.SyncResult(tmp_path / "opencode.json", "byom/model", False)

    def allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(cli, "_down", fail_down)
    monkeypatch.setattr(cli.opencode, "sync_config", sync)
    monkeypatch.setattr(cli.colab, "new", allocate)

    result = CliRunner().invoke(cli.app, ["up", "--force"])

    assert result.exit_code == 1
    assert "no new session was created" in result.output
    assert sync_called is False
    assert allocated is False


def test_up_force_aborts_if_orphan_assignment_remains(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    _save_active_state()
    sync_called = False
    allocated = False

    def down(_cfg, *, quiet=False):
        config.clear_state()
        return True

    def orphan_assignments():
        return [
            cli.colab.SessionLine(
                name=None,
                endpoint="orphan-endpoint",
                hardware="A100",
                shape="",
                variant="",
            )
        ]

    def sync(*args, **kwargs):
        nonlocal sync_called
        sync_called = True
        return opencode.SyncResult(tmp_path / "opencode.json", "byom/model", False)

    def allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(cli, "_down", down)
    monkeypatch.setattr(cli.colab, "orphan_assignments", orphan_assignments)
    monkeypatch.setattr(cli.opencode, "sync_config", sync)
    monkeypatch.setattr(cli.colab, "new", allocate)

    result = CliRunner().invoke(cli.app, ["up", "--force"])

    assert result.exit_code == 1
    assert "orphan-endpoint" in result.output
    assert sync_called is False
    assert allocated is False


def test_up_without_force_still_refuses_registered_session(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    _save_active_state()
    down_called = False

    def down(_cfg, *, quiet=False):
        nonlocal down_called
        down_called = True
        return True

    monkeypatch.setattr(cli, "_down", down)
    result = CliRunner().invoke(cli.app, ["up"])

    assert result.exit_code == 1
    assert "Session 'byom-old' is already registered" in result.output
    assert down_called is False


def test_up_aborta_com_estado_corrompido_sem_alocar(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{invalido", encoding="utf-8")
    allocated = False

    def allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(cli.colab, "new", allocate)

    result = CliRunner().invoke(cli.app, ["up"])
    output = result.output + (getattr(result, "stderr", "") or "")

    assert result.exit_code == 1
    assert "cannot be read or parsed" in output
    assert allocated is False


def test_up_sem_force_bloqueia_orfao_antes_de_alocar(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    orphan = cli.colab.SessionLine(
        name=None,
        endpoint="orphan-endpoint",
        hardware="L4",
        shape="",
        variant="",
    )
    monkeypatch.setattr(cli.colab, "orphan_assignments", lambda: [orphan])
    allocated = False
    synced = False

    def allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True

    def sync(*args, **kwargs):
        nonlocal synced
        synced = True
        return opencode.SyncResult(tmp_path / "opencode.json", "byom/model", False)

    monkeypatch.setattr(cli.colab, "new", allocate)
    monkeypatch.setattr(cli.opencode, "sync_config", sync)

    result = CliRunner().invoke(cli.app, ["up"])

    assert result.exit_code == 1
    assert "orphan-endpoint" in result.output
    assert synced is False
    assert allocated is False


def test_start_daemon_entrega_pid_ao_filho_antes_do_exec(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    captured = {}

    class FakeProc:
        pid = 9999

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    assert cli._start_daemon() == 9999
    with monkeypatch.context() as scope:
        scope.setenv("BYOM_DAEMON_EXPECTED_PID", "")
        captured["preexec_fn"]()
        assert os.environ["BYOM_DAEMON_EXPECTED_PID"] == str(os.getpid())


class _FakeRemote:
    """Remote sem SSH: o teste cobre só a contabilidade da alocação."""

    instancias: list[_FakeRemote] = []

    def __init__(self, session, cfg):
        self.session = session
        self.cfg = cfg
        _FakeRemote.instancias.append(self)

    def start(self, forwards=(), on_progress=None):
        return None

    def stop(self):
        return None


def test_up_registra_estado_antes_do_status_e_encerra_pela_sessao(tmp_path, monkeypatch):
    """`colab status` só enriquece o endpoint; falhar nele não apaga o registro.

    Com o state.json gravado logo após `colab new`, o teardown consegue encerrar
    a VM pelo nome mesmo quando o CLI de status está quebrado.
    """
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    eventos: list[str] = []
    estado_existia_no_status: list[bool] = []
    paradas: list[str] = []

    def allocate(name, *args, **kwargs):
        eventos.append("new")

    def status_quebrado(name):
        estado_existia_no_status.append(config.state_path().exists())
        raise cli.colab.ColabCliError("timeout no `colab status`")

    def bootstrap(*args, **kwargs):
        eventos.append("bootstrap")
        raise RuntimeError("falha simulada depois da alocação")

    monkeypatch.setattr(cli, "_sync_external_configs", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_orphans", lambda: None)
    monkeypatch.setattr(cli.colab, "new", allocate)
    monkeypatch.setattr(cli.colab, "status_line", status_quebrado)
    monkeypatch.setattr(cli.colab, "stop", lambda name: paradas.append(name))
    # O Colab não confirma a liberação: o registro local precisa sobreviver.
    monkeypatch.setattr(cli.colab, "is_allocated", lambda name, endpoint="": True)
    monkeypatch.setattr(cli, "_bootstrap_worker", bootstrap)
    monkeypatch.setattr(cli.remote, "Remote", _FakeRemote)

    resultado = CliRunner().invoke(cli.app, ["up"])

    assert resultado.exit_code == 1
    assert estado_existia_no_status == [True]
    assert paradas and paradas[0].startswith("byom-")
    estado = config.load_state()
    assert estado is not None
    assert estado.session == paradas[0]
    assert estado.endpoint == ""


def test_up_encerra_pela_sessao_quando_interrompido(tmp_path, monkeypatch):
    """Ctrl+C durante o bootstrap não pode deixar a VM alocada."""
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    paradas: list[str] = []

    def bootstrap(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_sync_external_configs", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_orphans", lambda: None)
    monkeypatch.setattr(cli.colab, "new", lambda name, *a, **k: None)
    monkeypatch.setattr(
        cli.colab,
        "status_line",
        lambda name: cli.colab.SessionLine(
            name=name, endpoint="m-s-abc", hardware="L4", shape="STANDARD", variant="GPU"
        ),
    )
    monkeypatch.setattr(cli.colab, "stop", lambda name: paradas.append(name))
    monkeypatch.setattr(cli.colab, "is_allocated", lambda name, endpoint="": False)
    monkeypatch.setattr(cli, "_bootstrap_worker", bootstrap)
    monkeypatch.setattr(cli.remote, "Remote", _FakeRemote)

    resultado = CliRunner().invoke(cli.app, ["up"])

    # O CliRunner converte Ctrl+C em saída != 0; o essencial é a VM encerrada.
    assert resultado.exit_code != 0
    assert len(paradas) == 1
    assert paradas[0].startswith("byom-")
    assert config.load_state() is None
