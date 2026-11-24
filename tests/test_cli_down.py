"""Tests for the shutdown flow, with a fake Colab CLI."""

from __future__ import annotations

import signal

import typer
from typer.testing import CliRunner

from byom import cli, colab, config, remote

runner = CliRunner()


class FakeRemote:
    """Replaces the multiplexed SSH connection without touching the network."""

    def __init__(self, *, session: str, cfg: config.Config, **kwargs) -> None:
        self.session = session
        self.cfg = cfg
        self.started = False
        self.stopped = False
        self.hardware = "NVIDIA L4"
        self.bootstrap_error: Exception | None = None
        self.uploaded: list[tuple[str, str]] = []
        self.commands: list[str] = []

    def start(self, *, forwards=()) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def gpu_name(self) -> str:
        return self.hardware

    def upload(self, local_path, remote_path: str) -> None:
        self.uploaded.append((str(local_path), remote_path))

    def run(self, command: str, **kwargs):
        self.commands.append(command)
        if self.bootstrap_error is not None:
            raise self.bootstrap_error
        return remote.RemoteResult(returncode=0, output="bootstrap ok\n")


def install_remote(monkeypatch, fake: FakeRemote) -> FakeRemote:
    monkeypatch.setattr(remote, "Remote", lambda **kwargs: fake)
    return fake


class FakeColab:
    """Replaces `byom.colab` without touching the real Colab."""

    def __init__(self, *, allocated_after_stop: bool = False, stop_fails: bool = False) -> None:
        self.allocated_after_stop = allocated_after_stop
        self.stop_fails = stop_fails
        self.stopped: list[str] = []
        self.sessions_calls = 0

    def stop(self, name: str) -> None:
        self.stopped.append(name)
        if self.stop_fails:
            raise colab.ColabCliError("simulated stop failure")

    def is_allocated(self, name: str, endpoint: str = "") -> bool:
        self.sessions_calls += 1
        return self.allocated_after_stop

    def find_by_name(self, name: str):
        return None

    def find_by_endpoint(self, endpoint: str):
        return None

    def status_line(self, name: str):
        return None

    def orphan_assignments(self):
        return []

    def unregistered_sessions(self, known=None):
        return []

    def usage(self) -> str:
        return "CCUs: 200"

    def ColabCliError(self):  # pragma: no cover - compatibility
        return colab.ColabCliError


def install_fake(monkeypatch, fake: FakeColab) -> None:
    for attribute in (
        "stop",
        "is_allocated",
        "find_by_name",
        "find_by_endpoint",
        "status_line",
        "orphan_assignments",
        "unregistered_sessions",
        "usage",
    ):
        monkeypatch.setattr(colab, attribute, getattr(fake, attribute))


def save_state() -> config.State:
    state = config.State(
        session="byom-abcdef12",
        endpoint="/tun/m/abc",
        nonce="nonce",
        created_at="2026-11-09T18:00:00+00:00",
        daemon_pid=None,
        watchdog_pid=None,
    )
    config.save_state(state)
    return state


def test_down_without_session_succeeds(monkeypatch):
    install_fake(monkeypatch, FakeColab())
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert "No active session" in result.output


def test_down_confirms_and_clears_state(monkeypatch):
    fake = FakeColab(allocated_after_stop=False)
    install_fake(monkeypatch, fake)
    save_state()

    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert fake.stopped == ["byom-abcdef12"]
    assert "Stopped" in result.output
    assert config.load_state() is None


def test_down_without_confirmation_keeps_state(monkeypatch):
    fake = FakeColab(allocated_after_stop=True)
    install_fake(monkeypatch, fake)
    save_state()

    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 1
    assert "Unknown" in result.output
    assert "compute units" in result.output
    # State is kept so a new verification can happen.
    assert config.load_state() is not None


def test_down_with_stop_failure_still_verifies(monkeypatch):
    fake = FakeColab(allocated_after_stop=False, stop_fails=True)
    install_fake(monkeypatch, fake)
    save_state()

    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert "Warning" in result.output
    assert config.load_state() is None


def test_repeated_down_is_safe(monkeypatch):
    install_fake(monkeypatch, FakeColab())
    result = runner.invoke(cli.app, ["down"])
    second = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert second.exit_code == 0


def test_up_refuses_registered_session(monkeypatch):
    install_fake(monkeypatch, FakeColab())
    save_state()

    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 1
    assert "is already registered" in result.output


def test_up_stops_session_on_hardware_mismatch(monkeypatch):
    fake = FakeColab(allocated_after_stop=False)
    install_fake(monkeypatch, fake)
    monkeypatch.setattr(colab, "new", lambda name, gpu="": None)
    client = install_remote(monkeypatch, FakeRemote(session="s", cfg=config.Config()))
    client.hardware = "Tesla T4"

    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 1
    assert "Allocated hardware" in result.output
    # The failure flow stops the allocated session and the SSH connection.
    assert fake.stopped
    assert client.stopped


def test_up_stops_session_when_bootstrap_fails(monkeypatch):
    fake = FakeColab(allocated_after_stop=False)
    install_fake(monkeypatch, fake)
    monkeypatch.setattr(colab, "new", lambda name, gpu="": None)
    client = install_remote(monkeypatch, FakeRemote(session="s", cfg=config.Config()))
    client.bootstrap_error = remote.RemoteError("remote command failed")

    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 1
    assert "Failed" in result.output
    assert fake.stopped
    assert client.stopped


def test_up_uses_ssh_to_upload_and_run_worker(monkeypatch):
    """The worker does not go through the kernel: `colab exec` may not return."""
    install_fake(monkeypatch, FakeColab())
    monkeypatch.setattr(colab, "new", lambda name, gpu="": None)
    client = install_remote(monkeypatch, FakeRemote(session="s", cfg=config.Config()))
    monkeypatch.setattr(cli, "_wait_ready", lambda cfg, timeout_s, *, expected_session=None: {})
    monkeypatch.setattr(cli, "_start_daemon", lambda: 9876)

    result = runner.invoke(cli.app, ["up"])

    assert result.exit_code == 0
    assert client.started
    assert client.uploaded
    assert client.uploaded[0][1] == cli.REMOTE_TARBALL
    assert client.commands == [
        f"mkdir -p /content/byom && tar xzf {cli.REMOTE_TARBALL} -C /content/byom "
        "&& bash /content/byom/bootstrap.sh"
    ]


def test_hardware_g4_matches_rtx_pro_6000():
    assert cli._hardware_matches("G4", "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert not cli._hardware_matches("G4", "NVIDIA A100-SXM4-80GB")


def test_up_passes_high_mem_to_colab(monkeypatch):
    cfg = config.Config(gpu="A100", gpu_high_mem=True)
    install_fake(monkeypatch, FakeColab())
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    captured = {}

    def new(name: str, gpu: str = "", *, high_mem: bool = False) -> None:
        captured.update(name=name, gpu=gpu, high_mem=high_mem)

    monkeypatch.setattr(colab, "new", new)
    client = install_remote(monkeypatch, FakeRemote(session="s", cfg=cfg))
    client.hardware = "NVIDIA A100-SXM4-80GB"
    monkeypatch.setattr(cli, "_wait_ready", lambda cfg, timeout_s, *, expected_session=None: {})
    monkeypatch.setattr(cli, "_start_daemon", lambda: 9876)

    result = runner.invoke(cli.app, ["up"])

    assert result.exit_code == 0
    assert captured["name"].startswith("byom-")
    assert captured["gpu"] == "A100"
    assert captured["high_mem"] is True


def test_down_stops_the_ssh_connection(monkeypatch):
    install_fake(monkeypatch, FakeColab(allocated_after_stop=False))
    save_state()
    client = install_remote(monkeypatch, FakeRemote(session="s", cfg=config.Config()))

    result = runner.invoke(cli.app, ["down"])

    assert result.exit_code == 0
    assert client.stopped


def test_status_warns_about_orphan_session(monkeypatch):
    """With no registered session, an orphan must still appear in status.

    This is the case of a creation that failed after allocating the VM.
    """
    install_fake(monkeypatch, FakeColab())
    orphan = colab.SessionLine(
        name=None,
        endpoint="m-s-usc1c8-35ulubuesk9zn",
        hardware="CPU",
        shape="Standard",
        variant="DEFAULT",
    )
    monkeypatch.setattr(colab, "orphan_assignments", lambda: [orphan])

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "m-s-usc1c8-35ulubuesk9zn" in result.output
    assert "No active session" in result.output
    assert "stop them from Colab" in result.output


def test_status_without_orphan_does_not_warn(monkeypatch):
    install_fake(monkeypatch, FakeColab())
    monkeypatch.setattr(colab, "orphan_assignments", list)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "no local record" not in result.output


def test_status_tolerates_orphan_listing_failure(monkeypatch):
    install_fake(monkeypatch, FakeColab())

    def failing():
        raise colab.ColabCliError("no network")

    monkeypatch.setattr(colab, "orphan_assignments", failing)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "No active session" in result.output


def test_daemon_status_rejects_another_session(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"session": "byom-antiga", "worker": {"state": "ready"}}

    monkeypatch.setattr(cli.httpx, "get", lambda *args, **kwargs: FakeResponse())

    payload = cli._daemon_status(config.Config(), expected_session="byom-atual")

    assert payload is not None
    assert payload["worker"]["state"] == "unavailable"
    assert payload["worker"]["error"] == "local daemon is serving a different session"


def test_build_worker_tar_includes_session_json(tmp_path):
    cfg = config.Config()
    state = config.State(session="s", endpoint="e", nonce="nonce-x", created_at="t")
    dest = tmp_path / "worker.tar.gz"
    cli._build_worker_tar(cfg, state, dest)

    import tarfile

    with tarfile.open(dest) as tar:
        names = set(tar.getnames())
        assert "session.json" in names
        assert {"bootstrap.sh", "gateway.py", "watchdog.py"} <= names
        payload = tar.extractfile("session.json")
        assert payload is not None
        import json

        data = json.loads(payload.read())
        assert data["nonce"] == "nonce-x"
        assert data["idle_timeout_min"] == cfg.idle_timeout_min
        assert data["max_job_min"] == cfg.max_job_min


def test_build_worker_tar_fails_without_files(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOM_WORKER_DIR", str(tmp_path))
    cfg = config.Config()
    state = config.State(session="s", endpoint="e", nonce="n", created_at="t")
    try:
        cli._build_worker_tar(cfg, state, tmp_path / "worker.tar.gz")
    except typer.Exit as exc:
        assert exc.exit_code == 1
    else:  # pragma: no cover - the test fails if it does not raise
        raise AssertionError("expected Exit when worker files are missing")


# ---------------------------------------------------------------------------
# PIDs locais: só matar o processo que o byom registrou
# ---------------------------------------------------------------------------
def _record_signals(monkeypatch) -> list[tuple[int, int]]:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    return sent


def _local_state(**overrides) -> config.State:
    values = {
        "session": "byom-abcdef12",
        "endpoint": "/tun/m/abc",
        "nonce": "nonce",
        "created_at": "2026-11-09T18:00:00+00:00",
    }
    values.update(overrides)
    return config.State(**values)


def test_pid_reciclado_nao_recebe_sinal(monkeypatch):
    """Depois de um reboot o PID pode ser de outro processo do usuário."""
    sent = _record_signals(monkeypatch)
    monkeypatch.setattr(cli.config, "process_start_ticks", lambda pid: 999)
    state = _local_state(daemon_pid=4242, daemon_start_ticks=555)

    cli._terminate_local(state)

    assert sent == []


def test_pid_registrado_recebe_sigterm(monkeypatch):
    sent: list[tuple[int, int]] = []
    alive = {"ticks": 555}

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        alive["ticks"] = 0  # encerrou depois do SIGTERM

    monkeypatch.setattr(cli.os, "kill", kill)
    monkeypatch.setattr(
        cli.config, "process_start_ticks", lambda pid: alive["ticks"] or None
    )
    state = _local_state(daemon_pid=4242, daemon_start_ticks=555)

    cli._terminate_local(state)

    assert sent == [(4242, signal.SIGTERM)]


def test_state_legado_com_cmdline_estranho_nao_mata(monkeypatch, capsys):
    sent = _record_signals(monkeypatch)
    monkeypatch.setattr(cli.config, "process_start_ticks", lambda pid: 777)
    monkeypatch.setattr(cli.config, "process_cmdline", lambda pid: "python3 outro-projeto.py")
    state = _local_state(daemon_pid=4242)

    cli._terminate_local(state)

    assert sent == []
    saida = capsys.readouterr().out
    assert "no signal sent" in saida
    assert "stale state after a reboot" in saida


def test_state_legado_com_cmdline_do_byom_mata(monkeypatch):
    sent: list[tuple[int, int]] = []
    alive = {"ticks": 777}

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        alive["ticks"] = 0

    monkeypatch.setattr(cli.os, "kill", kill)
    monkeypatch.setattr(cli.config, "process_start_ticks", lambda pid: alive["ticks"] or None)
    monkeypatch.setattr(
        cli.config, "process_cmdline", lambda pid: "python3 -m byom _daemon"
    )
    state = _local_state(daemon_pid=4242)

    cli._terminate_local(state)

    assert sent == [(4242, signal.SIGTERM)]


def test_sigterm_ignorado_promove_para_sigkill(monkeypatch):
    sent = _record_signals(monkeypatch)
    # O processo nunca desaparece do /proc: só o SIGKILL o encerra.
    monkeypatch.setattr(cli.config, "process_start_ticks", lambda pid: 555)
    monkeypatch.setattr(cli, "TERMINATE_GRACE_S", 0.01)
    state = _local_state(daemon_pid=4242, daemon_start_ticks=555)

    cli._terminate_local(state)

    assert sent == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


def test_pid_ausente_no_proc_nao_recebe_sinal(monkeypatch):
    sent = _record_signals(monkeypatch)
    monkeypatch.setattr(cli.config, "process_start_ticks", lambda pid: None)
    state = _local_state(watchdog_pid=4242, watchdog_start_ticks=555)

    cli._terminate_local(state)

    assert sent == []
