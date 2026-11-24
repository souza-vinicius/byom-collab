"""Tests for remote execution over multiplexed SSH."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from byom import config, remote


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    monkeypatch.setenv("BYOM_COLAB_BIN", "/usr/local/bin/colab")
    return config.Config()


class FakeSsh:
    """Replaces `ssh` with a configurable fake, without touching the network."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.master_returncode = 0
        self.run_returncode = 0
        self.run_stdout = b""
        self.run_stderr = b""
        self.create_socket = True
        self.raise_timeout = False
        self.socket: object = None
        self.master_log_text = "root@x: Permission denied (publickey).\n"

    def proc(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        return type(
            "Proc",
            (),
            {"returncode": returncode, "stdout": stdout, "stderr": stderr},
        )()

    def __call__(self, command, **kwargs):
        self.calls.append({"command": command, "kwargs": kwargs})
        if "-O" in command:  # -O check / -O exit
            return self.proc(0)
        if "-M" in command:  # master startup
            if self.master_returncode == 0 and self.create_socket:
                self.socket.parent.mkdir(parents=True, exist_ok=True)
                self.socket.touch()
            else:
                # Real `ssh` writes the reason to the `-E` file.
                log = Path(command[command.index("-E") + 1])
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(self.master_log_text, encoding="utf-8")
            return self.proc(self.master_returncode)
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout", 0))
        return self.proc(self.run_returncode, self.run_stdout, self.run_stderr)

    def master_calls(self) -> list[dict]:
        return [call for call in self.calls if "-M" in call["command"]]

    def client_calls(self) -> list[dict]:
        return [
            call
            for call in self.calls
            if "-O" not in call["command"] and "-M" not in call["command"]
        ]


@pytest.fixture
def fake_ssh(monkeypatch):
    fake = FakeSsh()
    monkeypatch.setattr(remote.subprocess, "run", fake)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    return fake


def build(cfg, session="byom-x") -> remote.Remote:
    return remote.Remote(session=session, cfg=cfg)


def live(cfg, fake_ssh, session="byom-x") -> remote.Remote:
    """Client with the master already considered active."""
    client = build(cfg, session)
    fake_ssh.socket = client.socket
    client.socket.parent.mkdir(parents=True, exist_ok=True)
    client.socket.touch()
    return client


# -- command building -----------------------------------------------------
def test_socket_lives_in_config_dir(cfg):
    """The name is short because Unix sockets have a path limit on macOS."""
    client = build(cfg)
    assert client.socket.parent == config.config_dir()
    assert client.socket.name == "cm-byom-x.sock"
    assert len("cm-byom-abcdef12.sock") < 32


def test_master_command_carries_identity_socket_and_forward(cfg):
    command = build(cfg).master_command([(4001, 8001)])

    assert command[0] == "ssh"
    assert "-M" in command
    assert "-fN" in command
    assert command[command.index("-E") + 1] == str(build(cfg).log_path)
    assert command[command.index("-S") + 1] == str(build(cfg).socket)
    assert command[command.index("-i") + 1] == str(cfg.ssh_key_path)
    assert "127.0.0.1:4001:127.0.0.1:8001" in command
    assert command[-3:] == ["-l", "root", "byom-x"]


def test_proxy_command_carries_the_same_key(cfg):
    """Without `-i`, the CLI uses the default key and authentication fails."""
    proxy = build(cfg).proxy_command()
    assert "--proxy-mode" in proxy
    assert str(cfg.ssh_key_path) in proxy
    assert "colab-sessions.json" in proxy


def test_client_command_does_not_repeat_the_forward(cfg):
    command = build(cfg).client_command("echo ok")
    assert "-L" not in command
    assert command[command.index("-S") + 1] == str(build(cfg).socket)
    assert command[-1] == "echo ok"


def test_control_command_uses_the_socket(cfg):
    command = build(cfg).control_command("exit")
    assert command[command.index("-O") + 1] == "exit"
    assert command[command.index("-S") + 1] == str(build(cfg).socket)


# -- start ----------------------------------------------------------------
def test_start_brings_up_the_master(cfg, fake_ssh):
    client = build(cfg)
    fake_ssh.socket = client.socket

    client.start([(4001, 8001)])

    assert client.socket.exists()
    assert len(fake_ssh.master_calls()) == 1


def test_start_does_not_capture_output(cfg, fake_ssh):
    """The detached master inherits descriptors: with pipes, the wait is eternal."""
    client = build(cfg)
    fake_ssh.socket = client.socket

    client.start()

    kwargs = fake_ssh.master_calls()[0]["kwargs"]
    assert "capture_output" not in kwargs
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_start_retries_and_explains_failure(cfg, fake_ssh):
    """A refused banner and a runtime 429 pass on a retry."""
    client = build(cfg)
    fake_ssh.socket = client.socket
    fake_ssh.master_returncode = 255

    with pytest.raises(remote.RemoteError) as excinfo:
        client.start()

    assert len(fake_ssh.master_calls()) == remote.START_ATTEMPTS
    assert "Permission denied" in str(excinfo.value)


def test_start_removes_the_stale_socket_when_giving_up(cfg, fake_ssh):
    client = build(cfg)
    fake_ssh.socket = client.socket
    fake_ssh.master_returncode = 255
    client.socket.parent.mkdir(parents=True, exist_ok=True)
    client.socket.touch()  # socket of a dead master

    with pytest.raises(remote.RemoteError):
        client.start()

    assert not client.socket.exists()


# -- run ------------------------------------------------------------------
def test_run_preserves_the_exit_code(cfg, fake_ssh):
    client = live(cfg, fake_ssh)
    fake_ssh.run_returncode = 3
    fake_ssh.run_stdout = b"output\n"

    result = client.run("exit 3", check=False)

    assert result.returncode == 3
    assert result.output == "output\n"
    assert not result.ok


def test_run_raises_error_on_nonzero_exit_code(cfg, fake_ssh):
    client = live(cfg, fake_ssh)
    fake_ssh.run_returncode = 2
    fake_ssh.run_stderr = b"boom\n"

    with pytest.raises(remote.RemoteError) as excinfo:
        client.run("exit 2")

    assert "exit code 2" in str(excinfo.value)
    assert "boom" in str(excinfo.value)


def test_run_without_active_master_refuses(cfg, fake_ssh):
    with pytest.raises(remote.RemoteError) as excinfo:
        build(cfg).run("echo ok")
    assert "is not active" in str(excinfo.value)


def test_run_bounds_the_wait(cfg, fake_ssh):
    """A background process can hold the output: the wait must be bounded."""
    client = live(cfg, fake_ssh)
    fake_ssh.raise_timeout = True

    with pytest.raises(remote.RemoteError) as excinfo:
        client.run("sleep 999", timeout=5)

    assert "timed out on the VM after 5s" in str(excinfo.value)


# -- upload ---------------------------------------------------------------
def test_upload_sends_over_stdin(cfg, fake_ssh, tmp_path):
    client = live(cfg, fake_ssh)
    payload = b"tarball contents"
    local = tmp_path / "worker.tar.gz"
    local.write_bytes(payload)

    client.upload(local, "/content/byom-worker.tar.gz")

    sent = fake_ssh.client_calls()[-1]
    assert sent["command"][-1] == "cat > /content/byom-worker.tar.gz"
    assert sent["kwargs"]["input"] == payload


# -- stop -----------------------------------------------------------------
def test_stop_stops_and_removes_the_socket(cfg, fake_ssh):
    client = live(cfg, fake_ssh)

    client.stop()

    exit_calls = [call for call in fake_ssh.calls if "-O" in call["command"]]
    assert exit_calls[0]["command"][exit_calls[0]["command"].index("-O") + 1] == "exit"
    assert not client.socket.exists()


def test_stop_without_master_is_noop(cfg, fake_ssh):
    build(cfg).stop()
    assert fake_ssh.calls == []


_real_validate_ssh_key = remote.validate_ssh_key


def test_validate_ssh_key_rejects_fake_and_accepts_real_key(tmp_path):
    import shutil

    if shutil.which("ssh-keygen") is None:
        pytest.skip("ssh-keygen not available")
    fake = tmp_path / "fake"
    fake.write_text("ssh-ed25519 AAAA test", encoding="utf-8")
    fake.chmod(0o600)
    assert "não é uma chave privada SSH válida" in (_real_validate_ssh_key(fake) or "")
    assert "não existe" in (_real_validate_ssh_key(tmp_path / "missing") or "")

    real = tmp_path / "real"
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(real)], check=True)
    assert _real_validate_ssh_key(real) is None


def test_start_reports_progress_for_each_attempt(cfg, fake_ssh):
    client = build(cfg)
    fake_ssh.socket = client.socket
    fake_ssh.master_returncode = 255
    messages: list[str] = []

    with pytest.raises(remote.RemoteError):
        client.start(on_progress=messages.append)

    assert messages[0] == f"tentativa 1 de {remote.START_ATTEMPTS}"
    assert any("nova tentativa em" in m for m in messages)
    assert messages[-1] == f"tentativa {remote.START_ATTEMPTS} de {remote.START_ATTEMPTS}"
