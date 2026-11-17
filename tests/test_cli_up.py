from __future__ import annotations

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
