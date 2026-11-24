"""Tests for the daemon's startup registration handshake."""

from __future__ import annotations

import os

from byom import config, daemon


def make_state() -> config.State:
    return config.State(
        session="byom-race-test",
        endpoint="/tun/m/race",
        nonce="race-nonce",
        created_at="test",
    )


async def test_pid_from_environment_wins_before_state_is_written(monkeypatch, tmp_path):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    state = make_state()
    config.save_state(state)
    # `byom up` has not persisted daemon_pid yet; the child already knows it.
    monkeypatch.setenv("BYOM_DAEMON_EXPECTED_PID", str(os.getpid()))

    assert await daemon._load_registered_state(timeout_s=0.5) == state


async def test_without_hint_daemon_waits_for_state_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    config.save_state(make_state())
    monkeypatch.delenv("BYOM_DAEMON_EXPECTED_PID", raising=False)

    assert await daemon._load_registered_state(timeout_s=0.05) is None


# ---------------------------------------------------------------------------
# Logs: rotação e `basicConfig` efetivo
# ---------------------------------------------------------------------------
def test_daemon_log_recebe_linha_info(tmp_path, monkeypatch):
    """`cli.main()` já configurou o root: sem `force=True` era no-op."""
    import logging

    from byom import config, daemon

    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    root = logging.getLogger()
    anteriores = list(root.handlers)
    nivel = root.level
    try:
        handler = daemon.configure_logging()
        logging.getLogger("byom.daemon").info("Daemon started (teste)")
        handler.flush()
        handler.close()
        texto = (config.config_dir() / "logs" / "daemon.log").read_text(encoding="utf-8")
    finally:
        root.handlers.clear()
        root.handlers.extend(anteriores)
        root.setLevel(nivel)

    assert "Daemon started (teste)" in texto
    assert "INFO" in texto
    assert "byom.daemon" in texto


def test_daemon_log_rotaciona(tmp_path, monkeypatch):
    import logging

    from byom import config, daemon

    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(daemon, "LOG_MAX_BYTES", 512)
    root = logging.getLogger()
    anteriores = list(root.handlers)
    nivel = root.level
    try:
        handler = daemon.configure_logging()
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 512
        assert handler.backupCount == daemon.LOG_BACKUP_COUNT == 2
        logger = logging.getLogger("byom.daemon")
        for index in range(50):
            logger.info("linha %s com recheio para forçar a rotação", index)
        handler.flush()
        handler.close()
    finally:
        root.handlers.clear()
        root.handlers.extend(anteriores)
        root.setLevel(nivel)

    logs = config.config_dir() / "logs"
    assert (logs / "daemon.log").exists()
    assert (logs / "daemon.log.1").exists()
    assert (logs / "daemon.log").stat().st_size <= 512
