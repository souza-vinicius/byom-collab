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
