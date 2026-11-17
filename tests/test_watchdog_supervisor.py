from __future__ import annotations

import asyncio

from byom import config
from byom.watchdog_supervisor import WATCHDOG_EXEC_TIMEOUT_S, WatchdogSupervisor


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.pid = 4321
        self.returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9


def make_state() -> config.State:
    return config.State(
        session="byom-watchdog-test",
        endpoint="/tun/m/watchdog",
        nonce="watchdog-nonce",
        created_at="test",
        daemon_pid=1234,
    )


async def test_supervisor_restarts_colab_exec_after_it_exits(monkeypatch, tmp_path):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    state = make_state()
    config.save_state(state)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls: list[tuple] = []
    first = FakeProcess(returncode=1)
    second = FakeProcess()

    def process_factory(session, file, *, timeout, log_path):
        calls.append((session, file, timeout, log_path))
        if len(calls) == 2:
            loop.call_soon_threadsafe(stop.set)
            return second
        return first

    monkeypatch.setattr("byom.watchdog_supervisor.RESTART_BACKOFF_MIN_S", 0.001)
    monkeypatch.setattr("byom.watchdog_supervisor.RESTART_BACKOFF_MAX_S", 0.001)
    supervisor = WatchdogSupervisor(
        state=state,
        cfg=config.Config(),
        process_factory=process_factory,
        heartbeat_probe=_no_heartbeat,
        check_interval_s=0.001,
    )

    await supervisor.run(stop)

    assert len(calls) == 2
    assert all(call[0] == state.session for call in calls)
    assert all(call[2] == WATCHDOG_EXEC_TIMEOUT_S for call in calls)
    assert all(call[1].name == "watchdog.py" for call in calls)
    assert all(call[3].name == "watchdog.log" for call in calls)
    assert second.terminated is True
    assert config.load_state().watchdog_pid is None


async def test_supervisor_restarts_before_gateway_watchdog_deadline():
    cfg = config.Config(watchdog_timeout_s=60)
    supervisor = WatchdogSupervisor(
        state=make_state(),
        cfg=cfg,
        process_factory=lambda *args, **kwargs: FakeProcess(),
        heartbeat_probe=_stale_heartbeat,
    )
    supervisor.process = FakeProcess()

    reason = await supervisor._restart_reason()

    assert reason == "No heartbeat for 30s"
    assert supervisor.restart_after_s < cfg.watchdog_timeout_s


async def _no_heartbeat() -> float | None:
    return None


async def _stale_heartbeat() -> float | None:
    return 30.0
