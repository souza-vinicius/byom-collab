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


# ---------------------------------------------------------------------------
# Supervisão com teto: parar de relançar quando não há heartbeat válido
# ---------------------------------------------------------------------------
async def test_supervisor_desiste_apos_cinco_falhas(monkeypatch, tmp_path):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    state = make_state()
    config.save_state(state)
    stop = asyncio.Event()
    calls: list[int] = []

    def process_factory(session, file, *, timeout, log_path):
        calls.append(1)
        return FakeProcess(returncode=1)

    monkeypatch.setattr("byom.watchdog_supervisor.RESTART_BACKOFF_MIN_S", 0.001)
    monkeypatch.setattr("byom.watchdog_supervisor.RESTART_BACKOFF_MAX_S", 0.001)
    supervisor = WatchdogSupervisor(
        state=state,
        cfg=config.Config(),
        process_factory=process_factory,
        heartbeat_probe=_no_heartbeat,
        check_interval_s=0.001,
    )

    # O `stop` nunca é sinalizado: o laço precisa terminar sozinho.
    await asyncio.wait_for(supervisor.run(stop), timeout=10)

    assert len(calls) == 5, "não pode haver relançamento depois do teto"
    assert supervisor.status.state == "gave_up"
    assert supervisor.status.failures == 5
    assert "watchdog desistiu após 5 reinícios" in (supervisor.status.last_error or "")
    assert "byom down" in (supervisor.status.last_error or "")
    assert stop.is_set() is False


async def test_heartbeat_valido_zera_o_contador(monkeypatch, tmp_path):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path))
    state = make_state()
    config.save_state(state)
    stop = asyncio.Event()
    procs: list[FakeProcess] = []

    def process_factory(session, file, *, timeout, log_path):
        # O segundo `colab exec` fica vivo e entrega heartbeat.
        process = FakeProcess(returncode=1) if len(procs) != 1 else FakeProcess()
        procs.append(process)
        return process

    async def heartbeat() -> float | None:
        if len(procs) >= 2:
            stop.set()
        return 1.0

    monkeypatch.setattr("byom.watchdog_supervisor.MAX_CONSECUTIVE_FAILURES", 2)
    monkeypatch.setattr("byom.watchdog_supervisor.RESTART_BACKOFF_MIN_S", 0.001)
    monkeypatch.setattr("byom.watchdog_supervisor.RESTART_BACKOFF_MAX_S", 0.001)
    supervisor = WatchdogSupervisor(
        state=state,
        cfg=config.Config(),
        process_factory=process_factory,
        heartbeat_probe=heartbeat,
        check_interval_s=0.001,
    )

    await asyncio.wait_for(supervisor.run(stop), timeout=10)

    assert supervisor.status.failures == 0, "heartbeat válido zera o contador"
    assert supervisor.status.state == "stopped"


async def test_restart_reason_limpa_contador_quando_ha_heartbeat():
    supervisor = WatchdogSupervisor(
        state=make_state(),
        cfg=config.Config(),
        process_factory=lambda *args, **kwargs: FakeProcess(),
        heartbeat_probe=_fresh_heartbeat,
    )
    supervisor.process = FakeProcess()
    supervisor.status.failures = 4
    supervisor.status.last_error = "`colab exec` exited with code 1"

    reason = await supervisor._restart_reason()

    assert reason is None
    assert supervisor.status.failures == 0
    assert supervisor.status.last_error is None


async def _fresh_heartbeat() -> float | None:
    return 1.0
