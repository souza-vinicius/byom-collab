"""Tests for the daemon's startup registration handshake."""

from __future__ import annotations

import asyncio
import logging
import os
import time

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


# -- 5.12: tarefa morta encerra o daemon, com motivo no log ------------------
async def test_tarefa_que_levanta_encerra_a_supervisao(monkeypatch, caplog):
    from byom import daemon

    stop = asyncio.Event()
    vivo = {"cancelada": False}

    async def falha():
        raise RuntimeError("túnel explodiu")

    async def fica_viva():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            vivo["cancelada"] = True
            raise

    with caplog.at_level(logging.ERROR, logger="byom.daemon"):
        inicio = time.monotonic()
        erros = await daemon._supervise({"tunnel": falha(), "proxy": fica_viva()}, stop)
        decorrido = time.monotonic() - inicio

    assert decorrido < 5
    assert stop.is_set()
    assert vivo["cancelada"] is True
    assert len(erros) == 1 and isinstance(erros[0], RuntimeError)
    assert "Task tunnel ended with RuntimeError" in caplog.text


async def test_stop_encerra_a_supervisao_sem_erros():
    from byom import daemon

    stop = asyncio.Event()

    async def serve():
        await asyncio.sleep(30)

    tarefa = asyncio.create_task(daemon._supervise({"proxy": serve()}, stop))
    await asyncio.sleep(0.05)
    stop.set()
    erros = await asyncio.wait_for(tarefa, timeout=5)

    assert erros == []


def test_porta_ocupada_remove_o_pid_do_state():
    from byom import daemon

    config.save_state(
        config.State(
            session="byom-x",
            endpoint="",
            nonce="n" * 32,
            created_at="t",
            daemon_pid=4242,
            daemon_start_ticks=99,
        )
    )

    daemon._clear_daemon_pid(config.load_state())

    atual = config.load_state()
    assert atual is not None
    assert atual.daemon_pid is None
    assert atual.daemon_start_ticks is None


def test_clear_daemon_pid_nao_toca_state_de_outra_sessao():
    from byom import daemon

    config.save_state(
        config.State(
            session="byom-nova", endpoint="", nonce="n" * 32, created_at="t", daemon_pid=7
        )
    )
    antiga = config.State(
        session="byom-antiga", endpoint="", nonce="o" * 32, created_at="t", daemon_pid=4242
    )

    daemon._clear_daemon_pid(antiga)

    atual = config.load_state()
    assert atual is not None
    assert atual.session == "byom-nova"
    assert atual.daemon_pid == 7
