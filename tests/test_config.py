"""Testes do `config.toml` e do `state.json`."""

from __future__ import annotations

import json

from byom import config


def test_padroes_quando_nao_existe_config():
    cfg = config.load_config()
    assert cfg.model == "philbert440/Qwen3.8-27B-W4A16-AWQ"
    assert cfg.proxy_port == 4000
    assert cfg.tunnel_port == 4001
    assert cfg.gateway_port == 8001
    assert cfg.reasoning_effort == "medium"
    assert cfg.vllm_language_model_only is True
    assert cfg.vllm_enforce_eager is False
    assert cfg.gpu_high_mem is False


def test_round_trip_do_config():
    cfg = config.load_config()
    cfg.idle_timeout_min = 5
    cfg.vllm_enforce_eager = True
    cfg.model = "outro/modelo"
    cfg.gpu_high_mem = True
    config.save_config(cfg)

    loaded = config.load_config()
    assert loaded == cfg


def test_chaves_desconhecidas_sao_ignoradas():
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model = "x"\nchave_nova = 1\n', encoding="utf-8")
    cfg = config.load_config()
    assert cfg.model == "x"
    assert not hasattr(cfg, "chave_nova")


def test_estado_ausente():
    assert config.load_state() is None


def test_round_trip_do_estado():
    state = config.State(
        session="byom-1234abcd",
        endpoint="/tun/m/abc",
        nonce="deadbeef",
        created_at="2026-11-09T18:00:00+00:00",
    )
    config.save_state(state)
    assert config.load_state() == state


def test_estado_atualizado_com_pids():
    state = config.State(session="s", endpoint="e", nonce="n", created_at="t")
    config.save_state(state)
    state.daemon_pid = 4321
    state.watchdog_pid = 8765
    config.save_state(state)
    loaded = config.load_state()
    assert loaded is not None
    assert loaded.daemon_pid == 4321
    assert loaded.watchdog_pid == 8765


def test_estado_corrompido_nao_quebra():
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{invalido", encoding="utf-8")
    assert config.load_state() is None


def test_estado_sem_campos_obrigatorios_e_ignorado():
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"endpoint": "/tun/m/x"}), encoding="utf-8")
    assert config.load_state() is None


def test_clear_state_e_idempotente():
    config.save_state(config.State(session="s", endpoint="e", nonce="n", created_at="t"))
    config.clear_state()
    config.clear_state()
    assert config.load_state() is None


def test_escrita_atomica_nao_deixa_temporario():
    config.save_config(config.load_config())
    leftovers = list(config.config_dir().glob("*.tmp"))
    assert leftovers == []
