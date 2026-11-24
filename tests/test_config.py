"""Testes do `config.toml` e do `state.json`."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from byom import catalog, config


def test_padroes_quando_nao_existe_config():
    cfg = config.load_config()
    assert cfg.model == "philbert440/Qwen3.8-27B-W4A16-AWQ"
    assert cfg.proxy_port == 4000
    assert cfg.tunnel_port == 4001
    assert cfg.gateway_port == 8001
    # Fonte: qwen3.8.md, "xhigh (default)" (checado em 2026-11-23).
    assert cfg.reasoning_effort == "xhigh"
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


def test_chaves_desconhecidas_sao_ignoradas(capsys):
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model = "x"\nchave_nova = 1\n', encoding="utf-8")
    cfg = config.load_config()
    assert cfg.model == "x"
    assert not hasattr(cfg, "chave_nova")
    # Aviso no stderr: não é mais silencioso.
    assert "chave_nova" in capsys.readouterr().err


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


def test_estado_corrompido_e_erro_explicito():
    """A damaged file is not the same as no session: never allow reallocation."""
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{invalido", encoding="utf-8")
    with pytest.raises(config.StateFileError) as excinfo:
        config.load_state()
    assert "cannot be read or parsed" in str(excinfo.value)


def test_estado_sem_campos_obrigatorios_e_rejeitado():
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"endpoint": "/tun/m/x"}), encoding="utf-8")
    with pytest.raises(config.StateFileError):
        config.load_state()


def test_pid_como_string_e_rejeitado():
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session": "s",
                "endpoint": "e",
                "nonce": "n",
                "created_at": "t",
                "daemon_pid": "4321",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(config.StateFileError):
        config.load_state()


def test_update_state_preserva_pid_de_outro_processo():
    state = config.State(session="s", endpoint="e", nonce="n", created_at="t")
    config.save_state(state)
    # Simulate the watchdog supervisor writing while `up` holds an old object.
    config.update_state(watchdog_pid=8765)
    config.update_state(daemon_pid=4321)
    loaded = config.load_state()
    assert loaded is not None
    assert loaded.daemon_pid == 4321
    assert loaded.watchdog_pid == 8765


def test_valida_fronteiras_do_config():
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in (
        ("proxy_port", 70000),
        ("idle_timeout_min", 0),
        ("gpu_memory_utilization", 1.5),
        ("reasoning_effort", "bogus"),
        ("reasoning_effort", 0.9),  # template aceita número, a fronteira não
    ):
        path.write_text(f'{key} = {json.dumps(value)}\n', encoding="utf-8")
        with pytest.raises(config.ConfigError):
            config.load_config()

    path.write_text('reasoning_effort = "xhigh"\n', encoding="utf-8")
    assert config.load_config().reasoning_effort == "xhigh"


@pytest.mark.parametrize("value", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_todos_os_valores_documentados_sao_aceitos(value):
    cfg = config.Config(reasoning_effort=value)
    config.validate_config(cfg)
    assert cfg.reasoning_effort == value


def test_reasoning_e_herdado_do_catalogo_pelo_perfil():
    expected = {
        "qwen3.8-27b-32k": "xhigh",
        "qwen3.8-27b-q4-gguf": "xhigh",
        "inkling-2-mini-q4-gguf": "high",
        "muse-glimmer-30b-q4-gguf": "high",
        # A página do Gemma 4 NÃO define reasoning_effort.
        "gemma-4-12b-q4-gguf": None,
        "qwen-image-2.1-q4km": None,
    }
    for profile_id, value in expected.items():
        cfg = config.Config()
        config.apply_profile(cfg, profile_id)
        assert cfg.reasoning_effort == value, profile_id

    # Nenhum perfil repete o valor: a herança vem do catálogo.
    for profile in config.available_profiles():
        assert "reasoning_effort" not in profile.settings
        if profile.catalog_model:
            assert profile.settings.get("reasoning_effort", "missing") == "missing"
            assert catalog.model_by_id(profile.catalog_model).reasoning_effort in (
                *config.REASONING_EFFORTS,
                None,
            )


def test_config_de_gemma_sem_valor_sobrevive_ao_round_trip():
    cfg = config.Config()
    config.apply_profile(cfg, "gemma-4-12b-q4-gguf")
    assert cfg.reasoning_effort is None
    config.save_config(cfg)
    assert "reasoning_effort" not in config.config_path().read_text(encoding="utf-8")
    loaded = config.load_config()
    assert loaded.reasoning_effort is None
    assert loaded.profile == "gemma-4-12b-q4-gguf"


def test_nonce_match_exige_valor_da_sessao():
    state = config.State(session="s", endpoint="e", nonce="n", created_at="t")
    assert config.nonce_matches({"nonce": "n"}, state) is True
    assert config.nonce_matches({"nonce": "outra"}, state) is False
    assert config.nonce_matches(None, state) is False


def test_clear_state_e_idempotente():
    config.save_state(config.State(session="s", endpoint="e", nonce="n", created_at="t"))
    config.clear_state()
    config.clear_state()
    assert config.load_state() is None


def test_escrita_atomica_nao_deixa_temporario():
    config.save_config(config.load_config())
    leftovers = list(config.config_dir().glob("*.tmp"))
    assert leftovers == []


def test_up_lock_recusa_aquisicao_concorrente_e_libera_no_fim():
    with config.up_lock(), pytest.raises(config.ConfigError) as exc, config.up_lock():
        pass
    assert "já existe um up em andamento" in str(exc.value)
    # A recusa não pode deixar o lock preso para o próximo `up`.
    with config.up_lock():
        pass


def test_up_lock_libera_quando_o_corpo_levanta_excecao():
    with pytest.raises(RuntimeError), config.up_lock():
        raise RuntimeError("falha simulada")
    with config.up_lock():
        pass


# ---------------------------------------------------------------------------
# Identidade do PID: start time do /proc
# ---------------------------------------------------------------------------
def test_process_start_ticks_le_o_processo_atual():
    import os

    ticks = config.process_start_ticks(os.getpid())
    assert ticks is not None and ticks > 0
    # Estável entre leituras: é o que permite detectar PID reciclado.
    assert config.process_start_ticks(os.getpid()) == ticks
    assert config.process_start_ticks(2**31 - 1) is None
    assert config.process_start_ticks(0) is None


def test_state_valida_start_ticks_positivos():
    state = config.State(
        session="s",
        endpoint="e",
        nonce="n",
        created_at="t",
        daemon_pid=10,
        daemon_start_ticks=1234,
        watchdog_pid=11,
        watchdog_start_ticks=5678,
    )
    config.validate_state(state)

    for campo in ("daemon_start_ticks", "watchdog_start_ticks"):
        for valor in (0, -1, "123", True):
            quebrado = config.State(session="s", endpoint="e", nonce="n", created_at="t")
            setattr(quebrado, campo, valor)
            with pytest.raises(config.StateFileError):
                config.validate_state(quebrado)


# -- 5.5: max_model_len não pode passar do contexto do catálogo ----------------
def test_max_model_len_acima_do_catalogo_e_recusado_na_carga():
    """400.000 num perfil de 256K não pode carregar em silêncio."""
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'profile = "qwen3.8-27b-256k"\nmax_model_len = 400000\n', encoding="utf-8"
    )

    with pytest.raises(config.ConfigError) as excinfo:
        config.load_config()

    mensagem = str(excinfo.value)
    assert "400000" in mensagem
    assert "262144" in mensagem
    assert "qwen3.8-27b-awq:q4-awq" in mensagem


def test_max_model_len_igual_ao_catalogo_e_aceito():
    cfg = config.Config(max_model_len=262_144, profile="qwen3.8-27b-256k")
    config.validate_context_limit(cfg, "qwen3.8-27b-awq", "q4-awq")
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.dumps_config(cfg), encoding="utf-8")
    assert config.load_config().max_model_len == 262_144


def test_validacao_de_contexto_ignora_variante_sem_contexto_documentado():
    """Imagem não tem janela autoregressiva: sentinela 1 não é limitada."""
    cfg = config.Config(max_model_len=1, profile="qwen-image-2.1-q4km")
    config.validate_context_limit(cfg, "qwen-image-2.1", "q4km")
    assert catalog.context_limit("qwen-image-2.1", "q4km") == 0


def test_perfis_embarcados_respeitam_o_contexto_do_catalogo():
    for profile in config.available_profiles():
        if not profile.catalog_model:
            continue
        limite = catalog.context_limit(profile.catalog_model, profile.catalog_variant)
        assert not limite or profile.settings["max_model_len"] <= limite, profile.id


# -- 5.8: escrita atômica preserva symlink e não cruza sistemas de arquivos ---
def test_atomic_write_preserva_symlink_de_dotfile(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text("antigo\n", encoding="utf-8")
    real.chmod(0o640)
    link = tmp_path / "config.toml"
    link.symlink_to(real)

    config.atomic_write_text(link, "novo\n")

    assert link.is_symlink(), "o link do dotfile não pode virar arquivo comum"
    assert os.readlink(link) == str(real)
    assert real.read_text(encoding="utf-8") == "novo\n"
    assert stat.S_IMODE(real.stat().st_mode) == 0o640


def test_atomic_write_cria_tmp_no_diretorio_real_do_alvo(tmp_path, monkeypatch):
    real = tmp_path / "dotfiles" / "arquivo.txt"
    real.parent.mkdir()
    real.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    vistos: list[str] = []
    original = tempfile.mkstemp

    def fake_mkstemp(*args, **kwargs):
        vistos.append(kwargs.get("dir"))
        return original(*args, **kwargs)

    monkeypatch.setattr(config.tempfile, "mkstemp", fake_mkstemp)

    config.atomic_write_text(link, "y")

    assert vistos == [str(real.parent)]


def test_atomic_write_em_alvo_de_outro_sistema_de_arquivos(tmp_path):
    """Link para outro FS não pode falhar com EXDEV."""
    outro = Path("/dev/shm")
    if not outro.is_dir() or not os.access(outro, os.W_OK):
        pytest.skip("/dev/shm indisponível neste ambiente")
    try:
        pasta = Path(tempfile.mkdtemp(dir=str(outro)))
    except OSError:  # pragma: no cover - ambiente sem tmpfs gravável
        pytest.skip("sem tmpfs gravável para simular outro FS")
    try:
        if os.stat(pasta).st_dev == os.stat(tmp_path).st_dev:
            pytest.skip("/dev/shm está no mesmo dispositivo")
        alvo = pasta / "config.toml"
        alvo.write_text("antigo\n", encoding="utf-8")
        link = tmp_path / "config.toml"
        link.symlink_to(alvo)

        config.atomic_write_text(link, "novo\n")

        assert alvo.read_text(encoding="utf-8") == "novo\n"
        assert link.is_symlink()
    finally:
        for item in pasta.iterdir():
            item.unlink()
        pasta.rmdir()


# -- 5.10: chave desconhecida é avisada e preservada, não apagada ------------
def test_round_trip_preserva_chaves_desconhecidas(capsys):
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'model = "x"\nmax_model_len = 16384\nchave_nova = 1\n'
        'outra = "valor"\nlista = ["a", "b"]\n',
        encoding="utf-8",
    )
    cfg = config.load_config()
    aviso = capsys.readouterr().err
    assert "chave_nova, lista, outra" in aviso
    assert "preservadas" in aviso

    cfg.proxy_port = 4123
    config.save_config(cfg)
    texto = path.read_text(encoding="utf-8")

    assert 'chave_nova = 1' in texto
    assert 'outra = "valor"' in texto
    assert 'lista = ["a", "b"]' in texto
    assert "proxy_port = 4123" in texto
    # E o round-trip continua estável: recarregar não perde nada.
    recarregado = config.load_config()
    assert recarregado.proxy_port == 4123
    assert 'chave_nova = 1' in config.dumps_config(recarregado)


def test_chave_desconhecida_em_tabela_recusa_a_gravacao():
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model = "x"\n[custom]\nflag = true\n', encoding="utf-8")
    cfg = config.load_config()

    with pytest.raises(config.ConfigError) as excinfo:
        config.save_config(cfg)

    assert "custom" in str(excinfo.value)
    assert "não reescreve tabelas" in str(excinfo.value)


def test_tui_mostra_o_aviso_das_chaves_desconhecidas():
    from byom import tui

    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model = "x"\nchave_nova = 1\n', encoding="utf-8")
    con = tui.criar_console(width=100, height=30, record=True)

    cfg = tui._carregar_config(con)

    assert cfg.model == "x"
    assert "chave_nova" in con.export_text()
