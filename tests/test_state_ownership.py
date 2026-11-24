"""Dono do `state.json`: nenhuma sessão apaga ou reescreve o registro de outra."""

from __future__ import annotations

import threading

import pytest
from typer.testing import CliRunner

from byom import cli, config


def _state(session: str, nonce: str = "n" * 32) -> config.State:
    return config.State(session=session, endpoint=f"m-s-{session}", nonce=nonce, created_at="t")


def test_clear_state_com_dono_errado_nao_apaga():
    config.save_state(_state("byom-nova"))

    with pytest.raises(config.StateFileError) as excinfo:
        config.clear_state(expected_session="byom-antiga")

    assert "byom-nova" in str(excinfo.value)
    assert config.state_path().exists()
    assert config.load_state().session == "byom-nova"


def test_clear_state_com_dono_certo_apaga():
    config.save_state(_state("byom-atual"))
    config.clear_state(expected_session="byom-atual")
    assert not config.state_path().exists()


def test_escritas_com_expected_session_errado_nao_alteram_o_arquivo():
    config.save_state(_state("byom-nova"))
    antes = config.state_path().read_text(encoding="utf-8")

    with pytest.raises(config.StateFileError):
        config.update_state(expected_session="byom-antiga", daemon_pid=1234)
    with pytest.raises(config.StateFileError):
        config.save_state(_state("byom-antiga"), expected_session="byom-atual")

    assert config.state_path().read_text(encoding="utf-8") == antes
    assert config.load_state().daemon_pid is None


def test_update_state_com_dono_certo_grava():
    config.save_state(_state("byom-atual"))

    atualizado = config.update_state(expected_session="byom-atual", daemon_pid=4321)

    assert atualizado.session == "byom-atual"
    assert config.load_state().daemon_pid == 4321


def test_down_antigo_nao_apaga_o_state_da_sessao_nova():
    """O `down` de uma sessão antiga não pode destruir o registro da nova."""
    config.save_state(_state("byom-antiga"))
    pronto = threading.Event()
    gravado = threading.Event()
    erros: list[BaseException] = []

    def up_novo():
        # Um `up` que já passou pela fase de preparação: sem state, registra o novo.
        pronto.wait(timeout=5)
        config.clear_state()
        config.save_state(_state("byom-nova"))
        gravado.set()

    def down_antigo():
        state = config.load_state()
        assert state is not None and state.session == "byom-antiga"
        pronto.set()
        assert gravado.wait(timeout=5)
        try:
            config.clear_state(expected_session=state.session)
        except config.StateFileError as exc:
            erros.append(exc)

    threads = [threading.Thread(target=up_novo), threading.Thread(target=down_antigo)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert erros, "o down antigo não pode limpar o state de outra sessão"
    assert config.load_state() is not None
    assert config.load_state().session == "byom-nova"


def test_duas_escritas_concorrentes_de_sessoes_diferentes_nao_se_misturam():
    config.save_state(_state("byom-a"))
    resultados: list[str] = []

    def escreve(session: str, outra: str) -> None:
        try:
            config.save_state(_state(session), expected_session=outra)
            resultados.append(f"ok:{session}")
        except config.StateFileError:
            resultados.append(f"recusado:{session}")

    threads = [
        threading.Thread(target=escreve, args=("byom-a", "byom-b")),
        threading.Thread(target=escreve, args=("byom-b", "byom-a")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    final = config.load_state()
    assert final is not None
    assert final.session in {"byom-a", "byom-b"}
    # Nenhuma escrita bem-sucedida pode ter trocado a sessão do arquivo.
    for resultado in resultados:
        if resultado.startswith("ok:byom-a"):
            assert final.session == "byom-a"


def test_down_da_cli_e_recusado_enquanto_ha_transacao_em_andamento(monkeypatch):
    config.save_state(_state("byom-viva"))
    alocacao: list[str] = []
    monkeypatch.setattr(cli.colab, "stop", lambda name: alocacao.append(name))

    with config.up_lock():
        resultado = CliRunner().invoke(cli.app, ["down"])

    assert resultado.exit_code != 0
    assert "está em andamento" in resultado.output
    assert alocacao == []
    assert config.load_state() is not None
