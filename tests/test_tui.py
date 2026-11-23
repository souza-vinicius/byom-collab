"""Testes da camada interativa (TUI), formatadores e mensagens centralizadas."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from byom import cli, colab, config, tui
from byom import mensagens as M

runner = CliRunner()


def _saida(resultado) -> str:
    return (getattr(resultado, "stdout", "") or "") + (getattr(resultado, "stderr", "") or "")


# ---------------------------------------------------------------------------
# Contratos de entrada da CLI / TUI
# ---------------------------------------------------------------------------


def test_sem_argumentos_fora_de_tty_mostra_ajuda_e_mantem_exit_2():
    resultado = runner.invoke(cli.app, [])
    assert resultado.exit_code == 2
    assert "Usage:" in _saida(resultado)


def test_no_tui_forca_modo_cli():
    resultado = runner.invoke(cli.app, ["--no-tui"])
    assert resultado.exit_code == 2
    assert "Usage:" in _saida(resultado)


def test_tui_sem_tty_tem_erro_explicativo():
    resultado = runner.invoke(cli.app, ["--tui"])
    assert resultado.exit_code == 2
    saida = _saida(resultado)
    assert "terminal interativo" in saida
    assert "Causa provável" in saida
    assert "O que fazer" in saida


def test_tui_e_no_tui_juntos_prevalece_no_tui(monkeypatch):
    chamados: list[object] = []
    monkeypatch.setattr(tui, "mostrar_menu", lambda modulo: chamados.append(modulo))
    resultado = runner.invoke(cli.app, ["--tui", "--no-tui"])
    assert resultado.exit_code == 2
    assert chamados == []


def test_com_tty_sem_subcomando_abre_menu(monkeypatch):
    chamados: list[object] = []
    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(tui, "mostrar_menu", lambda modulo: chamados.append(modulo))
    resultado = runner.invoke(cli.app, [])
    assert resultado.exit_code == 0
    assert chamados == [cli]


def test_subcomando_nao_abre_menu(monkeypatch):
    def falha(modulo):  # pragma: no cover - só deve ser chamada se houver bug
        raise AssertionError("menu não deve abrir com subcomando")

    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(tui, "mostrar_menu", falha)
    resultado = runner.invoke(cli.app, ["profiles"])
    assert resultado.exit_code == 0
    assert "qwen" in _saida(resultado).lower()


# ---------------------------------------------------------------------------
# Mensagens, glossário e formatadores pt-BR
# ---------------------------------------------------------------------------


def test_mensagem_de_erro_tem_tres_partes():
    texto = M.ERRO.format(falha="falha", causa="causa", acao="ação")
    assert "O que aconteceu:" in texto
    assert "Causa provável:" in texto
    assert "O que fazer:" in texto


def test_mensagem_de_resultado_vazio_explica_origem():
    falha, causa, acao = M.ERROS_COMUNS["vazio"]
    texto = M.ERRO.format(
        falha=falha.format(origem="profiles"),
        causa=causa.format(origem="profiles"),
        acao=acao.format(origem="profiles"),
    )
    assert "profiles" in texto
    assert "Verifique" in texto


def test_comando_equivalente_inclui_flags():
    assert tui._comando_equivalente("models", {"gpu": "L4"}) == "byom models --gpu L4"
    assert (
        tui._comando_equivalente("configure", {"profile_id": "inkling-2-mini-q4-gguf"})
        == "byom configure --profile inkling-2-mini-q4-gguf"
    )
    assert tui._comando_equivalente("up", {"force": True}) == "byom up --force"


def test_todas_as_acoes_have_ajuda_e_proximo_passo():
    assert set(M.ACOES) == set(M.AJUDA_ACOES)
    assert set(M.ACOES) == set(M.ANTES)
    assert set(M.ACOES) == set(M.PROXIMO)
    assert set(M.ACOES) == set(M.PROXIMO_ACAO)
    for destino_id, _rotulo in M.PROXIMO_ACAO.values():
        assert destino_id in M.ACOES


def test_formatadores_ptbr(isolated_home):
    assert M.formatar_numero(32768) == "32.768"
    assert M.formatar_numero(1_000_000) == "1.000.000"
    assert M.formatar_numero(17.6) == "17,6"
    assert M.formatar_gib(16) == "16 GiB"
    assert M.formatar_gib(16, aproximado=True) == "~16 GiB"
    assert M.formatar_gib(17.6) == "17,6 GiB"

    assert M.formatar_contexto(32_768) == "32K"
    assert M.formatar_contexto(65_536) == "64K"
    assert M.formatar_contexto(131_072) == "128K"
    assert M.formatar_contexto(262_144) == "256K"
    assert M.formatar_contexto(1_000_000) == "1M"
    assert M.formatar_contexto(1_048_576) == "1M"
    assert M.formatar_contexto(1, "image") == "não se aplica"
    assert M.formatar_contexto(0, "image") == "não se aplica"
    assert M.formatar_contexto_detalhe(32_768) == "32K (32.768 tokens)"
    assert "não se aplica" in M.formatar_contexto_detalhe(1, "image")

    assert M.formatar_data_ptbr("2026-11-23") == "23/11/2026"
    assert M.formatar_data_ptbr("2026-11-23T19:05:00+00:00") == "23/11/2026 às 19:05"

    caminho = isolated_home / ".ssh" / "byom_ed25519"
    assert M.encurtar_caminho(caminho) == "~/.ssh/byom_ed25519"
    assert M.encurtar_caminho("/usr/bin/ssh") == "/usr/bin/ssh"

    assert M.traduzir_raciocinio("xhigh") == "muito alto"
    assert M.traduzir_raciocinio("high") == "alto"
    assert M.traduzir_raciocinio(None, tem_thinking_prompt=True) == "automático"
    assert M.traduzir_raciocinio(None, tem_thinking_prompt=False) == "não informado"
    assert M.traduzir_raciocinio(None, runtime="image") == "não se aplica"


# ---------------------------------------------------------------------------
# Estado local e próximo passo recomendado
# ---------------------------------------------------------------------------


def test_proximo_passo_recomendado_por_estado():
    base = dict(
        config_existe=False,
        chave_ssh_existe=False,
        ambiente_configurado=False,
        perfil_id="",
        perfil_nome="",
        gpu_id="L4",
        gpu_rotulo="L4",
        sessao_ativa=False,
        sessao_nome="",
        sessao_criada_em="",
    )
    assert tui.proximo_passo_recomendado(tui.EstadoLocal(**base)) == "init"

    sem_perfil = {
        **base,
        "config_existe": True,
        "chave_ssh_existe": True,
        "ambiente_configurado": True,
    }
    assert tui.proximo_passo_recomendado(tui.EstadoLocal(**sem_perfil)) == "configure"

    pronto = {**sem_perfil, "perfil_id": "qwen3.8-27b-32k", "perfil_nome": "Qwen3.8 27B (32k)"}
    assert tui.proximo_passo_recomendado(tui.EstadoLocal(**pronto)) == "up"

    ativo = {**pronto, "sessao_ativa": True, "sessao_nome": "byom-1234"}
    assert tui.proximo_passo_recomendado(tui.EstadoLocal(**ativo)) == "status"


def test_ler_estado_local_nao_chama_rede_nem_colab(monkeypatch):
    def proibido(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Barra de estado não pode chamar colab/rede")

    monkeypatch.setattr(colab, "run", proibido)
    monkeypatch.setattr(colab, "status_line", proibido)
    monkeypatch.setattr(colab, "usage", proibido)
    estado = tui.ler_estado_local()
    assert estado.ambiente_configurado is False
    assert estado.sessao_ativa is False


# ---------------------------------------------------------------------------
# Renderização por largura/altura, NO_COLOR e busca por resíduos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("largura", [70, 80, 100, 120])
def test_renderizacao_responsiva_e_sem_residuos(largura: int):
    rec = tui.criar_console(width=largura, height=24, record=True)
    estado = tui.ler_estado_local()

    tui.renderizar_menu_estatico(rec, estado)
    itens = tui.renderizar_perfis(rec, estado)
    assert len(itens) == len(config.available_profiles())
    for item in itens:
        tui.renderizar_detalhe_perfil(rec, item, estado)
    tui.renderizar_gpus(rec, estado)
    tui.renderizar_modelos_por_gpu(rec, "L4", estado)

    texto = rec.export_text(clear=False)

    if largura >= 80:
        assert "…" not in texto

    for termo_proibido in (
        "reasoning not documented",
        "not documented",
        "Full native",
        "Lower-context",
        "Ready",
        "Version:",
        "already exists",
        "sentinela",
        "1 tokens",
        "1 token",
        "Duração: 0.0s",
        "nao ",
    ):
        assert termo_proibido not in texto, (
            f"Resíduo proibido em largura {largura}: {termo_proibido}"
        )

    assert not re.findall(r"\b\d{1,3},\d{3}\b", texto)


def test_altura_24_mostra_dica_paginacao():
    rec = tui.criar_console(width=80, height=24, record=True)
    tui.renderizar_perfis(rec, tui.ler_estado_local())
    assert "Terminal compacto (80×24)" in rec.export_text(clear=False)


def test_no_color_e_fallback_ascii(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    con = tui.criar_console(width=100, height=30)
    assert con.no_color is True
    tui.renderizar_gpus(con, tui.ler_estado_local())
    captured = capsys.readouterr().out
    assert "\x1b[" not in captured

    fake_con = SimpleNamespace(encoding="ascii")
    assert tui.obter_simbolos(fake_con) == tui.ASCII_SIMBOLOS


# ---------------------------------------------------------------------------
# Fluxo simulado completo e erros em 3 partes
# ---------------------------------------------------------------------------


def test_down_recusa_confirmacao_nao_chama_comando(monkeypatch):
    monkeypatch.setattr(tui, "_confirmar", lambda chave, **kwargs: False)
    with pytest.raises(KeyboardInterrupt):
        tui._opcoes_down()


def test_fluxo_simulado_completo_sem_chamadas_reais(monkeypatch, isolated_home):
    # Nunca chamar colab/remote de verdade
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/ssh")
    monkeypatch.setattr(cli.colab, "version", lambda: "Version: 0.7.5")
    monkeypatch.setattr(
        cli.colab,
        "run",
        lambda args, check=False: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    def fake_ssh_keygen(cmd, check=False, capture_output=False, text=False):
        idx = cmd.index("-f")
        Path(cmd[idx + 1]).write_text("PRIVATE", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="Generated key", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_ssh_keygen)
    monkeypatch.setattr(cli.colab, "orphan_assignments", lambda: [])
    monkeypatch.setattr(cli.colab, "usage", lambda: "")
    monkeypatch.setattr(cli.colab, "new", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli.colab,
        "status_line",
        lambda name: colab.SessionLine(
            name=name, endpoint="m-s-abc", hardware="L4", shape="standard", variant="GPU"
        ),
    )
    monkeypatch.setattr(cli.colab, "stop", lambda name: None)
    monkeypatch.setattr(cli.colab, "is_allocated", lambda name, ep="": False)
    monkeypatch.setattr(cli, "_sync_external_configs", lambda cfg, oc, dc: None)
    monkeypatch.setattr(cli, "_bootstrap_worker", lambda cfg, st, cl: None)
    monkeypatch.setattr(cli, "_start_daemon", lambda: 99999)
    monkeypatch.setattr(cli, "_wait_ready", lambda cfg, t, expected_session=None: {})
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "session": expected_session,
            "worker": {
                "state": "ready",
                "watchdog_silence_s": 2,
                "idle_s": 5,
                "idle_timeout_min": 20,
            },
            "tunnel": {"state": "connected", "reconnects": 0, "latency_ms": 25},
        },
    )

    class FakeRemote:
        def __init__(self, session, cfg):
            self.session = session

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(cli.remote, "Remote", FakeRemote)

    rec = tui.criar_console(width=100, height=30, record=True)
    monkeypatch.setattr(tui, "console", rec)
    monkeypatch.setattr(tui, "_confirmar", lambda *a, **kw: True)
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **kw: "menu")

    # 1. init simulado
    assert tui._executar_init(rec, cli, tui.ler_estado_local()) == "menu"
    assert tui.ler_estado_local().ambiente_configurado is True

    # 2. aplicar perfil (configure)
    itens = tui.listar_perfis_estruturados(tui.ler_estado_local())
    qwen32 = next(i for i in itens if i.profile.id == "qwen3.8-27b-32k")
    ok, _ = tui._aplicar_perfil_interativo(rec, qwen32, pedir_confirmacao=False)
    assert ok is True
    assert tui.ler_estado_local().perfil_id == "qwen3.8-27b-32k"

    # 3. up simulado
    assert tui._executar_up(rec, cli, tui.ler_estado_local()) == "menu"
    assert tui.ler_estado_local().sessao_ativa is True

    # 4. status simulado
    assert tui._executar_status(rec, cli, tui.ler_estado_local()) == "menu"

    # 5. down simulado
    assert tui._executar_down(rec, cli, tui.ler_estado_local()) == "menu"
    assert tui.ler_estado_local().sessao_ativa is False

    saida = rec.export_text(clear=False)
    assert "versão 0.7.5" in saida
    assert "Version: 0.7.5" not in saida
    assert "Duração: 0.0s" not in saida
    assert "Sessão iniciada e API pronta" in saida
    assert "encerrada e confirmada pelo Colab" in saida


def test_erro_forcado_ssh_ausente_exibe_tres_partes_e_detalhes_tecnicos(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    rec = tui.criar_console(width=100, height=30, record=True)
    monkeypatch.setattr(tui, "_confirmar", lambda *a, **kw: True)

    with pytest.raises(tui._ErroEtapa) as exc_info:
        tui._executar_init(rec, cli, tui.ler_estado_local())

    err = exc_info.value
    tui._renderizar_erro(
        rec, err.falha, err.causa, err.acao, detalhes_tecnicos=err.detalhes_tecnicos
    )
    saida = rec.export_text(clear=False)
    assert "O que aconteceu:" in saida
    assert "Causa provável:" in saida
    assert "O que fazer:" in saida
    assert "Detalhes técnicos" in saida
    assert "shutil.which('ssh') -> None" in saida


def test_menu_sem_atalhos_duplicados_e_foco_no_recomendado():
    import questionary

    estado = tui.EstadoLocal(
        config_existe=True,
        chave_ssh_existe=True,
        ambiente_configurado=True,
        perfil_id="inkling-2-mini-q4-gguf",
        perfil_nome="Inkling 2 Mini (4 bit, GGUF)",
        gpu_id="L4",
        gpu_rotulo="L4",
        sessao_ativa=False,
        sessao_nome="",
        sessao_criada_em="",
    )
    escolhas, padrao = tui._construir_escolhas_menu(estado, tui.UNICODE_SIMBOLOS)
    assert padrao is not None
    assert padrao.value == "up"
    for item in escolhas:
        if isinstance(item, questionary.Choice):
            assert not re.match(r"^[0-9hq]\.\s", str(item.title))
    assert ("selected", "noreverse") in tui._estilo_select().style_rules


def test_up_tui_nao_configura_harness_automaticamente(monkeypatch, isolated_home):
    cfg = config.Config()
    config.apply_profile(cfg, "inkling-2-mini-q4-gguf")
    cfg.ssh_key_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ssh_key_path.write_text("PRIVATE", encoding="utf-8")
    config.save_config(cfg)

    def proibido_sync(*args, **kwargs):  # pragma: no cover
        raise AssertionError("_executar_up na TUI não deve configurar harness automaticamente")

    monkeypatch.setattr(cli, "_sync_external_configs", proibido_sync)
    monkeypatch.setattr(cli.colab, "orphan_assignments", lambda: [])
    monkeypatch.setattr(cli.colab, "new", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli.colab,
        "status_line",
        lambda name: colab.SessionLine(
            name=name, endpoint="m-s-abc", hardware="L4", shape="standard", variant="GPU"
        ),
    )
    monkeypatch.setattr(cli, "_bootstrap_worker", lambda c, s, cl: None)
    monkeypatch.setattr(cli, "_start_daemon", lambda: 99999)
    monkeypatch.setattr(cli, "_wait_ready", lambda c, t, expected_session=None: {})

    class FakeRemote:
        def __init__(self, session, cfg):
            self.session = session

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(cli.remote, "Remote", FakeRemote)

    rec = tui.criar_console(width=100, height=30, record=True)
    monkeypatch.setattr(tui, "_confirmar", lambda *a, **kw: True)
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **kw: "menu")

    assert tui._executar_up(rec, cli, tui.ler_estado_local()) == "menu"
    assert "Sessão iniciada e API pronta" in rec.export_text(clear=False)


def test_renderizar_instrucoes_harness_todos(isolated_home):
    cfg = config.Config()
    config.apply_profile(cfg, "inkling-2-mini-q4-gguf")
    estado = tui.ler_estado_local()

    for harness_id in ("opencode", "dsh", "aider", "cline"):
        rec = tui.criar_console(width=100, height=30, record=True)
        tui.renderizar_instrucoes_harness(rec, harness_id, cfg, estado)
        saida = rec.export_text(clear=False)
        assert "http://127.0.0.1:4000/v1" in saida
        assert "inkling-2-mini" in saida
        assert "Configuração manual" in saida


def test_aplicar_harness_automatico_opencode_e_dsh(isolated_home):
    cfg = config.Config()
    config.apply_profile(cfg, "inkling-2-mini-q4-gguf")

    rec_ok = tui.criar_console(width=100, height=30, record=True)
    assert tui._aplicar_harness_automatico(rec_ok, "opencode", cfg) is True
    assert "OpenCode configurado para a API do Colab" in rec_ok.export_text(clear=False)
    assert (isolated_home / ".config" / "opencode" / "opencode.json").is_file()

    assert tui._aplicar_harness_automatico(rec_ok, "dsh", cfg) is True
    assert (isolated_home / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml").is_file()

    # Caso opencode.jsonc possua comentários: não sobrescreve e orienta configuração manual
    jsonc = isolated_home / ".config" / "opencode" / "opencode.jsonc"
    jsonc.write_text('{\n  // comentário\n  "mcp": {}\n}\n', encoding="utf-8")
    rec_err = tui.criar_console(width=100, height=30, record=True)
    assert tui._aplicar_harness_automatico(rec_err, "opencode", cfg) is False
    saida_err = rec_err.export_text(clear=False)
    assert "O que aconteceu:" in saida_err
    assert "opencode.jsonc" in saida_err
    assert "Detalhes técnicos" in saida_err
