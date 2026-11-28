"""Telas de geração de imagem e áudio da TUI."""

from __future__ import annotations

import base64

import questionary

from byom import config, tui


class Pergunta:
    def __init__(self, valor):
        self.valor = valor

    def ask(self):
        return self.valor


def _estado(runtime: str) -> tui.EstadoLocal:
    return tui.EstadoLocal(
        config_existe=True,
        chave_ssh_existe=True,
        ambiente_configurado=True,
        perfil_id="teste",
        perfil_nome="Teste",
        gpu_id="T4",
        gpu_rotulo="Tesla T4",
        sessao_ativa=True,
        sessao_nome="byom-teste",
        sessao_criada_em="",
        runtime=runtime,
    )


def _config_com(profile_id: str) -> config.Config:
    cfg = config.Config()
    config.apply_profile(cfg, profile_id)
    config.save_config(cfg)
    return config.load_config()


def test_tui_imagem_pede_prompt_e_salva_png(monkeypatch):
    _config_com("qwen-image-2.1-q4km")
    respostas = iter(["uma casa na neve", "8", "1.0"])
    monkeypatch.setattr(questionary, "text", lambda *a, **k: Pergunta(next(respostas)))
    monkeypatch.setattr(questionary, "select", lambda *a, **k: Pergunta("1024x1024"))
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")

    class Resposta:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(b"PNG").decode()}]}

    monkeypatch.setattr(tui.httpx, "post", lambda *a, **k: Resposta())

    rec = tui.criar_console(width=100, height=40, record=True)
    assert tui._executar_imagem(rec, None, _estado("image")) == "menu"

    destino = config.byom_home() / "images"
    arquivos = list(destino.glob("imagem-*.png"))
    assert len(arquivos) == 1
    assert arquivos[0].read_bytes() == b"PNG"
    assert "Imagem gerada" in rec.export_text(clear=False)


def test_tui_imagem_exibe_os_padroes_do_modelo_turbo(monkeypatch):
    _config_com("qwen-image-2.1-turbo-uncensored-q4k")
    respostas = iter(["uma casa na neve", "8", "1.0"])
    capturados: list[dict] = []

    def fake_text(mensagem, **kwargs):
        capturados.append(kwargs)
        return Pergunta(next(respostas))

    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "select", lambda *a, **k: Pergunta("1024x1024"))
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")

    class Resposta:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(b"PNG").decode()}]}

    monkeypatch.setattr(tui.httpx, "post", lambda *a, **k: Resposta())

    rec = tui.criar_console(width=100, height=40, record=True)
    assert tui._executar_imagem(rec, None, _estado("image")) == "menu"

    assert capturados[1]["default"] == "8"
    assert capturados[2]["default"] == "1.0"


def test_tui_audio_pede_texto_voz_e_salva_wav(monkeypatch):
    _config_com("qwen3-tts-0.6b-customvoice")
    vozes = iter(["vivian", "Portuguese"])
    monkeypatch.setattr(questionary, "text", lambda *a, **k: Pergunta("olá, mundo"))
    monkeypatch.setattr(questionary, "select", lambda *a, **k: Pergunta(next(vozes)))
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")

    class Resposta:
        status_code = 200
        headers = {"content-type": "audio/wav"}
        content = b"RIFFwav"
        text = ""

    monkeypatch.setattr(tui.httpx, "post", lambda *a, **k: Resposta())

    rec = tui.criar_console(width=100, height=40, record=True)
    assert tui._executar_audio(rec, None, _estado("tts")) == "menu"

    destino = config.byom_home() / "audio"
    arquivos = list(destino.glob("audio-*.wav"))
    assert len(arquivos) == 1
    assert arquivos[0].read_bytes() == b"RIFFwav"
    assert "Áudio gerado" in rec.export_text(clear=False)


def _capturar_proximo_passo(monkeypatch, acao_id: str, runtime: str) -> list[str]:
    capturadas: list[list[str]] = []

    def fake_select(mensagem, *, choices, **kwargs):
        capturadas.append([choice.title for choice in choices])
        return Pergunta("menu")

    monkeypatch.setattr(questionary, "select", fake_select)
    assert tui._perguntar_proximo_passo(acao_id, _estado(runtime)) == "menu"
    return capturadas[0]


def test_proximo_passo_de_configure_nao_oferece_harness(monkeypatch):
    titulos = _capturar_proximo_passo(monkeypatch, "configure", "image")

    assert any(titulo.startswith("Iniciar sessão no Colab") for titulo in titulos)
    assert all("Configurar harness" not in titulo for titulo in titulos)
    assert titulos[-1] == "Voltar ao menu principal"


def test_proximo_passo_de_up_oferece_geracao_de_imagem(monkeypatch):
    titulos = _capturar_proximo_passo(monkeypatch, "up", "image")

    assert "Gerar imagem" in titulos
    assert all("Configurar harness" not in titulo for titulo in titulos)
    assert all("Conversar com o modelo" not in titulo for titulo in titulos)


def test_proximo_passo_de_up_oferece_geracao_de_audio(monkeypatch):
    titulos = _capturar_proximo_passo(monkeypatch, "up", "tts")

    assert "Gerar áudio" in titulos
    assert all("Configurar harness" not in titulo for titulo in titulos)
    assert all("Conversar com o modelo" not in titulo for titulo in titulos)


def test_tui_imagem_recusa_runtime_de_chat(monkeypatch):
    _config_com("qwen3.8-27b-32k")
    monkeypatch.setattr(tui, "_confirmar", lambda *a, **k: False)
    chamadas: list[object] = []
    monkeypatch.setattr(tui.httpx, "post", lambda *a, **k: chamadas.append(k))

    rec = tui.criar_console(width=100, height=40, record=True)
    assert tui._executar_imagem(rec, None, _estado("vllm")) == "menu"

    assert chamadas == []
    assert "não é de imagem" in rec.export_text(clear=False)
