"""`worker/requirements.lock`: versões exatas com hashes, instaladas pelo bootstrap."""

from __future__ import annotations

import re
from pathlib import Path

from byom import cli

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "worker"
LOCK = WORKER / "requirements.lock"


def _requisitos() -> list[tuple[str, list[str]]]:
    """(linha do requisito, hashes) do lock."""
    itens: list[tuple[str, list[str]]] = []
    atual: str | None = None
    hashes: list[str] = []
    for linha in LOCK.read_text(encoding="utf-8").splitlines():
        if not linha.strip() or linha.lstrip().startswith("#"):
            continue
        if linha.startswith(" "):
            if atual and "--hash=sha256:" in linha:
                hashes.append(linha.strip())
            continue
        if atual:
            itens.append((atual, hashes))
        atual = linha.split("\\")[0].strip()
        hashes = []
        if "--hash=sha256:" in linha:
            hashes.append(linha.strip())
    if atual:
        itens.append((atual, hashes))
    return itens


def test_lock_existe_e_tem_versao_exata_em_toda_linha():
    assert LOCK.is_file()
    itens = _requisitos()

    assert itens, "o lock não pode estar vazio"
    for linha, _hashes in itens:
        assert "==" in linha, linha
        assert ">=" not in linha and "~=" not in linha and "*" not in linha, linha


def test_todo_pacote_do_lock_tem_hash():
    for linha, hashes in _requisitos():
        assert hashes, f"{linha} sem hash"
        for item in hashes:
            limpo = item.rstrip("\\").strip()
            assert re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", limpo), limpo


def test_lock_cobre_as_dependencias_declaradas_no_requirements():
    nomes_lock = {
        linha.split("==")[0].strip().lower() for linha, _ in _requisitos()
    }
    fonte = (WORKER / "requirements.txt").read_text(encoding="utf-8").splitlines()

    for linha in fonte:
        if not linha.strip() or linha.lstrip().startswith("#"):
            continue
        nome = re.split(r"[<>=!~ ]", linha.strip(), maxsplit=1)[0].lower()
        assert nome in nomes_lock, f"{nome} não está no lock"


def test_bootstrap_instala_o_lock_com_require_hashes():
    texto = (WORKER / "bootstrap.sh").read_text(encoding="utf-8")

    assert "--require-hashes" in texto
    assert "requirements.lock" in texto
    # O fallback continua existindo, mas avisa que perdeu o pin.
    assert "WARNING" in texto and "requirements.txt" in texto


def test_bootstrap_monta_o_argv_com_o_python_do_venv():
    """O TTS procura vllm_omni no sys.path; o python3 do sistema não o tem."""
    texto = (WORKER / "bootstrap.sh").read_text(encoding="utf-8")

    assert '"$VENV/bin/python" "$BYOM_DIR/build_vllm_command.py"' in texto


def test_lock_vai_no_bundle_do_worker():
    assert "requirements.lock" in cli.WORKER_FILES
    assert "requirements.txt" in cli.WORKER_FILES
