#!/usr/bin/env python3
"""Regenera os literais esperados de tests/test_cli_snapshots.py.

Ferramenta de manutenção (não faz parte do pacote): quando números do catálogo
mudam, os snapshots da CLI mudam junto. Uso:

    BYOM_CONFIG_DIR=/tmp/snap uv run python scripts/update_cli_snapshots.py

Reescreve apenas os blocos `esperado = (...)` de `profiles` e `gpus` e o bloco
`res_img.output == (...)` de `configure`. O caminho do config.toml da execução
vira `{cfg_file}`, como nos literais originais (f-string).
"""

from __future__ import annotations

from pathlib import Path

from byom import config

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "tests" / "test_cli_snapshots.py"
WRAP = 88
INDENT = "        "


def _quote(piece: str, placeholder: str) -> str:
    if placeholder in piece:
        escaped = piece.replace("{", "{{").replace("}", "}}").replace("{{cfg_file}}", "{cfg_file}")
        return f'f"{escaped}"'
    return f'"{piece}"'


def _literal(text: str, indent: str = INDENT) -> str:
    """Literal Python exato do texto, quebrado por tamanho de linha."""
    placeholder = str(config.config_path())
    text = text.replace(placeholder, "{cfg_file}")
    limit = WRAP - len(indent) - 3
    pieces: list[str] = []
    for raw in text.splitlines(keepends=True):
        line = raw.replace("\n", "\\n")
        while line:
            pieces.append(line[:limit])
            line = line[limit:]
    return "\n".join(f"{indent}{_quote(piece, '{cfg_file}')}" for piece in pieces)


def _replace(source: str, start_marker: str, end_marker: str, text: str) -> str:
    start = source.index(start_marker) + len(start_marker)
    end = source.index(end_marker, start)
    return f"{source[:start]}\n{_literal(text)}{source[end:]}"


def _run(args: list[str]) -> str:
    from typer.testing import CliRunner

    from byom import cli

    result = CliRunner().invoke(cli.app, args)
    if result.exit_code != 0:
        raise SystemExit(f"`byom {' '.join(args)}` falhou: {result.output}")
    return result.output


def main() -> int:
    source = TARGET.read_text(encoding="utf-8")

    source = _replace(
        source,
        'def test_snapshot_cli_profiles():\n    res = runner.invoke(cli.app, ["profiles"])\n'
        "    assert res.exit_code == 0\n    esperado = (",
        "    )\n    assert res.output == esperado",
        _run(["profiles"]),
    )
    source = _replace(
        source,
        'def test_snapshot_cli_gpus():\n    res = runner.invoke(cli.app, ["gpus"])\n'
        "    assert res.exit_code == 0\n    esperado = (",
        "    )\n    assert res.output == esperado",
        _run(["gpus"]),
    )
    source = _replace(
        source,
        "    assert res_img.output == (",
        "    )\n\n",
        _run(["configure", "--profile", "qwen-image-2.1-q4km"]),
    )
    TARGET.write_text(source, encoding="utf-8")
    print(f"{TARGET} atualizado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
