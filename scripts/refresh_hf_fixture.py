#!/usr/bin/env python3
"""Regrava a fixture de árvores do Hugging Face a partir da API pública.

Uso: python scripts/refresh_hf_fixture.py
Escreve `tests/fixtures/hf_files.json` com `path` e `size` de cada arquivo.
A fixture é o que permite testar `match_file` sem rede no CI; este script é o
único ponto que fala com a API, e é rodado à mão quando um repo muda.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tests" / "fixtures" / "hf_files.json"

REPOS = (
    "leejet/Qwen-Image-2.1-GGUF",
    "Comfy-Org/Qwen-Image-2.1",
    "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "unsloth/Inkling-2-Mini-GGUF",
    "unsloth/Muse-Glimmer-30B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from hf_tree import tree

    payload = {
        "source": "https://huggingface.co/api/models/{repo}/tree/main?recursive=true",
        "captured_at": date.today().isoformat(),
        "repos": {},
    }
    for repo in REPOS:
        files = [
            {"path": item["path"], "size": item.get("size", 0)}
            for item in sorted(tree(repo), key=lambda item: item["path"])
        ]
        payload["repos"][repo] = files
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{DEST}: {sum(len(v) for v in payload['repos'].values())} arquivos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
