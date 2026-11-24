#!/usr/bin/env python3
"""Lista arquivos publicados de um repo do Hugging Face (só leitura).

Uso: python scripts/hf_tree.py <repo> [<repo> ...]
Imprime `path<TAB>bytes<TAB>GiB` e, com --json, o payload bruto da API.
Depende só da biblioteca padrão: é uma sonda de conferência, não parte do pacote.
"""

from __future__ import annotations

import json
import sys
import urllib.request

API = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=false"


def tree(repo: str) -> list[dict]:
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise SystemExit(f"resposta inesperada para {repo}")
    return [item for item in payload if item.get("type") == "file"]


def main() -> None:
    as_json = "--json" in sys.argv
    repos = [arg for arg in sys.argv[1:] if arg != "--json"]
    if not repos:
        raise SystemExit(__doc__)
    result = {}
    for repo in repos:
        files = tree(repo)
        result[repo] = [item["path"] for item in files]
        print(f"== {repo}")
        for item in files:
            size = item.get("size")
            if size:
                print(f"  {item['path']}\t{size}\t{size / 2**30:.2f} GiB")
            else:
                print(f"  {item['path']}\t?")
    if as_json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
