#!/usr/bin/env python3
"""Prepare binários fixados e pesos do runtime llama.cpp/imagem no worker.

Executado uma vez por sessão, antes de iniciar o servidor. Nenhuma versão
"latest" é usada: URL, quantização e checksum vêm do perfil em session.json.
Só a biblioteca padrão é necessária neste arquivo.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

BYOM_DIR = Path("/content/byom")
MODEL_DIR = BYOM_DIR / "models"
MANIFEST = BYOM_DIR / "runtime_manifest.json"
HF_API = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=false"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"


def download(url: str, dest: Path, sha256: str | None = None) -> Path:
    """Baixe um artefato e valide o digest quando a fonte o fornece."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
    if sha256 and digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {url}: {digest.hexdigest()} != {sha256}")
    os.replace(tmp, dest)
    return dest


def _safe_join(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise ValueError(f"Unsafe path in archive: {relative!r}")
    return target


def _model_path(root: Path, path: str) -> Path:
    """Use `.gguf` locally even when the repo publishes `.gguf_file`."""
    local = path[:-5] if path.endswith(".gguf_file") else path
    return _safe_join(root, local)


def extract(archive: Path, dest: Path) -> None:
    """Extraia tar.gz ou zip para dest, recusando caminhos fora de dest."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            _check_members(dest, names)
            zf.extractall(dest)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            names = [member.name for member in tf.getmembers() if member.isfile()]
            _check_members(dest, names)
            tf.extractall(dest)


def _check_members(dest: Path, names: list[str]) -> None:
    for name in names:
        _safe_join(dest, name)


def find_binary(root: Path, name: str) -> Path:
    """Localize o binário depois da extração e crie um caminho estável."""
    matches = [path for path in root.rglob(name) if path.is_file()]
    if not matches:
        raise ValueError(f"{name} not found under {root}")
    binary = matches[0]
    binary.chmod(0o755)
    stable = BYOM_DIR / name
    if binary != stable:
        if stable.exists() or stable.is_symlink():
            stable.unlink()
        stable.symlink_to(binary)
    return stable


def hf_files(repo: str) -> list[str]:
    """Liste arquivos de um repo HF fixado, sem baixar modelo nesta etapa."""
    if "/" not in repo:
        raise ValueError(f"Invalid model repo: {repo!r}")
    with urllib.request.urlopen(HF_API.format(repo=repo), timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected HF tree response for {repo}")
    return [str(item["path"]) for item in payload if item.get("type") == "file"]


def match_file(paths: list[str], pattern: str) -> str:
    matches = [path for path in sorted(paths) if fnmatch.fnmatch(path, pattern)]
    if not matches:
        raise ValueError(f"No model file matches {pattern!r}")
    return matches[0]


def prepare(session_file: Path) -> dict:
    cfg = json.loads(session_file.read_text(encoding="utf-8"))
    runtime = cfg.get("runtime", "vllm")
    if runtime == "vllm":
        return {"runtime": runtime, "downloaded": []}

    worker = cfg.get("worker") or {}
    binary_url = worker.get("binary_url")
    binary_sha256 = worker.get("binary_sha256")
    if not binary_url or not binary_sha256:
        raise ValueError(f"Pinned binary_url and binary_sha256 are required for {runtime}")
    suffix = ".zip" if ".zip" in binary_url else ".tar.gz"
    archive = BYOM_DIR / ("binary" + suffix)
    download(binary_url, archive, binary_sha256)
    binary_name = "sd-server" if runtime == "image" else "llama-server"
    binary_root = BYOM_DIR / str(worker.get("binary_dir") or "runtime")
    extract(archive, binary_root)
    find_binary(binary_root, binary_name)

    model_repo = worker.get("model_repo")
    if runtime == "llama-cpp":
        if not model_repo:
            raise ValueError("model_repo is required for llama-cpp")
        paths = hf_files(model_repo)
        resolved: dict[str, str] = {}
        for key in ("main_pattern", "mmproj_pattern", "drafter_pattern"):
            pattern = worker.get(key)
            if pattern:
                resolved[key] = match_file(paths, str(pattern))
        wanted = set(resolved.values()) | {
            match_file(paths, str(pattern)) for pattern in worker.get("model_patterns", [])
        }
        for path in sorted(wanted):
            url = HF_RESOLVE.format(repo=model_repo, path=path)
            download(url, _model_path(MODEL_DIR / model_repo, path))
        manifest = {
            "runtime": runtime,
            "server": str(BYOM_DIR / "llama-server"),
            "main": str(_model_path(MODEL_DIR / model_repo, resolved["main_pattern"])),
        }
        for key in ("mmproj", "drafter"):
            pattern_key = key + "_pattern"
            if pattern_key in resolved:
                manifest[key] = str(_model_path(MODEL_DIR / model_repo, resolved[pattern_key]))
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    specs = worker.get("model_patterns") or []
    downloaded: list[str] = []
    for spec in specs:
        if "::" not in spec:
            raise ValueError(f"Invalid image model spec: {spec!r}")
        repo, pattern = spec.split("::", 1)
        path = match_file(hf_files(repo), pattern)
        download(HF_RESOLVE.format(repo=repo, path=path), _safe_join(MODEL_DIR / repo, path))
        downloaded.append(f"{repo}::{path}")
    manifest = {
        "runtime": runtime,
        "server": str(BYOM_DIR / "sd-server"),
        "downloaded": downloaded,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = sys.argv[1:]
    if len(args) != 1:
        print("usage: prepare_runtime.py <session.json>", file=sys.stderr)
        raise SystemExit(2)
    result = prepare(Path(args[0]))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
