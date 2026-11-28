#!/usr/bin/env python3
"""Prepare binários fixados e pesos do runtime llama.cpp/imagem/Strata no worker.

Executado uma vez por sessão, antes de iniciar o servidor. Nenhuma versão
"latest" é usada: URL, quantização e checksum vêm do perfil em session.json.
Só a biblioteca padrão é necessária neste arquivo.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

BYOM_DIR = Path("/content/byom")
MODEL_DIR = BYOM_DIR / "models"
MANIFEST = BYOM_DIR / "runtime_manifest.json"
HF_API = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=false"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"

#: Papéis de imagem e a flag do sd-server que carrega cada um. Confirmado na
#: documentação da build fixada (`master-1106-532d2c1`):
#: https://github.com/leejet/stable-diffusion.cpp/blob/532d2c1/docs/models/qwen_image_2.1.md
#: (consultado em 2026-11-24) — tabela "Model files": `--diffusion-model`
#: (obrigatório), `--vae` e `--llm` (text encoder). Sem inferência por nome de
#: arquivo: o papel vem do perfil.
IMAGE_ROLE_FLAGS = {
    "diffusion": "--diffusion-model",
    "vae": "--vae",
    "text_encoder": "--llm",
    # Edição de imagem (`/v1/images/edits`): projetor de visão do text encoder.
    "vision": "--llm_vision",
}

#: Papéis sem os quais o servidor não sobe. `vision` é opcional: o perfil sem
#: ele continua servindo geração, e só a edição fica indisponível.
REQUIRED_IMAGE_ROLES = ("diffusion", "vae", "text_encoder")

#: Runtime `strata`: o engine gráfico é um pacote fechado com as libs CUDA ao
#: lado do executável (`libcublas.so.12`, `libcublasLt.so.12`, `libcudart.so.12`,
#: `libgomp.so.1` — o tarball do 0.1.70 tem 655.962.569 B por causa delas). O
#: binário resolve as libs pelo próprio diretório (`$ORIGIN`), então o diretório
#: extraído fica inteiro e o path do executável vai absoluto para o manifesto.
STRATA_ENGINE_BINARY = "strata-inkling"
#: O control traz um CPython embarcado (3.12.15 no 0.1.70) com o servidor HTTP
#: em `serve/server.py`; é ele que fala OpenAI para o gateway.
STRATA_SERVER_PYTHON = "python/bin/python3"
#: Arquivos de tokenizer que o servidor procura AO LADO do pack quando o
#: `vocab.json` ainda não existe (`_ensure_shape_tokenizer`, passo 2 do
#: `serve/server.py`). Não é opcional na prática: o CPython embarcado não tem
#: `zstandard`, então o passo 1 (ler o manifest do `.strata`) falha com "this
#: Python cannot read zstd", e o `strata-inkling --export-tokenizer` do pack
#: SQ-10 devolve `tokenizer_json: 0 bytes` (o pack não traz essa seção) —
#: medido com o pack real em 2026-11-27. Com estes dois arquivos ao lado do
#: pack, o servidor gera `vocab.json`/`merges.txt`/`token_type.json` na primeira
#: subida e segue.
STRATA_TOKENIZER_FILES = ("tokenizer.json", "chat_template.jinja")
#: Driver NVIDIA mínimo do engine, lido do `engine-component.json` do próprio
#: tarball (`min_driver.nvidia`) na release v0.2.23, e confirmado no
#: `strata-feed.json` assinado da mesma release (2026-11-27). O cuda13 pede
#: 580.65; o perfil fixa o cuda12 porque o llama.cpp do repositório também usa
#: CUDA 12.8, e o Colab não garante driver novo.
STRATA_MIN_NVIDIA_DRIVER = "575.51"


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
    return [entrada["path"] for entrada in hf_entries(repo)]


def hf_entries(repo: str) -> list[dict]:
    """Tree de um repo HF fixado, com tamanho e o sha256 do LFS quando existe.

    O `lfs.oid` que a API devolve para um arquivo versionado no LFS **é** o
    sha256 do conteúdo: ele bate, nos quatro packs que o `SHA256SUMS.txt` do
    `strata-engine/Inkling-2-Mini-Strata` lista, com o digest daquele arquivo.
    É o que permite conferir um pack que o sums ainda não cobre.
    """
    if "/" not in repo:
        raise ValueError(f"Invalid model repo: {repo!r}")
    with urllib.request.urlopen(HF_API.format(repo=repo), timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected HF tree response for {repo}")
    entradas: list[dict] = []
    for item in payload:
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs")
        entradas.append(
            {
                "path": str(item["path"]),
                "size": int(item.get("size") or 0),
                "sha256": str(lfs.get("oid") or "") if isinstance(lfs, dict) else "",
            }
        )
    return entradas


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    """`SHA256SUMS.txt` -> {nome: sha256}.

    Aceita as duas formas do `sha256sum` (`<hash>  <nome>` e `<hash> *<nome>`),
    ignora linhas de comentário e guarda só o nome do arquivo, sem diretório: o
    pack é procurado pelo nome em que o `pack_pattern` casou.
    """
    sums: dict[str, str] = {}
    for line in text.splitlines():
        partes = line.strip().split(None, 1)
        if len(partes) != 2:
            continue
        digest, nome = partes[0].strip().lower(), partes[1].strip().lstrip("*").strip()
        if len(digest) == 64 and nome and all(c in "0123456789abcdef" for c in digest):
            sums[PurePosixPath(nome).name] = digest
    return sums


def _version_tuple(version: str, what: str) -> tuple[int, ...]:
    """`"575.51"` / `"575.51.03"` -> (575, 51)...; recusa o que não é numérico."""
    parts: list[int] = []
    for piece in str(version).strip().split(".")[:3]:
        if not piece.isdigit():
            raise ValueError(f"versão de {what} não reconhecida: {version!r}")
        parts.append(int(piece))
    if not parts:
        raise ValueError(f"versão de {what} vazia")
    return tuple(parts)


def _version_at_least(found: str, minimum: str, what: str) -> bool:
    """Compara major.minor(.patch) com padding, sem `packaging` (só stdlib)."""
    a = _version_tuple(found, what)
    b = _version_tuple(minimum, what)
    size = max(len(a), len(b))
    return a + (0,) * (size - len(a)) >= b + (0,) * (size - len(b))


def nvidia_driver_version() -> str:
    """Versão do driver NVIDIA desta máquina, como o `nvidia-smi` a reporta."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "nvidia-smi não encontrado: o engine do Strata exige driver NVIDIA "
            f"{STRATA_MIN_NVIDIA_DRIVER} ou mais novo e sem o nvidia-smi não há como "
            "confirmar a versão do driver antes de baixar 10+ GiB de pesos"
        ) from exc
    except OSError as exc:  # pragma: no cover - ambiente sem permissão de exec
        raise ValueError(f"nvidia-smi não pôde ser executado: {exc}") from exc
    linha = next((linha.strip() for linha in done.stdout.splitlines() if linha.strip()), "")
    if done.returncode != 0 or not linha:
        detalhe = (done.stderr or done.stdout).strip() or f"código {done.returncode}"
        raise ValueError(
            "nvidia-smi não respondeu à consulta de driver_version "
            f"({detalhe}): o engine do Strata exige driver {STRATA_MIN_NVIDIA_DRIVER}+"
        )
    return linha.splitlines()[0].strip()


def check_nvidia_driver(minimum: str = STRATA_MIN_NVIDIA_DRIVER) -> str:
    """Falha ANTES de baixar pesos quando o driver é mais velho que o exigido.

    O engine `strata-inkling` 0.1.70 declara `min_driver.nvidia = "575.51"` e se
    recusa a subir abaixo disso; descobrir isso depois de baixar o pack custaria
    ~10 GiB de download e uma sessão do Colab.
    """
    encontrado = nvidia_driver_version()
    if not _version_at_least(encontrado, minimum, "driver NVIDIA"):
        raise ValueError(
            f"driver NVIDIA {encontrado} é mais antigo que o mínimo {minimum} do engine "
            "do Strata (strata-inkling 0.1.70, release v0.2.23, campo "
            "`min_driver.nvidia` do `engine-component.json`); o engine se recusa a "
            "subir abaixo disso. Troque a GPU/imagem do Colab ou use um perfil de "
            "outro runtime"
        )
    return encontrado


def engine_min_driver(engine_root: Path) -> str:
    """`min_driver.nvidia` declarado pelo próprio pacote do engine extraído."""
    caminho = engine_root / "engine-component.json"
    if not caminho.is_file():
        return ""
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"engine-component.json ilegível em {engine_root}: {exc}") from exc
    minimo = (dados.get("min_driver") or {}).get("nvidia") if isinstance(dados, dict) else ""
    return str(minimo or "")


def match_file(paths: list[str], pattern: str) -> str:
    matches = [path for path in sorted(paths) if fnmatch.fnmatch(path, pattern)]
    if not matches:
        raise ValueError(f"No model file matches {pattern!r}")
    return matches[0]


def _prepare_strata(worker: dict) -> dict:
    """Componentes fixados + pack `.strata` do engine Strata.

    Ordem deliberada: o driver é conferido ANTES de qualquer download (o engine
    0.1.70 exige 575.51 e não sobe abaixo disso), depois vêm os dois tarballs da
    release (control e engine, ambos com sha256 do perfil), o `SHA256SUMS.txt` do
    repo do pack e só então o pack em si — 9,75 a 18,04 GiB conforme a variante.
    """
    driver = check_nvidia_driver()

    model_repo = str(worker.get("model_repo") or "").strip()
    if "/" not in model_repo:
        raise ValueError(f"model_repo inválido para o runtime strata: {model_repo!r}")
    pack_pattern = str(worker.get("pack_pattern") or "").strip()
    if not pack_pattern:
        raise ValueError("runtime strata requires pack_pattern (e.g. Inkling-2-Mini-SQ-10.strata)")

    control_url = str(worker.get("control_url") or "").strip()
    control_sha = str(worker.get("control_sha256") or "").strip()
    engine_url = str(worker.get("engine_url") or "").strip()
    engine_sha = str(worker.get("engine_sha256") or "").strip()
    for nome, valor in (
        ("control_url", control_url),
        ("control_sha256", control_sha),
        ("engine_url", engine_url),
        ("engine_sha256", engine_sha),
    ):
        if not valor:
            raise ValueError(f"runtime strata requires {nome} (pinned release component)")

    control_root = BYOM_DIR / str(worker.get("control_dir") or "strata-control")
    control_archive = BYOM_DIR / "strata-control.tar.gz"
    download(control_url, control_archive, control_sha)
    extract(control_archive, control_root)
    server = control_root / STRATA_SERVER_PYTHON
    if not server.is_file():
        raise ValueError(
            f"o tarball do control não traz {STRATA_SERVER_PYTHON} em {control_root}: "
            "a URL fixada não é o pacote linux-x86_64 esperado"
        )

    engine_root = BYOM_DIR / str(worker.get("engine_dir") or "strata-engine")
    engine_archive = BYOM_DIR / "strata-engine.tar.gz"
    download(engine_url, engine_archive, engine_sha)
    extract(engine_archive, engine_root)
    engine = engine_root / STRATA_ENGINE_BINARY
    if not engine.is_file():
        raise ValueError(
            f"o tarball do engine não traz {STRATA_ENGINE_BINARY} na raiz de "
            f"{engine_root}; o diretório extraído precisa ficar inteiro porque o "
            "binário carrega as libs CUDA do próprio diretório"
        )
    engine.chmod(0o755)
    # O pacote diz o próprio mínimo: se a release mudar e pedir mais do que
    # STRATA_MIN_NVIDIA_DRIVER, o download já feito não vale para este driver.
    exigido = engine_min_driver(engine_root)
    if exigido and not _version_at_least(
        driver, exigido, "driver NVIDIA"
    ):  # pragma: no cover - depende da release fixada
        raise ValueError(
            f"o engine extraído exige driver NVIDIA {exigido}, mais novo que o "
            f"{driver} desta máquina (o perfil fixou 575.51)"
        )

    sums_name = str(worker.get("sums_pattern") or "SHA256SUMS.txt").strip()
    sums_path = _safe_join(MODEL_DIR / model_repo, sums_name)
    download(HF_RESOLVE.format(repo=model_repo, path=sums_name), sums_path)
    sums = parse_sums(sums_path.read_text(encoding="utf-8"))

    entradas = hf_entries(model_repo)
    pack_path = match_file([entrada["path"] for entrada in entradas], pack_pattern)
    esperado = sums.get(PurePosixPath(pack_path).name, "")
    fonte = "SHA256SUMS.txt"
    if not esperado:
        # O sums pode não cobrir um pack novo (o SQv2-20 da revisão 5623ceb não
        # está lá). O `lfs.oid` do tree é o sha256 do conteúdo — a mesma fonte,
        # conferida nos packs que o sums lista.
        entrada = next(item for item in entradas if item["path"] == pack_path)
        esperado = entrada["sha256"]
        fonte = "hf-tree-lfs"
    if not esperado:
        raise ValueError(
            f"sem sha256 para {pack_path}: nem o {sums_name} do repo nem o `lfs.oid` "
            "do tree do HF trazem um digest, e baixar sem conferir não é opção"
        )

    destino = _safe_join(MODEL_DIR / model_repo, pack_path)
    download(HF_RESOLVE.format(repo=model_repo, path=pack_path), destino, esperado)

    # Tokenizer ao lado do pack: o servidor do control monta o `vocab.json` a
    # partir destes dois arquivos (o pack não traz `tokenizer_json`, e o CPython
    # embarcado não lê zstd). Ausentes no repo, o server decide o que fazer.
    caminhos = {entrada["path"] for entrada in entradas}
    for nome in STRATA_TOKENIZER_FILES:
        if nome in caminhos:
            download(
                HF_RESOLVE.format(repo=model_repo, path=nome),
                _safe_join(MODEL_DIR / model_repo, nome),
            )

    manifest = {
        "runtime": "strata",
        "server": str(server),
        "control_root": str(control_root),
        "engine": str(engine),
        "pack": str(destino),
        "sums_verified": True,
        "pack_sha256": esperado,
        "pack_sha256_source": fonte,
        "driver": driver,
        "tokenizer_files": [nome for nome in STRATA_TOKENIZER_FILES if nome in caminhos],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare(session_file: Path) -> dict:
    cfg = json.loads(session_file.read_text(encoding="utf-8"))
    runtime = cfg.get("runtime", "vllm")
    if runtime in ("vllm", "tts"):
        # Os dois deixam o download dos pesos para o `vllm serve`; o bootstrap
        # instala `vllm_spec` (vLLM) e, no TTS, também `[worker].vllm_omni_spec`.
        return {"runtime": runtime, "downloaded": []}
    if runtime == "strata":
        return _prepare_strata(cfg.get("worker") or {})

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
        # O drafter de uma spec pode morar dentro dela (`[worker.spec]
        # drafter_pattern`) ou no `[worker]` (atalho `dspark`): os dois baixam
        # e viram `manifest["drafter"]`, que o argv usa em `--model-draft`.
        spec = worker.get("spec") or {}
        drafter_pattern = worker.get("drafter_pattern")
        if not drafter_pattern and isinstance(spec, dict):
            drafter_pattern = spec.get("drafter_pattern")
        padroes = {
            "main_pattern": worker.get("main_pattern"),
            "mmproj_pattern": worker.get("mmproj_pattern"),
            "drafter_pattern": drafter_pattern,
        }
        for key, pattern in padroes.items():
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

    def fetch(spec: str) -> str:
        if "::" not in spec:
            raise ValueError(f"Invalid image model spec: {spec!r}")
        repo, pattern = spec.split("::", 1)
        path = match_file(hf_files(repo), pattern)
        target = _safe_join(MODEL_DIR / repo, path)
        download(HF_RESOLVE.format(repo=repo, path=path), target)
        downloaded.append(f"{repo}::{path}")
        return str(target)

    roles = worker.get("image_roles")
    if roles is not None:
        # `image_roles` é a fonte da verdade quando existe: cada papel tem uma
        # flag própria do sd-server, e o mapa vai absoluto para o manifest.
        if not isinstance(roles, dict) or not roles:
            raise ValueError("[worker] image_roles must be a non-empty table")
        unknown = set(roles) - set(IMAGE_ROLE_FLAGS)
        if unknown:
            raise ValueError(
                f"Unknown image roles: {', '.join(sorted(unknown))}; "
                f"expected {', '.join(IMAGE_ROLE_FLAGS)}"
            )
        image = {role: fetch(str(roles[role])) for role in IMAGE_ROLE_FLAGS if role in roles}
        manifest = {
            "runtime": runtime,
            "server": str(BYOM_DIR / "sd-server"),
            "image": image,
            "downloaded": downloaded,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    # Perfis antigos: só baixa os padrões. Sem papéis não há como montar o argv
    # do sd-server, e `build_vllm_command` recusa em vez de adivinhar.
    for spec in specs:
        fetch(str(spec))
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
