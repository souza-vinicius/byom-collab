"""DSpark de ponta a ponta, com o perfil embarcado e com um perfil temporário.

O binário com suporte (`--spec-type draft-dspark`) é o upstream: o PR #32218
entrou no release **v1.1.0** e o build oficial `b12800` é o primeiro que o
serve, sem fork. O perfil embarcado `inkling-2-mini-ad-q6k-dspark` fixa esse
build (URL + sha256 do asset conferidos em 2026-11-26). O que este arquivo
prova é o encanamento e o pin:

    perfil embarcado -> dspark = true, drafter_pattern, b12800 + sha256
    perfil temporário (tmp_path) com dspark = true
      -> config.available_profiles recusa dspark sem drafter_pattern
      -> prepare_runtime.prepare resolve o drafter e grava manifest["drafter"]
      -> build_llama_args emite --model-draft/--spec-type/--spec-n-max/layers
      -> o orçamento soma o drafter (~0,47 GiB) em required_vram_gib
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from byom import catalog, config

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"


@pytest.fixture
def prepare_module():
    """`worker/prepare_runtime.py` como módulo (não é pacote)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "byom_prepare_runtime_dspark", WORKER_DIR / "prepare_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_module():
    """`worker/build_vllm_command.py` como módulo."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "byom_build_command_dspark", WORKER_DIR / "build_vllm_command.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

#: Mesmo repositório e mesmos padrões dos perfis Unsloth do Inkling 2 Mini: o
#: drafter publicado está na raiz (`dspark-Inkling-2-Mini-Q8_0.gguf`,
#: 503.889.952 B = 0,47 GiB, tree da API do HF em 2026-11-24).
DRAFTER = "dspark-Inkling-2-Mini-Q8_0.gguf"

PERFIL_DSPARK = """
name = "Inkling 2 Mini com DSpark (temporário)"
description = "Perfil de teste: dspark ligado, sem binário com suporte publicado."
opencode_output_limit = 8192
runtime = "llama-cpp"
catalog_model = "inkling-2-mini"
catalog_variant = "ad-q6k"

[config]
model = "inkling-2-mini"
served_model_name = "inkling-2-mini"
max_model_len = 131072
gpu = "A100"
gpu_high_mem = false
vllm_spec = "vllm"
vllm_enforce_eager = false
vllm_language_model_only = true
gpu_memory_utilization = 0.92
enable_auto_tool_choice = true
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"

[worker]
binary_url = "https://example.invalid/llama-dspark.tar.gz"
binary_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
binary_dir = "llama.cpp"
model_repo = "unsloth/Inkling-2-Mini-GGUF"
model_patterns = ["Mainline_Support/Inkling-2-Mini-UD-Q6_K_XL.gguf_file", "*mmproj-F16*"]
main_pattern = "Mainline_Support/Inkling-2-Mini-UD-Q6_K_XL.gguf_file"
mmproj_pattern = "mmproj-F16.gguf"
drafter_pattern = "dspark-Inkling-2-Mini-Q8_0.gguf"
dspark = true
llama_jinja = true
"""


def _perfil_temporario(tmp_path: Path, *, com_drafter: bool = True) -> tuple[Path, config.Profile]:
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    texto = PERFIL_DSPARK if com_drafter else PERFIL_DSPARK.replace(
        f'drafter_pattern = "{DRAFTER}"\n', ""
    )
    (pasta / "inkling-2-mini-dspark-teste.toml").write_text(texto, encoding="utf-8")
    (perfil,) = config.available_profiles(pasta)
    return pasta, perfil


def test_dspark_sem_drafter_pattern_falha_com_mensagem_clara(tmp_path: Path):
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    texto = PERFIL_DSPARK.replace(f'drafter_pattern = "{DRAFTER}"\n', "")
    (pasta / "quebrado.toml").write_text(texto, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)
    mensagem = str(excinfo.value)
    assert "dspark = true" in mensagem
    assert "drafter_pattern" in mensagem
    assert "quebrado.toml" in mensagem


def test_perfil_temporario_carrega_o_drafter_e_o_variant_do_catalogo(tmp_path: Path):
    _pasta, perfil = _perfil_temporario(tmp_path)

    assert perfil.worker is not None
    assert perfil.worker["dspark"] is True
    assert perfil.worker["drafter_pattern"] == DRAFTER
    assert perfil.catalog_model == "inkling-2-mini"
    assert perfil.catalog_variant == "ad-q6k"


def test_orcamento_soma_o_drafter_quando_dspark_esta_ligado():
    perfil = config.load_profile("inkling-2-mini-ad-q6k")
    sem_drafter = catalog.check_fit("inkling-2-mini", "ad-q6k", "A100-40", 131_072)
    com_drafter = catalog.check_fit(
        "inkling-2-mini",
        "ad-q6k",
        "A100-40",
        131_072,
        drafter_gib=catalog.drafter_gib({"dspark": True, "drafter_pattern": DRAFTER}),
    )

    custo = catalog.drafter_gib(perfil.worker or {})
    assert custo == 0.0  # o perfil sem dspark continua custando zero
    perfil_com_drafter = {"dspark": True, "drafter_pattern": DRAFTER}
    assert catalog.drafter_gib(perfil_com_drafter) == pytest.approx(0.47)
    assert com_drafter.required_gib == pytest.approx(sem_drafter.required_gib + 0.47)
    assert com_drafter.fits is True  # 26,88 + 0,47 + 2,0 ainda cabe nos 40 GiB

    # A variante q4 no limite da L4 deixa de caber com o drafter: 21,73 + 0,47
    # = 22,20, mais 1,2 de folga = 23,40 <= 24 — ainda cabe; quem estoura é o
    # caso de 128K: 22,73 + 0,47 = 23,20, mais 1,2 = 24,40 > 24.
    assert catalog.check_fit("inkling-2-mini", "q4", "L4", 131_072).fits
    assert not catalog.check_fit(
        "inkling-2-mini", "q4", "L4", 131_072, drafter_gib=0.47
    ).fits


def test_drafter_so_e_cobrado_quando_ha_arquivo_separado():
    """MTP embute a cabeça no arquivo principal: não há segundo peso na VRAM.

    O caso existe porque `[worker.spec]` permite `draft-mtp` sem
    `drafter_pattern`: o `-mtp.gguf` vai em `-m`, e cobrar 0,47 GiB de drafter
    ali seria dinheiro inventado no orçamento.
    """
    assert catalog.drafter_gib({"dspark": True, "drafter_pattern": DRAFTER}) == (
        pytest.approx(0.47)
    )
    assert catalog.drafter_gib({"drafter_pattern": DRAFTER}) == pytest.approx(0.47)
    assert (
        catalog.drafter_gib(
            {"spec": {"type": "draft-dflash", "drafter_pattern": "dflash.gguf"}}
        )
        == pytest.approx(0.47)
    )
    assert (
        catalog.drafter_gib(
            {"spec": {"type": "draft-mtp", "n_min": 0, "n_max": 3, "p_min": 0.1}}
        )
        == 0.0
    )
    assert catalog.drafter_gib({"dspark": True}) == 0.0


def test_check_compatibility_do_perfil_dspark_soma_o_drafter(tmp_path, monkeypatch):
    _pasta, perfil = _perfil_temporario(tmp_path)
    monkeypatch.setattr(config, "available_profiles", lambda *a, **k: [perfil])

    cfg = config.Config()
    config.apply_profile(cfg, perfil.id)
    assert cfg.max_model_len == 131_072

    fit = config.check_compatibility(cfg)
    base = catalog.check_fit("inkling-2-mini", "ad-q6k", "A100-40", 131_072)
    assert fit.required_gib == pytest.approx(base.required_gib + 0.47)
    assert fit.fits is True


def test_drafter_vai_para_o_manifest_e_para_o_argv(tmp_path, monkeypatch, prepare_module):
    """prepare_runtime grava o drafter; build_llama_args emite as flags DSpark."""
    _pasta, perfil = _perfil_temporario(tmp_path)
    worker = dict(perfil.worker or {})

    def _tarball() -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            data = b"#!/bin/sh\necho llama-server\n"
            info = tarfile.TarInfo("llama.cpp/llama-server")
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    tarball = _tarball()
    arquivos = [
        "Mainline_Support/Inkling-2-Mini-UD-Q6_K_XL.gguf_file",
        "mmproj-F16.gguf",
        DRAFTER,
    ]

    byom = tmp_path / "byom"
    monkeypatch.setattr(prepare_module, "BYOM_DIR", byom)
    monkeypatch.setattr(prepare_module, "MODEL_DIR", byom / "models")
    monkeypatch.setattr(prepare_module, "MANIFEST", byom / "runtime_manifest.json")
    monkeypatch.setattr(prepare_module, "hf_files", lambda repo: arquivos)

    baixados: list[tuple[str, Path]] = []

    def _download(url: str, destino: Path, sha: str | None = None) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        corpo = tarball if url.endswith(".tar.gz") else f"peso:{url}".encode()
        if sha is not None:
            assert hashlib.sha256(corpo).hexdigest() == sha
        destino.write_bytes(corpo)
        baixados.append((url, destino))

    monkeypatch.setattr(prepare_module, "download", _download)

    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "runtime": "llama-cpp",
                "model": "inkling-2-mini",
                "served_model_name": "inkling-2-mini",
                "max_model_len": 131_072,
                "vllm_port": 8000,
                "worker": {
                    **worker,
                    "binary_sha256": hashlib.sha256(tarball).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_module.prepare(session)

    assert manifest["drafter"].endswith(DRAFTER)
    assert Path(manifest["drafter"]).is_file()
    assert manifest["mmproj"].endswith("mmproj-F16.gguf")
    assert manifest["main"].endswith("Inkling-2-Mini-UD-Q6_K_XL.gguf")
    on_disk = json.loads((byom / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["drafter"] == manifest["drafter"]
    assert any(DRAFTER in url for url, _destino in baixados)

    # O argv real, montado pelo script do worker a partir do manifest.
    script = WORKER_DIR / "build_vllm_command.py"
    cfg = {
        **perfil.settings,
        "runtime": "llama-cpp",
        "vllm_port": 8000,
        "worker": worker,
    }
    resultado = subprocess.run(
        [sys.executable, str(script), "/dev/stdin"],
        input=json.dumps(cfg),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "BYOM_RUNTIME_MANIFEST": str(byom / "runtime_manifest.json")},
    )
    assert resultado.returncode == 0, resultado.stderr
    argv = shlex.split(resultado.stdout.removeprefix("exec ").strip())

    assert argv[0].endswith("llama-server")
    assert argv[argv.index("--model-draft") + 1] == manifest["drafter"]
    assert argv[argv.index("--spec-type") + 1] == "draft-dspark"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "3"
    assert argv[argv.index("--n-gpu-layers-draft") + 1] == "99"
    assert "--jinja" in argv
    assert argv[argv.index("--ctx-size") + 1] == "131072"


def test_build_llama_args_sem_manifest_do_drafter_falha_explicito(build_module):
    """dspark sem drafter no manifest não vira servidor sem spec: estoura claro."""
    cfg = {
        "runtime": "llama-cpp",
        "model": "x",
        "served_model_name": "x",
        "max_model_len": 1024,
        "vllm_port": 8000,
        "worker": {"main_pattern": "x.gguf", "dspark": True},
    }
    with pytest.raises(ValueError, match="drafter_pattern"):
        build_module.build_llama_args(cfg)


def test_padroes_do_perfil_embarcado_dspark_casam_com_os_arquivos_publicados(
    prepare_module,
):
    """O padrão do drafter é nome exato: não pega o `_Old.gguf` do upstream."""
    arquivos = [
        "Inkling-2-Mini-AD-Q6_K.gguf",
        "mmproj-Inkling-2-Mini-F16.gguf",
        "dspark-Inkling-2-Mini-Q8_0.gguf",
        "dspark-Inkling-2-Mini-Q8_0_Old.gguf",
    ]
    perfil = config.load_profile("inkling-2-mini-ad-q6k-dspark")
    worker = perfil.worker or {}

    assert (
        prepare_module.match_file(arquivos, worker["drafter_pattern"])
        == "dspark-Inkling-2-Mini-Q8_0.gguf"
    )
    for padrao in worker["model_patterns"]:
        assert prepare_module.match_file(arquivos, padrao) in arquivos


def test_perfil_embarcado_dspark_pina_o_build_oficial_b12800():
    """O perfil embarcado usa o upstream, não o fork do PR #32218."""
    perfil = config.load_profile("inkling-2-mini-ad-q6k-dspark")
    worker = perfil.worker or {}

    assert perfil.catalog_model == "inkling-2-mini"
    assert perfil.catalog_variant == "ad-q6k"
    assert perfil.settings["max_model_len"] == 131_072
    assert worker["dspark"] is True
    assert worker["drafter_pattern"] == DRAFTER
    assert DRAFTER in worker["model_patterns"]
    assert worker["binary_url"] == (
        "https://github.com/ggml-org/llama.cpp/releases/download/b12800/"
        "llama-b12800-bin-ubuntu-cuda-12.8-x64.tar.gz"
    )
    # Digest da API do GitHub conferido contra o arquivo baixado em 2026-11-26.
    assert worker["binary_sha256"] == (
        "088bc70b93c4f973de0b1e6e493603ed71f03b89f08e0c256f8f2290d587892d"
    )

    assert catalog.drafter_gib(worker) == pytest.approx(0.47)
    fit = catalog.check_fit(
        "inkling-2-mini", "ad-q6k", "A100-40", 131_072, drafter_gib=0.47
    )
    assert fit.fits is True
    assert fit.required_gib == pytest.approx(27.35)


def test_argv_do_perfil_embarcado_dspark(build_module, tmp_path: Path, monkeypatch):
    """O argv do perfil embarcado passa o drafter e as flags do draft-dspark."""
    perfil = config.load_profile("inkling-2-mini-ad-q6k-dspark")
    manifest = tmp_path / "runtime_manifest.json"
    arquivos = {
        "server": tmp_path / "llama-server",
        "main": tmp_path / "Inkling-2-Mini-AD-Q6_K.gguf",
        "mmproj": tmp_path / "mmproj-Inkling-2-Mini-F16.gguf",
        "drafter": tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf",
    }
    manifest.write_text(
        json.dumps({chave: str(valor) for chave, valor in arquivos.items()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_module, "MANIFEST", manifest)

    argv = build_module.build_llama_args(
        {
            **perfil.settings,
            "runtime": "llama-cpp",
            "vllm_port": 8000,
            "worker": perfil.worker,
        }
    )

    assert argv[argv.index("--model") + 1] == str(arquivos["main"])
    assert argv[argv.index("--model-draft") + 1] == str(arquivos["drafter"])
    assert argv[argv.index("--spec-type") + 1] == "draft-dspark"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "3"
    assert argv[argv.index("--n-gpu-layers-draft") + 1] == "99"
    assert argv[argv.index("--ctx-size") + 1] == "131072"
