"""argv do llama-server: flash attention, sampling completo e `[worker.spec]`.

Flags conferidas no binário fixado, não na memória: o `llama-server --help` do
fork PrismML `prism-b12128-d7b2896` (build 12128, commit `d7b289683`) lista

    -fa,   --flash-attn [on|off|auto]       set Flash Attention use
    --presence-penalty N                    repeat alpha presence penalty
    --repeat-penalty N                      penalize repeat sequence of tokens
    --repeat-last-n N                       last n tokens to consider for penalize
    --spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,...
    --spec-draft-n-min N / --spec-draft-n-max N / --spec-draft-p-min P
    -md, --model-draft FNAME / -ngld, --n-gpu-layers-draft N

(asset `llama-prism-b12128-d7b2896-bin-ubuntu-x64.tar.gz`, rodado localmente em
2026-11-24).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from byom import config

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"


@pytest.fixture
def build_module():
    """`worker/build_vllm_command.py` como módulo (não é pacote)."""
    spec = importlib.util.spec_from_file_location(
        "byom_build_command_llama", WORKER_DIR / "build_vllm_command.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(**worker: object) -> dict:
    return {
        "runtime": "llama-cpp",
        "model": "ternary-bonsai-2.5-27b",
        "served_model_name": "ternary-bonsai-2.5-27b",
        "max_model_len": 32768,
        "vllm_port": 8000,
        "worker": {"main_pattern": "Ternary-Bonsai-2.5-27B-Q2_0.gguf", **worker},
    }


def _valor(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# Flash attention
# ---------------------------------------------------------------------------


def test_flash_attn_emite_a_flag_curta_com_on(build_module):
    """`-fa on` é a forma que os docs do publicador usam e o binário aceita."""
    argv = build_module.build_llama_args(_cfg(flash_attn=True))

    assert _valor(argv, "-fa") == "on"
    assert "--flash-attn" not in argv


def test_flash_attn_ausente_quando_o_perfil_nao_declara(build_module):
    argv = build_module.build_llama_args(_cfg())

    assert "-fa" not in argv
    assert "--flash-attn" not in argv


# ---------------------------------------------------------------------------
# Sampling: temp/top_p/top_k/min_p + presença e repetição
# ---------------------------------------------------------------------------


def test_sampling_emite_presenca_repeticao_e_repeat_last_n(build_module):
    argv = build_module.build_llama_args(
        _cfg(
            sampling={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "repeat_last_n": 64,
            }
        )
    )

    assert _valor(argv, "--temp") == "1.0"
    assert _valor(argv, "--top-p") == "0.95"
    assert _valor(argv, "--top-k") == "20"
    assert _valor(argv, "--min-p") == "0.0"
    assert _valor(argv, "--presence-penalty") == "0.0"
    assert _valor(argv, "--repeat-penalty") == "1.0"
    assert _valor(argv, "--repeat-last-n") == "64"


def test_perfil_com_flash_attn_e_sampling_carrega(tmp_path: Path):
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    (pasta / "bonsai-teste.toml").write_text(
        """
name = "Bonsai (teste)"
runtime = "llama-cpp"
catalog_model = "qwen3.8-27b-gguf"
catalog_variant = "q3"

[config]
model = "ternary-bonsai-2.5-27b"
served_model_name = "ternary-bonsai-2.5-27b"
max_model_len = 32768
gpu = "L4"
gpu_high_mem = false
vllm_spec = "vllm"
vllm_enforce_eager = false
vllm_language_model_only = true
gpu_memory_utilization = 0.92
enable_auto_tool_choice = true
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"

[worker]
binary_url = "https://example.invalid/llama.tar.gz"
binary_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
binary_dir = "llama.cpp"
model_repo = "prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
main_pattern = "Ternary-Bonsai-2.5-27B-Q2_0.gguf"
llama_jinja = true
flash_attn = true

[worker.sampling]
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
""",
        encoding="utf-8",
    )

    (perfil,) = config.available_profiles(pasta)

    assert perfil.worker is not None
    assert perfil.worker["flash_attn"] is True
    assert perfil.worker["sampling"]["top_k"] == 20


@pytest.mark.parametrize(
    ("linha", "esperado"),
    [
        ('flash_attn = "sim"', "flash_attn"),
        ("[worker.sampling]\ntemperature = 1.0\ntop_q = 0.95", "top_q"),
    ],
)
def test_perfil_com_tipo_invalido_falha_no_load(tmp_path: Path, linha: str, esperado: str):
    """`flash_attn` não booleano e chave de sampling desconhecida não passam."""
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    texto = """
name = "Bonsai (teste)"
runtime = "llama-cpp"

[config]
model = "ternary-bonsai-2.5-27b"
served_model_name = "ternary-bonsai-2.5-27b"
max_model_len = 32768
gpu = "L4"
gpu_high_mem = false
vllm_spec = "vllm"
vllm_enforce_eager = false
vllm_language_model_only = true
gpu_memory_utilization = 0.92
enable_auto_tool_choice = true
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"

[worker]
binary_url = "https://example.invalid/llama.tar.gz"
binary_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
binary_dir = "llama.cpp"
model_repo = "prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
main_pattern = "Ternary-Bonsai-2.5-27B-Q2_0.gguf"
"""
    texto += linha + "\n"
    (pasta / "invalido.toml").write_text(texto, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert esperado in str(excinfo.value)
    assert "invalido.toml" in str(excinfo.value)


# ---------------------------------------------------------------------------
# `[worker.spec]`: decodificação especulativa generalizada
# ---------------------------------------------------------------------------

BASE_PERFIL = """
name = "Spec (teste)"
runtime = "llama-cpp"

[config]
model = "ternary-bonsai-2.5-27b"
served_model_name = "ternary-bonsai-2.5-27b"
max_model_len = 32768
gpu = "L4"
gpu_high_mem = false
vllm_spec = "vllm"
vllm_enforce_eager = false
vllm_language_model_only = true
gpu_memory_utilization = 0.92
enable_auto_tool_choice = true
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"

[worker]
binary_url = "https://example.invalid/llama.tar.gz"
binary_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
binary_dir = "llama.cpp"
model_repo = "prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
main_pattern = "Ternary-Bonsai-2.5-27B-Q2_0.gguf"
"""


def _perfil(tmp_path: Path, extra: str) -> tuple[Path, config.Profile | None]:
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    (pasta / "spec.toml").write_text(BASE_PERFIL + extra, encoding="utf-8")
    return pasta, None


def test_spec_mtp_emite_as_flags_sem_model_draft(build_module):
    """MTP carrega a cabeça embutida no próprio arquivo: `-m`, nunca `-md`.

    Valores do preset do publicador (`presets/llama.cpp/mtp-cuda.json`, medido
    no build `prism-b12128-d7b2896`): n-min 0, n-max 3, p-min 0.1.
    """
    argv = build_module.build_llama_args(
        _cfg(
            spec={"type": "draft-mtp", "n_min": 0, "n_max": 3, "p_min": 0.1},
            main_pattern="Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf",
        )
    )

    assert _valor(argv, "--spec-type") == "draft-mtp"
    assert _valor(argv, "--spec-draft-n-min") == "0"
    assert _valor(argv, "--spec-draft-n-max") == "3"
    assert _valor(argv, "--spec-draft-p-min") == "0.1"
    assert "--model-draft" not in argv
    assert "--n-gpu-layers-draft" not in argv
    # O modelo principal continua sendo o arquivo `-mtp.gguf`.
    assert _valor(argv, "--model") == "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf"


def test_spec_com_drafter_pattern_emite_md_e_ngld(build_module):
    argv = build_module.build_llama_args(
        _cfg(
            spec={
                "type": "draft-dflash",
                "n_max": 4,
                "drafter_pattern": "Bonsai-DFlash2-Q8_0.gguf",
            }
        )
    )

    assert _valor(argv, "--model-draft") == "Bonsai-DFlash2-Q8_0.gguf"
    assert _valor(argv, "--spec-type") == "draft-dflash"
    assert _valor(argv, "--spec-draft-n-max") == "4"
    assert _valor(argv, "--n-gpu-layers-draft") == "99"


def test_dspark_legado_produz_o_mesmo_argv_de_antes(build_module):
    """`dspark = true` continua valendo, com o argv idêntico ao da versão antiga."""
    cfg = _cfg(dspark=True, drafter_pattern="dspark-Inkling-2-Mini-Q8_0.gguf")
    argv = build_module.build_llama_args(cfg)

    esperado = [
        "/content/byom/llama.cpp/llama-server",
        "--model",
        "Ternary-Bonsai-2.5-27B-Q2_0.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--ctx-size",
        "32768",
        "--alias",
        "ternary-bonsai-2.5-27b",
        "--model-draft",
        "dspark-Inkling-2-Mini-Q8_0.gguf",
        "--spec-type",
        "draft-dspark",
        "--spec-draft-n-max",
        "3",
        "--n-gpu-layers-draft",
        "99",
    ]
    assert argv == esperado


def test_spec_sem_type_falha_no_load_do_perfil(tmp_path: Path):
    pasta, _ = _perfil(tmp_path, '\n[worker.spec]\nn_max = 3\n')

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "type" in str(excinfo.value)
    assert "spec.toml" in str(excinfo.value)


def test_spec_sem_type_falha_tambem_no_argv(build_module):
    with pytest.raises(ValueError, match="type"):
        build_module.build_llama_args(_cfg(spec={"n_max": 3}))


def test_dspark_e_spec_juntos_falham(tmp_path: Path):
    pasta, _ = _perfil(
        tmp_path,
        '\ndspark = true\ndrafter_pattern = "d.guf"\n\n[worker.spec]\ntype = "draft-mtp"\n',
    )

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "dspark" in str(excinfo.value)
    assert "spec" in str(excinfo.value)


def test_spec_exige_runtime_llama_cpp(tmp_path: Path):
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    (pasta / "spec.toml").write_text(
        BASE_PERFIL.replace('runtime = "llama-cpp"', 'runtime = "vllm"')
        + '\n[worker.spec]\ntype = "draft-mtp"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "llama-cpp" in str(excinfo.value)


def test_spec_de_tipo_sem_cabeca_embutida_exige_drafter(tmp_path: Path):
    pasta, _ = _perfil(tmp_path, '\n[worker.spec]\ntype = "draft-dspark"\n')

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "drafter_pattern" in str(excinfo.value)


def test_spec_com_n_min_maior_que_n_max_falha(tmp_path: Path):
    pasta, _ = _perfil(
        tmp_path, '\n[worker.spec]\ntype = "draft-mtp"\nn_min = 5\nn_max = 2\n'
    )

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "n_min" in str(excinfo.value)


def test_spec_recusa_chave_desconhecida(tmp_path: Path):
    pasta, _ = _perfil(tmp_path, '\n[worker.spec]\ntype = "draft-mtp"\nn_maxx = 3\n')

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "n_maxx" in str(excinfo.value)


def test_perfil_mtp_temporario_carrega_a_spec(tmp_path: Path):
    pasta, _ = _perfil(
        tmp_path,
        '\nflash_attn = true\n\n[worker.spec]\ntype = "draft-mtp"\nn_min = 0\n'
        "n_max = 3\np_min = 0.1\n",
    )

    (perfil,) = config.available_profiles(pasta)

    assert perfil.worker is not None
    assert perfil.worker["spec"] == {
        "type": "draft-mtp",
        "n_min": 0,
        "n_max": 3,
        "p_min": 0.1,
    }


@pytest.fixture
def prepare_module():
    """`worker/prepare_runtime.py` como módulo (não é pacote)."""
    spec = importlib.util.spec_from_file_location(
        "byom_prepare_runtime_llama", WORKER_DIR / "prepare_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_carrega_o_mtp_como_modelo_e_nao_como_drafter(
    tmp_path, monkeypatch, prepare_module
):
    """O `-mtp.gguf` vai para `main`; nenhum `drafter` é gravado no manifest."""
    import hashlib
    import io
    import json
    import tarfile

    arquivos = [
        "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf",
        "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf",
    ]

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
    byom = tmp_path / "byom"
    monkeypatch.setattr(prepare_module, "BYOM_DIR", byom)
    monkeypatch.setattr(prepare_module, "MODEL_DIR", byom / "models")
    monkeypatch.setattr(prepare_module, "MANIFEST", byom / "runtime_manifest.json")
    monkeypatch.setattr(prepare_module, "hf_files", lambda repo: arquivos)

    def _download(url: str, destino: Path, sha: str | None = None) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        corpo = tarball if url.endswith(".tar.gz") else f"peso:{url}".encode()
        if sha is not None:
            assert hashlib.sha256(corpo).hexdigest() == sha
        destino.write_bytes(corpo)

    monkeypatch.setattr(prepare_module, "download", _download)

    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "runtime": "llama-cpp",
                "model": "ternary-bonsai-2.5-27b",
                "served_model_name": "ternary-bonsai-2.5-27b",
                "max_model_len": 65536,
                "vllm_port": 8000,
                "worker": {
                    "binary_url": "https://example.invalid/llama.tar.gz",
                    "binary_sha256": hashlib.sha256(tarball).hexdigest(),
                    "binary_dir": "llama.cpp",
                    "model_repo": "prism-ml/Ternary-Bonsai-2.5-27B-GGUF",
                    "main_pattern": "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf",
                    "mmproj_pattern": "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf",
                    "spec": {"type": "draft-mtp", "n_min": 0, "n_max": 3, "p_min": 0.1},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_module.prepare(session)

    assert manifest["main"].endswith("PTQ1_0-mtp.gguf")
    assert "drafter" not in manifest


def test_sonda_de_argv_imprime_o_argv_dos_tres_perfis_do_bonsai():
    """`scripts/llama_argv.py` mostra o argv real; o MTP não tem `--model-draft`."""
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parent.parent / "scripts" / "llama_argv.py"
    resultado = subprocess.run(
        [
            sys.executable,
            str(script),
            "ternary-bonsai-25-27b-q2",
            "ternary-bonsai-25-27b-ptq1",
            "ternary-bonsai-25-27b-ptq1-mtp",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "BYOM_CONFIG_DIR": os.devnull},
    )

    assert resultado.returncode == 0, resultado.stderr
    linhas = {
        linha.split("—")[0].strip().removeprefix("# "): linha
        for linha in resultado.stdout.splitlines()
        if linha.startswith("# ")
    }
    assert set(linhas) == {
        "ternary-bonsai-25-27b-q2",
        "ternary-bonsai-25-27b-ptq1",
        "ternary-bonsai-25-27b-ptq1-mtp",
    }
    assert "-fa on" in resultado.stdout
    assert "--min-p 0.0" in resultado.stdout
    assert "--spec-type draft-mtp" in resultado.stdout
    assert "--model-draft" not in resultado.stdout
