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
