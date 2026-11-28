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

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Runtime `strata`: run config JSON + argv do control
# ---------------------------------------------------------------------------
#
# O engine `strata-inkling` não recebe a linha de comando: o control lê um JSON
# (`--config`) com `exe`, `args`, `cwd`, `model_name`, `log` e `arch`, e sobe o
# binário com `--serve`. Os valores de `--kv` foram conferidos no `--help` do
# binário 0.1.70 (`auto|fp16|fp8|k8v8|k8v4`, e `int8` é recusado com
# "unknown --kv value 'int8'").


def _strata_files(tmp_path: Path) -> dict[str, Path]:
    control = tmp_path / "strata-control"
    (control / "python" / "bin").mkdir(parents=True)
    server = control / "python" / "bin" / "python3"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    engine_dir = tmp_path / "strata-engine"
    engine_dir.mkdir()
    engine = engine_dir / "strata-inkling"
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    pack = (
        tmp_path
        / "models"
        / "strata-engine"
        / "Inkling-2-Mini-Strata"
        / "Inkling-2-Mini-SQ-10.strata"
    )
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"pack")
    manifest = tmp_path / "runtime_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime": "strata",
                "server": str(server),
                "control_root": str(control),
                "engine": str(engine),
                "pack": str(pack),
                "sums_verified": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "control": control,
        "server": server,
        "engine": engine,
        "pack": pack,
        "manifest": manifest,
    }


def _strata_cfg(**worker: object) -> dict:
    return {
        "runtime": "strata",
        "model": "inkling-2-mini-strata",
        "served_model_name": "inkling-2-mini-strata",
        "max_model_len": 32768,
        "vllm_port": 8000,
        "kv_cache_dtype": "fp8",
        "worker": dict(worker),
    }


def test_strata_argv_e_run_config_com_pack_contexto_e_kv(build_module, tmp_path):
    arquivos = _strata_files(tmp_path)
    build_module.MANIFEST = str(arquivos["manifest"])

    argv = build_module.build_strata_args(_strata_cfg())

    assert argv == [
        str(arquivos["server"]),
        "-m",
        "serve.server",
        "--engine",
        "strata",
        "--config",
        str(tmp_path / "strata-inkling-2-mini-strata.json"),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--offline",
    ]

    config_json = json.loads((tmp_path / "strata-inkling-2-mini-strata.json").read_text())
    assert config_json["exe"] == str(arquivos["engine"])
    assert config_json["cwd"] == str(arquivos["engine"].parent)
    assert config_json["arch"] == "inkling2_loop"
    assert config_json["model_name"] == "inkling-2-mini-strata"
    assert config_json["log"].endswith("strata-engine.log")
    # O tokenizer é gerado pelo próprio servidor a partir do manifest do pack;
    # o diretório é explícito porque o default do control não existe no tarball.
    assert config_json["tokenizer"].endswith("strata-tokenizer")
    assert config_json["args"] == [
        "--native",
        str(arquivos["pack"]),
        "--max-context",
        "32768",
        "--kv",
        "fp8",
        "--parallel",
        "1",
    ]


def test_strata_respeita_kv_fp16_do_perfil_e_run_config_extra(build_module, tmp_path):
    arquivos = _strata_files(tmp_path)
    build_module.MANIFEST = str(arquivos["manifest"])
    cfg = _strata_cfg(run_config_extra=["--vram-reserve-mib", "512"])
    cfg["kv_cache_dtype"] = "fp16"

    build_module.build_strata_args(cfg)

    escrito = json.loads((tmp_path / "strata-inkling-2-mini-strata.json").read_text())
    assert _valor(escrito["args"], "--kv") == "fp16"
    assert escrito["args"][-2:] == ["--vram-reserve-mib", "512"]


def test_strata_recusa_kv_que_o_engine_nao_aceita(build_module, tmp_path):
    """`--kv int8` é do launcher, não do engine: o binário 0.1.70 o recusa."""
    arquivos = _strata_files(tmp_path)
    build_module.MANIFEST = str(arquivos["manifest"])
    cfg = _strata_cfg()
    cfg["kv_cache_dtype"] = "int8"

    with pytest.raises(ValueError) as excinfo:
        build_module.build_strata_args(cfg)

    assert "int8" in str(excinfo.value)
    assert "--kv" in str(excinfo.value)


def test_strata_sem_manifest_completo_recusa_sem_adivinhar(build_module, tmp_path):
    arquivos = _strata_files(tmp_path)
    build_module.MANIFEST = str(arquivos["manifest"])
    arquivos["pack"].unlink()

    with pytest.raises(RuntimeError) as excinfo:
        build_module.build_strata_args(_strata_cfg())

    assert "pack não existe" in str(excinfo.value)


def test_comando_strata_comeca_com_cd_no_control(tmp_path):
    """`-m serve.server` acha `serve`/`strata_control` pelo cwd: o cd vem antes."""
    arquivos = _strata_files(tmp_path)
    session = tmp_path / "session.json"
    session.write_text(json.dumps(_strata_cfg()), encoding="utf-8")

    resultado = subprocess.run(
        [sys.executable, str(WORKER_DIR / "build_vllm_command.py"), str(session)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "BYOM_RUNTIME_MANIFEST": str(arquivos["manifest"])},
    )

    assert resultado.returncode == 0, resultado.stderr
    linhas = resultado.stdout.splitlines()
    assert linhas[0] == f"cd {arquivos['control']}"
    assert linhas[1].startswith("exec ")
    argv = shlex.split(linhas[1].removeprefix("exec "))
    assert argv[0] == str(arquivos["server"])
    assert argv[argv.index("--config") + 1].endswith(".json")


# ---------------------------------------------------------------------------
# `prepare_runtime`: sha256 do pack contra o SHA256SUMS.txt do repo
# ---------------------------------------------------------------------------


@pytest.fixture
def strata_prepare(prepare_module, monkeypatch, tmp_path):
    """`prepare_runtime` sem rede: download, tree do HF e driver falsos."""
    import hashlib

    byom = tmp_path / "byom"
    monkeypatch.setattr(prepare_module, "BYOM_DIR", byom)
    monkeypatch.setattr(prepare_module, "MODEL_DIR", byom / "models")
    monkeypatch.setattr(prepare_module, "MANIFEST", byom / "runtime_manifest.json")
    monkeypatch.setattr(prepare_module, "check_nvidia_driver", lambda minimum="": "575.51")

    pack_nome = "Inkling-2-Mini-SQ-10.strata"
    conteudos = {
        "https://example.invalid/control.tar.gz": b"controle",
        "https://example.invalid/engine.tar.gz": b"engine",
        "https://example.invalid/SHA256SUMS.txt": b"",
    }
    estado = {"sums": "", "tree_oid": "", "sha_a_baixar": ""}

    def _download(url: str, destino: Path, sha: str | None = None) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("SHA256SUMS.txt"):
            corpo = estado["sums"].encode()
        elif url.endswith(pack_nome):
            corpo = b"pack real"
        elif url.endswith("tokenizer.json"):
            corpo = b'{"model": {"type": "BPE"}}'
        elif url.endswith("chat_template.jinja"):
            corpo = b"{{ messages }}"
        else:
            corpo = conteudos[url]
        if sha is not None and hashlib.sha256(corpo).hexdigest() != sha:
            raise ValueError(f"sha256 mismatch for {url}")
        destino.write_bytes(corpo)
        return destino

    def _extract(archive: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if archive.name == "strata-control.tar.gz":
            (dest / "python" / "bin").mkdir(parents=True)
            (dest / "python" / "bin" / "python3").write_text("#!/bin/sh\n")
        else:
            (dest / "strata-inkling").write_text("#!/bin/sh\n")

    monkeypatch.setattr(prepare_module, "download", _download)
    monkeypatch.setattr(prepare_module, "extract", _extract)
    monkeypatch.setattr(
        prepare_module,
        "hf_entries",
        lambda repo: [
            {
                "path": pack_nome,
                "size": 10_471_526_400,
                "sha256": estado["tree_oid"],
            },
            {"path": "tokenizer.json", "size": 26_178_678, "sha256": ""},
            {"path": "chat_template.jinja", "size": 6868, "sha256": ""},
        ],
    )
    estado["pack"] = pack_nome
    return prepare_module, estado


def _strata_session(tmp_path: Path, **worker: object) -> Path:
    dados = {
        "runtime": "strata",
        "worker": {
            "control_url": "https://example.invalid/control.tar.gz",
            "control_sha256": hashlib.sha256(b"controle").hexdigest(),
            "engine_url": "https://example.invalid/engine.tar.gz",
            "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
            "model_repo": "strata-engine/Inkling-2-Mini-Strata",
            "pack_pattern": "Inkling-2-Mini-SQ-10.strata",
            **worker,
        },
    }
    caminho = tmp_path / "session.json"
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    return caminho


def test_prepare_strata_confere_o_pack_contra_o_sums_do_repo(strata_prepare, tmp_path):
    prepare, estado = strata_prepare
    digest = hashlib.sha256(b"pack real").hexdigest()
    estado["sums"] = f"{digest}  {estado['pack']}\n"

    manifest = prepare.prepare(_strata_session(tmp_path))

    assert manifest["runtime"] == "strata"
    assert manifest["sums_verified"] is True
    assert manifest["pack_sha256"] == digest
    assert manifest["pack_sha256_source"] == "SHA256SUMS.txt"
    assert manifest["engine"].endswith("strata-inkling")
    assert manifest["server"].endswith("python/bin/python3")
    assert Path(manifest["pack"]).read_bytes() == b"pack real"
    # O tokenizer é gerado pelo servidor a partir destes dois arquivos, que
    # precisam estar AO LADO do pack: sem eles o `vocab.json` nunca nasce (o
    # CPython embarcado do control não lê zstd e o pack não traz a seção).
    assert manifest["tokenizer_files"] == ["tokenizer.json", "chat_template.jinja"]
    ao_lado = Path(manifest["pack"]).parent
    assert (ao_lado / "tokenizer.json").is_file()
    assert (ao_lado / "chat_template.jinja").is_file()


def test_prepare_strata_usa_o_lfs_oid_quando_o_sums_nao_lista_o_pack(strata_prepare, tmp_path):
    """O SQv2-20 da revisão atual não está no sums; o `lfs.oid` é o sha256."""
    prepare, estado = strata_prepare
    estado["sums"] = "0" * 64 + "  Inkling-2-Mini-SQ-15.strata\n"
    estado["tree_oid"] = hashlib.sha256(b"pack real").hexdigest()

    manifest = prepare.prepare(_strata_session(tmp_path))

    assert manifest["pack_sha256_source"] == "hf-tree-lfs"
    assert manifest["pack_sha256"] == estado["tree_oid"]


def test_prepare_strata_recusa_pack_com_digest_divergente(strata_prepare, tmp_path):
    prepare, estado = strata_prepare
    estado["sums"] = "1" * 64 + "  " + estado["pack"] + "\n"

    with pytest.raises(ValueError) as excinfo:
        prepare.prepare(_strata_session(tmp_path))

    assert "sha256 mismatch" in str(excinfo.value)


def test_prepare_strata_exige_os_dois_componentes_fixados(strata_prepare, tmp_path):
    prepare, estado = strata_prepare
    estado["sums"] = hashlib.sha256(b"pack real").hexdigest() + "  " + estado["pack"] + "\n"
    session = _strata_session(tmp_path)
    dados = json.loads(session.read_text(encoding="utf-8"))
    del dados["worker"]["engine_sha256"]
    session.write_text(json.dumps(dados), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        prepare.prepare(session)

    assert "engine_sha256" in str(excinfo.value)


def test_driver_mais_antigo_que_o_minimo_falha_antes_do_download(prepare_module, monkeypatch):
    monkeypatch.setattr(prepare_module, "nvidia_driver_version", lambda: "550.54.14")

    with pytest.raises(ValueError) as excinfo:
        prepare_module.check_nvidia_driver()

    assert "575.51" in str(excinfo.value)


def test_driver_igual_ou_mais_novo_passa(prepare_module, monkeypatch):
    for versao in ("575.51", "575.51.03", "580.65.06"):
        monkeypatch.setattr(prepare_module, "nvidia_driver_version", lambda v=versao: v)
        assert prepare_module.check_nvidia_driver() == versao
