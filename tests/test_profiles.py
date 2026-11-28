from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from byom import catalog, cli, config, opencode

PROFILE_IDS = {
    "qwen3.8-27b-32k",
    "qwen3.8-27b-256k",
    "inkling-2-mini-iq2-gguf",
    "inkling-2-mini-q4-gguf",
    "inkling-2-mini-q8-gguf",
    "inkling-2-mini-ad-q6k",
    "inkling-2-mini-ad-q6k-dspark",
    "inkling-2-mini-ad-q4k",
    "inkling-2-mini-ad-iq2xxs",
    "qwen3.8-27b-q3-gguf",
    "qwen3.8-27b-q4-gguf",
    "gemma-4-e4b-q4-gguf",
    "gemma-4-12b-q4-gguf",
    "gemma-4-26b-a4b-q4-gguf",
    "gemma-4-31b-q4-gguf",
    "muse-glimmer-30b-q2-gguf",
    "muse-glimmer-30b-q4-gguf",
    "qwen-image-2.1-q4km",
    "qwen-image-2.1-fp8",
    "qwen-image-2.1-turbo-uncensored-q4k",
    "qwen-image-2.1-turbo-uncensored-q8",
    "ternary-bonsai-25-27b-q2",
    "ternary-bonsai-25-27b-ptq1",
    "ternary-bonsai-25-27b-ptq1-mtp",
    "inkling-2-mini-strata-sq10",
    "inkling-2-mini-strata-sq15",
    "inkling-2-mini-strata-sq20",
    "qwen3-tts-0.6b-customvoice",
    "qwen3-tts-1.7b-customvoice",
}

#: Perfis que não entram nos aliases do OpenCode: imagem e TTS não conversam.
NON_CHAT_PROFILE_IDS = {
    "qwen-image-2.1-q4km",
    "qwen-image-2.1-fp8",
    "qwen-image-2.1-turbo-uncensored-q4k",
    "qwen-image-2.1-turbo-uncensored-q8",
    "qwen3-tts-0.6b-customvoice",
    "qwen3-tts-1.7b-customvoice",
}


def test_templates_definem_os_dois_contextos_e_aliases_opencode():
    profiles = {profile.id: profile for profile in config.available_profiles()}
    assert set(profiles) == PROFILE_IDS
    assert profiles["qwen3.8-27b-32k"].settings["max_model_len"] == 32768
    assert profiles["qwen3.8-27b-256k"].settings["max_model_len"] == 262144
    assert all(
        profiles[p].settings["gpu"] == "A100" for p in ("qwen3.8-27b-32k", "qwen3.8-27b-256k")
    )
    assert profiles["qwen3.8-27b-32k"].settings["gpu_high_mem"] is False
    assert profiles["qwen3.8-27b-256k"].settings["gpu_high_mem"] is True

    aliases = opencode.profile_aliases_from_config()
    assert set(aliases) == PROFILE_IDS - NON_CHAT_PROFILE_IDS
    assert aliases["qwen3.8-27b-32k"]["id"] == "qwen3.8-27b"
    assert aliases["qwen3.8-27b-32k"]["limit"]["context"] == 32768
    assert aliases["qwen3.8-27b-256k"]["limit"]["context"] == 262144
    assert all(alias["limit"]["output"] == 8192 for alias in aliases.values())


def test_apply_profile_preserves_local_connection_settings():
    cfg = config.Config(proxy_port=4550, tunnel_port=4551, ssh_key="~/custom_key")

    profile = config.apply_profile(cfg, "qwen3.8-27b-256k")

    assert profile.opencode_model == "byom/qwen3.8-27b-256k"
    assert cfg.profile == profile.id
    assert cfg.max_model_len == 262144
    assert cfg.proxy_port == 4550
    assert cfg.tunnel_port == 4551
    assert cfg.ssh_key == "~/custom_key"


def test_load_profile_rejects_path_input():
    try:
        config.load_profile("../qwen3.8-27b-32k")
    except ValueError as exc:
        assert "Invalid profile ID" in str(exc)
    else:
        raise AssertionError("path-like profile ID should be rejected")


def test_configure_cli_saves_selected_profile():
    result = CliRunner().invoke(cli.app, ["configure", "--profile", "qwen3.8-27b-32k"])

    assert result.exit_code == 0, result.output
    assert "byom/qwen3.8-27b-32k" in result.output
    assert "standard memory" in result.output
    saved = config.load_config()
    assert saved.profile == "qwen3.8-27b-32k"
    assert saved.max_model_len == 32768


def test_worker_payload_and_command_use_profile_values(tmp_path: Path):
    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-32k")
    tar_path = tmp_path / "worker.tar.gz"
    state = config.State(session="s", endpoint="", nonce="test", created_at="now")

    cli._build_worker_tar(cfg, state, tar_path)

    with tarfile.open(tar_path, "r:gz") as archive:
        payload = json.load(archive.extractfile("session.json"))
    assert payload["served_model_name"] == "qwen3.8-27b"
    assert payload["max_model_len"] == 32768
    assert payload["gpu_memory_utilization"] == 0.92
    assert payload["kv_cache_dtype"] == "fp8"

    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps(payload), encoding="utf-8")
    command_script = Path(cli._worker_dir()) / "build_vllm_command.py"
    result = subprocess.run(
        [sys.executable, str(command_script), str(session_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    args = shlex.split(result.stdout.removeprefix("exec ").strip())
    assert "--max-model-len" in args
    assert args[args.index("--max-model-len") + 1] == "32768"
    assert args[args.index("--served-model-name") + 1] == "qwen3.8-27b"
    assert "--tool-call-parser" in args


def test_perfis_llama_cpp_nao_declaram_kv_cache_dtype():
    """O llama-server não recebe --cache-type-k/-v: o KV é f16 e o campo confunde."""
    for profile in config.available_profiles():
        if profile.runtime == "llama-cpp":
            assert "kv_cache_dtype" not in profile.settings, profile.id
        elif profile.runtime == "vllm":
            assert profile.settings.get("kv_cache_dtype") == "fp8", profile.id


def test_perfis_nao_repetem_reasoning_nem_steps():
    for profile in config.available_profiles():
        assert "reasoning_effort" not in profile.settings, profile.id
        assert "steps" not in (profile.worker or {}), profile.id
        assert catalog.profile_context(profile.id).context_served == (
            profile.settings["max_model_len"]
        ) or profile.catalog_model in {"qwen3.8-27b-awq"}


@pytest.mark.parametrize(
    ("profile_id", "effort", "has_thinking", "has_generation"),
    [
        ("inkling-2-mini-q4-gguf", "high", False, False),
        ("qwen3.8-27b-32k", "xhigh", False, False),
        ("gemma-4-12b-q4-gguf", None, True, False),
        ("muse-glimmer-30b-q4-gguf", "high", False, False),
        ("qwen-image-2.1-q4km", None, False, True),
        ("qwen-image-2.1-turbo-uncensored-q4k", None, False, True),
    ],
)
def test_payload_da_sessao_propagas_propriedades_do_modelo(
    tmp_path: Path, profile_id: str, effort, has_thinking: bool, has_generation: bool
):
    cfg = config.Config()
    config.apply_profile(cfg, profile_id)
    tar_path = tmp_path / "worker.tar.gz"
    state = config.State(session="s", endpoint="", nonce="test", created_at="now")

    cli._build_worker_tar(cfg, state, tar_path)

    with tarfile.open(tar_path, "r:gz") as archive:
        payload = json.load(archive.extractfile("session.json"))
    assert payload["reasoning_effort"] == effort
    assert (payload["thinking_policy"] is not None) is has_thinking
    assert (payload["generation_defaults"] is not None) is has_generation
    if has_thinking:
        assert payload["thinking_policy"]["system_token"] == "<|think|>"
        assert payload["thinking_policy"]["strip_history_thought"] is True
    if has_generation:
        model_id, _variant_id = config.active_catalog_ref(cfg)
        assert payload["generation_defaults"] == catalog.model_by_id(model_id).generation_defaults
        assert "steps" not in (payload["worker"] or {})


# ---------------------------------------------------------------------------
# Perfis AD da AtomicChat para o Inkling 2 Mini
# ---------------------------------------------------------------------------


def test_perfis_atomicchat_reaproveitam_build_e_baixam_do_repo_certo():
    perfis = {profile.id: profile for profile in config.available_profiles()}
    irmao = perfis["inkling-2-mini-q4-gguf"].worker
    assert irmao is not None
    esperado = {
        "inkling-2-mini-ad-q6k": ("Inkling-2-Mini-AD-Q6_K.gguf", "A100", False, 131_072),
        "inkling-2-mini-ad-q4k": ("Inkling-2-Mini-AD-Q4_K.gguf", "L4", False, 32_768),
        "inkling-2-mini-ad-iq2xxs": ("Inkling-2-Mini-AD-IQ2_XXS.gguf", "T4", False, 32_768),
    }
    for profile_id, (arquivo, gpu, high_mem, contexto) in esperado.items():
        profile = perfis[profile_id]
        assert profile.catalog_model == "inkling-2-mini"
        assert profile.settings["gpu"] == gpu
        assert bool(profile.settings["gpu_high_mem"]) is high_mem
        assert profile.settings["max_model_len"] == contexto
        worker = profile.worker or {}
        assert worker["model_repo"] == "AtomicChat/Inkling-2-Mini-GGUF"
        assert worker["main_pattern"] == arquivo  # nome exato, sem subpasta
        assert worker["mmproj_pattern"] == "mmproj-Inkling-2-Mini-F16.gguf"
        assert worker["binary_dir"] == "llama.cpp"
        assert worker["binary_url"] == irmao["binary_url"]
        assert worker["binary_sha256"] == irmao["binary_sha256"]
        assert worker["llama_jinja"] is True
        # O layout antigo (PR #32695) não carrega em b12611: proibido no perfil.
        assert "Q4_K_M" not in arquivo and "Q5_K_M" not in arquivo
        # reasoning_effort herda do catálogo (o gateway injeta em
        # chat_template_kwargs), como nos perfis irmãos da Unsloth.
        assert "reasoning_effort" not in profile.settings
        cfg = config.Config()
        config.apply_profile(cfg, profile_id)
        fit = config.check_compatibility(cfg)
        assert fit.fits, (profile_id, fit.reason)
        assert fit.available_gib - fit.required_gib >= fit.headroom_gib


def test_perfis_atomicchat_nao_alteram_os_perfis_unsloth():
    perfis = {profile.id: profile for profile in config.available_profiles()}
    unsloth = perfis["inkling-2-mini-iq2-gguf"]
    assert unsloth.worker is not None
    assert unsloth.worker["model_repo"] == "unsloth/Inkling-2-Mini-GGUF"
    assert unsloth.worker["main_pattern"] == "Mainline_Support/Inkling-2-Mini-UD-IQ2_XXS.gguf_file"
    assert unsloth.settings["gpu"] == "T4"


# ---------------------------------------------------------------------------
# Runtime `strata`: perfis, campos proibidos e dtype do KV
# ---------------------------------------------------------------------------

_STRATA_TOML = """
name = "Inkling 2 Mini (Strata teste)"
runtime = "strata"
catalog_model = "inkling-2-mini-strata"
catalog_variant = "sq10"

[config]
model = "inkling-2-mini-strata"
served_model_name = "inkling-2-mini-strata"
max_model_len = 32768
gpu = "T4"
gpu_high_mem = false
vllm_spec = "vllm"
vllm_enforce_eager = false
vllm_language_model_only = true
gpu_memory_utilization = 0.92
kv_cache_dtype = "{kv}"
enable_auto_tool_choice = true
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"

[worker]
control_url = "https://example.invalid/control.tar.gz"
control_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
control_dir = "strata-control"
engine_url = "https://example.invalid/engine.tar.gz"
engine_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
engine_dir = "strata-engine"
model_repo = "strata-engine/Inkling-2-Mini-Strata"
pack_pattern = "Inkling-2-Mini-SQ-10.strata"
sums_pattern = "SHA256SUMS.txt"
{extra}
"""


def _perfil_strata(tmp_path: Path, extra: str = "", kv: str = "fp8"):
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    (pasta / "strata-teste.toml").write_text(
        _STRATA_TOML.format(kv=kv, extra=extra), encoding="utf-8"
    )
    return config.available_profiles(pasta)


def test_perfis_strata_pinam_engine_control_e_pack(tmp_path: Path):
    perfis = {p.id: p for p in config.available_profiles()}
    esperado = {
        "inkling-2-mini-strata-sq10": ("Inkling-2-Mini-SQ-10.strata", "T4"),
        "inkling-2-mini-strata-sq15": ("Inkling-2-Mini-SQ-15.strata", "L4"),
        "inkling-2-mini-strata-sq20": ("Inkling-2-Mini-SQv2-20.strata", "L4"),
    }
    for profile_id, (pack, gpu) in esperado.items():
        profile = perfis[profile_id]
        assert profile.runtime == "strata"
        assert profile.catalog_model == "inkling-2-mini-strata"
        assert profile.settings["kv_cache_dtype"] == "fp8"
        assert profile.settings["served_model_name"] == "inkling-2-mini-strata"
        assert profile.settings["gpu"] == gpu
        assert profile.settings["max_model_len"] == 32_768
        worker = profile.worker or {}
        assert worker["engine_url"].endswith(
            "v0.2.23/strata-inkling-0.1.70-linux-x86_64-cuda-cuda12.tar.gz"
        )
        assert worker["control_url"].endswith(
            "v0.2.23/strata-control-0.1.70-linux-x86_64.tar.gz"
        )
        assert worker["engine_sha256"] == (
            "cc078a9cf15e5e738a0396973748cbab352aed5f9cdf3a892802dfd407c6e526"
        )
        assert worker["control_sha256"] == (
            "f38d85bd5bd19393808e2bd0c7867e34c4fed968065371d77f37c11c451eeb6a"
        )
        assert worker["engine_dir"] == "strata-engine"
        assert worker["control_dir"] == "strata-control"
        assert worker["model_repo"] == "strata-engine/Inkling-2-Mini-Strata"
        assert worker["pack_pattern"] == pack
        assert worker["sums_pattern"] == "SHA256SUMS.txt"
        # A descrição precisa avisar do driver mínimo e da ausência de visão.
        assert "575.51" in profile.description
        assert "sem visão" in profile.description
        cfg = config.Config()
        config.apply_profile(cfg, profile_id)
        assert cfg.runtime == "strata"
        fit = config.check_compatibility(cfg)
        assert fit.fits, (profile_id, fit.reason)
        assert fit.available_gib - fit.required_gib >= fit.headroom_gib


def test_perfil_strata_valido_carrega(tmp_path: Path):
    (perfil,) = _perfil_strata(tmp_path, extra='run_config_extra = ["--vram-reserve-mib", "512"]')

    assert perfil.runtime == "strata"
    assert perfil.settings["kv_cache_dtype"] == "fp8"
    assert perfil.worker["run_config_extra"] == ["--vram-reserve-mib", "512"]


@pytest.mark.parametrize(
    ("extra", "esperado"),
    [
        ('drafter_pattern = "dspark.gguf"', "drafter_pattern"),
        ("dspark = true", "dspark"),
        ('main_pattern = "modelo.gguf"', "main_pattern"),
        ('mmproj_pattern = "mmproj.gguf"', "mmproj_pattern"),
    ],
)
def test_perfil_strata_recusa_campos_do_llama_cpp(tmp_path: Path, extra: str, esperado: str):
    with pytest.raises(ValueError) as excinfo:
        _perfil_strata(tmp_path, extra=extra)

    assert "not valid for runtime 'strata'" in str(excinfo.value)
    assert esperado in str(excinfo.value)


def test_perfil_strata_recusa_image_roles(tmp_path: Path):
    extra = '\n[worker.image_roles]\ndiffusion = "a/b::c.gguf"'
    with pytest.raises(ValueError) as excinfo:
        _perfil_strata(tmp_path, extra=extra)

    assert "image_roles" in str(excinfo.value)


def test_perfil_strata_recusa_worker_spec(tmp_path: Path):
    extra = '\n[worker.spec]\ntype = "draft-dspark"\nn_max = 3'
    with pytest.raises(ValueError) as excinfo:
        _perfil_strata(tmp_path, extra=extra)

    assert "[worker.spec]" in str(excinfo.value)
    assert "llama-cpp" in str(excinfo.value)


def test_perfil_strata_exige_os_componentes_fixados(tmp_path: Path):
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    texto = _STRATA_TOML.format(kv="fp8", extra="").replace(
        'engine_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"\n', ""
    )
    (pasta / "strata-teste.toml").write_text(texto, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "engine_sha256" in str(excinfo.value)


@pytest.mark.parametrize("kv", ["fp8", "fp16"])
def test_perfil_strata_aceita_os_dtypes_do_engine(tmp_path: Path, kv: str):
    (perfil,) = _perfil_strata(tmp_path, kv=kv)
    assert perfil.settings["kv_cache_dtype"] == kv


@pytest.mark.parametrize("kv", ["int8", "k8v8", "k8v4", "q4_0"])
def test_perfil_strata_recusa_dtypes_que_o_engine_nao_aceita(tmp_path: Path, kv: str):
    """`int8` é do launcher; `k8v8`/`k8v4`/`q4_0` não têm fator no catálogo."""
    with pytest.raises(ValueError) as excinfo:
        _perfil_strata(tmp_path, kv=kv)

    assert "runtime 'strata'" in str(excinfo.value)
    assert kv in str(excinfo.value)


def test_perfis_de_imagem_e_llama_cpp_continuam_proibidos_de_declarar_kv(tmp_path: Path):
    pasta = tmp_path / "profiles"
    pasta.mkdir()
    texto = (
        _STRATA_TOML.format(kv="fp8", extra="")
        .replace('runtime = "strata"', 'runtime = "llama-cpp"')
        .replace('catalog_model = "inkling-2-mini-strata"', 'catalog_model = "qwen3.8-27b-gguf"')
        .replace('catalog_variant = "sq10"', 'catalog_variant = "q4"')
    )
    (pasta / "llama.toml").write_text(texto, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        config.available_profiles(pasta)

    assert "kv_cache_dtype" in str(excinfo.value)
    assert "f16" in str(excinfo.value)
