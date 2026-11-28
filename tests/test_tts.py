"""Qwen3-TTS: catálogo, orçamento de VRAM e argv do vLLM-Omni."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from byom import catalog, config

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"


@pytest.fixture
def build_module():
    spec = importlib.util.spec_from_file_location(
        "byom_build_command_tts", WORKER_DIR / "build_vllm_command.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalogo_tts_e_orcamento_por_pipeline():
    entry = catalog.model_by_id("qwen3-tts")
    assert entry.runtime == "tts"
    assert entry.provider == "Qwen (Alibaba Cloud)"

    pequeno = catalog.variant_by_id("qwen3-tts", "0.6b-customvoice")
    assert pequeno.memory_gib == 2.32
    assert pequeno.overhead_gib == 6.0
    assert catalog.check_fit("qwen3-tts", "0.6b-customvoice", "T4", 1).fits
    assert catalog.check_fit("qwen3-tts", "0.6b-customvoice", "T4", 1).required_gib == 10.32

    grande = catalog.variant_by_id("qwen3-tts", "1.7b-customvoice")
    assert grande.memory_gib == 4.21
    assert grande.overhead_gib == 9.0
    assert not catalog.check_fit("qwen3-tts", "1.7b-customvoice", "T4", 1).fits
    assert catalog.check_fit("qwen3-tts", "1.7b-customvoice", "L4", 1).fits


def test_perfis_tts_pinam_vllm_omni_e_deploy_config():
    for profile_id in ("qwen3-tts-0.6b-customvoice", "qwen3-tts-1.7b-customvoice"):
        profile = config.load_profile(profile_id)
        assert profile.runtime == "tts"
        assert profile.settings["vllm_spec"] == "vllm==0.34.0"
        assert profile.worker["vllm_omni_spec"] == "vllm-omni==0.34.0"
        assert profile.settings["max_model_len"] == 1
        assert profile.worker["deploy_config"] == "qwen3_tts.yaml"
        assert profile.worker["trust_remote_code"] is True


def test_payload_do_worker_leva_os_dois_specs_do_tts(tmp_path):
    import json
    import tarfile

    from byom import cli

    cfg = config.Config()
    config.apply_profile(cfg, "qwen3-tts-0.6b-customvoice")
    state = config.State(session="s", endpoint="", nonce="n", created_at="now")
    tar_path = tmp_path / "worker.tar.gz"
    cli._build_worker_tar(cfg, state, tar_path)

    with tarfile.open(tar_path, "r:gz") as archive:
        payload = json.load(archive.extractfile("session.json"))

    assert payload["runtime"] == "tts"
    assert payload["vllm_spec"] == "vllm==0.34.0"
    assert payload["worker"]["vllm_omni_spec"] == "vllm-omni==0.34.0"
    assert payload["worker"]["deploy_config"] == "qwen3_tts.yaml"


def test_tui_aplica_tts_com_contexto_sentinela():
    from byom import tui

    item = next(
        p
        for p in tui.listar_perfis_estruturados(tui.ler_estado_local())
        if p.profile.id == "qwen3-tts-0.6b-customvoice"
    )
    rec = tui.criar_console(width=100, height=30, record=True)

    ok, _ = tui._aplicar_perfil_interativo(
        rec,
        item,
        pedir_confirmacao=False,
        gpu_id="T4",
        max_model_len=item.contexto_servido,
    )

    assert ok is True
    cfg = config.load_config()
    assert (cfg.runtime, cfg.gpu, cfg.max_model_len) == ("tts", "T4", 1)
    assert config.check_compatibility(cfg).fits


def test_build_tts_args_monta_o_serve_do_vllm_omni(build_module, tmp_path):
    deploy = tmp_path / "qwen3_tts.yaml"
    deploy.write_text("stages: []\n", encoding="utf-8")
    build_module._tts_deploy_config = lambda name: deploy
    cfg = {
        "runtime": "tts",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "vllm_port": 8000,
        "served_model_name": "qwen3-tts-0.6b",
        "worker": {"deploy_config": "qwen3_tts.yaml", "trust_remote_code": True},
    }

    assert build_module.build_command(cfg) == [
        "/content/byom/venv/bin/vllm",
        "serve",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "--omni",
        "--deploy-config",
        str(deploy),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "qwen3-tts-0.6b",
        "--trust-remote-code",
    ]


def test_tts_sem_vllm_omni_falha_com_mensagem(build_module):
    if importlib.util.find_spec("vllm_omni") is not None:
        pytest.skip("vllm_omni instalado neste ambiente")
    with pytest.raises(RuntimeError, match="vllm_omni"):
        build_module._tts_deploy_config("qwen3_tts.yaml")
