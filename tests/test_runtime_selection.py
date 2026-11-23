"""Testes de seleção de perfil por runtime e do contrato do worker."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from byom import cli, config

RUNTIMES = {"vllm", "llama-cpp", "image"}


def test_perfis_declaram_runtime_e_referencia_do_catalogo():
    profiles = {profile.id: profile for profile in config.available_profiles()}
    assert {profile.runtime for profile in profiles.values()} == RUNTIMES
    for profile in profiles.values():
        if profile.catalog_model:
            assert profile.catalog_variant
            assert config.catalog.model_by_id(profile.catalog_model).runtime == profile.runtime
            cfg = config.Config()
            config.apply_profile(cfg, profile.id)
            assert config.check_compatibility(cfg).fits


def test_apply_profile_define_runtime_no_config():
    cfg = config.Config()
    config.apply_profile(cfg, "qwen-image-2.1-q4km")
    assert cfg.runtime == "image"
    config.apply_profile(cfg, "qwen3.8-27b-32k")
    assert cfg.runtime == "vllm"


def test_binarios_fixados_com_checksum_e_sem_latest():
    for profile in config.available_profiles():
        worker = profile.worker or {}
        if not worker:
            continue
        assert "@latest" not in json.dumps(worker)
        assert worker["binary_url"].startswith("https://")
        assert len(worker["binary_sha256"]) == 64


def test_payload_carrega_runtime_e_worker_settings(tmp_path: Path):
    cfg = config.Config()
    config.apply_profile(cfg, "inkling-2-mini-q4-gguf")
    state = config.State(session="s", endpoint="", nonce="n", created_at="now")
    tar_path = tmp_path / "worker.tar.gz"
    cli._build_worker_tar(cfg, state, tar_path)
    import tarfile

    with tarfile.open(tar_path, "r:gz") as archive:
        payload = json.load(archive.extractfile("session.json"))
    assert payload["runtime"] == "llama-cpp"
    assert payload["catalog_model"] == "inkling-2-mini"
    assert payload["worker"]["dspark"] is True
    assert payload["worker"]["sampling"]["top_p"] == 1.0
    assert "binary_sha256" in payload["worker"]


def _command(cfg: dict) -> list[str]:
    script = Path(cli._worker_dir()) / "build_vllm_command.py"
    result = subprocess.run(
        [sys.executable, str(script), "/dev/stdin"],
        input=json.dumps(cfg),
        check=True,
        capture_output=True,
        text=True,
    )
    return shlex.split(result.stdout.removeprefix("exec ").strip())


def test_comando_por_runtime():
    base = {"model": "m", "served_model_name": "m", "vllm_port": 8000, "max_model_len": 32768}
    assert _command({**base, "runtime": "vllm"})[0].endswith("/vllm")

    llama = _command(
        {
            **base,
            "runtime": "llama-cpp",
            "worker": {
                "main_pattern": "model.gguf",
                "mmproj_pattern": "mmproj.gguf",
                "sampling": {"temperature": 1.0, "top_k": 64},
            },
        }
    )
    assert llama[0].endswith("llama-server")
    assert "--mmproj" in llama
    assert llama[llama.index("--top-k") + 1] == "64"

    image = _command({**base, "runtime": "image"})
    assert image[0].endswith("sd-server")
    assert "--listen-port" in image
    assert image[image.index("--listen-port") + 1] == "8000"


def test_gateway_rejeita_runtime_invalido(gateway_module):
    cfg = gateway_module.GatewayConfig(nonce="n", runtime="bogus")
    with pytest.raises(ValueError):
        cfg.validate()


def test_gateway_imagem_nao_e_chat(gateway_module):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "QUJD"}]})

    cfg = gateway_module.GatewayConfig(
        nonce="n", runtime="image", vllm_base_url="http://backend"
    )
    app = gateway_module.create_app(cfg, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        assert client.post("/v1/images/generations", json={"prompt": "cat"}).status_code == 200
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 501

    vllm_cfg = gateway_module.GatewayConfig(nonce="n", runtime="vllm")
    vllm_app = gateway_module.create_app(vllm_cfg, transport=httpx.MockTransport(handler))
    with TestClient(vllm_app) as client:
        assert client.post("/v1/images/generations", json={"prompt": "cat"}).status_code == 501
