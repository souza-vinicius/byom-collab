"""Testes de seleção de perfil por runtime e do contrato do worker."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from byom import catalog, cli, config

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
    assert payload["worker"].get("dspark", False) is False
    assert payload["worker"]["llama_jinja"] is True
    assert "Mainline_Support" in payload["worker"]["main_pattern"]
    assert payload["worker"]["main_pattern"].endswith(".gguf_file")
    assert payload["worker"]["sampling"]["top_p"] == 1.0
    assert "binary_sha256" in payload["worker"]


def _command(cfg: dict, *, env: dict | None = None) -> list[str]:
    script = Path(cli._worker_dir()) / "build_vllm_command.py"
    result = subprocess.run(
        [sys.executable, str(script), "/dev/stdin"],
        input=json.dumps(cfg),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return shlex.split(result.stdout.removeprefix("exec ").strip())


def write_image_manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Manifest de imagem com o binário e os três papéis em arquivos reais."""
    server = tmp_path / "sd-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    role_paths: dict[str, Path] = {}
    for role in ("diffusion", "vae", "text_encoder"):
        path = tmp_path / f"{role}.bin"
        path.write_text(role, encoding="utf-8")
        role_paths[role] = path
    manifest = tmp_path / "runtime_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime": "image",
                "server": str(server),
                "image": {role: str(path) for role, path in role_paths.items()},
            }
        ),
        encoding="utf-8",
    )
    return manifest, role_paths


def test_comando_por_runtime(tmp_path: Path):
    base = {
        "model": "m",
        "served_model_name": "m",
        "vllm_port": 8000,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.92,
    }
    assert _command({**base, "runtime": "vllm"})[0].endswith("/vllm")

    llama = _command(
        {
            **base,
            "runtime": "llama-cpp",
            "worker": {
                "main_pattern": "model.gguf",
                "mmproj_pattern": "mmproj.gguf",
                "llama_jinja": True,
                "sampling": {"temperature": 1.0, "top_k": 64},
            },
        }
    )
    assert llama[0].endswith("llama-server")
    assert "--jinja" in llama
    assert "--mmproj" in llama
    assert llama[llama.index("--top-k") + 1] == "64"

    image_manifest, role_paths = write_image_manifest(tmp_path)
    image = _command(
        {**base, "runtime": "image"}, env={"BYOM_RUNTIME_MANIFEST": str(image_manifest)}
    )
    assert image[0].endswith("sd-server")
    assert image[image.index("--listen-port") + 1] == "8000"
    assert image[image.index("--diffusion-model") + 1] == str(role_paths["diffusion"])
    assert image[image.index("--vae") + 1] == str(role_paths["vae"])
    assert image[image.index("--llm") + 1] == str(role_paths["text_encoder"])


def test_mainline_gguf_file_renamed_to_gguf():
    import importlib.util

    script = Path(cli._worker_dir()) / "prepare_runtime.py"
    spec = importlib.util.spec_from_file_location("byom_prepare_runtime", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path("/content/byom/models/repo")
    assert module._model_path(root, "Mainline_Support/x.gguf_file") == (
        root / "Mainline_Support/x.gguf"
    )
    assert module._model_path(root, "mmproj-F16.gguf") == root / "mmproj-F16.gguf"


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


# ---------------------------------------------------------------------------
# Runtime de imagem: papéis, manifest e fail-closed
# ---------------------------------------------------------------------------
WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"


@pytest.fixture
def build_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "byom_build_command_roles", WORKER_DIR / "build_vllm_command.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_image_args_usa_o_manifest(build_module, tmp_path):
    manifest, role_paths = write_image_manifest(tmp_path)
    build_module.MANIFEST = str(manifest)

    argv = build_module.build_image_args({"vllm_port": 8000})

    assert argv[0] == str(tmp_path / "sd-server")
    assert argv[argv.index("--diffusion-model") + 1] == str(role_paths["diffusion"])
    assert argv[argv.index("--vae") + 1] == str(role_paths["vae"])
    assert argv[argv.index("--llm") + 1] == str(role_paths["text_encoder"])
    assert argv[argv.index("--listen-ip") + 1] == "127.0.0.1"
    assert argv[argv.index("--listen-port") + 1] == "8000"


@pytest.mark.parametrize(
    ("payload_fn", "motivo"),
    [
        (lambda server: {"runtime": "image", "server": server}, "não registra image"),
        (
            lambda server: {
                "runtime": "image",
                "server": server,
                "image": {"diffusion": "/x", "vae": "/y"},
            },
            "text_encoder",
        ),
        (lambda server: {"runtime": "image", "image": {"a": "/x"}}, "sd-server"),
    ],
)
def test_build_image_args_recusa_sem_papel(build_module, tmp_path, payload_fn, motivo):
    server = tmp_path / "sd-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path / "runtime_manifest.json", payload_fn(str(server)))
    build_module.MANIFEST = str(manifest)

    with pytest.raises(RuntimeError) as excinfo:
        build_module.build_image_args({"vllm_port": 8000})

    assert "runtime de imagem não suportado nesta versão" in str(excinfo.value)
    assert motivo in str(excinfo.value)


def test_build_image_args_recusa_servidor_ausente(build_module, tmp_path):
    manifest, _ = write_image_manifest(tmp_path)
    (tmp_path / "sd-server").unlink()
    build_module.MANIFEST = str(manifest)

    with pytest.raises(RuntimeError) as excinfo:
        build_module.build_image_args({"vllm_port": 8000})

    assert "sd-server não existe" in str(excinfo.value)


def test_comando_imagem_falha_com_saida_legivel(tmp_path):
    """Sem manifest, o script termina com código 2 e a mensagem, não traceback."""
    script = Path(cli._worker_dir()) / "build_vllm_command.py"
    result = subprocess.run(
        [sys.executable, str(script), "/dev/stdin"],
        input=json.dumps({"runtime": "image", "vllm_port": 8000}),
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "BYOM_RUNTIME_MANIFEST": str(tmp_path / "ausente.json"),
        },
    )
    assert result.returncode == 2
    assert "runtime de imagem não suportado nesta versão" in result.stderr
    assert "Traceback" not in result.stderr


# -- prepare_runtime: papéis viram caminhos absolutos no manifest -------------
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf_files.json"


def hf_fixture() -> dict[str, list[str]]:
    """Lista real de arquivos por repo, congelada em `tests/fixtures/hf_files.json`.

    `match_file` é testado contra os nomes publicados, sem rede: a fixture é
    regravada por `scripts/refresh_hf_fixture.py` quando um repo muda.
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {repo: [item["path"] for item in items] for repo, items in data["repos"].items()}


HF_FILES = hf_fixture()



def _binary_tarball() -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"#!/bin/sh\necho sd-server\n"
        info = tarfile.TarInfo("sdcpp/sd-server")
        info.size = len(data)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@contextlib.contextmanager
def fake_hf_server():
    import http.server
    import threading
    import urllib.parse

    tarball = _binary_tarball()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - API do http.server
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/tree":
                files = HF_FILES[query["repo"][0]]
                body = json.dumps([{"path": p, "type": "file"} for p in files]).encode()
            elif parsed.path == "/resolve":
                body = f"peso:{query['path'][0]}".encode()
            elif parsed.path == "/binary.tar.gz":
                body = tarball
            else:  # pragma: no cover - rota inesperada
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def prepare_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "byom_prepare_runtime_roles", WORKER_DIR / "prepare_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_imagem_grava_os_tres_papeis_no_manifest(
    prepare_module, tmp_path, monkeypatch
):
    import hashlib

    byom = tmp_path / "byom"
    monkeypatch.setattr(prepare_module, "BYOM_DIR", byom)
    monkeypatch.setattr(prepare_module, "MODEL_DIR", byom / "models")
    monkeypatch.setattr(prepare_module, "MANIFEST", byom / "runtime_manifest.json")

    with fake_hf_server() as base:
        monkeypatch.setattr(prepare_module, "HF_API", f"{base}/tree?repo={{repo}}")
        monkeypatch.setattr(
            prepare_module, "HF_RESOLVE", f"{base}/resolve?repo={{repo}}&path={{path}}"
        )
        session = tmp_path / "session.json"
        session.write_text(
            json.dumps(
                {
                    "runtime": "image",
                    "worker": {
                        "binary_url": f"{base}/binary.tar.gz",
                        "binary_sha256": hashlib.sha256(_binary_tarball()).hexdigest(),
                        "binary_dir": "sdcpp",
                        "image_roles": {
                            "diffusion": "leejet/Qwen-Image-2.1-GGUF::qwen_image_2.1-Q4_K.gguf",
                            "vae": "Comfy-Org/Qwen-Image-2.1::vae/*.safetensors",
                            "text_encoder": "Qwen/Qwen3-VL-8B-Instruct-GGUF::*Q4_K_M.gguf",
                            "vision": "Qwen/Qwen3-VL-8B-Instruct-GGUF::mmproj-*F16.gguf",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = prepare_module.prepare(session)

    assert manifest["server"] == str(byom / "sd-server")
    assert set(manifest["image"]) == {"diffusion", "vae", "text_encoder", "vision"}
    for path in manifest["image"].values():
        target = Path(path)
        assert target.is_absolute()
        assert target.is_file()
        assert target.read_text(encoding="utf-8").startswith("peso:")
    assert manifest["image"]["diffusion"].endswith("qwen_image_2.1-Q4_K.gguf")
    assert manifest["image"]["vae"].endswith("qwen_image_2.1_vae_bf16.safetensors")
    assert manifest["image"]["text_encoder"].endswith("Qwen3VL-8B-Instruct-Q4_K_M.gguf")
    assert manifest["image"]["vision"].endswith("mmproj-Qwen3VL-8B-Instruct-F16.gguf")
    assert (byom / "sd-server").is_symlink()
    on_disk = json.loads((byom / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["image"] == manifest["image"]


def test_perfis_de_imagem_declaram_os_tres_papeis():
    for profile_id in ("qwen-image-2.1-q4km", "qwen-image-2.1-fp8"):
        profile = config.load_profile(profile_id)
        roles = (profile.worker or {}).get("image_roles") or {}
        assert set(roles) >= {"diffusion", "vae", "text_encoder"}, profile_id
        assert all("::" in spec for spec in roles.values())


# -- 4.1: o quant publicado é Q4_K, não Q4_K_M --------------------------------
def test_match_file_resolve_o_quant_publicado_e_recusa_o_inexistente(prepare_module):
    """`leejet/Qwen-Image-2.1-GGUF` publica Q4_K (4,20 GB) e não Q4_K_M."""
    paths = HF_FILES["leejet/Qwen-Image-2.1-GGUF"]

    assert prepare_module.match_file(paths, "qwen_image_2.1-Q4_K.gguf") == (
        "qwen_image_2.1-Q4_K.gguf"
    )
    with pytest.raises(ValueError, match="Q4_K_M"):
        prepare_module.match_file(paths, "*Q4_K_M*")


def test_match_file_tolera_o_caminho_do_repo_com_pasta(prepare_module):
    paths = HF_FILES["unsloth/Inkling-2-Mini-GGUF"]
    assert prepare_module.match_file(
        paths, "Mainline_Support/Inkling-2-Mini-UD-IQ2_XXS.gguf_file"
    ).startswith("Mainline_Support/")


def test_perfil_q4km_aponta_para_arquivos_que_existem_no_hf(prepare_module):
    profile = config.load_profile("qwen-image-2.1-q4km")
    roles = profile.worker["image_roles"]

    for role, spec in roles.items():
        repo, pattern = spec.split("::", 1)
        assert prepare_module.match_file(HF_FILES[repo], pattern), role


def test_orcamento_de_imagem_usa_os_bytes_publicados():
    """Somas conferidas na API do HF em 2026-11-24 (ver tests/fixtures/hf_files.json)."""
    q4km = catalog.variant_by_id("qwen-image-2.1", "q4km")
    fp8 = catalog.variant_by_id("qwen-image-2.1", "fp8-int8")
    sizes = {item["path"]: item["size"] for item in json.loads(
        FIXTURE.read_text(encoding="utf-8")
    )["repos"]["leejet/Qwen-Image-2.1-GGUF"]}
    q4km_bytes = (
        sizes["qwen_image_2.1-Q4_K.gguf"]
        + 5_027_784_800  # Qwen3VL-8B-Instruct-Q4_K_M.gguf
        + 675_509_688  # vae/qwen_image_2.1_vae_bf16.safetensors
    )
    assert q4km.memory_gib == pytest.approx(round(q4km_bytes / 2**30, 2))
    assert fp8.memory_gib == pytest.approx(round((q4km_bytes + 3_058_988_248) / 2**30, 2))


def test_t4_volta_a_comportar_o_q4km_e_o_fp8_continua_em_l4():
    """9,22 GiB de pesos: T4 (15 GiB) aprova; 12,07 GiB do INT8 não aprova."""
    q4km_t4 = catalog.check_fit("qwen-image-2.1", "q4km", "T4", 1)
    q4km_l4 = catalog.check_fit("qwen-image-2.1", "q4km", "L4", 1)
    fp8_l4 = catalog.check_fit("qwen-image-2.1", "fp8-int8", "L4", 1)
    fp8_t4 = catalog.check_fit("qwen-image-2.1", "fp8-int8", "T4", 1)

    assert q4km_l4.fits is True
    assert q4km_t4.fits is True
    assert fp8_l4.fits is True
    assert fp8_t4.fits is False
    assert fp8_t4.required_gib == pytest.approx(15.15)
    assert "Tesla T4" in fp8_t4.reason


def test_perfil_q4km_declara_a_gpu_que_comporta_o_arquivo():
    profile = config.load_profile("qwen-image-2.1-q4km")
    assert profile.settings["gpu"] == "T4"
    ref = catalog.profile_context("qwen-image-2.1-q4km")
    assert ref.gpu_id == "T4"
    assert catalog.check_fit(
        "qwen-image-2.1", "q4km", "T4", profile.settings["max_model_len"]
    ).fits


# -- 5.1: edição de imagem exige o projetor de visão --------------------------
def test_build_image_args_emite_llm_vision_quando_o_papel_existe(build_module, tmp_path):
    manifest, role_paths = write_image_manifest(tmp_path)
    role_paths["vision"] = tmp_path / "models" / "mmproj-F16.gguf"
    role_paths["vision"].parent.mkdir(parents=True, exist_ok=True)
    role_paths["vision"].write_text("mmproj", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["image"]["vision"] = str(role_paths["vision"])
    _write_manifest(manifest, payload)
    build_module.MANIFEST = str(manifest)

    argv = build_module.build_image_args({"vllm_port": 8000})

    assert argv[argv.index("--llm_vision") + 1] == str(role_paths["vision"])
    # As três flags obrigatórias continuam na linha de comando.
    for flag in ("--diffusion-model", "--vae", "--llm"):
        assert flag in argv


def test_build_image_args_sem_vision_continua_valido_para_gerar(build_module, tmp_path):
    """Perfil sem `vision` não impede o servidor de subir."""
    manifest, role_paths = write_image_manifest(tmp_path)
    build_module.MANIFEST = str(manifest)

    argv = build_module.build_image_args({"vllm_port": 8000})

    assert "--llm_vision" not in argv
    assert argv[argv.index("--llm") + 1] == str(role_paths["text_encoder"])


def test_perfis_de_imagem_declaram_o_papel_vision():
    for profile_id in ("qwen-image-2.1-q4km", "qwen-image-2.1-fp8"):
        profile = config.load_profile(profile_id)
        roles = (profile.worker or {}).get("image_roles") or {}
        assert "vision" in roles, profile_id
        assert roles["vision"].endswith("mmproj-Qwen3VL-8B-Instruct-F16.gguf")
        assert (
            "Qwen/Qwen3-VL-8B-Instruct-GGUF::mmproj-Qwen3VL-8B-Instruct-F16.gguf"
            in (profile.worker or {}).get("model_patterns", [])
        ), profile_id


def test_gateway_recusa_edits_sem_o_papel_vision(gateway_module):
    import httpx
    from fastapi.testclient import TestClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"url": "http://x/1.png"}]})

    sem_vision = gateway_module.GatewayConfig(
        nonce="n",
        runtime="image",
        image_roles={"diffusion": "a", "vae": "b", "text_encoder": "c"},
    )
    app = gateway_module.create_app(sem_vision, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        resposta = client.post("/v1/images/edits", json={"prompt": "cat"})
        geracao = client.post("/v1/images/generations", json={"prompt": "cat"})

    assert resposta.status_code == 501
    assert "`vision`" in resposta.json()["error"]["message"]
    assert "--llm_vision" in resposta.json()["error"]["message"]
    # A geração continua funcionando sem o projetor.
    assert geracao.status_code == 200


def test_gateway_aceita_edits_com_o_papel_vision(gateway_module):
    import httpx
    from fastapi.testclient import TestClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"url": "http://x/1.png"}]})

    cfg = gateway_module.GatewayConfig(
        nonce="n",
        runtime="image",
        image_roles={"diffusion": "a", "vae": "b", "text_encoder": "c", "vision": "d"},
    )
    app = gateway_module.create_app(cfg, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        resposta = client.post("/v1/images/edits", json={"prompt": "cat"})

    assert resposta.status_code == 200


# -- 5.2: o custo do projetor vem do arquivo que o padrão resolve -------------
def _size_gib(repo: str, path: str) -> float:
    dados = json.loads(FIXTURE.read_text(encoding="utf-8"))["repos"][repo]
    for item in dados:
        if item["path"] == path:
            return round(item["size"] / 2**30, 2)
    raise AssertionError(f"{path} não está na fixture de {repo}")


def test_vision_gib_por_variante_usa_o_projetor_publicado(prepare_module):
    assert catalog.variant_by_id("inkling-2-mini", "q2").vision_gib == 0.65
    assert catalog.variant_by_id("gemma-4", "26b-a4b-q4").vision_gib == 1.11
    assert catalog.variant_by_id("gemma-4", "31b-q4").vision_gib == 1.12
    assert catalog.variant_by_id("muse-glimmer-30b", "q2").vision_gib == 3.58
    assert catalog.variant_by_id("muse-glimmer-30b", "q4").vision_gib == 3.58

    # O padrão `*mmproj*` do Muse resolve para o BF16 (3,85 GB), não para o kquant.
    assert prepare_module.match_file(
        HF_FILES["unsloth/Muse-Glimmer-30B-GGUF"], "*mmproj*"
    ) == "mmproj-Muse-Glimmer-30B-BF16.gguf"
    assert (
        _size_gib("unsloth/Muse-Glimmer-30B-GGUF", "mmproj-Muse-Glimmer-30B-BF16.gguf")
        == 3.58
    )
    assert _size_gib("unsloth/gemma-4-26B-A4B-it-GGUF", "mmproj-BF16.gguf") == 1.11
    assert _size_gib("unsloth/gemma-4-31B-it-GGUF", "mmproj-BF16.gguf") == 1.12
    assert _size_gib("unsloth/Inkling-2-Mini-GGUF", "mmproj-F16.gguf") == 0.6



