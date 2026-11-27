#!/usr/bin/env python3
"""Sondas reais da revisão dos lotes 4 e 5 (rodar à mão, não é teste de CI).

Cada função exercita o caminho de produção com arquivos/processos reais e
imprime `[OK]`/`[FALHA]`. Uso:

    uv run python scripts/probe_revisao.py            # todas
    uv run python scripts/probe_revisao.py proxy dsh  # só algumas

Nenhuma sonda depende de rede externa: o HF é substituído pela fixture real
(`tests/fixtures/hf_files.json`), o upstream é um `http.server` local e o CLI
`colab` é simulado no ponto de `subprocess`.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = ROOT / "tests" / "fixtures" / "hf_files.json"
OK, FALHA = "[OK]", "[FALHA]"
_falhas: list[str] = []


def _root() -> Path:
    return ROOT


def _importar(nome: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(nome, ROOT / "worker" / f"{nome}.py")
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def _check(condicao: bool, mensagem: str) -> None:
    print(f"{OK if condicao else FALHA} {mensagem}")
    if not condicao:
        _falhas.append(mensagem)


# ---------------------------------------------------------------------------
# (a) match_file contra a lista real de arquivos do HF
# ---------------------------------------------------------------------------
def probe_match_file() -> None:
    print("\n== (a) match_file com a lista real do Hugging Face")
    prepare = _importar("prepare_runtime")
    dados = json.loads(FIXTURE.read_text(encoding="utf-8"))
    arquivos = {repo: [item["path"] for item in itens] for repo, itens in dados["repos"].items()}
    qwen = arquivos["leejet/Qwen-Image-2.1-GGUF"]

    achado = prepare.match_file(qwen, "qwen_image_2.1-Q4_K.gguf")
    print(f"   match_file({len(qwen)} arquivos, 'qwen_image_2.1-Q4_K.gguf') -> {achado}")
    try:
        prepare.match_file(qwen, "*Q4_K_M*")
    except ValueError as exc:
        print(f"   match_file(..., '*Q4_K_M*') -> ValueError: {exc}")
        nao_casa = True
    else:
        nao_casa = False
    _check(achado == "qwen_image_2.1-Q4_K.gguf" and nao_casa, "Q4_K casa e Q4_K_M não existe")

    from byom import config

    perfil = config.load_profile("qwen-image-2.1-q4km")
    for papel, spec in perfil.worker["image_roles"].items():
        repo, padrao = spec.split("::", 1)
        resolvido = prepare.match_file(arquivos[repo], padrao)
        print(f"   {papel:13s} {spec} -> {resolvido}")


# ---------------------------------------------------------------------------
# (b) proxy: non-stream de 90 s e stream sem primeiro token
# ---------------------------------------------------------------------------
class _Upstream(http.server.BaseHTTPRequestHandler):
    demora_imagem = 90.0
    demora_stream = 120.0

    def do_POST(self) -> None:  # noqa: N802 - API do http.server
        corpo = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
        payload = json.loads(corpo or b"{}")
        if self.path.endswith("/chat/completions") and payload.get("stream"):
            # Nunca manda header: é o caso que o deadline de primeiro byte cobre.
            time.sleep(type(self).demora_stream)
            return
        time.sleep(type(self).demora_imagem)
        body = json.dumps({"data": [{"url": "http://x/probe.png"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def probe_proxy() -> None:
    print("\n== (b) proxy: non-stream de 90 s e stream sem primeiro token")
    from fastapi.testclient import TestClient

    from byom import config, proxy
    from byom.proxy import Runtime, WorkerStatus, create_app
    from byom.tunnel import TunnelState, TunnelSupervisor

    servidor = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{servidor.server_address[1]}"

    porta_upstream = servidor.server_address[1]
    cfg = config.Config(tunnel_port=porta_upstream)
    tunnel = TunnelSupervisor(session="byom-probe", endpoint="/tun/m/x", cfg=cfg, nonce="n")
    tunnel.status = TunnelState(state="connected", reconnects=0, latency_ms=1.0)
    runtime = Runtime(
        state=config.State(session="byom-probe", endpoint="", nonce="n", created_at="t"),
        cfg=cfg,
        tunnel=tunnel,
    )
    runtime.worker = WorkerStatus(verified=True, model_ready=True)

    async def _idle(stop: asyncio.Event) -> None:
        await stop.wait()

    runtime.verify_loop = _idle  # type: ignore[method-assign]
    app = create_app(runtime)

    print(f"   upstream local em {base}; NONSTREAM_DEADLINE_S={proxy.NONSTREAM_DEADLINE_S:g}")
    with TestClient(app, base_url="http://127.0.0.1:4000") as client:
        inicio = time.monotonic()
        resposta = client.post("/v1/images/generations", json={"prompt": "cat"})
        decorrido = time.monotonic() - inicio
    print(f"   imagem non-stream: {resposta.status_code} em {decorrido:.1f}s")
    _check(
        resposta.status_code == 200 and decorrido > 60,
        "imagem non-stream de 90 s responde 200 (o prazo antigo de 60 s daria 503)",
    )

    print(f"   chat stream=true sem header; FIRST_BYTE_DEADLINE_S={proxy.FIRST_BYTE_DEADLINE_S:g}")
    with TestClient(app, base_url="http://127.0.0.1:4000") as client:
        inicio = time.monotonic()
        resposta = client.post("/v1/chat/completions", json={"model": "m", "stream": True})
        decorrido = time.monotonic() - inicio
    tipo = resposta.json()["error"]["type"]
    print(f"   chat stream: {resposta.status_code} {tipo} em {decorrido:.1f}s")
    print(f"   mensagem: {resposta.json()['error']['message']}")
    _check(
        resposta.status_code == 503 and tipo == "byom_deadline",
        "chat stream sem primeiro token responde 503 byom_deadline",
    )
    servidor.shutdown()


# ---------------------------------------------------------------------------
# (c) dsh: indentação de 4 espaços e de 3 espaços
# ---------------------------------------------------------------------------
AMOSTRA_DSH = """\
# Patch do usuário
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      anthropic:
        apiKeyEnv: SEGREDO
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: anthropic
    model: claude-x
"""


def _reindentar(texto: str, fator: int) -> str:
    passo = 2 * fator
    linhas = []
    for linha in texto.splitlines(keepends=True):
        sem = linha.lstrip(" ")
        recuo = len(linha) - len(sem)
        if sem.startswith("- "):
            linhas.append(" " * (recuo * fator) + "-" + " " * (passo - 1) + sem[2:])
        else:
            linhas.append(" " * (recuo * fator) + sem)
    return "".join(linhas)


def probe_dsh() -> None:
    print("\n== (c) dsh: merge com indentação de 4 espaços e recusa com 3")
    import yaml

    from byom import config, dsh

    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-32k")

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "cordis.patch.yml"
        quatro = _reindentar(AMOSTRA_DSH, 2)
        caminho.write_text(quatro, encoding="utf-8")
        resultado = dsh.sync_config(cfg, caminho)
        texto = caminho.read_text(encoding="utf-8")
        try:
            docs = yaml.safe_load(texto)
            valido = True
        except yaml.YAMLError as exc:
            print(f"   yaml.safe_load falhou: {exc}")
            valido = False
        byom = docs[0]["config"]["providers"]["byom"] if valido else {}
        print(f"   changed={resultado.changed} yaml_valido={valido}")
        print(f"   provider.byom.models[0].id = {byom.get('models', [{}])[0].get('id')}")
        _check(
            valido
            and byom.get("api") == "openai-completions"
            and texto.startswith("# Patch do usuário\n")
            and "SEGREDO" in texto,
            "4 espaços: YAML válido, comentário e chave do usuário preservados",
        )
        print(f"   idempotente: {dsh.sync_config(cfg, caminho).changed is False}")
        print(f"   backup existe: {(Path(tmp) / 'cordis.patch.yml.byom.bak').exists()}")

        tres = Path(tmp) / "tres.yml"
        original = (
            "-  id: llm-pi-ai\n"
            '   name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "   config:\n"
            "      providers:\n"
            "         byom:\n"
            "            apiKeyEnv: MEU\n"
        )
        tres.write_text(original, encoding="utf-8")
        try:
            dsh.sync_config(cfg, tres)
        except dsh.DshConfigError as exc:
            print(f"   3 espaços -> DshConfigError: {exc}")
            recusou = tres.read_text(encoding="utf-8") == original
        else:
            recusou = False
        _check(recusou, "3 espaços: recusa sem escrever")


# ---------------------------------------------------------------------------
# (d) `colab sessions`: ANSI, coluna extra, lixo e `down`
# ---------------------------------------------------------------------------
class _Proc:
    returncode = 0
    stderr = b""

    def __init__(self, texto: str) -> None:
        self.stdout = texto.encode()


_ANSI = (
    "\x1b[32m[byom-1a2b3c4d]\x1b[0m /tun/m/abcdef | \x1b[36mHardware:\x1b[0m L4 | "
    "Shape: STANDARD | Variant: GPU"
)
_EXTRA = (
    "[byom-1a2b3c4d] /tun/m/abcdef | Hardware: L4 | Shape: STANDARD | "
    "Variant: GPU | Status: IDLE | Region: us-central1"
)


def probe_colab() -> None:
    print("\n== (d) `colab sessions`: ANSI, coluna extra, lixo e `down`")
    from typer.testing import CliRunner

    from byom import cli, colab, config

    for rotulo, texto in (("ANSI", _ANSI), ("coluna extra", _EXTRA)):
        colab.run = lambda args, _t=texto, **kwargs: _Proc(_t)  # type: ignore[assignment]
        linhas = colab.sessions()
        print(
            f"   {rotulo:12s} -> {len(linhas)} linha(s): "
            f"{linhas[0].name} {linhas[0].endpoint} {linhas[0].hardware} {linhas[0].status}"
        )
        _check(len(linhas) == 1 and linhas[0].endpoint == "/tun/m/abcdef", f"{rotulo} parseia")

    colab.run = lambda args, **kwargs: _Proc("formato novo: sem campos\n")  # type: ignore[assignment]
    try:
        colab.sessions()
    except colab.ColabCliError as exc:
        print(f"   lixo -> ColabCliError: {exc}")
        lixo_erra = True
    else:
        lixo_erra = False
    _check(lixo_erra, "saída desconhecida vira ColabCliError, não lista vazia")

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BYOM_CONFIG_DIR"] = tmp
        config.save_config(config.Config())
        config.save_state(
            config.State(
                session="byom-viva", endpoint="m-s-viva", nonce="n" * 32, created_at="t"
            )
        )
        colab.stop = lambda name: None  # type: ignore[assignment]
        colab.sessions = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            colab.ColabCliError("formato do `colab sessions` não reconhecido: x")
        )
        saida = CliRunner().invoke(cli.app, ["down"])
        estado = config.load_state()
        print(f"   down com parse quebrado: exit={saida.exit_code} state={estado.session}")
        print(f"   saida: {saida.output.strip().splitlines()[-3:]}")
        _check(
            saida.exit_code != 0 and estado is not None and estado.session == "byom-viva",
            "`down` não limpa o state quando o parse falha",
        )


# ---------------------------------------------------------------------------
# (e) down/up concorrentes e escritas com dono
# ---------------------------------------------------------------------------
def probe_ownership() -> None:
    print("\n== (e) down antigo x up novo e escritas com `expected_session`")
    from byom import config

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BYOM_CONFIG_DIR"] = tmp
        config.save_state(
            config.State(session="byom-antiga", endpoint="", nonce="a" * 32, created_at="t")
        )
        pronto = threading.Event()
        gravado = threading.Event()
        erros: list[str] = []

        def up_novo() -> None:
            pronto.wait(timeout=5)
            config.clear_state()
            config.save_state(
                config.State(session="byom-nova", endpoint="", nonce="b" * 32, created_at="t")
            )
            gravado.set()

        def down_antigo() -> None:
            atual = config.load_state()
            assert atual is not None
            pronto.set()
            gravado.wait(timeout=5)
            try:
                config.clear_state(expected_session=atual.session)
            except config.StateFileError as exc:
                erros.append(str(exc))

        threads = [threading.Thread(target=up_novo), threading.Thread(target=down_antigo)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        final = config.load_state()
        print(f"   down antigo recusado: {erros[0] if erros else 'NÃO'}")
        print(f"   state final: {final.session if final else None}")
        _check(
            bool(erros) and final is not None and final.session == "byom-nova",
            "down antigo não apaga o state da sessão nova",
        )

        antes = config.state_path().read_text(encoding="utf-8")
        try:
            config.update_state(expected_session="byom-antiga", daemon_pid=1)
        except config.StateFileError as exc:
            print(f"   update com dono errado -> {exc}")
        _check(
            config.state_path().read_text(encoding="utf-8") == antes,
            "escrita com `expected_session` errado não altera o arquivo",
        )


# ---------------------------------------------------------------------------
# (f) falha do `up` depois do daemon: processo morto, porta livre, sem espera
# ---------------------------------------------------------------------------
def _filho_na_porta(porta: int) -> subprocess.Popen[bytes]:
    script = (
        "import socket, time\n"
        "s = socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        f"s.bind(('127.0.0.1', {porta}))\n"
        "s.listen(1)\n"
        "time.sleep(120)\n"
    )
    return subprocess.Popen([sys.executable, "-c", script])


def _porta_livre(porta: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", porta))
        except OSError:
            return False
    return True


def _porta_de_probe() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def probe_up_failure() -> None:
    print("\n== (f) falha do `up` depois do daemon: morto, porta livre, sem espera")
    from typer.testing import CliRunner

    from byom import cli, colab, config, remote

    porta = _porta_de_probe()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BYOM_CONFIG_DIR"] = tmp
        cfg = config.Config(proxy_port=porta)
        config.apply_profile(cfg, "inkling-2-mini-iq2-gguf")
        config.save_config(cfg)

        class FakeRemote:
            def __init__(self, session, cfg):
                self.session = session
                self.cfg = cfg

            def start(self, forwards=(), on_progress=None):
                return None

            def stop(self):
                return None

            def gpu_name(self):
                return "Tesla T4"

            def upload(self, *a, **k):
                return None

            def run(self, *a, **k):
                return remote.RemoteResult(returncode=0, output="")

        cli._sync_external_configs = lambda *a, **k: None
        cli._check_orphans = lambda: None
        # A validação real da chave abortaria o up antes do daemon; esta sonda
        # isola o teardown, então a checagem é neutralizada como nos testes.
        remote.validate_ssh_key = lambda path: None
        cli._bootstrap_worker = lambda *a, **k: None
        cli._check_proxy_port = lambda cfg: None
        cli.remote.Remote = FakeRemote
        colab.new = lambda *a, **k: None
        colab.stop = lambda name: None
        colab.is_allocated = lambda name, endpoint="": False
        colab.status_line = lambda name: colab.SessionLine(
            name=name, endpoint="m-s-x", hardware="T4", shape="STANDARD", variant="GPU"
        )

        filho = _filho_na_porta(porta)
        try:
            prazo = time.monotonic() + 5
            while time.monotonic() < prazo and _porta_livre(porta):
                time.sleep(0.02)
            # O "daemon" do up é este filho; a espera pelo worker falha de propósito.
            cli._start_daemon = lambda: filho.pid
            cli._pid_is_byom = lambda *a, **k: filho.poll() is None
            cli.TERMINATE_GRACE_S = 1.0

            def falha_rapida(*a, **k):
                raise TimeoutError("worker indisponível: túnel failed (sonda)")

            cli._wait_ready = falha_rapida
            inicio = time.monotonic()
            saida = CliRunner().invoke(cli.app, ["up"])
            decorrido = time.monotonic() - inicio
            filho.wait(timeout=10)
            livre = _porta_livre(porta)
            print(f"   up -> exit={saida.exit_code} em {decorrido:.1f}s")
            print(f"   filho (daemon) vivo? {filho.poll() is None}; porta {porta} livre? {livre}")
            _check(
                saida.exit_code == 1
                and filho.poll() is not None
                and livre
                and decorrido < 30,
                "falha encerra o daemon, libera a porta e não espera 30 min",
            )
        finally:
            if filho.poll() is None:
                filho.kill()
                filho.wait(timeout=5)
            os.environ.pop("BYOM_CONFIG_DIR", None)


# ---------------------------------------------------------------------------
# (g) identidade do processo
# ---------------------------------------------------------------------------
def probe_pid_identity() -> None:
    print("\n== (g) `_pid_is_byom` com 'byom' no argv mas sem o padrão")
    from byom import cli, config

    processo = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "byom", "rodando"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        prazo = time.monotonic() + 5
        while time.monotonic() < prazo and "byom" not in config.process_cmdline(processo.pid):
            time.sleep(0.02)
        cmdline = config.process_cmdline(processo.pid)
        veredito = cli._pid_is_byom(processo.pid, None, "byom-probe")
        print(f"   argv: {cmdline}")
        print(f"   _pid_is_byom(pid, None, 'byom-probe') -> {veredito}")
        print(f"   estrutura reconhecida: {cli._structural_kind(cmdline.split())}")
        _check(veredito is False, "'byom' no argv não basta para matar o processo")
    finally:
        if processo.poll() is None:
            processo.kill()
            processo.wait(timeout=5)

    daemon_argv = [sys.executable, "-m", "byom", "_daemon"]
    print(f"   argv de daemon {daemon_argv} -> {cli._structural_kind(daemon_argv)}")
    watchdog = ["colab", "exec", "-s", "byom-x", "-f", "/w/watchdog.py"]
    print(f"   argv de watchdog -> {cli._structural_kind(watchdog, 'byom-x')}")


# ---------------------------------------------------------------------------
# (h) tabela final de perfis
# ---------------------------------------------------------------------------
def probe_tabela() -> None:
    print("\n== (h) orçamento de VRAM dos perfis embarcados")
    from byom import catalog, config

    cabecalho = (
        f"{'perfil':28s} {'GPU':8s} {'ctx':>7s} {'required':>8s} {'KV':>4s} "
        f"{'folga':>6s} {'margem':>7s} fits"
    )
    print(cabecalho)
    for perfil in sorted(config.available_profiles(), key=lambda p: p.id):
        if not perfil.catalog_model:
            continue
        gpu = catalog.resolve_gpu(
            str(perfil.settings["gpu"]), bool(perfil.settings.get("gpu_high_mem"))
        )
        ctx = int(perfil.settings["max_model_len"])
        fit = catalog.check_fit(
            perfil.catalog_model,
            perfil.catalog_variant,
            gpu.id,
            ctx,
            kv_cache_dtype=perfil.settings.get("kv_cache_dtype"),
        )
        dtype = catalog.effective_kv_dtype(
            perfil.runtime, perfil.settings.get("kv_cache_dtype")
        )
        margem = gpu.vram_gib - fit.required_gib - fit.headroom_gib
        print(
            f"{perfil.id:28s} {gpu.id:8s} {ctx:7d} {fit.required_gib:8.2f} {dtype or '-':>4s} "
            f"{fit.headroom_gib:6.2f} {margem:7.2f} {'sim' if fit.fits else 'NAO'}"
        )
        _check(fit.fits, f"{perfil.id} cabe em {gpu.id}")


SONDAS = {
    "match": probe_match_file,
    "proxy": probe_proxy,
    "dsh": probe_dsh,
    "colab": probe_colab,
    "ownership": probe_ownership,
    "up": probe_up_failure,
    "pid": probe_pid_identity,
    "tabela": probe_tabela,
}


def main(argv: list[str]) -> int:
    alvos = argv or list(SONDAS)
    for nome in alvos:
        funcao = SONDAS.get(nome)
        if funcao is None:
            raise SystemExit(f"sonda desconhecida: {nome}; opções: {', '.join(SONDAS)}")
        funcao()
    print(f"\n{len(_falhas)} falha(s)" if _falhas else "\nTodas as sondas passaram")
    return 1 if _falhas else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
