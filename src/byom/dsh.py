"""Merge BYOM settings into the dsh (DeepSeek harness) profile patch.

The file (``~/.dsh/profiles/desktop/cordis.patch.yml``) is edited as text so comments,
secrets and unrelated entries survive. Only two things are written:

* ``providers.byom`` inside the ``llm-pi-ai`` entry (``apiKeyEnv`` when absent, ``api``,
  ``baseURL`` and the BYOM model; other keys and other models of that provider are kept);
* ``provider``/``model`` of the ``agent-default-model`` entry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config

PROVIDER_ID = "byom"
API_KEY_ENV = "BYOM_API_KEY"  # any value works: the local proxy does not check it
LLM_ENTRY = ("llm-pi-ai", "@deepseek-ai/dsh-llm-pi-ai")
DEFAULT_MODEL_ENTRY = ("agent-default-model", "@deepseek-ai/dsh-agent-default-model")
_MANAGED_KEYS = ("api", "baseURL", "models")

#: Passos de indentação aceitos. O arquivo do dsh é editado como texto, então o
#: merge tem de escrever no MESMO passo do arquivo: literais de 6/8/10 espaços
#: produziam YAML inválido em arquivos de 4 espaços. Passos fora desta lista
#: (3, 5, ...) são recusados em vez de reescritos.
INDENT_STEPS = (2, 4)

#: Sufixo do backup gravado antes da primeira alteração (nunca sobrescrito).
BACKUP_SUFFIX = ".byom.bak"



class DshConfigError(Exception):
    """Safe error raised while locating, reading, or updating the dsh config."""


@dataclass(frozen=True)
class SyncResult:
    path: Path
    model: str
    changed: bool


def default_config_path() -> Path:
    return Path.home() / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"


def resolve_config_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser()
    return default_config_path()


def is_installed() -> bool:
    return default_config_path().is_file()


def sync_config(cfg: config.Config, explicit_path: Path | None = None) -> SyncResult:
    """Register the BYOM provider/model in dsh and make it the default model."""
    path = resolve_config_path(explicit_path)
    original = _read(path)
    step = _indent_step(original)
    updated = _merge(original, cfg, step)
    changed = updated != original
    if changed:
        _assert_equivalent(original, updated, cfg)
        _backup(path)
        _atomic_write(path, updated)
    return SyncResult(path=path, model=f"{PROVIDER_ID}/{cfg.served_model_name}", changed=changed)


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    if not path.is_file():
        raise DshConfigError(f"dsh config path is not a file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")  # no newline translation
    except (OSError, UnicodeError) as exc:
        raise DshConfigError(f"Could not read {path}: {exc}") from exc
    if re.search(r"^\t", text, re.MULTILINE):
        raise DshConfigError(f"{path} uses tab indentation, which this tool does not rewrite")
    stripped = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if stripped and not stripped[0].startswith("- "):
        raise DshConfigError(f"{path} must be a top-level YAML list of patch entries")
    return text


# --- tiny line-oriented helpers -------------------------------------------------------
def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_content(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


class _LenientLoader(yaml.SafeLoader):
    """SafeLoader que aceita tags desconhecidas: o patch do dsh usa `!!js`."""


def _lenient_tag(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)  # type: ignore[arg-type]


# `None` registra o multi-construtor de tags desconhecidas do PyYAML.
_LenientLoader.add_constructor(None, _lenient_tag)  # type: ignore[arg-type]


def _load_yaml(text: str) -> object:
    """Parse tolerante: tags próprias do dsh (`!!js`) não impedem a validação."""
    return yaml.load(text, Loader=_LenientLoader)  # noqa: S506 - SafeLoader + tags tratadas


def _indent_step(text: str) -> int:
    """Passo de indentação do arquivo, inferido do primeiro filho de um item.

    O dsh escreve `- id: ...` na coluna 0 e as chaves do item no passo do
    arquivo (2 ou 4 espaços). Sem passo inferível, ou com passo fora de
    `INDENT_STEPS`, o merge recusa: reescrever com outro passo corrompe o YAML.
    """
    if not text.strip():
        return INDENT_STEPS[0]
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("- "):
            continue
        for inner in range(index + 1, len(lines)):
            if not _is_content(lines[inner]):
                continue
            if lines[inner].startswith("-"):
                break
            step = _indent(lines[inner])
            if step == 0:
                break
            if step not in INDENT_STEPS:
                raise DshConfigError(
                    f"indentação de {step} espaços não é suportada (aceitos: "
                    f"{', '.join(str(value) for value in INDENT_STEPS)}); "
                    "ajuste o arquivo para 2 ou 4 espaços por nível"
                )
            _assert_consistent_indent(lines, step)
            return step
    raise DshConfigError(
        "não foi possível inferir a indentação do arquivo: nenhum item da lista "
        "top-level tem chaves indentadas"
    )


def _assert_consistent_indent(lines: list[str], step: int) -> None:
    for line in lines:
        if not _is_content(line):
            continue
        indent = _indent(line)
        if indent and indent % step:
            raise DshConfigError(
                f"indentação inconsistente em {line.strip()[:60]!r}: {indent} espaços "
                f"não são múltiplos do passo {step} do arquivo"
            )


def _backup(path: Path) -> None:
    """Grava `<arquivo>.byom.bak` uma vez, sem sobrescrever backup existente."""
    if not path.exists():
        return
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        return
    try:
        backup.write_bytes(path.read_bytes())
        backup.chmod(0o600)
    except OSError as exc:
        raise DshConfigError(f"não foi possível gravar o backup {backup}: {exc}") from exc


def _managed_view(docs: list, model_id: str, keep_ids: set[str] | None = None) -> list:
    """Documento sem as chaves que o merge gerencia, para comparar o resto."""
    view = []
    for entry in docs:
        if not isinstance(entry, dict):
            view.append(entry)
            continue
        if keep_ids is not None and entry.get("id") not in keep_ids:
            # Entrada criada pelo próprio merge: não existe no original.
            continue
        copy = json.loads(json.dumps(entry))
        config_view = copy.get("config")
        if isinstance(config_view, dict):
            providers = config_view.get("providers")
            if isinstance(providers, dict):
                providers.pop(PROVIDER_ID, None)
                if not providers:
                    config_view.pop("providers", None)
            if copy.get("id") == DEFAULT_MODEL_ENTRY[0]:
                config_view.pop("provider", None)
                config_view.pop("model", None)
        view.append(copy)
    return view


def _byom_block(docs: list) -> dict:
    for entry in docs:
        if isinstance(entry, dict) and entry.get("id") == LLM_ENTRY[0]:
            providers = (entry.get("config") or {}).get("providers") or {}
            block = providers.get(PROVIDER_ID)
            return block if isinstance(block, dict) else {}
    return {}


def _assert_equivalent(original: str, updated: str, cfg: config.Config) -> None:
    """Recusa gravar quando o texto novo não é YAML válido ou perdeu conteúdo.

    A edição é textual e preserva comentários, então a única prova de que ela
    não corrompeu nada é reparsear e comparar a estrutura: original + o
    provider/model do byom, nada mais.
    """
    model_id = cfg.served_model_name
    try:
        new_docs = _load_yaml(updated)
    except yaml.YAMLError as exc:
        raise DshConfigError(
            f"o merge produziria YAML inválido ({exc}); o arquivo não foi alterado"
        ) from exc
    if not isinstance(new_docs, list):
        raise DshConfigError("o merge produziria um documento que não é uma lista YAML")
    if not original.strip():
        return
    try:
        old_docs = _load_yaml(original)
    except yaml.YAMLError as exc:  # pragma: no cover - `_read` só valida a forma
        raise DshConfigError(f"o arquivo atual não é YAML válido: {exc}") from exc
    if not isinstance(old_docs, list):
        raise DshConfigError("o arquivo atual não é uma lista YAML de entradas")
    old_ids = [entry.get("id") for entry in old_docs if isinstance(entry, dict)]
    new_ids = [entry.get("id") for entry in new_docs if isinstance(entry, dict)]
    added = [entry_id for entry_id in new_ids if entry_id not in old_ids]
    known = {LLM_ENTRY[0], DEFAULT_MODEL_ENTRY[0]}
    if any(entry_id not in known for entry_id in added) or [
        entry_id for entry_id in new_ids if entry_id in old_ids
    ] != old_ids:
        raise DshConfigError(
            "o merge mudaria a lista de entradas do arquivo "
            f"({old_ids} -> {new_ids}); nada foi gravado"
        )
    if _managed_view(old_docs, model_id) != _managed_view(
        new_docs, model_id, keep_ids=set(old_ids)
    ):
        raise DshConfigError(
            "o merge alteraria chaves fora de `providers.byom` e "
            "`agent-default-model`; nada foi gravado"
        )
    block = _byom_block(new_docs)
    expected_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    if block.get("api") != "openai-completions" or block.get("baseURL") != expected_url:
        raise DshConfigError("o merge não gravou `api`/`baseURL` do provider byom")
    models = block.get("models")
    if not isinstance(models, list) or not any(
        isinstance(item, dict) and item.get("id") == model_id for item in models
    ):
        raise DshConfigError(f"o merge não gravou o modelo {model_id!r} no provider byom")
    default = next(
        (
            entry
            for entry in new_docs
            if isinstance(entry, dict) and entry.get("id") == DEFAULT_MODEL_ENTRY[0]
        ),
        {},
    )
    default_config = default.get("config") or {}
    if default_config.get("provider") != PROVIDER_ID or default_config.get("model") != model_id:
        raise DshConfigError("o merge não tornou o modelo do byom o padrão do agente")


def _block_end(lines: list[str], start: int, limit: int, indent: int) -> int:
    """End (exclusive) of the block opened at ``start`` whose children are > ``indent``."""
    end = start + 1
    for j in range(start + 1, limit):
        if not _is_content(lines[j]):
            continue
        if _indent(lines[j]) <= indent:
            break
        end = j + 1
    return end


def _find_key(lines: list[str], start: int, end: int, indent: int, key: str) -> int | None:
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(key)}:(\s|$)")
    for i in range(start, end):
        if pattern.match(lines[i]):
            return i
    return None


def _find_entry(lines: list[str], entry_id: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"""^-\s+id:\s*["']?{re.escape(entry_id)}["']?\s*(#.*)?$""")
    for i, line in enumerate(lines):
        if pattern.match(line):
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("- "):
                    end = j
                    break
            while end > i + 1 and not _is_content(lines[end - 1]):
                end -= 1
            return i, end
    return None


def _item_prefix(indent: int, step: int) -> str:
    """Prefixo de um item de lista cujo primeiro campo alinha em `indent + step`.

    Em YAML o item é `- ` seguido do campo, então o campo começa em
    ``dash_col + 2``. Para o campo cair em ``item_indent + step`` o traço ganha
    ``step - 1`` espaços: com passo 2 sai ``- id:``; com passo 4, ``-   id:``.
    """
    return " " * indent + "-" + " " * (step - 1)


def _ensure_entry(lines: list[str], entry: tuple[str, str], step: int) -> tuple[int, int]:
    found = _find_entry(lines, entry[0])
    if found:
        return found
    if lines and lines[-1].strip():
        lines.append("")
    pad = " " * step
    lines += [
        f"{_item_prefix(0, step)}id: {entry[0]}",
        f'{pad}name: "{entry[1]}"',
        f"{pad}config:",
    ]
    return _find_entry(lines, entry[0])  # type: ignore[return-value]


def _ensure_key(
    lines: list[str], entry: tuple[str, str], parent: tuple[str, ...], step: int
) -> tuple[int, int]:
    """Locate (creating when missing) the mapping at ``parent`` below the entry.

    ``parent`` is the key chain under the entry (e.g. ("config", "providers"));
    returns the (line, end) of the last key. Keys follow the file's own indent
    step: ``step``, ``2*step``, ...
    """
    s, e = _ensure_entry(lines, entry, step)
    line = s
    end = e
    for depth, key in enumerate(parent):
        ind = step * (1 + depth)
        idx = _find_key(lines, line + 1, end, ind, key)
        if idx is None:
            idx = end
            lines.insert(idx, f"{' ' * ind}{key}:")
            end += 1
        else:
            text = lines[idx]
            value = text.split(":", 1)[1].split("#", 1)[0].strip()
            if value == "{}":
                lines[idx] = f"{' ' * ind}{key}:"
            elif value:
                raise DshConfigError(f"`{key}` in {entry[0]} must be a block mapping")
        line = idx
        end = _block_end(lines, idx, end, ind)
    return line, end


def _scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


#: Valores YAML "simples" que podem ir sem aspas na restauração. Escolher o
#: estilo do arquivo é o que faz o `down` devolver a linha como era.
_PLAIN_SCALAR_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def _restored_scalar(value: str) -> str:
    return value if _PLAIN_SCALAR_RE.fullmatch(value) else _scalar(value)


def default_model_values(path: Path) -> dict[str, str | None] | None:
    """`provider`/`model` do agente em `agent-default-model` (ou None se ausente)."""
    try:
        text = _read(path)
    except DshConfigError:
        return None
    if not text.strip():
        return None
    try:
        docs = _load_yaml(text)
    except yaml.YAMLError:
        return None
    if not isinstance(docs, list):
        return None
    for entry in docs:
        if isinstance(entry, dict) and entry.get("id") == DEFAULT_MODEL_ENTRY[0]:
            config_view = entry.get("config") or {}
            if not isinstance(config_view, dict):
                return None
            return {
                "provider": config_view.get("provider"),
                "model": config_view.get("model"),
            }
    return None


def restore_default_model(
    path: Path, provider: str | None, model: str | None
) -> bool:
    """Devolve `provider`/`model` do agente ao estado anterior.

    `None` remove a chave (ela não existia antes do `up`). A edição é textual,
    com o passo de indentação do próprio arquivo, e validada antes de gravar.
    """
    original = _read(path)
    if not original.strip():
        return False
    step = _indent_step(original)
    lines = original.splitlines()
    found = _find_entry(lines, DEFAULT_MODEL_ENTRY[0])
    if found is None:
        return False
    cfg_line, cfg_end = _ensure_key(lines, DEFAULT_MODEL_ENTRY, ("config",), step)
    key_indent = 2 * step
    alterado = False
    for key, value in (("provider", provider), ("model", model)):
        idx = _find_key(lines, cfg_line + 1, cfg_end, key_indent, key)
        if value is None:
            if idx is not None:
                del lines[idx]
                cfg_end -= 1
                alterado = True
            continue
        novo = f"{' ' * key_indent}{key}: {_restored_scalar(value)}"
        if idx is None:
            lines.insert(cfg_end, novo)
            cfg_end += 1
            alterado = True
        elif lines[idx] != novo:
            lines[idx] = novo
            alterado = True
    if not alterado:
        return False
    newline = "\r\n" if "\r\n" in original else "\n"
    updated = newline.join(lines) + (newline if original.endswith("\n") else "")
    _assert_valid_restore(original, updated)
    _backup(path)
    _atomic_write(path, updated)
    return True


def _assert_valid_restore(original: str, updated: str) -> None:
    """Valida a restauração: YAML válido e nada além do modelo padrão mudou."""
    try:
        new_docs = _load_yaml(updated)
        old_docs = _load_yaml(original)
    except yaml.YAMLError as exc:
        raise DshConfigError(
            f"a restauração produziria YAML inválido ({exc}); o arquivo não foi alterado"
        ) from exc
    if not isinstance(new_docs, list) or not isinstance(old_docs, list):
        raise DshConfigError("a restauração produziria um documento que não é uma lista")
    old_ids = [entry.get("id") for entry in old_docs if isinstance(entry, dict)]
    new_ids = [entry.get("id") for entry in new_docs if isinstance(entry, dict)]
    if old_ids != new_ids:
        raise DshConfigError("a restauração mudaria a lista de entradas; nada foi gravado")
    if _managed_view(old_docs, "") != _managed_view(new_docs, ""):
        raise DshConfigError(
            "a restauração alteraria chaves fora de `agent-default-model`; nada foi gravado"
        )


def _merge(text: str, cfg: config.Config, step: int) -> str:
    lines = text.splitlines()
    try:
        profile = config.load_profile(cfg.profile) if cfg.profile else None
    except ValueError as exc:
        raise DshConfigError(f"Invalid BYOM profile: {exc}") from exc
    model_id = cfg.served_model_name
    model_name = profile.name if profile else "Qwen3.8 27B (BYOM)"

    _merge_provider(lines, cfg, model_id, model_name, step)
    _set_default_model(lines, model_id, step)
    # Keep the file's own line endings and final-newline convention.
    newline = "\r\n" if "\r\n" in text else "\n"
    ends_with_newline = text.endswith("\n") or not text
    return newline.join(lines) + (newline if ends_with_newline else "")


def _merge_provider(
    lines: list[str], cfg: config.Config, model_id: str, model_name: str, step: int
) -> None:
    prov_line, prov_end = _ensure_key(lines, LLM_ENTRY, ("config", "providers"), step)
    byom_indent = 3 * step
    child_indent = 4 * step
    item_indent = 5 * step
    byom = _find_key(lines, prov_line + 1, prov_end, byom_indent, PROVIDER_ID)
    preserved_children: list[str] = []
    other_models: list[str] = []
    if byom is not None:
        value = lines[byom].split(":", 1)[1].split("#", 1)[0].strip()
        if value and value != "{}":
            raise DshConfigError("`providers.byom` must be a block mapping")
        byom_end = _block_end(lines, byom, prov_end, byom_indent)
        # Fail safely on layouts `byom up` does not write: children at another
        # indent would be dropped, and `- ` items at child level would be
        # re-nested into the mapping, corrupting the file.
        child_indent_found = next(
            (
                _indent(lines[j])
                for j in range(byom + 1, byom_end)
                if _is_content(lines[j])
            ),
            child_indent,
        )
        if child_indent_found != child_indent or any(
            _indent(lines[j]) == child_indent and lines[j].lstrip().startswith("- ")
            for j in range(byom + 1, byom_end)
            if _is_content(lines[j])
        ):
            raise DshConfigError(
                "`providers.byom` does not match the layout written by `byom up` "
                f"(child keys at {child_indent} spaces, list items deeper); "
                "fix or delete the block"
            )
        children = [
            i
            for i in range(byom + 1, byom_end)
            if _is_content(lines[i]) and _indent(lines[i]) == child_indent
        ]
        for n, i in enumerate(children):
            stop = children[n + 1] if n + 1 < len(children) else byom_end
            key = lines[i].strip().split(":", 1)[0]
            chunk = lines[i:stop]
            if key == "models":
                other_models = _other_models(chunk, model_id, step)
            elif key not in _MANAGED_KEYS:
                preserved_children += chunk
        del lines[byom:byom_end]
        insert_at = byom
    else:
        insert_at = prov_end

    pad = " " * child_indent
    if not any(ln.strip().startswith("apiKeyEnv:") for ln in preserved_children):
        preserved_children.insert(0, f"{pad}apiKeyEnv: {API_KEY_ENV}")
    # User-owned keys first, so re-running is a no-op regardless of who added them.
    item_start = _item_prefix(item_indent, step)
    item_key_pad = " " * (item_indent + step)
    block = [
        f"{' ' * byom_indent}{PROVIDER_ID}:",
        *preserved_children,
        f"{pad}api: openai-completions",
        f"{pad}baseURL: http://127.0.0.1:{cfg.proxy_port}/v1",
        f"{pad}models:",
        f"{item_start}id: {_scalar(model_id)}",
        f"{item_key_pad}name: {_scalar(model_name)}",
        f"{item_key_pad}contextWindow: {cfg.max_model_len}",
        *other_models,
    ]
    lines[insert_at:insert_at] = block


def _other_models(chunk: list[str], model_id: str, step: int) -> list[str]:
    """Model list items of ``chunk`` (a `models:` block) that are not ours."""
    first = chunk[0].split(":", 1)[1].split("#", 1)[0].strip()
    if first and first != "[]":
        raise DshConfigError("`providers.byom.models` must be a block list")
    item_indent = 5 * step
    # The first `- ` line in the chunk is necessarily a top-level item; items at
    # another indent would be silently dropped by the indent-anchored scan below.
    first_item = next(
        (i for i in range(1, len(chunk)) if re.match(rf"^ {{{item_indent}}}-\s+", chunk[i])),
        None,
    )
    if first_item is None:
        other_item = next((i for i in range(1, len(chunk)) if re.match(r"^ +-\s+", chunk[i])), None)
        if other_item is not None:
            raise DshConfigError(
                "`providers.byom.models` does not match the layout written by `byom up` "
                f"(items at {item_indent} spaces); fix or delete the list"
            )
    item_pattern = rf"^ {{{item_indent}}}-\s+"
    starts = [i for i, ln in enumerate(chunk) if re.match(item_pattern, ln)]
    kept: list[str] = []
    for n, i in enumerate(starts):
        stop = starts[n + 1] if n + 1 < len(starts) else len(chunk)
        item = chunk[i:stop]
        match = re.match(rf"""^{item_pattern}id:\s*["']?([^"'\s#]+)["']?""", item[0])
        if match and match.group(1) == model_id:
            continue
        kept += item
    return kept


def _set_default_model(lines: list[str], model_id: str, step: int) -> None:
    cfg_line, cfg_end = _ensure_key(lines, DEFAULT_MODEL_ENTRY, ("config",), step)
    key_indent = 2 * step
    for key, value in (("provider", PROVIDER_ID), ("model", model_id)):
        idx = _find_key(lines, cfg_line + 1, cfg_end, key_indent, key)
        new = f"{' ' * key_indent}{key}: {_scalar(value)}"
        if idx is None:
            lines.insert(cfg_end, new)
            cfg_end += 1
        else:
            lines[idx] = new



def _atomic_write(path: Path, contents: str) -> None:
    try:
        # `newline=""` keeps CRLF files unchanged.
        config.atomic_write_text(path, contents, newline="")
    except OSError as exc:
        raise DshConfigError(f"Could not write {path}: {exc}") from exc
