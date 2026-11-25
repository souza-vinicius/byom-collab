# Instruções para Agentes no Repositório

## Inicialização Obrigatória (`pstack-skill`)

**Sempre carregue e aplique a skill `pstack-skill` no início de cada sessão**, antes de iniciar qualquer tarefa de análise, código, depuração ou revisão:

1. Leia [`.agents/skills/pstack-skill/SKILL.md`](.agents/skills/pstack-skill/SKILL.md).
2. Carregue o mapeamento de ferramentas, subagentes e caminhos correspondente ao harness em uso:
   - **Antigravity**: [`.agents/skills/pstack-skill/references/harness/antigravity.md`](.agents/skills/pstack-skill/references/harness/antigravity.md)
   - **OpenCode**: [`.agents/skills/pstack-skill/references/harness/opencode.md`](.agents/skills/pstack-skill/references/harness/opencode.md)
   - **Claude Code**: [`.agents/skills/pstack-skill/references/harness/claude-code.md`](.agents/skills/pstack-skill/references/harness/claude-code.md)
3. Utilize sempre o **modelo em execução na sessão atual** para todos os papéis (`worker`, `builder`, `judge`, `peer`), conforme definido em [`.agents/pstack-models.md`](.agents/pstack-models.md) (`inherit-parent` / `Model: "inherit"`).

## Manutenção da Base de Conhecimento (`docs/`)

Este repositório utiliza `docs/` como uma base de conhecimento no formato **Open Knowledge Format (OKF v0.2)**. Siga a skill `okf-open-knowledge-format` em `.agents/skills/okf-open-knowledge-format/SKILL.md` sempre que ler, criar ou atualizar documentos da base de conhecimento.

### Quando Atualizar `docs/`

Mantenha `docs/` sincronizada sempre que alterar ou adicionar:
- Arquitetura, módulos ou ciclo de vida do worker (`src/byom/`, `worker/`)
- Catálogos de GPUs/modelos, regras de quantização ou lógica de cálculo de VRAM (`src/byom/catalog.py`)
- Comandos de CLI, flags, interface interativa (TUI) ou esquemas de configuração (`config.toml`, `state.json`)
- Runbooks operacionais, procedimentos de solução de problemas ou decisões de design

### Regras OKF a Seguir

1. **Um conceito por arquivo**: cada arquivo `.md` em `docs/` (exceto os arquivos reservados `index.md` e `log.md`) representa um único conceito e DEVE ter frontmatter YAML válido com um campo `type` não vazio.
2. **Frontmatter recomendado**: inclua `title`, `description`, `tags`, `status` (`draft` | `stable` | `deprecated`), `generated` (`{ by: <ator>, at: <ISO8601> }`) e `sources` (com referências relativas ao bundle ou ao repositório e notas de rodapé quando aplicável).
3. **Referências cruzadas**: use links Markdown padrão. Prefira caminhos relativos ao bundle começando com `/` (ex.: `[Catálogo de GPUs](/catalog/gpus.md)`) para que os links continuem válidos caso arquivos sejam movidos.
4. **Divulgação progressiva (`index.md`)**: mantenha `docs/index.md` (e eventuais arquivos `index.md` em subdiretórios) atualizado com links e descrições de uma linha para cada conceito. Arquivos `index.md` de subdiretórios NÃO devem ter frontmatter; apenas o `docs/index.md` raiz pode declarar `okf_version: "0.2"` (quando compatível com o validador).
5. **Histórico de alterações (`log.md`)**: registre adições, atualizações ou descontinuações em `docs/log.md` sob títulos de data ISO 8601 (`## YYYY-MM-DD`), do mais recente para o mais antigo, sem frontmatter YAML.

### Validação

Após criar ou editar arquivos em `docs/`, valide o bundle:
- Se `okflint` estiver instalado: `okflint validate docs/`
- Caso contrário, execute o validador incluído na skill: `.agents/skills/okf-open-knowledge-format/scripts/validate.sh docs/`
