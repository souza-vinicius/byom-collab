# tui-design (skill)

> Design, build, refactor, or review terminal interfaces — full-screen TUIs,
> interactive prompts, and command-line tools.

Skill de terceiros, instalada sem modificação a partir do upstream:

- Origem: [gfargo/tui-design-skill](https://github.com/gfargo/tui-design-skill) (MIT)
- Versão do plugin: **1.9.0**
- Commit fixado: `0a0ce996e00fc2a5db7f1f85495257381cc90df9` (também em `UPSTREAM_COMMIT`)
- Subpasta copiada: `plugins/tui-design/skills/tui-design/`

O que **não** veio junto, de propósito: manifestos de plugin
(`.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`), `evals/`,
`tests/`, `scripts/` e workflows do upstream. Este diretório é a skill
autocontida no formato Agent Skills (`SKILL.md` + `references/`), descoberta
por qualquer harness que leia `.agents/skills/`.

Para atualizar: copie de novo a subpasta acima do commit novo e reaproveite
`UPSTREAM_COMMIT`. Não edite os arquivos aqui — divergência local some no
próximo update; ajustes de comportamento pertencem ao upstream.
