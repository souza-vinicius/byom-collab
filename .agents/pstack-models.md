# Configuração de papéis de modelo da pstack-skill

# Todos os papéis herdam explicitamente o modelo em execução na sessão atual
# (compatível com Antigravity e OpenCode, inclusive modelos servidos via byom).
# budget: session

worker: inherit-parent
builder: inherit-parent
judge: inherit-parent
peer: inherit-parent
