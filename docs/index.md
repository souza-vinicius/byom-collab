# byom-collab Knowledge Base

Open Knowledge Format (OKF v0.2) knowledge base for `byom-collab`.

- [VRAM Budget and GPU Eligibility](./catalog.md) - Weights + vision projector + runtime + dtype-aware KV (heuristic or per-architecture) + DSpark drafter + mandatory headroom, and why equality never approves
- [Speculative Decoding in the Worker](./speculative-decoding.md) - `[worker.spec]` versus the legacy `dspark` shorthand, the types whose draft head is embedded in the main file, and how the drafter reaches the VRAM budget
- [Model Provider and Weight Package](./provider.md) - The two names shown per model: who develops it (catalog data with a dated source) and which repository publishes the weights byom downloads
- [Interactive TUI Architecture](./tui.md) - Single-screen Rich + questionary TUI with the `Selecionar GPU` submenu, GPU + model pinning, and responsive layout
- [Isolated llm chat Integration](./llm-chat.md) - Centralized `byom_home()` path resolution and `setup`, `chat`, and `uninstall` commands wrapping `llm chat`
- [Update Log](./log.md) - Chronological history of knowledge base updates
