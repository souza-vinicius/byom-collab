# byom-collab Knowledge Base

Open Knowledge Format (OKF v0.2) knowledge base for `byom-collab`.

- [VRAM Budget and GPU Eligibility](./catalog.md) - Weights + vision projector + runtime + dtype-aware KV (heuristic or per-architecture) + DSpark drafter + mandatory headroom, and why equality never approves
- [Speculative Decoding in the Worker](./speculative-decoding.md) - `[worker.spec]` versus the legacy `dspark` shorthand, the types whose draft head is embedded in the main file, and how the drafter reaches the VRAM budget
- [Model Provider and Weight Package](./provider.md) - The two names shown per model: who develops it (catalog data with a dated source) and which repository publishes the weights byom downloads
- [Worker Runtime, Binaries and CUDA Libraries](./worker-runtime.md) - Bootstrap steps, pinned binaries, the CUDA 12.8 decision with its SASS/PTX coverage, library resolution, and the gateway that owns readiness on the session's `vllm_port`
- [Connection Path: Local Proxy, SSH Tunnel and Control Sockets](./connection-path.md) - Proxy to gateway over a single `colab ssh` slot, the separate control sockets of `up` and the tunnel, nonce identity and the watchdog channel
- [Troubleshooting the Worker Session](./troubleshooting.md) - Symptom to cause to action: silent runtime exit, missing CUDA libraries, unsupported PTX, worker stuck initializing, tunnel drops, and how to read the VM logs without taking the SSH slot
- [Interactive TUI Architecture](./tui.md) - Single-screen Rich + questionary TUI with the `Selecionar GPU` submenu, GPU + model pinning, and responsive layout
- [Isolated llm chat Integration](./llm-chat.md) - Centralized `byom_home()` path resolution and `setup`, `chat`, and `uninstall` commands wrapping `llm chat`
- [Update Log](./log.md) - Chronological history of knowledge base updates
