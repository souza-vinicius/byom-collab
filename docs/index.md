# byom-collab Knowledge Base

Open Knowledge Format (OKF v0.2) knowledge base for `byom-collab`.

- [VRAM Budget and GPU Eligibility](./catalog.md) - Weights + vision projector + runtime + dtype-aware KV (heuristic or per-architecture) + DSpark drafter + mandatory headroom, and why equality never approves
- [Speculative Decoding in the Worker](./speculative-decoding.md) - `[worker.spec]` versus the legacy `dspark` shorthand, the types whose draft head is embedded in the main file, and how the drafter reaches the VRAM budget
- [Model Provider and Weight Package](./provider.md) - The two names shown per model: who develops it (catalog data with a dated source) and which repository publishes the weights byom downloads
- [Worker Runtime, Binaries and CUDA Libraries](./worker-runtime.md) - Bootstrap steps, pinned binaries, the CUDA 12.8 decision with its SASS/PTX coverage, library resolution, and the gateway that owns readiness on the session's `vllm_port`
- [Connection Path: Local Proxy, SSH Tunnel and Control Sockets](./connection-path.md) - Proxy to gateway over a single `colab ssh` slot, the separate control sockets of `up` and the tunnel, nonce identity and the watchdog channel
- [Troubleshooting the Worker Session](./troubleshooting.md) - Symptom to cause to action: silent runtime exit, CUDA libraries, PTX, stuck worker, ports in use, Colab allocation refusals (HTTP 5xx/412/400), timeout reasons, and how to read VM logs without taking the SSH slot
- [Interactive TUI Architecture](./tui.md) - Single-screen Rich + questionary TUI with the `Selecionar GPU` submenu, GPU + model pinning, responsive layout, and `NO_COLOR`/EOF handling
- [Isolated llm chat Integration](./llm-chat.md) - Isolated `llm` setup under `byom_home()`, `setup`/`chat`/`uninstall` flags, and clean Ctrl+C/Ctrl+D cancellation
- [Open WebUI Frontend](./webui.md) - `byom web` starts an isolated Open WebUI on 127.0.0.1:8080 with the byom proxy auto-configured, `DEFAULT_MODELS` preselected and user environment overrides preserved; the one-runtime limit still blocks image generation
- [Qwen3-TTS Runtime](./qwen3-tts.md) - Qwen3-TTS CustomVoice through vLLM-Omni, the two-stage VRAM budget, the `/v1/audio/speech` and `/v1/audio/voices` routes, and what is still unmeasured
- [Qwen-Image-2.1 Turbo Uncensored](./qwen-image-turbo-uncensored.md) - The Turbo image pipeline with an abliterated Qwen3-VL encoder, its two quantization pairs, the fixed 8-step sigma schedule, and why it is a separate catalog model
- [Update Log](./log.md) - Chronological history of knowledge base updates
