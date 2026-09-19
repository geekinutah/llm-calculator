# TODO / Roadmap

The calculator already supports dense and MoE models, mixed expert/attention
precision, Hugging Face config discovery, explicit expert structure, KV-cache
VRAM accounting, and roofline-based throughput estimates.

The next work should focus on architectural changes that materially affect
memory footprint or inference throughput.

## 1. Context-aware attention / KV-cache modeling *(highest priority)*

Current throughput math is dominated by model-weight traffic. Sequence length
affects VRAM, but does not yet materially affect decode throughput.

Add explicit modeling for:

- **MLA / latent KV representations** — model compressed latent KV state rather
  than assuming conventional K/V tensors per layer.
- **GQA / MQA** — preserve current `kvHeads` support but make attention type a
  first-class architecture field.
- **Sliding-window / hybrid attention** — account for models where only part of
  the context is retained by some layers.
- **Sparse attention** — support long-context designs where attention cost grows
  more slowly than dense O(n) decode attention / O(n²) prefill attention.
- **KV-cache quantization** — separate KV precision from weight precision and
  support FP8 / INT8 / lower-bit cache formats where the serving stack allows it.

Update both VRAM and throughput calculations. At long context, KV reads and
attention work can become large enough that the current weight-only decode model
is optimistic.

## 2. Separate prefill and decode performance models

The UI currently presents one roofline estimate, while real serving has two
different regimes:

- **Prefill** — high arithmetic intensity; attention and GEMMs dominate.
- **Decode** — low batch is typically bandwidth-sensitive; weight reads, KV
  reads, attention, and scheduler overhead all matter.

Expose separate estimates for:

- time to first token / prefill throughput
- decode tokens/sec
- aggregate batched throughput

Use sequence length explicitly in both paths.

## 3. Multi-GPU communication efficiency

Current tensor-parallel scaling assumes TFLOPS and memory bandwidth scale
linearly with GPU count.

Add a TP efficiency model based on:

- NVLink / NVSwitch bandwidth when present
- PCIe-only interconnects
- tensor-parallel degree
- model size / hidden dimension
- collective communication overhead

At minimum, surface an estimated TP efficiency factor rather than assuming 100%.

## 4. Modern architecture metadata from Hugging Face

Normalize newer config variants into first-class calculator fields instead of
adding one-off aliases indefinitely.

Track and map fields for:

- attention type
- latent / compressed KV dimensions
- sliding-window size and layer pattern
- sparse-attention configuration
- KV-cache dtype when declared
- MoE routing structure
- shared experts
- per-layer dense-vs-MoE patterns
- explicit `head_dim`

Keep architecture parsing isolated from the math engine so new DeepSeek, Qwen,
Kimi, GLM, and similar model families can be added without touching UI logic.

## 5. Refresh benchmark validation

Expand `tests/fixtures.json` with current open-model serving results covering:

- recent DeepSeek models
- recent Qwen models
- recent Kimi models
- recent GLM models
- dense vs MoE
- short vs long context
- single-GPU vs tensor-parallel
- BF16 / FP8 / INT4-class quantization

Prefer benchmarks that report enough detail to reproduce batch size, context,
precision, GPU count, and serving engine.

Validation should continue to execute the production math engine directly.

## 6. Model architecture presets / diagnostics

When Hugging Face metadata is incomplete or ambiguous:

- identify the detected architecture family
- show which fields were inferred vs explicitly supplied
- warn when the calculator falls back to conventional KV-cache assumptions
- show the exact formulas used for weight VRAM, KV VRAM, and throughput

This should make estimates easier to audit as architectures become less uniform.

---

## Recently completed

- Mixed expert vs attention/embed precision for MoE models
- First-class expert count / active expert / expert FFN fields
- Expert vs non-expert VRAM breakdown
- Mixed-precision MoE bandwidth calculations
- Quantization detection from Hugging Face configs
- `num_local_experts` / `experts_per_token` compatibility
- Roofline ridge-point batch calculation
- KV cache tied to attention precision rather than expert precision
- Explicit `head_dim` support for models where `hidden / heads` is incorrect
