# TODO

## MoE / Mixed-Precision Support

Production LLM serving is converging on MoE architectures with mixed-precision
quantization (e.g. gpt-oss-120b, DeepSeek-V3/R1, Llama 4). The calculator
currently applies a single precision to all parameters, which causes ~2×
throughput error for these models. The following changes address that.

### 1. Two-precision model  *(highest priority)*

Split the single precision selector into two:

- **Expert / FFN weights** — the quantized bulk (MXFP4, FP8, INT4, INT8)
- **Attention + embed weights** — always-on, typically BF16

Impact on the math:

| Formula | Current | Fixed |
|---|---|---|
| Weight VRAM | `params × bytesPerParam` | `expertParams × expertBytes + otherParams × otherBytes` |
| Bandwidth/token | `activeParams × bytesPerParam` | `activeExpertParams × expertBytes + nonExpertParams × otherBytes` |

For dense models both precision slots default to the same value — no UX
regression. For MoE models the split is meaningful.

The HuggingFace fetch should auto-populate both precisions from
`quantization_config.modules_to_not_convert` in config.json where available.

### 2. Expert structure as first-class model fields

Replace the opaque `activeParams` field with the actual architectural inputs
the user understands:

- `n_experts` — total routed experts per layer
- `n_active` — experts selected per token
- `moe_intermediate_size` — expert FFN hidden dim (may differ from main FFN dim)

Derive `expertParams` and `activeExpertParams` from these rather than asking
the user to compute and enter the final number. The HuggingFace fetch already
reads `n_routed_experts` and `num_experts_per_tok` from config.json — promote
them to UI fields instead of discarding them.

### 3. VRAM breakdown: surface expert vs. non-expert split

For MoE models, the expert FFN weights sitting cold in VRAM (e.g. 57 GB for
gpt-oss-120b) are conceptually separate from the always-active attention +
embed weights (~5 GB). Showing these as distinct segments in the VRAM bar
makes it immediately clear why a model with only 5.1B active params still
requires 62 GB of VRAM.

### 4. Bandwidth formula uses mixed bytes-per-token

Once (1) and (2) are in place, `calcThroughput` bandwidth per token becomes:

```
modelBytes = activeExpertParams × expertBytes + nonExpertParams × otherBytes
```

This replaces the current `activeParams × bytesPerParam` and is the fix that
closes the ~2× throughput error on gpt-oss-120b and DeepSeek-V3.
