// ═══════════════════════════════════════════════════════════
//  MATH ENGINE
// ═══════════════════════════════════════════════════════════

function getTFLOPS(gpu, prec) {
  if (!gpu) return null;
  const map = {
    fp32: gpu.fp32, bf16: gpu.bf16, fp16: gpu.fp16,
    fp8: gpu.fp8, int8: gpu.int8, int4: gpu.int4, fp4: gpu.fp4
  };
  return map[prec] ?? null;
}

function inferParams(cfg, vocabFallback) {
  // Some models expose safetensors metadata with total count
  if (cfg.safetensors?.total) return cfg.safetensors.total;
  // Estimate from architecture if we have the dims
  const L = cfg.num_hidden_layers;
  const H = cfg.hidden_size;
  const V = cfg.vocab_size ?? vocabFallback;
  if (!L || !H || !V) return null;

  const attn = 4 * H * H; // Q+K+V+O projections
  const emb = 2 * V * H; // embedding + lm_head

  // MoE architecture
  const nRouted = cfg.n_routed_experts ?? cfg.num_experts ?? cfg.num_local_experts ?? 0;
  // gpt_oss and similar models store expert FFN size in intermediate_size
  const moeFFN = cfg.moe_intermediate_size ?? (nRouted > 0 ? cfg.intermediate_size : null);
  if (moeFFN && nRouted > 0) {
    const nShared = cfg.n_shared_experts ?? 0;
    const freq = cfg.moe_layer_freq ?? 1;
    const moeLayers = Math.ceil(L / freq);
    const denseLayers = L - moeLayers;
    const ffnPerMoeLayer = (nRouted + nShared) * 3 * H * moeFFN;
    const denseFFN = cfg.intermediate_size;
    const ffnPerDenseLayer = denseFFN ? 3 * H * denseFFN : 0;
    return emb + moeLayers * (attn + ffnPerMoeLayer) + denseLayers * (attn + ffnPerDenseLayer);
  }

  // Dense architecture
  const FFN = cfg.intermediate_size ?? cfg.ffn_dim ?? cfg.inner_dim;
  if (!FFN) return null;
  return emb + L * (attn + 3 * H * FFN);
}

function inferActiveParams(cfg) {
  const nRouted = cfg.n_routed_experts ?? cfg.num_experts ?? cfg.num_local_experts ?? 0;
  const moeFFN = cfg.moe_intermediate_size ?? (nRouted > 0 ? cfg.intermediate_size : null);
  const activePerTok = cfg.num_experts_per_tok ?? cfg.n_activated_experts ?? cfg.experts_per_token ?? null;
  if (!moeFFN || !nRouted || activePerTok === null) return null;

  const L = cfg.num_hidden_layers;
  const H = cfg.hidden_size;
  const V = cfg.vocab_size;
  if (!L || !H || !V) return null;

  const attn = 4 * H * H;
  const emb = 2 * V * H;
  const nShared = cfg.n_shared_experts ?? 0;
  const freq = cfg.moe_layer_freq ?? 1;
  const moeLayers = Math.ceil(L / freq);
  const denseLayers = L - moeLayers;

  const activeFFNPerMoeLayer = (activePerTok + nShared) * 3 * H * moeFFN;
  const denseFFN = cfg.intermediate_size;
  const ffnPerDenseLayer = denseFFN ? 3 * H * denseFFN : 0;

  return (emb + moeLayers * (attn + activeFFNPerMoeLayer) + denseLayers * (attn + ffnPerDenseLayer)) / 1e9;
}

function splitMoEParams(model) {
  if (
    model.nExperts != null && model.nActive != null &&
    model.moeIntermediateSize != null &&
    model.layers != null && model.hidden != null
  ) {
    const expertParamsB = model.nExperts * model.layers * 3 * model.hidden * model.moeIntermediateSize / 1e9;
    const activeExpertParamsB = (model.nActive / model.nExperts) * expertParamsB;
    const nonExpertParamsB = (model.params ?? 0) - expertParamsB;
    const otherParamsB = nonExpertParamsB;
    return { expertParamsB, activeExpertParamsB, nonExpertParamsB, otherParamsB };
  } else if (model.activeParams != null) {
    return { expertParamsB: null, activeExpertParamsB: model.activeParams, nonExpertParamsB: null, otherParamsB: null };
  } else {
    return { expertParamsB: null, activeExpertParamsB: null, nonExpertParamsB: null, otherParamsB: null };
  }
}

function calcVRAM(model, gpu, precision, gpuCount, batch, seqLen, expertPrecision = precision, otherPrecision = precision) {
  const expertBytes = PREC_META[expertPrecision]?.bytes ?? 2;
  const otherBytes  = PREC_META[otherPrecision]?.bytes  ?? 2;
  const totalVram = (gpu?.vram ?? 80) * gpuCount;

  // 1. Weights
  const paramsB = model.params ?? estimateParams(model);
  const { expertParamsB, otherParamsB } = splitMoEParams(model);

  let weightsGB, expertWeightsGB = null, otherWeightsGB = null;
  if (expertParamsB !== null && otherParamsB !== null) {
    expertWeightsGB = (expertParamsB * 1e9 * expertBytes) / 1e9;
    otherWeightsGB  = (otherParamsB  * 1e9 * otherBytes)  / 1e9;
    weightsGB = expertWeightsGB + otherWeightsGB;
  } else if (paramsB) {
    // Legacy / dense path: single precision
    weightsGB = (paramsB * 1e9 * expertBytes) / 1e9;
  } else {
    weightsGB = null;
  }

  // 2. KV Cache
  // Per token: 2 (K+V) * layers * kvHeads * head_dim * kvBytes
  // head_dim may be set explicitly (e.g. Qwen3: hidden=2048, heads=32, head_dim=128 ≠ 64).
  // Fall back to hidden/heads when not provided.
  // KV is produced by attention layers, which run at otherPrecision (typically BF16),
  // not at the quantized expert precision. Floor at 1 byte (no sub-byte KV in practice).
  const kvBytes = Math.max(1, otherBytes);
  let kvGB = null;
  const kvSeq = seqLen || model.context;
  if (model.layers && model.kvHeads && model.hidden && model.heads && kvSeq) {
    const headDim = model.headDim || (model.hidden / model.heads);
    const kvPerToken = 2 * model.layers * model.kvHeads * headDim * kvBytes;
    kvGB = (kvPerToken * kvSeq * batch) / 1e9;
  }

  // 3. Activations (peak prefill: hidden * seq_len * batch * 4 bytes)
  // Decode-phase activations are per-token and negligible; this models the prefill peak.
  let actGB = null;
  const actSeq = seqLen || model.context;
  if (model.hidden && actSeq) {
    actGB = (model.hidden * actSeq * batch * 4) / 1e9;
  }

  // 4. CUDA overhead (fixed)
  const cudaGB = 1.0;

  const used = (weightsGB ?? 0) + (kvGB ?? 0) + (actGB ?? 0) + cudaGB;
  const freeGB = Math.max(0, totalVram - used);
  const overflow = Math.max(0, used - totalVram);

  return { weightsGB, expertWeightsGB, otherWeightsGB, kvGB, actGB, cudaGB, freeGB, totalVram, used, overflow };
}

function calcThroughput(model, gpu, precision, gpuCount, batch, expertPrecision = precision, otherPrecision = precision) {
  if (!gpu) return null;
  const prec = precision;

  // Effective TFLOPS across N GPUs
  let tflops = getTFLOPS(gpu, prec);
  if (tflops === null) return null;
  tflops = tflops * gpuCount;

  // Effective memory bandwidth across N GPUs (BW scales with GPU count for TP)
  const bwGBs = (gpu.bw ?? 3350) * gpuCount;

  const totalParamsB = model.params ?? estimateParams(model);
  if (!totalParamsB) return null;

  const expertBytes = PREC_META[expertPrecision]?.bytes ?? 2;
  const otherBytes  = PREC_META[otherPrecision]?.bytes  ?? 2;
  const { activeExpertParamsB, nonExpertParamsB } = splitMoEParams(model);

  // For MoE: active params drive FLOPs per token (must come before effectiveBytesPerParam)
  const paramsB = activeExpertParamsB ?? (model.activeParams ?? totalParamsB);

  let modelBytes;
  let effectiveBytesPerParam; // used for ridgeBatch calc at end
  if (activeExpertParamsB !== null && nonExpertParamsB !== null) {
    // Full mixed-precision: active expert weights + always-on non-expert weights
    modelBytes = activeExpertParamsB * 1e9 * expertBytes + nonExpertParamsB * 1e9 * otherBytes;
    // ridgeBatch uses paramsB (active FLOPs params), not totalParamsB
    effectiveBytesPerParam = modelBytes / (paramsB * 1e9);
  } else if (activeExpertParamsB !== null) {
    // Legacy activeParams path: just activeParams at expertPrecision
    modelBytes = activeExpertParamsB * 1e9 * expertBytes;
    effectiveBytesPerParam = expertBytes;
  } else {
    // Dense / single precision
    modelBytes = totalParamsB * 1e9 * expertBytes;
    effectiveBytesPerParam = expertBytes;
  }

  const bytesPerParam = PREC_META[prec]?.bytes ?? 2;

  // Arithmetic intensity threshold (ops/byte) = TFLOPS / BW
  const ridgePoint = (tflops * 1e12) / (bwGBs * 1e9); // ops/byte

  // Decode step: per-token FLOPs ≈ 2 * active_params (each active weight touched once)
  const flopsPerToken = 2 * paramsB * 1e9;

  // Memory-bound time per token (decode): load all weights from VRAM
  const memTimeSec = modelBytes / (bwGBs * 1e9);

  // Compute-bound time per token (prefill-like, batched):
  // At large batch, arithmetic intensity = 2 * batch * params / modelBytes
  const arithIntensity = (2 * batch * paramsB * 1e9) / modelBytes;

  const isComputeBound = arithIntensity >= ridgePoint;

  // Throughput estimate using roofline
  let tokensPerSecPerGpu;
  if (isComputeBound) {
    // Compute bound: limited by TFLOPS
    const computeTimeSec = flopsPerToken / (tflops * 1e12 / batch);
    tokensPerSecPerGpu = batch / computeTimeSec;
  } else {
    // Memory bound: limited by bandwidth
    // Each new token requires loading model weights
    tokensPerSecPerGpu = (bwGBs * 1e9) / modelBytes * batch;
  }

  // Apply realistic MFU (35-50% for memory-bound, 40-55% for compute-bound)
  const mfu = isComputeBound ? 0.45 : 0.40;
  const tps = tokensPerSecPerGpu * mfu;

  // FLOPS utilization % = (actual FLOPs used) / (peak FLOPs)
  const actualFlopsUsed = flopsPerToken * tps;
  const peakFlops = tflops * 1e12;
  const flopsUtil = Math.min(100, (actualFlopsUsed / peakFlops) * 100);

  const ridgeBatch = Math.max(1, Math.ceil(ridgePoint * effectiveBytesPerParam / 2));

  return {
    tps: Math.round(tps),
    ridgeBatch,
    flopsUtil,
    isComputeBound,
    ridgePoint,
    arithIntensity,
    mfu,
    peakTFLOPS: tflops,
    bwGBs,
    memTimeSec,
    flopsPerToken,
    bytesPerParam,
  };
}

function calcMaxBatch(model, gpu, precision, gpuCount, seqLen, expertPrecision = precision, otherPrecision = precision) {
  const expertBytes = PREC_META[expertPrecision]?.bytes ?? 2;
  const otherBytes  = PREC_META[otherPrecision]?.bytes  ?? 2;
  const totalVram = (gpu?.vram ?? 80) * gpuCount;
  const paramsB = model.params ?? estimateParams(model);
  if (!paramsB) return 1;

  const { expertParamsB, otherParamsB } = splitMoEParams(model);

  let weightsGB;
  if (expertParamsB !== null && otherParamsB !== null) {
    weightsGB = (expertParamsB * 1e9 * expertBytes + otherParamsB * 1e9 * otherBytes) / 1e9;
  } else {
    weightsGB = (paramsB * 1e9 * expertBytes) / 1e9;
  }

  const freeGB = totalVram - weightsGB - 1.0; // minus CUDA overhead
  if (freeGB <= 0) return 1;

  const kvBytes = Math.max(1, otherBytes); // KV lives in attention layers → otherPrecision
  let perReqGB = 0;
  if (model.layers && model.kvHeads && model.hidden && model.heads && seqLen) {
    const headDim = model.headDim || (model.hidden / model.heads);
    perReqGB += (2 * model.layers * model.kvHeads * headDim * kvBytes * seqLen) / 1e9;
  }
  if (model.hidden && seqLen) {
    perReqGB += (model.hidden * seqLen * 4) / 1e9;
  }
  if (perReqGB <= 0) return Math.max(1, Math.floor(freeGB / 0.1)); // fallback

  return Math.max(1, Math.floor(freeGB / perReqGB));
}

function estimateParams(model) {
  if (!model.layers || !model.hidden || !model.ffn || !model.vocab) return null;
  const attn = 4 * model.hidden * model.hidden;
  const ffn = 3 * model.hidden * model.ffn;
  const emb = 2 * model.vocab * model.hidden;
  return (emb + model.layers * (attn + ffn)) / 1e9;
}

if (typeof module !== 'undefined') {
  global.PREC_META = require('./gpus.js').PREC_META;
  module.exports = { getTFLOPS, estimateParams, inferParams, inferActiveParams, calcVRAM, calcThroughput, calcMaxBatch };
}
