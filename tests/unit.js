#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════
//  Unit Tests — LLM Calculator
//
//  Tests internal mathematical invariants. Every assertion is
//  an exact value derived from the formulas in app.js — not a
//  range check, not a benchmark comparison (see validate.js).
//
//  GPU data (GPU_DB, PREC_META) comes from static/gpus.js.
//  Calculator functions (calcVRAM, calcThroughput, etc.) come
//  from static/app.js.
//
//  Run: npm test  |  node tests/unit.js
//  Requires Node 18+ (uses node:test built-in)
// ═══════════════════════════════════════════════════════════

const { test } = require('node:test');
const assert   = require('assert/strict');
const {
  getTFLOPS, estimateParams, inferParams, inferActiveParams,
  calcVRAM, calcThroughput, calcMaxBatch,
} = require('../static/math.js');
const { GPU_DB, PREC_META } = require('../static/gpus.js');

// ── Shared fixtures ──────────────────────────────────────────
//
// MODEL_FULL: Llama-3.1-8B-like architecture with all dims set.
// Used for KV-cache and maxBatch tests that need full arch info.
const MODEL_FULL = {
  params: 7, layers: 32, hidden: 4096, ffn: 14336,
  heads: 32, kvHeads: 8, vocab: 128256, context: 4096,
};
// MODEL_PARAMS: params only — weight and throughput tests that
// don't need KV-cache geometry.
const MODEL_PARAMS  = { params: 7 };
const MODEL_MOE     = { params: 235, activeParams: 22 };
const MODEL_MOE_SPLIT = {
  params: 671, layers: 94, hidden: 7168,
  heads: 128, kvHeads: 8, vocab: 201088,
  nExperts: 128, nActive: 4, moeIntermediateSize: 2048,
};
const MODEL_DENSE22 = { params: 22 };

const A100P = GPU_DB.a100_pcie;   // 80 GB, 1935 GB/s, bf16=312, int8=624
const H100  = GPU_DB.h100;        // 80 GB, 3350 GB/s, bf16=989, fp8=1979
const A100  = GPU_DB.a100;        // 80 GB, 2000 GB/s, fp8=null

// ═══════════════════════════════════════════════════════════
//  getTFLOPS
// ═══════════════════════════════════════════════════════════

test('getTFLOPS — returns correct value for supported precision', () => {
  assert.strictEqual(getTFLOPS(H100,  'bf16'), 989);
  assert.strictEqual(getTFLOPS(H100,  'fp8'),  1979);
  assert.strictEqual(getTFLOPS(A100P, 'bf16'), 312);
  assert.strictEqual(getTFLOPS(A100P, 'int8'), 624);
});

test('getTFLOPS — returns null for unsupported precision', () => {
  assert.strictEqual(getTFLOPS(A100,  'fp8'),  null);  // Ampere: no FP8 hw
  assert.strictEqual(getTFLOPS(A100P, 'fp8'),  null);
  assert.strictEqual(getTFLOPS(H100,  'fp4'),  null);  // Hopper: no FP4
});

// ═══════════════════════════════════════════════════════════
//  estimateParams
// ═══════════════════════════════════════════════════════════
//
//  Formula (dense):
//    emb  = 2 × vocab × hidden
//    attn = 4 × hidden²          (Q,K,V,O projections)
//    ffn  = 3 × hidden × ffn_dim (SwiGLU: gate, up, down)
//    total = (emb + layers × (attn + ffn)) / 1e9

test('estimateParams — Llama-3.1-8B-like dims produce exact param count', () => {
  // emb  = 2 × 128256 × 4096 = 1,050,673,152
  // attn = 4 × 4096²          =    67,108,864
  // ffn  = 3 × 4096 × 14336   =   176,160,768
  // total = 1,050,673,152 + 32 × 243,269,632 = 8,835,301,376
  assert.strictEqual(
    estimateParams({ layers: 32, hidden: 4096, ffn: 14336, vocab: 128256 }),
    8_835_301_376 / 1e9,
  );
});

test('estimateParams — Llama-3.1-70B-like dims produce exact param count', () => {
  // emb  = 2 × 128256 × 8192 = 2,101,346,304
  // attn = 4 × 8192²          =   268,435,456
  // ffn  = 3 × 8192 × 28672   =   704,643,072
  // total = 2,101,346,304 + 80 × 973,078,528 = 79,947,628,544
  assert.strictEqual(
    estimateParams({ layers: 80, hidden: 8192, ffn: 28672, vocab: 128256 }),
    79_947_628_544 / 1e9,
  );
});

test('estimateParams — returns null when any required field is missing', () => {
  assert.strictEqual(estimateParams({ layers: 32, hidden: 4096, ffn: 14336 }), null); // no vocab
  assert.strictEqual(estimateParams({ layers: 32, hidden: 4096, vocab: 32000 }),  null); // no ffn
  assert.strictEqual(estimateParams({}), null);
});

// ═══════════════════════════════════════════════════════════
//  calcVRAM
// ═══════════════════════════════════════════════════════════
//
//  Weight formula:  params × bytes/param
//  KV formula:      2 × layers × kvHeads × headDim × kvBytes × seqLen × batch / 1e9
//  Act formula:     hidden × seqLen × batch × 4 / 1e9
//  CUDA overhead:   1.0 GB (fixed)

test('calcVRAM — weight footprint: BF16 is exactly 2× INT4', () => {
  const bf16 = calcVRAM(MODEL_PARAMS, A100P, 'bf16', 1, 1, null);
  const int4 = calcVRAM(MODEL_PARAMS, A100P, 'int4', 1, 1, null);
  // BF16: 7B × 2 bytes = 14 GB  |  INT4: 7B × 0.5 bytes = 3.5 GB
  assert.strictEqual(bf16.weightsGB, 14);
  assert.strictEqual(int4.weightsGB, 3.5);
  assert.strictEqual(bf16.weightsGB, int4.weightsGB * 4);
});

test('calcVRAM — weight footprint scales linearly with param count', () => {
  const m7  = calcVRAM({ params: 7  }, A100P, 'bf16', 1, 1, null);
  const m14 = calcVRAM({ params: 14 }, A100P, 'bf16', 1, 1, null);
  assert.strictEqual(m7.weightsGB,  14);
  assert.strictEqual(m14.weightsGB, 28);
});

test('calcVRAM — multi-GPU multiplies totalVram but not weightsGB', () => {
  const g1 = calcVRAM(MODEL_PARAMS, A100P, 'bf16', 1, 1, null);
  const g2 = calcVRAM(MODEL_PARAMS, A100P, 'bf16', 2, 1, null);
  assert.strictEqual(g1.totalVram, 80);
  assert.strictEqual(g2.totalVram, 160);
  assert.strictEqual(g1.weightsGB, g2.weightsGB); // weights are the same either way
});

test('calcVRAM — KV cache scales exactly with batch size', () => {
  const b1 = calcVRAM(MODEL_FULL, A100P, 'bf16', 1, 1,  2048);
  const b2 = calcVRAM(MODEL_FULL, A100P, 'bf16', 1, 2,  2048);
  // headDim = 4096/32 = 128
  // kvPerToken = 2 × 32 × 8 × 128 × 2 = 131,072 bytes
  // kvGB(batch=1) = 131,072 × 2048 / 1e9 = 0.268435456
  assert.strictEqual(b1.kvGB, 0.268435456);
  assert.strictEqual(b2.kvGB, 0.536870912);
  assert.strictEqual(b2.kvGB, b1.kvGB * 2);
});

test('calcVRAM — KV cache scales exactly with sequence length', () => {
  const s2048 = calcVRAM(MODEL_FULL, A100P, 'bf16', 1, 1, 2048);
  const s4096 = calcVRAM(MODEL_FULL, A100P, 'bf16', 1, 1, 4096);
  assert.strictEqual(s2048.kvGB, 0.268435456);
  assert.strictEqual(s4096.kvGB, 0.536870912);
  assert.strictEqual(s4096.kvGB, s2048.kvGB * 2);
});

test('calcVRAM — KV cache uses max(1 byte, bytesPerParam) — INT4 floors at 1 byte', () => {
  const bf16 = calcVRAM(MODEL_FULL, A100P, 'bf16', 1, 1, 2048); // kvBytes=2
  const int4 = calcVRAM(MODEL_FULL, A100P, 'int4', 1, 1, 2048); // kvBytes=max(1,0.5)=1
  // BF16: 131,072 bytes/tok × 2048 / 1e9 = 0.268435456
  // INT4: 65,536 bytes/tok  × 2048 / 1e9 = 0.134217728
  assert.strictEqual(bf16.kvGB, 0.268435456);
  assert.strictEqual(int4.kvGB, 0.134217728);
  assert.strictEqual(bf16.kvGB, int4.kvGB * 2);
});

test('calcVRAM — CUDA overhead is always 1.0 GB', () => {
  const v = calcVRAM(MODEL_PARAMS, A100P, 'bf16', 1, 1, null);
  assert.strictEqual(v.cudaGB, 1.0);
});

test('calcVRAM — overflow detected when model exceeds VRAM', () => {
  // 79B @ BF16 = 158 GB weights + 1 GB CUDA = 159 GB > 80 GB
  const v = calcVRAM({ params: 79 }, A100P, 'bf16', 1, 1, null);
  assert.strictEqual(v.overflow, 79);
  assert.strictEqual(v.freeGB,   0);
});

test('calcVRAM — freeGB is positive when model comfortably fits', () => {
  // 7B @ BF16 = 14 GB + 1 GB = 15 GB used; 80 - 15 = 65 GB free
  const v = calcVRAM(MODEL_PARAMS, A100P, 'bf16', 1, 1, null);
  assert.strictEqual(v.freeGB,   65);
  assert.strictEqual(v.overflow, 0);
});

// ═══════════════════════════════════════════════════════════
//  calcThroughput
// ═══════════════════════════════════════════════════════════
//
//  Memory-bound TPS = (BW × N × 1e9) / modelBytes × batch × MFU(0.40)
//  Compute-bound TPS = (TFLOPS × N × 1e12 / (2×params×1e9)) × batch × MFU(0.45)
//  Ridge point (ops/byte) = TFLOPS×1e12 / (BW×1e9)

test('calcThroughput — returns null for unsupported precision', () => {
  assert.strictEqual(calcThroughput(MODEL_PARAMS, A100, 'fp8', 1, 1), null);
});

test('calcThroughput — ridge point is exactly TFLOPS/BW in ops/byte', () => {
  // A100 PCIe BF16: 312e12 / 1935e9 = 161.24031007751938
  const tp = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 1);
  assert.strictEqual(tp.ridgePoint, (312 * 1e12) / (1935 * 1e9));
});

test('calcThroughput — small batch is memory-bound', () => {
  // arithIntensity(batch=1, 7B BF16) = 1.0 < ridgePoint(≈161) → memory-bound
  const tp = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 1);
  assert.strictEqual(tp.isComputeBound, false);
});

test('calcThroughput — batch at ridge point crossover becomes compute-bound', () => {
  // ridgePoint ≈ 161.24 for A100 PCIe BF16
  // arithIntensity = batch (for 7B BF16, where modelBytes = 14B)
  // batch=161 → 161 < 161.24 → memory-bound
  // batch=162 → 162 ≥ 161.24 → compute-bound
  const below = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 161);
  const above = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 162);
  assert.strictEqual(below.isComputeBound, false);
  assert.strictEqual(above.isComputeBound, true);
});

test('calcThroughput — memory-bound: INT8 produces exact token/s values', () => {
  // BF16: rawTPS = 1935/14 = 138.2142857…  → Math.round(×0.40) = 55
  // INT8: rawTPS = 1935/7  = 276.4285714…  → Math.round(×0.40) = 111
  const bf16 = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 1);
  const int8 = calcThroughput(MODEL_PARAMS, A100P, 'int8', 1, 1);
  assert.strictEqual(bf16.tps, 55);
  assert.strictEqual(int8.tps, 111);
});

test('calcThroughput — memory-bound: 2× GPU count produces exact token/s values', () => {
  // 2-GPU BF16: bwGBs=3870, modelBytes=14B → rawTPS=3870/14=276.42857… → tps=111
  const g1 = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 1);
  const g2 = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 2, 1);
  assert.strictEqual(g1.tps, 55);
  assert.strictEqual(g2.tps, 111);
});

test('calcThroughput — compute-bound: exact token/s at crossover batch', () => {
  // batch=162, A100 PCIe BF16, 7B:
  // computeTimeSec = 14e9 / (312e12 / 162) → tokensPerSec = 312000/14 = 22285.714…
  // tps = Math.round(22285.714… × 0.45) = 10029
  const tp = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 162);
  assert.strictEqual(tp.isComputeBound, true);
  assert.strictEqual(tp.tps, 10029);
});

test('calcThroughput — ridgeBatch is exactly ceil(ridgePoint × bytesPerParam / 2)', () => {
  // A100 PCIe BF16: ridgePoint = 312e12/1935e9 ≈ 161.24, bytesPerParam = 2
  // ridgeBatch = ceil(161.24 × 2 / 2) = ceil(161.24) = 162
  const tp = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 1);
  assert.strictEqual(tp.ridgeBatch, Math.ceil(tp.ridgePoint * tp.bytesPerParam / 2));
  assert.strictEqual(tp.ridgeBatch, 162);
});

test('calcThroughput — ridgeBatch is the exact compute-bound crossover point', () => {
  // ridgeBatch must be compute-bound; ridgeBatch-1 must still be memory-bound
  const tp = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, 1);
  const atRidge  = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, tp.ridgeBatch);
  const belowRidge = calcThroughput(MODEL_PARAMS, A100P, 'bf16', 1, tp.ridgeBatch - 1);
  assert.strictEqual(atRidge.isComputeBound,    true);
  assert.strictEqual(belowRidge.isComputeBound, false);
});

test('calcThroughput — MoE active params drive throughput, not total params', () => {
  // MoE {params:235, activeParams:22} should equal dense {params:22}
  // because paramsB = activeParams ?? params
  const moe   = calcThroughput(MODEL_MOE,     A100P, 'bf16', 1, 1);
  const dense = calcThroughput(MODEL_DENSE22, A100P, 'bf16', 1, 1);
  assert.strictEqual(moe.tps, dense.tps);
  assert.strictEqual(moe.isComputeBound, dense.isComputeBound);
});

// ═══════════════════════════════════════════════════════════
//  calcMaxBatch
// ═══════════════════════════════════════════════════════════
//
//  freeGB = totalVram - weightsGB - 1.0 (CUDA)
//  perReqGB = kvCachePerReq + activationsPerReq
//  maxBatch = floor(freeGB / perReqGB)

test('calcMaxBatch — exact value for known model/GPU/seqLen', () => {
  // freeGB = 80 - 14 - 1 = 65
  // headDim=128; kvPerToken=2×32×8×128×2=131,072 bytes
  // perReqKV  = 131,072 × 2048 / 1e9 = 0.268435456
  // perReqAct = 4096 × 2048 × 4 / 1e9 = 0.033554432
  // perReqGB  = 0.301989888
  // maxBatch  = floor(65 / 0.301989888) = 215
  assert.strictEqual(calcMaxBatch(MODEL_FULL, A100P, 'bf16', 1, 2048), 215);
});

test('calcMaxBatch — doubling seqLen roughly halves maxBatch', () => {
  // seqLen=2048 → 215  |  seqLen=4096 → floor(65/0.603979776) = 107
  assert.strictEqual(calcMaxBatch(MODEL_FULL, A100P, 'bf16', 1, 2048), 215);
  assert.strictEqual(calcMaxBatch(MODEL_FULL, A100P, 'bf16', 1, 4096), 107);
});

test('calcMaxBatch — more GPUs means more free VRAM and a larger maxBatch', () => {
  // 2-GPU: freeGB = 160 - 14 - 1 = 145 → floor(145 / 0.301989888) = 480
  assert.strictEqual(calcMaxBatch(MODEL_FULL, A100P, 'bf16', 1, 2048), 215);
  assert.strictEqual(calcMaxBatch(MODEL_FULL, A100P, 'bf16', 2, 2048), 480);
});

test('calcMaxBatch — returns 1 when model does not fit in VRAM', () => {
  // 79B @ BF16 = 158 GB > 80 GB → freeGB <= 0 → maxBatch = 1
  assert.strictEqual(calcMaxBatch({ params: 79 }, A100P, 'bf16', 1, 2048), 1);
});

// ═══════════════════════════════════════════════════════════
//  inferParams
// ═══════════════════════════════════════════════════════════
//
//  Dense formula (same as estimateParams but keyed on HF config names):
//    emb  = 2 × vocab_size × hidden_size
//    attn = 4 × hidden_size²
//    ffn  = 3 × hidden_size × intermediate_size
//    total = emb + num_hidden_layers × (attn + ffn)
//
//  MoE formula:
//    ffnPerMoeLayer = (n_routed_experts + n_shared_experts) × 3 × hidden × moe_ffn
//    total = emb + moeLayers × (attn + ffnPerMoeLayer) + denseLayers × (attn + denseFFN)

test('inferParams — dense: Llama-3.1-8B HF config keys map correctly', () => {
  // emb  = 2 × 128256 × 4096 = 1,050,673,152
  // attn = 4 × 4096²          =    67,108,864
  // ffn  = 3 × 4096 × 14336   =   176,160,768
  // total = 1,050,673,152 + 32 × 243,269,632 = 8,835,301,376
  assert.strictEqual(
    inferParams({
      num_hidden_layers: 32, hidden_size: 4096,
      vocab_size: 128256, intermediate_size: 14336,
    }),
    8_835_301_376,
  );
});

test('inferParams — dense: returns null when a required field is missing', () => {
  assert.strictEqual(inferParams({ num_hidden_layers: 32, hidden_size: 4096, intermediate_size: 14336 }), null); // no vocab
  assert.strictEqual(inferParams({ num_hidden_layers: 32, hidden_size: 4096, vocab_size: 32000 }),          null); // no ffn (dense path)
});

test('inferParams — MoE: n_routed_experts + moe_intermediate_size (Mixtral/DeepSeek style)', () => {
  // emb  = 2 × 32000 × 4096 = 262,144,000
  // attn = 4 × 4096²         =  67,108,864
  // ffnPerMoeLayer = 8 × 3 × 4096 × 14336 = 1,409,286,144
  // moeLayers = ceil(32/1) = 32, denseLayers = 0
  // total = 262,144,000 + 32 × (67,108,864 + 1,409,286,144) = 47,506,784,256
  assert.strictEqual(
    inferParams({
      num_hidden_layers: 32, hidden_size: 4096,
      vocab_size: 32000,
      n_routed_experts: 8, moe_intermediate_size: 14336,
    }),
    47_506_784_256,
  );
});

test('inferParams — MoE: num_local_experts + intermediate_size fallback (gpt_oss style)', () => {
  // emb  = 2 × 201088 × 2880 = 1,158,266,880
  // attn = 4 × 2880²          =    33,177,600
  // ffnPerMoeLayer = 128 × 3 × 2880 × 2880 = 3,185,049,600
  // moeLayers = ceil(36/1) = 36, denseLayers = 0
  // total = 1,158,266,880 + 36 × (33,177,600 + 3,185,049,600) = 117,014,446,080
  assert.strictEqual(
    inferParams({
      num_hidden_layers: 36, hidden_size: 2880,
      vocab_size: 201088,
      num_local_experts: 128, intermediate_size: 2880,
    }),
    117_014_446_080,
  );
});

test('inferParams — MoE: safetensors.total shortcut takes priority over architecture', () => {
  // When the HF metadata includes a direct param count, use it as-is.
  assert.strictEqual(
    inferParams({ safetensors: { total: 7_000_000_000 }, num_hidden_layers: 32, hidden_size: 4096, vocab_size: 128256, intermediate_size: 14336 }),
    7_000_000_000,
  );
});

// ═══════════════════════════════════════════════════════════
//  inferActiveParams
// ═══════════════════════════════════════════════════════════

test('inferActiveParams — returns null for dense models (no expert keys)', () => {
  assert.strictEqual(
    inferActiveParams({ num_hidden_layers: 32, hidden_size: 4096, vocab_size: 128256, intermediate_size: 14336 }),
    null,
  );
});

test('inferActiveParams — returns null when active-expert count is missing', () => {
  // Has experts but no num_experts_per_tok / n_activated_experts / experts_per_token
  assert.strictEqual(
    inferActiveParams({ num_hidden_layers: 32, hidden_size: 4096, vocab_size: 32000, n_routed_experts: 8, moe_intermediate_size: 14336 }),
    null,
  );
});

test('inferActiveParams — num_local_experts + num_experts_per_tok (gpt_oss style)', () => {
  // emb  = 2 × 201088 × 2880 = 1,158,266,880
  // attn = 4 × 2880²          =    33,177,600
  // activeFFNPerMoeLayer = 4 × 3 × 2880 × 2880 = 99,532,800
  // total = (1,158,266,880 + 36 × (33,177,600 + 99,532,800)) / 1e9
  //       = 5,935,841,280 / 1e9 = 5.93584128
  assert.strictEqual(
    inferActiveParams({
      num_hidden_layers: 36, hidden_size: 2880,
      vocab_size: 201088,
      num_local_experts: 128, intermediate_size: 2880,
      num_experts_per_tok: 4,
    }),
    5_935_841_280 / 1e9,
  );
});

test('inferActiveParams — experts_per_token alias resolves correctly', () => {
  // Same result as above but using experts_per_token instead of num_experts_per_tok
  assert.strictEqual(
    inferActiveParams({
      num_hidden_layers: 36, hidden_size: 2880,
      vocab_size: 201088,
      num_local_experts: 128, intermediate_size: 2880,
      experts_per_token: 4,
    }),
    5_935_841_280 / 1e9,
  );
});

// ═══════════════════════════════════════════════════════════
//  Mixed-precision MoE (Phase 1A)
// ═══════════════════════════════════════════════════════════

test('calcVRAM — MoE mixed precision: INT4 expert + BF16 other < uniform BF16', () => {
  const bf16only = calcVRAM(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 1, 4096);
  const mixed    = calcVRAM(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 1, 4096, 'int4', 'bf16');
  assert.ok(mixed.weightsGB < bf16only.weightsGB);
});

test('calcVRAM — MoE mixed precision: returns non-null expertWeightsGB and otherWeightsGB', () => {
  const r = calcVRAM(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 1, 4096, 'int4', 'bf16');
  assert.ok(r.expertWeightsGB !== null);
  assert.ok(r.otherWeightsGB  !== null);
  assert.ok(r.expertWeightsGB > 0);
  assert.ok(r.otherWeightsGB  > 0);
});

test('calcThroughput — MoE mixed precision: INT4/BF16 split has higher TPS than uniform BF16', () => {
  const bf16only = calcThroughput(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 1);
  const mixed    = calcThroughput(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 1, 'int4', 'bf16');
  assert.ok(mixed.tps > bf16only.tps);
});

test('calcThroughput — dense model: single precision = two identical precisions (no regression)', () => {
  const single = calcThroughput(MODEL_FULL, A100P, 'bf16', 1, 1);
  const double = calcThroughput(MODEL_FULL, A100P, 'bf16', 1, 1, 'bf16', 'bf16');
  assert.strictEqual(single.tps, double.tps);
});

test('calcMaxBatch — MoE mixed precision: INT4 expert weights allow larger batch than BF16', () => {
  const bf16only = calcMaxBatch(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 256);
  const mixed    = calcMaxBatch(MODEL_MOE_SPLIT, A100P, 'bf16', 1, 256, 'int4', 'bf16');
  assert.ok(mixed >= bf16only);
});
