#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════
//  Calculator Validation Against Public Benchmarks
//  Run: node tests/validate.js
// ═══════════════════════════════════════════════════════════

const fixtures = require('./fixtures.json').fixtures;

// ── GPU DB (must match app.js) ────────────────────────────
const GPU_DB = {
  h100: { name: 'H100 SXM', vram: 80,  bw: 3350, fp16: 989,  fp32: 67,  bf16: 989,  fp8: 1979, int8: 1979, int4: 1979, fp4: null },
  h200: { name: 'H200 SXM', vram: 141, bw: 4800, fp16: 989,  fp32: 67,  bf16: 989,  fp8: 1979, int8: 1979, int4: 1979, fp4: null },
  a100: { name: 'A100 SXM', vram: 80,  bw: 2000, fp16: 312,  fp32: 19.5, bf16: 312, fp8: null, int8: 624,  int4: 624,  fp4: null },
  b200: { name: 'B200 SXM', vram: 192, bw: 8000, fp16: 2250, fp32: 90,  bf16: 2250, fp8: 4500, int8: 4500, int4: 4500, fp4: 9000 },
};

const PREC_META = {
  fp32: { bytes: 4  },
  bf16: { bytes: 2  },
  fp16: { bytes: 2  },
  fp8:  { bytes: 1  },
  int8: { bytes: 1  },
  int4: { bytes: 0.5 },
  fp4:  { bytes: 0.5 },
};

// ── Core calc (must stay in sync with app.js calcThroughput) ──
function calcThroughput(model, gpu, precision, gpuCount, batch) {
  const tflopsRaw = getTFLOPS(gpu, precision);
  if (tflopsRaw === null) return null;
  const tflops = tflopsRaw * gpuCount;
  const bwGBs  = gpu.bw * gpuCount;

  const paramsB = model.activeParams ?? model.paramsB;
  const bytesPerParam = PREC_META[precision].bytes;
  const modelBytes = paramsB * 1e9 * bytesPerParam;

  const ridgePoint     = (tflops * 1e12) / (bwGBs * 1e9);
  const arithIntensity = (2 * batch * paramsB * 1e9) / modelBytes;
  const isComputeBound = arithIntensity >= ridgePoint;

  let rawTPS;
  if (isComputeBound) {
    const flopsPerToken  = 2 * paramsB * 1e9;
    const computeTimeSec = flopsPerToken / (tflops * 1e12 / batch);
    rawTPS = batch / computeTimeSec;
  } else {
    rawTPS = (bwGBs * 1e9) / modelBytes * batch;
  }

  const mfu = isComputeBound ? 0.45 : 0.40;
  return {
    tps: Math.round(rawTPS * mfu),
    isComputeBound,
    ridgePoint,
    arithIntensity,
    mfu,
  };
}

function getTFLOPS(gpu, prec) {
  const map = { fp32: gpu.fp32, bf16: gpu.bf16, fp16: gpu.fp16,
                fp8: gpu.fp8, int8: gpu.int8, int4: gpu.int4, fp4: gpu.fp4 };
  return map[prec] ?? null;
}

// ── Run validation ────────────────────────────────────────
const COL = { id: 42, bench: 9, calc: 9, ratio: 8, regime: 14, note: 0 };
const line = (s, w) => String(s).padEnd(w ?? 0);

console.log('\n' + '═'.repeat(100));
console.log(' LLM Calculator — Validation Against Public Benchmarks');
console.log('═'.repeat(100));
console.log(
  line('Fixture', COL.id) +
  line('Bench TPS', COL.bench) +
  line('Calc TPS', COL.calc) +
  line('Ratio', COL.ratio) +
  line('Regime', COL.regime) +
  'Notes'
);
console.log('─'.repeat(100));

const results = [];

for (const f of fixtures) {
  const gpu = GPU_DB[f.gpu];
  if (!gpu) { console.log(`  SKIP ${f.id} — unknown GPU "${f.gpu}"`); continue; }

  const result = calcThroughput(f.model, gpu, f.precision, f.gpuCount, f.batchForCalc);
  if (!result) { console.log(`  SKIP ${f.id} — precision not supported`); continue; }

  const ratio = result.tps / f.benchmarkTPS;
  const ratioStr = ratio.toFixed(2) + '×';
  const regime = result.isComputeBound ? 'compute-bound' : 'memory-bound';
  const flag = ratio < 0.6 ? '⚠ LOW' : ratio > 2.0 ? '⚠ HIGH' : ratio > 1.4 ? '△' : '✓';

  console.log(
    line(f.id, COL.id) +
    line(f.benchmarkTPS.toLocaleString(), COL.bench) +
    line(result.tps.toLocaleString(), COL.calc) +
    line(ratioStr, COL.ratio) +
    line(regime, COL.regime) +
    `${flag}  ${f.engine} | batch=${f.batchForCalc} | ${f.precision.toUpperCase()}`
  );

  results.push({ ...f, calcTPS: result.tps, ratio, result });
}

console.log('─'.repeat(100));
console.log('\nRatio = calc ÷ benchmark  (1.0 = perfect,  >1 = over-estimate,  <1 = under-estimate)\n');

// ── Interpretation ────────────────────────────────────────
console.log('═'.repeat(100));
console.log(' Interpretation');
console.log('═'.repeat(100));

const overPredicted = results.filter(r => r.ratio > 1.4);
const underPredicted = results.filter(r => r.ratio < 0.7);
const good = results.filter(r => r.ratio >= 0.7 && r.ratio <= 1.4);

console.log(`\n  ✓ Within ±40%: ${good.length}/${results.length} fixtures`);
console.log(`  △ Over-predicted (>1.4×): ${overPredicted.length}/${results.length}`);
if (overPredicted.length) overPredicted.forEach(r => console.log(`      - ${r.id}: ${r.ratio.toFixed(2)}×`));
console.log(`  ⚠ Under-predicted (<0.7×): ${underPredicted.length}/${results.length}`);
if (underPredicted.length) underPredicted.forEach(r => console.log(`      - ${r.id}: ${r.ratio.toFixed(2)}×`));

console.log(`
  MFU CALIBRATION NOTE
  ─────────────────────────────────────────────────────────────────────────────
  The calculator uses MFU=0.40 (memory-bound) and MFU=0.45 (compute-bound).
  These are conservative production baselines (vLLM, LMDeploy, moderate opt).

  Specialized engines (TRT-LLM with CUDA graphs, SGLang w/ FlashInfer) achieve
  0.65–0.80 effective bandwidth utilization at low batch, explaining why the
  calculator under-predicts latency-optimized single-stream benchmarks.

  At high batch (offline / saturation), batch size dominates the formula and
  MFU is less critical — this is where the calculator is most accurate.

  KNOWN SYSTEMATIC BIASES
  ─────────────────────────────────────────────────────────────────────────────
  1. Low batch (<4), premium engines: calc under-predicts by ~2×
     → MFU=0.40 is conservative vs TRT-LLM CUDA-graph-optimized kernels

  2. Large models at high batch (70B, batch=64): calc over-predicts by ~2×
     → KV cache memory traffic not modeled in bandwidth calculation
     → Scheduling overhead grows with model size

  3. Multi-GPU TP (8× H100): calc over-predicts by ~1.6×
     → NVLink synchronization overhead not modeled
     → Real-world TP efficiency ~80-85%, not 100%

  BOTTOM LINE
  ─────────────────────────────────────────────────────────────────────────────
  The calculator is best used for order-of-magnitude sizing and
  VRAM feasibility checks. For production capacity planning, apply:
    · Small model (<13B), optimized engine:  multiply calc TPS by ~2×
    · Large model (70B+), production batch:  multiply calc TPS by ~0.6×
    · Multi-GPU TP:                          multiply calc TPS by ~0.8×
`);
