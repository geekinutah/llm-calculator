#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════
//  Calculator Validation Against Public Benchmarks
//  Run: node tests/validate.js
//
//  IMPORTANT: this harness intentionally imports the production math
//  engine and GPU database. Benchmark validation should exercise the
//  same code and hardware data that power the web UI.
// ═══════════════════════════════════════════════════════════

const fixtures = require('./fixtures.json').fixtures;
const { GPU_DB } = require('../static/gpus.js');
const { calcThroughput } = require('../static/math.js');

// Fixtures predate the production engine's `params` field name.
// Normalize only the schema boundary here; do not duplicate calculator math.
function normalizeModel(model) {
  return {
    ...model,
    params: model.params ?? model.paramsB,
  };
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

  const model = normalizeModel(f.model);
  const result = calcThroughput(model, gpu, f.precision, f.gpuCount, f.batchForCalc);
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
  higher effective bandwidth utilization at low batch, explaining why the
  calculator can under-predict latency-optimized single-stream benchmarks.

  At high batch (offline / saturation), batch size dominates the formula and
  MFU is less critical — this is where the calculator is most accurate.

  KNOWN SYSTEMATIC BIASES
  ─────────────────────────────────────────────────────────────────────────────
  1. Low batch (<4), premium engines: calc may under-predict substantially
     → MFU=0.40 is conservative vs highly optimized kernels

  2. Large models at high batch: calc can over-predict
     → KV cache memory traffic is not modeled in the throughput calculation
     → Scheduling overhead grows with model size

  3. Multi-GPU TP: calc can over-predict
     → Inter-GPU synchronization overhead is not modeled
     → Current production math assumes ideal TFLOPS and bandwidth scaling

  BOTTOM LINE
  ─────────────────────────────────────────────────────────────────────────────
  The calculator is best used for order-of-magnitude sizing and
  VRAM feasibility checks. Benchmark fixtures quantify where the simple
  roofline model diverges from real serving engines.
`);
