// ═══════════════════════════════════════════════════════════
//  GPU DATA  (browser: loaded from gpus.js <script> tag)
//            (Node.js: require'd here so app.js is self-contained)
// ═══════════════════════════════════════════════════════════
if (typeof module !== 'undefined') {
  // In Node.js: load GPU_DB and PREC_META from gpus.js.
  // Using global assignment (not var) to avoid var-hoisting conflicting
  // with the const declarations in gpus.js when loaded as a browser <script>.
  const _gpus = require('./gpus.js');
  global.GPU_DB = _gpus.GPU_DB;
  global.PREC_META = _gpus.PREC_META;
}

// ═══════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════
let state = {
  gpu: null,
  gpuCount: 1,
  precision: 'bf16',
  model: {
    params: null, layers: null, hidden: null,
    ffn: null, heads: null, kvHeads: null,
    context: null, vocab: null,
  },
  batch: { batch: null, output: null },
  // animated gauge value
  gaugeTarget: 0,
  gaugeCurrent: 0,
  gaugeRaf: null,
};

// ═══════════════════════════════════════════════════════════
//  GPU CHANGE
// ═══════════════════════════════════════════════════════════
function onGpuChange() {
  const v = document.getElementById('gpuSelect').value;
  const customFields = document.getElementById('customGpuFields');

  if (v === 'custom') {
    customFields.classList.add('show');
    state.gpu = buildCustomGpu();
  } else if (v && GPU_DB[v]) {
    customFields.classList.remove('show');
    state.gpu = { ...GPU_DB[v] };
  } else {
    customFields.classList.remove('show');
    state.gpu = null;
  }
  
  updatePcieWarning();
  updatePrecisionPills();
  recalculate();
}

function buildCustomGpu() {
  return {
    name: 'Custom GPU',
    vram: +document.getElementById('cVram').value || null,
    bw: +document.getElementById('cBw').value || null,
    bf16: +document.getElementById('cBf16').value || null,
    fp16: +document.getElementById('cBf16').value || null,
    fp32: null,
    fp8: +document.getElementById('cFp8').value || null,
    fp4: +document.getElementById('cFp4').value || null,
    int8: +document.getElementById('cInt8').value || null,
    int4: +document.getElementById('cInt8').value || null,
  };
}

function stepGpu(d) {
  state.gpuCount = Math.max(1, Math.min(16, state.gpuCount + d));
  document.getElementById('gpuCountVal').textContent = state.gpuCount;
  updatePcieWarning();
  recalculate();
}

function updatePcieWarning() {
  const noNvlink = state.gpu?.interconnect?.nvlink_gbps == null;
  const show = !!state.gpu && noNvlink && state.gpuCount > 1;
  document.getElementById('pcieDisclaimer').classList.toggle('show', show);
}

// ═══════════════════════════════════════════════════════════
//  PRECISION
// ═══════════════════════════════════════════════════════════
function setPrecision(p) {
  const pill = document.querySelector(`[data-prec="${p}"]`);
  if (pill.classList.contains('disabled')) return;
  state.precision = p;
  document.querySelectorAll('.pill').forEach(el => el.classList.remove('active'));
  pill.classList.add('active');
  recalculate();
}

function updatePrecisionPills() {
  const gpu = state.gpu;
  const noteEl = document.getElementById('precNote');
  document.querySelectorAll('.pill').forEach(pill => {
    const p = pill.dataset.prec;
    const supported = !gpu || getTFLOPS(gpu, p) !== null;
    pill.classList.toggle('disabled', !supported);
  });

  // If current precision became unsupported, fall back
  if (gpu && getTFLOPS(gpu, state.precision) === null) {
    const fallback = ['bf16', 'fp16', 'int8', 'fp32'].find(p => getTFLOPS(gpu, p) !== null) || 'bf16';
    setPrecision(fallback);
  }

  // INT4 note
  if (state.precision === 'int4') {
    noteEl.className = 'fetch-status';
    noteEl.textContent = '⚠ INT4 dequantizes to INT8/FP16 at runtime — no additional TFLOPS gain over INT8.';
    noteEl.style.color = 'var(--yellow)';
  } else {
    noteEl.textContent = '';
  }
}



// ═══════════════════════════════════════════════════════════
//  HUGGINGFACE FETCH
// ═══════════════════════════════════════════════════════════




// Active params for MoE: only the experts that fire per token contribute to
// memory bandwidth and FLOPs. Returns null for dense models (active = total).


function setFieldVal(id, val) {
  const el = document.getElementById(id);
  if (val != null && val !== '') {
    el.value = val;
  } else {
    el.value = '';
  }
}

// ═══════════════════════════════════════════════════════════
//  CALCULATIONS
// ═══════════════════════════════════════════════════════════
function readModel() {
  return {
    params: +document.getElementById('mParams').value || null,
    activeParams: +document.getElementById('mActiveParams').value || null,
    layers: +document.getElementById('mLayers').value || null,
    hidden: +document.getElementById('mHidden').value || null,
    ffn: +document.getElementById('mFFN').value || null,
    heads: +document.getElementById('mHeads').value || null,
    kvHeads: +document.getElementById('mKVHeads').value || null,
    context: +document.getElementById('mContext').value || null,
    vocab: +document.getElementById('mVocab').value || null,
  };
}

function readBatch() {
  return {
    batch: +document.getElementById('bBatch').value || 32,
    output: +document.getElementById('bOutput').value || 2048,
  };
}







// ═══════════════════════════════════════════════════════════
//  SCENARIOS
// ═══════════════════════════════════════════════════════════


function renderScenarios(model, gpu, precision, gpuCount, seqLen) {
  const panel = document.getElementById('scenariosPanel');
  const tp = calcThroughput(model, gpu, precision, gpuCount, 1);
  if (!tp) {
    panel.className = 'scenarios-empty';
    panel.innerHTML = '— configure GPU &amp; model —';
    return;
  }

  const maxBatch = calcMaxBatch(model, gpu, precision, gpuCount, seqLen);
  const tpMax = calcThroughput(model, gpu, precision, gpuCount, maxBatch);
  const tpMid = calcThroughput(model, gpu, precision, gpuCount, 32);

  const rows = [
    {
      label: 'Min Latency',
      sub: 'batch 1 · lowest response time',
      tps: tp.tps,
      cls: 'best',
      tip: 'Batch=1: one user, no queuing. The GPU streams all weights once per token — pure memory-bandwidth limit.\n\nFormula: BW × N_GPUs / model_bytes × MFU\n\nThis is the fastest a single user can receive tokens. Latency is minimized; GPU utilization is low.',
    },
    {
      label: 'Balanced',
      sub: 'batch 32 · production baseline',
      tps: tpMid?.tps ?? 0,
      cls: '',
      tip: 'Batch=32: a reasonable continuous-batching baseline for a production API.\n\nThroughput scales linearly with batch while memory-bound. Each user waits slightly longer, but the system serves 32× more tokens per second than batch=1.\n\nTypical starting point for vLLM / TRT-LLM deployments.',
    },
    {
      label: 'Max Throughput',
      sub: `batch ${maxBatch} · VRAM ceiling`,
      tps: tpMax?.tps ?? 0,
      cls: 'worst',
      tip: `Batch=${maxBatch}: the largest batch that fits in VRAM given your seq length.\n\nFree VRAM after weights is divided by (KV cache + activations) per request at the configured seq length.\n\nHighest total tokens/sec for the system, but each user's response latency is highest. Beyond this batch size the model OOMs.`,
    },
  ];

  panel.className = '';
  panel.innerHTML = rows.map(r => `
    <div class="scenario-row">
      <span class="scenario-name">${r.label} <i class="tip" data-tip="${r.tip}"></i></span>
      <span class="scenario-sub">${r.sub}</span>
      <span class="scenario-tps ${r.cls}">${r.tps.toLocaleString()} <span class="scenario-unit">tok/s</span></span>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════
//  MAIN RECALCULATE
// ═══════════════════════════════════════════════════════════
function recalculate() {
  if (document.getElementById('gpuSelect').value === 'custom') {
    state.gpu = buildCustomGpu();
  }

  const model = readModel();
  const batch = readBatch();
  const gpu = state.gpu;
  const prec = state.precision;
  const N = state.gpuCount;

  const hasGPU = !!gpu;
  const hasModel = !!(model.params || (model.layers && model.hidden && model.ffn));

  renderScenarios(model, gpu, prec, N, batch.output);

  if (!hasGPU || !hasModel) {
    document.getElementById('emptyState').style.display = 'flex';
    document.getElementById('outputPanel').querySelectorAll('.results-block').forEach(e => e.remove());
    return;
  }

  document.getElementById('emptyState').style.display = 'none';

  const vram = calcVRAM(model, gpu, prec, N, batch.batch, batch.output);
  const throughput = calcThroughput(model, gpu, prec, N, batch.batch);

  renderResults(model, gpu, prec, N, batch, vram, throughput);
  serializeState();
}

// ═══════════════════════════════════════════════════════════
//  RENDER
// ═══════════════════════════════════════════════════════════
function renderResults(model, gpu, prec, N, batch, vram, tp) {
  const panel = document.getElementById('outputPanel');

  // Remove old results
  panel.querySelectorAll('.results-block').forEach(e => e.remove());

  // ── Hero metrics ──
  const heroEl = mkEl('div', 'results-block');
  const tpsVal = tp ? tp.tps.toLocaleString() : '—';
  const tpsSub = tp ? `${PREC_META[prec].label} · batch ${batch.batch} · ${N > 1 ? N + '× GPU TP' : '1 GPU'}` : '';
  const utilVal = tp ? tp.flopsUtil.toFixed(1) + '%' : '—';
  const modelGB = vram.weightsGB ? vram.weightsGB.toFixed(1) : '—';

  // Cost card — uses Balanced (batch=32) TPS as the pricing anchor
  const costPerHr = (+document.getElementById('gpuCostPerHr').value || 2.50) * N;
  const tpBalanced = calcThroughput(model, gpu, prec, N, 32);
  let costVal = '—', costSub = '';
  if (tpBalanced) {
    const cPerMTok = (costPerHr / (tpBalanced.tps * 3600)) * 1e6;
    costVal = cPerMTok < 1
      ? '$' + cPerMTok.toFixed(3)
      : '$' + cPerMTok.toFixed(2);
    costSub = `per M output tokens · batch 32 · $${costPerHr.toFixed(2)}/hr`;
  }

  heroEl.innerHTML = `
    <div class="hero">
      <div class="metric-card primary">
        <div class="metric-label">Tokens / Second <i class="tip" data-tip="Roofline minimum of memory-bound and compute-bound limits, with MFU applied.\n\nMemory-bound: (BW × N_GPUs) / model_bytes × batch\nCompute-bound: (TFLOPS × N_GPUs × MFU) / (2 × params / batch)\n\nDecode is almost always memory-bound. Large prefill batches can cross into compute-bound."></i></div>
        <div class="metric-value" id="tpsVal">${tpsVal}</div>
        <div class="metric-sub">${tpsSub}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">FLOPS Utilization <i class="tip" data-tip="How much of peak TFLOPS the workload actually uses.\n\nActual TFLOPS = 2 × params × TPS ÷ 1e12\nPeak TFLOPS = GPU peak × N_GPUs × MFU\n\nLow % is normal for memory-bound decode — the GPU sits idle waiting for weights to stream in. High % indicates compute-bound prefill."></i></div>
        <div class="metric-value">${utilVal}</div>
        <div class="metric-sub">of ${tp ? tp.peakTFLOPS.toFixed(0) : '—'} TFLOPS peak</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Model Footprint <i class="tip" data-tip="Weight-only VRAM: params × bytes_per_param.\n\nDoes not include KV cache, activations, or CUDA overhead — those are shown in the VRAM breakdown below.\n\nFormula: params_B × 1e9 × bytes/param ÷ 1e9"></i></div>
        <div class="metric-value">${modelGB} <span style="font-size:1rem;color:var(--text-dim)">GB</span></div>
        <div class="metric-sub">of ${vram.totalVram} GB total VRAM</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Output Token Cost <i class="tip" data-tip="Cost per million output tokens at Balanced throughput (batch=32).\n\nFormula: (GPU $/hr × GPU count) ÷ (balanced_TPS × 3600) × 1,000,000\n\nOutput tokens dominate inference cost — they are generated sequentially, one at a time. Input (prefill) tokens are ~5–10× cheaper to process and are not included here.\n\nAdjust GPU Cost ($/hr) in the GPU Configuration section."></i></div>
        <div class="metric-value">${costVal}</div>
        <div class="metric-sub">${costSub}</div>
      </div>
    </div>
  `;
  panel.appendChild(heroEl);

  // ── Bottleneck badge ──
  if (tp) {
    const badgeEl = mkEl('div', 'results-block bottleneck-row');
    const cls = tp.isComputeBound ? 'compute' : 'memory';
    const icon = tp.isComputeBound ? '⚡' : '💾';
    const label = tp.isComputeBound ? 'Compute-Bound' : 'Memory-Bound';
    const desc = tp.isComputeBound
      ? `Arithmetic intensity (${tp.arithIntensity.toFixed(0)} ops/B) exceeds ridge point (${tp.ridgePoint.toFixed(0)} ops/B). TFLOPS is the limiting factor.`
      : `Arithmetic intensity (${tp.arithIntensity.toFixed(0)} ops/B) is below ridge point (${tp.ridgePoint.toFixed(0)} ops/B). Memory bandwidth is the limiting factor.`;
    badgeEl.innerHTML = `
      <div class="bottleneck-badge ${cls}">${icon} ${label}</div>
      <div class="bottleneck-desc">${desc}</div>
    `;
    panel.appendChild(badgeEl);
  }

  // ── FLOPS Gauge ──
  const gaugeEl = mkEl('div', 'results-block viz-card');
  const peakTF = tp ? tp.peakTFLOPS : (gpu.bf16 ?? 0) * N;
  const usedTF = tp ? (tp.tps * tp.flopsPerToken / 1e12) : 0;
  gaugeEl.innerHTML = `
    <div class="viz-title">FLOPS Utilization — Roofline</div>
    <div class="gauge-wrap">
      <div class="gauge-svg-wrap">
        <canvas id="gaugeCanvas" width="220" height="130"></canvas>
        <div class="gauge-center-text">
          <span class="gauge-pct" id="gaugePctText">0%</span>
          <span class="gauge-sub-label">util</span>
        </div>
      </div>
      <div class="gauge-legend">
        <div class="gauge-legend-row">
          <span class="gauge-legend-label">Peak TFLOPS</span>
          <span class="gauge-legend-val">${peakTF.toFixed(0)} TF</span>
        </div>
        <div class="gauge-legend-row">
          <span class="gauge-legend-label">Used TFLOPS</span>
          <span class="gauge-legend-val">${usedTF.toFixed(1)} TF</span>
        </div>
        <div class="gauge-legend-row">
          <span class="gauge-legend-label">MFU Applied</span>
          <span class="gauge-legend-val">${tp ? (tp.mfu * 100).toFixed(0) : '—'}%</span>
        </div>
        <div class="gauge-legend-divider"></div>
        <div class="gauge-legend-row">
          <span class="gauge-legend-label">Precision</span>
          <span class="gauge-legend-val">${PREC_META[prec].label}</span>
        </div>
        <div class="gauge-legend-row">
          <span class="gauge-legend-label">Bytes/param</span>
          <span class="gauge-legend-val">${PREC_META[prec].bytes}</span>
        </div>
        <div class="gauge-legend-divider"></div>
        <div class="mfu-note">
          Theoretical peak is rarely achieved.<br>
          Typical MFU: 35–55% on optimized<br>
          inference engines (vLLM, TRT-LLM).
        </div>
      </div>
    </div>
  `;
  panel.appendChild(gaugeEl);

  // Animate gauge
  const targetPct = tp ? tp.flopsUtil : 0;
  animateGauge(targetPct);

  // ── VRAM Bar ──
  const vramEl = mkEl('div', 'results-block viz-card');
  vramEl.innerHTML = buildVRAMHTML(vram);
  panel.appendChild(vramEl);
  // Animate segments: start at 0%, then set target widths after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      vramEl.querySelectorAll('.vram-seg[data-pct]').forEach(seg => {
        seg.style.width = seg.dataset.pct + '%';
      });
    });
  });

  // ── Breakdown Table ──
  if (tp) {
    const bkEl = mkEl('div', 'results-block viz-card');
    bkEl.innerHTML = buildBreakdownHTML(tp, model);
    panel.appendChild(bkEl);
  }
}

function buildVRAMHTML(v) {
  const total = v.totalVram;
  const segs = [
    { key: 'weights', gb: v.weightsGB, label: 'Weights', cls: 'vram-seg-weights' },
    { key: 'kv', gb: v.kvGB, label: 'KV Cache', cls: 'vram-seg-kv' },
    { key: 'activations', gb: v.actGB, label: 'Activations', cls: 'vram-seg-activations' },
    { key: 'cuda', gb: v.cudaGB, label: 'CUDA/FW', cls: 'vram-seg-cuda' },
  ].filter(s => s.gb && s.gb > 0);

  const segHTML = segs.map(s => {
    const pct = Math.min(100, (s.gb / total) * 100);
    return `<div class="vram-seg ${s.cls}" style="width:0%" data-pct="${pct}"></div>`;
  }).join('');

  const overflowPct = v.overflow > 0 ? Math.min(100, (v.overflow / total) * 100) : 0;

  // Legend entries
  const dotColors = ['var(--c-weights)', 'var(--c-kvcache)', 'var(--c-activations)', 'var(--c-cuda)', 'var(--c-free)'];
  const legItems = [
    ...segs.map((s, i) => ({ label: s.label, gb: s.gb, color: dotColors[i] })),
    { label: 'Free', gb: v.freeGB > 0 ? v.freeGB : 0, color: 'var(--c-free)', border: '1px solid var(--border2)' },
  ];

  const legHTML = legItems.map(l => `
    <div class="vleg">
      <div class="vleg-dot" style="background:${l.color};${l.border ? 'border:' + l.border : ''}"></div>
      <div class="vleg-name">${l.label}</div>
      <div class="vleg-val">${l.gb.toFixed(1)} GB</div>
      <div class="vleg-pct">${((l.gb / v.totalVram) * 100).toFixed(1)}%</div>
    </div>
  `).join('');

  return `
    <div class="viz-title">VRAM Allocation — ${v.totalVram} GB total</div>
    <div class="vram-bar-wrap">
      <div class="vram-bar">${segHTML}</div>
      <div class="vram-overflow ${v.overflow > 0 ? 'show' : ''}">
        ⚠ OVERFLOW +${v.overflow.toFixed(1)} GB — model does not fit
        <div class="vram-overflow-bar"><div class="vram-overflow-fill" style="width:${overflowPct}%"></div></div>
      </div>
    </div>
    <div class="vram-legend">${legHTML}</div>
  `;
}

function buildBreakdownHTML(tp, model) {
  const isMoE = model.activeParams && model.params && model.activeParams < model.params;
  const rows = [
    ...(isMoE ? [
      ['Total Params', model.params.toFixed(2) + ' B'],
      ['Active Params', model.activeParams.toFixed(2) + ' B  ← MoE: drives BW & FLOPs'],
    ] : []),
    ['Mem Bandwidth', (tp.bwGBs / 1000).toFixed(2) + ' TB/s'],
    ['Ridge Point', tp.ridgePoint.toFixed(0) + ' ops/byte'],
    ['Arith. Intensity', tp.arithIntensity.toFixed(0) + ' ops/byte'],
  ];

  const rowsHTML = rows.map(([l, v]) => `
    <div class="brow">
      <span class="brow-label">${l}</span>
      <span class="brow-val">${v}</span>
    </div>
  `).join('');

  return `
    <div class="viz-title">Parameter Breakdown</div>
    <div class="breakdown-grid">${rowsHTML}</div>
  `;
}

// ═══════════════════════════════════════════════════════════
//  GAUGE CANVAS
// ═══════════════════════════════════════════════════════════
function animateGauge(targetPct) {
  if (state.gaugeRaf) cancelAnimationFrame(state.gaugeRaf);
  state.gaugeTarget = targetPct;

  function step() {
    const diff = state.gaugeTarget - state.gaugeCurrent;
    if (Math.abs(diff) < 0.1) {
      state.gaugeCurrent = state.gaugeTarget;
      drawGauge(state.gaugeCurrent);
      return;
    }
    state.gaugeCurrent += diff * 0.08;
    drawGauge(state.gaugeCurrent);
    state.gaugeRaf = requestAnimationFrame(step);
  }
  step();
}

function drawGauge(pct) {
  const canvas = document.getElementById('gaugeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 18;
  const r = 88;
  const startAngle = Math.PI;
  const endAngle = 2 * Math.PI;
  const fraction = Math.min(100, Math.max(0, pct)) / 100;

  ctx.clearRect(0, 0, W, H);

  // Track background
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = '#1e2530';
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  // "Realistic max" zone marker at 55%
  const maxAngle = startAngle + 0.55 * Math.PI;
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, maxAngle);
  ctx.strokeStyle = 'rgba(255,215,64,0.12)';
  ctx.lineWidth = 14;
  ctx.lineCap = 'butt';
  ctx.stroke();

  // Realistic max tick
  const tickAngle = startAngle + 0.55 * Math.PI;
  const tx1 = cx + (r - 10) * Math.cos(tickAngle);
  const ty1 = cy + (r - 10) * Math.sin(tickAngle);
  const tx2 = cx + (r + 4) * Math.cos(tickAngle);
  const ty2 = cy + (r + 4) * Math.sin(tickAngle);
  ctx.beginPath();
  ctx.moveTo(tx1, ty1);
  ctx.lineTo(tx2, ty2);
  ctx.strokeStyle = 'rgba(255,215,64,0.5)';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Label for realistic max
  const lx = cx + (r + 16) * Math.cos(tickAngle);
  const ly = cy + (r + 16) * Math.sin(tickAngle);
  ctx.fillStyle = 'rgba(255,215,64,0.5)';
  ctx.font = '500 9px "IBM Plex Mono"';
  ctx.textAlign = 'center';
  ctx.fillText('55%', lx, ly);

  // Value arc
  if (fraction > 0) {
    const valAngle = startAngle + fraction * Math.PI;
    const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
    grad.addColorStop(0, '#00e676');
    grad.addColorStop(0.5, '#ffd740');
    grad.addColorStop(1, '#ff5252');

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, valAngle);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  // Tick marks at 0, 25, 50, 75, 100%
  [0, 0.25, 0.5, 0.75, 1.0].forEach(t => {
    const a = startAngle + t * Math.PI;
    const ix = cx + (r - 17) * Math.cos(a);
    const iy = cy + (r - 17) * Math.sin(a);
    const ox = cx + (r - 5) * Math.cos(a);
    const oy = cy + (r - 5) * Math.sin(a);
    ctx.beginPath();
    ctx.moveTo(ix, iy);
    ctx.lineTo(ox, oy);
    ctx.strokeStyle = '#2e3540';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Labels
    const lbx = cx + (r - 28) * Math.cos(a);
    const lby = cy + (r - 28) * Math.sin(a);
    ctx.fillStyle = '#5a6475';
    ctx.font = '400 8px "IBM Plex Mono"';
    ctx.textAlign = 'center';
    ctx.fillText((t * 100) + '%', lbx, lby);
  });

  // Center pct text
  const pctEl = document.getElementById('gaugePctText');
  if (pctEl) pctEl.textContent = pct.toFixed(1) + '%';
}

// ═══════════════════════════════════════════════════════════
//  URL STATE — serialize / deserialize
// ═══════════════════════════════════════════════════════════
function serializeState() {
  if (typeof document === 'undefined') return;
  const p = new URLSearchParams();
  const gpuVal = document.getElementById('gpuSelect').value;

  if (gpuVal) p.set('g', gpuVal);
  if (state.gpuCount !== 1) p.set('n', state.gpuCount);
  if (state.precision !== 'bf16') p.set('p', state.precision);

  const cost = document.getElementById('gpuCostPerHr').value;
  if (cost) p.set('cost', cost);

  const fields = [
    ['mp', 'mParams'], ['ml', 'mLayers'], ['mh', 'mHidden'], ['mf', 'mFFN'],
    ['ma', 'mHeads'], ['mk', 'mKVHeads'], ['mc', 'mContext'], ['mv', 'mVocab'],
    ['map', 'mActiveParams'], ['bb', 'bBatch'], ['bs', 'bOutput'],
  ];
  for (const [key, id] of fields) {
    const v = document.getElementById(id).value;
    if (v) p.set(key, v);
  }

  const hf = document.getElementById('hfModelId').value.trim();
  if (hf) p.set('hf', hf);

  const qs = p.toString();
  history.replaceState(null, '', qs ? '?' + qs : location.pathname);
}

function deserializeState() {
  const p = new URLSearchParams(location.search);
  if (!p.size) return;

  const gpuVal = p.get('g');
  if (gpuVal) document.getElementById('gpuSelect').value = gpuVal;

  const n = parseInt(p.get('n'));
  if (n >= 1 && n <= 16) {
    state.gpuCount = n;
    document.getElementById('gpuCountVal').textContent = n;
  }

  // Set precision on state before onGpuChange so pill fallback logic has context
  const prec = p.get('p');
  if (prec) state.precision = prec;

  const cost = p.get('cost');
  if (cost) document.getElementById('gpuCostPerHr').value = cost;

  const fields = [
    ['mp', 'mParams'], ['ml', 'mLayers'], ['mh', 'mHidden'], ['mf', 'mFFN'],
    ['ma', 'mHeads'], ['mk', 'mKVHeads'], ['mc', 'mContext'], ['mv', 'mVocab'],
    ['map', 'mActiveParams'], ['bb', 'bBatch'], ['bs', 'bOutput'],
  ];
  for (const [key, id] of fields) {
    const v = p.get(key);
    if (v) document.getElementById(id).value = v;
  }

  const hf = p.get('hf');
  if (hf) document.getElementById('hfModelId').value = hf;

  // onGpuChange wires up state.gpu, precision pills, pcie warning, and calls recalculate()
  onGpuChange();

  // Re-apply precision after onGpuChange (it may have fallen back to a supported default)
  if (prec) setPrecision(prec);
}

function copyShareLink() {
  navigator.clipboard.writeText(location.href).then(() => {
    const btn = document.getElementById('shareBtn');
    const prev = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = prev; }, 1500);
  });
}

// ═══════════════════════════════════════════════════════════
//  GPU DROPDOWN RENDERER
// ═══════════════════════════════════════════════════════════
function renderGpuDropdown() {
  const select = document.getElementById('gpuSelect');

  // Build map: arch → { order, entries: [[id, gpu], ...] }
  const archMap = {};
  for (const [id, gpu] of Object.entries(GPU_DB)) {
    const arch = gpu.arch;
    if (!archMap[arch]) archMap[arch] = { order: gpu.archOrder ?? 99, entries: [] };
    archMap[arch].entries.push([id, gpu]);
  }

  // Sort archs by archOrder, GPUs within each arch by VRAM descending
  const sortedArchs = Object.entries(archMap)
    .sort(([, a], [, b]) => a.order - b.order);
  for (const [, group] of sortedArchs) {
    group.entries.sort(([, a], [, b]) => b.vram - a.vram);
  }

  // Remove existing optgroups; leave placeholder ('') and 'custom' options intact
  select.querySelectorAll('optgroup').forEach(g => g.remove());
  select.querySelectorAll('option:not([value=""]):not([value="custom"])').forEach(o => o.remove());

  const customOption = select.querySelector('option[value="custom"]');
  for (const [arch, group] of sortedArchs) {
    const optgroup = document.createElement('optgroup');
    optgroup.label = arch.charAt(0).toUpperCase() + arch.slice(1);
    for (const [id, gpu] of group.entries) {
      const opt = document.createElement('option');
      opt.value = id;
      // gpu.name already includes ★ for entries with estimated:true
      const memPart = gpu.memType ? ` — ${gpu.vram} GB ${gpu.memType}` : ` — ${gpu.vram} GB`;
      opt.textContent = `NVIDIA ${gpu.name}${memPart}`;
      optgroup.appendChild(opt);
    }
    select.insertBefore(optgroup, customOption);
  }
}

// ═══════════════════════════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════════════════════════
function mkEl(tag, cls) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  return el;
}

// ═══════════════════════════════════════════════════════════
//  EXPORTS (Node.js / test harness)
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════
if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', () => {
  // Populate GPU dropdown from GPU_DB
  renderGpuDropdown();
  // Restore state from URL query string (if any)
  deserializeState();

  // Wire up input events for custom gpu fields
  ['cVram', 'cBw', 'cBf16', 'cFp8', 'cFp4', 'cInt8'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => {
      if (document.getElementById('gpuSelect').value === 'custom') {
        state.gpu = buildCustomGpu();
        recalculate();
      }
    });
  });

  // Floating tooltip — delegated so it works on dynamically rendered .tip icons too
  const tipFloat = document.createElement('div');
  tipFloat.id = 'tip-float';
  document.body.appendChild(tipFloat);

  function showTip(el) {
    const text = el.getAttribute('data-tip');
    if (!text) return;
    tipFloat.textContent = text;
    tipFloat.style.display = 'block';

    const r = el.getBoundingClientRect();
    const GAP = 8;
    const TIP_W = 260;
    const TIP_H = tipFloat.offsetHeight;

    let left = r.left;
    if (left + TIP_W > window.innerWidth - 8) left = window.innerWidth - TIP_W - 8;
    if (left < 8) left = 8;

    let top = r.top - TIP_H - GAP;
    if (top < 8) top = r.bottom + GAP;

    tipFloat.style.left = left + 'px';
    tipFloat.style.top = top + 'px';
  }

  document.body.addEventListener('mouseover', e => {
    const tip = e.target.closest('.tip');
    if (tip) showTip(tip);
  });
  document.body.addEventListener('mouseout', e => {
    if (e.target.closest('.tip')) tipFloat.style.display = 'none';
  });

  // Init token on load
  const savedToken = localStorage.getItem('hfToken');
  if (savedToken) {
    const tEl = document.getElementById('hfToken');
    if (tEl) tEl.value = savedToken;
  }
});

function toggleSettings(show) {
  const m = document.getElementById('settingsModal');
  if (m) m.classList.toggle('show', show);
}

function saveToken() {
  const t = document.getElementById('hfToken').value.trim();
  localStorage.setItem('hfToken', t);
}