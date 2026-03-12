// ═══════════════════════════════════════════════════════════
//  GPU DATABASE
// ═══════════════════════════════════════════════════════════
const GPU_DB = {
  h100: {
    name: 'H100 SXM',
    vram: 80,
    bw: 3350,      // GB/s
    fp32: 67,      // TFLOPS
    bf16: 989,
    fp16: 989,
    fp8: 1979,
    int8: 1979,
    int4: 1979,    // software dequant, no hardware gain over int8
    fp4: null,
    nvlink: 900,
  },
  h200: {
    name: 'H200 SXM',
    vram: 141,
    bw: 4800,
    fp32: 67,
    bf16: 989,
    fp16: 989,
    fp8: 1979,
    int8: 1979,
    int4: 1979,
    fp4: null,
    nvlink: 900,
  },
  b200: {
    name: 'B200 SXM ★',
    vram: 192,
    bw: 8000,
    fp32: 90,
    bf16: 2250,
    fp16: 2250,
    fp8: 4500,
    int8: 4500,
    int4: 4500,
    fp4: 9000,
    nvlink: 1800,
    estimated: true,
  },
  a100: {
    name: 'A100 SXM',
    vram: 80,
    bw: 2000,
    fp32: 19.5,
    bf16: 312,
    fp16: 312,
    fp8: null,     // not supported
    int8: 624,
    int4: 624,     // software dequant
    fp4: null,
    nvlink: 600,
  },
};

// Precision metadata
const PREC_META = {
  fp32: { bytes: 4, label: 'FP32' },
  bf16: { bytes: 2, label: 'BF16' },
  fp16: { bytes: 2, label: 'FP16' },
  fp8:  { bytes: 1, label: 'FP8' },
  int8: { bytes: 1, label: 'INT8' },
  int4: { bytes: 0.5, label: 'INT4', softwareOnly: true },
  fp4:  { bytes: 0.5, label: 'FP4' },
};

// KV cache always stays at bf16 (common default)
const KV_BYTES = 2;

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
  const disclaimer   = document.getElementById('b200Disclaimer');

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

  disclaimer.classList.toggle('show', v === 'b200');
  updatePrecisionPills();
  recalculate();
}

function buildCustomGpu() {
  return {
    name: 'Custom GPU',
    vram:  +document.getElementById('cVram').value  || null,
    bw:    +document.getElementById('cBw').value    || null,
    bf16:  +document.getElementById('cBf16').value  || null,
    fp16:  +document.getElementById('cBf16').value  || null,
    fp32:  null,
    fp8:   +document.getElementById('cFp8').value   || null,
    fp4:   +document.getElementById('cFp4').value   || null,
    int8:  +document.getElementById('cInt8').value  || null,
    int4:  +document.getElementById('cInt8').value  || null,
  };
}

function stepGpu(d) {
  state.gpuCount = Math.max(1, Math.min(16, state.gpuCount + d));
  document.getElementById('gpuCountVal').textContent = state.gpuCount;
  recalculate();
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
    const fallback = ['bf16','fp16','int8','fp32'].find(p => getTFLOPS(gpu, p) !== null) || 'bf16';
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

function getTFLOPS(gpu, prec) {
  if (!gpu) return null;
  const map = { fp32: gpu.fp32, bf16: gpu.bf16, fp16: gpu.fp16,
                fp8: gpu.fp8, int8: gpu.int8, int4: gpu.int4, fp4: gpu.fp4 };
  return map[prec] ?? null;
}

// ═══════════════════════════════════════════════════════════
//  HUGGINGFACE FETCH
// ═══════════════════════════════════════════════════════════
async function fetchHFConfig() {
  const modelId = document.getElementById('hfModelId').value.trim();
  if (!modelId) return;

  const statusEl = document.getElementById('fetchStatus');
  const fetchBtn = document.getElementById('fetchBtn');

  statusEl.className = 'fetch-status loading';
  statusEl.textContent = '⟳ Fetching config.json…';
  fetchBtn.disabled = true;

  try {
    const url = `https://huggingface.co/${modelId}/resolve/main/config.json`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();

    // Map common HF config keys → our fields
    const mappings = {
      params:   inferParams(cfg),
      layers:   cfg.num_hidden_layers ?? cfg.n_layer ?? cfg.num_layers ?? null,
      hidden:   cfg.hidden_size ?? cfg.d_model ?? cfg.n_embd ?? null,
      ffn:      cfg.intermediate_size ?? cfg.ffn_dim ?? cfg.inner_dim ?? null,
      heads:    cfg.num_attention_heads ?? cfg.n_head ?? null,
      kvHeads:  cfg.num_key_value_heads ?? cfg.num_attention_heads ?? cfg.n_head ?? null,
      context:  cfg.max_position_embeddings ?? cfg.n_positions ?? cfg.max_seq_len ?? null,
      vocab:    cfg.vocab_size ?? null,
    };

    // Infer precision from torch_dtype
    const dtype = cfg.torch_dtype ?? cfg.dtype;
    if (dtype) {
      const dtypeMap = { float32:'fp32', float16:'fp16', bfloat16:'bf16' };
      const mapped = dtypeMap[dtype];
      if (mapped) {
        setPrecision(mapped);
      }
    }

    // Fill fields
    setFieldVal('mParams',  mappings.params   ? (mappings.params / 1e9).toFixed(2) : '');
    setFieldVal('mLayers',  mappings.layers);
    setFieldVal('mHidden',  mappings.hidden);
    setFieldVal('mFFN',     mappings.ffn);
    setFieldVal('mHeads',   mappings.heads);
    setFieldVal('mKVHeads', mappings.kvHeads);
    setFieldVal('mContext', mappings.context);
    setFieldVal('mVocab',   mappings.vocab);

    statusEl.className = 'fetch-status ok';
    statusEl.textContent = `✓ Loaded ${cfg.model_type ?? modelId}`;

    recalculate();
  } catch(e) {
    statusEl.className = 'fetch-status err';
    statusEl.textContent = `✗ ${e.message} — check model ID`;
  } finally {
    fetchBtn.disabled = false;
  }
}

function inferParams(cfg) {
  // Some models expose safetensors metadata with total count
  if (cfg.safetensors?.total) return cfg.safetensors.total;
  // Estimate from architecture if we have the dims
  const L = cfg.num_hidden_layers;
  const H = cfg.hidden_size;
  const V = cfg.vocab_size;
  if (!L || !H || !V) return null;

  const attn = 4 * H * H; // Q+K+V+O projections
  const emb  = 2 * V * H; // embedding + lm_head

  // MoE architecture
  const nRouted = cfg.n_routed_experts ?? cfg.num_experts ?? 0;
  const moeFFN  = cfg.moe_intermediate_size;
  if (moeFFN && nRouted > 0) {
    const nShared   = cfg.n_shared_experts ?? 0;
    const freq      = cfg.moe_layer_freq ?? 1;
    const moeLayers = Math.ceil(L / freq);
    const denseLayers = L - moeLayers;
    const ffnPerMoeLayer   = (nRouted + nShared) * 3 * H * moeFFN;
    const denseFFN = cfg.intermediate_size;
    const ffnPerDenseLayer = denseFFN ? 3 * H * denseFFN : 0;
    return emb + moeLayers * (attn + ffnPerMoeLayer) + denseLayers * (attn + ffnPerDenseLayer);
  }

  // Dense architecture
  const FFN = cfg.intermediate_size ?? cfg.ffn_dim ?? cfg.inner_dim;
  if (!FFN) return null;
  return emb + L * (attn + 3 * H * FFN);
}

function setFieldVal(id, val) {
  const el = document.getElementById(id);
  if (val != null && val !== '') {
    el.value = val;
  }
}

// ═══════════════════════════════════════════════════════════
//  CALCULATIONS
// ═══════════════════════════════════════════════════════════
function readModel() {
  return {
    params:  +document.getElementById('mParams').value   || null,
    layers:  +document.getElementById('mLayers').value   || null,
    hidden:  +document.getElementById('mHidden').value   || null,
    ffn:     +document.getElementById('mFFN').value      || null,
    heads:   +document.getElementById('mHeads').value    || null,
    kvHeads: +document.getElementById('mKVHeads').value  || null,
    context: +document.getElementById('mContext').value  || null,
    vocab:   +document.getElementById('mVocab').value    || null,
  };
}

function readBatch() {
  return {
    batch:  +document.getElementById('bBatch').value  || 32,
    output: +document.getElementById('bOutput').value || 256,
  };
}

function calcVRAM(model, gpu, precision, gpuCount, batch) {
  const bytesPerParam = PREC_META[precision]?.bytes ?? 2;
  const totalVram = (gpu?.vram ?? 80) * gpuCount;

  // 1. Weights
  const paramsB = model.params ?? estimateParams(model);
  const weightsGB = paramsB ? (paramsB * 1e9 * bytesPerParam) / 1e9 : null;

  // 2. KV Cache
  // Per token: 2 (K+V) * layers * kvHeads * (hidden/heads) * KV_BYTES
  let kvGB = null;
  if (model.layers && model.kvHeads && model.hidden && model.heads && model.context) {
    const headDim = model.hidden / model.heads;
    const kvPerToken = 2 * model.layers * model.kvHeads * headDim * KV_BYTES;
    kvGB = (kvPerToken * model.context * batch) / 1e9;
  }

  // 3. Activations (rough: hidden * context * batch * 4 bytes, peak during fwd pass)
  let actGB = null;
  if (model.hidden && model.context) {
    actGB = (model.hidden * model.context * batch * 4) / 1e9;
  }

  // 4. CUDA overhead (fixed)
  const cudaGB = 1.0;

  const used = (weightsGB ?? 0) + (kvGB ?? 0) + (actGB ?? 0) + cudaGB;
  const freeGB = Math.max(0, totalVram - used);
  const overflow = Math.max(0, used - totalVram);

  return { weightsGB, kvGB, actGB, cudaGB, freeGB, totalVram, used, overflow };
}

function estimateParams(model) {
  if (!model.layers || !model.hidden || !model.ffn || !model.vocab) return null;
  const attn = 4 * model.hidden * model.hidden;
  const ffn  = 3 * model.hidden * model.ffn;
  const emb  = 2 * model.vocab * model.hidden;
  return (emb + model.layers * (attn + ffn)) / 1e9;
}

function calcThroughput(model, gpu, precision, gpuCount, batch) {
  if (!gpu) return null;
  const prec = precision;

  // Effective TFLOPS across N GPUs
  let tflops = getTFLOPS(gpu, prec);
  if (tflops === null) return null;
  tflops = tflops * gpuCount;

  // Effective memory bandwidth across N GPUs (BW scales with GPU count for TP)
  const bwGBs = (gpu.bw ?? 3350) * gpuCount;

  const paramsB = model.params ?? estimateParams(model);
  if (!paramsB) return null;

  const bytesPerParam = PREC_META[prec]?.bytes ?? 2;
  const modelBytes = paramsB * 1e9 * bytesPerParam;

  // Arithmetic intensity threshold (ops/byte) = TFLOPS / BW
  const ridgePoint = (tflops * 1e12) / (bwGBs * 1e9); // ops/byte

  // Decode step: per-token FLOPs ≈ 2 * params (each param touched once)
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

  return {
    tps: Math.round(tps),
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

// ═══════════════════════════════════════════════════════════
//  MAIN RECALCULATE
// ═══════════════════════════════════════════════════════════
function recalculate() {
  if (document.getElementById('gpuSelect').value === 'custom') {
    state.gpu = buildCustomGpu();
  }

  const model = readModel();
  const batch = readBatch();
  const gpu   = state.gpu;
  const prec  = state.precision;
  const N     = state.gpuCount;

  const hasGPU   = !!gpu;
  const hasModel = !!(model.params || (model.layers && model.hidden && model.ffn));

  if (!hasGPU || !hasModel) {
    document.getElementById('emptyState').style.display = 'flex';
    document.getElementById('outputPanel').querySelectorAll('.results-block').forEach(e => e.remove());
    return;
  }

  document.getElementById('emptyState').style.display = 'none';

  const vram    = calcVRAM(model, gpu, prec, N, batch.batch);
  const throughput = calcThroughput(model, gpu, prec, N, batch.batch);

  renderResults(model, gpu, prec, N, batch, vram, throughput);
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
  const tpsSub = tp ? `${PREC_META[prec].label} · batch ${batch.batch} · ${N > 1 ? N+'× GPU TP' : '1 GPU'}` : '';
  const utilVal = tp ? tp.flopsUtil.toFixed(1) + '%' : '—';
  const modelGB = vram.weightsGB ? vram.weightsGB.toFixed(1) : '—';

  heroEl.innerHTML = `
    <div class="hero">
      <div class="metric-card primary">
        <div class="metric-label">Tokens / Second</div>
        <div class="metric-value" id="tpsVal">${tpsVal}</div>
        <div class="metric-sub">${tpsSub}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">FLOPS Utilization</div>
        <div class="metric-value">${utilVal}</div>
        <div class="metric-sub">of ${tp ? tp.peakTFLOPS.toFixed(0) : '—'} TFLOPS peak</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Model Footprint</div>
        <div class="metric-value">${modelGB} <span style="font-size:1rem;color:var(--text-dim)">GB</span></div>
        <div class="metric-sub">of ${vram.totalVram} GB total VRAM</div>
      </div>
    </div>
  `;
  panel.appendChild(heroEl);

  // ── Bottleneck badge ──
  if (tp) {
    const badgeEl = mkEl('div', 'results-block bottleneck-row');
    const cls  = tp.isComputeBound ? 'compute' : 'memory';
    const icon = tp.isComputeBound ? '⚡' : '💾';
    const label = tp.isComputeBound ? 'Compute-Bound' : 'Memory-Bound';
    const desc  = tp.isComputeBound
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
          <span class="gauge-legend-val">${tp ? (tp.mfu*100).toFixed(0) : '—'}%</span>
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

  // ── Breakdown Table ──
  if (tp) {
    const bkEl = mkEl('div', 'results-block viz-card');
    bkEl.innerHTML = buildBreakdownHTML(model, gpu, prec, N, batch, tp, vram);
    panel.appendChild(bkEl);
  }
}

function buildVRAMHTML(v) {
  const total = v.totalVram;
  const segs = [
    { key: 'weights',     gb: v.weightsGB,   label: 'Weights',     cls: 'vram-seg-weights'     },
    { key: 'kv',          gb: v.kvGB,        label: 'KV Cache',    cls: 'vram-seg-kv'           },
    { key: 'activations', gb: v.actGB,       label: 'Activations', cls: 'vram-seg-activations'  },
    { key: 'cuda',        gb: v.cudaGB,      label: 'CUDA/FW',     cls: 'vram-seg-cuda'         },
  ].filter(s => s.gb && s.gb > 0);

  const segHTML = segs.map(s => {
    const pct = Math.min(100, (s.gb / total) * 100);
    return `<div class="vram-seg ${s.cls}" style="width:${pct}%"></div>`;
  }).join('');

  const freeGBDisplay = Math.max(0, v.freeGB).toFixed(1);
  const overflowPct = v.overflow > 0 ? Math.min(100, (v.overflow / total) * 100) : 0;

  // Legend entries
  const dotColors = ['var(--c-weights)','var(--c-kvcache)','var(--c-activations)','var(--c-cuda)','var(--c-free)'];
  const legItems = [
    ...segs.map((s,i) => ({ label: s.label, gb: s.gb, color: dotColors[i] })),
    { label: 'Free', gb: v.freeGB > 0 ? v.freeGB : 0, color: 'var(--c-free)', border: '1px solid var(--border2)' },
  ];

  const legHTML = legItems.map(l => `
    <div class="vleg">
      <div class="vleg-dot" style="background:${l.color};${l.border ? 'border:'+l.border : ''}"></div>
      <div class="vleg-name">${l.label}</div>
      <div class="vleg-val">${l.gb.toFixed(1)} GB</div>
      <div class="vleg-pct">${((l.gb/v.totalVram)*100).toFixed(1)}%</div>
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

function buildBreakdownHTML(model, gpu, prec, N, batch, tp, vram) {
  const paramsB = model.params ?? (vram.weightsGB ? (vram.weightsGB / PREC_META[prec].bytes) : null);
  const rows = [
    ['Model Params',         paramsB ? paramsB.toFixed(2)+'B' : '—'],
    ['Bytes / Param',        PREC_META[prec].bytes],
    ['Weight Memory',        vram.weightsGB ? vram.weightsGB.toFixed(2)+' GB' : '—'],
    ['KV Cache Memory',      vram.kvGB ? vram.kvGB.toFixed(2)+' GB' : '—'],
    ['Peak Compute',         tp.peakTFLOPS.toFixed(0)+' TFLOPS'],
    ['Mem Bandwidth',        (tp.bwGBs/1000).toFixed(2)+' TB/s'],
    ['Ridge Point',          tp.ridgePoint.toFixed(0)+' ops/byte'],
    ['Arith. Intensity',     tp.arithIntensity.toFixed(0)+' ops/byte'],
    ['MFU',                  (tp.mfu*100).toFixed(0)+'%'],
    ['Batch Size',           batch.batch],
    ['Context Length',       model.context ? model.context.toLocaleString() : '—'],
    ['GPU Count (TP)',       N],
  ];

  const rowsHTML = rows.map(([l,v]) => `
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
  const endAngle   = 2 * Math.PI;
  const fraction   = Math.min(100, Math.max(0, pct)) / 100;

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
    grad.addColorStop(0,   '#00e676');
    grad.addColorStop(0.5, '#ffd740');
    grad.addColorStop(1,   '#ff5252');

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
    const ox = cx + (r - 5)  * Math.cos(a);
    const oy = cy + (r - 5)  * Math.sin(a);
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
//  UTILS
// ═══════════════════════════════════════════════════════════
function mkEl(tag, cls) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  return el;
}

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  // Wire up input events for custom gpu fields
  ['cVram','cBw','cBf16','cFp8','cFp4','cInt8'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => {
      if (document.getElementById('gpuSelect').value === 'custom') {
        state.gpu = buildCustomGpu();
        recalculate();
      }
    });
  });
});
