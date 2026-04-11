// ═══════════════════════════════════════════════════════════
//  GPU DATABASE
//
//  Each entry has the following fields:
//    name        — display name (include ★ for estimated specs)
//    arch        — microarchitecture family; used for optgroup label
//    archOrder   — sort key for optgroup ordering (lower = appears first)
//    category    — 'datacenter' | 'prosumer' | 'consumer'; for future filtering
//    memType     — memory technology string; used in dropdown option label
//    vram        — total VRAM in GB
//    bw          — memory bandwidth in GB/s
//    fp32/bf16/fp16/fp8/int8/int4/fp4 — TFLOPS (null = not supported)
//    interconnect — { nvlink_gbps, pcie_gen }
//    estimated   — true if specs are pre-release estimates (shows ★ disclaimer)
// ═══════════════════════════════════════════════════════════
const GPU_DB = {
  // ── Blackwell ──────────────────────────────────────────────
  b200: {
    name: 'B200 SXM ★',
    arch: 'blackwell',
    archOrder: 10,
    category: 'datacenter',
    memType: 'HBM3e',
    vram: 192,
    bw: 8000,
    fp32: 90,
    bf16: 2250,
    fp16: 2250,
    fp8: 4500,
    int8: 4500,
    int4: 4500,
    fp4: 9000,
    interconnect: { nvlink_gbps: 1800, pcie_gen: 5 },
    estimated: true,
  },
  // ── Hopper ─────────────────────────────────────────────────
  h200: {
    name: 'H200 SXM',
    arch: 'hopper',
    archOrder: 20,
    category: 'datacenter',
    memType: 'HBM3e',
    vram: 141,
    bw: 4800,
    fp32: 67,
    bf16: 989,
    fp16: 989,
    fp8: 1979,
    int8: 1979,
    int4: 1979,
    fp4: null,
    interconnect: { nvlink_gbps: 900, pcie_gen: 5 },
  },
  h100: {
    name: 'H100 SXM',
    arch: 'hopper',
    archOrder: 20,
    category: 'datacenter',
    memType: 'HBM3',
    vram: 80,
    bw: 3350,
    fp32: 67,
    bf16: 989,
    fp16: 989,
    fp8: 1979,
    int8: 1979,
    int4: 1979,    // software dequant, no hardware gain over int8
    fp4: null,
    interconnect: { nvlink_gbps: 900, pcie_gen: 5 },
  },
  // ── Ampere ─────────────────────────────────────────────────
  a100: {
    name: 'A100 SXM',
    arch: 'ampere',
    archOrder: 30,
    category: 'datacenter',
    memType: 'HBM2e',
    vram: 80,
    bw: 2000,
    fp32: 19.5,
    bf16: 312,
    fp16: 312,
    fp8: null,     // not supported on Ampere
    int8: 624,
    int4: 624,     // software dequant
    fp4: null,
    interconnect: { nvlink_gbps: 600, pcie_gen: 4 },
  },
  a100_pcie: {
    name: 'A100 PCIe',
    arch: 'ampere',
    archOrder: 30,
    category: 'datacenter',
    memType: 'HBM2e',
    vram: 80,
    bw: 1935,      // HBM2e — PCIe variant is ~3% slower than SXM
    fp32: 19.5,
    bf16: 312,
    fp16: 312,
    fp8: null,     // not supported on Ampere
    int8: 624,
    int4: 624,     // software dequant
    fp4: null,
    interconnect: { nvlink_gbps: null, pcie_gen: 4 },
  },
};

// ═══════════════════════════════════════════════════════════
//  PRECISION METADATA
// ═══════════════════════════════════════════════════════════
const PREC_META = {
  fp32: { bytes: 4, label: 'FP32' },
  bf16: { bytes: 2, label: 'BF16' },
  fp16: { bytes: 2, label: 'FP16' },
  fp8:  { bytes: 1, label: 'FP8' },
  int8: { bytes: 1, label: 'INT8' },
  int4: { bytes: 0.5, label: 'INT4', softwareOnly: true },
  fp4:  { bytes: 0.5, label: 'FP4' },
};

// ═══════════════════════════════════════════════════════════
//  EXPORTS (Node.js / test harness)
// ═══════════════════════════════════════════════════════════
if (typeof module !== 'undefined') {
  module.exports = { GPU_DB, PREC_META };
}
