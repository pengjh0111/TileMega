# TileMega pre-construction verification

Evidence labels: ✅ compiled/executed and observed; ⚠️ documented,
source-inspected, cross-compiled without running, or explicitly unconfirmed;
❌ conjecture.

| Priority | Experiment | State | Evidence |
|---:|---|---|---|
| 1 | V-A cross-block event synchronization | ✅ complete on RTX 4090 | [V-A result](experiments/V_A/result.md) |
| 2 | V-I four-target cross compilation | ✅ all four targets compile | [V-I result](experiments/V_I/result.md) |
| 3 | V-B CUTLASS collective in persistent loop | ✅ sm_89 run; ⚠️ sm_90/120 compile | [V-B result](experiments/V_B/result.md) |
| 3 | V-D compile-time traits query | ✅ complete | [V-D result](experiments/V_D/result.md) |
| 4 | V-G shared-storage union | ✅ complete on sm_89; ⚠️ target projections | [V-G result](experiments/V_G/result.md) |
| 4 | V-H `torch.export` coverage | ✅ strict two-layer Llama export | [V-H result](experiments/V_H/result.md) |
| 5 | V-E nvcc compilation baseline | ✅ complete | [V-E result](experiments/V_E/result.md) |
| 5 | V-F symbolic CuTe IR behavior | ✅ pinned LLVM build and tests | [V-F result](experiments/V_F/result.md) |
| 6 | V-C cluster DSMEM | ✅ sm_90/120 compile; ⚠️ not run | [V-C result](experiments/V_C/result.md) |
| 4 | V-J tile-size and backward-wait controls | ✅ complete on RTX 4090 | [V-J result](experiments/V_J/result.md) |
| 2 | E2E L0 → L0.5 → L1 | ✅ 50/50 on RTX 4090 | [E2E result](experiments/E2E/result.md) |
| 1 | P1/P2 generated CG → L0.5/L1 | ✅ importer + codegen, 50/50 on RTX 4090 | [E2E_GEN result](experiments/E2E_GEN/result.md) |

## Reproduction policy

Synchronization and race experiments use at least 50 fresh processes per
cell, except the explicitly requested 20-run expected-hang V-J backward scan.
Hang capture uses a timeout and preserves raw process/probe logs. Resource and
grid values come from `TargetSpec`, CUDA occupancy queries, or cubin metadata;
they are not copied into business code.

Every experiment directory contains source, a standalone `run.sh`, raw data,
and a result. An optional-environment blocker exits 77 and is recorded as ⚠️,
never converted into a pass.

## Key decisions carried forward

- Cross-CTA publication retains the fence and CTA-wide release sequence.
- Barriers are forbidden inside a thread-divergent spin loop.
- Full-grid co-residency is a property of the realized wait-for graph and
  scheduling frontier, not of `grid > occupancy_capacity` alone.
- TaskBody stage count and union feasibility are functions of `TargetSpec`.
- Architecture selection uses exact capability tags. In particular, sm_120
  has no `tcgen05` path even though it is numerically newer than sm_100.
- CUTLASS traits are suitable for early shared-memory pruning; final survivors
  still require true compilation for register and occupancy evidence.
- `torch.export` symbol expressions and executable guards must both enter the
  frontend parameter domain; dense KV cache is explicit input/output state in
  the tested decoder.
- CuTe MLIR represents dynamic layout algebra but rejects a dynamic-shape
  right inverse; the CuTe-to-Presburger bridge owns that fallback.
- The fixed E2E lowering directly calls the SM80 cp.async collective family
  and validates the full-stage global-barrier path at the resident grid.
- The product frontend is now a thin, torch-versioned Python serialization
  bridge followed by a C++ JSON-to-CG `ModuleOp` importer. Codegen accepts only
  that verified dialect module; generated L0.5 matches the handwritten
  reference bitwise and generated L1 passes 50/50 fresh processes.
