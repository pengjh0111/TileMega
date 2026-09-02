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
| 1 | P1/P2 generated CG → L0.5/L1/L2 | ✅ importer + table-driven codegen, 50/50 on RTX 4090 | [E2E_GEN result](experiments/E2E_GEN/result.md) |
| 1 | P3.2/P3.3 coupling derivation (`W⁻¹∘R`) | ✅ all 13 §2.7 rows auto-derived and asserted (now isl-backed) | [table27](experiments/P3/table27.md) |
| 1 | P3.4 Tier classification and I2 containment | ✅ `containment_test` both directions; Tier 0-3 all exercised | `test/unit/containment_test.cpp`, `test/unit/table27_test.cpp` |
| 2 | P3.5 L2 fine-grained events vs. L1 barrier | ✅ bitwise identical, 50/50 fresh processes; ⚠️ slower than L1 (conservative relaxation) | [E2E_L2 result](experiments/E2E_L2/result.md) |
| 1 | P3.6 generator generalization (second structurally different model) | ✅ 4-layer MHA passes end to end, no model-structure constants in generated `.cu` | [P3_GENERALIZATION result](experiments/P3_GENERALIZATION/result.md) |
| 5 | P3.1 CuTe↔ISL three-level inverse policy | ✅ classifier and Tier consequence tested; ⚠️ isl_map round trip not implemented | `test/unit/layout_bridge_test.cpp` |
| 1 | P3 isl/barvinok dependency + solving-authority migration | ✅ builds and coexists with MLIR; §2.7 re-derived through isl (one table correction); Coarsen, I2 and a quasi-polynomial case covered; E2E bit-identical | [DEPENDENCIES](DEPENDENCIES.md), [P3_ISL result](experiments/P3_ISL/result.md) |

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
- `C = W⁻¹ ∘ R` is a genuine `isl_map`, and `wait`/`fanout` are barvinok
  counts over it rather than a structural formula; `Contains` is
  `isl_map_is_subset`. Re-deriving §2.7 this way reproduces every tabulated
  quantity except row 3's fanout, where the table and the pre-migration
  heuristic were both wrong for the same reason (F-28). Coarsen — §2.3's
  `C_κ`, previously inexpressible — is now available, and its algebraic laws
  are what catch implementation errors in it (F-29). wait and fanout need
  opposite bounding treatments, a barvinok boundary found by failing (F-30).
- isl's divisor must be a literal, so tile sizes are substituted before any
  isl object is built while workload dimensions stay isl parameters
  (invariant I1 preserved) — F-27. A misaligned-tile coupling makes the
  quasi-polynomial genuinely load-bearing: its `wait` has no single scalar
  value, which the pre-migration scalar-typed metric could not represent.
- The verified derivation still does not drive the generated code: the real
  frontend fills couplings from a placeholder and Codegen reads only their
  structure, so the E2E bit hashes are unchanged by the entire migration
  (F-31). That connection is the remaining step for L3b.
- Model structure (layer count, hidden/intermediate width, GQA-vs-MHA head
  ratio) is derived structurally from FX parameter shapes by
  `lib/Frontend/ModelPlan.cpp` and reaches the generator only as CG module
  attributes; the generated `.cu` carries model *data* tables, never a
  model-*structure* control-flow constant. This is verified by a second,
  structurally different model passing end to end with no `#include` of a
  handwritten runtime (F-25). The pattern matcher still only recognizes the
  Llama decoder-layer dataflow family — a structurally distinct model needs a
  new rule in `ModelPlan.cpp`, not automatic support.
- L2 per-edge events are synthesized from the real derived `C` and are
  bitwise identical to L1's global-barrier output, but every emitted
  dependency uses the conservative `Map::kAll` relaxation (a whole producer
  stage, not the exact producer CTAs `C` identifies) because the TaskBody ABI
  does not yet carry a CTA→task ownership fact. L2 is therefore currently
  slower than L1, not faster; closing that gap is an ABI change, not an
  algorithm change (F-24).
