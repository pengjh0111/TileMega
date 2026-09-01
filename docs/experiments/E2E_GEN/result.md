# Generated L0.5/L1 acceptance

Evidence labels: ✅ executed/observed; ⚠️ scope limitation; ❌ conjecture.

## Contract and frontend

✅ The run starts from V-H's real torch 2.13 `ExportedProgram`, not from a
handwritten CG. `python/tilemega/export_bridge.py` serializes only stable graph
facts. The C++ importer owns operator classification, structured W/R/C
skeletons, whitelist checks, symbolic binding, and stage formation, and emits
the sole compiler contract: a verified CG dialect `ModuleOp`.

| Imported property | Result |
|---|---:|
| `call_function` task spaces | 179 |
| Tensor-use couplings | 222 |
| Semantic stages | 24 |
| Range symbols | 4 |
| Equality guards | 4 |
| Unsupported operators | 0 |

All four guards remain constraints. After cancellation and union-find,
`s61 → s14` and `s65 → s14`; the original guard count remains four. `view` and
`transpose` nodes survive as task spaces whose structured access-map kinds are
`view` and `transpose`. The unsupported-operator unit fixture is rejected with
the exact name `aten.imaginary.default`.

✅ `tilemega-opt` parsed, verified, printed, and reparsed the imported module.
Its negative lit tests rejected an event extent different from
`image(C_kappa)` and Tier 3 with cluster synchronization. A C++ round-trip test
proved `ClosedForm → ClosedFormAttr → ClosedForm` semantic equality.

## Generated numerical ladder

`CouplingGraphToCUDA::Lower()` accepts only `mlir::ModuleOp`. It traverses all
`tilemega.task_space`, `tilemega.coupling`, and `tilemega.placement` ops,
evaluates metrics via `ClosedForm::Eval`, and emits TaskBody specialization,
stage-count schedule, §8 synchronization helpers, and the TargetSpec-driven
launcher.

| Comparison | Evidence | Result |
|---|---|---|
| Generated L0.5 vs PyTorch L0 | 0 mismatches; max abs `1.5497208e-6` | ✅ within `3e-5` |
| Generated L1 vs generated L0.5 | 0 mismatches; max abs `0` | ✅ bitwise |
| Generated L0.5 vs handwritten L0.5 | both hash `5245714bc5d3ab4d` | ✅ bitwise |
| Generated L1, fresh processes | 50 pass / 0 fail / 0 hang / 0 error | ✅ 50/50 |

The retained `docs/experiments/E2E/e2e.cu` is now explicitly a handwritten
reference baseline. It is not read by the importer or Codegen.

## Resources and launch

| Quantity | Generated | Handwritten reference |
|---|---:|---:|
| Threads/CTA | 256 | 256 |
| Registers/thread (L1) | 168 | 168 |
| Dynamic shared-memory union | 49,536 B | 49,536 B |
| Spills | 0 | 0 |
| Active CTA/SM | 1 | 1 |
| Probed SM count | 128 | 128 |
| Grid | 128 | 128 |

The launch computes
`grid = TargetSpec.num_sms × ActiveBlocksPerSM(l1_kernel, block, union_bytes)`.
Because this Phase-2 schedule places a global barrier after every stage, the
grid equals the resident limit. No hardware resource value is embedded in the
generator. The union contains GEMM, RMSNorm, RoPE, attention, and pointwise
storage and is statically asserted to equal the maximum member size; no member
address escapes dispatch.

The direct GEMM specialization is the V-B/V-E2E SM80 cp.async FP32 family. Its
adapter documents CUTLASS B's logical `(N,K)` coordinates and `(K,1)` stride for
contiguous PyTorch `[N,K]` weights. The 14-call parameter table is passed as one
device pointer; ptxas reports a 32-byte stack and zero spills.

## Performance baseline

Across the 50 generated-process logs:

| Path | Median kernel time |
|---|---:|
| Generated L0.5 | 1.094608 ms |
| Generated L1 | 1.079296 ms |
| L1 / L0.5 | 0.986897× |

This is a correctness baseline, not an optimization claim. The non-GEMM
TaskBodies remain intentionally naive.

## Scope and corrections

⚠️ The explicit stage-forming rule is validated for the two-layer Llama
structure (seven linears per layer). It fails loudly for a different structure
instead of silently guessing; general stage formation remains in
`PROPOSED_SKELETON_CHANGES.md`.

The first MLIR reuse configure failed because V-F's exported CMake files retain
absolute source/build paths after the tree was moved to `/tmp`. Restoring
temporary path mappings reused the exact pinned artifacts without rebuilding.
The initial lit command then exposed two infrastructure assumptions (an empty
`LLVM_EXTERNAL_LIT` and unbuilt default-substitution tools); the final target
invokes the pinned source `lit.py` and substitutes only `tilemega-opt` and
`FileCheck`. These failed attempts were not counted as passes.

The first direct `.so` build also exposed an address-space decision hidden by
the handwritten source: a host `constexpr` schedule is not device-readable.
The generator now derives matching host and device-constant schedules from one
initializer. `tilemega-compile raw/cg.mlir /tmp/tilemega-e2e-final.so` then
exited 0 and produced a 647 KiB ELF shared object; see
`raw/compile_shared.txt`.

## Reproduction

```bash
docs/experiments/E2E_GEN/run.sh
cat docs/experiments/E2E_GEN/raw/import_summary.txt
cat docs/experiments/E2E_GEN/raw/hash_compare.txt
cat docs/experiments/E2E_GEN/raw/fresh_process_summary.txt
cat docs/experiments/E2E_GEN/raw/timing_median.txt
```
