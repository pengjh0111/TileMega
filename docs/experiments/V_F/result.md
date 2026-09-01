# V-F — symbolic shapes in CuTe IR

Evidence labels: ✅ observed; ⚠️ source-inspected or documented but not run;
❌ conjecture.

## Result

⚠️ Execution was not confirmed in this environment. CUTLASS pins an exact LLVM
revision in `cutlass_compiler/LLVM_COMMIT`, but the corresponding
`external/llvm-project` checkout is absent and no system `mlir-opt`/`cute-opt`
is installed. Building that dependency would require a large new LLVM checkout
and build, so the experiment stopped at the reproducible preflight instead of
expanding scope. `run.sh` exits 77 and records this condition.

The checked-in tests nevertheless establish the following narrower facts by
source inspection:

- ⚠️ `cute-fold-static` has explicit `no_fold_dynamic.mlir` coverage.
- ⚠️ dynamic `composition` and `zipped_divide` operations are represented and
  deliberately preserved when they cannot be proven static.
- ⚠️ static `right_inverse` folding is covered, but a dynamic
  `right_inverse` legality/result test was not found; that case is unconfirmed.
- ⚠️ integration coverage is small and is not evidence for production
  code-generation completeness.

## Phase 3 decision

Use the BSD-3-Clause CUTLASS **CuTe MLIR dialect as the analysis-layer layout
representation**, pinned to the CUTLASS/LLVM revisions. It already represents
partially dynamic layouts and leaves unresolved algebra in IR, unlike a Python
runtime-only representation. Keep TileMega's Presburger/ISL analysis as the
authority for symbolic legality, and do not depend on CuTe IR for codegen.

This is a provisional ⚠️ decision until `check-cute`, the four requested
dynamic operations, and the CPU integration tests are executed with the pinned
toolchain. `pycute` was not selected because this repository intentionally does
not import CuTe DSL, and a new layout algebra would duplicate an existing IR
before its confirmed gap is known.

## Reproduction

```bash
docs/experiments/V_F/run.sh
cat docs/experiments/V_F/raw/environment.txt
cat docs/experiments/V_F/raw/source_audit.txt
```

The exact missing-dependency record is in `raw/environment.txt`; the static
test inventory is in `raw/source_audit.txt`.
