# TileMega tools

The default build provides `tilemega` and the independent MLIR driver
`tilemega-opt`. Host tests share `tilemega-unit`; CTest still runs each case in
its own process, with its original name and arguments, and assertions enabled.
CUDA tests and the different-library `isl_crosslink_test` remain independent.

`tilemega compile` accepts the old compiler arguments. `--options FILE.json`
expands a JSON array of argument strings at that position; this preserves
argument order and repeated flags without a second compiler option parser.

```json
["model.pt2", "plan.cu", "--emit", "serving", "--serving", "decode"]
```

| Old executable | Unified command |
| --- | --- |
| tilemega-compile | tilemega compile |
| tilemega-calibrate | tilemega calibrate |
| tilemega-wait-policy | tilemega calibrate wait-policy |
| tilemega-occupancy | tilemega occupancy |
| tilemega-import / derive / wiring / dram-floor / runtime-projection / parametric | tilemega inspect import / derive / wiring / dram-floor / runtime-projection / parametric |
| tilemega-target-audit / op-audit / skeleton-audit / flow-audit | tilemega audit target / op / skeleton / flow |
| tilemega-attention-work / event-cost / interface-probe / scalar-error-probe / task-work-probe | tilemega probe attention-work / event-cost / interface-probe / scalar-error-probe / task-work-probe |
| tilemega-opt | unchanged |

Historical standalone tools in `tools/experimental/` and the four experiment
executables are built only with `-DTILEMEGA_BUILD_EXPERIMENTAL=ON`.
The occupancy/calibrate shell compatibility entry points forward to the unified
driver; they no longer compile an unmanaged binary or choose a CUDA directory.

R11 implementation is in progress. The serving calibration suite, fingerprint
command, architecture/SASS audits and cached Python end-to-end entry point are
tracked in TODO §5.9.2; this initial migration does not claim their completion.

`tilemega audit sass` counts FP64 instructions in a compiled serving library.
`tilemega audit arch` cross-compiles generated CUDA for requested architectures;
it does not claim execution on an unavailable device. `tilemega inspect
request-floor` evaluates the CG-derived DRAM and compute floor at each past
position of a request.

The Python entry point is `PYTHONPATH=python python -m tilemega`. Its
`doctor`, `calibrate`, `build`, and `run` commands read TOML on Python 3.11 or
newer and equivalent JSON on Python 3.10. The full parameter surface is in
[`configs/e2e/llama_b1.toml`](../configs/e2e/llama_b1.toml). The configured
`[device].cache_dir` stores stamped calibration in `targets/`, export graphs
in `exports/`, solved plans in `plans/`, and compiled libraries in
`artifacts/`. Keys include source and calibration fingerprints, options, and
compiler dependencies. Paged decode accepts `features.handoff="auto"` and
measures the handoff binary against its event-plan control; prefill uses the
event plan. Tiled weight lowering remains unavailable, so serving examples
use `features.weight_layout="row"`. The narrowed Llama B=1/16 acceptance
configuration is `configs/e2e/llama_r11_acceptance.json` on Python 3.10
and the equivalent `.toml` on Python 3.11 or newer.

On a new accelerator, `PYTHONPATH=python python -m tilemega doctor --hwcheck`
builds and runs the native page tests, calibrates and builds a Llama B=1 plan,
checks 64 generated steps in L1/L2, and compares two-step PDL on/off output
when the device supports PDL. The report prints the actual native `-arch`
used; compiling a lower PTX target and relying on JIT would check the wrong
transport path. The command uses row-major, handoff-off geometry because its
purpose is to validate the architecture path; TF-1 has a separate gate.
Hardware checking materializes one explicit legal 16×128×128×2 Llama seed
configuration through the ordinary solver/codegen path. It does not run
coordinate descent or establish a performance optimum; `build` and `run`
perform the full search.
