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
