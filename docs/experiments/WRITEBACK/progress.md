# R6 writeback progress

Verified: `tilemega-opt` registers a real `PlacementSolvePass`. It consumes a
verified CG, calibrated TargetSpec and bound theta, derives node costs through
TaskInstanceNs, invokes the six-placement solver catalog, and writes the
selected mode/parameters/slot table and resident constraints to CG. The first
run on freshly imported gqa2 chooses rotate, grid128 at seq4/past3.

```
build-portable/tools/tilemega-import docs/experiments/SEQSCAN/raw/export/gqa2.json > docs/experiments/WRITEBACK/gqa2_input.mlir
build-portable/tools/tilemega-opt docs/experiments/WRITEBACK/gqa2_input.mlir --tilemega-solve-placement='target=configs/targets/sm_89.json seq=4 past=3' > docs/experiments/WRITEBACK/gqa2_solved.mlir
```

This initial pass is explicitly restricted to kappa1 and one CTA/SM. It does
not claim the compiler-driver closure gate or the final research gate. Higher
residencies require compiled resource confirmation; grouped-event dependencies
must be supplied before accepting kappa>1. Geometry search remains the outer
solver's responsibility and is not silently replaced by a pass-local default.
The API accepts an explicit HopCurve; the initial opt probe uses a zero-hop
binding-bound diagnostic, not calibrated synchronization timing.
