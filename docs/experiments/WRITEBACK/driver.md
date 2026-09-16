# Production compiler solve path (R6, verified initial scope)

```sh
build-portable/tools/tilemega-compile \
  docs/experiments/SEQSCAN/raw/export/gqa2.json \
  docs/experiments/WRITEBACK/gqa2_auto.cu \
  --solve configs/targets/sm_89.json --seq 4 --past 3 \
  --search-capacity 6 --dump-cg docs/experiments/WRITEBACK/gqa2_auto.mlir
```

Verified: the command imports each selected geometry, evaluates all six placement
families for each configuration, writes the winning PlacementOps, and generates
CUDA using the written plan. This run selected 32x16x16s2, split4, kappa1, EFT,
128 workers. The generated binary passed 50/50 fresh processes (`driver_gqa2_s4`).
The written CG table and generated host queue compare byte-for-byte in
`roundtrip_gqa2_s4/plan.diff` after canonical serialization by stage/task.

The source fixes kappa, resident cap, and the solved theta/grid. Conflicting
compile macros fail compilation; a different runtime theta/grid fails before
launch. Legacy modules emit no new definitions and retain existing behavior.
The required variant partition starts at seq=1; the concrete-solve guard keeps
this table from silently claiming an interval proof.

Explicit degraded scope: the finite run evaluated 6 of 3654 uniform-geometry,
split and kappa candidates; 3648 are deferred, not proven dominated. Residency
currently admits one CTA/SM until whole-kernel compiled resource evidence is
supplied. No hop curve was supplied in this channel test (zero-hop binding bound);
`--hop-curve PATH` accepts the calibrated TSV. Neither this shortlist nor these
absolute predictions claim J-group rank, calibration, or performance acceptance.
The 18-point rank replay, higher residency search, all-model round trips, CTest
and SEQSCAN acceptance are tracked separately and remain pending.

The initial command failed on a singleton variant beginning at seq=4, before
CUDA generation. `driver_solve.log` and `driver_first_search.tsv` retain it.
The retry respects the existing partition contract and adds a concrete theta
guard; no Plan execution rule or numerical tolerance was changed.
