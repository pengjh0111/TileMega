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

## Compiled residency (supersedes the initial residency restriction)

The compiler now compiles an unsolved candidate's complete generated TaskBody,
queries both L1 and L2 with CUDA's occupancy calculator, then evaluates every
resident level from 1 through their minimum. Each run records source, command,
ptxas log and `resources.json` under `OUTPUT.cu.resources/<session>/<probe>/`.
It rejects an architecture/SM-count mismatch and hard-checks 8192 MiB before
compiling. `--resource-probes 0` explicitly requests the earlier one-CTA degraded
scope; it is no longer the default for `--solve`.

Verified command (one retained geometry, six placements at each of five levels):

```sh
build-portable/tools/tilemega-compile \
  docs/experiments/SEQSCAN/raw/export/gqa2.json \
  docs/experiments/WRITEBACK/gqa2_resident_auto.cu \
  --solve configs/targets/sm_89.json --seq 4 --past 3 \
  --search-capacity 1 --dump-cg docs/experiments/WRITEBACK/gqa2_resident_auto.mlir \
  --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv
```

CUDA reports 5 CTA/SM for both entry points, 80/85 registers, 16384 dynamic shared
bytes, 128 threads. The solver selected 4 CTA/SM, and that generated binary
passed 50/50 fresh processes (`resident_gqa2_s4/correctness`). This is production
channel/resource validation, not a J-a performance result or shortlist coverage
claim. It was collected before the separate R6 body-calibration input was wired.
Protocol compile options that change occupancy require a corresponding resource
probe; the final host still independently rejects a nonresident requested grid.
