# Finite theta interval writeback

Verified: the production pass now solves every integer seq in a requested
interval, retaining the actual winner among all six placement families at
each point. Geometry, kappa, past and compiled resident grid are held fixed.
The compiler selects geometry and kappa at the upper endpoint before invoking
the interval placement pass. This does not claim joint geometry optimality over
the interval, unbounded extrapolation, or portability of materialized tables.

The CG dictionary carries all point tables and their selected families. Codegen
serializes them into one RuntimeVariantDesc, with host selection by seq. Requests
outside the interval, fixed past or grid are rejected. Legacy descriptors have
zero interval count. No kernel synchronization or TaskSmem lifecycle changes.

Command (rebuild the compiler first):

```sh
build-portable/tools/tilemega-compile docs/experiments/SEQSCAN/raw/export/gqa2.json docs/experiments/WRITEBACK/interval_closure/gqa2.cu --solve docs/experiments/COSTMODEL/event_fit/target.json --seq 5 --seq-begin 1 --past 3 --search-capacity 1 --resource-probes 0 --search-domain docs/experiments/COSTMODEL/event_fit/search_domain.json --dump-cg docs/experiments/WRITEBACK/interval_closure/gqa2.mlir --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv
```

This focused smoke campaign uses one resident CTA per SM without resource
probes; it is not the performance search. `interval_campaign.py` records 50
fresh correctness processes at each of 1,2,3,4,5, all using the same binary.
`interval_check.cpp` independently re-solves every point, compares worker/slot
arrays, checks CG serialization leaves emitted CUDA byte-identical and rejects
inconsistent seq/past/grid metadata. See `correctness/` and `roundtrip.log`.

The pre-existing S5 ISL template certificates remain separate. This bounded
materialized carrier preserves EFT winners that do not fit those templates;
it does not relabel them as grid-stride, rotate, band or wavefront.
