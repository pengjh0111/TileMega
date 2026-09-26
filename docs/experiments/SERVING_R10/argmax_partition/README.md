# Candidate lm_head partitions

The initial unrestricted Llama decode B=1 search stopped before materialization:
`FlowPreparation.cpp` reported 4008 runtime ownership tasks but only 1002
priced tasks at `serving.s130`. The candidate used `tile_n=32` for lm_head,
while its imported argmax-partial relation still used the 128-column seed.

`SkeletonSearch.cpp::Prepare` now rebuilds serving semantics whenever the
candidate lm_head `tile_n` changes. The raw `argmax32.search.tsv` evaluates a
six-class candidate with lm_head `16x32x64s6k1`, `Ec=256`, `Rq=4`, kappa 1,
residency 1. It completed without an ownership mismatch and returned
4,777,778.960 ns. This is a Level 1 prediction, not GPU performance.

Command (from `/root/TileMega`):

```sh
build-portable/tools/tilemega-compile /root/r10_work/export/llama_decode/bridge.json /root/r10_work/argmax32_check.cu --serving decode --batch 1 --past-range 64:1086 --capacity 1088 --solver skeleton --solve /root/r10_work/target_r10_inflight.json --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv --variant-cache /root/r9_work/variant_resources --emit serving --flow-search-only 1 --evaluate-configs /root/r10_work/argmax32_case.json
```

The test also checks `argmax_tile_n` 32, 64, and 128 against the number of
partial-result tiles in the serving ModelPlan. A full search and
materialization are needed to validate the integrated path.
