# B2.1 — producer-wide fence eligibility

✅ Six offline instances, 78 mappings, all queue DAG checks pass and all six
processes explicitly report `ISL_CONTEXT remaining=0`. This is not a GPU
synchronization validation. Original `balanced/` negative and positive data
remain unchanged.

`tools/tilemega-affine-probe.cpp` accumulates two predicates per producer task
over the exact clipped coupling: has a consumer, and has a remote consumer.
Only the first without the second is fence-free. Terminal outputs are excluded:
an empty consumer set does not justify removing publication to external readers.
The full 78-row table is `mappings.tsv`; the following picks the greatest
fence-free count among mappings with no queue growth.

| seq/workers | stage-major fence-free | selected fence-free | selected mapping | longest queue, both |
|---|---:|---:|---|---:|
| 4/16 | 47 | 47 | stage-major | 18 |
| 4/256 | 47 | 47 | stage-major | 18 |
| 128/16 | 95 | 585 | affine_balanced_100 | 142 |
| 128/256 | 35 | 541 | affine_balanced_100 | 22 |
| 512/16 | 1145 | 2929 | affine_balanced_100 | 913 |
| 512/256 | 185 | 2154 | affine_balanced_100 | 70 |

The seq=4 edge-locality improvement does not improve this stronger metric.
No measured fence rate is available (`not_calibrated`), so its price rebate
remains zero even in rows with more eligible producers. Poll-count changes
must be priced separately after runtime projection; these numbers alone do
not establish B2.5's nonzero predicted benefit.

⚠️ B2.2 choice: production integration must require `grid <= resident_limit`
as a checked L-sched constraint. The table proves only the six fixed instances
have resident grids; it does not prove an overresident parameter-domain bound.
`overresident_proven=0` is intentionally retained. Writing and lowering that
constraint, runtime projection integration and GPU validation are not claimed
by this offline result.

Reproduction:

```sh
ninja -C build-portable tilemega-affine-probe
python3 docs/experiments/AFFINE_PROBE/run_balanced.py --out NEW_DIRECTORY
```
