# SM90 search quality v2

Run these commands from the experimental worktree. Use a Release build for
`sm_90` with the same dependency versions as the v1 experiment. Build the
`tilemega` and `tilemega-sq2-space` targets.

Phase 0 requires the authentic export, coordinates, checkpoint, and guard
receipt from the v1 run directory. Run the sampler and the reproduction
independently. Both must finish before collecting new calibration data.

```bash
sq2=runs/sm90_search_quality_v2
v1=/root/TileMega-search-quality/runs/sm90_search_quality/qwen3_1_7b-B1-20261123
case_dir="$sq2/qwen3_1_7b_b1"
mkdir -p "$case_dir"
"$sq2/build-release/tools/tilemega-sq2-space" \
  configs/targets/sm_90_h800.json "$PWD" \
  "$v1/inputs/export.json" "$v1/inputs/seed.json" 100 "$case_dir/qspace.json"
python3 experiments/search_quality_v2/phase0.py \
  --old-run "$v1" --compiler "$sq2/build-release/tools/tilemega" \
  --paper /root/Prompt/tilemega.pdf
```

The sampler draws one uniform integer over the union of the residency-specific
products. These products have different cardinalities and zero-volume strata
receive no draws. GEMM and attention slots come from the matching native
catalog entries. The output records both the `Legal()` acceptance estimate and
the estimate after the native search's additional rules. Only the latter
supplies the samples and random starts.

The reproduction compares `q000` and `q079` on the same physical UUID as v1.
Each process and guard attempt has a separate directory. Guard rejection and
interruption are recorded at that granularity, and accepted processes are
read from their receipts on resume.
