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

Phase 1 requires `summary.json` from the Phase 0 reproduction to pass.
Export the complete native geometry domains for each B1 case using
`tilemega-native-domains`. Put the primary domain receipt in
`qwen3_1_7b_b1/native-domains.jsonl` and the secondary receipt in
`qwen3_1_7b_b1/calibration/llama-native-domains.jsonl`. Retain the original
inputs and hashes in the case directory.

Capture the production header hashes before starting any calibration. Store
the backend and codegen header map in
`calibration/production_source_binding.json` under `production_headers`,
and the stage-body header map in `calibration/body_binding.json` under
`header_sha256`. The latter includes the GEMM, epilogue and attention bodies.
The collection checks both the compiled probe receipts and the live sources.

```bash
scripts=experiments/search_quality_v2
calibration="$case_dir/calibration"
python3 "$scripts/make_gemm_probes.py"
python3 "$scripts/make_attention_probes.py"
for kind in gemm attention; do
  python3 "$scripts/build_calibration.py" "$kind"
  python3 "$scripts/calibration_queue.py" --job-spec "$calibration/$kind/queue.json" \
    --output "$calibration/$kind/measurement"
done
```

For the legacy dense constants, compile `scripts/sm90r3_constants.cu` with
`-std=c++20 -O3 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr
--extended-lambda -Iinclude -Ithird_party/cutlass/include -lcuda`. The
current source provides `--sq2-base` and `--sq2-epilogue`. Save that
executable as `calibration/legacy/constants-sq2`. The other modes can use
the same binary at `calibration/legacy/constants` when starting a new run.
The legacy collector only reuses a prior attention result when its exact
command and binary hash match.

```bash
python3 "$scripts/legacy_calibration.py"
python3 "$scripts/collect_stage_calibration.py"
```

The guard owns an individual physical card for one process. Queue items are
identified by suite and replicate. Contention or interference returns the
affected item to the queue. A successful guard receipt is retained on resume.

Phase 2 builds `tilemega`, `tilemega-native-evaluate`, and
`tilemega-sq2-check-cost` in Release mode. Generate the three reserved cost
tables before running the development gate.

```bash
mkdir -p "$case_dir/phase2"
for variant in overlap head-startup tail-publication; do
  python3 "$scripts/make_cost_table.py" "$variant" "$case_dir/phase2/$variant.json"
done
${TILEMEGA_SQ2_PYTHON:-/workspace/llm-stack/envs/vllm/bin/python} "$scripts/rescore_dev.py"
```

The optional cost-table path is fail-closed. The solver checks exact keys for
the fixed dense B1 TP1 workload. The calibration check compares the pooled
body interval with the measured body interval and treats incoming stage
synchronization separately. The development gate only reads the v1 TPOTs
after the candidate variants and binary hashes are registered.

The Phase 3 CPU diagnostic uses the existing offline terminal:

```bash
"$sq2/build-release/tools/tilemega-native-search" \
  configs/targets/sm_90_h800_sq2.json "$PWD" "$v1/cells.json" \
  "$case_dir/phase3/owner-audit.jsonl" \
  --search-profile sq2 --native-cost-table "$case_dir/phase2/overlap.json" \
  --extra-seeds "$case_dir/qspace.json" --diagnose-owners 1
```

`--diagnose-owners` compares a CPU Eq. (5) assignment against the implemented
template. Its hypothetical makespan is diagnostic. It does not participate
in finalist selection.
