# MX5 torch.compile recovery on C550

The partial-compile recovery below is retained as history. New fullgraph decode results for all5B of Qwen and OLMoE are complete; see [the native/fullgraph follow-up](MX5_olmoe_native_c550.md). The fixed MX4 baselines and old failures remain unchanged.

Evidence: verified unless explicitly marked inferred. This is a new,
user-authorized recovery campaign; the fixed MX4 baseline CSV is unchanged.
Prompt SHA256: `75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.

Original Qwen3-MoE-L24 B2/B4/B8/B16, both rounds, exited124 at the900s
GPU guard timeout. None recorded an OOM exception; minimum sampled free
memory across those attempts was31189.996MiB. This establishes timeout,
not a proven absence of an unsampled memory peak. Original B1 completed,
recording grouped_mm unsupported, actual eager experts and fullgraph=False.
Its retained fullgraph traceback stops at HF's data-dependent expert_hit loop.
The larger-batch logs also record Dynamo's limit8 and layer_idx specialization.

The opt-in `--moe-compile-boundary experts-eager` keeps the original HF
expert forward unchanged and explicitly excludes it from Dynamo capture.
Decode surrounding the experts remains compiled. fullgraph=False is chosen
up front. The specialization limit becomes max(original_limit, layers+8):
32 for this24-layer model, to accommodate distinct static-cache layer indices.
The original CLI default and all old baseline data are preserved.
This is a partial-compile implementation, separately labelled in the new CSV;
it is not grouped_mm or fullgraph capture. It changes the compilation boundary
and budget, not the HF expert arithmetic or measurement protocol.

CPU regression tests2/2 pass: real small HF BF16 experts retain exact output,
surrounding operations are captured, and dense models reject the MoE option.
A C550 B2/8-token diagnostic captures30 graphs and completes; its two generated
sequences equal a separate wholly eager HF process bit for bit. The diagnostic
is not accepted as standard performance. No recompile-limit warning appears
in the successful diagnostic. Eager expert dispatch still incurs host/device
control overhead; this is an observed execution repair, not a speed claim.

The original repaired collection ran in fresh guarded processes on physical7:
ISL64/OSL1024, BF16 greedy, warmup1/timed2, both rounds attempted for all5B.
Each of those original jobs retained the900s limit. A later timeout/error
remains a failed cell.
The standard recovery campaign is complete: all10 guarded attempts have
closed. B1/B2/B4 each complete both fresh rounds. Median TPOT is
71.823444/122.764771/215.622438ms respectively. B8/B16 both original rounds
hit900s; four explicit failure rows retain blank timing fields in the CSV.
They are not OOM or accepted performance rows. B8 warmup/first timed
request take406.344754/370.787272s in round1 and389.472/375.381s in round2;
the second timed request starts before each timeout. B16 completes only
warmup (618.523s/662.806s), then starts its first timed request. Completed
requests report30 graphs, with no generation exception. B8 round1 sampled
free memory remains at least33077.418MiB. This remaining limit is real
generation runtime: three required requests cannot fit the observed rate
within900s. In that original campaign the timing cap, expert arithmetic,
warmup/repeats and OSL were unchanged. The compile-boundary repair does
not make all batches viable under900s.
New append-only data: [MX5_torchcompile_repair_c550.csv](MX5_torchcompile_repair_c550.csv).

Raw evidence: `runs/mx5/torchcompile_repair_20261030_v1/`:
`REPAIR_SHORT_VALIDATED.json`, `diagnostic_B2_v2/`, `reference_B2/`,
`full/`, `COLLECTION_STATUS.json`, frozen sources and controller receipts.
Every request records compile progress, exceptions and Dynamo counters.
The first diagnostic stopped before generation because its frozen helper
layout omitted the relative guard-policy path. V2 restores the source tree
layout. The first collector CPU validation used the wrong guard-result key;
it stopped before launching a reference/timing job. Both failed attempts remain.

The repair does not establish the sole cause of every original timeout.
It demonstrates that explicit expert boundaries and the model-sized compile
budget complete the B2 diagnostic and both B2/B4 standard1024-token rounds.
Further completed rounds must be read from their process and guard receipts.

The user subsequently explicitly authorized1800s for B8 and2700s for B16.
Both extended rounds now complete for both batches with the identical
checkpoint, frozen repair, warmup1/timed2, ISL64/OSL1024 and BF16 greedy.
All four old900s timeout rows remain unchanged; new rows are labelled
extended_1/extended_2/extended_median. This is a job-duration exception,
not a change to the measurement protocol or expert math. The complete
generation records support runtime exceeding the earlier cap, not OOM.

Verified two-round TPOT medians (ms):

| B | TPOT ms | Per-job cap s |
|---:|---:|---:|
|1|71.823444|900|
|2|122.764771|900|
|4|215.622438|900|
|8|364.138602|1800|
|16|554.719642|2700|

Extended receipts: runs/mx5/torchcompile_repair_20261030_v1/extended_B8_B16/.
Full new-model tables and remaining native gaps are reported in
[MX5_collection_c550.md](MX5_collection_c550.md).

## Fullgraph decode recovery (new implementation)

Verified: an explicit opt-in independent Triton provider repairs the missing
MetaX grouped-mm prefill call. HF5.12 then selects its own batched_mm experts
for decode. Decode forward is compiled fullgraph=True in one graph, with no
recorded graph break. Prefill remains eager HF execution with the custom matrix
multiply. This does not claim stock MetaX grouped_mm support or whole-request
fullgraph capture. Both phase selections and restoration are measured separately.
The frozen baseline has no TileMega runtime/kernel dependency. The old CLI
default, old partial data and original failures remain intact.

All20 fresh standard processes complete at the original900s cap. Initial B1
setup failures (two per model) are retained; the frozen layout was missing the
relative guard-policy file. Same frozen arithmetic, repaired layout and new
processes produced the accepted B1 rounds. All runs use ISL64/OSL1024, BF16
greedy, warmup1/timed2; two process medians follow the original protocol.

Qwen3-30B-A3B-L24, new implementation:

|B|TPOT ms|TTFT ms|E2E ms|output tok/s|
|---:|---:|---:|---:|---:|
|1|13.978311|58.430059|14358.242382|71.317934|
|2|21.796115|75.544187|22372.969985|91.539035|
|4|37.030689|121.502253|38003.897348|107.778885|
|8|68.124411|199.767045|69891.039016|117.213973|
|16|127.765839|344.229775|131048.683098|125.022241|

OLMoE-1B-7B, new implementation:

|B|TPOT ms|TTFT ms|E2E ms|output tok/s|
|---:|---:|---:|---:|---:|
|1|8.798108|35.236460|9035.701288|113.328457|
|2|13.839850|48.064294|14206.230483|144.163607|
|4|22.909596|76.525138|23513.041895|174.201201|
|8|41.199638|117.669863|42264.899436|193.825159|
|16|77.068176|208.103142|79048.847574|207.265322|

The new Qwen results are5.14/5.63/5.82/5.35/4.34 times faster than the old
partial eager-expert implementation; this is a baseline implementation change,
not a TileMega speedup. The earlier timeouts establish no OOM diagnosis.
Frozen-source identity, guard/process receipts and generated tokens are under
runs/mx5/olmoe_native_20261030_v1/. CSV implementation labels distinguish
metax_triton_grouped_mm_prefill from HF_batched_mm_decode explicitly.
