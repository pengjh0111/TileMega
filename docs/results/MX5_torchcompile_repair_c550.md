# MX5 torch.compile recovery on C550

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

Standard collection runs in fresh guarded processes on physical7:
ISL64/OSL1024, BF16 greedy, warmup1/timed2, both rounds queued for all5B.
Each job retains the900s limit. A later timeout/error remains a failed cell.
The standard recovery campaign is complete: all10 guarded attempts have
closed. B1/B2/B4 each complete both fresh rounds. Median TPOT is
71.823444/122.764771/215.622438ms respectively. B8/B16 both rounds still
hit900s; four explicit failure rows retain blank timing fields in the CSV.
They are not OOM or accepted performance rows. B8 warmup/first timed
request take406.344754/370.787272s in round1 and389.472/375.381s in round2;
the second timed request starts before each timeout. B16 completes only
warmup (618.523s/662.806s), then starts its first timed request. Completed
requests report30 graphs, with no generation exception. B8 round1 sampled
free memory remains at least33077.418MiB. This remaining limit is real
generation runtime: three required requests cannot fit the observed rate
within900s. The timing cap, expert arithmetic, warmup/repeats and OSL are
unchanged. The compile-boundary repair does not make all batches viable.
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
