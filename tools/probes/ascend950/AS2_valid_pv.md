# FP32 PV valid-extent gate

`as2_pv_valid` uses the same CATLASS FP32 `GemmTile<...,32,128,64,false>`
as persistent attention. It compares capacity1088 against a K slice rounded
to64 keys while retaining physical P stride1088 and V row strideD. Four epochs
reuse the same resource with ordinary per-recipe MMAD construction/drain.
An independently poisoned V suffix begins after the rounded boundary; output
guards and full/sliced/poisoned results must agree bitwise. CPU FP64 products
are checked independently under pre-recorded FP32 tolerances.

Build serially with the existing private CANN environment:

```sh
source scripts/as2_cpu_env.sh
nice -n 10 cmake --build "$AS2_WORK/build/taskbody_pv_valid_20261102_v1" \
  --target as2_pv_valid --parallel 1
```

The CLI is `as2_pv_valid rows D valid_keys input_dir new_output_dir`.
Inputs are FP32 `p.bin` [rows,1088], `v.bin`/`poison.bin` [1088,D]. Use only the
single-card `as2_run` wrapper; its output includes actual physical/logical ID,
software/command identity and timeout snapshots. Probe timing excludes MMAD
construction/drain and is diagnostic, never full-request performance evidence.

`verified-run`: D128/GQA2/key65 passes five fresh processes, four epochs each;
D128/GQA3/key129 passes one process. Full/sliced/poisoned outputs agree
bitwise; independent CPU max absolute error is at most1.79e-7 in these cases.
Raw inputs/results: `runs/as2/probes/pv_valid_20261102_v1/`.
Further D128/key127,799,1087 cases pass in `pv_valid_20261102_v2/`.

The following D64/key127 process hits its60s Host limit under external load;
the original image has no phase markers, so this does **not** establish a
kernel deadlock or a numerical failure. No repeat is relabelled a pass.
The producer exits; a later read-only snapshot observes HealthOK/util0 with
the own PID absent. Source/image/logs are protected under
`runs/as2/failures/pv_valid_timeout_20261102_v1/`. The next image adds Host
init/device/allocation/launch-return/completion markers. Recovery is finite,
one group, after repeated idle observations; no reset or other-process action.
Both marked recovery processes (D128/key65 and D64/key127) reach
`kernelComplete` and pass. Their receipts remain separate in
`runs/as2/probes/pv_valid_20261102_recovery_v3/`; the original timeout's phase
is still unknown and D64 is excluded from the model candidate.

Persistent implementation12 is **disabled by default**, D128 only, replay
precision contract1. Old images reject unknown variant12; new Host and device
checks also require `TILEMEGA_LLM_VALID_PV`. The ordinary cached variant9 and
both streaming variants retain separate identities. Component evidence alone
does not admit model correctness, prices, local channels or TPOT.

`verified-run`: native Import/CTG/Solver-generated Qwen3-0.6B B1 plans run
with this candidate and a matched variant9 control in image
`cc49ab31b14aa7b1a3585dbb65b558bfaf115f0bf9574ddff4ce4440ff6f0683`.
Both pass full64/1024 greedy/logits/KV, teacher512 (67 captures, no predicted
token differences), fast/diagnostic C2 and five fresh-process C3 (two reset
requests each). Evidence: `runs/as2/correctness/taskbody_valid_pv_20261102_v1/`.
Queues/dependencies/allocations are identical between these native bootstrap
plans; only the explicit attention variant changes9→12. This isolates the
Body but does not establish cost-driven selection or a performance gain.
