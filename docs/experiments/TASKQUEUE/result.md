# Task-queue execution

## Result

✅ L2 now consumes one materialized queue per physical CTA.  A queue row is a
`TaskRef` carrying the stage, the logical coordinate in that stage, the
original incoming-edge interval, and the deduplicated `TaskWait` interval.
The generator emits one compact, variant-exact `ScheduleStageDesc` program;
the host binds `seq`, `past`, placement and the resident grid and expands it to
the concrete per-worker queues.  This split is necessary because task counts
are runtime-symbolic and therefore cannot be emitted as a fixed C++ table.

✅ Event groups are indexed by **logical producer task**, not by its resident
CTA. A runtime prefix sum gives each stage the aggregate completion row and/or
`ceil(task_count/κ)` fine rows that its outgoing edges actually reference;
every `TaskRef` calls `NotifyTask`, including later grid-stride tasks. A
narrowed edge waits on fine rows, while `kAll` waits once on the aggregate row.
This avoids a superficially queue-shaped but CTA-granular publisher, an
O(producer task count) expansion for one CG edge, and unused atomic streams.

✅ The generated order comes from `ListScheduler` and is not stage-number
order.  For gqa2 it begins `0, 2, 1, 3, 5, 4, 7, 6, ...`.  Generation rejects
a cyclic graph, a non-permutation, or any dependency reversed by the emitted
order.  The host repeats the proof after split-K inserts combiner stages, before
the kernel can launch.

✅ Waiting is per task.  Each task polls only its unique `(producer, group)`
set; polls already satisfied earlier in the same worker queue are lifted, and
the remaining polls are distributed over CTA threads.  At κ=1, a producer
owned by the same worker is discharged by queue order without a global poll.
The two reference models passed 50/50 fresh processes with identical
L0.5/L1/L2 hashes and no timeout.  The exact counts are in
[`correctness.tsv`](correctness.tsv).

✅ The post-lifting wait body is bounded by the task's own interval, rather
than scanning all graph edges. Static `sm_89` disassembly contains 50
instructions from the wait-count branch through its acquire fence
(`0x420..0x740`), including 23 in the retry loop (`0x4f0..0x660`; see
[`sass.tsv`](sass.tsv)). Dynamic lifted waits average 2.50 per task on gqa2 and
2.10 on mha4; the actual retry count is data-dependent and is measured
separately by the κ/attribution experiments.

⚠️ MPK-style fan-in-one normalization was evaluated but not selected. A lower
bound that inserts only `wait_count-1` dummy tasks already needs 344 dummies
for gqa2 and 696 for mha4: total queue task counts would grow from 200 to at
least 544 (2.72×), and from 512 to at least 1,208 (2.36×), before encoding the
dummy fan-out. The aggregate event implements the useful normalization for
full fan-in; narrowed edges then remove repeated `(producer,group)` polls,
lift polls already satisfied earlier in a worker queue, and keep CTA-parallel
polling without multiplying the task table.

✅ I3 is enforced conservatively.  At the tested resident grid the maximum
forward worker dependency span is 15 for both models, below the 256-worker
resident limit.  Both generated graphs retain at least one global-fan-in edge,
so an over-resident launch is rejected rather than inferred safe from stage
distance.  This is a deliberate no-fallback policy: L2 launches exactly the
minimum resident grid supported by both persistent kernels.

⚠️ The 16-layer, 973M-parameter model generated, compiled and completed
**50/50 without a hang**. Its L0.5/L1/L2 paths are bit-identical to one another
in all 50 processes, but all three retain the same pre-existing 198-element
BF16 mismatch against PyTorch, so it is **0/50 end-to-end correct**. At
`seq=4,past=3` its materialized
table contains 8,768 `TaskRef` rows (245,504 B) and 55,556 lifted waits
(444,448 B).  The compact generated schedule is only 240 rows and the
generated translation unit is 102,192 B; the large concrete table is allocated
in device global memory, not constant memory.  L2/L1 in this single diagnostic
run is not reused as a performance result.

✅ These rows cover the two reference fixtures at κ=1. The independent
`seq × past` sweep in `../SEQSCAN/raw/matrix.tsv` passed **1500/1500** fresh
processes, and the complete queue-era κ sweep is in `../COARSEN/`. No
pre-task-queue result is reused.
