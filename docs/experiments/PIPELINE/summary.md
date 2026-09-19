# R7 B1 — cross-task shared-memory pipelining

Written for: the TileMega maintainers reviewing R7 §5.2.

This file is built up as B1 proceeds. Only **B1-b (the occupancy budget)** is
complete; B1-a, B1-c, B1-d and B1-e are not yet measured and are not reported
here.

## B1-b — what a prefetch page costs

✅ Verified on RTX 4090 / sm_89. Reproduce with `occupancy.py`; raw tables in
`raw/occupancy.tsv` (per page) and `raw/occupancy_cells.tsv` (per cell).

B1-b was measured **before** the mechanism was built, on purpose. The page B1
appends after the `TaskSmem` union is dynamic shared memory on the L2 worker
kernel, so it can cost residency, and the grid is sized from that residency
(`ModelHarness.cuh:2931-2946`) — a cap the launch cannot meet is *rejected*,
not silently lowered. Building the mechanism first and discovering the budget
afterwards would repeat the error B0 was created to fix.

### Method

Two independent answers per cell, cross-checked:

- **the driver**, `cuOccupancyMaxActiveBlocksPerMultiprocessor` on a real cubin
  built with the shipped flags (`JOINT/measure.py:32-41`'s `PROTOCOL`, plus the
  cell's frozen `kappa`/`residency`/`placement`). This is the same query
  production makes through `TargetSpec::ActiveBlocksPerSM`
  (`lib/Target/TargetSpec.cpp:461-471`). Going through the driver API on a
  cubin is what lets the page size vary without editing the cell source;
- **F-40's closed form** (`ORACLE/occupancy.sh:23-45`), generalised off its
  hard-coded 256-thread/8-warp shape, because these kernels run **128** threads.

Page 0 is anchored against the harness's own `E2E_RESOURCE` line from the runs
that actually executed (`PHASE2/raw/<cell>/correctness/selected/r0.log`). It
agrees on all four cells, for both the L1 and L2 kernels.

### Before

| cell | configuration | residency used | regs (prod / instrumented) | smem | L2 CTA/SM |
|---|---|---:|---|---:|---:|
| gqa2 s4 | `32x16x64s2k1_k1_r1` | 1 | 94 / 96 | 16384 | 5 |
| gqa2 s128 | `32x16x32s2k1_k1_r5` | 5 | 86 / 96 | 16384 | 5 |
| mha4 s4 | `32x16x64s2k1_k1_r1` | 1 | 94 / 96 | 16384 | 5 |
| mha4 s128 | `32x16x16s2k1_k2_r5` | 5 | 85 / 87 | 16384 | 5 |

All four are **register bound at 5** and all four run 128-thread CTAs. The
`ctas_per_sm=1` the s4 cells print is their chosen `RESIDENCY_CAP`, not a
hardware limit: the hardware allows 5 there too.

Two register counts are given because the numbers the harness logged come from
the **instrumented** build (`TRACE_PHASE`+`TRACE_KLOOP`+`TRACE_SIMT`, B0) while
the budget below is the production build. They differ by up to 10 registers and
both give 5 CTA/SM — which also says B0's probe cost no occupancy.

### The cost curve

Driver CTA/SM against page bytes appended after the union (`raw/occupancy.tsv`;
gqa2 s128 shown, the others differ only in where registers bind):

| page | 0 | 1024 | 2048 | **3072** | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CTA/SM | 5 | 5 | 5 | **5** | 4 | 4 | 3 | 2 |

| cell | free page at 5 CTA/SM | affordable at the residency it runs |
|---|---:|---:|
| gqa2 s4 | 3072 | 84992 |
| gqa2 s128 | 3072 | 3072 |
| mha4 s4 | 3072 | 84992 |
| mha4 s128 | 3072 | 3072 |

The two s4 cells run at `RESIDENCY_CAP=1`, so a page costs them nothing until
it reaches the opt-in ceiling — 84992 B, i.e. 101376 less the union. The two
s128 cells run at 5 and are the binding constraint: **3072 bytes**.

### Is 3072 enough?

The page must hold the successor's no-in-edge operand. For these GEMMs that is
the B tile, `K x 16 x 2` bytes:

| cell | tile K | B tile bytes | stages that fit in the free page |
|---|---:|---:|---:|
| gqa2 s4 | 64 | 2048 | 1 free, 41 within its own budget |
| gqa2 s128 | 32 | 1024 | 3 |
| mha4 s4 | 64 | 2048 | 1 free, 41 within its own budget |
| mha4 s128 | 16 | 512 | 6 |

So yes, and with room: **every cell fits at least one B stage at zero occupancy
cost**, and the constraint falls the convenient way round — the cells that want
the largest page (s4, 2048 B) are the ones running at residency 1 where shared
memory is nearly free, while the cells pinned at 5 want the smallest pages.

### F-40 needs one correction to be used as a shared-memory budget

F-40's closed form predicts the driver exactly on 52 of 56 rows. The four it
misses are the **same row in every cell**: `page=4096`, i.e. `smem=20480`,
where `102400/20480 = 5` exactly. The driver says 4.

The cause is a per-CTA shared-memory reservation the driver adds on top of the
request. Adding 1024 B per CTA makes the form exact on **56 of 56**:

```
ctas_smem = floor(102400 / (smem + 1024))
```

Bisected cliff on gqa2 s128: `dyn=19456` holds 5, `dyn=19584` drops to 4, and
`(19456 + 1024) * 5 = 102400` exactly.

F-40 was fit on sm_89 configurations that were overwhelmingly register bound
(605 register-bound, 150 smem-bound, 322 tie), and the reservation only shows
up when the shared-memory term is the binding one and lands on an exact
boundary — which is precisely the regime B1 moves these kernels into. Quoting
F-40 unamended here would have authorised a 4096 B page and silently cost 20%
of residency. Recorded as F-223.

### What this does not say

- It is a **prediction of resident CTAs**, from the same API production sizes
  the grid with — not an observed count of concurrently running CTAs. It is the
  operative number because the launch is accepted or rejected on it, but it is
  not a measurement of achieved overlap. That is B1-c.
- It is sm_89 only. `optin_cap`, the per-CTA reservation and the register file
  all differ on sm_120; re-derive before quoting.
- The "after" half of B1-b's report — shared bytes, registers and CTA/SM with
  the page actually in place — needs the mechanism and is not in this file yet.
  The page cost is bounded above here; the register cost of splitting the
  TaskBody ABI into `Prefetch`/`Wait`/`Compute` is not, and registers are what
  binds these kernels today.
