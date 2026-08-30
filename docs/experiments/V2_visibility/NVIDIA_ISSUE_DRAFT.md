# NVIDIA issue draft — Tile IR: CTA-collective `BAR.SYNC` inside a spin-wait `loop` aliases the barrier used by `reduce`, producing silent wrong results

> Draft only. Not submitted. Review before filing.

---

## Summary

`tileiras` lowers a Tile IR `loop` into a **CTA-collective** construct: the loop body
contains a `BAR.SYNC.DEFER_BLOCKING 0x0` (all-warp rendezvous on hardware barrier 0),
while the **back edge is per-thread predicated**. When the loop is a spin-wait whose
exit condition comes from a memory load, the predicate is *not* CTA-uniform: threads
observe the flag flip on different iterations, so the CTA splits across barrier
instances.

Because the spin loop's barrier and the barrier that `reduce` uses to combine
per-warp partials through shared memory are **the same hardware barrier 0**, an
arrival by a lagging warp still spinning can satisfy the reduction's barrier. Warp 0
then executes its `LDS` before the other warps have executed their `STS`, and reads
whatever was in the shared-memory partial slots — for a first launch, zeros; for a
subsequent launch on the same context, the **previous launch's partials**.

The result is a **silently wrong reduction**, with no error from the compiler, the
runtime, or `compute-sanitizer`.

The same defect manifests as a **hang** rather than a wrong answer on builds where
`EIATTR_REQNTID` is 256: there the loop lowers to `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80`
(named barrier 1 with an explicit arrival count of 128) and the count is never met.

## Impact

This makes the combination **"spin-wait on a memory flag, then reduce"** unusable in
Tile IR. That combination is the core primitive of any in-kernel producer/consumer
task graph (persistent megakernel, cross-tile-block event synchronisation), and the
reduction is present in GEMM epilogues, softmax, layernorm, and every reduction
operator. Elementwise consumers are unaffected.

Severity: **silent data corruption** with no diagnostic. It is easy to miss in
testing — see "Why this is easy to miss" below.

## Environment

```
GPU                : NVIDIA GeForce RTX 5090 (sm_120a), 170 SM
Driver             : 580.65.06
tileiras           : release 13.3, V13.3.36   (/data/cuda-13.3.1/bin/tileiras)
                     also reproduced on 13.1  (/usr/local/cuda-13.1/bin/tileiras)
Tile IR bytecode   : 13.3 (13.1 for the 13.1 run)
cuda-tile          : af2417041cc939b87ef56d92cfdcf61737c5457e
Target             : --gpu-name sm_120
Launch             : cuLaunchCooperativeKernel, blockDim (1,1,1)
```

Reproduced on **both** tileiras 13.1 and 13.3, with identical generated code
structure in every respect audited (instruction count, barrier flavour, REQNTID,
store/load expansion and placement).

## Minimal reproducer

`repro.mlir` — one producer block writes 1024 floats of `1.0` and releases a flag;
every other block spins on the flag, acquires, loads those 1024 floats and reduces
them. The correct answer is `1024.0` for every consumer.

```mlir
cuda_tile.module @cuda_tile_module {
  entry @repro(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %is_prod = cmpi equal %bx, %c0, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %dv = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %st = store_view_tko relaxed device %ones, %dp[%c0] : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
      %flag_1 = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %one_1 = constant <i32: 1> : tile<1xi32>
      %old, %atok = atomic_rmw_tko release device %flag_1, xchg, %one_1 token=%st : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
      yield
    } else {
      %flag_1c = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %it = make_token : token
      %lt = loop iter_values(%tok = %it) : token -> token {
        %v, %t2 = load_ptr_tko relaxed device %flag_1c token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
        %vs = reshape %v : tile<1xi32> -> tile<i32>
        %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
        if %rd { break %t2 : token }
        continue %t2 : token
      }
      %a, %at = load_ptr_tko acquire device %flag_1c token=%lt : tile<1xptr<i32>> -> tile<1xi32>, token
      %dv2 = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp2 = make_partition_view %dv2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %d, %dt = load_view_tko relaxed device %dp2[%c0] token=%at : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %s = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%e: tile<f32>, %ra: tile<f32>) { %z = addf %e, %ra : tile<f32>
          yield %z : tile<f32> }
      %r1 = reshape %s : tile<f32> -> tile<1xf32>
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko relaxed device %r1, %op[%bx] token=%dt : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
```

In this repo the exact text above is [`repro.mlir`](repro.mlir). Build it with the
commands below to get `repro.cubin` / `repro.sass` (cubins and disassembly are
gitignored); every SASS address quoted below is from that disassembly. It is identical to
[`sass_audit/v1_min.mlir`](sass_audit/v1_min.mlir) except that the final `out` store
uses `relaxed device` instead of `weak`, so that the reproducer contains no spec
§7.2 violation for a reviewer to get distracted by; both fail 50/50.
Host harness: [`v2_host.cpp`](v2_host.cpp).

Verified as written: `repro.mlir` → `REG:15 SHARED:1052`, **50/50 runs fail at
grid=340**, sample histogram `512 x11`.

### Build and run

```bash
cuda-tile-opt repro.mlir -o /dev/null
cuda-tile-translate repro.mlir --mlir-to-cudatilebc --no-implicit-module \
    --bytecode-version=13.3 -o repro.tilebc
tileiras --gpu-name sm_120 repro.tilebc -o repro.cubin

g++ -O2 -I/data/cuda-13.3.1/include v2_host.cpp -o v2_host -lcuda
# grid = 340, one fresh process per run, 50 runs
for i in $(seq 50); do ./v2_host repro.cubin repro 340 scalar 1024 1; done
```

The host allocates `data`/`out`/`flag`, memsets `flag` to 0, fills `data` with a
poison value, launches with `cuLaunchCooperativeKernel(f, grid,1,1, 1,1,1, ...)`,
synchronises, and checks `out[b] == 1024.0f` for `b in [1, grid)`.

### Expected vs actual

| | |
|---|---|
| **Expected** | `out[b] == 1024.0` for every consumer block |
| **Actual** | `out[b] ∈ {768.0, 512.0, 256.0}` for a subset of blocks — **50/50 runs affected at grid=340** |

The wrong values are always `1024 − k·256` for small integer `k`. **256 = 32 lanes ×
8 elements = exactly one warp's partial** of a `tile<1024xf32>` at REQNTID 128
(4 warps × 256). `k` partials are missing.

## Evidence that this is a barrier-pairing bug, not a memory-visibility bug

Four independent observations, each reproducible.

### 1. The corruption occurs with data the kernel never wrote

Replace `%dp2[%c0]` with `%dp2[%c200]`, where chunk 200 is written by the **host via
`cuMemcpyHtoD` before the launch** and touched by no tile block. Cross-block
visibility is physically not in play.

Result: **50/50 failures at grid=340** (36/50 at grid=170), with the identical
`768`/`512` signature. See [`V2_f/v2_f_hostdata.mlir`](V2_f/v2_f_hostdata.mlir).

### 2. Removing only the `reduce` makes it 100% clean; nothing else does

grid = 340, 50 runs per cell, one fresh process each:

| consumer body | cross-warp shared-memory stage? | failures |
|---|---|---|
| load 1 element, no reduce | no | **0/50** |
| load `tile<1024xf32>`, store all 1024 back elementwise, no reduce | no | **0/50** |
| load `tile<32xf32>`, reduce (fits one warp, `SHFL` only, no `STS`/`LDS`) | no | **0/30** |
| load `tile<64xf32>`, reduce | **yes** | **30/30** |
| load `tile<1024xf32>`, reduce | **yes** | **30/30** |

Eight `LDG` instructions are harmless. The switch is exactly whether the reduction
needs `STS` / `BAR.SYNC` / `LDS` to combine warp partials. And the loss is always
**half** the correct answer for a 2-warp reduction — `tile<64xf32>` loses 32, not 128,
which rules out any explanation in units of load/store instructions.

### 3. Direct capture: a later launch reads the *previous* launch's partials

Run N launches in one process, alternating the host fill value between `1.0` and
`3.0`, so the correct answer alternates 1024 / 3072:

```
rep 0 expect=1024   17 bad :  512 x17
rep 1 expect=3072   17 bad :  1536 x2   2048 x14  2560 x1
rep 2 expect=1024   17 bad :  2048 x16  2560 x1
rep 3 expect=3072   33 bad :  1536 x3   2048 x30
rep 4 expect=1024   26 bad :  2048 x26
```

`rep 2` expects **1024** and returns **2048** — *larger* than the correct answer.
No missing store, no stale global data, and no poison value can produce that; those
can only make the sum too small. `2048 = 2×256 + 2×768`: two of the four warp-partial
slots hold this launch's correct partial (256), and two hold the **previous**
launch's partial (768, from the `3.0` fill). The shared-memory slots are being read
before they are written.

With a constant fill, the same kernel fails only on `rep 0` (`512 = 1024 − 2×256`,
cold shared memory reads as 0) and looks clean afterwards, because the stale value
from the previous launch happens to equal the correct one.

Script: [`mechanism_altfill.sh`](mechanism_altfill.sh).

### 4. The generated SASS shows the aliasing directly

Consumer, `cuobjdump -sass` (13.3; 13.1 is structurally identical):

```
/*0320*/  LDG.E.STRONG.GPU R3, desc[UR6][R4.64] ;   each thread loads the same flag
/*0330*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;             <-- CTA-collective, INSIDE the spin loop
/*0350*/  @P0 BRA 0x320 ;                           <-- per-thread predicated back edge
...
/*0370*/  CCTL.IVALL ;                              (from the acquire)
/*0390*/  LDG.E.STRONG.GPU ...    x8                the data load, correctly after CCTL.IVALL
/*0420*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;
          SHFL.BFLY  x5                             intra-warp reduction
/*05b0*/  @!P0 STS ...                              each warp writes its partial
/*05c0*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;             <-- SAME barrier 0
/*05f0*/  @!P1 LDS ...                              warp 0 reads the partials
          SHFL       x2
/*0650*/  @!P0 STS ...
/*0660*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0680*/  LDS ...
/*0690*/  @!P0 STG.E.STRONG.GPU ...
```

`/*0330*/` and `/*05c0*/` are the same hardware barrier. `BAR.SYNC 0x0` rendezvouses
by counting arriving warps; it cannot distinguish "this warp arrived at the spin
loop's barrier" from "this warp arrived at the reduction's barrier". A warp still
spinning at `/*0330*/` therefore satisfies `/*05c0*/` on behalf of a warp that has
not yet reached `/*05b0*/`.

Producer side, for completeness: the `store_view_tko relaxed device` of
`tile<1024xf32>` expands to exactly 8 `STG.E.STRONG.GPU` (0x0120–0x0190, stride
0x200), **all** before `BAR.SYNC` (0x01a0), `MEMBAR.ALL.GPU` (0x01e0) and the release
`ATOMG.E.EXCH.STRONG.GPU` (0x0210), with uniform qualifiers. The data path is
correctly ordered; the defect is not there.

## The 256-thread variant: same defect, presenting as a hang

On an older cuda-tile frontend (`8a775693`) the same source compiles with
`EIATTR_REQNTID = 256`, and the loop's barrier lowers differently:

| REQNTID | barrier in spin loop | consequence of a mismatched arrival count |
|---|---|---|
| 256 | `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` — named barrier 1, **explicit count 128** | count never met → **deadlock** |
| 128 | `BAR.SYNC.DEFER_BLOCKING 0x0` — barrier 0, implicit all-warp | no deadlock → **silent wrong result** |

In the 256-thread build, `cuda-gdb` sampling shows stuck blocks parked at an `LDS`
immediately following a `BAR.SYNC.DEFER_BLOCKING` and an `@!P2 STS` — the same
instruction pair identified above. We believe these are one defect with two
presentations, though we have not run a cross-version controlled experiment to
confirm that (e.g. reproducing the silent corruption on a REQNTID-256 build).

## Ruled out

| hypothesis | how it was excluded |
|---|---|
| Missing/incorrect memory ordering by the user | Producer stores are `relaxed device`, release via `atomic_rmw_tko release device`, consumer does `acquire device` after the loop, all token-chained. SASS confirms `STG.E.STRONG.GPU`, `MEMBAR.ALL.GPU`, `CCTL.IVALL` in the right places. |
| `weak` used for inter-thread communication (spec §7.2 violation) | Changing every data `store_view_tko weak` to `relaxed device` changes the SASS (`STG.E` → `STG.E.STRONG.GPU`) and changes the failure rate **not at all** (50/50 → 50/50). |
| Load/store scheduled on the wrong side of the barrier | Instruction-by-instruction audit: 8/8 stores before the barrier, 8/8 loads after `CCTL.IVALL`. |
| Cross-block data visibility | Fails identically on data written by the host before launch (§1 above). |
| Producer had not finished / spin not actually waiting | Verified: with the flag pre-set so nobody spins, 0/50 failures. The defect requires a real spin. |
| Reduction itself miscomputing | The same reduction with no preceding spin, and with a register-constant input, is correct. |
| Hardware / driver | A structurally equivalent CUDA C++ kernel (`cuda_control.cu`: same grid, same flag protocol, `__syncthreads()` + shared-memory reduction) passes. |
| Occupancy / cooperative-launch admission | grid=8 already fails (11/50) with the serial-chain variant; grid is not the driver — see below. |

## Additional characterisation

Holding **grid fixed at 340** and varying only the number of blocks that participate
in the handshake (the rest exit immediately):

```
participants  2      8      32     128    339
failures     0/50   2/50   22/50  50/50  50/50
```

The driver is the number of blocks concurrently spinning-then-reducing, not the grid
size. With a strictly serial producer→consumer chain (each block waits on its own
flag, reads its own exclusive chunk), failures start at **grid = 8** (11/50) and
reach 50/50 by grid = 80.

## Why this is easy to miss

Running the kernel repeatedly in one process and aggregating hides it: after the
first launch, the stale shared-memory partial from the previous launch equals the
correct value whenever the input does not change, so only the very first launch
reports a wrong answer. Our own earlier measurements produced an apparent "sharp
threshold in grid size" for exactly this reason. Testing requires either a fresh
process per run, or per-launch pass/fail with a changing input.

`compute-sanitizer` does not report anything, and perturbs the timing enough to mask
the failure.

## Suggested fix direction

Either:

1. Do not place a CTA-collective `BAR.SYNC` inside a loop whose back edge is
   per-thread predicated — make the exit condition CTA-uniform (e.g. reduce the
   predicate across the CTA before branching), or
2. Allocate a **distinct named barrier** to the loop construct so that its arrivals
   can never be counted against a reduction's barrier.

Option 2 alone is not sufficient for the REQNTID-256 case, where the explicit arrival
count is itself the problem when the CTA diverges.

## Attachments to include when filing

- `repro.mlir` (above) and `v1_min.mlir`
- `v2_f_hostdata.mlir` — the host-written-data variant that isolates the cause
- `v2_host.cpp` — host harness
- `cuda_control.cu` — passing CUDA C++ control
- `repro.sass` (13.3) and `repro_131.sass` (13.1)
- `mechanism_altfill.sh` and its output
- `.env.json` toolchain snapshots
