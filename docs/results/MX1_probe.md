# MX1 — MetaX C550 device conventions (MX-P)

Platform: 8× MetaX C550 64 GB, container `tm-metax` (maca-sglang 0.5.14 image),
MACA 3.8.2.6, driver 3.8.1, mx-smi 2.3.4, torch 2.10.0+metax3.8.2.2,
triton 3.6.0+metax. GPUs 7/6 only. Probes: `tools/probes/maca/`, raw output
under `runs/mx1/probes/`. Sync claims use ≥5 fresh processes.

## §1.2 checked item by item

| Item | Result | Mark |
|---|---|---|
| Host/container versions | MACA 3.8.2.6 in container (image tag says 122; `Version.txt` 3.8.2.6), driver 3.8.1, mx-smi **2.3.4** (stated 2.3.1) | verified, corrected |
| Scale | 104 APs. Through cu-bridge `cudaGetDeviceProperties` reports cc **10.2**; torch and plain `mcruntime` report 8.0 | verified, corrected |
| ISA | `/sys/.../smi/showisaversion` = 10.2 → **xcore1002**; `mxcc` defaults to xcore1000, `cucc` probes 1002 | verified (new) |
| wave | 64; `warpSize` is a builtin but not usable in `constexpr`; masks are ignored by `__shfl_*_sync`/`__syncwarp` (P-5) | verified |
| Per-AP resources | smem 64 KiB per AP and per block (opt-in = 64 KiB, 64 KiB + 1 KiB launch fails); 131072 regs/AP (512 KiB); max 1024 threads/block, 2048/AP, 16 blocks/AP | verified, corrected (stated 512 threads/block) |
| Matrix unit | `__builtin_mxc_mma_16x16x16bf16` native; wmma 32×8 / 8×32 are emulated with two 16×16 MMAs. Builtins also exist for f16, i8, tf32 (16×16×8) and f32 (16×16×4) | verified, corrected (fp32/tf32 inputs exist) |
| MMA rate | ≈20 cycles per MMA per wave, one MMA unit per wave; 4 waves/AP saturate: 276 TFLOPS (2 waves: 139) | verified |
| Async / waits | `__builtin_mxc_ldg_b128_bsm(dst,src,0,-1,ret0,…)`; `__builtin_mxc_arrive_gvmcnt(n)` needs a constant; with n loads issued, `gvmcnt(n-1)` completes the first (20 reps × 4 depths) | verified |
| Sync & memory | `__syncthreads` = block fence + barrier; `__threadfence` = seq_cst device fence; `__builtin_mxc_fence(order,"warp"/"block"/"device"/"system")`; `__maca_atomic_*` with `memory_scope_device`; clang `__scoped_atomic_*` scopes are rejected by the backend; `__ldcg`/`__ldcv` are plain volatile loads; `__builtin_mxc_inv_kl1` needs xcore2000 | verified |
| Inline asm | PTX in emitted device code fails in the MACA assembler; unused functions, uninstantiated templates and `if constexpr(false)` bodies are fine (P-3) | verified |
| Macros | cucc: host+device `__CUDACC__`, `__NVCC__`, `__MACACC__`; device `__CUDA_ARCH__` **and** `__MACA_ARCH__=1002`. mxcc: only `__MACACC__`, device `__MACA_ARCH__` | verified |
| Memory traits | read BW needs ≥16 KiB in flight per AP (P-10) | verified |
| Performance | DRAM read 1.64 TB/s (4096-bit × 1.8 GHz × 2 = 1.84 TB/s theoretical); BF16 MMA 276 TFLOPS | verified |
| Toolchain | `cucc` = nvcc front end over mxcc (`-arch`, `-Xptxas`, `--expt-relaxed-constexpr` filtered); host-only C++: g++ with cu-bridge + `/opt/maca/include{,/mcr}` headers, link `-lruntime_cu -lmcruntime` | verified |
| mcTlass | not used | — |

## Decisions (G-PROBE)

**§2.5 threads.** 128 compute threads (2 waves) per CTA, one CTA per AP,
grid = 104. Decode bandwidth: 128 threads reach 1.59 TB/s (97 %) with 8
outstanding 16 B loads per thread (P-10); 256 threads only need 4. MMA at 2
waves/AP caps at 139 TFLOPS, which bounds prefill only; the 256-thread
expansion is deferred until an end-to-end measurement shows prefill limited.
Co-residency: 104×1 and 104×2 persistent grids rendezvous (P-8).

**§2.6 protocol.** Publish: data stores → `__syncthreads()` → thread 0
`__threadfence()` → `atomicAdd(event, 1)`. Wait: thread 0 polls with a volatile
load and `__nanosleep` backoff → `__threadfence()` → `__syncthreads()`. Plain
loads afterwards are correct (no stale L1 observed).

Evidence (P-7, 104 blocks, one per AP, 52 pairs, 5 processes each):

| Test | Result |
|---|---|
| MP with stale-line pre-read, REL∈{fence+atomicAdd, release fetch_add} × WAIT∈{atomicAdd(0), relaxed load, volatile, acquire load} × READ∈{plain, volatile, ldg_b128_bsm} | 0 violations / 5.2·10⁷ each, 24 variants × 5 |
| Single-thread MP, 64 lines, consumer volatile reads, no fences anywhere | 0 / 6.66·10⁸ × 5 |
| Single-thread MP, consumer plain reads without consumer fence | ~1.35·10⁸ violations (compiler reuses the pre-read) |
| Same with consumer `__threadfence` | 0 × 5 (with or without producer fence) |
| SB with / without `__threadfence` | 0 / 2·10⁵ × 5 each |

Hardware ordering was not observed to fail (verified for these patterns); the
consumer fence is required as a compiler barrier (verified). The producer fence
is kept because the hardware model is undocumented (inferred). Round trip:
1.2–1.4 µs per handoff pair; volatile poll is the fastest.

**§2.7 decode main loop.** B (register direct): each lane issues 16 B loads
straight into MMA/GEMV fragments with ≥8 loads in flight. `ldg_b128_bsm` with
an 8-deep ring reaches only 1.28 TB/s at 128 threads.

**§2.8 MMA.** `__builtin_mxc_mma_16x16x16bf16(a, b, c)` (a, b: 4×bf16 as
`__fp16 __attribute__((vector_size(8)))`, c: float4), fp32 accumulate. The
accumulation is not bit-exact IEEE (|err| ≈ 2⁻²¹ on integer sums) but is
deterministic. Fragment layout (lane l, element i):

| Operand | Mapping |
|---|---|
| A (M×K) | m = l & 15, k = 4·(l >> 4) + i |
| B (K×N) | n = l & 15, k = 4·(l >> 4) + i |
| C (M×N) | n = l & 15, m = 4·(l >> 4) + i |

A [N,K] row-major weight is B in col-major form: each lane reads 8 contiguous
bytes of one row; a 16 B load feeds two MMAs if A and B share a k permutation.

**§2.9 primitives** (`Target/Maca/DevicePrimitives.h`):

| Primitive | Implementation |
|---|---|
| `Trap()` | `__trap()` |
| event publish / wait | §2.6 above |
| `ComputeSync()` | `__syncthreads()` (≈20 cycles at 128/256 threads); an smem compute-only barrier works but costs ≈500 cycles (P-6) |
| `GlobalTimerNs()` | `__builtin_mxc_get_realtime() * 10` (100 MHz, 10 ns, consistent across blocks) |
| `Clock64()` | `clock64()` (≈1.37 GHz under load, DVFS-dependent) |
| `SmId()` | none found (hwreg 0–31 scanned); trace uses the persistent block index |
| async copy / wait | `__builtin_mxc_ldg_b128_bsm` / `__builtin_mxc_arrive_gvmcnt(N·k)` |
| L2 prefetch | none; `nonpaged_la` prefetch is removed from candidates |
| cache-hinted loads | none distinct from volatile |
| math | `exp2f`, `rsqrtf` |
| `__nanosleep(ns)` | available; sleeps ≈1.9× the request at idle clocks |
| `printf`, `assert` | work |
| TMA, mbarrier, bulk, PDL, cluster, wgmma | absent; excluded at compile time |

## Notes for later phases

- A failed launch (e.g. smem over limit) is reported by `cudaGetLastError()`
  right after the launch, not by the following `cudaDeviceSynchronize()`.
- mx-smi inside the container lists only this container's processes, with
  container PIDs; other tenants show up only as unowned `vram used` (guard uses
  that). Idle board power ≈95–97 W; idle `vram used` 859 MiB.
- Both `CUDA_VISIBLE_DEVICES` and `MACA_VISIBLE_DEVICES` restrict devices; the
  job wrapper sets both.
- A combination of `__builtin_mxc_fence(…,"device")` and atomics inside the
  litmus loop crashes the backend ("sync scope"), while the same code in
  isolation compiles: avoid that builtin, use `__threadfence()`.
- L2 bandwidth was not measured validly (launch-bound); deferred to MX-C.
- Strict aliasing (verified): TileMega puns BF16 arrays through `uint4`;
  MACA's clang at -O3 dropped such stores (RMSNorm produced zeros/NaN), so all
  device builds use `-fno-strict-aliasing`.
- Resource report (verified): `cucc -Xcompiler -resource-usage` prints, per
  kernel, the stack frame and "Used N MTregisters, M STregisters, S bytes
  shared mem"; MT = per-thread vector registers. No separate spill counter.
- MACA torch 2.10 (verified): a contiguous 4-D CPU tensor copied with
  `.cuda()` comes back with channels-last-like strides; `torch.cat` of an empty
  1-D tensor with a 4-D one raises (breaks transformers' DynamicCache; shimmed
  in `hf_check`).
- `mx-smi` may list a process only after its first allocation registers;
  exclusivity is "no other owner and no unowned memory above baseline".
