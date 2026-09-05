# Runtime GEMM variants

Evidence status: ✅ measured on RTX 4090 (`sm_89`) on 2026-09-05 unless a row
is explicitly marked otherwise.

`tilemega-compile --variants PLAN.json` now imports and lowers every variant
independently.  A generated `ModelSpec` carries the per-GEMM tile, stages and
split-K values, its matching dependency table, ownership flags, and a compact
`seq -> variant` array.  The host selection is one indexed load and validates
the selected interval before launch.  Up to 16 distinct CUTLASS template
instantiations coexist in one executable.

The dependency table belongs to the same runtime variant as the GEMM plan.
Consequently every affine `kWindow` constant is derived with that variant's
own granularity.  The old compile-granularity comparison and
`wait_table=degraded` fallback were removed; runs report
`dependency_table=variant_exact`.

## Correctness

[`multi_variant.tsv`](multi_variant.tsv) selects both intervals of the same
two-variant BF16 GQA executable and records 50 fresh-process passes per
interval.  The corresponding MHA rows are produced by `run.sh` as part of the
same protocol.

The configuration which previously failed 0/50 (`16x64x16s2`, split-K 16)
was regenerated as a runtime plan.  Both reference models now pass 50/50 with
the exact variant-owned wait table; see
[`coarsen_regression.tsv`](coarsen_regression.tsv).  It is tested in FP32
because tile-M 16 is a legal member of the original SIMT FP32 family and is
not legal for the BF16 Tensor Core family (whose tile-M is a multiple of 32).

## Shared-memory union and occupancy

[`curve.tsv`](curve.tsv) contains the measured curve for both requested
composition styles.  `task_union_bytes` is the actual C++ TaskBody union,
`gemm_union_bytes` is its GEMM member, registers come from the generated L2
entry in `ptxas -v`, and occupancy comes from CUDA's occupancy calculator.

| Variants | Task union | Registers | CTA/SM |
| ---: | ---: | ---: | ---: |
| 1 | 16 KiB | 80 | 5 |
| 2 | 16 KiB | 85 | 5 |
| 4 | 16 KiB | 114 | 4 |
| 8 | 18 KiB | 218 | 2 |
| 16 | 96 KiB | 255 | 1 |

Thus the measured no-occupancy-loss capacity is **two variants per binary**.
The first loss is already at four variants and is register-bound (the union
is still 16 KiB); eight variants lose both register headroom and shared
memory, while 16 reaches 255 registers, a 96 KiB union and a 3208-byte local
stack.  Same-operator and per-operator mixtures produced the same maxima for
the selected shape ladder: C++ unions take the largest member rather than the
sum, whereas dispatching more template paths increases compiled register
pressure.  Phase 5 should therefore start with at most two intervals per
kernel, or partition intervals across binaries if more are justified.

Raw sources, binaries and logs are regenerated under `raw/` by `run.sh`; they
are not source artifacts.
