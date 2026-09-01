# V-B — CUTLASS collective inside a persistent loop

Evidence labels: ✅ executed/compiled; ⚠️ cross-compiled or source-inspected;
❌ inference.

## Ada result

The custom kernel owns the persistent tile counter and dynamic shared memory,
then directly invokes CUTLASS `CollectiveMma`; it does not call
`GemmUniversal`. A separate, same-shape `GemmUniversal` kernel is the baseline.

| M=N=K | Repeats | Custom | GemmUniversal | custom / baseline | Output |
|---:|---:|---:|---:|---:|---|
| 2048 | 20 | 0.111616 ms | 0.124928 ms | 0.893443× | ✅ exact match |

The custom implementation delivered 1.119× the baseline throughput, exceeding
the 90% target. It used 49,152 B shared memory, 2 CTA/SM, and a 256-CTA grid.
ptxas reported 246 registers for the custom kernel and 254 for the baseline.

## Interface findings

In this CUTLASS revision, the usable SM80 cp.async mainloop has the effective
call contract:

```cpp
collective(accumulator, gA, gB, source_accumulator,
           k_tile_iterator, k_tile_count, residue_mnk,
           thread_idx, shared_memory);
```

Extracting it from a universal kernel means the caller owns tensor/stride and
residue construction, accumulator initialization, shared storage, task and
K-tile scheduling, inter-iteration CTA synchronization, epilogue/output, and
launch/occupancy configuration.

✅ The current `CollectiveBuilder` does not provide an SM80/SM89 specialization
for this combination, so V-B constructs the public `MainloopSm80CpAsync`
collective through CUTLASS's default configuration helper. The builder did
instantiate the SM90 half-precision TMA/GMMA warp-specialized path. On sm_120,
the same half-precision builder was rejected because the current SM120 builder
supports narrow MMA types; selecting FP8 instantiated
`MainloopSm120TmaWarpSpecialized` successfully. This is a capability/type
choice, not a numeric architecture comparison.

Architecture-independent pieces are TaskContext, persistent scheduling,
storage-union ownership, dependency events, result contracts, and launch
policy. Collective construction and orchestration are per-family: SM80 uses a
monolithic cp.async mainloop call, while SM90/SM120 use TMA and
producer/consumer warp-specialized roles.

No Mirage source was copied or adapted; the implementation was derived from
CUTLASS's public interfaces and tests, so no third-party license decision was
needed for code reuse.

## Reproduction

```bash
docs/experiments/V_B/run.sh
cat docs/experiments/V_B/raw/performance.txt
```

⚠️ The SM90 and SM120 builder probes were cross-compiled only.
