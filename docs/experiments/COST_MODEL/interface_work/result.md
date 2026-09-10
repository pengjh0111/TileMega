# A8: price physical interface incidence and retain live endpoints

✅ CPU model verification, not an end-to-end timing claim. The old Carry
control remains selectable; `TILEMEGA_CG_INTERFACE_DP` defaults OFF until
the broader unified-model acceptance is complete.

`lib/Solver/CostModel.cpp:652` prices
`(sum(wait) - |domain(C)|) * volume * element_bytes * memory_ns_per_byte`.
Both counts come from the physical relation / CG quasi-polynomials.
Subtracting the whole consumer task space would produce negative work on
partial-writer edges; only consumers in domain(C) have a first producer read.
`lib/Solver/TaskModel.cpp:57` reinstantiates candidate granularity and invokes
the existing coupling derivation; it does not retype the alignment formula.

## Two-shape effect and exact optimization

✅ The reference graph contains residual interfaces between GEMMs 3 and 6,
6 and 10, etc. They are not adjacent in the list of GEMMs. Treating the new
price as a unary term or charging only adjacent GEMMs loses real factors.
`lib/Solver/CouplingInterfaceDP.cpp:14` keeps each selected shape until its
last unpriced coupling endpoint, with Residency pinned outside the DP.
The frontier width in both reference graphs is one. Internal split partial
to combine couplings are also priced. The historical chain path remains a
separate control, not an implicit fallback.

At BF16 seq=128/past=3 on the real GEMM3→GEMM6 residual edge:

| Producer M | Consumer M | interface ns |
|---:|---:|---:|
| 32 | 32 | 0 |
| 32 | 128 | 19.230769396298506 |
| 128 | 32 | 0 |
| 128 | 128 | 0 |

N=128, K=16, stages=3, split=1 here. Seq=4 makes this particular edge
zero because its physical M domain is shorter than both tiles; the other
candidate interface factors still have nonzero spread. No positive number
was substituted for this legitimate zero.

## Controlled candidate set

Eight candidates: M/N∈{32,128}, K=16, stages=3, split∈{1,2}; archived
per-shape ptxas registers and the actual max TaskSmem determine residency.
This is not a claim about the complete 1077-candidate DP search.

The following prices hold task pricing on the historical path and vary only
the interface model; seq=128/past=3, BF16, all units ns:

| Model | Old uniform | New uniform | Old per-op | New per-op | interface spread |
|---|---:|---:|---:|---:|---:|
| gqa2 | 127226.8179254006 | 147701.52763689353 | 111119.04882271661 | 131509.97447900119 | 615.3908806976317 |
| mha4 | 255092.89547138085 | 297656.0969595394 | 212266.9549467552 | 254648.56588847894 | 615.3908806976317 |

Uniform→per-op predicted savings are 12.661%→10.962% for gqa2 and
16.788%→14.449% for mha4 (rounded). They are not GPU speedups. The new
interface increases the denominator; it does not guarantee that per-op
shape selection becomes more valuable. Per-op-split equals uniform in this
explicit eight-candidate set.

✅ Each model's exact-frontier selection was compared with exhaustive 16
assignments of the two non-adjacent candidate axes while fixing all other
operators. Whole-Evaluate selected prices are bit-identical. The check was
repeated with unified task prices enabled at gqa2 seq=4 and mha4 seq=128,
including uniform/per-op-split/per-op modes. Raw logs are adjacent to this
report. Five executed rejection branches retain zero isl references; the
negative-wait injection intentionally emits the retained barvinok diagnostic.
All successful runs explicitly finish with `ISL_CONTEXT remaining=0`.

Decomposed DP sums differ from whole-model summation by about 1e-10 ns
because their FP64 addition order differs. This is reported, not used as a
GEMM-bit-gate tolerance. The chosen full-Evaluate price comparison uses
`memcmp`. No numerical/GPU criterion or target rate was changed.

## Reproduction and switches

`build-portable/tools/tilemega-interface-probe /root/TileMega gqa2`
and the analogous mha4 command reproduce the historical-task comparison.
Add `--unified 4` or `--unified 128` for the combined-path checks.
`--unified` is a test option; `CostModelOptions::cg_interface` and
`::unified_task_cost` select the independent production paths.

`TILEMEGA_CG_INTERFACE_COST=0` rejects explicit new-price calls;
`TILEMEGA_CG_INTERFACE_DP=0` retains the old DP default. Neither switch
changes the CUDA event protocol. L2 candidate-specific event DP remains
explicitly rejected and is not established by this L1+interface probe.
