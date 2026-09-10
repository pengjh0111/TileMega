# Real Task Prices and Cubic Stop

Verified CPU evidence, BF16, target sm_89, fixed past=3. This is not a GPU
experiment or a completed design (a). `verify_symbolic_prices.py` archives
the exact executable hash, source diff and each stdout/stderr.

| Control | gqa2 | mha4 |
|---|---:|---:|
| First GEMM, splits 1/2/4/8/16, seq=1..512 | 2560/2560 | 2560/2560 |
| Every scalar stage, seq=1..16 | 256/256 | 512/512 |
| Full CG-interface symbolic DP | degree stop | degree stop |
| ISL references after every process | 0 | 0 |

`lib/Solver/SymbolicTaskCost.cpp` constructs nominal collective work/rate
lanes, exact floor wave partitions, scalar physical R/W lanes, per-task and
per-wave maxima, combine prices and CG-interface prices. Calibration's
binary64 rates are lifted to exact rational constants. This is algebraic
arithmetic, not a promise of identical IEEE operation order: the probe
prints the maximum rounding difference separately. No concrete seq samples
are used to fit a polynomial. The seq loops only validate the finished QP.
These checks do not replace A6's bit-pattern gate.

`lib/Solver/SymbolicChainDP.cpp` retains the CG factor frontier and the outer
residency pin. Each transition minimizes polynomial prices by exact roots
on explicit feasibility intervals. It currently covers the unchunked L1
model only; L2, new fusion and placement choices are explicitly rejected,
not priced as zero. It is not the default or a replacement for (b).

## Observed Stop

Both models reject interface 6->8 (KVAppend -> attention) before completing
the first DP solve. The exact incidence work is

```text
repeated = 512*seq^2 - 4*seq
volume = 1
bytes = 2*repeated
```

The measured cache service coefficient has a nonzero linear term on this
domain. `bytes * ((1-miss)/l2_gbps + miss/dram_gbps)` therefore contains a
nonzero cubic term. Full rational coefficients are in `gqa2_dp.txt` and
`mha4_dp.txt`; the cubic signs differ because their measured-curve segments
have different slopes. Neither coefficient is numerical noise or a fit to
seq samples.

This disproves the assumption that replacing the CDF makes every complete
cost comparison quadratic. A QP remains a QP under multiplication, but its
degree can increase. The user-required degree>2 guard fires, exit=2,
`ISL_CONTEXT remaining=0`. B3.2 stops here. Do not disable the CG-interface
term, freeze the cache factor or sample seq to evade this gate. The required
S=1..16 symbolic-vs-finite DP choice gate has NOT passed; (b) is NOT retired.
B1 is independent and continues.

## Corrections During Implementation

1. Summing a QP with zero coordinate axes previously called barvinok's
   domain-sum routine, which rejects scalar input. It now preserves the
   scalar unchanged; constant zero/nonzero and parameterized tests cover it.
2. This bundled isl aligns parameters in addition but not multiplication.
   Constant x parameterized work initially failed a space assertion. Named
   parameter alignment now precedes multiplication. Tests cover constants
   and reversed parameter order, and require zero reference delta.
3. First extraction used the domain space instead of the polynomial's
   complete space and returned zero. The algebra test caught it before
   acceptance; extraction now uses the complete space. No expected price
   was changed to accommodate it.
