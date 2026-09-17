# Signed intervention differences and varying controls

Verified: the real-s128 process logs contain two broad timing bands in L0.5
and L1, including the controls that the L2 fence probe does not modify.
`kernel_identity.tsv` compares exact disassembled function bodies: removing
the notify fence changes only L2; L1 and the other kernels remain identical.
Removing grid synchronization changes only L1; L2 remains identical.

Some same-round `full.L1 - l1nosync.L1` differences are consequently negative.
`barrier_pairs.tsv` retains every completed pair, its sign and the separation
between process start times. `unchanged_controls.tsv` retains the three
timings from every process, without a fast/slow classification used for
filtering. `audit.log` records the exact coverage at the time of the audit;
the audit is rerun after all 1250 processes complete.

The completed run contains 44/250 nonpositive pairs, all strictly negative,
and no zero denominator. Pair start-time separation has median 59.017 s and
maximum 1213.832 s. These are observed separations, not an identified cause
of the control timing bands.

The analyzer originally assumed every paired barrier difference was positive.
That assumption is not a gate in the R6 prompt. The corrected analyzer keeps
the same subtraction and per-pair ratio, including negative values, and reports
nonpositive/zero denominator counts. If a denominator is zero, the complete
ratio statistic is undefined; that pair is not dropped. All 25 paired rounds
still contribute to every defined statistic. No raw log is changed.

Inference is limited: these samples establish the observed intervention
differences, not isolated physical fence/barrier service costs. The underlying
cause of the control timing bands is not established by the current logs.
One GPU-process snapshot contained only the benchmark process; it cannot
exclude unobserved interference during earlier samples.

The next diagnostic should record clocks/power and allocation information
per process. `REBASE/run.py` currently rotates the flattened fifty-arm list;
a configuration can straddle a round boundary, separating paired controls by
nearly an entire round. A future campaign should rotate the five arms within
each configuration and independently rotate configuration order. That change
is not applied retroactively to this campaign, and does not license filtering
the current slow or negative samples.

Recompute from archived SASS and logs:

```sh
python3 docs/experiments/REBASE/probe_audit.py
```

Add `--dump-sass` only after rebuilding the recorded sm_89 binaries locally.
