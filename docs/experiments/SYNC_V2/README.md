# SYNC_V2: the wait policy (EX-E3 step 0)

Round two left 910 of the 1235 ns hop inside `__nanosleep(64)` (F-145).  This
directory measures what the wait costs under other policies and answers the two
questions R3 §5.1 attaches to the step: whether a pure spin slows a *computing*
worker sharing the SM, and whether sm_89 and sm_120 want the same answer.

Instruments: [`backoff.cu`](backoff.cu) sweeps `hop_ns(spin_iters, backoff_ns)`,
[`spin_interference.cu`](spin_interference.cu) puts a waiter and a computer on
one SM, [`backoff_fit.py`](backoff_fit.py) fits and picks, and
[`run_backoff.sh`](run_backoff.sh) builds and runs both.  All three are new
here: H1 freezes `SIMULATOR/`, so the round-two microbenchmark was derived from
rather than edited, and the padded row, sense-reversing barrier, per-round phase
dither and 0.1% trimmed mean carry over unchanged.

Verified on an RTX 4090 (sm_89, 128 SMs), 3 processes x 4096 rounds per cell,
8 policies x 7 (consumers, rows) cells, arm order rotated across the three
passes.  Zero inversions in every cell.

## The policy curve

`hop` is each consumer's own publish-to-observe time; `last` is the slowest
consumer of a row.  A task cannot start on somebody else's observation, so both
are reported.  `c0` is the constant of
`hop_ns(N, R) = c0 + c1*log2(1 + N/R) + c2*log2(R)`.

| arm | spin | backoff | mean ns | N=1 | N>=64 | hop c0 | last c0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `literal64` | 0 | 64 | 1232.3 | 1231.2 | 1233.5 | 1206.5 | 1202.1 |
| `spin` | 0 | 0 | 349.2 | 355.4 | 354.3 | 308.3 | 321.6 |
| `bo64` | 0 | 64 | 1213.9 | 1210.3 | 1215.8 | 1197.6 | 1190.5 |
| `bo16` | 0 | 16 | 1213.9 | 1209.5 | 1216.1 | 1199.7 | 1192.7 |
| `spin64_bo64` | 64 | 64 | 257.1 | 357.1 | 228.6 | 165.3 | 37.1 |
| `spin256_bo64` | 256 | 64 | 341.7 | 356.7 | 339.4 | 328.5 | 335.9 |
| `spin64_bo16` | 64 | 16 | 258.6 | 356.6 | 231.4 | 173.3 | 51.2 |
| `grow16_1024` | 64 | 16 | 256.9 | 348.9 | 230.0 | 167.8 | 42.0 |

Verified: `literal64` reproduces F-145.  It reaches the status quo through an
immediate `__nanosleep(64)` operand exactly as Codegen emits it, `bo64` reaches
the same policy through a register, and the two agree to 1.5%, so routing the
policy through `TargetSpec` does not itself move the number.

Verified: `bo16` and `bo64` are indistinguishable.  The argument to
`__nanosleep` is not what the backoff costs -- the sleep is quantized far above
either value, which is why cutting 64 to 16 buys nothing and only taking the
sleep off the common path does.

Verified: the arm order is not a confound.  Per-arm means across the three
rotations spread by at most 7.2 ns, against the 90 ns effect being ranked.

Inferred: the win is non-monotone in `spin_iters` (0, 64, 256 give 349, 257 and
342 ns pooled) because 64 polls run out roughly when the publish lands, so those
consumers are asleep -- and not hammering the row -- at the moment it matters,
while 256 polls never run out and behave as a pure spin.  At N=1 the ordering
disappears (all arms 350-365 ns), which is what a congestion explanation
predicts and a code-path explanation does not.  This is reasoned from the shape
of the sweep, not separately instrumented.

## Does a spinner slow a computer on the same SM

Two CTAs per SM, the low half of the grid computing and the high half waiting,
co-residency reconstructed offline from `%smid` rather than assumed.  That check
earned its place: even/odd assignment paired nothing at all (`paired_fraction`
0.0000), because the scheduler puts block *s* and block *s+128* on the same SM
and both have the same parity.

| compute | arm | p50 cycles | ratio to idle | paired | samples |
| --- | --- | ---: | ---: | ---: | ---: |
| fma | `idle` | 390190.0 | 1.0000 | 1.0000 | 5120 |
| fma | `spin` | 390189.0 | 1.0000 | 1.0000 | 5120 |
| fma | `backoff64` | 390190.0 | 1.0000 | 1.0000 | 5120 |
| mem | `idle` | 12580703.0 | 1.0000 | 1.0000 | 5120 |
| mem | `spin` | 12623825.0 | 1.0034 | 1.0000 | 5120 |
| mem | `backoff64` | 12628232.0 | 1.0038 | 1.0000 | 5120 |

Verified: no.  Against an issue-bound FMA computer the spinner costs 0.0000;
against a memory-bound one it costs 0.0034, and the backoff arm costs 0.0038 --
more than the spin -- against the idle arm's own round-to-round spread of
0.0303.  The cost of spinning is inside the noise of not spinning, and what
little there is does not separate spin from backoff.

The first pass measured only the FMA shape and reported exactly zero.  That
would have been an answer to the wrong question: §8.3's stated reason for the
backoff is that compact polling can saturate the *memory subsystem*, which an
FMA loop never touches.  The memory shape was added for that reason.

## The chosen policy

Rule, fixed before the numbers were read: smallest `c0`; if that policy carries
no backoff, keep it only when spinning costs a co-resident computer no more than
the idle arm's own spread.  The rule selected a policy that keeps a backoff, so
the interference result did not decide it.

```text
spin_iters = 64, backoff_ns = 64, backoff_grow = 1, backoff_cap_ns = 64
hop c0 1206.5 ns -> 165.3 ns (13.7% of the status quo)
```

These are the values in `configs/targets/sm_89.json`, in both `calibration.sync`
and `calibration_by_dtype.bf16.sync`.  Provenance is
[`backoff_policy.tsv`](backoff_policy.tsv), emitted by the fit.

sm_120 is not calibrated here.  H9 keeps this machine off Blackwell, and its hop
sits near a 400 ns floor with a backoff worth about 32 ns, so its optimum is a
different question and is expected to differ; `run_sm120.sh` recomputes the
sweep and emits its own policy file on the target machine.  Until it runs,
`configs/targets/sm_120.json` carries `wait_spin_iters = 0, wait_backoff_ns =
64, wait_backoff_grow = 1, wait_backoff_cap_ns = 64` -- the status quo, byte for
byte what Codegen emitted before this step existed, so the file parses to the
generated wait unchanged.  An earlier draft of this paragraph said sm_120
carried no wait keys at all; that was true when it was written and stopped being
true when every target config was given the block, and the values are the
status quo rather than a calibration.

Raw: [`backoff.tsv`](backoff.tsv) (pooled), `backoff_rot{0,1,2}.tsv` (per
rotation), [`spin_interference.tsv`](spin_interference.tsv),
[`backoff_fit.txt`](backoff_fit.txt), [`sha256.txt`](sha256.txt).


## E3-4: is one release fence by thread 0 enough?

[`litmus.cu`](litmus.cu) builds four release orders against one acquire, so a
difference between arms is attributable to the release and nothing else:
`per_writer` (§8.5 as built), `thread0_fence` (the candidate: barrier, then one
fence by thread 0, then publish), `no_barrier` (negative control) and `no_fence`
(sensitivity control).

The first scan was vacuous, for two reasons that are now measured rather than
argued.  The consumer's `__threadfence()` was unconditional in every arm, and it
invalidates the L1 the stale read needs (F-10): with it present, zero of nine
cells can detect a missing release (`SENSITIVE acquire=1 cells=0`).  A
`--no-acquire-fence` axis was added, and six cells become readable without it.
Separately, `no_barrier` had removed the release-side barrier while V_A's
removes the consumer-side one, so it raced only over intra-CTA write skew -- a
window the epoch poll dwarfs -- and passed 50/50 in all 18 cells.  Dropping both
barriers makes the arm literally barrier-free, and it now fails.

A cell is readable only where the sensitivity control actually failed and
`per_writer` held.  Six of eighteen qualify, all with the acquire fence dropped:

```text
acquire grid tile per_writer_pass no_fence_mismatch thread0_pass no_barrier_mismatch
0   64   1024   50   50   50   50
0   64   4096   50   50   50   50
0  128   1024   50   50   50   50
0  128   4096   50   50   50    0
0  256   1024   50   50   50   50
0  256   4096   50   50   50   50
```

Verified: `thread0_fence` held 50/50 in all 18 cells, including all six readable
ones -- 300 fresh processes, no mismatch.  Where the harness can tell the two
shapes apart, the candidate is indistinguishable from per-writer release.

One anomaly is recorded rather than reconciled: `no_barrier` passed 50/50 at
grid 128 / tile 4096 while mismatching 50/50 at tile 1024 and 49/50 at tile
16384 on the same grid.  That is non-monotonic in tile size, in one cell of
nine, and it is why the readable set is six and not seven.

The SASS census confirms each arm was built as intended -- `no_fence` loses the
release fence, `no_barrier` loses both barriers (`bar_sync` 6 -> 4):

```text
release  bar_sync  membar  atom_red  insns
3 (no_fence)      6   1   11   800
2 (no_barrier)    4   2   11   824
1 (thread0_fence) 6   2   11   800
0 (per_writer)    6   2   11   800
```

Per H3 this step only produces a conclusion: §8.5 stays exactly as written.  The
measurement is what E3-5 (async publish) needs before it can move that rule.

Raw: [`raw_litmus/litmus.tsv`](raw_litmus/litmus.tsv),
[`raw_litmus/readable.tsv`](raw_litmus/readable.tsv),
[`raw_litmus/census.tsv`](raw_litmus/census.tsv), and
[`raw_litmus/litmus_acquire_axis_only.tsv`](raw_litmus/litmus_acquire_axis_only.tsv)
(scan 1, preserved as the intermediate state it was).
