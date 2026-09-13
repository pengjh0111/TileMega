# sm_120 SIMULATOR / PLACE_EFT, 2026-09-13

Verified from the serial session and saved artifacts: both CPU-side
SELF_CHECK commands passed. The SIMULATOR sweep started at 12:04:43 and
PLACE_EFT started at 12:04:56 (Asia/Shanghai). The session stopped during
PLACE_EFT; no experiment processes remained when checked for this report.
The base revision of this evidence commit is
`9eb072abb0bdbc5cb01f515ab974426cc316233b`.

Verified command environment from the session: `REALWIDTH=0`;
SIMULATOR reports 4096 rounds per cell and device NVIDIA GeForce RTX 5090,
170 SMs, 4080 resident CTAs. The disk check before PLACE_EFT reported
16666 MiB free, above its 4096 MiB requirement. This failure was not the
previous round's disk-exhaustion failure.

## SIMULATOR: PASS

Verified: RMW and load sweeps each contain 48 cells (24 each at backoff
64 and 0), with 4096 rounds per cell and zero recorded inversions.
The three saved checksums match `contention.tsv`, `contention_load.tsv`,
and `hop_ns.tsv`.

The default RMW/backoff-64 trimmed-mean fit is:

```text
hop_ns(N,R) = 448.277066 + 1.134462*log2(1+N/R) - 0.436421*log2(R)
```

Verified fit report: RMS residual 15.0 ns, maximum residual 42.6 ns,
reduced chi-square 1936.5. Sweep completion/PASS is not a claim that this
fit is statistically adequate or that a scheduling improvement was proven.
The generated coefficients remain experiment-local; no sm_89 calibration
or repository default is replaced.

Evidence: [RMW sweep](SIMULATOR/raw_sm120/contention.tsv),
[load sweep](SIMULATOR/raw_sm120/contention_load.tsv),
[fit](SIMULATOR/raw_sm120/hop_fit.txt),
[coefficients](SIMULATOR/raw_sm120/hop_ns.tsv), and
[checksums](SIMULATOR/raw_sm120/sha256.txt).

## PLACE_EFT: FAIL before the fresh-process gates

Verified: the four reference legacy-source provenance comparisons are
byte-identical. All 24 reference-arm executables were compiled. During
the initial placement-statistics pass, the gqa2/seq=4 legacy, balanced,
and rotate controls reported `RESULT status=PASS`. These are single
invocations, not the planned 50-fresh-process correctness gates.

The next invocation, gqa2/seq=4 EFT, stopped with:

```text
the eft plan table is pinned to nodes=200 seq=4 past=3 grid=256, not nodes=200 seq=4 past=3 grid=340
```

Verified: the copied sm_89 plan expects grid 256, while this sm_120
execution selected grid 340. The runner's pre-launch plan guard refused
the mismatch. The schedule cannot be transferred unchanged under the
current device-dependent grid selection; no configuration was changed to
force the expected value or bypass this guard.

The 50-fresh-process correctness matrix, paired timing, sync decomposition,
and summary were not reached. REALWIDTH was disabled, so no real-width
success is claimed. Compiled plans are not evidence that those plans ran.

Evidence: [provenance](PLACE_EFT/raw_sm120/provenance.tsv),
[EFT rejection](PLACE_EFT/raw_sm120/log/stats_gqa2_s4_eft.out),
[partial control statistics](PLACE_EFT/raw_sm120/place_stats.txt),
[status](PLACE_EFT/raw_sm120/status.txt), and nvcc logs under
`PLACE_EFT/raw_sm120/log/`.

## Artifact scope

Verified: outputs are confined to the two `raw_sm120/` trees; existing
sm_89 sweeps, plans, and results are unchanged by this evidence commit.
The [serial log](SIMULATOR/raw_sm120/serial_session.log) is preserved.
Compiler executables are excluded from Git, while source plans, compiler
logs, measured tables, and failure evidence are retained. No tests were
rerun or implementation changes made while preparing this commit.
