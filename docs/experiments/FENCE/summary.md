# R4 device fence pricing

✅ Verified: five arms x two placements x four cells x 25 rotating paired
fresh-process rounds = 1,000 processes in one recorded session. Full passes
200/200. Unsafe-arm numerical results are not acceptance evidence. Each log
has a companion JSON with binary digest, command, session, round and order.

All builds enable R3 B: calibrated wait policy, BARRIER_V2, EVENT_SOLO and
EVENT_RED_PUBLISH; C1/C2/C3 are absent. The original probe-only commit was
79fe7549. This completion commit precedes resumed C implementation.

Times are paired medians in microseconds; shares are medians of paired ratios
and therefore need not equal ratios of the displayed medians. Full percentile
bootstrap 95% intervals (10,000 paired resamples, seed 167) are in
`raw/measurements.tsv`, reproducible with `run.py summarize`.

| cell | placement | fence | notify | full-neither | barrier | fence/notify | fence/protocol |
|---|---:|---:|---:|---:|---:|---:|---:|
| gqa2 s4 | 0 | 15.360 | 35.840 | 69.632 | 26.848 | 0.419 | 0.215 |
| gqa2 s4 | 5 | 13.376 | 2.048 | 224.256 | 26.848 | 6.410 | 0.060 |
| gqa2 s128 | 0 | 22.528 | 33.664 | 77.824 | 32.576 | 0.664 | 0.286 |
| gqa2 s128 | 5 | 16.480 | 15.360 | 287.744 | 31.648 | 1.071 | 0.057 |
| mha4 s4 | 0 | 31.744 | 66.560 | 135.136 | 53.248 | 0.467 | 0.232 |
| mha4 s4 | 5 | 24.576 | 7.168 | 427.200 | 54.304 | 3.539 | 0.058 |
| mha4 s128 | 0 | 56.160 | 79.968 | 187.392 | 86.016 | 0.711 | 0.331 |
| mha4 s128 | 5 | 45.056 | 40.960 | 609.280 | 56.320 | 1.099 | 0.073 |

Definitions: fence = full - nofence; wait = full - nowait;
notify = nowait - neither; barrier = full.L1 - l1nosync.L1.
The fence probe removes only NotifyTask's top fence; it retains the barrier,
arrivals, publication and polling.

⚠️ Inferred from the paired probes: fence is substantial but is not the whole
remaining protocol cost. Its protocol share is 21–33% at placement 0 and
about 6–7% at placement 5. C1 should therefore aim at a fraction of the
full-minus-nofence effect, not assume all notification or wait cost disappears.
C2's independent publications and overlap with next-slot waits become the
next implementation focus; C3(a) tests shared completion state under W>1.

Fence/notify above one in rotate is retained, not clipped: notify is priced
with waiting disabled, whereas fence is priced with waiting enabled. They
are marginal costs in different execution contexts, not disjoint instruction
buckets. This interaction prevents attributing the observed fence delta only
to the nowait-minus-neither term and explains why protocol gains cannot be
transferred between placements. The prior R3 2.8%/4.8% improvement claims refer
to protocol variants, not to the total unsafe-probe cost measured here.
