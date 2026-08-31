# Findings

## F-1 — CTA-wide publication needs a CTA-wide release sequence

- Finding: when all producer threads write the tile but only thread 0 signals,
  a fence performed only by thread 0 does not order writes performed by the
  other threads. Each writer fences before a CTA barrier; thread 0 signals
  after that barrier.
- Evidence: `docs/experiments/V_A/event_sync.cu`; the compliant path passed
  1,250/1,250 runs and the no-barrier control mismatched 100/100 runs across
  the two fill modes.
- Skeleton impact: §8.5 should distinguish single-thread production from
  cooperative CTA production and show the required CTA convergence.
- Confidence: high.

## F-2 — All-thread polling is not inherently a correctness negative control

- Finding: independent polling by every thread generated excess atomic traffic
  but passed 100/100 correctness runs. In the supplement its kernel-time ratio
  versus single-thread polling was 0.996× / 0.971× / 0.946× at grid 64/128/256,
  so this short-wait workload did not measure the expected contention cost.
- Finding: moving `__syncthreads()` into the divergent polling loop is a
  different and genuinely dangerous pattern: it hung 50/50 runs at each of
  grid 64, 128, and 256. PC samples changed without completion, consistent
  with a collective stall/livelock.
- Evidence: `docs/experiments/V_A/negative_controls.txt`,
  `supplement_polling_timing_raw.txt`, `supplement_barrier_spin_raw.txt`, and
  `supplement_hang_probe.txt`.
- Skeleton impact: §8.1 should separate two rules: single-thread polling avoids
  unnecessary atomic traffic; collective barriers must never occur inside a
  thread-divergent spin loop. The former cost was not measurable here, while
  the latter received direct negative-control evidence.
- Confidence: high for correctness on this kernel/GPU; low for the measured
  performance cost outside this short-wait workload.

## F-3 — Fence-removal passing is not evidence that fences are redundant

- Finding: the original no-fence control used fresh addresses each iteration,
  excluding a stale L1 hit. The combined hostile variant (reuse + no backoff +
  8,192-float tile) still passed 150/150, but isolating address reuse at the
  original tile size failed 150/150. Every sampled mismatch was the exact
  previous-iteration value. Removing backoff alone and increasing tile alone
  each passed 150/150; the fenced reuse control passed 150/150.
- Evidence: `docs/experiments/V_A/supplement_hostile_raw.txt`,
  `supplement_hostile_isolation_raw.txt`, `supplement_reuse_correct_control_raw.txt`,
  and `supplement_mismatch_samples.txt`.
- Skeleton impact: retain §8.5. A fence negative control must reuse addresses;
  fresh per-iteration storage structurally suppresses stale-line failures.
  Hostile changes are not monotonic—larger writes changed timing enough to hide
  the reuse-only failure—so each stressor must also be tested independently.
- Confidence: high.

## F-4 — Occupancy capacity is not the same as a co-residency requirement

- Finding: cubin metadata gave 256 threads, 40 registers/thread, 0 static and
  32 bytes dynamic shared memory, hence 6 CTA/SM × 128 SM = 768 resident CTAs.
  Nevertheless the current circular workload passed 50/50 even at 1,536 CTAs.
- Evidence: `docs/experiments/V_A/supplement_occupancy.txt` and
  `supplement_residency_boundary_raw.txt`.
- Skeleton impact: §8.7 should retain the portable capacity formula, but only
  require `grid ≤ resident_limit` when analysis of the generated wait-for graph
  proves full-grid co-residency is necessary. Local successor dependencies can
  drain progressively above the capacity.
- Confidence: high.
