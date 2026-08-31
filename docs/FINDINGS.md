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
  but passed 100/100 runs. A barrier reached after every thread exits its spin
  is not, by itself, mismatched.
- Evidence: `docs/experiments/V_A/negative_controls.txt`.
- Skeleton impact: §8.1 should motivate single-thread polling primarily by
  reduced contention and simpler control flow. A concrete illegal barrier
  placement is needed to demonstrate silent barrier mismatch.
- Confidence: high for this kernel and GPU; medium as a general statement.

## F-3 — Fence-removal passing is not evidence that fences are redundant

- Finding: the no-fence control passed 100/100 runs on SM 8.9. This is an
  observed implementation outcome, not a portable memory-ordering guarantee.
- Evidence: `docs/experiments/V_A/negative_controls.txt`.
- Skeleton impact: retain §8.5 and label this control as an inability to provoke
  the forbidden outcome on the tested GPU.
- Confidence: high.
