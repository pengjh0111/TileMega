# D-a: the whole Llama decoder, end to end

Sixteen layers, the token embedding, both per-layer normalizations, the rotary
phase table, the final normalization and the language-model head, imported from
one `torch.export` program and solved, generated, built and run as one
megakernel. Nothing is cut out and nothing is passed in as a pre-computed
tensor. This is the first round in which the whole decoder was solved at all:
before the plan hoist (F-230) the outer search did not reach its first row.

## What was run

| | |
|---|---|
| export | `/root/r7_work/llama/exported_program.pt2`, public Llama-3.2-1B config, seeded random weights, golden on CPU in bf16 |
| dims | seq 4, past 3; runtime variant interval `[1,4]` |
| solve | `--search-capacity 12`, evaluated 12, deferred 69, 5167.2 s, head `f6b00ac1` (`solve.json`) |
| chosen | `32x16x64s2 split4 kappa1`, residency 4, `eft`, grid 512, floor 5.23145e6 ns, predicted 6.35483e6 ns |
| generated | 388 tasks, 467 couplings, 243 stages, 467 symbolic windows, 0 fallback |
| built | `build/selected.json`, `MIDPOINT_REFINE=1`, 94 registers, 16384 B task smem, 4 CTAs/SM |
| ran | 50 fresh processes, `TILEMEGA_WARMUP=0 TILEMEGA_REPEAT=1` (`correctness/r*.log`) |

## Result: 0/50 PASS, and the reason is not the megakernel

Verified, all 50 processes:

- `E2E_HASH l05=l1=l2=735ddfc6d445292e` in every one of the 50. The task-by-task
  launch, the resident kernel and the L2 kernel produce bit-identical output,
  and so do all 50 processes: `l1_vs_l05_mismatch=0` and `l2_vs_l1_mismatch=0`
  everywhere, and `E2E_ITER l2_iter1_vs_iter0_mismatch=0`. On the whole model,
  W=1 FIFO, the event protocol and the solved slot order are exact at 50/50.
- `l05_vs_l0_mismatch=147` in every one of the 50 — the same 147 elements, the
  same `max_abs=0.03125`. The comparison that fails is against the CPU golden,
  and it already fails at L0.5, which launches one task at a time and uses no
  event, no worker queue and no placement.

So `RESULT status=MISMATCH` is an arithmetic difference against the golden, not
a synchronization failure. It is recorded as a failed gate all the same: D-a
asks for 50/50 against the golden and this is 0/50. The harness bound is
`1.6e-2 + 1.6e-2*|expected|` for bf16 and is not moved to fit this run.

## Where the 147 elements are

146 of them are in output 0, the 4x128256 logits; the other is one element of
output 25 (buffer 301), 0.01715 against a bound of 0.01640. Every one of the 32
KV-cache outputs is otherwise within bound. `diff_dump.log` prints the first 25
offending elements with both values (`TILEMEGA_DIFF_DUMP=25`).

`head_cancellation.py` prices the head for each offending logit from the dumped
buffers (`head_cancellation.txt`): verified,

- recomputing a logit in double precision from *this run's own* hidden state and
  the head weights reproduces the run's bf16 logit to ~1e-3 relative
  (-0.113550 recomputed vs -0.113770 stored), so the GEMM is right for the input
  it was given;
- the head sums 2048 products whose absolute mass is ~18 into a result of
  magnitude ~0.1-0.3: cancellation of 48.6 to 5.6e4, median 284 over the 146,
  against a median of 43 over 64 randomly drawn tokens. The offending logits are
  the strongly cancelling ones;
- 18 * 2^-9 = 0.035 is the scale a one-ulp bf16 difference in the hidden state
  reaches after that cancellation, and 0.029 is the largest difference observed.

Inferred, not observed: the hidden state entering the head differs from the CPU
golden's by about one bf16 ulp per element. The golden's hidden state is not an
output of the exported program, so this round cannot measure it, only the logits
it produces. What is verified is that the amplification needed to turn a
last-place bf16 difference into 0.029 is present in this operator and that no
other output in the model has it.

## What this does not say

- It does not say the model is numerically correct: 146 logits out of 513024
  exceed the harness bound and a token argmax could differ at those positions.
- It does not say the bound is wrong. Fixing the comparison is not in R7's
  scope; what R7 records is the difference.
- It does not measure a speedup. `E2E_TIME` is reported per round
  (l2 41.049-41.128 ms over the 50, l05 54.4 ms, l1 49.4 ms) but the timing
  gates are D-b and D-c, not this one.
