# T4.4: original 4×4096 scene still fails with FP32 partials

✅ The requested original blocking scene was rerun on RTX 4090: four layers,
hidden=4096, intermediate=14336, heads=32, KV heads=8, seq=4/past=3,
BF16, unchanged seed and CPU-default golden. This is **one failed diagnostic
process**, not a 50-process acceptance result.

| Candidate | Result after FP32 partials | mismatch | max_abs | max_rel |
|---|---|---:|---:|---:|
| split8, 32×128×32, stages3 | 0/1; stopped | 1 | .046875 | 3906.25 |
| split16, 32×256×16, stages3 | Compiled, not run after stop | — | — | — |

`condition9/k8_r0.txt` confirms `fp32=1`, element_bytes=4 and
total partial bytes=22020096. L0.5/L1/L2 share hash `1ebe85ae6a9095f1`,
and inter-level/second-iteration comparisons report zero. The sole
elementwise failure is in final hidden (output0), not KV outputs.
There was no CUDA launch failure. One process does not establish race freedom.

The runner stopped immediately on this failure, before split16 or another
round. It did not relax `.016 + .016 * |expected|`, change golden thread
count, disable large split, or continue toward a claimed 100/100.
**Phase 5 condition 9 remains open.** The earlier two-reference-model
500/500 results on each architecture do not transfer to this wider scene.

## Reproduction and resources

`run_condition9.py:17` validates the exact original fixture shape, keeps
both old leader plans, enables FP32 partials explicitly, and alternates
split state order within each round. Logs, fixture manifest, commands and
both complete ptxas logs are in `condition9/`. Regenerated binary/weights
are ignored. Warmup=5/repeat=11; single failed-process timing is not a
comparative performance result.

```sh
python3 docs/experiments/REALMODEL/run_condition9.py
```

Observed split8 L2 resources: block128, reg168, actual dynamic shared30720,
static shared0, chosen ctas_per_sm2, grid256. The log separately reports
L1's 2 and L2's 3 occupancy-query CTAs; shared grid uses the common bound.
Schedule: 13552 task refs, 28884 waits, 26480 lifted polls, 6764 waiting
tasks, max_worker_span224, resident_limit256, current I3 pass and
overresident rejection. Do not reuse the old reg212 occupancy assertion
for this different tile configuration.

## Numerical feasibility: design required before unrestricted DP selection

⚠️ A validated feasibility model is **not implemented**. The residual
failure after FP32 partials disproves the claim that that repair alone
closes the original condition, but does not identify the remaining cause
or establish that error is monotone in split. No global split ban follows.

The intended interface is a pre-cost legality result for each candidate,
analogous to AlignmentPropagation, with `{proven_feasible, infeasible,
unresolved}` plus a reason and the input/parameter assumptions. A proof
must cover the actual mixed tolerance at each output. Shape and split alone
cannot do this: cancellation, magnitudes, operator conditioning and the
comparison reference's reduction order are data-dependent.

For a bounded numerical domain, propagate magnitude/error bounds through
FP32 partial dot products, their FP32 combination, the single BF16 output
rounding and the preserved residual rounding boundary. Standard gamma_n
accumulation bounds are only one input; RMSNorm/attention and layer
composition also need conditioning bounds. An error bound against real
arithmetic is not automatically a bound against PyTorch, so include the
reference error bound as well. Feed the resulting legality predicate to
candidate enumeration **before** any cost comparison, never as a soft
penalty. Missing input bounds must remain unresolved with an explicit
reason, not be silently accepted or replaced by a conservative runtime.

This design requires actual numerical-domain assumptions and validation;
one recorded pass is not a proof and a blacklist of this one shape would
not meet it. Next diagnosis must localize the failing element and compare
the same prefix/reference arithmetic before selecting such bounds. No
criterion change or unvalidated feasibility filter was installed here.
