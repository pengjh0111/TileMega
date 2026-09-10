# Mixed Task Price Checks

Verified CPU analysis, BF16 model inputs, seq=4/past=3. No new GPU timing or
race claim. `verify_prices.py` freezes the tool hash and reads registers from
the archived compiled attention-plan receipts, not a guessed resource value.

| Pair | Separate task ns | Fused task ns | Global bytes before/after |
|---|---:|---:|---:|
| gqa2 RoPE -> KVAppend | 1720.5076632 | 1438.6544790 | 12288 / 8192 |
| gqa2 existing GEMM -> add | 18621.0466716 | 17921.4449031 | 557056 / 548864 |

All gqa2/mha4 candidate rows are in their TSVs. The GEMM/add row is an
existing runtime epilogue, NOT a newly obtained runtime event discount.
The logical RoPE/KV row owns 1024 tasks, not the later runtime CTA grouping.
Neither row is a prediction of a complete newly compiled fused megakernel.

`PriceFusionTasks` in `lib/Solver/FusionResources.cpp` uses the producer's
exact inverse fiber and `CostModel::TaskInstanceNs` for each phase. Internal
writes and reads move to shared; externally needed writes remain global.
Each phase retains its own arithmetic pipe. The consumer coordinate set
determines fused waves. Scratch is max(phase scratch)+intermediate tile;
the reported 6-byte RoPE/KV intermediate is NOT the whole-kernel TaskSmem.

The nonunit-fanout control narrows consumer rows without changing producer
tiles. Four producers feed sixteen consumers: twelve producer reexecutions
and 219645.236054 ns of separately reported recompute work. Global traffic
rises to 2170880 bytes. This diagnostic work is not added twice: repeated
phase instances already occur in the fused wave sum. More work can still
fit one device wave, so no assertion forces latency to grow with fanout.

Verified regressions:

- 4308 GEMM configuration groups, BF16 and FP32 x both models; both stage
  entry points preserve bit patterns. Complete PASS lines and zero ISL
  references are in `gemm_gate.txt`.
- `../scalar_controls_complete/summary.json`: 2720 attention-phase checks,
  fourteen scalar tables byte-identical, two historical invalid-domain
  rejections retained. This rerun is distinct from the older A7 receipt.
- Mixed-price rejection paths: bad inverse fanout, mismatched threads and
  zero residency, 9/15 checks respectively, all zero reference delta.

The earlier `../task_prices/` and `../scalar_controls/` runs were interrupted
by the connection/tool restart. Their partial output is retained, never
counted as a completed gate. Streaming the long GEMM gate now leaves a
receipt per configuration; status still requires normal process completion.

Remaining: candidate-specific runtime event delta, interval optimization,
compiled fused resources and two genuine GPU fusion arms. Task-only
numbers must not stand in for those checks.
