# Serving attention body-fit units

The interrupted Qwen3 decode B=1 search at `/root/r10_work/plans_superseded_attention_price/qwen3_decode_B1/plan.so.search.tsv` priced its seed at 762058433.951 ns. Its first fused-attention block had `compute_ns=6735205.881` in `plan.so.m1A.flow_parts.tsv`; the raw `serving_task_bodies.tsv` calibration for D=128 and 64 positions is 27194 ns. The input to the fitted `flop_ns` coefficient had been expanded semantic work rather than the per-task QK/PV operations measured by the calibrator.

After fixing the input units in `TaskPriceParts.cpp`, the same explicit six-class configuration (the JSON case is `/root/r10_work/price_audit/qwen3_seed.json`) scores 9011250.207 ns. The raw fixed-case result is `/root/r10_work/price_audit/qwen3_seed.cu.search.tsv`. The earlier matrix and pruning processes were stopped; their files are retained under `/root/r10_work/plans_superseded_attention_price` and this directory's `superseded_plans`. None of those scores are used as R10 performance results.

The candidate timing guard also rejected a run with only its own PID visible because NVML's device-wide and per-process memory counters differed by 615 MiB after allocations. `guard.jsonl` remains under the archived Llama decode B=2 candidate. The guard now follows the specified process-list test and records the memory delta only for diagnosis.
