# Single-block attention dependency regression

The original full Llama decode B=1 candidate used one KV block (`E_c=1088`).
Its attention body wrote `ctx`, while its semantic result named the split-KV
partial buffer. The coupling graph consequently omitted each `attention ->
o_proj` dependency. The resulting L2 token stream was not a valid performance
measurement. `lib/Frontend/ServingSemanticLifting.cpp` now gives the
single-block task the context result and omits the unused LSE write.

`test/unit/serving_model_plan_test.cpp` checks that each single-block
attention stage has an edge to the following projection. The fixed geometry
was then re-solved and compiled into
`/root/r10_work/plans/llama_decode_B1/repaired.so`; the previous placement
could not be reused because the new dependency changes queue legality.

The independent-process check runs each process with real Llama weights,
creates one plan instance, generates eight tokens in L1, then generates eight
tokens in L2 on that *same instance*. This also exercises the separate
per-mode event iteration counters. The raw JSONL has one result and PID per
fresh process. Run it with:

```sh
PYTHONPATH=python /root/venvs/tilemega-torch213-cu126/bin/python \
  docs/experiments/SERVING_R10/check_single_block.py \
  --model /root/models/llama3_2_1b \
  --prefill /root/r10_work/attention_prefill_top1.so \
  --decode /root/r10_work/plans/llama_decode_B1/repaired.so \
  --prompts docs/experiments/SERVING_R10/prompts/llama_ids.json \
  --out docs/experiments/SERVING_R10/single_block_dependency/fresh_processes.jsonl \
  --processes 50
```

The repaired fixed-config full-generation record is
`/root/r10_work/preliminary_e2e/llama_B1_repaired/`. It passed HF C-1 and
L1/L2 token equality. Its 5.068 s E2E median is preliminary: it is a fixed
configuration with a restricted prefill plan, not a full SV-17 winner.
