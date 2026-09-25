# Restricted decode interval check

Verified on sm_89 with the R10 measured target at
`/root/r10_work/target_r10_inflight.json`. The five `*.search.tsv` files are
raw `tilemega-compile` search logs. They use real exported serving graphs and a
restricted GEMM domain (`/root/r10_work/serving_smoke_domain.json`), one search
pass, and `--flow-search-only 1`; they are not full-domain performance results.

The command template for each point is:

```sh
./build-portable/tools/tilemega-compile /root/r10_work/llama_decode_seed.so.export.json /root/r10_work/flow_b16_price_p64.cu --serving decode --emit serving --batch 16 --past-range 64:64 --capacity 1088 --solver skeleton --solve /root/r10_work/target_r10_inflight.json --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv --variant-cache /root/r9_work/variant_resources --search-domain /root/r10_work/serving_smoke_domain.json --search-passes 1 --flow-search-only 1
```

The other Llama endpoints substitute `575:575`, `1086:1086`, or `64:1086`;
the Qwen boundary check uses `/root/r10_work/qwen3_decode_seed.so.export.json`.
Run `python3 check.py` to recompute the number of common finite candidates,
the number whose price changes with past, and the Simpson identity error.

The boundary assertion previously occurred in `PriceBoundaryPieces` when
Barvinok summed the fused attention task's causal, piecewise access polynomial
at B=16 and past=64. The serving attention space is now priced by exact task
coordinates. Empty KV blocks pay only their early-return fixed term; the
mainloop scales with the number of 64-position KV tiles that execute.
