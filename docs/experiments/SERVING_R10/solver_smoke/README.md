# Restricted serving solve smoke (development evidence)

This is **not** EV-1 or a gate result. It fixes GEMM geometry to
`16x128x128s2` and searches split-K, κ, and residency for Llama decode B=1.
The purpose is to check the complete solve → materialize → real residency
query → compile → top-candidate L1/L2 measurement → winner-manifest path.
The full legal domain, attention coordinates, other batch sizes, and Qwen3
were not evaluated here.

The command ran in a fresh process on sm_89 with
`/root/r10_work/target_r10_inflight.json`, itself copied from
`../calibration/target.json`. The restriction file was
`/root/r10_work/serving_smoke_domain.json`:

```json
{"geometries":[{"tile_m":16,"tile_n":128,"tile_k":128,"stages":2}]}
```

```sh
./build-portable/tools/tilemega-compile \
  /root/r10_work/llama_decode_seed.so.export.json \
  /root/r10_work/serve_solve_measured_smoke.so \
  --serving decode --emit serving --batch 1 --past-range 64:1086 \
  --capacity 1088 --solver skeleton \
  --solve /root/r10_work/target_r10_inflight.json \
  --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv \
  --variant-cache /root/r9_work/variant_resources \
  --search-domain /root/r10_work/serving_smoke_domain.json \
  --search-passes 1 --top-m 1 \
  --measure-cmd 'PYTHONPATH=/root/TileMega/python /root/venv_vllm/bin/python -m tilemega.serving.measure_candidate --model /root/models/llama3_2_1b'
```

The measured candidate was L1 6.25847 ms and L2 6.34479 ms at past=575
(32 timed launches each, random finite resident data); the CLI selected L1.
The real occupancy query returned one resident CTA per SM. The final `.so`
had zero FP64 SASS instructions in its L1/L2 kernels. A separate fresh
process used real Llama weights with the prior seed prefill plan to generate
32 tokens through this solved decode plan; the 32 tokens matched the earlier
HF-checked seed output exactly. That short check does not establish the
1024-token C-1/C-2 gates.

Raw search, timing, materialization, measurement, ptxas, SASS audit, and
token-output files are adjacent. The binary, exported program, checkpoint,
and variant cache remain outside the repository under `/root/r10_work` and
`/root/models`.
