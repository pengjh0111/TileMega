# Cross-batch warm start smoke test

The Llama prefill B=2 search used the B=1 winner manifest as a second
coordinate-descent start. The raw search log records `WARM_START accepted`
with all six class geometries, kappa, residency, Ec, and Rq. This smoke test
used a restricted geometry domain and `--flow-search-only 1`; it does not
replace the full-domain B=2 search or GPU validation.

Command (from `/root/TileMega`):

```sh
build-portable/tools/tilemega-compile /root/r10_work/export/llama_prefill/bridge.json /root/r10_work/warm_b2_check.cu --serving prefill --batch 2 --capacity 1088 --solver skeleton --solve /root/r10_work/target_r10_inflight.json --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv --variant-cache /root/r9_work/variant_resources --emit serving --flow-search-only 1 --search-passes 1 --search-domain /root/r10_work/top1_shapes_domain.json --serving-warm-start /root/r10_work/attention_prefill_top1.so.plan.json
```
