# T4.5 FP32 GPU partial-storage regression

✅ Freshly exported FP32 models, seq=128/past=3; two models × split={1,16}
× TILEMEGA_FP32_PARTIALS={0,1} × 50 fresh processes = **400/400 PASS**.
Every state pair has identical output hashes, as do L0.5/L1/L2 within each
run. No numerical criterion changed. This tests FP32 as a zero-regression
control; it is not a BF16 ranking or depth-accuracy result.

| Model | split | Off | On | max_abs vs PyTorch | Partial bytes, both states |
|---|---:|---:|---:|---:|---:|
| gqa2 | 1 | 50/50 | 50/50 | 2.2649765e−6 | 0 |
| gqa2 | 16 | 50/50 | 50/50 | 1.9073486e−6 | 67108864 |
| mha4 | 1 | 50/50 | 50/50 | 2.3841858e−6 | 0 |
| mha4 | 16 | 50/50 | 50/50 | 2.8610229e−6 | 150994944 |

Code: `run_splitk.py:24` adds dtype/split selection without changing default
BF16 sweep scope; `run_splitk.py:110` rotates the full build-state matrix,
records execution_index/dtype/hash, checks inter-level hashes, and requires
both FP32 states to pass. For BF16 the old-partial numerical negative control
remains permitted; the new path is still required to pass.

Reproduction:

```
python3 docs/experiments/BF16/run_splitk.py \
  --out docs/experiments/BF16/raw_splitk_f32 --dtype f32 --splits 1 16 \
  --jobs 2 --phases build
python3 docs/experiments/BF16/run_splitk.py \
  --out docs/experiments/BF16/raw_splitk_f32 --dtype f32 --splits 1 16 \
  --phases correctness
```

Evidence: `raw_splitk_f32/correctness.tsv`, all process logs, all eight ptxas
logs, plan files and header-hash manifest. Timing columns are diagnostic only;
no latency improvement or isolated performance attribution is claimed.

⚠️ This closes the requested FP32 GPU regression for these split endpoints,
not every possible configuration. The separate measured FP32-partial combine
coefficient remains outstanding; its existing traffic approximation was not
rebranded as a new measurement.
