# A6 GEMM component — not the complete A6 gate

✅ `tilemega-task-cost-gate` compares actual `double` object bits of
`CostModel::TaskCostNs` and the independent historical `GemmStageNs`.
It imports CG semantic payloads; it does not read a generated CUDA table as
the source of work. All 1077 archived ORACLE configurations are used for
each of BF16/FP32 × gqa2/mha4. Each production GEMM stage is checked at
seq={1,4,128,512,2048}, past=3, residency={1,2}:
4308/4308 groups, 904680/904680 stage prices, zero isl references.

The tested frozen executable SHA256 is
`bf05b0b3160e4af5e045aebd0991a4ae2d90413ff8f8b6f2d1afd7dfdf6595b9`.
`prices.tsv`, `status.txt`, and `build.txt` retain its evidence; run
`python3 docs/experiments/COST_MODEL/verify_gemm_price.py` for the independent
coverage audit. Configuration identity comes from the archived FP32 ORACLE
universe, but the BF16 passes import BF16 CG and use BF16 calibration.

Work is a QP until theta substitution. Nominal issued read work is used for
the calibrated collective lanes, while predicated physical R/W remains a
separate access-domain quantity (the user-approved A3 distinction).
Factorized setup multiplication and the tail-wave accumulation order are
preserved, rather than changing the expected result to accept FP64 drift.

⚠️ This entry currently rejects scalar traits. `Evaluate` and `ChainDP`
still use the historical split GEMM/non-GEMM path. This evidence does not
close A6: scalar per-task dependency structure, ownership projection,
non-GEMM difference explanations and ranking regression remain required.
Consequently it does not release B. `TILEMEGA_DERIVED_TASK_COST=0` rejects
the explicit experimental entry, leaving the historical evaluator intact.
