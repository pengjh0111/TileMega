# PG-1 stage-equivalence search

The paged GEMM executes a two-stage local mainloop and uses page ring depth for cross-task pipelining. This restricted-domain CPU-only run tests the stage-2 normalization and logs the removed equivalent variants. It evaluated 100 configurations with no errors; the best Level 1 score was 3.739 ms. These scores are not measured GPU times and do not establish an end-to-end winner. See `result.json` and the raw TSVs.
