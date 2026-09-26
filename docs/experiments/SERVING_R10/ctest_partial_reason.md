The first sequential full CTest attempt was interrupted after test 32, with all completed tests passing. Its command held the exclusive GPU timing lock across CPU-only tests; test 33 (`pipeline_sigma`) previously required about four minutes and would delay top-3 GPU candidate timing. The remaining CPU tests were restarted without that lock in two disjoint groups (33–54 and 66–80). Serving GPU tests (55–65) run separately under the lock. The partial output is retained as evidence of tests 1–32, not represented as a full-suite pass.

The disjoint follow-up runs completed: tests 33–54 passed 22/22
(`ctest_cpu_33_54.txt`), tests 55–65 passed 11/11
(`ctest_gpu_55_65.txt`), and tests 66–80 passed 15/15
(`ctest_cpu_66_80.txt`). Together with the 32 completed tests in the first
run, all 80 registered CTest cases passed. After tightening the attention
cache-write check to one BF16 ULP, test 64 passed again
(`ctest_attention_decode_matrix_strict.txt`).
