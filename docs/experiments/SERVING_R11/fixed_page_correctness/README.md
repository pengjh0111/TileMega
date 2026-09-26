# Full-request correctness of fixed-geometry paged plans

These are the four user-selected endpoint cells, using the already generated R10 geometry with the R11 8 KiB page transport, not the final SV-18 winner. `check_modes` launched 1024 steps in L1 and L2 on the same plan instance and checked every token; `hf_check` performed teacher-forced near-tie evaluation on the L2 output. Free-running HF divergence was skipped in this diagnostic check. The page binaries previously passed 50 fresh-process 64-step comparisons per cell. Raw tokens, guard observations and command logs are archived in `raw.tar.xz` (SHA256 `0dab33768621bf5c6221cad15f595cfe7f20aa2a7f8550e1598c684836d209b2`).

All four cells passed C-1 and had zero L1/L2 token mismatches (34,816 tokens total). These results do not close EV-2 because final solver-selected plans and fused/unfused comparisons are not yet available.
