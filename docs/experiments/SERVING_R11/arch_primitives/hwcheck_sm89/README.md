# Native sm_89 hardware check

`python -m tilemega doctor --hwcheck --config configs/e2e/llama_b1.json`
passed on the RTX 4090. It used one explicit 16×128×128×2 Llama B=1
configuration per phase to exercise ordinary import, solver materialization,
codegen, native nvcc compilation, and the serving runtime. This is an
architecture smoke check, not the EV-2 configuration search.

- Five CUDA task/page tests passed (`unit_command.json`).
- The prefill and paged decode builds took 87.998 s and 94.682 s respectively
  (`*_build_command.json`). Both build commands specify `sm_89`.
- `sass.json` reports zero FP64 instructions across both L1/L2 kernels in
  both `.so` files. Binary SHA256 values: prefill
  `3ceca5940e0abf6714317c6fc85d09776f5668698474c67dc970ec1588d334c3`,
  decode `b4827cc0870c518834f0ac64505e21e08f179a41b86d53d92829436a554aa73e`.
- `mode_check.json` reports 64/64 identical tokens for L1/L2 on the same
  plan instances. The token lists and process guard are retained.
- `cache_first.json` records first-build misses for both smoke plans;
  `cache_repeat.json` records all calibration, export and smoke-plan hits.
  The fully cached command took 30.304 s, including repeat CUDA tests,
  SASS audit and 64-step generation (`cached_wall_seconds.txt`).
- PDL is inapplicable on sm_89. sm_90/sm_120 transport and PDL still require
  execution on their respective hardware; the cross-architecture compilation
  and SASS checks are separate evidence in `../`.

The first attempt to use full coordinate descent inside `--hwcheck` was
stopped after the prefill search exceeded ten minutes. The smoke path now
uses the existing single-case solver entry and a hash-validated plan cache.
An initial complete smoke run exposed and fixed two CLI errors: `doctor()`
returns a wrapped device object, and the nvcc command quotes its `-arch`
argument. The successful run above was after both fixes.
