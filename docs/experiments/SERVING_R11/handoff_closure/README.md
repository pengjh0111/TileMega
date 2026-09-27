# R11 handoff integration checks

The three paired Llama B=16 decode measurements use the same selected
geometry and paged executor at past 575. Each arm warms eight steps and
times 32 steps in L1 and L2. `llama_b16_handoff_timing.tsv` contains the
per-round means. `llama_b16_1024_tokens.json` records the four complete
L1/L2 × event/handoff token hashes and mismatch counts. The event plan is
faster in both modes, so the final 2% measured gate should reject the
handoff for this geometry.

The restricted Llama B=1 end-to-end solver test used one GEMM geometry,
one search pass and top-M=1. It evaluated 27 configurations and priced
handoff mask 1 below mask 0, but the selected graph had no eligible
normalization→GEMM edge. The final compiler therefore emitted an event
plan. This mismatch is evidence that the current class-level handoff
score is an approximation, not a proof of a profitable runtime handoff.
The same test compiled and executed the actual resource probe (`resident=1`)
after the probe link fix. Its files are under
`/root/r11_work/handoff_try/restricted_B1.so*`.

The access verifier previously visited all spaces while proving one edge.
Generated split-K combine spaces have no standalone L-sem, so this
rejected unrelated eligible edges. The verifier now visits lifted spaces
and refuses recompute when an unlifted predecessor may supply the input.
On the restricted Llama graph, the post-fix audit found two generic
recompute edges and 18 last-arriver edges; none of the two recompute edges
is a norm→GEMM serving handoff. A Qwen3 B=1 selected graph passed a
separate serving rewrite/codegen check with 57 norm recomputes and 28
last-arriver reductions. This is structural evidence; it is not a
full-request correctness or performance result for that Qwen3 plan.

The synthetic one-to-one `smem_direct` runtime projection now has a positive
test: it removes the producer stage and preserves the remaining dependency.
`direct_projection_ctest.txt` records the passing host test. This does not
establish end-to-end `smem_direct` code generation; the full CG-to-runtime
lowering remains an implementation gap.
