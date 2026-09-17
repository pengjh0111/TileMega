# R6 model subset and explicit boundaries

The public Llama configuration is only partially covered; see `coverage.tsv`.
`export_mlp.py` exports 16 independent, already-normalized SwiGLU regions at
hidden=2048 and intermediate=8192, with distinct random weights per region.
The 32 boundary tensors (normalized activation and residual per region) are
explicit user inputs. This tests all 16 MLP components, not a sequential decoder
or a full model. Attention, normalization, RoPE, embeddings and the output head
are cut. Covered attention components have not been assembled into a maximal
whole-graph subset; A1 end-to-end remains a declared degraded result.

The frontend adapter recognizes the structural MLP pattern and lowers to the
existing GEMM and elementwise TaskBodies. It adds no missing operator and no
operator-specific solver cost rule. The ordinary decoder pattern takes priority.

Reproduce from the repository root (requires approximately 6 GiB temporary disk):

```sh
python3 docs/experiments/MODELS/export_mlp.py --model llama --seq 4 --out docs/experiments/MODELS/llama_mlp
build-portable/tools/tilemega-compile docs/experiments/MODELS/llama_mlp/exported_program.pt2 docs/experiments/MODELS/llama_mlp/direct_pt2.cu --solve docs/experiments/COSTMODEL/event_fit/target.json --seq 4 --past 3 --search-capacity 3 --search-domain docs/experiments/COSTMODEL/event_fit/search_domain.json --dump-cg docs/experiments/MODELS/llama_mlp/direct_pt2.mlir --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv
```

The final command imports, solves geometry/kappa/placement/residency, writes the
CG, and generates CUDA. The reduced geometry domain is explicit and does not
claim global optimality. `llama_mlp/auto.cu.search.tsv` retains evaluated choices.
`llama_mlp/correctness/r{0..49}.{log,json}` contains 50 fresh process PASS results,
including all 16 CPU golden output comparisons. Warmup=0, repeat=1 makes their
latencies single-launch observations, not a steady-state performance claim.
Large regenerated random fixtures, torch archives and executables are omitted
from git; their hashes and build commands remain in the evidence.

The direct `.pt2` invocation was executed in this round; its export bridge and
compiled occupancy logs are in `llama_mlp/direct_pt2.log`. Its generated CUDA
is byte-identical to `llama_mlp/auto.cu`, which produced the 50/50 results above
(`direct_pt2.equivalence.tsv`). The bridge remains internal to this command.

## Continuation: maximal supported graph and numerical stop

The continuation supersedes the earlier omission of supported attention and
output-head regions. `export_covered.py` exports all 16 Llama layers' Q/K/V and
output projections, KV append, attention, both residual additions, SwiGLU MLP,
and the final vocabulary projection. The residual chain stays connected across
layers. Only the known embedding, RMSNorm-parameter and FP32-RoPE gaps are cut;
their outputs are explicit graph inputs. There are 98 inputs and 66 checked
outputs. No missing TaskKind or approximate replacement was added.

The production compiler imports this `.pt2`, solves geometry/kappa/residency and
all six inner placements, writes the CG, and generates a 209-stage CUDA program.
The first solver-selected 64x128x16s2 split8 and the admitted split4 replacement
both fail one final-residual CPU-golden element. Independent numerical probes
at split1 with 64x128x16s2, 32x16x64s2 and 32x16x32s2 fail the same element.
The probe configurations are explicitly manual diagnostics, not solver wins.
All five retain the unchanged numerical tolerance, input seed and outputs.
L0.5, L1 and L2 are mutually bit-identical; this is a new graph's numerical
admission failure, not a regression of an existing accepted configuration.

The diagnostic at tensor 0, index 389 has actual -0.41796875 versus expected
-0.44921875, difference 0.03125 and tolerance 0.0231875014. CPU FX-interpreter
comparisons localize the first discrepancy to the first layer's V projection
(one BF16 element), then three attention-context elements and the output
projection. One residual rounding unit becomes 0.03125 after repeated additions;
the fixed tolerance first fails at the penultimate layer. Head-major FX context
values are transposed to the backend's token-major layout before comparison in
`cpu_buffers_aligned.tsv`; the earlier unaligned diagnostic is not evidence of
an attention indexing error.

**A1's maximal-graph 50/50 result is FAIL, not completed by the old 16-region
MLP result.** R6 §9.2 stops numerical admission of this new graph after the
five failed candidates. The §2/§7.3 exclusion concerns missing operators; it
is not a blanket prohibition on fixing existing backend numerical behavior.
The R7 action is to specify and validate the BF16 contraction accumulation and
rounding contract in `GemmStageTaskBody.h` against the exported CPU operation,
then recheck propagation through `AttentionChunkTaskBody.h` and residual Add.
Changing the reference, seed, tolerance, outputs or residual connectivity to
obtain a pass is not an accepted remedy. Other R6 groups continue independently.

Reproduce the maximal export and failed admitted solve:

```sh
python3 docs/experiments/MODELS/export_covered.py --seq 4 --out docs/experiments/MODELS/covered_llama
build-portable/tools/tilemega-compile docs/experiments/MODELS/covered_llama/exported_program.pt2 docs/experiments/MODELS/covered_llama_admitted2/auto.cu --solve docs/experiments/COSTMODEL/event_fit/target.json --seq 4 --past 3 --search-capacity 3 --search-domain docs/experiments/COSTMODEL/event_fit/search_domain.json --dump-cg docs/experiments/MODELS/covered_llama_admitted2/auto.mlir --hop-curve docs/experiments/SIMULATOR/hop_ns.tsv --numerical-rejections docs/experiments/MODELS/covered_llama/numerical_rejections.json
python3 docs/experiments/MODELS/run_covered.py --root docs/experiments/MODELS/covered_llama_admitted2
```

Use fresh output directories when reproducing: collectors refuse to overwrite
raw process logs. The maximal graph's diagnostic timings are retained in raw
logs but are not promoted to accepted model-performance results. The prior MLP
subset's 50/50 remains a narrower, explicitly degraded result.

The independent FP64 diagnostic strengthens the localization: first V token 0,
column 463 has accumulator -0.450195362966042, just below the BF16 rounding
midpoint -0.4501953125. FP64 rounded to BF16 agrees with the frozen CPU output
(-0.451171875); GPU returns -0.44921875. Replacing only V with the captured GPU V
in the diagnostic CPU attention removes all three context differences. This
is a diagnostic intervention, not a replacement reference or accepted run.
Raw diagnostic outputs are `diagnostic/first_v_rounding.json` and
`diagnostic/first_context_isolation.json` under `covered_llama_admitted2`.
A concrete next implementation is compensated accumulation or selective
recomputation near BF16 rounding midpoints in `GemmStageTaskBody.h`, with its
backend cost declared and the full residual chain revalidated.
