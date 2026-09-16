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
python3 docs/experiments/MODELS/export_mlp.py --model llama --seq 4 --out /tmp/r6-llama-mlp
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
