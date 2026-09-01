# V-H — `torch.export` coverage

Evidence labels: ✅ observed; ⚠️ source or environment inspected; ❌ conjecture.

## Result

⚠️ The export could not be run in this environment because `torch`,
`transformers`, and `sympy` are absent. An isolated CPU PyTorch install was
attempted, but the package transfer was not making useful progress; it was
cancelled rather than spending the remaining validation window on a large
environment mutation. No export or operator whitelist is claimed.

The checked-in probe is complete and uses a randomly initialized, two-layer
Llama configuration, so it downloads no model weights. It applies
`torch.export.Dim("seq", min=2, max=128)`, records the graph signature, symbolic
range constraints, buffer mutation entries, and the complete set of
`call_function`/ATen targets. It can be rerun unchanged once the CPU
dependencies are available.

KV-cache form therefore remains **unconfirmed** here. The probe sets
`use_cache=False` for the baseline export; a follow-up run with an explicit
cache input/output is required before Phase 1 decides whether cache state is a
buffer mutation or a graph parameter for the installed Transformers version.

## Reproduction

```bash
docs/experiments/V_H/run.sh
cat docs/experiments/V_H/raw/environment.txt
```

The script exits 77 for a missing optional environment, rather than reporting a
false pass.
