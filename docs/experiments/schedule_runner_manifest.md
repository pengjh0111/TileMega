# Manual sm_120 Comparisons

The FUSION and PLACE wrappers accept `--manifest FILE --out NEW_DIRECTORY`.
They do not compile candidates or invent predictions. Prepare actual solver/codegen
artifacts on the sm_120 host first. This runner has only CPU unit-test coverage;
no sm_120 experiment has been run by this change.

The JSON object has `arch: "sm_120"`, `dtype: "bf16"`, and a `cases` array.
Each case has `model`, `seq`, `group`, `state`, `fixture`, and four file paths:
`binary`, `source`, `ptxas`, `prediction`. Paths are relative to the manifest.
Each file has a corresponding `<field>_sha256`. The fixture is a directory.
Prediction JSON must contain `state`, `total_ns`, `event_ns`, `ctas_per_sm`.
These must be actual model outputs, not constants manufactured for the runner.

Required product: models gqa2/mha4, seq 4/128; fusion groups
gemm_elementwise/gemm_rmsnorm with states unfused/fused; placement group
placement with states stage_major/affine_balanced. Missing or duplicate cells
are errors. Compilation logs must include registers and spill stores.

Select exactly one real compute-capability 12.0 device with
`CUDA_VISIBLE_DEVICES` when multiple devices are present. No inherited
`TILEMEGA_*` overrides are accepted. Every round rotates the entire case list,
including compilation state. Default 50 rounds, warmup 5, repeat 11.

Outputs: frozen manifest, device identity, each process log, paired.tsv with
full resource/schedule records and predictions, and status.txt. A correctness
failure stops this experiment. PASS denotes correctness only; paired statistical
analysis and model-sign acceptance remain separate report steps.
