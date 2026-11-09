# CATLASS49 local Attention TaskBody component

This opt-in component is **not model-admitted**. It assembles the pinned
CATLASS QK/PV BlockMmad and online-softmax/rescale epilogues, without its
FAInferKernel, block scheduler or Host operator launcher. The caller supplies
one independent Q/K/V domain, its coordinates and the AIC/AIV role.

Source: CATLASS `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`,
`examples/49_ascend950_flash_attention_infer/fai_kernel.h` and the included
Block/epilogue components. The adapted header retains the CANN notice; its
license is in `include/tilemega/Backend/Ascend/LICENSE.CATLASS`.
No dependency upgrade is required.

Contract: BF16 Q/K/V and P; FP32 scores, online statistics and output
accumulation; physical M32, D64/D128, KV blocks128, valid keys1..1088.
Q/K/V addresses differ between the two probe tasks. P goes UB→L1;
scores and partial outputs go L0C→UB. Scores/P and expanded FP32 KV are
not stored in GM by this source. This is a declared dataflow, not a measured
HBM traffic counter. The probe has no causal/prefill semantic claim.

One G1 MIX launch processes two tasks over four reuse epochs. Each AIV owns
16 physical rows; M3 has useful work only on AIV0. Padding is private,
initialized and checked. Mode4 credits7..10, score11, output12 and P13 use
paired base/+16 channels, separate from the CATLASS local pipe event banks.
Only AIC publishes external completion, after both MTE3 acknowledgements.
Both QK and PV use physical32×128×128 MMAD banks, including D64; their shared
L0 event rings require equal buffer strides. AIVs wait for owner release before
seeding the next Task's credits. Logical D64 remains64 in data/cast/output.
Internal hardware waits are not abortable; the Host timeout remains required.
Five independent successful processes are required before a synchronization
claim. A passing component does not admit a model, C3 or TPOT improvement.

Use the existing private environment and explicit targets, nice10/parallel1:

```bash
source scripts/as2_cpu_env.sh
cmake -S tools/ascend -B "$AS2_WORK/build/taskbody_fai_tile_20261104_v1" \
  -DAS2_GENERATED_SOURCE="$REPO/runs/as2/models/qwen3_0_6b/batched_residual_candidate_20261104_decode_v1/persistent_llm.cpp"
nice -n 10 cmake --build "$AS2_WORK/build/taskbody_fai_tile_20261104_v1" \
  --target as2_fai_tile as2_host_fai_contracts --parallel 1
"$AS2_WORK/build/taskbody_fai_tile_20261104_v1/as2_host_fai_contracts"
```

The generated-source CMake argument satisfies the shared build entry; this
explicit probe target does not execute that model or establish CTG provenance.
Prepare one new controlled input directory with
`python -m tilemega.bench.as2_body prepare-fai-tile --rows 3 --head-dim 128 --keys 129 --out INPUT`.
Run through `as2_run --physical-device 1 --out JOB --timeout-s 60 -- .../as2_fai_tile 3 128 129 INPUT OUTPUT`,
after rechecking the physical/PCI mapping and process list. Then use
`python -m tilemega.bench.as2_body compare-fai-tile --input INPUT --output OUTPUT --job JOB`.
All paths must be fresh canonical paths beneath `runs/as2`; never bypass the
wrapper or overwrite a failed attempt. BF16 P has a separate precision ID;
the frozen model/greedy tolerances are unchanged.
