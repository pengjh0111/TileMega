# DRAM floor phase probe

The updated search times `DeriveModelDramFloor` separately. With one
Llama decode B1 configuration, the two structural floor derivations took
0.734 s. With one Qwen3 decode B16 configuration, they took 1.074 s;
pricing/release consumed 31.726 s and relation preparation 12.300 s.
The case JSON and complete aggregate timing TSVs are preserved here.
Both runs were CPU-only (`CUDA_VISIBLE_DEVICES` empty). These are
single-case phase observations, not full-plan budget results.
