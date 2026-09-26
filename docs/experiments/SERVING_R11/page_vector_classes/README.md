# Current paged-class transport microbenchmarks

Fixed selected geometry: Llama decode B=1 and Qwen3 decode B=16. Each benchmark compares the standard and paged GEMM mainloops for one class on 128 SMs. This measures isolated stage transport, not full-request performance or numerical correctness. The raw archive preserves each build command and run output; executable hashes are in `summary.tsv`. Small-shape numerical checks are separate.

Raw archive SHA256: `1addef759be7a67f3a7bbe8caa703d189719b6cc334bf4ed0427dc463f4d343c`.
