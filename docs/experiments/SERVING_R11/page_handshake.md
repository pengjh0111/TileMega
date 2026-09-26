# Page-ring handshake lower bound

`page_handshake_bench.cu` runs 128 CTAs with one loader warp and four compute
warps. Each CTA performs 1024 full/empty cycles over the generated Llama B1
page layouts, with one 16 B `cp.async` per loader lane per cycle and no GEMM.
Six runs after one warmup yield a median of **241.45 ns/page** at 8 KiB and
**241.97 ns/page** at 16 KiB. Raw outputs are in `page_handshake_8k.txt` and
`page_handshake_16k.txt`. Both builds use 101376 B shared memory and sm_89.

This is a lower bound on a real full-page transfer: the benchmark copies only
512 B per cycle, whereas a GEMM page carries 8 or 16 KiB and uses more
`cp.async` instructions. The similar per-cycle values show that the 16 KiB
gain comes from fewer page cycles per GEMM, while the exact transfer cost
still requires the full class controls in `paged_class_bench/`.
