# V-J — fence masking and non-streaming wait graphs

Evidence labels: ✅ measured; ⚠️ interpretation; ❌ conjecture.

## Tile-size scan

The only changed variable was tile size. Every cell used 50 fresh processes at
grid 128 with address reuse, no fence, and no backoff.

| Floats/tile | Correct | Mismatch | Mismatch rate |
|---:|---:|---:|---:|
| 512 | 0/50 | 50/50 | 100% |
| 1,024 | 0/50 | 50/50 | 100% |
| 2,048 | 0/50 | 50/50 | 100% |
| 4,096 | 0/50 | 50/50 | 100% |
| 8,192 | 49/50 | 1/50 | 2% |
| 16,384 | 50/50 | 0/50 | 0% |

✅ The predicted cliff exists between 4,096 and 8,192 floats. ⚠️ L1 eviction is
consistent with the result but was not directly measured with performance
counters. The negative engineering implication is supported: a missing fence
can be easier to observe for small norm/RoPE-like tiles than for large GEMM
tiles, so a large-tile pass is not evidence for §8.5 correctness.

## Backward dependency

The resident limit was probed as `CTA_per_SM × num_SM = 6 × 128 = 768`.

| Grid | Pass | Hang |
|---:|---:|---:|
| 767 | 20/20 | 0/20 |
| 768 | 20/20 | 0/20 |
| 769 | 20/20 | 0/20 |
| 1,536 | 0/20 | 20/20 |

At 1,536, `hang_probe.sh` observed unchanged samples and classified
`no-observed-progress`; `cuda-gdb` was unavailable, so this is a timeout/dead
progress observation, not a PC-level proof of deadlock.

✅ A non-streaming wait-for graph can hang above the resident limit, but
**`grid > resident_limit` is not sufficient to guarantee a hang**: grid 769
passed 20/20 because the scheduled resident subset still admitted progress.
The correct launch rule is:

```text
resident_limit(T, kernel) = num_SM(T) *
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel, block_size, dynamic_smem)
```

and full-grid co-residency is required only when wait-for/scheduling analysis
cannot prove that every possible resident subset has a progress frontier.

## Reproduction

```bash
docs/experiments/V_J/run.sh
cat docs/experiments/V_J/raw/tile_scan.txt
cat docs/experiments/V_J/raw/backward_scan.txt
```
