# Llama AB4 batch-trend investigation

Verified: all 12 original model/batch cells use identical coordinates, grid and shared memory within each AB4b/AB4c pair. Across all four models, the 339 common TileMega body headers have identical recorded hashes. Within each model, the checkpoint, prompt, model config, runner, software versions, L1 mode, prompt prefill, decode-loop setting and sample pattern match across B1/B4/B16. B1/B16 used the original multi-arm same-card protocol; B4 used the user-authorized dynamic process queue for every model. No Llama-only timing-method exception was found.

Verified: coordinates are independently selected at each batch. For Llama, SwiGLU interleave U is 8/8/16 at B1/B4/B16; attention chunks are 8/4/1. The gate_up (TN,TK,slots) changes from (112,64,4) at B1/B4 to (128,128,2) at B16. The other three models keep their respective U fixed across these batches. These are optimizer-selected program differences, shared by both arms in each cell. Thus the original plot compares separately optimized programs, rather than varying batch while keeping one program fixed. Inferred: configuration changes can contribute to a different trend; this repeat does not isolate any single configuration parameter.

The user requested performance only. The repeat reuses the six selected binaries without compilation, search, C1, C2 or new occupancy probes. It measures three fresh processes per arm/batch, each with one warmup and three samples at N=1 and N=1024, using the original timing driver and interference guard. Independent items use any idle GPU 0–3. TPOT is (median(T1024)-median(T1))/1023. Results are separate from the original dataset and retain per-process GPU and binary provenance.

| Batch | Old AB4b ms | Old AB4c ms | Old C/B − 1 | Repeat AB4b ms | Repeat AB4c ms | Repeat C/B − 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.616666 | 6.009993 | 7.0029% | 5.625087 | 6.012871 | 6.8938% |
| 4 | 5.594464 | 5.962232 | 6.5738% | 5.592682 | 5.960330 | 6.5737% |
| 16 | 6.154418 | 7.165814 | 16.4337% | 6.142400 | 7.149481 | 16.3956% |

Verified: all 18 performance processes completed with accepted guards. Raw design inspection and attempts: `/root/TileMega-r6/runs/sm90r6/ab4-llama-trend-repeat-v1`. Full-precision summaries and all 18 process records accompany this report. No correctness tests were rerun.
