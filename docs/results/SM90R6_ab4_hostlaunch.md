# Stage-granular host-launch performance variant

Verified implementation: opt-in `TILEMEGA_NATIVE_SEQUENCE_HOST_LAUNCH=1` bypasses CUDA Graph construction and replay. Each decode step enqueues the same per-stage kernels through ordinary CUDA host launches. The selected coordinates, stage bodies, grid and shared memory come from the existing AB4b/AB4c program at each model/batch. The new arm is labeled `AB4c_hostlaunch`; the original graph-based AB4c data remains separately labeled and retained. The runtime option defaults to zero.

The user requested performance only. No new C1/C2 or standalone occupancy experiments were run. Each arm/cell uses three fresh processes, one warmup and three timed samples at N=1 and N=1024, and the existing timing driver and interference guard. Independent processes run on any idle GPU 0–3. The initial matrix contains 12 cells and 72 final timing processes. GPU assignments, all measured values and binary provenance are retained in CSV.

The user subsequently requested Llama-3.2-3B unconditionally, overriding the original batch-trend trigger. This extension retains the Llama-3.1-8B results. Extension enabled: `true`. When triggered, Llama-3.2-3B uses independent 600-second aggregate searches at B1/B4/B16, four compiled finalists and eight P-SEL processes per batch, then the same three-process performance pair; correctness is not rerun.

| Model | Batch | AB4b TPOT ms | Host-launch TPOT ms | AB4b speedup vs host-launch | Original graph AB4c ms (historical) |
|---|---:|---:|---:|---:|---:|
| llama3_1_8b | 1 | 5.622929 | 6.326321 | 1.125094× | 6.009993 |
| llama3_1_8b | 4 | 5.593729 | 6.248908 | 1.117127× | 5.962232 |
| llama3_1_8b | 16 | 6.143554 | 7.417908 | 1.207430× | 7.165814 |
| llama3_2_3b | 1 | 2.717425 | 3.378992 | 1.243453× | N/A |
| llama3_2_3b | 4 | 2.795401 | 3.471402 | 1.241826× | N/A |
| llama3_2_3b | 16 | 3.155544 | 3.737194 | 1.184326× | N/A |
| olmoe_1b_7b | 1 | 1.412677 | 2.280950 | 1.614630× | 2.147745 |
| olmoe_1b_7b | 4 | 2.643505 | 3.066380 | 1.159968× | 2.917894 |
| olmoe_1b_7b | 16 | 5.071337 | 5.449059 | 1.074482× | 5.301788 |
| qwen3_1_7b | 1 | 1.773782 | 3.167906 | 1.785961× | 3.036318 |
| qwen3_1_7b | 4 | 1.850483 | 2.915717 | 1.575652× | 2.670822 |
| qwen3_1_7b | 16 | 2.153173 | 2.635415 | 1.223968× | 2.436227 |
| qwen3_30b_a3b | 1 | 3.815536 | 4.982955 | 1.305964× | 4.668591 |
| qwen3_30b_a3b | 4 | 7.028166 | 9.749251 | 1.387169× | 9.316013 |
| qwen3_30b_a3b | 16 | 14.874147 | 16.019737 | 1.077019× | 15.604925 |

The original graph AB4c column is historical data; its timing sessions differ from this collection. This dataset measures the explicit host-launch implementation and does not replace the graph-based baseline. Raw source, build receipts and all attempts: `/root/TileMega-r6/runs/sm90r6/stage-granular-variant-v1`. CSV values retain full precision.
