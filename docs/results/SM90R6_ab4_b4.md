# SM90R6 AB4b / AB4c B4 supplement

Verified collection of Qwen3-1.7B, Llama-3.1-8B, Qwen3-30B-A3B and OLMoE-1B-7B at batch 4 with the retained R6 search, compiler, exports, target and timing driver. AB4b uses native aggregate selection; AB4c emits a kernel sequence for exactly the selected AB4b coordinates, grid and shared-memory allocation.

The user authorized independent process scheduling across all four visible H800 GPUs. Each card has a worker polling approximately every 2 seconds. A process holds a cooperative claim only during its own attempt. Dependencies unlock separately per model, and up to four independent items can execute concurrently. Rejected interference attempts release the claim and rejoin the queue after 2 seconds; within each priority they rotate behind items attempted less recently. The original physical-card guard, 25-second preflight, ownership checks and throttle checks remain in force.

This B4 dataset uses protocol `user-authorized dynamic independent-process supplement`. Its process slots retain the original counts and arm labels, while physical start order and GPU assignment are dynamic. AB4b and AB4c medians may combine different cards and concurrent host activity. Per-process GPU, PID, process group, actual start/end times and per-GPU summaries are retained so those effects can be examined. Original B1/B16 protocols remain explicitly labeled in the combined table.

P-SEL uses all four linked finalists in logical ABCDDCBA slots: 8 fresh processes per model, median TPOT, with the original top-two prediction tie rule. Each selected arm passes 5 fresh C2 processes, 1024 decode steps in both loop modes, and C1 at 256 positions per sequence (1024 total positions for B4). Final timing uses 3 fresh processes per arm, each with 1 warmup and 3 timed samples at N=1 and N=1024. TPOT is `(median(T1024)-median(T1))*1000/1023`. The original range-based decision threshold is `max(0.5%, 2*max(arm ranges))`. Every accepted timing process is re-audited against its actual raw samples. All final timing processes also emit identical full tokens across the two arms.

Verified results (2026-11-09T11:56:42.395141+00:00):

| Model | AB4b TPOT (ms) | AB4c TPOT (ms) | AB4c change | Threshold | Decision |
|---|---:|---:|---:|---:|---|
| llama3_1_8b | 5.594463557 | 5.962231933 | +6.573792% | 0.500000% | slower |
| olmoe_1b_7b | 2.639050514 | 2.917894170 | +10.566060% | 5.185884% | slower |
| qwen3_1_7b | 1.849657676 | 2.670822325 | +44.395493% | 0.581096% | slower |
| qwen3_30b_a3b | 7.089350816 | 9.316012627 | +31.408543% | 2.596345% | slower |

Verified acceptance: 32 P-SEL processes, 40/40 C2 processes, 8/8 C1 checks and 24 final timing processes, totaling 104 accepted processes plus 4 occupancy items. Twelve accepted P-SEL processes were imported from the stopped whole-unit queue; all imports preserve their original paths, GPU and state provenance. Rejected attempts remain in the attempt ledger and raw logs.

CSV floating-point values retain full precision. The six CSV files contain all eight arm summaries; all 24 final timing processes; all eight correctness gates; the combined 12 model/batch comparisons at B1, B4 and B16; all 108 accepted queue items with complete definitions and results; and every dynamic attempt with raw guard hashes. The arm table includes geometry, structure, build times, search predictions and binary/input hashes. No fields or rows were removed to meet a size limit.

Raw collection: `/root/TileMega-r6/runs/sm90r6/ab4-b4-v2`. Frozen search/build inputs and original raw data: `/root/TileMega-r6/runs/sm90r6/ab4-b4-v1`. The original `environment.json` records hardware and software; the new `queue/inventory.json` rechecks the current GPU UUID mapping. Run `sm90r6_gpu_items.py` for the process queue and `sm90r6_dynamic_ab4.py` for dependency advancement. Re-audit and regenerate tables with `sm90r6_merge_dynamic_ab4.py --root <raw> --out <tables>`.

Verified OLMoE seed correction: collection uses the R5 A1 B4 seed family from the original R6 methodology. Its four finalist coordinates equal those from the superseded initial REF-R4-seed attempt; both receipts remain.
