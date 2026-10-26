# MX-5 LLM C550 results (in progress)

Verified: current cc29/A719 has 4/25 dense selections; MoE remains0/5.
G-A passes; remaining program/solver/LLM/final gates stay open. Default remains MX4.

Authority: `/root/Prompt/TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
Current source: `cc29d78304d297346d657c0fbca0a86e76b1f60f`; producerSHA
`24fa50ed913bce7791f575447a97e06c18a9bfce91c337c0bcc9191179c7c3c7`.
Old3ef/Af8 selections remain [historical](MX5_llm_prior_body_c550.md);
earlier four-cell details are [preserved separately](MX5_llm_development_details_c550.md).

All timings use BF16 greedy64/1024, two fresh processes, each warm1/timed2,
normal WD0/no trace/no DIAG, step-events0; TPOT=(E2E−TTFT)/1023. Original
top4 order is0/1/2/3 then1/2/3/0 on one physical card. Winner improves anchor0
by≥1% or0 stays. Each accepted selected cell passes full512-step C-1/C-2.
Existing cells checked all4; the explicitly separate future selected-only schema
marks other candidates correctness “not run” and stops if its fixed winner fails.

MBU is inferred from streamed linear/head BF16 weights plus KV at average L576,
divided by1.55TB/s×TPOT. It is not a measured DRAM counter. Stored external
baselines and MX4 rows are unchanged. Aggregate throughput uses aggregate E2E.

| Cell | TPOT ms | TTFT ms | E2E s | tok/s | MBU | ΔMX4 | /vLLM | /SGLang |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen0.6B B1 | 1.954394 | 9.285 | 2.008630 | 509.80 | 0.4153 | -1.470% | 0.6357 | 0.7951 |
| Qwen0.6B B2 | pending | — | — | — | — | — | — | — |
| Qwen0.6B B4 | pending | — | — | — | — | — | — | — |
| Qwen0.6B B8 | pending | — | — | — | — | — | — | — |
| Qwen0.6B B16 | 3.580819 | 60.760 | 3.723937 | 4399.64 | 0.4052 | -21.077% | 0.8907 | 0.8239 |
| Qwen1.7B B1 | pending | — | — | — | — | — | — | — |
| Qwen1.7B B2 | pending | — | — | — | — | — | — | — |
| Qwen1.7B B4 | pending | — | — | — | — | — | — | — |
| Qwen1.7B B8 | pending | — | — | — | — | — | — | — |
| Qwen1.7B B16 | pending | — | — | — | — | — | — | — |
| Llama1B B1 | 2.157002 | 8.958 | 2.215571 | 462.18 | 0.7449 | -2.445% | 0.7232 | 0.8537 |
| Llama1B B2 | pending | — | — | — | — | — | — | — |
| Llama1B B4 | pending | — | — | — | — | — | — | — |
| Llama1B B8 | pending | — | — | — | — | — | — | — |
| Llama1B B16 | pending | — | — | — | — | — | — | — |
| Llama3B B1 | pending | — | — | — | — | — | — | — |
| Llama3B B2 | pending | — | — | — | — | — | — | — |
| Llama3B B4 | pending | — | — | — | — | — | — | — |
| Llama3B B8 | pending | — | — | — | — | — | — | — |
| Llama3B B16 | pending | — | — | — | — | — | — | — |
| Llama8B B1 | 10.574720 | 41.277 | 10.859216 | 94.30 | 0.9203 | -1.314% | 0.9090 | 0.9536 |
| Llama8B B2 | pending | — | — | — | — | — | — | — |
| Llama8B B4 | pending | — | — | — | — | — | — | — |
| Llama8B B8 | pending | — | — | — | — | — | — | — |
| Llama8B B16 | pending | — | — | — | — | — | — | — |

Four original predicted and measured medians below are ID order0,1,2,3 (ms).
Per-process raw refs and complete comparisons: `MX5_selection_c550.csv`.

| Cell | Predicted4 | Measured4 | Selected | selected pred/meas |
|---|---|---|---:|---:|
| Llama8B B1 | 10.388443/10.389083/10.389083/10.390603 | 10.574720/10.559957/10.575258/10.570999 | 0 | 0.982385 |
| Llama1B B1 | 2.180651/2.181271/2.181671/2.181831 | 2.157002/2.157537/2.158015/2.160963 | 0 | 1.010964 |
| Qwen0.6B B1 | 1.996530/1.996600/1.996710/1.996820 | 1.954394/1.955735/1.958664/1.953756 | 0 | 1.021560 |
| Qwen0.6B B16 | 3.641533/3.644603/3.645143/3.645333 | 3.580819/3.580893/3.581747/3.585770 | 0 | 1.016955 |

Actual selected TN/WN/D/Pw for QKV/O/GU/down/head; suffix T=TileRange,
S=StreamK. Exact Part/source/pin metadata remain in the CSV and executed pin.

| Cell | QKV; O; GU; down; head | b_min/PK | Threads | Stages/step |
|---|---|---:|---:|---:|
| Llama8B B1 | 64/1/1/1T; 64/1/1/1T; 32/8/4/8T; 64/1/3/3T; 64/4/1/4T | 1/0 | 512 | 161 |
| Llama1B B1 | 32/1/1/1T; 32/1/2/2T; 64/2/2/4T; 32/1/2/2T; 32/8/2/8T | 1/2 | 576 | 81 |
| Qwen0.6B B1 | 32/2/2/2T; 16/1/2/2T; 64/1/1/1T; 16/1/2/2T; 128/8/1/2T | 1/0 | 512 | 141 |
| Qwen0.6B B16 | 64/1/1/1T; 16/1/1/1T; 64/1/1/1T; 16/1/1/1T; 128/8/1/2T | 2/0 | 512 | 141 |

All current normal selected artifacts have frame0 and actual resident1.
Llama1B B1 uses a genuine independently solved576 domain;512 exceptions retain
candidate-bound canonical576 histories, shared reduction budgets and D7 proofs.
Controlled227 diagnostics, including frame0/resident0, never qualify for execution.
Pins under `configs/pins/mx5/blockinit_cc29/` preserve executed bytes/metadata;
source-qualified origins and real measurements establish selection, not a metadata flip.

Selected-pin no-trace diagnostic interventions (us/nominal layer); values are
nonadditive whole-step differences. Each completed ledger has16 fresh processes,
two reverse-order rounds, warm1/timed2. Comparison uses exact matching-source
uninstrumented stage bench; no older manual ledger/PK/geometry substitution.

| Cell | net qkv/A/o/GU/down | bench qkv/A/o/GU/down | head/history net/bench us | floor63 us |
|---|---|---|---:|---:|
| Llama1B B1 | 9.708/13.609/6.078/43.138/21.335 | 12.510/15.195/10.160/52.660/25.340 | 317.360/326.810 | 341.048 |
| Qwen0.6B B1 | 6.723/23.151/4.704/9.823/6.566 | 10.100/22.070/8.360/13.770/10.030 | 190.932/195.300 | 331.180 |
| Qwen0.6B B16 | 9.425/65.835/8.621/12.743/12.820 | 13.550/65.584/11.810/16.410/15.570 | 199.857/199.670 | 311.723 |
| Llama8B B1 | 31.343/20.936/21.425/149.165/71.324 | 34.850/18.622/24.920/149.750/76.540 | 637.630/638.630 | 718.590 |
| Llama8B B16 | pending | pending | pending | pending |

Head differences are confounded by future-token history. Floor63 retains prefetch,
barriers/control/timer and is not pure synchronization. No additive latency sum
or unmeasured site-time allocation is claimed. Selected attention exceeds the
appendix inference; QwenB16 o/down do too, with elevated matching bench costs.

QwenB1 normal static source attribution is byte/section/resource/final-MIR equal
to one line-info twin: kernel161560B/20469MIR; device.text161816B including
stub32B/alignment224B. Body bytes: QKV10072,A82440,O7464,GU14352,down14656,
head17576; all23 buckets reconcile, shared184B, unattributed16B/2899MIR.
These are compiler-source PC unions and allocated MIR records, not decoded ISA
counts, exclusive semantic ownership or current I1/cache latency. Other plans
need their own equality proof; failure stops that mapping item. Metadata and
observed compile-pipeline wall are append-only in `MX5_resources_c550.csv`.

Unfinished: remaining dense selections, MoE native/prompt/route checks and5cells,
LLM30 acceptance, current I1 and the recorded crosssegment strong-overlap gap.
MX4-final P-PAIR is not measured. A separately labelled fresh22a reference is
being prepared with real new source/producer/identity; old strict-reuse failures
and fixed historical CSVs stay intact. DNN follows LLM acceptance.
