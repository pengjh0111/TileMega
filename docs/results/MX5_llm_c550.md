# MX-5 LLM C550 results (in progress)

Verified: historical cc29/A719 has7/25 completed dense top-four selections;MoE remains0/5.
G-A passes; remaining program/solver/LLM/final gates stay open. Default remains MX4.

Authority: `/root/Prompt/TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
Historical selected source: `cc29d78304d297346d657c0fbca0a86e76b1f60f`;producerSHA
`24fa50ed913bce7791f575447a97e06c18a9bfce91c337c0bcc9191179c7c3c7`.
Old3ef/Af8 selections remain [historical](MX5_llm_prior_body_c550.md);
earlier four-cell details are [preserved separately](MX5_llm_development_details_c550.md).

All timings use BF16 greedy64/1024, two fresh processes, each warm1/timed2,
normal WD0/no trace/no DIAG, step-events0; TPOT=(E2E−TTFT)/1023. Original
top4 order is0/1/2/3 then1/2/3/0 on one physical card. Winner improves anchor0
by≥1% or0 stays. Each accepted selected cell passes full512-step C-1/C-2.
The first five cells checked all4. The explicit selected-only schema checks its
fixed winner and marks other candidates correctness “not run”; failure stops the cell.

MBU is inferred from streamed linear/head BF16 weights plus KV at average L576,
divided by1.55TB/s×TPOT. It is not a measured DRAM counter. Stored external
baselines and MX4 rows are unchanged. Aggregate throughput uses aggregate E2E.

| Cell | TPOT ms | TTFT ms | E2E s | tok/s | MBU | ΔMX4 | /vLLM | /SGLang |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen0.6B B1 | 1.954394 | 9.285 | 2.008630 | 509.80 | 0.4153 | -1.470% | 0.6357 | 0.7951 |
| Qwen0.6B B2 | 2.318045 | 11.266 | 2.382626 | 859.56 | 0.3685 | 10.621% | 0.6881 | 0.8272 |
| Qwen0.6B B4 | 2.601075 | 21.301 | 2.682201 | 1527.10 | 0.3612 | 8.640% | 0.7559 | 0.8690 |
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
| Llama8B B16 | 13.188281 | 733.762 | 14.225374 | 1151.74 | 0.7933 | -9.304% | 0.9323 | 0.9348 |

Performance gate failures against fixed MX4 (numerical checks still passed):
- Qwen0.6B B2: 2.318045ms, +10.621%; D7 required, no default acceptance.
- Qwen0.6B B4: 2.601075ms, +8.640%; D7 required, no default acceptance.
Subsequent E2E dispatch is stopped for D7; B4 completed and its actual results were independently consumed.

Four original predicted and measured medians below are ID order0,1,2,3 (ms).
Per-process raw refs and complete comparisons: `MX5_selection_c550.csv`.

| Cell | Predicted4 | Measured4 | Selected | selected pred/meas |
|---|---|---|---:|---:|
| Llama8B B1 | 10.388443/10.389083/10.389083/10.390603 | 10.574720/10.559957/10.575258/10.570999 | 0 | 0.982385 |
| Llama8B B16 | 13.179178/13.180138/13.181108/13.183958 | 13.188281/13.183734/13.194895/13.208248 | 0 | 0.999310 |
| Llama1B B1 | 2.180651/2.181271/2.181671/2.181831 | 2.157002/2.157537/2.158015/2.160963 | 0 | 1.010964 |
| Qwen0.6B B1 | 1.996530/1.996600/1.996710/1.996820 | 1.954394/1.955735/1.958664/1.953756 | 0 | 1.021560 |
| Qwen0.6B B2 | 2.160459/2.160719/2.160819/2.161089 | 2.318045/2.320833/2.319673/2.319388 | 0 | 0.932018 |
| Qwen0.6B B4 | 2.432917/2.434427/2.434707/2.434757 | 2.601075/2.607365/2.606601/2.605314 | 0 | 0.935351 |
| Qwen0.6B B16 | 3.641533/3.644603/3.645143/3.645333 | 3.580819/3.580893/3.581747/3.585770 | 0 | 1.016955 |

Actual selected TN/WN/D/Pw for QKV/O/GU/down/head; suffix T=TileRange,
S=StreamK. Exact Part/source/pin metadata remain in the CSV and executed pin.

| Cell | QKV; O; GU; down; head | b_min/PK | Threads | Stages/step |
|---|---|---:|---:|---:|
| Llama8B B1 | 64/1/1/1T; 64/1/1/1T; 32/8/4/8T; 64/1/3/3T; 64/4/1/4T | 1/0 | 512 | 161 |
| Llama8B B16 | 64/1/1/1T; 64/1/1/1T; 64/1/1/1S; 64/1/1/4S; 128/8/2/2T | 1/0 | 512 | 161 |
| Llama1B B1 | 32/1/1/1T; 32/1/2/2T; 64/2/2/4T; 32/1/2/2T; 32/8/2/8T | 1/2 | 576 | 81 |
| Qwen0.6B B1 | 32/2/2/2T; 16/1/2/2T; 64/1/1/1T; 16/1/2/2T; 128/8/1/2T | 1/0 | 512 | 141 |
| Qwen0.6B B2 | 32/2/2/2T; 16/1/2/2T; 64/1/1/1T; 16/1/2/2T; 128/8/1/2T | 4/0 | 512 | 141 |
| Qwen0.6B B4 | 32/2/2/2T; 16/1/2/2T; 64/1/1/1T; 16/1/2/2T; 128/8/1/2T | 8/0 | 512 | 141 |
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
| Llama8B B16 | 35.526/61.127/26.511/166.236/83.019 | 37.620/57.557/31.620/174.180/90.350 | 655.180/656.700 | 686.870 |
| Qwen0.6B B2 D7 | 7.518/36.680/5.068/10.146/7.050 | 10.080/27.509/8.500/13.810/10.290 | 190.205/195.160 | 334.980 |

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

Verified local-LDS amendment and both indistinguishable manual B2 P-PAIRs
are recorded in MX5_local_lds_c550.md. The new source passes5fresh×336
required attention cases andmanual C1/C2512;new400-key costs and25scope replays pass.
These checks are not additional final solver selections anddo not change defaults.

## Current local-LDS source: completed solver selections (verified)

Frozen88327/A19dd8 has4/25 completed dense selections;the earlier table
remains historicalcc29,not new-source completion. Qwen0.6B B1 exacttop4
passes allfour C-1/C2512 checks andtwo fresh timing processes per candidate.

| Candidate | Predicted ms | Measured ms | pred/meas | Chosen |
|---|---:|---:|---:|---|
|0|2.021542|1.976266|1.0229|yes|
|1|2.021612|1.977535|1.0223|no|
|2|2.021722|1.980171|1.0210|no|
|3|2.021832|1.974518|1.0240|no|

Candidate0 TPOT1.976266ms vsfixedMX41.983552ms:−.367%,indistinguishable.
The fastest measured candidate3 improves less than1%,so anchor0 is retained.
No default/globalLLM acceptance follows. Raw:local_lds_tail_qwenB1_top4_execution_v1.

Qwen0.6B B16 also passes allfour C-1/C2512 checks andtwo fresh timing
processes per candidate. Actualsource576 histories andD7 bind its512 plans.

| Candidate | Predicted ms | Measured ms | pred/meas | Chosen |
|---|---:|---:|---:|---|
|0|3.554417|3.524260|1.0086|yes|
|1|3.557487|3.523996|1.0095|no|
|2|3.558027|3.528153|1.0085|no|
|3|3.558217|3.529580|1.0081|no|

Anchor0 remains selected:TPOT3.524260ms vsfixedMX44.537082ms(−22.323%),
and/fastestBF16baseline=.876616(vLLM4.0203ms);a distinguishable improvement.
Raw:local_lds_tail_qwen3_0_6b_B16_top4_execution_v1. No default promotion.

Llama3.1-8B B1 also passes allfour C-1/C2512 checks andtwo fresh
timing processes per candidate;allselected WD0/WD1 resource queries pass.

| Candidate | Predicted ms | Measured ms | pred/meas | Chosen |
|---|---:|---:|---:|---|
|0|10.402495|10.585899|0.9827|yes|
|1|10.403135|10.587368|0.9826|no|
|2|10.403135|10.581901|0.9831|no|
|3|10.404655|10.580788|0.9834|no|

Anchor0 remains selected:TPOT10.585899ms vsfixedMX410.715531ms(-1.210%),
a distinguishable improvement. Pred/meas0.98267. Defaults unchanged.
Raw:local_lds_tail_llama3_1_8b_B1_top4_execution_v1.

Llama3.1-8B B16 allfour C-1/C2512 andtwo fresh timings each pass.

| Candidate | Predicted ms | Measured ms | pred/meas | Chosen |
|---|---:|---:|---:|---|
|0|13.056443|13.150235|0.9929|yes|
|1|13.057403|13.163448|0.9919|no|
|2|13.058373|13.173144|0.9913|no|
|3|13.061223|13.163197|0.9923|no|

Anchor0 TPOT13.150235ms vsfixedMX414.541197ms
(-9.566%,distinguishable),/fastestbaseline=0.932112.
TTFT734.198ms;E2E14.186888s;MBU0.7956(inferred bytes).
Raw:local_lds_tail_llama3_1_8b_B16_top4_execution_v1;defaults unchanged.

Current selected matching-source bench costs(us);these are not new program ledgers.

| Model/B | QKV | Attention | O | GateUp | Down | Head |
|---|---:|---:|---:|---:|---:|---:|
|Qwen0.6B/1|10.100|22.963|8.360|13.770|10.030|195.300|
|Qwen0.6B/16|13.550|62.472|11.810|16.410|15.570|199.670|
|Llama8B/1|34.850|19.061|24.920|149.750|76.540|638.630|
|Llama8B/16|37.620|53.722|31.620|174.180|90.350|656.700|

Attention bench exceeds appendix inference;current source per-stage DIAG/I1 remains pending.
