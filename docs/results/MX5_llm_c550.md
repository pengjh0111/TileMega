# MX-5 C550 LLM development checkpoint

Verified current A19 source: 6/25 dense cells measured; 2 distinguishable MX4 regressions. G-LLM remains open.
MoE native0/5;DNN native0/25. All defaults remain MX4; final=False.
Authority SHA256:75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73.
Current body7c9658ac5,frozen normal producer88327cbc4;historicalcc29 results remain separate.

## Current 25-cell coverage (verified)

Fixed MX4 serving andexternal baseline CSVs are never remeasured ormodified.
Pending rows contain no fabricated MX5 measurements. Fastest baseline isminimum recorded BF16 TPOT.

| Model | B | MX5 TPOT ms | MX4 ms | Fastest baseline ms | vs MX4 | MX5/fastest | Status |
|---|---:|---:|---:|---:|---:|---:|---|
|Qwen0.6B|1|1.976266|1.983552|2.457900|-0.367%|0.8040|nonregression|
|Qwen0.6B|2|2.301333|2.095492|2.802200|+9.823%|0.8213|FAIL G-LLM|
|Qwen0.6B|4|2.575954|2.394211|2.993300|+7.591%|0.8606|FAIL G-LLM|
|Qwen0.6B|8|—|3.040548|3.668500|—|—|pending C1/C2,E2E|
|Qwen0.6B|16|3.524260|4.537082|4.020300|-22.323%|0.8766|nonregression|
|Qwen1.7B|1|—|3.424746|3.801700|—|—|pending C1/C2,E2E|
|Qwen1.7B|2|—|3.555443|4.265800|—|—|pending C1/C2,E2E|
|Qwen1.7B|4|—|3.894393|4.386800|—|—|pending C1/C2,E2E|
|Qwen1.7B|8|—|4.603874|5.019400|—|—|pending C1/C2,E2E|
|Qwen1.7B|16|—|6.252656|5.464700|—|—|pending C1/C2,E2E|
|Llama1B|1|—|2.211073|2.526700|—|—|576 cost blocked|
|Llama1B|2|—|2.272015|2.849600|—|—|576 cost blocked|
|Llama1B|4|—|2.438731|2.913400|—|—|576 cost blocked|
|Llama1B|8|—|2.771217|3.045700|—|—|576 cost blocked|
|Llama1B|16|—|3.435274|3.359300|—|—|576 cost blocked|
|Llama3B|1|—|5.317005|5.836400|—|—|pending C1/C2,E2E|
|Llama3B|2|—|5.477695|6.493000|—|—|pending C1/C2,E2E|
|Llama3B|4|—|5.807209|6.650300|—|—|pending C1/C2,E2E|
|Llama3B|8|—|6.532365|6.982200|—|—|pending C1/C2,E2E|
|Llama3B|16|—|8.296789|7.573900|—|—|pending C1/C2,E2E|
|Llama8B|1|10.585899|10.715531|11.089700|-1.210%|0.9546|nonregression|
|Llama8B|2|—|10.929705|12.524500|—|—|pending C1/C2,E2E|
|Llama8B|4|—|11.352761|12.703000|—|—|pending C1/C2,E2E|
|Llama8B|8|—|12.206898|13.156900|—|—|pending C1/C2,E2E|
|Llama8B|16|13.150235|14.541197|14.108000|-9.566%|0.9321|nonregression|

## Actual selected request metrics (verified)

| Cell | TTFT ms | E2E s | output tok/s | MBU | pred/meas |
|---|---:|---:|---:|---:|---:|
|Qwen0.6B/B1|9.302|2.031022|504.180|0.4107|1.02291|
|Qwen0.6B/B16|60.790|3.666108|4469.044|0.4117|1.00856|
|Llama8B/B1|41.190|10.870565|94.199|0.9193|0.98267|
|Llama8B/B16|734.198|14.186888|1154.869|0.7956|0.99287|
|Qwen0.6B/B2|11.334|2.365598|865.743|0.3712|0.93463|
|Qwen0.6B/B4|21.262|2.656463|1541.900|0.3647|0.95646|

MBU uses inferred streamed BF16 linear weights plusKV atmeanL576 dividedby1.55TB/s×TPOT.
TTFT isreported,not optimized. Request numbers aremedians oftwo fresh process medians.

## Exact original top4 comparisons (verified)

Each candidate passes512-step C-1/C2 andtwo fresh timing processes beforepublication.
ISL64/OSL1024,greedyBF16,warm1+timed2,TPOT=(E2E−TTFT)/1023.
WD0/uninstrumented timing;WD1 checks. Selection:take fastest onlyif>=1% faster thananchor0.

| Cell | Candidate | Predicted ms | Measured ms | pred/meas | Chosen |
|---|---:|---:|---:|---:|---|
|Qwen0.6B/B1|0|2.021542|1.976266|1.02291|yes|
|Qwen0.6B/B1|1|2.021612|1.977535|1.02229|no|
|Qwen0.6B/B1|2|2.021722|1.980171|1.02098|no|
|Qwen0.6B/B1|3|2.021832|1.974518|1.02396|no|
|Qwen0.6B/B16|0|3.554417|3.524260|1.00856|yes|
|Qwen0.6B/B16|1|3.557487|3.523996|1.00950|no|
|Qwen0.6B/B16|2|3.558027|3.528153|1.00847|no|
|Qwen0.6B/B16|3|3.558217|3.529580|1.00811|no|
|Llama8B/B1|0|10.402495|10.585899|0.98267|yes|
|Llama8B/B1|1|10.403135|10.587368|0.98260|no|
|Llama8B/B1|2|10.403135|10.581901|0.98311|no|
|Llama8B/B1|3|10.404655|10.580788|0.98335|no|
|Llama8B/B16|0|13.056443|13.150235|0.99287|yes|
|Llama8B/B16|1|13.057403|13.163448|0.99194|no|
|Llama8B/B16|2|13.058373|13.173144|0.99129|no|
|Llama8B/B16|3|13.061223|13.163197|0.99225|no|
|Qwen0.6B/B2|0|2.150885|2.301333|0.93463|yes|
|Qwen0.6B/B2|1|2.151145|2.304746|0.93335|no|
|Qwen0.6B/B2|2|2.151245|2.299368|0.93558|no|
|Qwen0.6B/B2|3|2.151515|2.308777|0.93189|no|
|Qwen0.6B/B4|0|2.463801|2.575954|0.95646|yes|
|Qwen0.6B/B4|1|2.465311|2.578473|0.95611|no|
|Qwen0.6B/B4|2|2.465591|2.579156|0.95597|no|
|Qwen0.6B/B4|3|2.465641|2.577437|0.95663|no|

## Executed coordinates and matching-source bench (verified)

Coordinate notation=TN/WN/D/Part/Pw,T=TileRange,S=StreamK. Allselectedplans512threads;5L+1 stages.
Prefetch quantity andR_next values also remain in append-only serving/configuration CSV fields.

| Cell | b_min | Attn PK | qkv | o | gate_up | down | head |
|---|---:|---:|---|---|---|---|---|
|Qwen0.6B/B1|1|0|32/2/2/T/2|16/1/2/T/2|64/1/1/T/1|16/1/2/T/2|128/8/1/T/2|
|Qwen0.6B/B16|4|0|64/1/1/T/1|16/1/1/T/1|64/1/1/T/1|16/1/1/T/1|128/8/1/T/2|
|Llama8B/B1|1|0|64/1/1/T/1|64/1/1/T/1|32/8/4/T/8|64/1/3/T/3|64/4/1/T/4|
|Llama8B/B16|4|0|64/1/1/T/1|64/1/1/T/1|64/1/1/S/1|64/1/1/S/4|128/8/2/T/2|
|Qwen0.6B/B2|4|0|32/2/2/T/2|16/1/2/T/2|64/1/1/T/1|16/1/2/T/2|128/8/1/T/2|
|Qwen0.6B/B4|8|0|32/2/2/T/2|16/1/2/T/2|64/1/1/T/1|16/1/2/T/2|128/8/1/T/2|

| Cell | QKV us | Attention us | O us | GateUp us | Down us | Head us |
|---|---:|---:|---:|---:|---:|---:|
|Qwen0.6B/B1|10.100|22.963|8.360|13.770|10.030|195.300|
|Qwen0.6B/B16|13.550|62.472|11.810|16.410|15.570|199.670|
|Llama8B/B1|34.850|19.061|24.920|149.750|76.540|638.630|
|Llama8B/B16|37.620|53.722|31.620|174.180|90.350|656.700|
|Qwen0.6B/B2|10.080|27.167|8.500|13.810|10.290|195.160|
|Qwen0.6B/B4|10.400|36.361|9.040|14.000|11.250|194.370|

These bench values areoriginal qualified three-process measurements,not program stage latencies.
Attention is the specified five-L-grid interpolation averaged overL65..1087.

## D7 and remaining boundaries

Verified QwenB2 DIAG:16fresh child processes(two rounds×8masks),eachwarm1/timed2.
Attention nonadditive intervention=35.923/35.842us perlayer,versusbench27.167us;ratio≈1.32.
Other interventions andhead/token-history confounding areexplicit inMX5_stage_ledger_c550.csv.
Verified three-fresh off-grid diagnostics:weighted32-block midpoint estimate33.674us vs5-grid27.167us.
Inferred model total usingthat estimate2.333086ms vsactual2.301333ms;thisisnot an officialcost replacement.
Original5grid remeasurements differ0.202–0.715%;intermediate segmentation explains muchof theprediction gap.
Otherbmin midpoint estimates(us):1=32.040,2=33.989,4=33.674,8=39.179;not default-selection evidence.
Current-source I1 timing remains unmeasured;no instruction-cache cause isclaimed.

Llama1B:normal576 program167MT/frame0/actualresident1;required PK2 self-next bench44–48B frame.
Three CPU-only bench operand/capacity/lifetime diagnostics remain rejected,withfailures preserved.
G576 single-element reference criterion remainsfailed;instrumented copy hasidentical fulloutputs.
Observed gate is1.192e−6 belowBF16 midpoint;CPU gate2.861e−6 above. Oracle/tolerance remain unchanged.

The15-cell continuation queue has60 actual512queries/D7 and15top4 protocolsREADY;completedcells areincludedabove.
The source-bound queue publishes only complete,independently reverified results,never pending orfailed items.
A segment-merge factor-reuse candidate staysisolated until numeric/MIR/resource andE2E gates pass.
MoE native andprompt,fullLLM30 acceptance andDNN native remain mandatory andincomplete.
The user waived the original60-card-hour total cutoff;allcard/concurrency/guard/per-job rules remain.

Raw evidence:local_lds_tail_remaining15_full_execution_v1;local_lds_tail_selected_qwenB2_D7_diag_v1;
local_lds_tail_qwenB2_length_geometry_d7_v1;gemm576_failed_coordinate_intermediates_v1.
Historicalcc29 seven measured cells remain in append-only CSVs andgit history;nonecount as current-source coverage.
