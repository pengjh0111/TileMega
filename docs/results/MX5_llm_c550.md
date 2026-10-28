# MX-5 C550 LLM development checkpoint

Verified A19 source: 10/25 dense cells measured; 3 distinguishable MX4 regressions. G-LLM remains open.
MoE native0/5;DNN native0/25. All defaults remain MX4; final=False.
Authority SHA256:75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73.
Measured A19 body7c9658ac5/producer88327cbc4;main4b7d83686 consumed-prefetch body has0 solverselected cells;manualLlama1B2.135083ms;new costs pending.

## Current 25-cell coverage (verified)

Fixed MX4 serving andexternal baseline CSVs are never remeasured ormodified.
Pending rows contain no fabricated MX5 measurements. Fastest baseline isminimum recorded BF16 TPOT.

| Model | B | MX5 TPOT ms | MX4 ms | Fastest baseline ms | vs MX4 | MX5/fastest | Status |
|---|---:|---:|---:|---:|---:|---:|---|
|Qwen0.6B|1|1.976266|1.983552|2.457900|-0.367%|0.8040|nonregression|
|Qwen0.6B|2|2.301333|2.095492|2.802200|+9.823%|0.8213|FAIL G-LLM|
|Qwen0.6B|4|2.575954|2.394211|2.993300|+7.591%|0.8606|FAIL G-LLM|
|Qwen0.6B|8|2.810956|3.040548|3.668500|-7.551%|0.7662|nonregression|
|Qwen0.6B|16|3.524260|4.537082|4.020300|-22.323%|0.8766|nonregression|
|Qwen1.7B|1|3.270282|3.424746|3.801700|-4.510%|0.8602|nonregression|
|Qwen1.7B|2|3.610137|3.555443|4.265800|+1.538%|0.8463|FAIL G-LLM|
|Qwen1.7B|4|—|3.894393|4.386800|—|—|pending C1/C2,E2E|
|Qwen1.7B|8|—|4.603874|5.019400|—|—|pending C1/C2,E2E|
|Qwen1.7B|16|5.185553|6.252656|5.464700|-17.066%|0.9489|nonregression|
|Llama1B|1|—|2.211073|2.526700|—|—|576 costs/selection pending|
|Llama1B|2|—|2.272015|2.849600|—|—|576 costs/selection pending|
|Llama1B|4|—|2.438731|2.913400|—|—|576 costs/selection pending|
|Llama1B|8|—|2.771217|3.045700|—|—|576 costs/selection pending|
|Llama1B|16|—|3.435274|3.359300|—|—|576 costs/selection pending|
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
|Qwen0.6B/B8|31.191|2.906799|2818.220|0.3949|1.01933|
|Qwen1.7B/B1|13.216|3.358714|304.879|0.6919|1.02065|
|Qwen1.7B/B16|142.664|5.447485|3007.627|0.5596|0.99592|
|Qwen1.7B/B2|20.539|3.713709|551.470|0.6385|0.95968|

MBU uses inferred streamed BF16 linear weights plusKV atmeanL576 dividedby1.55TB/s×TPOT.
TTFT isreported,not optimized. Request numbers aremedians oftwo fresh process medians.

## Exact original top4 comparisons (verified)

Each candidate passes512-step C-1/C2 andtwo fresh timing processes beforepublication.
ISL64/OSL1024,greedyBF16,warm1+timed2,TPOT=(E2E−TTFT)/1023.
WD0/uninstrumented timing;WD1 checks. Selection:take fastest onlyif>=1% faster thananchor0.

| Cell | Four predicted ms (0–3) | Four measured ms (0–3) | Four pred/meas (0–3) | Chosen |
|---|---|---|---|---:|
|Qwen0.6B/B1|2.021542, 2.021612, 2.021722, 2.021832|1.976266, 1.977535, 1.980171, 1.974518|1.02291, 1.02229, 1.02098, 1.02396|0|
|Qwen0.6B/B16|3.554417, 3.557487, 3.558027, 3.558217|3.524260, 3.523996, 3.528153, 3.529580|1.00856, 1.00950, 1.00847, 1.00811|0|
|Llama8B/B1|10.402495, 10.403135, 10.403135, 10.404655|10.585899, 10.587368, 10.581901, 10.580788|0.98267, 0.98260, 0.98311, 0.98335|0|
|Llama8B/B16|13.056443, 13.057403, 13.058373, 13.061223|13.150235, 13.163448, 13.173144, 13.163197|0.99287, 0.99194, 0.99129, 0.99225|0|
|Qwen0.6B/B2|2.150885, 2.151145, 2.151245, 2.151515|2.301333, 2.304746, 2.299368, 2.308777|0.93463, 0.93335, 0.93558, 0.93189|0|
|Qwen0.6B/B4|2.463801, 2.465311, 2.465591, 2.465641|2.575954, 2.578473, 2.579156, 2.577437|0.95646, 0.95611, 0.95597, 0.95663|0|
|Qwen0.6B/B8|2.863608, 2.865288, 2.865708, 2.866408|2.853298, 2.810956, 2.859982, 2.858719|1.00361, 1.01933, 1.00200, 1.00269|1|
|Qwen1.7B/B1|3.337822, 3.338992, 3.339082, 3.339342|3.270282, 3.269273, 3.269568, 3.266685|1.02065, 1.02133, 1.02126, 1.02224|0|
|Qwen1.7B/B16|5.164377, 5.168567, 5.168857, 5.168867|5.185553, 5.187592, 5.222659, 5.195082|0.99592, 0.99633, 0.98970, 0.99495|0|
|Qwen1.7B/B2|3.464585, 3.465135, 3.465725, 3.465895|3.610137, 3.609771, 3.608008, 3.610523|0.95968, 0.95993, 0.96056, 0.95994|0|

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
|Qwen0.6B/B8|4|0|64/1/1/T/1|16/1/2/T/2|64/1/1/T/1|16/1/2/T/2|128/8/1/T/2|
|Qwen1.7B/B1|1|0|64/1/1/T/1|32/1/2/T/2|128/1/1/T/1|32/1/2/T/2|64/8/2/T/4|
|Qwen1.7B/B16|4|0|64/1/1/T/1|32/1/1/T/1|64/2/2/T/2|32/1/1/T/1|128/8/1/T/2|
|Qwen1.7B/B2|4|0|64/1/1/T/1|32/1/2/T/2|128/1/1/T/1|32/1/2/T/2|64/8/2/T/4|

| Cell | QKV us | Attention us | O us | GateUp us | Down us | Head us |
|---|---:|---:|---:|---:|---:|---:|
|Qwen0.6B/B1|10.100|22.963|8.360|13.770|10.030|195.300|
|Qwen0.6B/B16|13.550|62.472|11.810|16.410|15.570|199.670|
|Llama8B/B1|34.850|19.061|24.920|149.750|76.540|638.630|
|Llama8B/B16|37.620|53.722|31.620|174.180|90.350|656.700|
|Qwen0.6B/B2|10.080|27.167|8.500|13.810|10.290|195.160|
|Qwen0.6B/B4|10.400|36.361|9.040|14.000|11.250|194.370|
|Qwen0.6B/B8|11.270|46.464|10.160|14.650|12.790|195.930|
|Qwen1.7B/B1|15.460|22.963|10.330|36.250|20.500|383.740|
|Qwen1.7B/B16|19.510|62.472|14.520|41.450|32.510|391.430|
|Qwen1.7B/B2|15.500|27.167|10.420|36.340|20.610|383.540|

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
Verified newbody manual Llama1B B1:165MT/frame0/resident1,C1/C2512PASS;TPOT2.134865/2.135300ms.
Manual median2.135083ms is3.437% belowfixedMX4;not a solverselection orold/newP-PAIR.

Llama1B:old576 PK2 self-next bench44–48B frame;new139/141MT/frame0,actualresident1,cold12keys×5freshPASS.
Original Rcap220/248 retained;full336/null-idle/helper/self-next each5fresh PASS;new main normalproducer/costs/E2E pending.
G576 single-element reference criterion remainsfailed;instrumented copy hasidentical fulloutputs.
Observed gate is1.192e−6 belowBF16 midpoint;CPU gate2.861e−6 above. Oracle/tolerance remain unchanged.

The15-cell continuation queue has60 actual512queries/D7 and15top4 protocolsREADY;completedcells areincludedabove.
The source-bound queue publishes only complete,independently reverified results,never pending orfailed items.
Merge-factor candidates measured+4.599/+7.636%slower inoff-grid micro;geometry pruning+0.167%indistinguishable;none adopted.
MoE native andprompt,fullLLM30 acceptance andDNN native remain mandatory andincomplete.
The user waived the original60-card-hour total cutoff;allcard/concurrency/guard/per-job rules remain.

Raw evidence:local_lds_tail_remaining15_full_execution_v1;local_lds_tail_selected_qwenB2_D7_diag_v1;
local_lds_tail_qwenB2_length_geometry_d7_v1;gemm576_failed_coordinate_intermediates_v1.
Historicalcc29 seven measured cells remain in append-only CSVs andgit history;nonecount as current-source coverage.
