# MX-5 C550 LLM checkpoint

Verified current consumed-prefetch source with L1 prefill: 7/25 dense solver selections; 0 distinguishable MX4 regressions.
Body4b7d83686;normalproducer230a572b atd889;G14/H15 unchanged,A20=b4555ea2e0dbde380c37c48793e2fd5ff4a9f6dc767b0ebb9606e108c4fd34b3.
Authority SHA256:75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73.
Allgates/default promotion remain open;MoE native0/5,DNN native0/25;defaults MX4.

## Current 25-cell coverage (verified)

| Model | B | MX5 TPOT ms | MX4 ms | Fastest baseline ms | vs MX4 | MX5/fastest | Status |
|---|---:|---:|---:|---:|---:|---:|---|
|Qwen0.6B|1|1.973795|1.983552|2.457900|-0.492%|0.8030|nonregression|
|Qwen0.6B|2|—|2.095492|2.802200|—|—|pending|
|Qwen0.6B|4|—|2.394211|2.993300|—|—|pending|
|Qwen0.6B|8|—|3.040548|3.668500|—|—|pending|
|Qwen0.6B|16|3.517579|4.537082|4.020300|-22.470%|0.8750|nonregression|
|Qwen1.7B|1|—|3.424746|3.801700|—|—|pending|
|Qwen1.7B|2|—|3.555443|4.265800|—|—|pending|
|Qwen1.7B|4|—|3.894393|4.386800|—|—|pending|
|Qwen1.7B|8|—|4.603874|5.019400|—|—|pending|
|Qwen1.7B|16|—|6.252656|5.464700|—|—|pending|
|Llama1B|1|2.134784|2.211073|2.526700|-3.450%|0.8449|nonregression|
|Llama1B|2|2.264237|2.272015|2.849600|-0.342%|0.7946|nonregression|
|Llama1B|4|2.428614|2.438731|2.913400|-0.415%|0.8336|nonregression|
|Llama1B|8|2.648571|2.771217|3.045700|-4.426%|0.8696|nonregression|
|Llama1B|16|—|3.435274|3.359300|—|—|pending|
|Llama3B|1|—|5.317005|5.836400|—|—|pending|
|Llama3B|2|—|5.477695|6.493000|—|—|pending|
|Llama3B|4|—|5.807209|6.650300|—|—|pending|
|Llama3B|8|—|6.532365|6.982200|—|—|pending|
|Llama3B|16|—|8.296789|7.573900|—|—|pending|
|Llama8B|1|—|10.715531|11.089700|—|—|pending|
|Llama8B|2|—|10.929705|12.524500|—|—|pending|
|Llama8B|4|—|11.352761|12.703000|—|—|pending|
|Llama8B|8|—|12.206898|13.156900|—|—|pending|
|Llama8B|16|13.136274|14.541197|14.108000|-9.662%|0.9311|nonregression|

## Selected request metrics (verified)

| Cell | Threads | TTFT ms | E2E s | output tok/s | MBU | pred/meas |
|---|---:|---:|---:|---:|---:|---:|
|Llama1B/B2|576|13.458|2.329772|879.056|0.7150|0.96869|
|Llama1B/B4|576|28.330|2.512803|1630.053|0.6766|0.96629|
|Llama8B/B16|512|688.275|14.126684|1159.791|0.7965|0.99385|
|Llama1B/B8|576|45.294|2.754782|2973.739|0.6388|0.97958|
|Qwen0.6B/B1|512|9.041|2.028233|504.873|0.4112|1.02455|
|Qwen0.6B/B16|512|52.587|3.651070|4487.452|0.4125|1.00977|
|Llama1B/B1|576|8.800|2.192684|467.008|0.7526|1.01338|

MBU=(inferred streamedBF16 linearweights+KV atmeanL576)/(1.55TB/s×TPOT);TTFT reported,not optimized.

## Exact original top4 (verified)

Eachcandidate:C1/C2512;twofreshWD0 timingprocesses,warm1/timed2,ISL64/OSL1024,greedyBF16,step-events0.
TPOT=(E2E−TTFT)/1023;choosefastest onlyif>=1% fasterthanprediction0;otherwisechoose0.
Predictions use actual executed widths;original predictions and bounded pair histories remain in raw results.
| Cell | Four predicted ms (0–3) | Four measured ms (0–3) | Four pred/meas | Chosen |
|---|---|---|---|---:|
|Llama1B/B2|2.193345, 2.194565, 2.194505, 2.200185|2.264237, 2.262324, 2.266004, 2.257325|0.96869, 0.97005, 0.96845, 0.97469|0|
|Llama1B/B4|2.346735, 2.350795, 2.351015, 2.347665|2.428614, 2.426905, 2.427556, 2.431681|0.96629, 0.96864, 0.96847, 0.96545|0|
|Llama8B/B16|13.055541, 13.056501, 13.057471, 13.060321|13.136274, 13.142208, 13.144467, 13.144505|0.99385, 0.99348, 0.99338, 0.99360|0|
|Llama1B/B8|2.594494, 2.594654, 2.598654, 2.598814|2.648571, 2.648759, 2.664316, 2.665914|0.97958, 0.97957, 0.97536, 0.97483|0|
|Qwen0.6B/B1|2.022251, 2.022321, 2.022431, 2.022541|1.973795, 1.978077, 1.980416, 1.975140|1.02455, 1.02237, 1.02122, 1.02400|0|
|Qwen0.6B/B16|3.551934, 3.555004, 3.555544, 3.555734|3.517579, 3.518736, 3.520886, 3.524120|1.00977, 1.01031, 1.00984, 1.00897|0|
|Llama1B/B1|2.163351, 2.164121, 2.164481, 2.164631|2.134784, 2.133451, 2.132636, 2.134002|1.01338, 1.01438, 1.01493, 1.01435|0|

## Executed coordinates and matching-source bench (verified)

NotationTN/WN/D/Part/Pw;T=TileRange,S=StreamK. Each dense step5L+1stages;R_next/Pw alsoinCSV.
| Cell | b_min | Attn PK | QKV | O | GateUp | Down | Head |
|---|---:|---:|---|---|---|---|---|
|Llama1B/B2|4|2|32/1/1/T/1|32/1/2/T/2|64/2/2/T/2|32/1/2/T/2|32/8/2/T/4|
|Llama1B/B4|8|2|32/1/1/T/1|32/1/2/T/2|64/2/2/T/2|32/1/2/T/2|64/4/1/T/4|
|Llama8B/B16|4|0|64/1/1/T/1|64/1/1/T/1|64/1/1/S/1|64/1/1/S/4|128/8/2/T/2|
|Llama1B/B8|8|2|32/1/1/T/1|32/1/1/T/1|64/2/2/T/2|32/1/1/S/2|64/8/4/T/4|
|Qwen0.6B/B1|1|0|32/2/2/T/2|16/1/2/T/2|64/1/1/T/1|16/1/2/T/2|128/8/1/T/2|
|Qwen0.6B/B16|4|0|64/1/1/T/1|16/1/1/T/1|64/1/1/T/1|16/1/1/T/1|128/8/1/T/2|
|Llama1B/B1|1|2|32/1/1/T/1|32/1/2/T/2|64/2/2/T/4|32/1/2/T/2|32/8/2/T/8|

| Cell | QKV us | Attention us | O us | GateUp us | Down us | Head us |
|---|---:|---:|---:|---:|---:|---:|
|Llama1B/B2|12.450|15.922|10.280|52.610|25.410|326.600|
|Llama1B/B4|12.760|20.242|10.880|54.200|27.820|332.300|
|Llama8B/B16|37.620|53.694|31.620|174.180|90.350|656.700|
|Llama1B/B8|13.570|26.675|11.800|56.750|32.480|334.090|
|Qwen0.6B/B1|10.100|22.988|8.360|13.770|10.030|195.300|
|Qwen0.6B/B16|13.550|62.384|11.810|16.410|15.570|199.670|
|Llama1B/B1|12.430|14.297|10.140|52.640|25.280|326.760|
Bench=qualified threefresh stage rows,not program stage latencies;attentionfivegrid interpolation overL65..1087.

## Aligned internal MX4 P-PAIR (verified)

Samecard5;old/new/old/new,freshprocesses,warm1/timed2,ISL64/OSL1024,L1prefill.
Old=rebuilt22a finalsource/pins,not exacthistoricalELF;oldprefillWD1,newprefillWD0,decodeWD0.
| Cell | Old ms | New ms | Change | Round range | Threshold | Status |
|---|---:|---:|---:|---:|---:|---|
|Qwen0.6B/B1|1.983132|1.977543|-0.282%|0.127%|1.000%|indistinguishable|
|Qwen0.6B/B16|4.521433|3.519038|-22.170%|0.414%|1.000%|faster|
|Llama8B/B16|14.494706|13.137143|-9.366%|0.060%|1.000%|faster|
|Llama1B/B1|2.208144|2.135033|-3.311%|0.102%|1.000%|faster|
Source-bound fresh-process rows:MX5_ppair_c550.csv;no fixedbaseline overwrite orselection change.

## Historical A19 reference (verified, separate source)

A19=7c9658ac5/producer88327cbc4;11historical selections,eightnonregressions,threefailures listedbelow.
| Cell | A19 TPOT ms | vs fixedMX4 | Status |
|---|---:|---:|---|
|Qwen0.6B/B2|2.301333|+9.823%|FAIL|
|Qwen0.6B/B4|2.575954|+7.591%|FAIL|
|Qwen1.7B/B2|3.610137|+1.538%|FAIL|

## D7 and remaining work
EarlierA20 fourL2-prefill selections remain inCSVs but outside alignedL1 acceptance;MX4 standardprefill isL1.
OldB2 L2timing rejects nonidentical1024token repetitions;allthree correctedL1 WDpair selections pass.
Original576 WD1frames16/24/56B remain preserved;all12 boundedpair histories,24queries andfive3fresh changed-width costs pass.
G-B original576 criterion stillfails21.75vs22.125,error.375>tol.374000013;oracle/seed/tolerance unchanged.
Observednative gate lies1.192e−6 belowBF16midpoint,CPU2.861e−6 above;instrumented fulloutputs identical.
A19 QwenB2 attentionDIAG35.923/35.842us vsbench27.167;offgrid diagnostic33.674us doesnot replace officialfivegrid.
Currentbody I1 fivefresh orderdeltasQKV+6.304%,attn+3.668%,O+8.104%,GU+7.224%,down−.478%;nocachecause claimed.
Consumed-prefetch336required/null28/helper6/cold12 eachpass5fresh;helper104CTAs×2000;D64Q4PK2frame0/resident1.
A20attention500keys×3fresh complete;300576resource rejects unexecuted;G/H unchanged;4545pool/25scopes replayPASS.
Userwaives only60card-hour totalcap;cards/concurrency/guard/per-joblimits remain,cumulative accounting retained.
Complete current25dense,MoE native5+prompt64+routing,LLM30 acceptance/CP5,beforeDNN native25.
G-B/G-PROGRAM/G-SOLVER/G-LLM/G-FWD andfinaldefaults remain open;detailsinMX5_c550.md/MX5_local_lds_c550.md.
Raw:consumed_prefetch_attention_epoch_v1/attention_consumer25_v1 andsource-specific runnerdirectories.
