# MX-5 C550 LLM checkpoint

Verified current consumed-prefetch source: 4/25 dense solver selections; 0 distinguishable MX4 regressions.
Body4b7d83686;normalproducer230a572b atd889;G14/H15 unchanged,A20=b4555ea2e0dbde380c37c48793e2fd5ff4a9f6dc767b0ebb9606e108c4fd34b3.
Authority SHA256:75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73.
Allgates/default promotion remain open;MoE native0/5,DNN native0/25;defaults MX4.

## Current 25-cell coverage (verified)

| Model | B | MX5 TPOT ms | MX4 ms | Fastest baseline ms | vs MX4 | MX5/fastest | Status |
|---|---:|---:|---:|---:|---:|---:|---|
|Qwen0.6B|1|1.973800|1.983552|2.457900|-0.492%|0.8030|nonregression|
|Qwen0.6B|2|—|2.095492|2.802200|—|—|pending|
|Qwen0.6B|4|—|2.394211|2.993300|—|—|pending|
|Qwen0.6B|8|—|3.040548|3.668500|—|—|pending|
|Qwen0.6B|16|3.517390|4.537082|4.020300|-22.475%|0.8749|nonregression|
|Qwen1.7B|1|—|3.424746|3.801700|—|—|pending|
|Qwen1.7B|2|—|3.555443|4.265800|—|—|pending|
|Qwen1.7B|4|—|3.894393|4.386800|—|—|pending|
|Qwen1.7B|8|—|4.603874|5.019400|—|—|pending|
|Qwen1.7B|16|—|6.252656|5.464700|—|—|pending|
|Llama1B|1|2.135062|2.211073|2.526700|-3.438%|0.8450|nonregression|
|Llama1B|2|—|2.272015|2.849600|—|—|pending|
|Llama1B|4|—|2.438731|2.913400|—|—|pending|
|Llama1B|8|—|2.771217|3.045700|—|—|pending|
|Llama1B|16|3.006219|3.435274|3.359300|-12.490%|0.8949|nonregression|
|Llama3B|1|—|5.317005|5.836400|—|—|pending|
|Llama3B|2|—|5.477695|6.493000|—|—|pending|
|Llama3B|4|—|5.807209|6.650300|—|—|pending|
|Llama3B|8|—|6.532365|6.982200|—|—|pending|
|Llama3B|16|—|8.296789|7.573900|—|—|pending|
|Llama8B|1|—|10.715531|11.089700|—|—|pending|
|Llama8B|2|—|10.929705|12.524500|—|—|pending|
|Llama8B|4|—|11.352761|12.703000|—|—|pending|
|Llama8B|8|—|12.206898|13.156900|—|—|pending|
|Llama8B|16|—|14.541197|14.108000|—|—|pending|

## Selected request metrics (verified)

| Cell | Threads | TTFT ms | E2E s | output tok/s | MBU | pred/meas |
|---|---:|---:|---:|---:|---:|---:|
|Llama1B/B1|576|9.018|2.193187|466.901|0.7525|1.01325|
|Qwen0.6B/B1|512|9.348|2.028546|504.795|0.4112|1.02455|
|Llama1B/B16|576|82.020|3.157382|5189.110|0.5952|1.00489|
|Qwen0.6B/B16|512|60.591|3.658881|4477.872|0.4125|1.00982|

MBU=(inferred streamedBF16 linearweights+KV atmeanL576)/(1.55TB/s×TPOT);TTFT reported,not optimized.

## Exact original top4 (verified)

Eachcandidate:C1/C2512;twofreshWD0 timingprocesses,warm1/timed2,ISL64/OSL1024,greedyBF16,step-events0.
TPOT=(E2E−TTFT)/1023;choosefastest onlyif>=1% fasterthanprediction0;otherwisechoose0.
| Cell | Four predicted ms (0–3) | Four measured ms (0–3) | Four pred/meas | Chosen |
|---|---|---|---|---:|
|Llama1B/B1|2.163351, 2.164121, 2.164481, 2.164631|2.135062, 2.135004, 2.133439, 2.133940|1.01325, 1.01364, 1.01455, 1.01438|0|
|Qwen0.6B/B1|2.022251, 2.022321, 2.022431, 2.022541|1.973800, 1.976463, 1.980066, 1.977786|1.02455, 1.02320, 1.02140, 1.02263|0|
|Llama1B/B16|3.020928, 3.021193, 3.021248, 3.021513|3.006219, 2.998767, 3.014646, 3.009563|1.00489, 1.00748, 1.00219, 1.00397|0|
|Qwen0.6B/B16|3.551934, 3.555004, 3.555544, 3.555734|3.517390, 3.519564, 3.523198, 3.526506|1.00982, 1.01007, 1.00918, 1.00829|0|

## Executed coordinates and matching-source bench (verified)

NotationTN/WN/D/Part/Pw;T=TileRange,S=StreamK. Each dense step5L+1stages;R_next/Pw alsoinCSV.
| Cell | b_min | Attn PK | QKV | O | GateUp | Down | Head |
|---|---:|---:|---|---|---|---|---|
|Llama1B/B1|1|2|32/1/1/T/1|32/1/2/T/2|64/2/2/T/4|32/1/2/T/2|32/8/2/T/8|
|Qwen0.6B/B1|1|0|32/2/2/T/2|16/1/2/T/2|64/1/1/T/1|16/1/2/T/2|128/8/1/T/2|
|Llama1B/B16|4|2|32/1/2/T/2|32/1/1/T/1|64/1/3/T/3|32/1/1/S/2|64/8/1/T/4|
|Qwen0.6B/B16|4|0|64/1/1/T/1|16/1/1/T/1|64/1/1/T/1|16/1/1/T/1|128/8/1/T/2|

| Cell | QKV us | Attention us | O us | GateUp us | Down us | Head us |
|---|---:|---:|---:|---:|---:|---:|
|Llama1B/B1|12.430|14.297|10.140|52.640|25.280|326.760|
|Qwen0.6B/B1|10.100|22.988|8.360|13.770|10.030|195.300|
|Llama1B/B16|15.810|35.190|14.270|63.610|38.680|339.960|
|Qwen0.6B/B16|13.550|62.384|11.810|16.410|15.570|199.670|
Bench=qualified threefresh stage rows,not program stage latencies;attentionfivegrid interpolation overL65..1087.

## Historical A19 reference (verified, separate source)

A19=7c9658ac5/producer88327cbc4;these11source-bound selections are not current-body acceptance.
| Cell | A19 TPOT ms | vs fixedMX4 | Status |
|---|---:|---:|---|
|Qwen0.6B/B1|1.976266|-0.367%|nonregression|
|Qwen0.6B/B16|3.524260|-22.323%|nonregression|
|Llama8B/B1|10.585899|-1.210%|nonregression|
|Llama8B/B16|13.150235|-9.566%|nonregression|
|Qwen0.6B/B2|2.301333|+9.823%|FAIL|
|Qwen0.6B/B4|2.575954|+7.591%|FAIL|
|Qwen0.6B/B8|2.810956|-7.551%|nonregression|
|Qwen1.7B/B1|3.270282|-4.510%|nonregression|
|Qwen1.7B/B16|5.185553|-17.066%|nonregression|
|Qwen1.7B/B2|3.610137|+1.538%|FAIL|
|Qwen1.7B/B4|3.920668|+0.675%|nonregression|

## D7 and remaining work

A19 QwenB2 attention DIAG=35.923/35.842us vsbench27.167us (16freshchildren,total2rounds×8masks).
Threefresh off-grid32-block midpoint estimate33.674us explains muchof thatmodelgap;fixedgrid/formula unchanged.
Midpoint diagnostic total2.333086ms vsactual2.301333ms;not an official replacementcost.
A20 current-source I1 order/repeat timing andper-body byte attribution remain unmeasured;no cachecause claimed.
True576 Llama1B B2/B4/B8 WD1 original0frames16/24/56B despite fittingWD0;runner stops before numerical launch.
Watchdog lifetime/lazy-record prototypes are not applied;prescribed maximum3 widthdiagnostics are not approvedresults.
OriginalG576 single-element referencefails21.75vs22.125,error.375>tol.374000013;oracle/seed/tolerance unchanged.
Nativegate1.192e−6 belowBF16 midpoint,CPUgate2.861e−6 above;instrumented fulloutput identical.
Consumed-prefetch repair passes5fresh×336required,5fresh×null28,5fresh×helper6×2000×104CTAs,and5fresh×coldselfnext12.
576D64Q4PK2now139/141MT/frame0/actualresident1;normalRcap220/248 andrule-derivedPK retained.
ManualnewbodyLlama1B B1control2.134865/2.135300ms,median2.135083ms;not solverdata orold/newP-PAIR.
A20attention500legalkeys×3fresh complete;300576resource rejects unexecuted;G/H unchanged;4545pool/25scopes replayPASS.
OldA19queue stoppedbefore9unstartedcells byf_bavail<1GiB;actualUID0reservedfree remainsmonitored;raw unchanged.
Original60card-hour totalcap waivedbyuser;card/concurrency/guard/per-job limits unchanged.
Complete current25dense selections,MoE native5+prompt64+routing,LLM30 acceptance/CP5,beforeDNN native25.
G-B/G-PROGRAM/G-SOLVER/G-LLM/G-FWD andfinaldefaults remain open.
Raw:consumed_prefetch_attention_epoch_v1;consumed_prefetch_attention_consumer25_v1;source-specific runnerdirectories.
