# MX-5 prior-body LLM selections (historical)

Verified 3ef/Af8-source selections retained separately from current cc29/A719.
Raw measurements and append-only CSV rows remain unchanged.

## Qwen B1 actual original top four

The native solver ranks and candidate IDs are retained. Every candidate passed actual 512-step C-1 and C-2 before measurement. The process order is 0/1/2/3, then 1/2/3/0 on card 7.

| Original ID | Predicted ms | Process 1 ms | Process 2 ms | Median ms | pred/meas | Selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.990283 | 1.946521 | 1.941521 | 1.944021 | 1.023797 | yes |
| 1 | 1.990353 | 1.946268 | 1.951853 | 1.949060 | 1.021186 | no |
| 2 | 1.990463 | 1.947532 | 1.952561 | 1.950046 | 1.020726 | no |
| 3 | 1.990573 | 1.946135 | 1.949988 | 1.948062 | 1.021822 | no |

Candidate 0 is the fastest median and remains the anchor under the prescribed 1% selection rule. Its improvement over MX-4 is distinguishable using max(1%, round spread); all four pred/meas values are within [0.9, 1.1].

## Executed configuration

| Body | TN | WN | D | Part | Pw |
|---|---:|---:|---:|---|---:|
| qkv | 32 | 2 | 2 | kTileRange | 2 |
| o | 16 | 1 | 2 | kTileRange | 2 |
| gate_up | 64 | 1 | 1 | kTileRange | 1 |
| down | 16 | 1 | 2 | kTileRange | 2 |
| head | 128 | 8 | 1 | kTileRange | 2 |

Attention: D128/Q2/hkv8, b_min=1, prefetch kind 0. Actual thread shape is 512, with eight compute waves and thread-0 synchronization. There are 141 stages/barriers per step (5×28+1). DT branches remain W576/R role-fence+atomic/P-B/X-A/block-fragments; this cell uses the documented per-plan thread exception.

All four actual canonical 576 histories preserve their original ranks, positive P0 references and shared retry counts. Canonical normal builds remain at MT168 with nonzero frames; exact-source/pin cap227 diagnostics retain frames 68/68/100/68 and actual zero residency. The independent normal 512 originals are MT239/239/241/239 with frame0 and resident1. Four substantive ordered D7 reviews and the complete ResolvedTop4 v2 audit passed. These observations do not prove a physical register lower bound or global 576 impossibility.

Selected WD0 resources: MT239, ST108, frame0 B, static shared 0 B, device text 161304 B, allocated MIR records 20437. Per-body code attribution remains unknown because the target ELF lacks the required mapping.

The stored pin is byte-for-byte the executed physical configuration, including its original diagnostic/manual metadata. Selection is established by the qualified native origin, D7/ResolvedTop4 proofs and eight actual guarded measurements. This is not global default promotion.

## Evidence and remaining work

- Raw selection and immutable process artifacts: `runs/mx5/qwenB1_actual_resolved_top4_execution_v1/`.
- Source-bound D7 reports and protocol: `runs/mx5/qwenB1_actual_d7_review_v1/`.
- Reviewable selected binaries: `runs/mx5/tm/qwen3_0_6b/B1/`; pin: `configs/pins/mx5/qwen3_0_6b_B1.json`.
- All 50 dense prefill libraries now pass complete-source/watchdog checks and final L1 API0/positive-residency queries. Historical ten libraries were preserved and excluded after strict source rejection; fresh twins replace them.
- The selected-pin non-trace stage ledger is still required. Earlier five manual-control ledgers remain separately labelled and are not substituted for it.
- Prompt §9C MX-4-final P-PAIR has no completed proof: the old producer SHA is not archived, and the original source-root snapshot fails current strict validation. Existing ten MX-4 measurement rounds are not a new P-PAIR. Evidence: `runs/mx5/mx4_development_pair_readiness_v1_61/`.
- Remaining 22 prior-body dense selections, five MoE selections/native implementation/routing, full LLM acceptance and DNN remain unfinished. No external baseline or old result row was changed.

## Qwen B16 actual original top four

Verified: all four original candidates passed 512-step C-1/C-2 and two fresh WD0 timing processes on card 6. Candidate 0 wins under the 1% anchor rule. Its /fastest baseline ratio is 0.887308 (vLLM), a 21.3758% improvement over MX-4.

| Original ID | Predicted ms | Process 1 ms | Process 2 ms | Median ms | pred/meas | Selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3.640890 | 3.566934 | 3.567552 | 3.567243 | 1.020646 | yes |
| 1 | 3.643960 | 3.573029 | 3.574071 | 3.573550 | 1.019703 | no |
| 2 | 3.644500 | 3.568471 | 3.568705 | 3.568588 | 1.021272 | no |
| 3 | 3.644690 | 3.578439 | 3.577348 | 3.577893 | 1.018669 | no |

Configuration: qkv TN64/WN1/D1/Pw1; o TN16/WN1/D1/Pw1; gate_up TN64/WN1/D1/Pw1; down TN16/WN1/D1/Pw1; head TN128/WN8/D1/Pw2. All use TileRange. Attention b_min4/PK0; threads512; 141 stages per step.

Verified: canonical576 shared reductions were 1/0/0/2. All four actual final512 originals have frame0 and API0/resident1. Exact-source cap227 diagnostic frames remain 56/56/108/64 with API0/resident0; all four ordered D7 reviews and ResolvedTop4 audit passed. No controlled diagnostic is execution eligible.

Evidence: `runs/mx5/qwenB16_actual_resolved_top4_execution_v1/`, `runs/mx5/qwenB16_actual_d7_review_v1/`, published `runs/mx5/tm/qwen3_0_6b/B16/` and exact executed pin `configs/pins/mx5/qwen3_0_6b_B16.json`. Manual metadata remains unchanged, with external native-selection proof; default remains off.

## Llama8B B1 prior-body original top four

Verified: all four 3ef/Af8 originals pass C-1/C-2 at 512 steps; eight fresh WD0 measurements on card 6 complete the interleaved selection. Candidate 0 is retained by the 1% anchor rule, though candidate 1 has the smallest median. No new cc29 measurement or default promotion is claimed.

| Original ID | Predicted ms | Process 1 ms | Process 2 ms | Median ms | pred/meas | Selected |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 10.389647 | 10.558385 | 10.563261 | 10.560823 | 0.983791 | yes |
| 1 | 10.390287 | 10.559950 | 10.556142 | 10.558046 | 0.984111 | no |
| 2 | 10.390287 | 10.563979 | 10.563167 | 10.563573 | 0.983596 | no |
| 3 | 10.391807 | 10.559733 | 10.557912 | 10.558822 | 0.984182 | no |

Actual configuration: qkv TN64/WN1/D1/Pw1/kTileRange, o TN64/WN1/D1/Pw1/kTileRange, gate_up TN32/WN8/D4/Pw8/kTileRange, down TN64/WN1/D3/Pw3/kTileRange, head TN64/WN4/D1/Pw4/kTileRange; attention b_min1/PK0, threads512, stages161.

Evidence: `runs/mx5/llama8bB1_actual_resolved_top4_execution_v1/`, ordered D7 in `runs/mx5/llama8bB1_actual_d7_review_v1/`, exact executed pin `configs/pins/mx5/llama3_1_8b_B1.json`. New A719 costs and25 scoped512 origins are complete; their final acceptance remains pending.

