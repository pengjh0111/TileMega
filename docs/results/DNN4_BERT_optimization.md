# DNN4 BERT optimization progress

Verified guarded comparisons use the same BF16 model inputs and the coordinates identified by each artifact. All candidates passed five fresh FP32-reference processes with identical cross-process output hashes. P-PAIR uses A/B/A/B/A/B, three fresh processes per side, 50 warmups and 200 events per process. Acceptance requires a gain larger than max(0.5%, twice the larger process range). GPU1 diagnostics guide optimization; G22 diagnostics compare on GPU3 against rank0 and do not replace selection against the eventual final winner or three-process P-DNN.

| Experiment | GPU | Reference ms | Candidate ms | Gain % | Noise threshold % | Significant gain |
| --- | --- | --- | --- | --- | --- | --- |
| attention B1 | 1 | 0.629600 | 0.618880 | +1.703 | 5.581 | False |
| attention B2 | 1 | 0.680416 | 0.670224 | +1.498 | 5.343 | False |
| TK128 B16 | 1 | 1.522272 | 1.526256 | -0.262 | 1.221 | False |
| residual TMA B16 | 1 | 1.528224 | 1.490880 | +2.444 | 1.322 | True |
| in-situ B16 | 1 | 1.521344 | 1.468064 | +3.502 | 1.942 | True |
| in-situ B16 | 3 | 1.519872 | 1.463936 | +3.680 | 1.143 | True |
| in-situ B1 | 1 | 0.630176 | 0.614240 | +2.529 | 5.433 | False |
| register40/232 B1 | 2 | 0.607888 | 0.616752 | -1.458 | 4.659 | False |
| FFN1 128x256 B16 | 2 | 1.475936 | 1.670880 | -13.208 | 1.791 | False |
| prefetch1 B16 | 2 | 1.464960 | 1.458768 | +0.423 | 2.580 | False |
| Repeat inline B1 | 2 | 0.614864 | 0.622000 | -1.161 | 3.272 | False |
| Repeat inline B16 | 2 | 1.464112 | 1.478592 | -0.989 | 1.377 | False |
| GELU four B16 | 2 | 1.469824 | 1.470416 | -0.040 | 1.855 | False |
| Repeat single B1 | 2 | 0.615280 | 0.497904 | +19.077 | 6.023 | True |
| Repeat single B1 G22 | 3 | 0.615024 | 0.497616 | +19.090 | 5.489 | True |
| Repeat single B16 | 2 | 1.468864 | 1.488016 | -1.304 | 2.561 | False |
| Repeat declined B2 | 2 | 0.685648 | 0.694240 | -1.253 | 4.354 | False |
| Repeat single B4 | 2 | 0.785616 | 0.784976 | +0.081 | 1.386 | False |
| Repeat inline16 B1 | 2 | 0.498656 | 0.507488 | -1.771 | 0.937 | False |
| FFN1 TMA192 B16 | 2 | 1.467632 | 1.495152 | -1.875 | 1.786 | False |
| P1 normalization B16 | 2 | 1.473008 | 1.428352 | +3.032 | 1.768 | True |
| P1 normalization B16 G22 | 3 | 1.463872 | 1.424240 | +2.707 | 2.521 | True |

Verified attention experiment: QK/PV columns were distributed across eight consumer warps without changing any output accumulation order. Per-head unit5/5 and sanitizer memcheck1/1, initcheck1/1, racecheck5/5, synccheck5/5 passed; whole-model outputs also matched the control byte for byte. Neither timing gain exceeded process variability, so the original header was restored. The initial driver added an unnecessary3% cutoff, which does not affect either rejection because both measured noise thresholds exceed3%. Frozen sources and the patch remain under runs/dnn4/bert_priority.

Verified TK128 experiment: one deterministic max-legal-slot configuration on12 QKV stages; other stages and tile/split/store coordinates unchanged. No significant gain. Verified residual TMA experiment: only24 residual-reuse stages change store mode, with no additional shared allocation or changes to tiles/splits/TK/Slots.

Verified in-situ experiments use the six already authorized candidate programs, three fresh processes per variant and20 warmups/50 trace windows. Group sums select a coordinate, then the assembled nontrace program faces a whole-model P-PAIR gate. These experiments add no parameter scan.

in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate1, group5→candidate1.
in-situ B1: group0→candidate3, group1→candidate0, group2→candidate0, group3→candidate0, group4→candidate4.

Verified post-collection checks reject40/232: the89-case scan still spills28B on256x128 residual_ln_rowstats, and the full B16 program spills36B; it is never executed. B1 compiles without spill but has no significant gain. FFN1 128x256 uses one deterministic legal four-slot register-store variant; it regresses and is rejected. Distance1 has no significant gain.

Verified Repeat inline tables use grid-constant parameter storage instead of dependent global index reads. Unit5/5 and sanitizer memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5 pass; both whole-model comparisons fail DT-CODE. The implementation patch is archived and the production headers are restored. Four-element GELU scheduling also has no end-to-end gain: five full precision probes report0 bad values and preserve scalar bits; the132-CTA resident epilogue ratio is1.0072, and its full B16 comparison is rejected.

Verified single-layer Repeat12x5 is an explicit deviation from the prescribed two-layer Repeat6x10. Logical stage numbers preserve arena parity for odd loop bodies. Odd and even body unit cohorts pass5/5; each passes isolated memcheck1/1,initcheck1/1,racecheck5/5, synccheck5/5. B1 and B16 strict whole-model checks pass5/5 with identical cross-process output hashes. GPU2 B1 improves19.08%;GPU3 complete P-PAIR confirms19.09%. B16 and B4 have no significant gain. The B2 published rank1 coordinates decline Repeat and remain flat; a separate periodic B2 candidate spills4B and is never executed. The mode remains explicitly selected rather than enabled for unmeasured coordinates.

Verified B1 boundary trace sum535456ns exceeds fresh nontrace smoke498320ns by7.45%, so these windows are excluded from P-TUNE. Four of the six original complete variants spill and are never executed. Nontrace P-SEL uses the two legal loop variants, the accepted loop control and an existing distinct model-rank candidate; it retains the loop control. GPU3 P-DNN has median0.498176ms,p900.500640ms,process range0.013232ms. This is still above the unchanged0.4858ms BF16 goal. Failed or killed partial paired cohorts are discarded in full; occupancy exit75 retries the complete cohort.

Verified follow-up rejects16-bit inline parameter tables: odd-loop unit5/5 and sanitizer1/1/5/5 pass;full B1 FP32/byte checks5/5 pass, but the complete P-PAIR against the accepted loop is0.498656→0.507488ms,ratio1.017712,noise0.9369%. The prototype headers are restored. Verified B8 has incompatible canonical buffer aliases;codegen now retains its exact flat program instead of failing. Four real CPU codegen cases cover B1 one/two-layer loops,B2 nonperiodic fallback and B8 alias fallback;all retain63 logical stages. No new B8 performance is claimed.

Verified P1 row normalization is explicitly selected only for B16: the side warp reduces the existing statistics in their original order and publishes cached (mean,inv) through32 individual release arrivals before consumer side_full. This deviates from consumer-side normalization in section5.2. The first prototype failed racecheck with89 errors and is rejected regardless of timing. The repaired implementation passes unit5/5 and isolated memcheck1/1,initcheck1/1,racecheck5/5, synccheck5/5. B1 loop and B16 whole-model FP32/byte checks each pass5/5. B16 GPU2/GPU3 complete paired gains are3.03%/2.71%;GPU3 P-DNN median1.421744ms, p901.446144ms,process range0.001824ms. This still exceeds the unchanged1.0869ms BF16 goal. Combining P1 normalization with the periodic B2 loop spills8B stores/32B loads and is never executed;the standalone periodic B2 loop also remains rejected.

GPU3 foreign occupancy delayed additional same-card G22 probes; only our pending schedulers were cancelled, and fresh candidate/control pairs moved to authorized GPU2. The separately identified B1 follow-up does not silently replace the earlier frozen20-cell collection. That BERT batch collection completed on GPU3: B1/B2/B4/B8/B16 =0.614016/0.695504/0.790144/1.043264/1.463168ms. None beats its frozen BF16 target. CNN collection is complete and paused; existing CNN results and all60 baseline arms are reused. GPU ownership remains qualified by the container PID namespace. See DNN4_BERT_optimization.csv for identities, clocks, correctness receipts and raw timing sources.
