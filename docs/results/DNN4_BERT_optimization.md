# DNN4 BERT optimization progress

Verified guarded comparisons use the same BF16 model inputs and the coordinates identified by each artifact. All candidates passed five fresh FP32-reference processes with identical cross-process output hashes. P-PAIR uses A/B/A/B/A/B, three fresh processes per side, 50 warmups and 200 events per process. Acceptance requires a gain larger than max(0.5%, twice the larger process range). GPU1 diagnostics guide optimization; G22 diagnostics compare on GPU3 against the explicitly identified control and do not replace selection against the eventual final winner or three-process P-DNN.

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
| scalar phase B1 | 2 | 0.497920 | 0.499296 | -0.276 | 6.703 | False |
| FFN2 arena B1 | 3 | 0.498064 | 0.487936 | +2.033 | 6.361 | False |
| FFN2 split4 B1 | 3 | 0.483856 | 0.512000 | -5.817 | 5.800 | False |
| attention Rows64 B16 | 2 | 1.428832 | 1.420240 | +0.601 | 1.695 | False |
| 96-lane normalization B1 | 2 | 0.486912 | 0.501776 | -3.053 | 6.605 | False |
| 96-lane normalization B16 | 3 | 1.425968 | 1.419824 | +0.431 | 0.655 | False |
| Repeat reg64 B2 GPU2 | 2 | 0.694512 | 0.642496 | +7.490 | 7.989 | False |
| Repeat reg64 B2 G22 | 3 | 0.695536 | 0.659808 | +5.137 | 0.500 | True |
| Repeat alias B8 G22 | 3 | 1.051232 | 1.070064 | -1.791 | 2.965 | False |
| prefetch table B1 | 3 | 0.498800 | 0.518432 | -3.936 | 6.813 | False |
| prefetch table B2 | 3 | 0.658608 | 0.631904 | +4.055 | 5.636 | False |
| P1 normalization B2 | 3 | 0.658144 | 0.625248 | +4.998 | 5.611 | False |
| P1 normalization B4 | 3 | 0.789984 | 0.760960 | +3.674 | 1.944 | True |
| P1 normalization B8 | 3 | 1.050832 | 1.020016 | +2.933 | 2.939 | False |
| FFN2 TK128 B1 | 3 | 0.482416 | 0.498784 | -3.393 | 4.850 | False |
| GELU reciprocal B1 | 3 | 0.483824 | 0.498944 | -3.125 | 6.581 | False |
| GELU reciprocal B16 | 3 | 1.427840 | 1.415392 | +0.872 | 1.605 | False |
| attention unroll B1 | 3 | 0.498944 | 0.495792 | +0.632 | 7.254 | False |
| FFN1 TM256 B16 | 3 | 1.427968 | 1.635488 | -14.533 | 3.027 | False |
| Repeat P1 B8 | 3 | 1.045552 | 1.037936 | +0.728 | 2.439 | False |
| binding arithmetic B1 | 3 | 0.484800 | 0.500544 | -3.248 | 6.304 | False |
| binding arithmetic B1 | 0 | 0.484256 | 0.487056 | -0.578 | 5.880 | False |
| binding arithmetic B2 | 3 | 0.659808 | 0.611152 | +7.374 | 1.351 | True |
| binding arithmetic B2 | 0 | 0.633008 | 0.610976 | +3.481 | 1.372 | False |
| Repeat P1 normalization B16 | 3 | 1.423392 | 1.426320 | -0.206 | 1.643 | False |
| Repeat P1 normalization B16 | 0 | 1.421936 | 1.426672 | -0.333 | 2.923 | False |
| statistics TMA B16 | 3 | 1.421776 | 1.465552 | -3.079 | 0.943 | False |
| Repeat arithmetic P1 reg64 B4 | 3 | 0.761344 | 0.739344 | +2.890 | 2.627 | False |
| Repeat arithmetic P1 reg56 B4 | 3 | 0.760288 | 0.733952 | +3.464 | 0.500 | True |
| in-situ P1 B16 | 3 | 1.428960 | 1.427312 | +0.115 | 0.898 | False |
| in-situ P1 B8 | 3 | 1.049424 | 1.043456 | +0.569 | 1.929 | False |
| FFN1 TK128 B16 | 3 | 1.423696 | 1.466832 | -3.030 | 1.349 | False |
| FFN1 TK128 single-store B16 | 3 | 1.424992 | 1.453520 | -2.002 | 0.980 | False |
| FFN1 TK128 single-store B16 | 3 | 1.428400 | 1.456560 | -1.971 | 351.287 | False |
| FFN1 TN256 single-store B16 | 0 | 1.423696 | 1.569584 | -10.247 | 2.093 | False |
| consumer216 B8 diagnostic | 0 | 1.019408 | 1.017712 | +0.166 | 2.310 | False |
| consumer216 in-situ B8 | 0 | 1.015856 | 1.040640 | -2.440 | 2.903 | False |
| consumer216 in-situ B16 | 0 | 1.397248 | 1.375600 | +1.549 | 3.323 | False |
| two-layer arithmetic B16 | 0 | 1.426400 | 1.438864 | -0.874 | 2.663 | False |
| attention load pairs B1 | 0 | 0.483904 | 0.485600 | -0.350 | 1.157 | False |
| consumer216 tuned B16 G22 | 3 | 1.424880 | 1.385024 | +2.797 | 1.938 | True |
| phase72 B1 G22 | 3 | 0.497392 | 0.500160 | -0.556 | 6.028 | False |
| attention Rows64 B1 | 3 | 0.484480 | 0.516128 | -6.532 | 7.738 | False |
| coalesced partial B1 | 3 | 0.499696 | 0.478160 | +4.310 | 4.537 | False |
| cyclic slots B1 | 3 | 0.497808 | 0.486592 | +2.253 | 5.287 | False |
| cyclic slots tuned B16 prototype | 3 | 1.423984 | 1.366288 | +4.052 | 1.021 | True |
| cyclic216 B16 integrated | 3 | 1.424304 | 1.364432 | +4.204 | 1.771 | True |
| cyclic coalesced B1 prototype | 3 | 0.498032 | 0.467984 | +6.033 | 4.745 | True |
| coalesced cyclic B1 integrated | 3 | 0.499216 | 0.478000 | +4.250 | 0.904 | True |
| coalesced cyclic B2 integrated | 3 | 0.610832 | 0.619792 | -1.467 | 6.386 | False |
| cyclic admitted B8 | 3 | 1.019536 | 1.013904 | +0.552 | 2.014 | False |
| cyclic in-situ B16 | 3 | 1.367568 | 1.374224 | -0.487 | 1.129 | False |
| repaired in-situ B8 | 3 | 1.019488 | 1.023504 | -0.394 | 0.807 | False |
| current whole-program B4 | 3 | 0.712720 | 0.707152 | +0.781 | 2.896 | False |
| producer cyclic B1 | 3 | 0.476240 | 0.469920 | +1.327 | 4.542 | False |
| producer cyclic B16 | 3 | 1.368400 | 1.369504 | -0.081 | 2.383 | False |
| present bindings B2 | 3 | 0.627584 | 0.615488 | +1.927 | 9.265 | False |
| present bindings B16 | 3 | 1.363936 | 1.367488 | -0.260 | 1.595 | False |
| side216 B2 | 3 | 0.611360 | 0.591104 | +3.313 | 8.710 | False |
| matrix store B8 prototype | 3 | 1.019168 | 1.009264 | +0.972 | 3.997 | False |
| matrix store B16 prototype | 3 | 1.365888 | 1.329440 | +2.668 | 1.197 | True |
| matrix store B16 integrated | 3 | 1.375664 | 1.336912 | +2.817 | 1.460 | True |
| matrix existing in-situ B16 | 3 | 1.330560 | 1.343088 | -0.942 | 2.441 | False |
| warpgroup QK B4 | 3 | 0.713056 | 0.713488 | -0.061 | 7.818 | False |
| warpgroup QK B8 | 3 | 1.009680 | 1.005200 | +0.444 | 2.336 | False |
| warpgroup QK B16 | 3 | 1.327888 | 1.320416 | +0.563 | 2.079 | False |
| warpgroup QK/PV B4 | 3 | 0.712048 | 0.707440 | +0.647 | 1.497 | False |
| warpgroup QK/PV B8 | 3 | 1.019920 | 1.012320 | +0.745 | 2.605 | False |
| warpgroup QK/PV B16 | 3 | 1.330352 | 1.319936 | +0.783 | 0.987 | False |
| phase72 B2 GPU2 diagnostic | 2 | 0.628096 | 0.599952 | +4.481 | 6.669 | False |
| phase72 B4 GPU2 diagnostic | 2 | 0.736144 | 0.710800 | +3.443 | 2.339 | True |
| phase72 B8 GPU2 diagnostic | 2 | 1.052320 | 1.009456 | +4.073 | 2.349 | True |
| phase72 B16 GPU2 diagnostic | 2 | 1.426560 | 1.399680 | +1.884 | 1.923 | False |
| phase72 B2 G22 | 3 | 0.627840 | 0.601776 | +4.151 | 6.721 | False |
| phase72 B4 G22 | 3 | 0.733344 | 0.712320 | +2.867 | 2.540 | True |
| phase72 B8 G22 | 3 | 1.051376 | 1.019376 | +3.044 | 0.735 | True |
| phase72 B16 G22 | 3 | 1.423520 | 1.399776 | +1.668 | 1.059 | False |

Verified attention experiment: QK/PV columns were distributed across eight consumer warps without changing any output accumulation order. Per-head unit5/5 and sanitizer memcheck1/1, initcheck1/1, racecheck5/5, synccheck5/5 passed; whole-model outputs also matched the control byte for byte. Neither timing gain exceeded process variability, so the original header was restored. The initial driver added an unnecessary3% cutoff, which does not affect either rejection because both measured noise thresholds exceed3%. Frozen sources and the patch remain under runs/dnn4/bert_priority.

Verified TK128 experiment: one deterministic max-legal-slot configuration on12 QKV stages; other stages and tile/split/store coordinates unchanged. No significant gain. Verified residual TMA experiment: only24 residual-reuse stages change store mode, with no additional shared allocation or changes to tiles/splits/TK/Slots.

Verified in-situ experiments use the six already authorized candidate programs, three fresh processes per variant and20 warmups/50 trace windows. Group sums select a coordinate, then the assembled nontrace program faces a whole-model P-PAIR gate. These experiments add no parameter scan.

in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate1, group5→candidate1.
in-situ B1: group0→candidate3, group1→candidate0, group2→candidate0, group3→candidate0, group4→candidate4.
in-situ P1 B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate4, group5→candidate1.
in-situ P1 B8: group0→candidate2, group1→candidate0, group2→candidate1, group3→candidate2, group4→candidate2, group5→candidate0.
consumer216 in-situ B8: group0→candidate0, group1→candidate0, group2→candidate0, group3→candidate0, group4→candidate1, group5→candidate0.
consumer216 in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate4, group5→candidate1.
cyclic in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate1, group5→candidate1.
repaired in-situ B8: group0→candidate0, group1→candidate4, group2→candidate0, group3→candidate0, group4→candidate1, group5→candidate4.
matrix existing in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate1, group5→candidate1.

Verified post-collection checks reject40/232: the89-case scan still spills28B on256x128 residual_ln_rowstats, and the full B16 program spills36B; it is never executed. B1 compiles without spill but has no significant gain. FFN1 128x256 uses one deterministic legal four-slot register-store variant; it regresses and is rejected. Distance1 has no significant gain.

Verified Repeat inline tables use grid-constant parameter storage instead of dependent global index reads. Unit5/5 and sanitizer memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5 pass; both whole-model comparisons fail DT-CODE. The implementation patch is archived and the production headers are restored. Four-element GELU scheduling also has no end-to-end gain: five full precision probes report0 bad values and preserve scalar bits; the132-CTA resident epilogue ratio is1.0072, and its full B16 comparison is rejected.

Verified single-layer Repeat12x5 is an explicit deviation from the prescribed two-layer Repeat6x10. Logical stage numbers preserve arena parity for odd loop bodies. Odd and even body unit cohorts pass5/5; each passes isolated memcheck1/1,initcheck1/1,racecheck5/5, synccheck5/5. B1 and B16 strict whole-model checks pass5/5 with identical cross-process output hashes. GPU2 B1 improves19.08%;GPU3 complete P-PAIR confirms19.09%. B16 and B4 have no significant gain. The B2 published rank1 coordinates decline Repeat and remain flat; a separate periodic B2 candidate spills4B and is never executed. The mode remains explicitly selected rather than enabled for unmeasured coordinates.

Verified B1 boundary trace sum535456ns exceeds fresh nontrace smoke498320ns by7.45%, so these windows are excluded from P-TUNE. Four of the six original complete variants spill and are never executed. Nontrace P-SEL uses the two legal loop variants, the accepted loop control and an existing distinct model-rank candidate; it retains the loop control. GPU3 P-DNN has median0.498176ms,p900.500640ms,process range0.013232ms. This is still above the unchanged0.4858ms BF16 goal. Failed or killed partial paired cohorts are discarded in full; occupancy exit75 retries the complete cohort.

Verified follow-up rejects16-bit inline parameter tables: odd-loop unit5/5 and sanitizer1/1/5/5 pass;full B1 FP32/byte checks5/5 pass, but the complete P-PAIR against the accepted loop is0.498656→0.507488ms,ratio1.017712,noise0.9369%. The prototype headers are restored. Earlier B8 canonical buffer aliases required retaining its exact flat program instead of failing. Four real CPU codegen cases cover B1 one/two-layer loops,B2 nonperiodic fallback and B8 alias fallback;all retain63 logical stages. No new B8 performance is claimed.

Verified P1 row normalization is explicitly selected only for B16: the side warp reduces the existing statistics in their original order and publishes cached (mean,inv) through32 individual release arrivals before consumer side_full. This deviates from consumer-side normalization in section5.2. The first prototype failed racecheck with89 errors and is rejected regardless of timing. The repaired implementation passes unit5/5 and isolated memcheck1/1,initcheck1/1,racecheck5/5, synccheck5/5. B1 loop and B16 whole-model FP32/byte checks each pass5/5. B16 GPU2/GPU3 complete paired gains are3.03%/2.71%;GPU3 P-DNN median1.421744ms, p901.446144ms,process range0.001824ms. This still exceeds the unchanged1.0869ms BF16 goal. Combining P1 normalization with the periodic B2 loop spills8B stores/32B loads and is never executed;the standalone periodic B2 loop also remains rejected.

Verified explicit scalar phase selection eliminates B1 local-memory instructions and passes odd-body unit5/5 and sanitizer1/1/5/5, but has no significant E2E gain; the prototype is archived and restored. B2 still spills28B and is never executed. Combining scalar phase with FFN2 Slots6 restores one-arena preissue and measures 2.03% faster, below its6.36% noise threshold;it is not admitted. A separate FFN2 split4/Slots6 variant regresses5.82% and is also rejected. Fixed BF16 targets and the50/200 measurement protocol remain unchanged.

Verified attention Rows64 B16 passes per-head unit5/5,sanitizer1/1/5/5 and whole-model FP32/byte checks5/5, but its0.60% gain over the accepted Rows128 P1-normalization control is below1.70% noise. Three-warp row normalization also passes unit5/5,sanitizer1/1/5/5 and B1/B16 model checks5/5;B1 regresses3.05%, and B16 improves0.43% below0.66% noise. Both mechanisms are rejected for performance admission;the three-warp implementation is archived and restored. Codegen exposes explicit SIDE_NORM and ATTENTION_ROWS settings while preserving the default source. A real three-profile CPU test checks all12 query-map boxes, all63 stage bindings and invalid setting rejection.

Verified opt-in register profile64/216 keeps the per-SMSP sum496 below the entry pool504. It deviates from the prescribed56/224 fallback. The periodic B2 single-layer loop now compiles without spill;focused odd-body unit5/5 and memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5 pass. Whole-model FP32/byte checks5/5 pass. GPU2 gain7.49% is below7.99% noise and is not admitted;the required GPU3 comparison confirms5.14% gain with0.50% noise and passes DT-CODE. B2 SASS falls from1868800B to275584B. Three existing B1 variants become nonspilling;their strict model checks5/5 pass,but G22 four-candidate P-SEL retains the previously accepted loop. One remaining B1 variant spills12B and is never executed. No new coordinate scan is added;the profile stays explicit. A real four-profile CPU test also verifies invalid register-budget rejection.

Verified B8 alias handling now gives each repeated operand role a virtual index when physical aliases vary across layers. Physical allocations and prefetch IDs stay unchanged. Six real CPU codegen cases certify63-stage coverage;B1 one/two-layer and accepted periodic B2 sources remain byte-identical. B8 one/two-layer contract tests each reject six remap corruptions and preserve the flat source. Virtual-ID units under56/224 and64/216 each pass5/5 and sanitizer1/1/5/5. The B8 model spills 4B stores/28B loads under56/224 and is never executed;64/216 compiles without spill and passes strict model checks5/5. GPU2 was cancelled before measurement because a separate baseline controller held its lock;that controller was left running. GPU3 P-PAIR is1.051232→1.070064ms,ratio1.017914,noise2.9649%;DT-CODE fails, so B8 keeps the published flat winner. The newly valid Repeat path remains opt-in.

Verified host-selected Repeat prefetch tables preserve prefetch distance and byte budgets, pass odd-body unit5/5,sanitizer1/1/5/5 and B1/B2 model checks5/5. Complete GPU3 P-PAIR gives B1 ratio1.039358,noise6.8132% and B2 ratio0.959454,noise5.6363%. Neither gain exceeds noise;the prototype header is archived and restored.

Verified middle-batch P1 normalization preserves accepted coordinates and passes strict model checks5/5. B2 GPU3 ratio0.950017 is below5.61% noise,so its old loop stays selected. B4 GPU3 ratio0.963260 beats1.94% noise;P-DNN median0.760672ms,p900.764224ms,range0.006784ms. The64/216 side-normalization unit also passes5/5 and sanitizer1/1/5/5. B8 ratio0.970675 gives2.9325% gain below2.9386% noise and is rejected. B4 remains above its fixed BF16 goal and retains the original anchor without re-anchoring.

Verified FFN2 TK128 retains split3 and the tile,using the largest legal six-slot ring. B1 model checks5/5 pass;GPU3 ratio1.033929,noise4.85% gives no gain,and accepted coordinates stay unchanged. Verified GELU reciprocal without Newton refinement passes the2^24-point grid and all finite BF16 values5/5 with max normalized error1.972e-7. The explicitly reserved GPU3 precision cohort is used; earlier scheduler-environment attempts are excluded. Unit5/5,sanitizer1/1/5/5 and B1/B16 model checks5/5 pass. The132-CTA three-task epilogue ratio is0.923058 over the current approximation on GPU3, but B1 E2E ratio1.031251 and B16 ratio0.991282 fail noise6.58%/1.60%. The prototype is archived and both production header and temporary gate option are restored.

Verified attention QK contraction-loop unrolling preserves MMA order and passes per-head unit5/5 plus sanitizer1/1/5/5. The B1 program compiles without spill and passes model checks5/5, but its0.63% gain is below7.25% noise. The B16 program spills16B stores/244B loads and is never executed. Initial pragma-macro compilation failures are retained separately. The prototype header and temporary gate option are restored.

Verified cooperative FFN1 TM256/TN128 uses two m64 fragments per WG,maximum legal four slots and register stores. Its exact NT12/spanning body passes unit5/5 and sanitizer1/1/5/5; the full B16 program is nonspilling and passes model checks5/5. GPU3 P-PAIR ratio1.145325 exceeds3.03% noise;reject and retain the accepted B16 coordinates. The B8 alias-safe loop combined with P1 normalization and64/216 also passes model checks5/5, but its0.73% gain fails2.44% noise and DT-CODE3%;retain the flat B8 winner.

GPU3 foreign occupancy delayed additional same-card G22 probes; only our pending schedulers were cancelled, and fresh candidate/control pairs moved to authorized GPU2. The separately identified B1/B2/B16 follow-ups do not silently replace the earlier frozen20-cell collection. That BERT batch collection completed on GPU3: B1/B2/B4/B8/B16 =0.614016/0.695504/0.790144/1.043264/1.463168ms. None beats its frozen BF16 target. CNN collection is complete and paused; existing CNN results and all60 baseline arms are reused. GPU ownership remains qualified by the container PID namespace. See DNN4_BERT_optimization.csv for identities, clocks, correctness receipts and raw timing sources.

Verified arithmetic remaps use compile-time checked constant, affine or period-two index columns; irregular columns retain lookup. Odd56/224 and alias64/216 units each pass5/5 and sanitizer1/1/5/5;both models pass strict5/5 without spill. Four real codegen cases and seven related ctests pass. B2 GPU3 P-PAIR 0.659808→0.611152ms beats1.35% noise;P-DNN 0.608016ms passes B2 admission. B1 has no significant gain;B16 Repeat plus P1 normalization also fails DT-CODE and keeps the flat control. GPU0 allocation-deviation diagnostics are separately identified and cannot replace G22 admission. The arithmetic feature remains default0 and is explicitly selected by admitted profiles.

Verified hardware-counter diagnostic: ncu2025.1.1 exits1 with ERR_NVGPUCTRPERM; no counter samples are available and no device/driver settings were changed. Fresh host CMake configuration also fails because the external MLIR installation lacks obj.MLIRCAPIDebug/Debug.cpp.o. The existing configured incremental build succeeds.

Verified statistics-TMA prototype maps even-M statistics as a BF16 byte view and loads all NT parts with one unswizzled3D TMA;odd M and NT1 keep Bulk. This explicitly deviates from per-part Bulk in section5.2. Focused units pass5/5,memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5;the complete second guarded cohort is used and the interrupted first is excluded. B16 compiles without spill and passes model5/5. GPU3 P-PAIR ratio1.030790 exceeds 0.94% noise. Reject;the prototype patch is archived and all production files are restored.

Verified P1 B8 in-situ follow-up: boundary overhead 3.682% passes5%;4/6 existing variants complete three fresh20/50 traces each. Spilled variants, if any, are excluded before execution and remain recorded. Group-window winners describe local selection;only the assembled nontrace program can qualify through the complete GPU3 end-to-end pair.
Verified assembled B8 P-PAIR ratio0.994313;noise1.929%;accepted=False.

Verified P1 B16 in-situ follow-up: boundary overhead 3.258% passes5%;6/6 existing variants complete three fresh20/50 traces each. Spilled variants, if any, are excluded before execution and remain recorded. Group-window winners describe local selection;only the assembled nontrace program can qualify through the complete GPU3 end-to-end pair.
Verified assembled B16 P-PAIR ratio0.998847;noise0.898%;accepted=False.

Verified FFN1 TK128 uses three slots and register stores on the12 FFN1 stages;all other accepted B16 coordinates remain unchanged. The exact 2048x3072x768,NT12 unit passes5/5,memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5;the whole model passes5/5 without spill. GPU3 P-PAIR ratio1.030299 regresses and is rejected. The regression unit remains as coverage for wide-K spanning slots.

Verified scalar phase plus arithmetic alias component passes unit5/5 and sanitizer1/1/5/5. The B8 Repeat/P1 whole-model build still spills 8/32B stores/loads under64/216;it is never executed. The prototype patch is archived and its production header is restored.

Verified TK128 single-store prototype explicitly deviates from section6.2: one8KiB staging box per WG replaces two and uses wait_group.read0 before each reuse. This fits three TK128 slots while retaining TMA writes. Exact FFN1 and cross-stage ordering cohorts each pass unit5/5 and sanitizer1/1/5/5;the whole model passes5/5 without spill. The first pair has351.287% noise and is inconclusive;all six processes remain recorded. A complete second pair measures ratio1.020020 against 0.980% noise. Reject and restore the production files;the archived patch retains the prototype.

Verified TN256 single-buffer follow-up retains TK64 and four slots on the12 FFN1 stages. The first model fails correctness because its serialized weight-packing recipes still use TN128;that artifact is never timed. A separate corrected artifact updates all12 recipes,compiles without spill and passes model5/5. The exact TN256 component passes unit5/5 and sanitizer1/1/5/5;cross-stage ordering reuses the prior17/17 single-buffer protocol gate. GPU2 never launches under foreign occupancy;GPU0 allocation-deviation P-PAIR gives ratio1.102471,noise2.093%. Reject and restore the prototype headers;no G22 admission.

Verified attention operand scheduling issues two independent shared matrix loads before their dependent MMAs,without changing any output accumulation order or Rows16 geometry. Per-head unit5/5 and sanitizer1/1/5/5 pass;the B1 model passes5/5 and its output bytes match the admitted control. SASS is259456B,maxR195,zero spill. GPU0 diagnostic P-PAIR gives ratio1.003505,noise1.157%. No significant end-to-end gain;archive and restore the isolated header.

Verified72/216 B8 re-selection qualifies3.304% boundary overhead;3/6 existing variants complete3/3 fresh traces. B8 deliberately uses reduced three-member coverage;this is a diagnostic deviation,not complete P-TUNE coverage. The assembled nontrace model passes5/5;GPU0 P-PAIR givesratio1.024397 versus2.903% noise. Admission and current artifacts remain unchanged.

Verified72/216 B16 re-selection qualifies2.234% boundary overhead;5/6 existing variants complete3/3 fresh traces. The assembled nontrace model passes5/5;GPU0 P-PAIR givesratio0.984507 versus3.323% noise. Admission and current artifacts remain unchanged.

Verified prescribed two-layer Repeat6x10 combined with arithmetic bindings and P1 retains the B16 coordinates and56/224 roles. Alias and P1 component gates each pass17/17;the model passes5/5 without spill. GPU0 diagnostic P-PAIR gives ratio1.008738 versus2.663% noise and is rejected. The B8 counterpart spills52/192B stores/loads and is never executed.

Verified boundary-trace cache is rejected:thread0 keeps the four markers in unused header padding and flushes at leave;three fresh GPU3 processes compare output bytes against the admitted control. Window sum differs by7.057% from nontrace,above5%,so these windows never select coordinates. The first static-shared variant exceeds the full dynamic-memory allocation and is rejected before launch. Both patches remain raw and the production header is restored. B1 instead tests existing whole-program candidates using nontrace P-SEL and P-PAIR.

Verified profile2 B8 boundary overhead is3.419% on GPU0. Only2/6 existing variants are nonspilling;each completes three fresh20/50 traces. Four spilled variants are never executed. Required candidate coverage is incomplete;no assembled selection or admission is claimed. GPU2 partial traces remain separate and are not mixed with the GPU0 windows.

Verified optional profile2 uses72/208 registers and explicit scalar phase selection only for that profile. Profiles0/1 retain their prior paths and the default remains0. Scalar/alias and P1 component cohorts each pass unit5/5 and sanitizer1/1/5/5;B2/B4/B8/B16 whole models each pass5/5 without spill. B8/B16 SASS is328576/395776B with zero local-memory instructions. The integrated B8 kernel has identical instruction bytes to its prototype. Seven related regression checks, real profile codegen and invalid-setting checks pass. GPU2 comparisons remain diagnostics. Complete GPU3 pairs subsequently admit B4/B8;exact-binary ledgers and three-process P-DNN results are in the current catalog. B2 fails its noise gate and B16 fails the firstRepeat3% gate. B8/B16 first-time Repeat admission still requires3% plus noise;B2/B4 already use accepted Repeat and require a gain beyond noise. Fixed BF16 goals and original anchors remain unchanged.

Verified B8 arithmetic Repeat plus P1 normalization fails the compilation gate under64/216 and prescribed56/224 (repeat_arithmetic_middle_v1/b8:8,24B stores/loads;repeat_arithmetic_b8_reg56_v1:32,104B stores/loads). Neither spilled binary is executed;profile2 later resolves this spill gate.

Verified B4 arithmetic Repeat plus P1 normalization retains the accepted coordinates. The64/216 profile gains2.89%,above2.63% noise but below DT-CODE3%,so it is rejected. Prescribed56/224 compiles without spill and passes strict model5/5. Its GPU3 P-PAIR 0.760288→0.733952ms gives 3.46% gain against 0.50% noise;the final P-DNN is recorded separately. Arithmetic and P1 component units pass5/5 and sanitizer1/1/5/5 separately.

Verified B4 admitted P-DNN median0.734352ms,p900.744128ms,process range0.007152ms. SASS and identities are retained with the default56/224 profile;no production header changes are needed. It remains above the fixed BF16 target and retains the original GPU3 anchor without re-anchoring.

Verified B2 P1 normalization plus the admitted arithmetic loop spills 8/16B stores/loads under64/216. The compilation gate rejects it before execution;the admitted arithmetic-only B2 profile remains selected.

Verified B2 current-profile boundary overhead 8.351% exceeds5%;those windows do not select stage coordinates. Four distinct nontrace whole programs use only original Phase Z candidates and the admitted control. Every candidate passes strict model5/5;spilled builds are never executed. G22 P-SEL winner=A,tie threshold=9.099%. The admitted control remains selected unless a separate fresh P-PAIR passes its unchanged noise gate.

Verified B4 current-profile boundary overhead 7.059% exceeds5%;those windows do not select stage coordinates. Four distinct nontrace whole programs use only original Phase Z candidates and the admitted control. Every candidate passes strict model5/5;spilled builds are never executed. G22 P-SEL winner=B,tie threshold=0.559%. The admitted control remains selected unless a separate fresh P-PAIR passes its unchanged noise gate.

Verified original thread-major and opt-in vector-major split-K layouts produce identical output bytes on six frozen shapes:odd rows,S3/4/6/16 and TK64/128. The first five accepted fresh original-layout processes use one frozen binary (four on GPU2,one on GPU3),each with guard0;all match the accepted coalesced-suite output hash. This is a functional allocation deviation with no mixed-card performance comparison.

Verified opt-in cyclic slot indexing and vector-major normal split-K partials retain the admitted B1 coordinates,56/224 register roles and ascending slice addition order. Integrated component gates pass17/17 on GPU3;the extended S2/TK128/odd-row suite passes17/17 on GPU2. The full model passes strict FP32 and cross-process byte checks5/5. G22 P-PAIR0.499216→0.478000ms passes0.904% noise;P-DNN 0.464496ms,p900.477696ms,process range0.012608ms. The exact accepted binary has a P-TRACE1 ledger. The same B2 mechanism fails its paired noise gate and is not enabled there. Defaults,original anchors,CNN results and frozen20-cell Phase Z remain unchanged.

Verified current accepted artifacts (GPU3;50/200;three-process P-DNN). B1/B2/B4/B8/B16 are follow-ups that retain the original anchor without re-anchoring. Fixed BF16 targets reached for B1. DNN4_BERT_current.csv records complete identities,profiles and evidence paths;DNN4_BERT_current_ledger.csv preserves all315 current stage rows with exact accepted reference hashes and coordinates.

| Batch | Median ms | p90 ms | Process range ms | Fixed BF16 ms | Ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.464496 | 0.477696 | 0.012608 | 0.4858 | 0.9561 |
| 2 | 0.608016 | 0.629664 | 0.020512 | 0.5171 | 1.1758 |
| 4 | 0.706304 | 0.715328 | 0.007824 | 0.5905 | 1.1961 |
| 8 | 1.016400 | 1.025344 | 0.004272 | 0.7457 | 1.3630 |
| 16 | 1.327520 | 1.345760 | 0.007824 | 1.0869 | 1.2214 |

Verified largest three stages for each current unmet BERT cell (P-TRACE1 attribution only;segments can overlap and are not timing results). Each ledger checks the exact accepted reference binary SHA256.

| Batch | Largest stage | Second | Third |
| --- | --- | --- | --- |
| 2 | s5 gemm 18.82us (main_loop=11.10,slot_wait=3.74) | s30 gemm 18.66us (main_loop=10.05,epilogue=4.19) | s35 gemm 18.30us (main_loop=10.21,epilogue=4.29) |
| 4 | s24 gemm 19.33us (epilogue=10.37,main_loop=8.58) | s40 gemm 19.33us (main_loop=14.98,slot_wait=4.96) | s25 gemm 18.94us (main_loop=15.62,slot_wait=5.12) |
| 8 | s14 gemm 30.53us (main_loop=15.74,epilogue=12.96) | s59 gemm 30.14us (main_loop=15.58,epilogue=13.18) | s49 gemm 30.05us (main_loop=15.33,epilogue=12.83) |
| 16 | s4 gemm 42.05us (main_loop=23.46,epilogue=16.03) | s19 gemm 41.63us (main_loop=24.03,epilogue=14.75) | s29 gemm 41.18us (main_loop=24.61,epilogue=13.92) |
