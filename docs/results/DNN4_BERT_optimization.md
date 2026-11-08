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

Verified attention experiment: QK/PV columns were distributed across eight consumer warps without changing any output accumulation order. Per-head unit5/5 and sanitizer memcheck1/1, initcheck1/1, racecheck5/5, synccheck5/5 passed; whole-model outputs also matched the control byte for byte. Neither timing gain exceeded process variability, so the original header was restored. The initial driver added an unnecessary3% cutoff, which does not affect either rejection because both measured noise thresholds exceed3%. Frozen sources and the patch remain under runs/dnn4/bert_priority.

Verified TK128 experiment: one deterministic max-legal-slot configuration on12 QKV stages; other stages and tile/split/store coordinates unchanged. No significant gain. Verified residual TMA experiment: only24 residual-reuse stages change store mode, with no additional shared allocation or changes to tiles/splits/TK/Slots.

Verified in-situ experiments use the six already authorized candidate programs, three fresh processes per variant and20 warmups/50 trace windows. Group sums select a coordinate, then the assembled nontrace program faces a whole-model P-PAIR gate. These experiments add no parameter scan.

in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate1, group5→candidate1.
in-situ B1: group0→candidate3, group1→candidate0, group2→candidate0, group3→candidate0, group4→candidate4.

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

Verified B8 arithmetic Repeat plus P1 normalization fails the compilation gate under64/216 and prescribed56/224 (repeat_arithmetic_middle_v1/b8:8,24B stores/loads;repeat_arithmetic_b8_reg56_v1:32,104B stores/loads). Neither spilled binary is executed;B8 retains its accepted flat program.

Verified B4 arithmetic Repeat plus P1 normalization retains the accepted coordinates. The64/216 profile gains2.89%,above2.63% noise but below DT-CODE3%,so it is rejected. Prescribed56/224 compiles without spill and passes strict model5/5. Its GPU3 P-PAIR 0.760288→0.733952ms gives 3.46% gain against 0.50% noise;the final P-DNN is recorded separately. Arithmetic and P1 component units pass5/5 and sanitizer1/1/5/5 separately.

Verified B4 admitted P-DNN median0.734352ms,p900.744128ms,process range0.007152ms. SASS and identities are retained with the default56/224 profile;no production header changes are needed. It remains above the fixed BF16 target and retains the original GPU3 anchor without re-anchoring.

8Verified B2 P1 normalization plus the admitted arithmetic loop spills /16B stores/loads under64/216. The compilation gate rejects it before execution;the admitted arithmetic-only B2 profile remains selected.

Verified current accepted artifacts (GPU3;50/200;three-process P-DNN). B8 reuses the original measurements;B1/B2/B4/B16 are follow-ups that retain the original anchor without re-anchoring. All fixed BF16 targets remain unmet. DNN4_BERT_current.csv records complete identities,profiles and evidence paths.

| Batch | Median ms | p90 ms | Process range ms | Fixed BF16 ms | Ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.498176 | 0.500640 | 0.013232 | 0.4858 | 1.0254 |
| 2 | 0.608016 | 0.629664 | 0.020512 | 0.5171 | 1.1758 |
| 4 | 0.734352 | 0.744128 | 0.007152 | 0.5905 | 1.2436 |
| 8 | 1.043264 | 1.055872 | 0.009120 | 0.7457 | 1.3990 |
| 16 | 1.421744 | 1.446144 | 0.001824 | 1.0869 | 1.3081 |

Verified largest three stages for each current unmet BERT cell (P-TRACE1 attribution only;segments can overlap and are not timing results). Each ledger checks the exact accepted reference binary SHA256.

| Batch | Largest stage | Second | Third |
| --- | --- | --- | --- |
| 1 | s5 gemm 17.57us (main_loop=9.28,slot_wait=4.83) | s50 gemm 16.70us (main_loop=9.28,slot_wait=3.94) | s60 gemm 16.22us (main_loop=8.35,slot_wait=3.74) |
| 2 | s5 gemm 18.82us (main_loop=11.10,slot_wait=3.74) | s30 gemm 18.66us (main_loop=10.05,epilogue=4.19) | s35 gemm 18.30us (main_loop=10.21,epilogue=4.29) |
| 4 | s15 gemm 19.42us (main_loop=15.42,slot_wait=5.95) | s4 gemm 19.07us (epilogue=9.50,main_loop=8.29) | s5 gemm 19.07us (main_loop=15.26,slot_wait=6.40) |
| 8 | s29 gemm 31.33us (main_loop=15.42,epilogue=13.66) | s34 gemm 31.20us (main_loop=15.78,epilogue=13.66) | s19 gemm 31.01us (main_loop=15.36,epilogue=13.47) |
| 16 | s54 gemm 44.10us (main_loop=22.75,epilogue=17.66) | s4 gemm 44.06us (main_loop=23.07,epilogue=18.27) | s19 gemm 44.03us (main_loop=24.00,epilogue=17.02) |
