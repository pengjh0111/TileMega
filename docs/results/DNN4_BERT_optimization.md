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
| FFN1 PP matrix B16 | 3 | 1.332672 | 1.511520 | -13.420 | 1.364 | False |
| FFN1 PP P1 matrix B16 | 3 | 1.327824 | 1.489680 | -12.190 | 2.627 | False |
| attention vector stores B8 | 3 | 1.009888 | 0.998048 | +1.172 | 2.598 | False |
| attention vector stores B16 | 3 | 1.330752 | 1.311216 | +1.468 | 1.335 | True |
| combined attention B4 | 3 | 0.712624 | 0.693728 | +2.652 | 2.820 | False |
| combined attention B8 | 3 | 1.017120 | 0.995744 | +2.102 | 2.640 | False |
| combined attention B16 | 3 | 1.307184 | 1.297856 | +0.714 | 1.664 | False |
| FFN1 TMA192 matrix B8 | 3 | 1.019792 | 1.017264 | +0.248 | 1.258 | False |
| FFN1 TMA192 matrix B16 | 3 | 1.310288 | 1.356144 | -3.500 | 0.538 | False |
| FFN2 TMA256x64 matrix B16 | 3 | 1.310656 | 1.359696 | -3.742 | 0.752 | False |
| cyclic phase72 B4 | 3 | 0.712832 | 0.719360 | -0.916 | 2.658 | False |
| FFN1 reg64 side216 B2 | 3 | 0.610384 | 0.597888 | +2.047 | 0.500 | True |
| FFN2 reg64 split4 B2 | 3 | 0.597776 | 0.723808 | -21.083 | 9.946 | False |
| FFN1 TMA128 phase72 B4 | 3 | 0.712848 | 0.741712 | -4.049 | 2.415 | False |
| FFN1 TMA256 matrix B16 | 3 | 1.313136 | 1.457424 | -10.988 | 0.585 | False |
| QKV TMA192 matrix B16 | 3 | 1.311696 | 1.341520 | -2.274 | 1.076 | False |
| FFN1 TMA128 matrix B8 | 3 | 1.019552 | 1.011648 | +0.775 | 3.796 | False |
| FFN1 PP full-wait repair B16 | 3 | 1.312176 | 1.409968 | -7.453 | 2.912 | False |
| FFN2 TMA128 split2 B8 | 3 | 1.017536 | 1.212080 | -19.119 | 2.409 | False |
| FFN2 TMA128 split2 coalesced B8 | 3 | 1.019584 | 1.085408 | -6.456 | 1.427 | False |
| FFN1 M64N128 B8 | 3 | 1.016048 | 1.042416 | -2.595 | 2.460 | False |
| FFN1 PP nonblocking side B16 | 3 | 1.309920 | 1.408976 | -7.562 | 1.056 | False |
| Current coalesced B2 | 3 | 0.597728 | 0.578048 | +3.292 | 13.663 | False |
| cached-boundary216 in-situ B4 | 3 | 0.706016 | 0.678688 | +3.871 | 1.600 | True |
| Current prefetch1 B16 | 3 | 1.310240 | 1.297216 | +0.994 | 2.341 | False |
| FFN2 register store B8 | 3 | 1.016416 | 1.018768 | -0.231 | 1.363 | False |
| phase72 B2 GPU2 diagnostic | 2 | 0.628096 | 0.599952 | +4.481 | 6.669 | False |
| phase72 B4 GPU2 diagnostic | 2 | 0.736144 | 0.710800 | +3.443 | 2.339 | True |
| phase72 B8 GPU2 diagnostic | 2 | 1.052320 | 1.009456 | +4.073 | 2.349 | True |
| phase72 B16 GPU2 diagnostic | 2 | 1.426560 | 1.399680 | +1.884 | 1.923 | False |
| phase72 B2 G22 | 3 | 0.627840 | 0.601776 | +4.151 | 6.721 | False |
| phase72 B4 G22 | 3 | 0.733344 | 0.712320 | +2.867 | 2.540 | True |
| phase72 B8 G22 | 3 | 1.051376 | 1.019376 | +3.044 | 0.735 | True |
| phase72 B16 G22 | 3 | 1.423520 | 1.399776 | +1.668 | 1.059 | False |
| fixed-candidate216 B4 revalidation | 3 | 0.748320 | 0.678592 | +9.318 | 11.135 | False |
| partial TN96 store B8 prototype | 3 | 1.016624 | 1.018784 | -0.212 | 2.276 | False |
| partial TN96 matrix store B8 prototype | 3 | 1.019632 | 0.998592 | +2.063 | 2.928 | False |
| QKV four-slot B2 | 3 | 0.603536 | 0.596848 | +1.108 | 10.954 | False |
| QKV seven-slot B4 | 3 | 0.730912 | 0.738096 | -0.983 | 2.693 | False |
| QKV plus attention stores B2 | 3 | 0.572432 | 0.591184 | -3.276 | 8.385 | False |
| first WGMMA initialization B16 | 3 | 1.319024 | 1.307904 | +0.843 | 2.120 | False |
| fixed H800 grid B16 | 3 | 1.315328 | 1.295936 | +1.474 | 0.659 | True |
| fixed grid producer recomputation B8 | 3 | 1.019888 | 0.989792 | +2.951 | 0.693 | True |
| producer plus four-slot QKV B2 | 3 | 0.572096 | 0.575120 | -0.529 | 12.060 | False |
| producer recomputation B2 | 3 | 0.603760 | 0.546592 | +9.469 | 10.478 | False |
| producer recomputation B4 | 3 | 0.731296 | 0.703984 | +3.735 | 0.621 | True |
| producer recomputation B16 | 3 | 1.296816 | 1.293920 | +0.223 | 2.016 | False |
| paired residual read B16 prototype | 3 | 1.307504 | 1.310624 | -0.239 | 1.819 | False |
| paired residual read B16 GPU0 diagnostic | 0 | 1.319680 | 1.313648 | +0.457 | 0.548 | False |
| full-tile B8 | 3 | 0.978864 | 0.950512 | +2.896 | 0.500 | True |
| full-tile B16 | 3 | 1.295728 | 1.282784 | +0.999 | 0.878 | True |
| full-tile B2 | 3 | 0.577040 | 0.543808 | +5.759 | 5.961 | False |
| full-tile B4 | 3 | 0.681776 | 0.635504 | +6.787 | 1.502 | True |
| full-tile pipeline B2 | 3 | 0.577264 | 0.518016 | +10.264 | 10.915 | False |
| full-bounds prefetch1 B8 | 3 | 0.948320 | 0.928992 | +2.038 | 2.433 | False |
| full-bounds pipeline prefetch1 B2 | 3 | 0.569136 | 0.529936 | +6.888 | 11.262 | False |
| full-tile stores B8 | 3 | 0.939984 | 0.914528 | +2.708 | 2.359 | True |
| full-tile stores B4 | 3 | 0.644480 | 0.628480 | +2.483 | 3.111 | False |
| pipeline reductions B2 | 3 | 0.568672 | 0.508128 | +10.647 | 10.607 | True |
| full-store bounds B8 | 3 | 0.922384 | 0.920976 | +0.153 | 2.151 | False |
| full-profile cyclic B4 | 3 | 0.641984 | 0.640176 | +0.282 | 2.164 | False |
| GELU quartet B16 | 3 | 1.273744 | 1.284864 | -0.873 | 3.377 | False |
| relative descriptors B16 | 3 | 1.283648 | 1.281904 | +0.136 | 1.246 | False |
| two-row statistics plus recompute B16 | 3 | 1.281712 | 1.268032 | +1.067 | 1.592 | False |
| repaired catalog candidate5 B16 | 3 | 1.282736 | 1.478768 | -15.282 | 2.115 | False |
| statistics read ahead B16 | 3 | 1.282160 | 1.286784 | -0.361 | 1.641 | False |
| shared terminal FFN2 B16 | 3 | 1.279280 | 1.299168 | -1.555 | 2.749 | False |
| main-loop pair B4 | 3 | 0.654688 | 0.693184 | -5.880 | 0.587 | False |
| direct statistics B16 | 3 | 1.270656 | 1.273568 | -0.229 | 3.496 | False |

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
cached-boundary216 in-situ B4: group0→candidate0, group1→candidate0, group2→candidate5, group3→candidate0, group4→candidate0, group5→candidate0.

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

Verified B1 boundary-trace cache is rejected:thread0 keeps the four markers in unused header padding and flushes at leave;three fresh GPU3 processes compare output bytes against the admitted control. Window sum differs by7.057% from nontrace,above5%,so these windows never select coordinates. The first static-shared variant exceeds the full dynamic-memory allocation and is rejected before launch. Both patches remain raw and the production header is restored. B1 instead tests existing whole-program candidates using nontrace P-SEL and P-PAIR.

Verified current B2 reuses the exact prior header-padding boundary cache;nontrace instruction words equal the accepted binary. Three fresh guarded GPU3 trace processes compare output bytes with that binary;this is diagnostic coverage,not five-process P-CORR. Boundary sum versus fresh nontrace P-SMOKE differs by13.416%;fails5%;these windows never select coordinates. The production marker header and current artifacts remain unchanged.

Verified current B4 reuses the exact prior header-padding boundary cache;nontrace instruction words equal the accepted binary. Three fresh guarded GPU3 trace processes compare output bytes with that binary;this is diagnostic coverage,not five-process P-CORR. Boundary sum versus fresh nontrace P-SMOKE differs by2.175%;qualifies for existing-candidate in-situ selection. The production marker header and current artifacts remain unchanged.

Verified current72/208 B4 cached-boundary qualification passes, but only3/6 fixed candidates have both production and boundary builds without spill. Candidate5 trace is legal,but its nontrace build spills8/8B and never executes. The other two rejected trace variants spill52/104B and96/180B. Three legal variants each pass model5/5 and three matched-coordinate byte-checked trace processes. Coverage below four prohibits an assembled selection;the existing72/216 profile is qualified separately.

Verified CPU-only B8 register-profile3 follow-up preserves the six existing coordinate variants. Only0/1/2 have zero-spill production and boundary builds;3/4/5 spill. Coverage remains3/6,below four;these partial windows cannot assemble stage coordinates. A separate whole-program fallback may compare the three legal programs with the incumbent after strict model5/5;an ambiguous selection retains the incumbent.

Verified B4 fixed-candidate72/216 prototype on historical GPU3: P-PAIR0.706016→0.678688ms;gain3.871% versus1.600% noise. P-DNN0.678704ms. This remains a prototype;the current continuation separately checks four-program selection,explicit-profile components,independent paired timing and final admission. It does not replace the current accepted B4 row.

Verified B2 trace prototype checks first_ready only on the first local GEMM task;its clockpoint and four-marker layout are preserved. Production instruction words exactly match the accepted control;trace compilation has zero spill. Hardware qualification was skipped before all three trace processes: all48 normal GEMM stages have at most one task per CTA,so the first-task predicate removes no repeated baseline checks. The fresh GPU0 reference smoke0.572192ms is diagnostic only;no new performance is admitted.

Verified CPU-only B2 first-task trace preparation keeps all six existing coordinate variants and the accepted72/216 profile. Only0/1 have zero-spill production and boundary builds;2/3/4/5 spill. Coverage2/6 prohibits assembled selection;baseline trace qualification is independent.

Verified explicit72/216 alias component on GPU0 passes unit5/5 and sanitizer mem1/1,init1/1,race5/5,sync5/5 in seventeen independent guarded reservations with released gaps. The resource.py generic entry was renamed worker.py after two model jobs failed before measurement because it shadowed the standard library;failed logs remain raw. No numerical or timing thresholds change,and completed component work is reused after the server restart.

Verified private B8 producer operand lifetime trial keeps side bindings inside P1 and preserves the fixed coordinates. Candidates3/4 still spill 24/32B and28/52B stores/loads versus20/28B and24/48B before the change. Reject at the CPU gate;no trace builds,hardware execution or production change.

Verified B2 fixed4 producer index lifetime drafts preserve coordinates and arithmetic. Recomputing task counts reduces spill8/8B to4/4B;also recomputing task addresses returns to8/8B. Both fail the zero-spill gate and never execute on GPU. The main header is unchanged.

Stated2026-11-12 allocation:all four H800 cards0–3 are authorized;intermittent gaps are no longer required. New schedules preserve the six-sample guard,300s occupancy cooldown and /root/python entry paths. B4/B8 timing uses original GPU3;functional checks may use the other cards. Previously paused queues stay retired and completed checks are reused.

Verified GPU3 idle footprint changed from4MiB to6MiB: six owner-free samples5s apart record6MiB and idle SM345MHz. A separate guard entry updates only that measured floor;original workers remain frozen. Locks,six-sample preflight,foreign-process rejection,cooldowns and measurement thresholds remain in force. The pending body cohort has no accepted processes;its replacement inherits the cooldown. Completed component/model cohorts are reused. Timing continues to record actual start/end SM clocks.

Verified foreign GPU1 occupancy prevents dispatch of one B8 functional cohort. All five jobs have zero attempts and no guarded children when its controllers retire. The cohort relocates to GPU0 under new original guards;the other two live cohorts and all completed checks are retained. Performance timing stays on GPU3.

Verified private fixed-coordinate B16 residual-read prototype compiles without spill;SASS355968B. One aligned shared32 read supplies each adjacent BF16 pair while preserving per-element arithmetic. CPU compilation establishes feasibility only;complete unit/sanitizer,model5/5 and same-GPU3 whole-model gates decide admission.

Verified explicit72/216 P1 normalization component on GPU0:unit5/5,memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5;seventeen distinct native processes have identical output hashes and clean guard receipts. Four completed unit processes are reused;the remaining thirteen run under one original guarded reservation after the inherited cooldown. These are functional prerequisites;whole-model correctness and same-GPU3 timing decide performance admission.

Verified private paired-residual-read component on GPU2:unit5/5,memcheck1/1,initcheck1/1,racecheck5/5,synccheck5/5;seventeen distinct native processes have identical output hashes and clean guard receipts. All seventeen run under one original guarded reservation. These are functional prerequisites;whole-model correctness and same-GPU3 timing decide performance admission.

Verified b8_corr1 on GPU0/3:strict FP32 model5/5,minimum cosine0.99995768,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified b8_corr2 on GPU0/3:strict FP32 model5/5,minimum cosine0.99995768,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified b8_corr3 on GPU2:strict FP32 model5/5,minimum cosine0.99995768,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified residual_corr on GPU0/2:strict FP32 model5/5,minimum cosine0.99995708,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified store96_corr on GPU3:strict FP32 model5/5,minimum cosine0.99995768,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified store96_matrix_corr on GPU3:strict FP32 model5/5,minimum cosine0.99995768,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified qkv_slots4_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995983,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified qkv_slots7_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995583,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified qkv_attention_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995983,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified mma_zero_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995708,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified fixed_grid_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995708,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified grid_recount_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995768,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified grid_recount_b2_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995983,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified grid_recount_b4_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995583,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified grid_recount_b16_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995708,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified recount_qkv4_corr on GPU0:strict FP32 model5/5,minimum cosine0.99995983,identical output bytes across all five fresh processes. These completed correctness checks do not establish whole-model performance or production admission.

Verified B4 fixed-candidate revalidation completes fresh GPU3 P-SEL8/8 and independent P-PAIR6/6. The selected candidate measures 0.678592ms against0.748320ms;gain9.318% is below11.135% process-noise threshold. Retain all six processes and the earlier prototype evidence;do not replace the admitted B4 result or rerun this pair to seek acceptance.

Verified CPU-only residual-pair follow-up on the existing fixed B8 candidate3 retains its coordinates and72/216 profile. It still spills 20/28B stores/loads,matching the original candidate,and never executes on GPU. The three legal whole-program alternatives remain unchanged.

Verified B8 nontrace whole-program P-SEL8/8 on GPU3: A=1.017400ms,D=1.049176ms,B=1.050376ms,C=1.065768ms;tied=False,noise0.626%. The incumbent wins and remains selected. Incomplete stage-window coverage supplies no tie priority or assembled selection.

Verified private partial TN96 store extends section6.2 under the user-authorized implementation discretion:only the first64 columns use TMA;the remaining32 use generic stores,with the full-box bound enforced inside each WG tile. The12 B8 FFN1 stages change Slots7 to6 to fit 218112B including staging;Slots7 would require246784B and is illegal. The current72/208 whole-model build spills4/20B and never executes. The existing72/216 profile compiles without spill,SASS331008B. Regeneration with unchanged coordinates/profile matches all41072 encoded instruction words of the incumbent. Solver enumeration is unchanged;exact-profile body/order gates,model5/5 and a same-GPU3 gain above3% and noise are required before enabling this extension.
Verified store96_body:unit5/5,mem1/1,init1/1,race5/5,sync5/5 on GPU3;all seventeen fresh processes have identical output hashes. Functional evidence does not establish E2E benefit.
Verified store96_profile3_ordering:unit5/5,mem1/1,init1/1,race5/5,sync5/5 on GPU3;all seventeen fresh processes have identical output hashes. Functional evidence does not establish E2E benefit.
Verified store96_profile3_body:unit5/5,mem1/1,init1/1,race5/5,sync5/5 on GPU3;all seventeen fresh processes have identical output hashes. Functional evidence does not establish E2E benefit.
Verified store96_matrix_ordering:unit5/5,mem1/1,init1/1,race5/5,sync5/5 on GPU3;all seventeen fresh processes have identical output hashes. Functional evidence does not establish E2E benefit.
Verified store96_matrix_body:unit5/5,mem1/1,init1/1,race5/5,sync5/5 on GPU3;all seventeen fresh processes have identical output hashes. Functional evidence does not establish E2E benefit.
Verified rejected partial-store attribution:the12 FFN1 windows sum357.152→362.080us;main-loop CTA maxima sum184.704→185.248us;epilogue149.568→155.136us;store47.296→46.304us. The historical same-GPU3 reference and fresh candidate trace are diagnostic;segment maxima overlap. These data do not support a large main-loop penalty from losing one slot;the store reduction is small and total epilogue increases. Nontrace pair rejects the candidate.
Verified CPU-only follow-up retains those coordinates and72/216 roles while enabling the existing matrix-store1 path;SASS321024B,zero spill. The flag affects both12 partial FFN1 stores and23 existing TMA stages. This tests the observed epilogue cost without a slot/tile scan;separate matrix-store1 component/model gates and the predeclared3%-plus-noise whole-model gate decide acceptance.
Verified CPU-only B16 preparation applies the same partial store to11 repeated QKV stages,keeping the first NT1 QKV and the admitted profile unchanged. The control matches all44736 encoded words;the candidate has zero spill and360960B SASS. No GPU checks or performance are claimed;this follow-up is conditional on B8 mechanism qualification and needs separate matrix-store1 gates.

Verified CPU-only B2 pipeline follow-up applies the seed one-arena maximum to the admitted128x64 QKV tile:11 repeated stages change Slots3 to4;the first QKV and all other coordinates/profile remain fixed. The canonical resource function gives87040/111616/136192B for3/4/5 slots;four fits one arena and five spans. Current QKV attribution averages8.33us main loop with4.55us slot wait. The fixed candidate has zero spill and272128B SASS;regenerated control matches all33120 encoded words. This is one seed-rule choice,not a parameter scan;component17/model5 and independent same-GPU3 E2E gain above noise decide acceptance.

Verified CPU-only B4 pipeline follow-up changes only11 repeated 128x96 QKV stages from3 to7 slots. Prompt4.5 selects spanning because the one-arena maximum is3;seven slots fit214016B and eight do not fit. The previous FFN2 already spans and attention cannot preissue,so this change loses no existing adjacent B-page preissue. The72/208 profile is unchanged;zero spill,SASS282624B,control regeneration matches all35296 encoded words. Complete component/model checks and an independent same-GPU3 pair are required for acceptance.

Verified partial TN96 matrix-store B8 completes component/model gates and GPU3 P-PAIR6/6:reference1.019632ms,candidate0.998592ms,gain2.063%,noise2.928%,minimum gain3.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified four-slot QKV B2 completes component/model gates and GPU3 P-PAIR6/6:reference0.603536ms,candidate0.596848ms,gain1.108%,noise10.954%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified seven-slot QKV B4 completes component/model gates and GPU3 P-PAIR6/6:reference0.730912ms,candidate0.738096ms,gain-0.983%,noise2.693%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified QKV plus attention vector stores B2 completes component/model gates and GPU3 P-PAIR6/6:reference0.572432ms,candidate0.591184ms,gain-3.276%,noise8.385%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified first WGMMA initialization B16 completes component/model gates and GPU3 P-PAIR6/6:reference1.319024ms,candidate1.307904ms,gain0.843%,noise2.120%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified producer plus four-slot QKV B2 completes component/model gates and GPU3 P-PAIR6/6:reference0.572096ms,candidate0.575120ms,gain-0.529%,noise12.060%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified producer recomputation B2 completes component/model gates and GPU3 P-PAIR6/6:reference0.603760ms,candidate0.546592ms,gain9.469%,noise10.478%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified producer recomputation B16 completes component/model gates and GPU3 P-PAIR6/6:reference1.296816ms,candidate1.293920ms,gain0.223%,noise2.016%,minimum gain0.0%. Reject;retain the incumbent and all measured processes. No final collection or production admission follows this pair.

Verified B2 ring attribution across11 QKV stages:windows130.848→120.800us;main91.456→82.304us;slot wait50.048→38.848us;first ready25.888→21.920us. Historical accepted GPU3 trace and fresh exact-candidate trace are diagnostic;segment maxima overlap. They do not override the independent nontrace pair or qualify a new performance result.

Verified B4 ring attribution across11 QKV stages:windows147.168→147.744us;main99.456→101.760us;slot wait47.424→46.080us;first ready20.192→24.896us. Historical accepted GPU3 trace and fresh exact-candidate trace are diagnostic;segment maxima overlap. They do not override the independent nontrace pair or qualify a new performance result.
Verified conditional B16 partial-store work remains CPU-only because the B8 mechanism did not pass its end-to-end gate;no B16 GPU job is queued.

Verified private default-off first-WGMMA initialization uses the CUTLASS explicit first-slot prologue:each m64 fragment starts with ScaleOutZero,then retains the original k16 order,slot and release protocol. The dynamic-predicate prototype leaves clearing instructions and never executes on GPU. The explicit prologue reduces static CS2R zero instructions289→56;SASS357888→362880B,zero spill,maxR207. Default0 matches all44736 incumbent instruction words. Coordinates and the accepted B16 profile stay fixed;end-to-end benefit is unproven.
Verified five fresh control processes and the candidate unit5/5 plus mem1/1,init1/1,race5/5,sync5/5 have identical output hashes on GPU0. The five shapes cover GELU,residual statistics,TM256 two-fragment ownership,TK128/split2,odd rows and ring wrap. These component checks do not establish an end-to-end gain.
Verified B16 strict model5/5 also matches the incumbent output bytes. The independent GPU3 pair rejects this private candidate; the production headers and accepted B16 result remain unchanged.

Verified private fixed132 H800 grid specialization retains the B16 coordinates,register profile,task ownership and arithmetic order. It folds runtime task division and prefetch partitioning; SASS357888→343168B,static RCP97→76,zero spill. Default0 matches all44736 incumbent instruction words. This extends the prompt's index specialization under the user-authorized implementation discretion;separate functional and E2E gates decide admission.
Verified five fresh CPU-only host processes reject grids 0/1/131/133/264 in both serving modes before changing the epoch. GPUs are hidden and no valid-grid launch is attempted.
Verified fixed-grid body on GPU0: unit5/5,mem1/1,init1/1,race5/5,sync5/5;output bytes match the frozen control. These are functional gates.
Verified fixed-grid attention on GPU3: unit5/5,mem1/1,init1/1,race5/5,sync5/5;output bytes match the frozen control. These are functional gates.
Verified CPU-only B8 portability check spills20/20B under its current72/208 profile. It is rejected before GPU execution;no B8 hardware result is claimed.
Verified existing72/216 also spills20/20B for fixed-grid B8; that CPU-only follow-up is rejected without a register sweep.
Verified fixed-grid B16 is admitted after main on/off instruction equivalence,repository fixture equivalence, host rejection5/5,strict model5/5,independent GPU3 pair and an exact63-stage ledger. P-DNN median1.295600ms,p901.313632ms,process range0.011296ms. Other four cells and original anchors remain unchanged. CMake regeneration is blocked by a missing installed MLIRCAPIDebug object;the unchanged certified cached archive and rebuilt codegen object pass real codegen14profiles/8invalid-settings checks,41admission tests and two related CTest checks. The full CMake build is not claimed.

Verified private B8 producer repair recomputes the CTA index at task/address uses. Earlier fixed-grid spills all occur after producer deallocation to72registers;the larger consumer budget does not resolve them. Current72/208 coordinates/profile now compile with zero spill,SASS328576→311680B,RCP77→56. Default0 matches all41072 current B8 words and all42896 admitted B16 words. This is a compiler register-lifetime repair;independent whole-model performance decides acceptance.
Verified producer-recomputation body on GPU0: unit5/5,mem1/1,init1/1,race5/5,sync5/5;output bytes match the frozen shape controls. These are functional gates.
Verified producer-recomputation attention on GPU3: unit5/5,mem1/1,init1/1,race5/5,sync5/5;output bytes match the frozen shape controls. These are functional gates.

Verified fixed-grid-only B2 under its current72/216 profile spills144/188B. It is rejected at the CPU gate and does not execute on GPU.

Verified the original disabled BlockIndex helper changes B2 instructions and is rejected before candidate compilation or GPU execution. The repaired compile-time macro restores the literal original expression when disabled. All five current BERT kernels match their original instruction words;enabled B8 matches all38960 words of the independently qualified prototype. No numerical or timing process is repeated for this source repair.

Verified B8 producer recomputation is integrated and admitted: main on/off kernels match all38960/41072 tested instruction words;committed body/attention fixtures match the two17-process gates;the unchanged fixed-grid runtime retains its five-process host rejection proof. Real codegen16profiles/9invalid settings and 46admission tests pass. GPU3 independent pair gains2.951% versus0.693% noise;P-DNN0.977680ms,p900.997472ms,process range0.014976ms. The exact tested binary and63-stage ledger are retained. Other cells and original anchors remain unchanged.

Verified CPU-only B2 uses the same producer repair with current72/216 coordinates/profile:zero spill,SASS264960→250752B.The repaired default path matches all33120 incumbent words. Separate functional and independent GPU3 timing gates decide acceptance;code size alone does not admit this candidate.

Verified B4 reuses the exact profile2 body/attention17-process gates and completes fresh model5/5 with unchanged output bytes. Main on/off33072/35296 instruction words match the paired candidate/incumbent;the exact GPU3 trace has63 stages. Independent pair gains3.735% versus0.621% noise. Admitted P-DNN=0.704128ms,p900.708320ms,process range0.002160ms. Other cells and original anchors remain unchanged.

Verified CPU-only B8 check:all six exact existing repaired P-TUNE coordinate programs compile without spills under the currently admitted72/208 producer profile. No new coordinate or register sweep is introduced. Legal compilation alone does not select a program;boundary qualification,strict model checks and an independent current-incumbent pair remain required.

Verified private boundary-only CTA recomputation passes GPU0 unit5/5 and sanitizer1/1/5/5. All38960 production and44288 full-trace instruction words are preserved. The uniform boundary source makes five of six existing candidates legal;candidate2 still spills8/8B. Reusing one CTA value also leaves8/8B and is rejected before GPU execution. No complete six-way P-TUNE result is claimed;missing windows are not synthesized.

Verified private B8 boundary qualification:GPU3 overhead4.029%,passed=True. Three fresh20/50 trace processes compare exact output bytes with the nontrace reference;one fresh50/200 reference supplies the timing comparison. This is diagnostic qualification, not performance admission.

Verified B8 diagnostic windows cover five of six existing programs, three fresh processes each; six groups choose [0, 4, 3, 4, 0, 4]. The missing candidate2 remains in the nontrace selection. An untied selection and an independent current-incumbent pair are required; this is not complete six-way P-TUNE coverage.

Verified current-profile B16 existing candidates1/2/3 compile without spills;0/4/5 are rejected before hardware execution. The whole-program fallback compares those three with the incumbent and requires an untied selection followed by an independent pair. No boundary assembly or P-TUNE completion is claimed.

Verified current-profile B8 nontrace selection: B=0.988760ms, A=1.000064ms, C=1.023392ms, D=1.045472ms; tied=False, noise=0.500%. Selection medians are diagnostic; the accepted catalog changes only after an independent pair, final measurement and ledger audit.

Verified current-profile B16 nontrace selection: A=1.292264ms, B=1.431624ms, D=1.435920ms, C=1.450376ms; tied=False, noise=2.825%. Selection medians are diagnostic; the accepted catalog changes only after an independent pair, final measurement and ledger audit.

Verified private Repeat codegen reserves operand roles used only by first or last repetitions. Existing optional-map candidates B8/B16 now use319488/359424B of SASS without spills. All five current generated sources remain identical; four real codegen cases reject optional-map and alias corruption. These are CPU results, not performance admission.

Verified the first optional-map fixture fails initcheck on generic reads after TMA output, consistent with the documented 2025.2.1 TMA initialization-shadow limitation. The rejected cohort is preserved. The replacement poisons every output before every launch and checks all intermediate values against positive CPU references; its CUDA instruction words are unchanged. This workaround retains the tool coverage limitation.

Verified optional Repeat B8 ordering fixture passes unit5/5 and sanitizer1/1/5/5 on GPU0. Whole-model byte equivalence and both code-size and current-incumbent performance gates remain separate.

Verified private optional Repeat B8 same-coordinate P-PAIR: 1.080000 to 1.090816ms, gain -1.001%, noise 2.124%, DT-CODE minimum3%. Passes this prerequisite=False. The private optional-role validator supplies identity checks; the unchanged main contract does not support these artifacts. This remains outside the production catalog.

Verified optional Repeat B16 ordering fixture passes unit5/5 and sanitizer1/1/5/5 on GPU3. Whole-model byte equivalence and both code-size and current-incumbent performance gates remain separate.

Verified private optional Repeat B16 same-coordinate P-PAIR: 1.436288 to 1.427808ms, gain 0.590%, noise 1.188%, DT-CODE minimum3%. Passes this prerequisite=False. The private optional-role validator supplies identity checks; the unchanged main contract does not support these artifacts. This remains outside the production catalog.

Verified matched GPU3 epilogue recheck: deferred_ln=859.3→764.0ns, deferred_ln_gelu=2618.0→2099.7ns, residual_ln_rowstats=1989.7→1478.9ns. Original56/224 and current72/216 P1/matrix implementations each use five fresh processes,132 CTAs and three tasks. These are coefficient measurements, not new end-to-end gains; constants remain uninstalled.

Verified current-coordinate model audit, actual/predicted: B1=1.531, B2=1.751, B4=1.918, B8=1.886, B16=1.616. The CPU utility reproduces all63 original rank0 stage predictions exactly and reuses current accepted P-DNN observations; it does not claim new P-MODEL traces. Every cell exceeds1.3. Lower epilogue coefficients alone would worsen underprediction, so no epilogue-only table is installed. Inferred next implementation check: eliminate runtime tail conditions for dimensions known to contain only full tiles.

Verified reuse of pre-full-tile B8 qualified boundary windows (prediction→observation, group sums): small 9.7→15.8µs, qkv 97.9→198.8µs, attention 59.5→133.7µs, projection 64.6→146.4µs, ffn1 134.2→296.7µs, ffn2 152.6→235.1µs. The existing three20/50 processes passed4.0288% overhead qualification on GPU3. FFN1 has the largest underprediction. This is an incumbent-coordinate diagnostic, not a new rank0 P-MODEL gate; boundary terms alone do not assign the gap to mainloop or epilogue. No coefficient is changed.

Verified private full-tile B8 CPU preparation: SASS 311680→292864B, zero spills; disabled kernels preserve current instruction words. Complete unit5/sanitizer1/1/5/5 suites: body, order. The initial B8 ordering cohort exceeded its aggregate1200s limit and is rejected; its replacement retains all200-launch cases with separately guarded processes. These are implementation and correctness results; no performance admission is implied.

Verified full-tile B8 strict model5/5 on GPU0, with exact current-incumbent output bytes.

Verified full-tile B8 GPU3 pair ratio 0.971036, noise 0.500%; passes the independent gain gate=True. Main integration, final measurement and an exact-binary ledger remain separate admission requirements.

Verified full-tile B8 admission: GPU3 P-DNN 0.950512ms, p90 0.956864ms, process range 0.002448ms. Main emitted kernels and focused fixtures match the tested instruction words; disabled defaults are preserved. The exact measured binary, strict model gate and63-stage ledger are retained. Eight corrupted admission receipts are rejected in CPU checks. Other cells, fixed goals and original anchors are unchanged.

Verified private full-tile B16 CPU preparation: SASS 343168→322560B, zero spills; disabled kernels preserve current instruction words. B16 also rematerializes the first/last Repeat selector to remove a measured4B consumer predicate spill. Complete unit5/sanitizer1/1/5/5 suites: body, order, repeat_even, repeat_odd. The initial B8 ordering cohort exceeded its aggregate1200s limit and is rejected; its replacement retains all200-launch cases with separately guarded processes. These are implementation and correctness results; no performance admission is implied.

Verified full-tile B16 strict model5/5 on GPU0, with exact current-incumbent output bytes.

Verified full-tile B16 GPU3 pair ratio 0.990010, noise 0.878%; passes the independent gain gate=True. Main integration, final measurement and an exact-binary ledger remain separate admission requirements.

Verified full-tile B16 admission: GPU3 P-DNN 1.282384ms, p90 1.298976ms, process range 0.014208ms. Main emitted kernels and focused fixtures match the tested instruction words; disabled defaults are preserved. The exact measured binary, strict model gate and63-stage ledger are retained. Eight corrupted admission receipts are rejected in CPU checks. Other cells, fixed goals and original anchors are unchanged.

Verified fixed full-tile B2/B4 candidates each pass strict whole-model5/5 on GPU0 and preserve incumbent output bytes. Real main codegen matches31360/31424 encoded64-bit instruction words of the prepared binaries, with zero spill and unchanged coordinates/profiles. B2 has a separate72/216 body cohort (unit5,mem1,init1,race5,sync5); its model uses only register stores. B4 requires the matching72/208 B8 body and ordering cohorts. Independent GPU3 performance and exact-binary ledger checks remain separate admission requirements.

Verified full-tile B4 admission: GPU3 P-DNN 0.634688ms, p90 0.638592ms, process range 0.003456ms. Its independent pair gains 6.787% against 1.502% noise. The main emitted kernel matches the measured binary; complete B8 body and ordering cohorts are reused only after matching their full72/208 feature profiles. Nine corrupted receipts, including mismatched inherited profiles, are rejected. The exact63-stage GPU3 ledger, fixed target and original anchor are retained.

Verified GPU3 isolated FFN diagnostics: B8 FFN1 12.096us, B8 FFN2 13.216us, B16 FFN1 21.376us, B16 FFN2 18.464us. Each fixed production shape/profile uses132 CTAs, five fresh 20/50 processes, real TMA and unchanged epilogue arithmetic; integer model shape/stride constants are preserved and every output is checked against an independent constant-input FP32 reference. The qualified pre-full-tile B8 FFN1 boundary mean is24.728us versus 12.096us isolated. The workloads differ in cache state, code context, bindings and stage gates; this gap is not assigned to one cause. These historical probes used synthetic epsilon1e-5; emitted BERT uses1e-12. No F4 constant or end-to-end result is replaced.

Verified GPU0 128MiB data-write contrast for the same isolated B8 FFN1 kernel: hot 12.256us and cold 13.120us, ratio 1.070496. Each context uses five fresh20/50 processes and checks every output against the constant-input FP32 reference. All6304 encoded64-bit Probe words match the original diagnostic; eviction runs outside its measured consumer window. This same-card contrast does not establish the complete model gap or replace F4 coefficients or end-to-end admission. Both sides use the historical synthetic epsilon1e-5.

Verified GPU3 1MiB straight-line-code contrast for the same isolated B8 FFN1 kernel: hot 12.128us and cold 12.416us, ratio 1.023747. Each context uses five fresh20/50 processes and checks every output against the constant-input FP32 reference. All6304 encoded64-bit Probe words match the original diagnostic; eviction runs outside its measured consumer window. This same-card contrast does not establish the complete model gap or replace F4 coefficients or end-to-end admission. Both sides use the historical synthetic epsilon1e-5.

Verified probe audit corrects its default epsilon from synthetic1e-5 to the emitted BERT1e-12. The old setting reproduces all6304 original instruction words; the corrected build changes only seven FP32 FADD immediates, retaining the opcode/register/scheduling words. Prior within-binary contrasts remain synthetic diagnostics, not exact production constants. Accepted model builds and installed coefficients were never changed by these probes.

Verified GPU0 fixed FFN1 input-pattern contrast at epsilon1e-5: uniform 12.160us and varying 12.288us. The same kernel uses32 deterministic BF16 row/column patterns; every output passes the independent FP32/GELU reference with cosine≥0.99999 and maximum relative error≤0.01. Each pattern has identical output bytes across five fresh 20/50 processes. This synthetic-data diagnostic does not replace a whole-model measurement or coefficient.

Verified GPU0 fixed FFN1 input-pattern contrast at epsilon1e-12: uniform 12.160us and varying 12.384us. The same kernel uses32 deterministic BF16 row/column patterns; every output passes the independent FP32/GELU reference with cosine≥0.99999 and maximum relative error≤0.01. Each pattern has identical output bytes across five fresh 20/50 processes. This synthetic-data diagnostic does not replace a whole-model measurement or coefficient.

Verified full-tile ordering dispatch correction: the first individual-process wrapper omitted the device environment on its scheduler subprocess, which replaced per-job lock paths with one shared default lock. All of those individual ordering results are excluded. The replacement passes the per-device environment and checks the actual lock before each CUDA child; all17 processes are recollected. Grouped body, Repeat and model gates already held the correct device locks. The earlier foreign GPU-lock and NUMA-setting diagnoses were not established.

Verified full-tile B16 individual racecheck durations vary: two complete guarded processes took603s and825s with zero errors; a third reached the900s child timeout and has no valid result. Its subsequent ownership rejection does not establish foreign interference. The timed-out attempt is excluded; nine completed processes were retained and only the eight missing checks were scheduled with line-buffered output and a3600s child limit. The frozen binary, cases,200 iterations, guard and correctness thresholds are unchanged. The replacement ordering gate now completes all17 processes; performance admission remains a separate requirement.

Verified profile2 B8 boundary overhead is3.419% on GPU0. Only2/6 existing variants are nonspilling;each completes three fresh20/50 traces. Four spilled variants are never executed. Required candidate coverage is incomplete;no assembled selection or admission is claimed. GPU2 partial traces remain separate and are not mixed with the GPU0 windows.

Verified optional profile2 uses72/208 registers and explicit scalar phase selection only for that profile. Profiles0/1 retain their prior paths and the default remains0. Scalar/alias and P1 component cohorts each pass unit5/5 and sanitizer1/1/5/5;B2/B4/B8/B16 whole models each pass5/5 without spill. B8/B16 SASS is328576/395776B with zero local-memory instructions. The integrated B8 kernel has identical instruction bytes to its prototype. Seven related regression checks, real profile codegen and invalid-setting checks pass. GPU2 comparisons remain diagnostics. Complete GPU3 pairs subsequently admit B4/B8;exact-binary ledgers and three-process P-DNN results are in the current catalog. B2 fails its noise gate and B16 fails the firstRepeat3% gate. B8/B16 first-time Repeat admission still requires3% plus noise;B2/B4 already use accepted Repeat and require a gain beyond noise. Fixed BF16 goals and original anchors remain unchanged.

Verified B8 arithmetic Repeat plus P1 normalization fails the compilation gate under64/216 and prescribed56/224 (repeat_arithmetic_middle_v1/b8:8,24B stores/loads;repeat_arithmetic_b8_reg56_v1:32,104B stores/loads). Neither spilled binary is executed;profile2 later resolves this spill gate.

Verified B4 arithmetic Repeat plus P1 normalization retains the accepted coordinates. The64/216 profile gains2.89%,above2.63% noise but below DT-CODE3%,so it is rejected. Prescribed56/224 compiles without spill and passes strict model5/5. Its GPU3 P-PAIR 0.760288→0.733952ms gives 3.46% gain against 0.50% noise;the final P-DNN is recorded separately. Arithmetic and P1 component units pass5/5 and sanitizer1/1/5/5 separately.

Verified B4 admitted P-DNN median0.734352ms,p900.744128ms,process range0.007152ms. SASS and identities are retained with the default56/224 profile;no production header changes are needed. It remains above the fixed BF16 target and retains the original GPU3 anchor without re-anchoring.

Verified B2 P1 normalization plus the admitted arithmetic loop spills 8/16B stores/loads under64/216. The compilation gate rejects it before execution;the admitted arithmetic-only B2 profile remains selected.

Verified B2 current-profile boundary overhead 8.351% exceeds5%;those windows do not select stage coordinates. Four distinct nontrace whole programs use only original Phase Z candidates and the admitted control. Every candidate passes strict model5/5;spilled builds are never executed. G22 P-SEL winner=A,tie threshold=9.099%. The admitted control remains selected unless a separate fresh P-PAIR passes its unchanged noise gate.

Verified B4 current-profile boundary overhead 7.059% exceeds5%;those windows do not select stage coordinates. Four distinct nontrace whole programs use only original Phase Z candidates and the admitted control. Every candidate passes strict model5/5;spilled builds are never executed. G22 P-SEL winner=B,tie threshold=0.559%. The admitted control remains selected unless a separate fresh P-PAIR passes its unchanged noise gate.

Verified original thread-major and opt-in vector-major split-K layouts produce identical output bytes on six frozen shapes:odd rows,S3/4/6/16 and TK64/128. The first five accepted fresh original-layout processes use one frozen binary (four on GPU2,one on GPU3),each with guard0;all match the accepted coalesced-suite output hash. This is a functional allocation deviation with no mixed-card performance comparison.

Verified resumption: all new GPU tests use physical GPU3 only. Previously completed component/model receipts are reused;historical GPU0–2 queues remain inactive. The four pending candidates complete fresh same-card pairs;none clears its acceptance gate. Accepted profiles,original anchors and frozen Phase Z results are retained.

Verified rejected B8 FFN2 split2/TMA attribution on GPU3: FFN2 windows sum311.968→479.712us;main-loop CTA maxima sum267.104→233.408us;split-K maxima sum130.944us and epilogue maxima sum32.928→100.672us. Segment maxima are nonadditive diagnostic attribution. The fixed coalesced-partial follow-up has separate component/model/pair gates;a main-loop improvement alone does not admit either combination.

Verified private asynchronous PP side acquisition: normal unit5/5, memcheck1/1,initcheck1/1 and racecheck5/5 pass on GPU3, but synccheck0/1 reports8 Missing init errors. Reject before model or timing;the partial cohort is not a complete17 gate. Default0 full B16 instruction words match the accepted binary. The isolated follow-up restores direct acquisition by all128 WG consumers;its gates remain separate.

Verified direct acquisition by all128 consumers also fails: unit5/5,mem1/1,init1/1,race5/5 pass on GPU3; synccheck0/1 reports1024 Missing init errors. Reject before model/timing. The next private refinement retains the previously complete named160 handoff and makes only the producer arrival nonblocking;its full protocol and end-to-end gates are independent.

Verified nonblocking named160 PP side handoff passes two fresh GPU3 component cohorts17/17 each and strict model5/5. Default0 full B16 instruction words remain identical to the incumbent. GPU3 full-model pair ratio1.075620;noise1.056%,DT-PP minimum3%. The synchronization repair is functional evidence;performance and production admission follow the independent end-to-end gate.

Verified opt-in cyclic slot indexing and vector-major normal split-K partials retain the admitted B1 coordinates,56/224 register roles and ascending slice addition order. Integrated component gates pass17/17 on GPU3;the extended S2/TK128/odd-row suite passes17/17 on GPU2. The full model passes strict FP32 and cross-process byte checks5/5. G22 P-PAIR0.499216→0.478000ms passes0.904% noise;P-DNN 0.464496ms,p900.477696ms,process range0.012608ms. The exact accepted binary has a P-TRACE1 ledger. The earlier B2 comparison fails its paired noise gate;the later full-pipeline combination has a separate admission below. Defaults,original anchors,CNN results and frozen20-cell Phase Z remain unchanged.

Verified current accepted artifacts (GPU3;50/200;three-process P-DNN). B1/B2/B4/B8/B16 are follow-ups that retain the original anchor without re-anchoring. Fixed BF16 targets reached for B1,B2. DNN4_BERT_current.csv records complete identities,profiles and evidence paths;DNN4_BERT_current_ledger.csv preserves all315 current stage rows with exact accepted reference hashes and coordinates.

| Batch | GPU | Median ms | p90 ms | Process range ms | Fixed BF16 ms | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0.464496 | 0.477696 | 0.012608 | 0.4858 | 0.9561 |
| 2 | 3 | 0.508096 | 0.511104 | 0.012352 | 0.5171 | 0.9826 |
| 4 | 3 | 0.634688 | 0.638592 | 0.003456 | 0.5905 | 1.0748 |
| 8 | 3 | 0.923120 | 0.929184 | 0.012176 | 0.7457 | 1.2379 |
| 16 | 3 | 1.282384 | 1.298976 | 0.014208 | 1.0869 | 1.1799 |

Verified largest three stages for each current unmet BERT cell (P-TRACE1 attribution only;segments can overlap and are not timing results). Each ledger checks the exact accepted reference binary SHA256.

| Batch | Largest stage | Second | Third |
| --- | --- | --- | --- |
| 4 | s45 gemm 18.24us (main_loop=15.07,slot_wait=4.19) | s40 gemm 17.89us (main_loop=14.53,slot_wait=3.97) | s30 gemm 17.86us (main_loop=14.72,slot_wait=4.45) |
| 8 | s34 gemm 29.38us (main_loop=15.58,epilogue=11.65) | s24 gemm 29.28us (main_loop=15.74,epilogue=11.17) | s49 gemm 29.12us (main_loop=15.84,epilogue=10.62) |
| 16 | s4 gemm 42.82us (main_loop=23.74,epilogue=15.81) | s44 gemm 40.35us (main_loop=22.98,epilogue=13.63) | s39 gemm 40.06us (main_loop=23.49,epilogue=13.09) |

Verified current B8 combines existing matrix and attention vector stores with the admitted full-tile profile. Three exact-profile component cohorts each pass17/17;strict model5/5 preserves incumbent bytes. Main real codegen matches all35232 encoded instruction words and the63-stage ledger identifies the measured binary. GPU3 P-DNN0.923120ms,p900.929184ms. Ten corrupted admission receipts are rejected. Other cells,default flags and fixed anchors remain unchanged.

Verified current B2 combines full-tile/grid/recompute with existing coalesced partials and cyclic slots at unchanged coordinates and72/216 registers. Eleven exact/odd-tail fixtures pass control5 and candidate17 with identical bytes;strict model5 preserves incumbent bytes. Main real codegen matches29984 encoded words;the exact measured binary has a63-stage GPU3 ledger. P-DNN0.508096ms,p900.511104ms. Pair gain10.64656% narrowly exceeds noise10.60716%;this variability remains in the receipt. Twelve corrupted admission receipts are rejected;fixed targets are unchanged.

Verified private P1 direct-statistics reads preserve ascending sums, normalization arithmetic,shared allocation and synchronization. Control5 and candidate17 cover ten exact/odd-tail fixtures;model5 preserves incumbent bytes. GPU3 pair ratio1.002292,noise3.496%. The candidate has no end-to-end gain and is rejected;private headers restored. The initial illegal synthetic shape failed compilation in both arms and was replaced before GPU execution with the actual legal QKV shapes.

Verified formerly spilled B16 catalog candidate5 compiles without spill after the existing task-index repair;14 exact/odd-tail fixtures pass control5 and candidate17, and strict model5 preserves incumbent bytes. Its independent GPU3 pair regresses15.282% and is rejected. Exact-binary P-TRACE1 group-window increases are projection+65.376us, ffn2+60.224us, qkv+42.528us. Projection/FFN2 epilogue CTA-maxima sums rise73.280→146.304us and76.608→146.304us. These diagnostic maxima are nonadditive; no P-DNN or current-artifact replacement follows the rejection.

Verified B16 statistics-read-ahead model checks produce identical bytes in all five processes and all six execution cases, with unchanged FP32 gates. One reported tanh maximum error varies from 0.016677532345 to 0.016679663211. The original wrapper incorrectly required the entire metric dictionary to match. A CPU-only finalizer validates each process against the original threshold and requires identical output hashes; raw checks and the initial assertion are retained. Thirteen corrupted cohorts are rejected. No hardware result or accuracy threshold was changed.

Verified B4 fixed two-iteration main-loop unrolling passes control5, candidate17 on12 exact/odd-K/odd-M fixtures and strict model5 with unchanged bytes. The GPU3 pair regresses5.880% and is rejected. P-TRACE1 GEMM main-loop CTA-maxima sums increase 436.608→460.608us; the intended segment did not improve. These maxima are nonadditive diagnostic attribution. The private header is restored.

Verified private terminal FFN2 sharing compiles without spill; SASS decreases 322560→298624 bytes. Default0 matches all40320 current B16 instruction words. The first attempt compiled but failed the old artifact contract; its raw failure is retained. The explicit optional-statistics contract checks identical geometry, arithmetic and the absent terminal output;13 corrupted sources are rejected. Production kernels and defaults remain unchanged.
Verified ten cases form five normal/terminal pairs and cover exact/odd rows, TMA/reg stores, TM64 merging and split-K. Control5 and candidate17 have identical bytes; strict model5 matches the incumbent. The separate full-model pair requires at least3% gain and the measured noise threshold before admission.
Verified GPU3 pair 1.279280→1.299168ms, noise2.749%. The code-size reduction provides no end-to-end gain; reject. Private headers and verifier are restored; raw evidence is retained.

Verified fixed WG64 PP/P1/TMA candidate compiles without spill and both normal unit cohorts pass5/5, but ordering synccheck reports1024 errors and residual-chain racecheck reports21 errors. Reject before model correctness or timing; partial sanitizer cohorts are not complete gates. Frozen failures remain raw; current profiles and targets are unchanged.
