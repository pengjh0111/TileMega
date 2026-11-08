# DNN4 BERT optimization progress

Verified diagnostic comparisons below reuse the current frozen production coordinates and inputs. All candidates passed five fresh FP32-reference processes with identical cross-process output hashes. P-PAIR uses A/B/A/B/A/B, three fresh processes per side, 50 warmups and 200 events per process. Acceptance requires a gain larger than max(0.5%, twice the larger process range). GPU1 diagnostics guide optimization; G22 diagnostics compare on GPU3 against rank0 and do not replace selection against the eventual final winner or three-process P-DNN.

| Experiment | GPU | Reference ms | Candidate ms | Gain % | Noise threshold % | Significant gain |
| --- | --- | --- | --- | --- | --- | --- |
| attention B1 | 1 | 0.629600 | 0.618880 | +1.703 | 5.581 | False |
| attention B2 | 1 | 0.680416 | 0.670224 | +1.498 | 5.343 | False |
| TK128 B16 | 1 | 1.522272 | 1.526256 | -0.262 | 1.221 | False |
| residual TMA B16 | 1 | 1.528224 | 1.490880 | +2.444 | 1.322 | True |
| in-situ B16 | 1 | 1.521344 | 1.468064 | +3.502 | 1.942 | True |
| in-situ B16 | 3 | 1.519872 | 1.463936 | +3.680 | 1.143 | True |
| in-situ B1 | 1 | 0.630176 | 0.614240 | +2.529 | 5.433 | False |

Verified attention experiment: QK/PV columns were distributed across eight consumer warps without changing any output accumulation order. Per-head unit5/5 and sanitizer memcheck1/1, initcheck1/1, racecheck5/5, synccheck5/5 passed; whole-model outputs also matched the control byte for byte. Neither timing gain exceeded process variability, so the original header was restored. The initial driver added an unnecessary3% cutoff, which does not affect either rejection because both measured noise thresholds exceed3%. Frozen sources and the patch remain under runs/dnn4/bert_priority.

Verified TK128 experiment: one deterministic max-legal-slot configuration on12 QKV stages; other stages and tile/split/store coordinates unchanged. No significant gain. Verified residual TMA experiment: only24 residual-reuse stages change store mode, with no additional shared allocation or changes to tiles/splits/TK/Slots.

Verified in-situ experiments use the six already authorized candidate programs, three fresh processes per variant and20 warmups/50 trace windows. Group sums select a coordinate, then the assembled nontrace program faces a whole-model P-PAIR gate. These experiments add no parameter scan.

in-situ B16: group0→candidate5, group1→candidate1, group2→candidate0, group3→candidate2, group4→candidate1, group5→candidate1.
in-situ B1: group0→candidate3, group1→candidate0, group2→candidate0, group3→candidate0, group4→candidate4.

Final BERT batch collection continues on GPU3 after completion of all15 CNN cells and the CNN anchors. Existing CNN results and all60 baseline arms are reused. GPU ownership remains qualified by the container PID namespace. See DNN4_BERT_optimization.csv for identities, clocks, correctness receipts and raw timing sources.
