# BERT BF16 baseline recheck

Verified BERT-base, seq128, H800. Each cell uses three fresh50/200 CUDA-event processes and two additional FP32 correctness processes;5/5 pass. Compilation is excluded. torch.compile uses max-autotune/fullgraph with requested and actual freezing=False. P90 is the median of three process P90s.

| Batch | TensorRT ms | torch.compile, no freezing ms | TRT GPU | TC GPU |
| --- | --- | --- | --- | --- |
| 1 | 0.460800 | 1.025360 | 1 | 0 |
| 2 | 0.518816 | 0.618720 | 1 | 0 |
| 4 | 0.605200 | 0.695968 | 1 | 0 |
| 8 | 0.788192 | 0.858688 | 1 | 0 |
| 16 | 1.145872 | 1.234768 | 0 | 1 |

Verified source receipts, all200 individual events, five FP32 correctness checks, input/checkpoint identities and actual freezing=False are reconstructed by scripts/dnn4_bert_baselines_recheck_report.py. The CSV retains P90, process range, versions, boundary clocks and raw sources. Baseline hash stability is diagnostic. Random-weight exports have no dataset accuracy certification; GPU PID attribution uses the existing container-namespace qualification.

Historical TileMega data, the earlier TRT recheck and fixed DNN4 targets are preserved. These fresh baseline measurements do not replace the frozen TC+freezing targets.

Verified drift: unfrozen torch.compile B1 is1.025359988ms versus historical0.577535987ms (+77.540450%). Its three process medians are1.021008015/1.025359988/1.029551983ms; all50/200 protocols, boundary1980MHz clocks, BF16 inputs/checkpoint, max-autotune/fullgraph and requested/actual freezing=False pass. Torch2.8.0/Triton3.4.0 and libtorch_cuda.so match the historical receipt. Generated wrapper identities differ in the new dedicated cache; the cause of the timing difference is unresolved. This observation is retained without selecting a replacement run or changing fixed targets.
