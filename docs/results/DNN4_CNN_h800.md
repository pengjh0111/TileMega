# DNN4 CNN collection

Verified: 9/15 CNN cells collected; 3/9 beat the fixed fastest BF16 target.

Existing TileMega results are reused. All 60 CNN baseline arms are reused from DNN3_baselines_phase_z.csv with their original GPU, input, checkpoint and measurement receipts; this export copies those prior measurements without running benchmarks. New Phase Z anchor checks and any drift-triggered measurements are tracked separately in DNN4_h800.md. ResNet18 uses GPU3, ResNet50 GPU0 and VGG16 GPU2. All inputs/weights are BF16 and accumulations FP32.

Measured TileMega winners passed five fresh FP32-reference processes with identical output byte hashes. Timing uses a nontrace build, 50 warmups and 200 CUDA events in each of three fresh processes; the reported latency is the median of process medians. Full CSV includes p90, process range, coordinates, artifact identities, SM clocks and raw receipts. GPU ownership remains provisional because the container PID namespace cannot resolve every host process.

| Model | B | TileMega ms | DNN3 TileMega ms | Speedup over DNN3 | Fixed BF16 ms | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 1 | 0.154848 | 0.217984 | 1.408x | 0.190624 | 0.8123 |
| resnet18 | 2 | pending | 0.259456 | — | 0.219552 | — |
| resnet18 | 4 | pending | 0.316352 | — | 0.253152 | — |
| resnet18 | 8 | pending | 0.420016 | — | 0.307808 | — |
| resnet18 | 16 | pending | 0.601792 | — | 0.432960 | — |
| resnet50 | 1 | 0.366480 | 0.478496 | 1.306x | 0.360224 | 1.0174 |
| resnet50 | 2 | 0.401408 | 0.549552 | 1.369x | 0.390272 | 1.0285 |
| resnet50 | 4 | pending | 0.649616 | — | 0.456976 | — |
| resnet50 | 8 | pending | 0.843504 | — | 0.576224 | — |
| resnet50 | 16 | 0.913392 | 1.201424 | 1.315x | 0.824688 | 1.1076 |
| vgg16 | 1 | 0.313536 | 0.521792 | 1.664x | 0.317232 | 0.9883 |
| vgg16 | 2 | 0.420640 | 0.687504 | 1.634x | 0.384576 | 1.0938 |
| vgg16 | 4 | 0.544912 | 0.954400 | 1.751x | 0.523296 | 1.0413 |
| vgg16 | 8 | 0.818544 | 1.398464 | 1.708x | 0.807184 | 1.0141 |
| vgg16 | 16 | 1.260448 | 2.382448 | 1.890x | 1.398176 | 0.9015 |

| Group | Cells | Geometric ratio to baseline | Geometric speedup over DNN3 | Baseline wins |
| --- | --- | --- | --- | --- |
| model:resnet18 | 1/5 | 0.8123 | 1.408x | 1 |
| model:resnet50 | 3/5 | 1.0504 | 1.330x | 0 |
| model:vgg16 | 5/5 | 1.0058 | 1.727x | 2 |
| batch:1 | 3/3 | 0.9348 | 1.452x | 2 |
| batch:2 | 2/3 | 1.0607 | 1.496x | 0 |
| batch:4 | 1/3 | 1.0413 | 1.751x | 0 |
| batch:8 | 1/3 | 1.0141 | 1.708x | 0 |
| batch:16 | 2/3 | 0.9992 | 1.577x | 1 |
| overall:CNN | 9/15 | 0.9965 | 1.548x | 3 |

Ratios below 1 beat the fixed fastest BF16 target. Improvement over DNN3 TileMega does not imply a win against that target. Partial aggregates describe measured members only. BERT optimization continues independently; this CNN table does not claim BERT completion.

CSV files: DNN4_CNN_final.csv, DNN4_CNN_ratios.csv, DNN4_CNN_baselines_reused.csv.
