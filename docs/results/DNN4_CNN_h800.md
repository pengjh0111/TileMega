# DNN4 CNN collection

Verified: 15/15 CNN cells collected; 7/15 beat the fixed fastest BF16 target.

Verified frozen production source digest: `13b4e4deb59e335a167b6ac54b45253955e85c7f721f64ec890cc08d8995f6d2`.

Existing TileMega results are reused. All 60 CNN baseline arms are reused from DNN3_baselines_phase_z.csv with their original GPU, input, checkpoint and measurement receipts; this export copies those prior measurements without running benchmarks. New Phase Z anchor checks and any drift-triggered measurements are tracked separately in DNN4_h800.md. ResNet18 uses GPU3, ResNet50 GPU0 and VGG16 GPU2. All inputs/weights are BF16 and accumulations FP32.

Measured TileMega winners passed five fresh FP32-reference processes with identical output byte hashes. Timing uses a nontrace build, 50 warmups and 200 CUDA events in each of three fresh processes; the reported latency is the median of process medians. Full CSV includes p90, process range, coordinates, artifact identities, SM clocks and raw receipts. GPU ownership remains provisional because the container PID namespace cannot resolve every host process.

| Model | B | TileMega ms | DNN3 TileMega ms | Speedup over DNN3 | Fixed BF16 ms | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 1 | 0.154848 | 0.217984 | 1.408x | 0.190624 | 0.8123 |
| resnet18 | 2 | 0.182384 | 0.259456 | 1.423x | 0.219552 | 0.8307 |
| resnet18 | 4 | 0.215648 | 0.316352 | 1.467x | 0.253152 | 0.8519 |
| resnet18 | 8 | 0.287680 | 0.420016 | 1.460x | 0.307808 | 0.9346 |
| resnet18 | 16 | 0.413120 | 0.601792 | 1.457x | 0.432960 | 0.9542 |
| resnet50 | 1 | 0.366480 | 0.478496 | 1.306x | 0.360224 | 1.0174 |
| resnet50 | 2 | 0.401408 | 0.549552 | 1.369x | 0.390272 | 1.0285 |
| resnet50 | 4 | 0.474720 | 0.649616 | 1.368x | 0.456976 | 1.0388 |
| resnet50 | 8 | 0.619072 | 0.843504 | 1.363x | 0.576224 | 1.0744 |
| resnet50 | 16 | 0.913392 | 1.201424 | 1.315x | 0.824688 | 1.1076 |
| vgg16 | 1 | 0.313536 | 0.521792 | 1.664x | 0.317232 | 0.9883 |
| vgg16 | 2 | 0.420640 | 0.687504 | 1.634x | 0.384576 | 1.0938 |
| vgg16 | 4 | 0.544912 | 0.954400 | 1.751x | 0.523296 | 1.0413 |
| vgg16 | 8 | 0.818544 | 1.398464 | 1.708x | 0.807184 | 1.0141 |
| vgg16 | 16 | 1.260448 | 2.382448 | 1.890x | 1.398176 | 0.9015 |

| Group | Cells | Geometric ratio to baseline | Geometric speedup over DNN3 | Baseline wins |
| --- | --- | --- | --- | --- |
| model:resnet18 | 5/5 | 0.8749 | 1.443x | 5 |
| model:resnet50 | 5/5 | 1.0528 | 1.344x | 0 |
| model:vgg16 | 5/5 | 1.0058 | 1.727x | 2 |
| batch:1 | 3/3 | 0.9348 | 1.452x | 2 |
| batch:2 | 3/3 | 0.9777 | 1.471x | 1 |
| batch:4 | 3/3 | 0.9731 | 1.521x | 1 |
| batch:8 | 3/3 | 1.0060 | 1.504x | 1 |
| batch:16 | 3/3 | 0.9840 | 1.536x | 2 |
| overall:CNN | 15/15 | 0.9748 | 1.496x | 7 |

Ratios below 1 beat the fixed fastest BF16 target. Improvement over DNN3 TileMega does not imply a win against that target. Partial aggregates describe measured members only. BERT optimization continues independently; this CNN table does not claim BERT completion.

CSV files: DNN4_CNN_final.csv, DNN4_CNN_ratios.csv, DNN4_CNN_baselines_reused.csv.
