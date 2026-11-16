# Admitted DNN4 BERT profiles

Verified: `bert_profiles.json` contains the five admitted batch-specific coordinate
and codegen-environment recipes from `docs/results/DNN4_BERT_current.csv`.
Use the coordinate file and **all** of its `environment` entries together with
`--abi f3`; a coordinate file alone does not select the measured profile.
Coordinates require a matching BERT-base/sequence128 DNN export and H800 target.

Verified integration checks: all five recipes produce the same complete generated
CUDA bodies as the native rebuild, with identical encoded GPU instructions and
zero spills. B2 differs from its historical source only in the order of
independent numeric macro definitions. No GPU timing was repeated for this merge.
Historical latency/accuracy claims retain the source report's checkpoint and
container-PID limitations; these recipes do not certify new model weights.

Private DSM, statistics-TMA and reader-policy prototypes, and the separate
performance-only R6 host-launch branch are excluded. Existing disabled diagnostic
options retain their original defaults. This package does not assert DNN4 or R6
full-stage acceptance.
