# DNN correctness CLI

implemented: `python -m tilemega dnn {export,build,run,check,report} --config
configs/dnn/<model>.json`. The five configs retain official data and checkpoint
paths. Builds validate pretrained upstream export manifests, bind each batch,
use the prediction-only solver path and write source/build/ptxas identities.
No latency benchmark is provided under the user's scope update.

verified: configuration validation and inherited GPU-lock exclusivity pass
`test/python/test_dnn_cli.py`. Unknown features are rejected. Numerical checks
retain the original G-DNN criteria and join receipts by artifact identity.
The complete public build/run entry still needs a native replay; existing
native model results were generated through the experiment/compile entry.

Current feature surface: pg, forward_executor, reuse and deferred_ln. Structural
fusion and other unfinished features are deliberately rejected until their
complete runtime integration is available. Their implementation remains due.
