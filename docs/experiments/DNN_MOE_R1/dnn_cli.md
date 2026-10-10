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

verified: the accepted feature surface now includes pg, forward_executor,
reuse, deferred_ln, dwpw_fuse, global_la and explicit nonpaged_la/paged_la/
paged_la_splitk overrides. Unknown features remain errors. Three CLI host
checks pass, including inherited GPU-lock exclusivity and reduction controls.

verified: the generated-model checker accepts `--synthetic-weights`, replacing
the exported module state with one fixed seed and packing that same state for
the native library. Normalization variances stay positive and integer buffers
stay integral. The reference is that synthetic BF16 state promoted to FP32,
recorded with its seed in the receipt; it is not a real-weight model gate. One
host exported Conv/BN check verifies deterministic, finite execution and dtypes.
Native full-graph synthetic replay remains pending.
