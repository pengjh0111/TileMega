# Dependencies

Verified versions were read from the sm_89 development environment. Pinned
versions are configuration requirements (stated), not a claim that every R15
gate has passed. Build instructions are in [README](../README.md) and
[BUILD_MLIR](BUILD_MLIR.md). Keep serving, DNN and vLLM in separate environments.

| Dependency | Version or pin | Required / optional | Purpose |
|---|---|---|---|
| C++17 compiler | GCC 11.4.0 (verified) | Required | Host compiler |
| CMake | 3.22.1 (verified) | Required | Build configuration |
| Ninja | 1.10.1 (verified) | Required for documented build | Build executor |
| CUDA Toolkit | 12.8, NVCC 12.8.93 (verified) | Required | Device compilation and binary inspection |
| NVIDIA driver | 610.43.02 (verified) | GPU execution | CUDA runtime support |
| LLVM / MLIR | `23a60f15f2fcafcf67b95b0a035053579958b732` (pinned) | Required | CG dialect and lowering |
| CUTLASS / CuTe | `dc45f979ae336a235da1676b311f35efeb30149a` (submodule pin) | Required | GPU task implementations |
| isl | 0.28 (submodule pin) | Required | Symbolic dependence relations |
| polylib | 5.22.9 (submodule pin) | Required | Polyhedral arithmetic |
| barvinok | 0.41.9 (submodule pin) | Required | Parametric cardinality |
| GMP | 6.2.1 (verified) | Required | Integer backend |
| NTL | 11.5.1 (verified) | Required | Polynomial arithmetic |
| Autotools | autoconf 2.71, automake 1.16.5, libtool 2.4.6 (verified) | Required to bootstrap | Polyhedral library builds |
| pkg-config / m4 | 0.29.2 / 1.4.18 (verified) | Required to bootstrap | Dependency discovery and macros |
| Python | 3.10.12 (verified); TOML needs 3.11+ | Required | Export, tests and CLI |
| PyTorch | Serving 2.13.0+cu126 (verified); DNN 2.14.1 (pinned) | Python model paths | Export and numerical references |
| transformers | Serving 5.5.3 (verified); DNN 5.19.0 (pinned) | LLM / BERT | Checkpoint reference implementations |
| safetensors | 0.8.0 (verified) | Model loading | Checkpoint tensors |
| huggingface_hub | 1.33.0 (verified) | Model download | Pinned repository snapshots |
| vLLM | 0.30.0 (verified) | Optional baseline | End-to-end reference timing |
| torchvision | 0.29.1 (DNN pin) | DNN | Vision models and input transforms |
| timm | 1.0.30 (DNN pin) | DNN | MobileNet reference |
| LMDB / PyArrow | 3.0.0 / 25.0.1 (DNN pins) | DNN datasets | SIDD and text data |
| OpenCV headless | 5.0.0.93 (DNN pin) | DNN datasets | Image decoding |

The exact DNN environment snapshots and direct requirements are under
[`requirements/`](../requirements/). The CUDA wheel runtime is separate from
the Toolkit used by NVCC. Use the recorded environment for numerical or
performance comparisons; changing a reference implementation requires a new
correctness receipt.
