# Supported device matrix

Evidence labels: ✅ compiled or queried locally; ⚠️ vendor-documented but not
run on that device; ❌ inference. Resource values are configuration inputs, not
architecture-wide constants. H100 means the 132-SM SXM configuration and A100
means the 108-SM configuration; `TargetSpec::Probe()` replaces these values on
the installed SKU.

| Property | A100 / sm_80 | RTX 4090 / sm_89 | H100 SXM / sm_90 | RTX 5090 / sm_120 |
|---|---:|---:|---:|---:|
| SM count | ⚠️ 108 [S3] | ⚠️ 128 [S4] | ⚠️ 132 [S3] | ⚠️ 170 [S5] |
| Shared memory per SM | ⚠️ 164 KiB [S1,S6] | ⚠️ 100 KiB [S1] | ⚠️ 228 KiB [S1,S7] | ⚠️ 100 KiB [S1] |
| Max dynamic shared memory per block | ⚠️ 163 KiB [S1] | ✅ 99 KiB [S1, V-A] | ⚠️ 227 KiB [S1,S7] | ⚠️ 99 KiB [S1] |
| Cluster / distributed shared memory | ⚠️ no [S1] | ✅ no [S1, V-C] | ✅ yes [S1, V-C] | ✅ yes [S1, V-C] |
| TMA | ⚠️ no [S1] | ✅ no [S1, V-B] | ✅ yes [S1,S2, V-B] | ✅ yes [S1,S2, V-B] |
| Warp-specialized collective | ⚠️ no [S2] | ✅ no [S2, V-B] | ✅ yes [S2, V-B] | ✅ yes for narrow types [S2, V-B] |
| `tcgen05` / TMEM | ⚠️ no [S8] | ⚠️ no [S8] | ⚠️ no [S8] | ✅ **no** [S8, V-I] |
| `cp.async` | ⚠️ yes [S1] | ✅ yes [S1, V-B] | ⚠️ yes [S1] | ⚠️ yes [S1] |
| `mbarrier` | ⚠️ yes [S9] | ✅ yes [S9, V-A] | ⚠️ yes [S9] | ⚠️ yes [S9] |
| CUTLASS mainloop observed in this checkout | ✅ `MainloopSm80CpAsync` [V-I] | ✅ `MainloopSm80CpAsync` [V-B,V-I] | ✅ `MainloopSm90TmaGmmaWarpSpecialized` [V-B,V-I] | ✅ `MainloopSm120TmaWarpSpecialized` with FP8 [V-B,V-I] |

The two portability consequences are direct:

- `sm_120` is numerically newer than `sm_100`, but PTX documents `tcgen05` for
  the `sm_100f`/`sm_101f` families, not `sm_120f`. Exact capability switches
  are therefore required; an `arch >= N` policy is incorrect.
- Shared-memory budgets differ by roughly two times between the checked-in
  data-center and consumer configurations. TaskBody `Stages` is computed as
  `min(max_stages, floor((budget - overhead) / bytes_per_stage))` from
  `TargetSpec`, never from an architecture or device-name literal.

## Calibrated constants

The table above is capability and resource data, which `TargetSpec::Probe()`
reads from the driver. The cost model additionally needs *measured* rates and
latencies, which no query returns. Those live in the same
`configs/targets/sm_XX.json` files under `"calibration"` and come only from
running `docs/experiments/CALIB/run.sh` on the device itself.

| Target | Calibration | Evidence |
|---|---|---|
| A100 / sm_80 | ❌ not run | no device here; every field 0, `"calibrated": false` |
| RTX 4090 / sm_89 | ✅ measured | [CALIB/result.md](experiments/CALIB/result.md), 55 records, 6.95 s |
| H100 SXM / sm_90 | ❌ not run | no device here; every field 0, `"calibrated": false` |
| RTX 5090 / sm_120 | ❌ not run | no device here; every field 0, `"calibrated": false` |
| `cluster_sync_ns`, all targets | ❌ not run | needs a `cudaLaunchKernelEx` cluster launch (Part 7) |

An uncalibrated profile carries zeros, never a value interpolated from sm_89 or
taken from a datasheet, and `run.sh` refuses to write a profile for an
architecture the machine does not have.

## Sources

- [S1] [CUDA C++ Programming Guide, Compute Capabilities](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [S2] [CUTLASS GEMM API 3.x](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/gemm_api_3x.html)
- [S3] [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [S4] [NVIDIA Ada GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf?ncid=no-ncid)
- [S5] [NVIDIA RTX Blackwell GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [S6] [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/archive/13.0.1/ampere-tuning-guide/index.html)
- [S7] [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/11.8.0/hopper-tuning-guide/index.html)
- [S8] [PTX ISA 8.8, architecture feature tables](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html)
- [S9] [PTX ISA, parallel synchronization and communication instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)

The `[V-*]` markers refer to the matching local experiment directory. Vendor
documentation does not replace a run on migration hardware; those cells remain
⚠️ until the experiment is rerun there.
