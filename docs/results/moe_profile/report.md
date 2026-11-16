# Qwen3-30B-A3B：BF16 闭合请求组基线与 Decode 瓶颈证据

## 结论与完成范围

【已测事实 / verified】完成 vLLM、SGLang、TensorRT-LLM 的 **36/36 个普通测速点**，每点充分预热后同一进程测 3 轮，共 108 个有效测量轮次。完成 **18 段原生配置 Nsight Systems 时间线**（3 框架 × 3 代表配置 × Prefill/Decode）、11 个独立源码/NVTX 归属诊断、9 个真实路由诊断（每个 48 层 × 17 个 Decode forward，共 7,344 份逐层路由样本）。接受标准是 worker 成功、实际 token 数正确且硬件守卫返回 0；被排除的历史尝试保留在 [excluded_attempts.csv](excluded_attempts.csv)。

【已测事实】主要观察是：B64 短上下文 Decode 的专家计算/融合 MoE 区域占 kernel 时间和约 74–76%；B1 短上下文的贡献更分散；B4 长上下文的 Attention/KV 区域升至约 21–26%。这些是 profiler ROI 的贡献，不能直接当作无 profiler 端到端占比。

【已测事实】两项受控验证没有建立可直接启用的模型优化：静态裁剪空 CTA 启动上界的名义收益小于顺序漂移；Nsight node→graph 粒度对照说明，vLLM B1 node 时间线的大量额外空隙来自测量扰动。**没有开发或启用新的优化系统。**

【证据支持的推断 / inferred】闭合请求组足以作为本轮执行瓶颈研究的主实验，不需要把在线 arrival/queue 调度作为研究主题。本轮没有执行在线 serving 补测或 TP2；所有结果 TP1，不能推断专家并行通信或在线排队瓶颈。底层引擎仍可能 admission、subbatch、chunked prefill 或 preemption，因此“同时提交 B 个请求”不等于每个 kernel 的实际 M=B。

【已测事实】B64、I8192/O256 的三个点全部完成，但**仅属于容量受限的闭合请求组结果**。权重文件 56.873 GiB，模型 BF16 KV 98,304 B/token，64×(8192+256) KV 需 49.5 GiB；两者下限已超过单张 H800 80GB。不能把这三个点称为全量驻留 fixed-B64 Decode，也没有用 TP2 或缩短输入/输出替代。

## 环境、输入与方法

【已测事实】当前环境为 3×H800（不是历史 4 卡），每卡报告 81,559 MiB、132 SM，NUMA0、NV8 peer。主测速和原生时间线均用物理 GPU1（UUID `GPU-344fa712-6ca8-3673-b04b-33b9c3cbffac`），CPU16–31，逐项取得 GPU/NUMA 计时锁。独立源码/路由诊断主要用 GPU0，最后两个 TRT 路由重试用 GPU1；这些运行不提供正式性能。其他任务保留，未改频率、功耗或系统权限。可见 owner 受容器 PID namespace 限制。

【已测事实】驱动 580.167.08，nvcc 12.8.93，Nsight Systems 2024.6.2.225，Nsight Compute 2025.1.1。Nsight GPU canary 通过；NCU canary 返回 `ERR_NVGPUCTRPERM`，因此 SM/Tensor Core 吞吐、occupancy、DRAM bytes/GB/s、L2、stall、合并访问及 Roofline **均未测得**。kernel 资源来自 CUPTI，不代替硬件计数器。Nsight 提示 CUDA13 driver 兼容性并使用 CUPTI12.8 fallback；CPU perf sampling 也因权限不可用。详见 [environment.json](environment.json)。

| 引擎 | 版本 | 实际 Attention / MoE / 采样 |
|---|---|---|
| vLLM | 0.11.0 `gf71952c1c`；Torch2.8.0+cu128 | FlashAttention SM90 FA3 / 非量化 Triton / Torch fallback |
| SGLang | 0.5.5；sgl-kernel0.3.16.post5；Torch2.8.0+cu128 | FA3 / 参数 auto、实际 Triton helper / FlashInfer |
| TensorRT-LLM | 0.21.0；TRT10.11.0.33；Torch2.7.1+cu128 | TRTLLM Attention / CUTLASS MoE；**PyTorch backend** |

【已测事实】版本/source tree 指纹在 environment.json；代表配置的实际初始化参数/命令在 [launch_parameters.json](launch_parameters.json)，全部初始化参数、graph/chunked prefill/token budget、backend 类和每次命令还保存在各原始证据目录的 `command.json`、`worker/engine.json`、`worker.log`。这是现有可运行环境的系统基线比较，不是控制 Torch/实现版本完全相同的 backend 因果实验。TRTLLM TP1 使用记录过的单进程 worker 设置，避免当前容器 MPI 启动问题。

【已测事实】模型固定为 [Qwen/Qwen3-30B-A3B@ad44e777](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39)，16 个实际预训练 BF16 shard 的大小和 SHA256 与该 revision 的 LFS 一致；下载经官方 Qwen ModelScope 镜像，不使用随机权重。权重身份在 [data_identity.json](data_identity.json)。BF16 权重/激活/KV，无量化、offload、speculation；图与融合保留，前缀跨请求/轮次复用关闭。

【已测事实】输入为基线提交 `ebe2fb0ed9d56b9b1e1abdd57c8fa8fd6a835f4c` 的 577 个公开技术文档/源代码文件组成的 1,465,627-token 语料，按固定窗口各生成 64 条精确长度 128/2048/8192 的 token IDs。固定 Qwen tokenizer/chat template、thinking=false、greedy、seed20261116；ignore-EOS 强制输出 128/256，实际数量逐轮核对。输入 SHA256 `81b43d7c6a501b1aea6ec9a32a4c63941f4b9e01bce4a446ea951f66947c2edf`，保留 [manifest](../../../configs/moe_profile/inputs_manifest.json) 与确定性生成脚本。语料可能跨文件边界，仅代表技术工作负载；不能用它代表一般对话路由分布。随机权重/随机 token 没有替代主实验。

【已测事实】各点 3 个完整 warmup cohort 后测 3 个完整 cohort，计时不含加载、编译/autotuning、capture。TTFT 为组提交到每个请求首 token 的主机观测值；TPOT=(last−first)/(实际 O−1)；E2E 为组 wall time；吞吐=B×O/E2E。取轮次中位数、极差，原始 chunk 数/输出 hash 保留。它们不是 GPU iteration 时长。轮间 JSON 写出也不计时；不是三个独立进程的置信区间。

【已测事实】各引擎 KV 均 BF16，但预分配策略分母不同：vLLM 总显存比例 .93，SGLang static .93，TRTLLM 权重后剩余 .85；容量约 175.7k、175.7k、185.3k tokens。SGLang 实测 BF16 KV pool16.08GiB，TRT pool约16.96GiB（启动日志）；不宣称容量完全相同。CSV 中峰值显存是加载/预热/计时期间 1s NVML 最大样本，可能漏峰；Torch allocator 峰值另列且不含外部分配。峰值活跃 KV 不可测，诊断保存实际 KV 序列长度。

【已测事实】轮次首尾 SM 时钟在 CSV 单列；最早 vLLM B1/I128 接受运行尚未增加边界 NVML 采样，仅保留守卫采样时钟范围，该格边界值留空。守卫的 running 时钟范围包含加载/预热，不冒充计时区间精确频率；其余可得边界记录与原始温度/功耗一并保留。

【已测事实】SGLang 8192+256 的 context_limit 需 8450：安装版本在 tp_worker 和 scheduler 各预留一个位置。8448 拒绝、8449 输出被夹成255的尝试均排除并保留；8450 接受且实际8192/256不变。SGLang cached_tokens 实测为0。其 queue/prefill_latency/forward_time 等原生 SDK 秒值已写 CSV，不能把 forward_time 解释为纯 GPU Decode；vLLM/TRT SDK 引擎内分段时间不可取得，留空。

## 无 profiler 基线性能

【已测事实】下表都是 3 轮中位数。TTFT/TPOT/E2E 单位 ms，吞吐单位输出 tokens/s；全部 batch/长度/实际 token 数和每项极差在 [benchmark.csv](benchmark.csv)。`*` 为容量受限 cohort。

| 引擎 | B | I/O | TTFT | TPOT | E2E | 输出 tokens/s | TPOT轮极差 |
|---|---:|---|---:|---:|---:|---:|---:|
| vllm | 1 | 128/128 | 22.643 | 5.833 | 763.710 | 167.6 | 0.021 |
| vllm | 4 | 128/128 | 59.577 | 8.574 | 1147.124 | 446.3 | 0.088 |
| vllm | 16 | 128/128 | 96.950 | 12.953 | 1741.772 | 1175.8 | 0.050 |
| vllm | 64 | 128/128 | 229.749 | 18.423 | 2583.570 | 3170.8 | 0.184 |
| vllm | 1 | 2048/256 | 68.431 | 5.975 | 1591.952 | 160.8 | 0.043 |
| vllm | 4 | 2048/256 | 167.299 | 9.031 | 2473.641 | 414.0 | 0.014 |
| vllm | 16 | 2048/256 | 569.417 | 15.150 | 4430.270 | 924.5 | 0.432 |
| vllm | 64 | 2048/256 | 1722.429 | 27.600 | 8888.708 | 1843.2 | 0.429 |
| vllm | 1 | 8192/256 | 232.788 | 6.195 | 1812.204 | 141.3 | 0.024 |
| vllm | 4 | 8192/256 | 809.596 | 9.919 | 3348.607 | 305.8 | 0.079 |
| vllm | 16 | 8192/256 | 2210.131 | 22.176 | 7946.983 | 515.4 | 0.073 |
| vllm | 64* | 8192/256 | 13943.428 | 39.621 | 34805.029 | 470.7 | 0.046 |
| sglang | 1 | 128/128 | 37.759 | 4.844 | 653.140 | 196.0 | 0.024 |
| sglang | 4 | 128/128 | 40.266 | 7.625 | 1009.044 | 507.4 | 0.001 |
| sglang | 16 | 128/128 | 66.635 | 11.582 | 1537.599 | 1331.9 | 0.006 |
| sglang | 64 | 128/128 | 209.507 | 16.455 | 2300.939 | 3560.3 | 0.008 |
| sglang | 1 | 2048/256 | 64.741 | 4.971 | 1332.088 | 192.2 | 0.002 |
| sglang | 4 | 2048/256 | 207.617 | 7.846 | 2208.990 | 463.6 | 0.005 |
| sglang | 16 | 2048/256 | 522.288 | 13.794 | 4040.903 | 1013.6 | 0.014 |
| sglang | 64 | 2048/256 | 1758.760 | 26.096 | 8415.802 | 1946.8 | 0.022 |
| sglang | 1 | 8192/256 | 236.624 | 5.207 | 1564.418 | 163.6 | 0.004 |
| sglang | 4 | 8192/256 | 595.410 | 9.939 | 3133.954 | 326.7 | 0.025 |
| sglang | 16 | 8192/256 | 2018.812 | 22.305 | 7707.920 | 531.4 | 0.109 |
| sglang | 64* | 8192/256 | 12737.956 | 27.981 | 32442.040 | 505.0 | 0.031 |
| trtllm | 1 | 128/128 | 29.110 | 5.272 | 698.782 | 183.2 | 0.003 |
| trtllm | 4 | 128/128 | 56.660 | 7.825 | 1051.312 | 487.0 | 0.027 |
| trtllm | 16 | 128/128 | 82.654 | 12.029 | 1610.057 | 1272.0 | 0.091 |
| trtllm | 64 | 128/128 | 191.457 | 17.249 | 2382.537 | 3438.4 | 0.136 |
| trtllm | 1 | 2048/256 | 56.860 | 5.288 | 1405.589 | 182.1 | 0.004 |
| trtllm | 4 | 2048/256 | 169.507 | 8.011 | 2213.744 | 462.6 | 0.052 |
| trtllm | 16 | 2048/256 | 460.414 | 13.832 | 4003.997 | 1023.0 | 0.091 |
| trtllm | 64 | 2048/256 | 1411.544 | 24.352 | 7765.455 | 2109.9 | 0.478 |
| trtllm | 1 | 8192/256 | 182.190 | 5.531 | 1591.918 | 160.8 | 0.004 |
| trtllm | 4 | 8192/256 | 663.650 | 9.224 | 3031.635 | 337.8 | 0.069 |
| trtllm | 16 | 8192/256 | 1838.018 | 20.499 | 7200.819 | 568.8 | 0.418 |
| trtllm | 64* | 8192/256 | 12560.046 | 36.757 | 32061.883 | 511.0 | 0.451 |

【已测事实】三个轮次的输出 hash 不完全一致的配置数：vllm=3/12, sglang=0/12, trtllm=12/12。固定输入/种子不等于各系统或各轮数值逐位一致；路由统计只描述各自实际生成的独立诊断，不将跨引擎路由差异归咎于调度，不做质量等价声明。

## 原生 Prefill / Decode 阶段贡献

【已测事实】代表点是 B1/I128、B64/I128、B4/I8192，各采 Prefill 与稳定 Decode 一段短 ROI。Prefill 从提交到各请求首 token，尾部可能混入已经开始的 Decode，**不称为隔离纯 Prefill**；Decode 以前端最低输出数32→48触发，流水线可领先，原生时间线不与另一个诊断逐 token 强配。路由诊断按实际模型 pure-Decode ordinal32..48取样并保存 KV 长度。

【证据支持的推断】阶段归属通过独立 eager/关闭 compile 的诊断，以实际模块/helper 源码路径、行号、SHA、NVTX 与同进程 CUDA runtime correlation 确认，再将相同符号/资源映射到原生配置；原生时间用于统计，**独立 eager 时间未冒充生产时间**。精确匹配同配置 kernel+grid/block/寄存器/smem 优先，fallback 为在源码诊断实际观测到的完整符号；多义专家 GEMM 合并 gate/up/down，不可解析的泛型内核保留 unresolved。`source_stage_candidates` 保存全部候选，单条 source_evidence 只是证据样例，绝不按名称单独猜测。TRT opaque MoE 按整个 `fused_routing_experts_combine` 区域报告，不人为拆分内部 kernel 为不可测操作。

【已测事实】下表 Σ 是 kernel 时间和，busy 是所有 kernel 区间并集，span 是首 kernel 到末 kernel 的 GPU 时间范围，单位 ms；copy/memset、graph span 另列 [gpu_windows.csv](gpu_windows.csv)。Σ 可重叠、stage union 不可直接相加；span 不含外部主机工作，三者都不是 E2E。所有比例分母均为该 ROI 的 kernel 时间和，node tracing 存在下节量化过的扰动。

| 引擎 | B/I | ROI | kernelΣ ms | busy ms | span ms | 专家/融合MoE % | Attention/KV % | QKV+O % | unresolved % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang | 1/128 | decode | 75.626 | 73.732 | 80.395 | 38.1 | 13.6 | 19.8 | 0.7 |
| sglang | 1/128 | prefill | 27.949 | 27.763 | 73.445 | 66.3 | 6.7 | 6.7 | 4.8 |
| sglang | 4/8192 | decode | 136.977 | 133.448 | 144.162 | 49.5 | 20.8 | 10.9 | 3.2 |
| sglang | 4/8192 | prefill | 945.631 | 944.975 | 953.376 | 57.9 | 19.1 | 8.5 | 2.1 |
| sglang | 64/128 | decode | 269.401 | 262.676 | 269.848 | 75.2 | 7.0 | 5.9 | 2.0 |
| sglang | 64/128 | prefill | 211.743 | 210.830 | 214.432 | 68.9 | 3.4 | 10.0 | 2.6 |
| trtllm | 1/128 | decode | 79.297 | 79.297 | 100.495 | 44.3 | 19.4 | 18.7 | 2.3 |
| trtllm | 1/128 | prefill | 28.219 | 28.219 | 40.324 | 62.1 | 6.9 | 6.6 | 15.8 |
| trtllm | 4/8192 | decode | 125.370 | 125.311 | 130.601 | 51.7 | 26.3 | 11.4 | 1.4 |
| trtllm | 4/8192 | prefill | 730.277 | 730.270 | 743.625 | 7.5 | 1.5 | 0.3 | 87.8 |
| trtllm | 64/128 | decode | 272.171 | 272.062 | 277.593 | 75.9 | 12.0 | 5.8 | 1.0 |
| trtllm | 64/128 | prefill | 163.211 | 163.199 | 185.531 | 15.4 | 2.8 | 1.2 | 76.0 |
| vllm | 1/128 | decode | 78.792 | 78.058 | 124.733 | 38.4 | 11.8 | 18.6 | 10.2 |
| vllm | 1/128 | prefill | 18.790 | 18.681 | 22.661 | 80.9 | 1.5 | 0.0 | 11.3 |
| vllm | 4/8192 | decode | 129.384 | 128.595 | 174.064 | 47.9 | 20.8 | 11.5 | 6.3 |
| vllm | 4/8192 | prefill | 924.443 | 924.295 | 949.435 | 63.9 | 21.1 | 8.8 | 4.1 |
| vllm | 64/128 | decode | 264.882 | 264.364 | 350.234 | 77.1 | 6.3 | 5.9 | 3.5 |
| vllm | 64/128 | prefill | 213.635 | 213.600 | 267.009 | 78.4 | 4.2 | 0.5 | 14.0 |

【已测事实】TRT Prefill unresolved 最高87.8%，这些 kernel 不能从本轮独立源码追踪可靠迁移归属；不补估计值。具体未覆盖的 tactic/异步路径仍【待验证】。Decode unresolved 为0.7–10.2%（随框架/配置），足够区分已确认的大区域，但不构成每个操作完整覆盖。完整分项含 Router Projection、TopK/归一化、index/padding、Norm/Residual、LM Head 等，见 [stage_breakdown.csv](stage_breakdown.csv)。TP1 无跨卡推理通信；没有据此评价多卡专家负载或通信效率。

## 热点 kernel 与可测指标

【已测事实】下表每个代表 Decode ROI 取累计时长最大的一个 kernel/资源组合；同一融合 C++ MoE 区域可包含多个内核，单个热点不等于完整区域。完整符号、实际 grid/block、寄存器/smem、调用数、中位时长与来源在 [kernel_metrics.csv](kernel_metrics.csv)，shape 由独立真实 helper/路由诊断补充，不把源诊断的 shape 误标成每次原生 launch 的已测 M/N/K。

| 引擎 | B/I | 来源确认的区域 | 次数 | 单次中位µs | 累计ms | grid | block | 寄存器/线程 | smem bytes(静态+动态) |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| vllm | 1/128 | expert_gate_up_gemm | 768 | 21.54 | 16.595 | (384, 1, 1) | (128, 1, 1) | 40 | 12288 |
| vllm | 64/128 | expert_gate_up_gemm | 768 | 168.56 | 129.774 | (7296, 1, 1) | (128, 1, 1) | 40 | 12288 |
| vllm | 4/8192 | expert_gate_up_gemm | 768 | 49.71 | 36.893 | (1536, 1, 1) | (128, 1, 1) | 40 | 12288 |
| sglang | 1/128 | expert_gate_up_gemm | 768 | 22.37 | 17.213 | (5856, 1, 1) | (128, 1, 1) | 40 | 12288 |
| sglang | 64/128 | expert_gate_up_gemm | 768 | 169.12 | 130.944 | (7344, 1, 1) | (128, 1, 1) | 40 | 12288 |
| sglang | 4/8192 | expert_gate_up_gemm | 768 | 55.97 | 41.464 | (5904, 1, 1) | (128, 1, 1) | 40 | 12288 |
| trtllm | 1/128 | fused_routing_experts_combine | 768 | 22.62 | 17.425 | (264, 1, 1) | (128, 1, 1) | 188 | 106496 |
| trtllm | 64/128 | fused_routing_experts_combine | 768 | 168.57 | 130.901 | (264, 1, 1) | (128, 1, 1) | 186 | 81920 |
| trtllm | 4/8192 | fused_routing_experts_combine | 720 | 49.50 | 37.746 | (264, 1, 1) | (128, 1, 1) | 188 | 106496 |

【已测事实】NCU 的原始错误与 canary 输出位于 `runs/moe_profile/setup/ncu_canary_metrics.csv`，没有 replay/cache/clock 参数有效采样，后续热点 NCU 未伪造结果。kernel_metrics 的 throughput/occupancy/DRAM/L2/FLOPs/AI 字段空白即 unavailable；NVML utilization 或理论 occupancy 没有填进这些字段。未获得可靠内存流量分母，因此没有画实测 Roofline，也没有把 kernel 利用率声称为整模型利用率。

## 真实路由、shape 与物化

【已测事实】每个 case 的48层×17步均验证 Σn_e=B×8、实际 token 行数=B；SGLang 还将 consumed top-k IDs 计数与原生 expert_distribution_recorder 逐层交叉核对。样本保留在设备上，ROI 后批量导出，未在每步同步复制到 CPU。独立诊断不参与正式测速。每专家均值/最大值、n_e 直方图、活跃/空专家、负载偏斜、KV范围在 [routing_stats/per_layer.csv](routing_stats/per_layer.csv)；每个原始 n_e 向量见其中指向的本机 `routing.json`。其原始统计为诊断数据，未作为权重/输入清理目标。

| 引擎 | B/I | 平均活跃/128 | 平均空专家 | 全样本最大n_e | 实际KV长度范围 | 平均padded行数(每GEMM) |
|---|---|---:|---:|---:|---|---:|
| sglang | 1/128 | 8.00 | 120.00 | 1 | 160–176 | 128.00 |
| sglang | 4/8192 | 22.89 | 105.11 | 4 | 8224–8240 | 366.20 |
| sglang | 64/128 | 78.67 | 49.33 | 57 | 160–176 | 1373.35 |
| trtllm | 1/128 | 8.00 | 120.00 | 1 | 160–176 | 不可测 |
| trtllm | 4/8192 | 21.68 | 106.32 | 4 | 8224–8244 | 不可测 |
| trtllm | 64/128 | 78.29 | 49.71 | 58 | 160–177 | 不可测 |
| vllm | 1/128 | 8.00 | 120.00 | 1 | 160–176 | 128.00 |
| vllm | 4/8192 | 22.17 | 105.83 | 4 | 8224–8244 | 354.73 |
| vllm | 64/128 | 77.48 | 50.52 | 59 | 160–177 | 1358.88 |

【已测事实】TRT query_seq_lens=1 是 query 长度，不是 KV 长度；最终取 `kv_seq_lens`，旧缺KV诊断不进入最终统计。B4 的各请求由于 Prefill/调度先后可能有不同 KV 长度，故列范围而非只写 I+ordinal。B1 恰好8个专家各1条：max/全128专家平均=16源自 top8 稀疏性，不能据此认定负载失衡；报告同时给出 max/活跃专家均值。真实路由没有被均匀/倾斜合成路由替代。

【已测事实】Vllm/SGLang Decode 的实测 helper 为 Gate/Up 权重 `[128,1536,2048]`、Down `[128,2048,768]`，每专家逻辑 GEMM 分别 `(M=n_e,N=1536,K=2048)`、`(M=n_e,N=2048,K=768)`；诊断配置为 M16/N32/K64、GROUP_M1。消费的张量、stride、top_k、sorted_ids上界与实际GPU padding计数见 [gemm_shapes.csv](routing_stats/gemm_shapes.csv)。TRT 权重 shape 可见，CUTLASS 内部 tile/padding/workspace 在可用接口中不可见，明确记不可测，不沿用 Triton 的 tile。

【证据支持的推断】安装源码及真实 helper 参数表明，Triton 第一 GEMM 用 sorted token IDs / top_k 从原始 A 读入，把 input Gather 融入 GEMM；没有独立物化完整 `[B×8,2048]` 输入复制。Gate/Up 输出、激活输出、Down/Combine 前缓冲显式存在；不把 Gather 预设为独立热点。输出链可由实际 shape 确认：BF16 Gate/Up `B×8×1536`、激活 `B×8×768`、Down临时 `B×8×2048`，逻辑容量分别24,576B、12,288B、32,768B（每个 B），按顺序/复用不代表同时驻留总额，更不等于实际 DRAM bytes。Combine 通过独立 `moe_sum`/归约路径（原生图下可能另融合）完成；TRT opaque MoE 的内部物化无法直接量化。

## 最小受控验证

【已测事实】控制一：同卡同输入/参数，原生 vLLM B1 短 Decode 仅切换 Nsight `--cuda-graph-trace=node` 与 `graph`。均16次整模型 graph launch，worker 边界镜像序列长度160→176；graph/kernel span 包含内部间隙，不是SM busy。结果在 [controls.json](controls.json)：

| 模式 | GPU graph起点间隔中位ms | graph span中位ms | 对无profiler TPOT比值 |
|---|---:|---:|---:|
| node | 7.646547 | 4.748208 | 1.3108 |
| graph | 5.853734 | 4.618658 | 1.0035 |

【证据支持的推断】node cadence约+31%，graph约+0.35%，无profiler TPOT5.833ms。因此 node 额外间隙大部分由 profiling 引入，**没有证据把它认定为系统 CPU 调度缺陷**。主机侧 profile start/stop、RPC 与 CUPTI flush 又引入更大暂停，不用诊断运行的host TPOT评价引擎。这是 B1 上的控制，不把同一校正系数套到其他 case。

【已测事实】控制二：保留真实模型第0层 Decode40 的输入/权重/路由与原始 Triton kernel/tile/算术，仅把 sorted_ids launch 上界裁为安全静态 bound=`assignments×BLOCK_M`；每个 assignment 至多贡献一个 padded block。两侧同样创建 view，Python dispatch 在 capture 外。50次warmup后 CUDA Graph replay，ABBA各100个GPU event样本；输出均逐位一致。该探针不是生产系统改动。

| B | ABBA中位µs | 实际padded行 | 安全上界 | 名义下降 | 同侧顺序极差µs |
|---|---|---:|---:|---:|---:|
| 1 | 46.768, 46.848, 46.960, 48.992 | 128 | 128 | 2.04% | 2.224 |
| 4 | 118.016, 118.000, 118.304, 119.936 | 480 | 512 | 0.69% | 1.920 |

【已测事实】B4 Down 的启动 grid 上界可由1952行裁至512行（122×64→32×64 CTA），实际 padded math不变；但约0.69%的名义下降小于顺序漂移，B1不改变有效上界的负对照同样有漂移，**没有建立可靠收益**。此前未capture且view操作不对称的试验被 superseded，不进入结论。单层反复重放缓存与全模型不同，没有把其时间乘48当作模型收益。

## 按贡献排序的三个研究候选

1. **专家 GEMM/融合 MoE 的小M与跨专家执行。**【已测事实】B64/I128 Decode，vLLM/SG 的 Up+Down kernelΣ占74.4%/73.8%，TRT整个融合区域75.9%；真实路由平均约77–79个活跃专家、最大n_e57–59，多数专家行数仍小。【证据支持的推断】它是最值得继续测硬件流量与执行尾部的区域；小batch权重复用不足和padding是工作负载/实现共同约束，不能仅用低TC指标或活跃专家数定性为缺陷。【已测事实】静态空CTA裁剪未验证明显收益。【待验证 / hypothesis】真实route replay下的tile/padding、权重与L2行为、融合边界是否可改善；仍缺DRAM/L2/TC计数、CUTLASS内部padding与同KV/同route的整模型A/B。证据：stage_breakdown、routing_stats、controls.json。
2. **长上下文 Attention/KV 区域。**【已测事实】B4/I8192 Decode Attention/KV占 kernelΣ20.8%/20.8%/26.3%（vLLM/SG/TRT），KV实测约8224–8244。【证据支持的推断】比短B64的Attention占比更高，是上下文增长后的实质贡献，且不能以Gather/路由热点替代解释。【待验证】KV读带宽、Attention拆分/归约、工作划分哪项主导；当前没有同路由同KV backend单因素对照与DRAM计数，不认定某Attention实现低效。证据：对应decode native_trace、attention源码/NVTX归属、KV诊断。
3. **B1 稠密投影和小算子链。**【已测事实】B1/I128的QKV+O占约18.6–19.8%；vLLM另有TopK/归一化约7.7%，SG Norm/Residual约8.7%。vLLM源码TopK路径包含转换、topk_softmax、sum、div多次launch。【证据支持的推断】融合/布局/调度边界值得研究，但投影本身权重流量不因消除launch而消失，已有图/融合必须保留。【已测事实】node/graph控制排除了把大段node空隙直接归因CPU缺陷的论据。【待验证】小链局部融合在保留相同数值/route下的端到端收益；缺逐操作生产归属与同KV的真实数值A/B，不预报收益。

【证据支持的推断】编译优化可以介入上述源代码确认的helper/融合边界、shape特化、布局与内存生命周期。仅作敏感度上界：若局部贡献为p且完全可移除、无重叠/其他变化，则 wall时间最多下降p、加速上限1/(1−p)。由于这里p来自受扰动的kernelΣ，**以下不是已验证E2E上界**，只是待无profiler A/B校准的范围：B64专家候选约0–74/76%的kernelΣ，B4长Attention约0–21/26%，B1 QKV+O约0–19/20%（TopK链另0–8%，不得重复计融合部分）。目前测得并可归因的端到端优化收益均为“未建立”；最低可实现收益包括0。不得把这些比例相加或承诺同等E2E下降。

## 失败、限制与复现材料

【已测事实】历史失败包括：vLLM可信本地callable RPC开关未开；SGLang lazy tokenizer event loop未启动、skip-tokenizer路径不支持min_new_tokens、长context边界；最早vLLM Decode capture无GPU表；TRT诊断在推理/导出/LLM.shutdown后C++全局析构SIGSEGV；以及守卫因他人占卡返回75。全部有原始日志且排除。TRT仅对成功完成的profile/routing在shutdown与flush后执行本进程`os._exit(0)`，规避已记录的全局退出缺陷；异常仍失败，普通benchmark路径未用该绕过。仅终止过本任务已确认的卡住进程，未终止他人任务，未更改共享core_pattern。

【已测事实】不同已安装软件栈、技术语料、单进程三轮、跨引擎/轮次生成差异、容量受限cohort、node tracing扰动、Prefill mixed尾部、部分kernel归属未解析、NCU权限以及TRT内部opaque区域共同限定结论。所有低利用率/带宽原因仍缺计数器时保留待验证，不将源码推断升级为已测事实。

【已测事实】可复现工具和协议在 [README.md](README.md)、`scripts/moe_profile_*.py`、`configs/moe_profile/`。最终CSV/identity/控制摘要随本分支推送；原始trace/sqlite、日志、失败尝试、逐步n_e数据和生成输出hash留在本机忽略目录 `runs/moe_profile/`，未提交原始大文件。本轮专用权重、tokenizer副本、完整输入token-ID payload以及含权重/激活的replay文件，按用户要求在数据验证后删除，记录见 [cleanup.json](cleanup.json)。保留hash和确定性配方，复跑需重新下载模型并生成相同输入；不承诺git历史中的旧输入blob被抹除。
