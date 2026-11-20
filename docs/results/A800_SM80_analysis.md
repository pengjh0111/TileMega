# A800 单卡 decode 性能定位

2026-11-20。审计分支为 `tilemega`，提交 `7425a5242580af16a7ea43be742e313a6eedb351`。H800 历史实验保留各自记录的设备条件。下表同时保留 H800 绝对时间、A800 正式时间和 A800 硬件诊断。A800 的原始请求时间见[完整时间表](A800_execution_times.md)。

## 同一模型的时间预算

已核验六个 dense 模型的 config SHA 与 H800 R6 对应行一致。下表为 B1。逻辑字节数沿用 R6 的权重和 KV 定义。估算列取该字节数除以 A800 实测传输率 1685.62 GB/s。它假设每步按此口径经过 HBM，没有考虑缓存命中、额外事务或阶段重叠。它用于估计时间量级，不能当作硬件计数器读数或可回收时间。H800 的 bulk 读探针测得 3235.332 GB/s，两边探针的指令和访存方式不同。[H800 字节与计时来源](SM90R6_hw_bound.csv) · [H800 各配置带宽与计时条件](SM90R6_hw_peaks.csv)

| 模型 | 每步逻辑数据量，GB | A800 按实测带宽折算，ms | A800 正式 TPOT，ms | H800 图表对应 TPOT，ms |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1.258 | 0.746 | 1.658 | 0.929 |
| Qwen3-1.7B | 3.507 | 2.081 | 3.561 | 1.675 |
| Llama-3.2-1B | 2.490 | 1.477 | 2.603 | 1.120 |
| Llama-3.2-3B | 6.491 | 3.851 | 5.183 | 2.829 |
| Llama-3.1-8B | 15.085 | 8.949 | 12.233 | 5.568 |
| Mistral-7B-v0.3 | 14.303 | 8.485 | 11.764 | 5.331 |

按上述假设，在较大的 dense 模型上，单靠消除软件开销，达到 H800 的绝对毫秒数会很困难。A800 本机的实测 baseline 提供更直接的参照。Llama-3.1-8B B1 的 SGLang 为 10.997 ms/token，Mistral 为 10.553 ms/token。两个数都低于当前 TileMega。MoE 的每步活跃专家数取决于实际路由，因此这里没有把检查点总大小当作每步读取量。[A800 正式 baseline](A800_baselines.csv) · [R6 路由审计](SM90R6_h800.md)

## 从 SM90 开发过程复核 SM80

| H800 分支实验 | 依据及可迁移的结论 | 已在 A800 查明的情况 |
|---|---|---|
| R1 的 scratch 生命周期、barrier 退役及 MoE combine 所有权 | 历史失败依靠完整输出、独立进程和针对性探针定位。五次新进程与 step/loop 用于区分不稳定结果。 [R1](SM90R1_h800.md) | 全长输出与 HF 检查保留。Qwen-30B 作业 345794 的 HF 进程在 CPU 载入权重时被 Slurm OOM-kill。保留另外四轮，改由 HF 将分片直接加载到单张 GPU。在 345934 中原 checkpoint、全部 1024 个位置、原阈值均通过，最大 gap 0.03125，五轮输出哈希一致。 |
| R2 到 R3 的 GEMM 搬运、寄存器流水、ring、分块注意力和实测选择 | R3 的预测第一名只在 30 格中的 8 格成为实测第一。跨阶段 weight stream 在四个 dense 开发格均更慢。 [R2](SM90R2_h800.md) · [R3](SM90R3_h800.md) | A800 有 108 个 SM，每 CTA 可申请 166912 字节共享内存。SM80 的 32 个 producer lanes 搬运权重和 activation，采用显式地址和 mbarrier。B16 Llama-1B 只将 gate-up、down 两个 ring 加深到 3，正式 TPOT 从 3.025 ms 降到 2.864 ms。全面放宽 ring 后选出的深流水在实测中更慢。 |
| R4 到 R5 的 MoE 页布局、路由与缓存 | R5 对 Qwen-30B 启用一次性权重流的 L2 evict-first 时，同卡 B1 配对实验的 1024 个 token 一致。路由和 combine 的改动在较大 batch 更有价值。 [R4](SM90R4_h800.md) · [R5](SM90R5_h800.md) · [H800 源码变更](https://github.com/pengjh0111/TileMega/commit/8724b8f) | A800 生成的 MoE 源码定义了权重优先淘汰宏，但在原 SM80 分支中权重块走普通 `cp.async.cg`。实验分支 `86c881c` 为一次性权重的 16 字节搬运接上已有缓存提示，覆盖连续块、专家页与预发。激活、残差和 KV 维持对照策略。本实验单独验证权重策略。两模型实验二进制已完成编译，生成 CUDA 与各自对照逐行一致，编译时使用实验分支的后端头文件。OLMoE、Qwen 的交错配对作业分别为 345972、346025。 |
| R6 的阶段回放、归属和硬件预算 | 按实际资源和每一阶段工作量解释预测误差，再在完整请求中验收。 [R6](SM90R6_h800.md) · [阶段回放实现](../../lib/Solver/Sm90NativePrices.cpp) | 模型的 MMA 某些 key 缺测时，阶段预测与流水评分会用零计入该项算术成本。其余带宽、延迟及同步成本仍计入。当前 65 个实测 key 中没有两个已选的关键形状。 |

编译依赖固定在 CUTLASS `dc45f979ae336a235da1676b311f35efeb30149a`。这里复用分块、共享内存布局和寄存器流水的原则。SM80 producer/consumer 分支需要保持各自的参与范围，直接插入带全 CTA barrier 的 mainloop 会改变执行约定。[固定版 CUTLASS 多级流水](https://github.com/NVIDIA/cutlass/blob/dc45f979ae336a235da1676b311f35efeb30149a/include/cutlass/gemm/threadblock/mma_multistage.h) · [NVIDIA Ampere 调优指南](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)

## 实测与下一轮问题

| 模型与阶段 | 已核验的证据 | 对照实验及验收标准 |
|---|---|---|
| Llama-3.2-1B B1 FFN down | 作业 `345633` 中 16 个标记 step 的阶段归属合计为 849.9 µs，预测为 391.0 µs。attention 为 283.6 / 117.5 µs，O 为 251.4 / 134.6 µs。trace 和无插桩对照在这次 520-token 诊断中输出相同，trace 开销约 1%。down、O 的 `108:8:1024:1:1` 均缺测。 | 用链接产物核对寄存器、栈与 spill，单卡五个独立进程补测大 K。用新 target 重新排名，比较四个 shortlist 候选的 1024-token 请求，再用全长正确性与独立计时验证。阶段差值含所有未建模成本，尚无法归因于单一缺项。 |
| Llama-3.1-8B 与 Mistral-7B B1 | 两个实际选中的 down 都缺少 `108:16:512:2:1`。它们的 gate-up、O、qkv 与 head 均有匹配的 MMA key。 | Llama-8B 作业 345932 已完成。FFN down 为 3327.0 µs，预测 2442.6 µs。attention 为 912.9 / 306.2 µs。gate-up 为 5548.0 / 5817.0 µs。16 个诊断 step 的总时长为 12.334 ms，520-token 输出一致，插桩开销为 0.82%。已用此记录触发同坐标的 down ring 深度对照。Mistral 作业 345938 独立复现，down 为 3311.6 / 2442.6 µs，attention 为 912.9 / 306.2 µs，插桩开销 0.64%。Llama-8B 的另一组固定坐标试验把 8 个 KV heads 的分块数从 8 改为 13，分别对应 64 和 104 个注意力任务，测试在 108 个 SM 上填满一波任务的效果。该候选的数值正确性与收益仍待验证。 |
| Qwen-30B 与 OLMoE B1 | 两个程序的权重 `CopySpan` 均未走已有的带 hint 指令。Qwen 使用共享专家页，OLMoE B1 使用自己的专家布局。 | 分别在同一张 A800 上交错比较相同坐标的六个独立进程，核对权重、设备及 token 哈希。OLMoE 两组都使用原有 prompt prefill。Qwen 先完成原始 artifact 的 HF 检查。只让请求计时和 TPOT 均改善的候选进入独立搜索、正确性与正式计时。 |

补充探针的 21 个大 K 内核已通过链接后审计，未发现被审计的栈、local 指令或 spill。五进程校准已排队，作业 346013。现有目标上按同一 TN、warp 划分、寄存器级数和最近的另一 TK 做留一验证，59 个可验证样本的绝对相对误差中位数为 24.1%。补充 target 因此只写入实测键。

五个校准进程各自验证单张物理 A800 的进程归属、设备和资源。补充 target 保留原始 65 项校准的来源，另记新增 key、GPU UUID、探针二进制及原始结果目录。运行中的历史搜索继续使用自己的 target SHA。原始证据位于服务器工作目录的 `TileMega-a800/runs/a800/` 和 `logs/`。诊断、调优与正式时间在采集时分别存放。[执行时间表](A800_execution_times.md) · [A800 结果记录](A800_report.md)

## Llama-3.2-1B B1 正式复测

| Solver 产物 | T1，ms | T1024，s | TPOT，ms/token | 正确性进程 | 计时进程 |
|---|---:|---:|---:|---:|---:|
| 原配置 | 7.61 | 2.670 | 2.603 | 5 | 3 |
| 新配置，down TN=16 | 7.58 | 2.467 | 2.404 | 5 | 3 |

作业数组 345891 和 345965 分别验证正确性、采集独立请求计时。两个产物使用相同的 prefill、模型配置、共享权重和 prompt。每个产物的五个进程生成相同的 1024 个 token，各自通过 step/loop 和完整 HF teacher-forced 检查。两次 HF 检查的最大 logit gap 均为 0.03125，所有位置均在 0.5 以内。

新旧产物的自由生成序列有 23 个位置不同，首次在第 88 个 token 出现。因此，要求两个产物逐 token 一致的交错配对实验仍记为失败。正式表按仓库既定的独立参考校验和计时协议选取新产物。两者使用的校准 target 也不同，这组正式时间反映整个产物的变化。服务器证据为 `runs/a800/optimization/llama-b1-down16-solver-independent.json`。

## 大 K 校准与专家权重缓存复测

单卡作业 346013 的五个进程均通过 guard。21 个新增形状与原 65 个形状没有重叠，校准表现有 86 个实测形状。下表的时间是每个 K16 步、每个 split-K warp 组的算术探针时间。参数顺序为 grid、TN、TK、WN、寄存器流水级数。[A800 校准 target](../../configs/targets/sm_80_a800_sxm4_80gb_mma.json)

| A800 形状 | 相关模型与阶段 | 中位数，ns | 进程内 P99 的中位数，ns |
|---|---|---:|---:|
| `108:8:1024:1:1` | Llama-3.2-1B B1 原配置的 FFN down 和 O | 42.234 | 42.550 |
| `108:16:512:2:1` | Llama-3.1-8B 与 Mistral-7B B1 的 FFN down | 41.225 | 41.506 |

目标文件 SHA-256 为 `318cc5e6d4d314f32bacf03f20d5c674267f26f3c4269296db1ce20ad77fc5ff`。新 target 已交给独立的 pipeline 搜索与后续正确性、正式计时流程。

OLMoE B1 作业 345972 在同一张卡上交错运行了六个进程。标准权重读取的 T1024 为 2.733 s、TPOT 为 2.518 ms/token。使用 evict-first 提示后分别为 2.919 s 和 2.690 ms/token，三组请求均变慢，正式配置继续使用标准读取。Qwen3-30B-A3B 的独立对照保留在作业 346025。
