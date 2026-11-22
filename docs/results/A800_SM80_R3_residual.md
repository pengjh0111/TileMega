# A800 SM80 · 剩余 decode 缺口

## 生产路径与轮次核对

**Verified** · 冻结矩阵以 BF16 TP1、64 输入和 1024 输出运行。GPU job 使用一个 A800，A/B 对照在同一张卡上顺序运行，不同 job 可以使用不同 A800。生产路径依次经过 `torch.export`、ModelPlan/CTG、pipeline 评分与 replay、SM80 tmexec 和 CUDA 生成、链接后驻留核验、persistent executor、完整请求复测。每个模型沿用正式 prefill 和权重配方。

| 轮次 | 已完成 | 待验证 |
|---|---|---|
| R0 | 已复核三格。Llama-3B B16 和 OLMoE B8 已有新正式数据。Llama-1B B16 在 0.50% 阈值内。 | 无需重复原对照。 |
| R1 | MMA 注册键由 92 增至 413。40 格正式坐标已重放。 | 新几何须补对应的实测价格。 |
| R2 | Llama-8B 和 Mistral B16 已采用 216 CTA 的 TN224。Llama-3B 当前正式产物采用 216 CTA、TN128 和 Q3 attention。 | 侧重 B16 的 gate/up 任务分配和驻留。 |
| R3 | 前一轮已正式采纳 9 个双驻留配置，覆盖 Llama-8B、Mistral 和 Llama-3B。 | 重点 dense B16 仍未达到 0.83 带宽效率。 |
| R4 | 已对照 MoE padding、attention mask 和 dense live-row，并为五个 residual 几何补测 epilogue。 | 保留经过端到端验证的现有选择。 |
| R5 | activation issue、wave tail、register stages 和 fresh KV 已进入普通成本路径。40 格重放误差中位数为 5.398%。Qwen B16 两个分页策略的排序已补价。 | 实际跨阶段 HBM 流量仍受权限限制。作业 `352601` 在 CUPTI 可用性查询返回 `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`。 |
| R6 | 有界 L2 预取已做同卡对照，收益落在平局阈值内。 | 保留关闭状态。 |
| R7 | Qwen30 B16 已正式采纳四槽共享环。两种路由变体均在平局阈值内。 | 检查相同共享环在 Qwen30 B1/B2/B4/B8 上是否有益。 |
| R8 | 生产路径仍使用 128 consumer 和 32 producer。 | 尚未开启新角色 ABI。更改参与范围需要新的校准和同步验证。 |

**Verified** · 正式起点为 31 胜、3 平、6 负，几何平均加速比 1.098623。三个重点 dense B16 的效率为 Llama-3B 0.7207、Llama-8B 0.8079 和 Mistral 0.7982。六个落后格为 Qwen30 的五个 batch 和 OLMoE B16。原基线冻结为 180 个有效格和 20 个 N/A。 [前轮审计](A800_SM80_R3_followup.md) · [正式矩阵](A800_llm.csv) · [效率表](A800_SM80_R3_followup_efficiency.csv)

## 本轮判读规则

**Stated** · 受控 stage 探针对照使用相同 CTA 数和共享内存预约。要求 guard 严格通过、链接驻留核验、探针输出逐位一致，以及最多 5% 的插桩扰动。Stage span 用于筛选，正式采用需要完整请求 TPOT 考核。完整请求的 control 与 trial 各用一个进程，在进程内做 warmup 和三次 N1/N1024 计时。三个计时对均改善、TPOT 改善超过 0.50% 且冻结 token 和权重检查通过，才进入校准、普通 solver 和正式验收。

## R2 · 双驻留下的 TN256

**Verified** · 作业 `353159` 将 SwiGLU 的 FP32 工作区放入已读完的 operand ring。仅在每个 CTA 最多一个 gate tile、没有预发、且 128 个消费者统一完成最后一轮读取时启用。TN256 的单 CTA 工作区原需 87,552 B 或 103,936 B，分别对应一个或两个 K split。复用后，两个几何均能在正式程序的 79,872 B 预约下达到两个 CTA/SM。所有探针均通过严格 guard，并与相同几何、独立工作区的输出逐位一致。 [Ampere 共享内存与异步复制](https://docs.nvidia.com/cuda/ampere-tuning-guide/)

| B16 gate/up | 正式几何 span (µs) | 同 tile 复用 (µs) | TN256 复用 (µs) | TN256 K split |
|---|---:|---:|---:|---:|
| Llama-8B | 151.552 | 150.528 | 160.768 | 1 |
| Mistral | 152.576 | 148.480 | 161.792 | 1 |
| Llama-3B | 70.656 | 71.680 | 73.728 | 2 |
| Llama-3B | 70.656 | 71.680 | 71.680 | 1 |

**Verified** · 对 Llama-8B 和 Mistral，TN256 比正式几何慢 6.1% 和 6.0%。Llama-3B 保持两个 K split 时慢 4.3%，改为单 split 的差异为一个计时档。TN256 未进入正式请求筛选。Llama-8B 和 Mistral 从 128 个活动 CTA 减到 112 个，同时有两个活动 CTA 的 SM 从 20 个减到 4 个。因此追加一组更窄、允许补齐末 tile 的候选，检验提高活动 CTA 覆盖是否能缩短 gate/up。工作区复用只存在于实验提交 `42a5ce2`。

## R5 · 当前正式程序的阶段归因

**Verified** · 作业 `353171_0–3` 各对照一份同卡未插桩参考。四格均匹配冻结 checkpoint，插桩与参考的 520-token 输出逐位一致，分析使用窗口内 15 个完整 decode step。QKV 与 attention 通过组级依赖衔接时，最早 release 会将一部分 QKV 尾部计入 attention，因此按相邻两阶段合计归因。表中的步进时间用于诊断。

| B16 模型 | 步进 (ms) | Gate/up (ms) | Down (ms) | QKV + attention (ms) | Fused expert (ms) | 扰动 | QKV 与 attention 有重叠 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-8B | 11.981 | 4.759 | 3.236 | 2.352 | — | 0.78% | 85/480 |
| Mistral | 11.607 | 4.769 | 3.239 | 2.367 | — | 0.95% | 65/480 |
| Llama-3B | 6.253 | 2.053 | 1.281 | 1.796 | — | 1.33% | 134/420 |
| OLMoE | 9.256 | — | — | 1.627 | 7.079 | 0.55% | 0/240 |

**Verified** · 三个 dense 模型的 QKV 与 attention 活动跨度重叠中位数均为 0，P90 均为 1.024 µs。Gate/up 占 dense 步进的 32.8%–41.1%，融合专家占 OLMoE 步进的 76.5%。这些跨度来自软件时间戳。实际跨阶段 HBM 带宽仍缺硬件计数。

## R2 · 窄 tile 与末 tile 补齐

**Verified** · 作业 `353177` 固定 216 CTA 和 79,872 B，比较精确整除与全宽补齐的 gate/up 探针。所有探针通过 guard，扰动不超过 3.18%。两个新 N 的补齐开销分别为 128/28,672 和 32/16,384 行。

| 模型 | TN | Slots | 活动 CTA | 同时承载两个活动 CTA 的 SM | 未插桩 span (µs) |
|---|---:|---:|---:|---:|---:|
| Llama-8B | 224 | 2 | 128 | 20 | 151.552 |
| Llama-8B | 160 | 2 | 180 | 72 | 156.672 |
| Llama-8B | 160 | 3 | 180 | 72 | 148.480 |
| Llama-8B | 192 | 2 | 150 | 42 | 148.480 |
| Llama-3B | 128 | 2 | 128 | 20 | 72.704 |
| Llama-3B | 96 | 2 | 171 | 63 | 65.536 |
| Llama-3B | 96 | 3 | 171 | 63 | 64.512 |
| Llama-3B | 96 | 4 | 171 | 63 | 64.512 |

**Inferred** · Llama-3B 的 TN96 三槽在该独立阶段比正式几何短 11.3%，且四槽没有额外收益。全请求候选沿用相同的两个 K split、216 CTA、对应 prefill 和其余 stage，权重增加 32 行零值。SwiGLU 保持逻辑行宽，只写有效列，避免改变下游 down 的输入布局。开始正式筛选前需要通过完整 HF、step/loop 和冻结 token 检查。Llama-8B 的探针只显示约 2.0% 的阶段收益，优先验证 Llama-3B。

## R7 · 融合专家环

**Verified** · 作业 `353198` 比较 OLMoE B16 同一张 A800 上的两槽与三槽 ring。两臂均保留 216 CTA、相同的 Gu/Dn tile、prompt prefill 和 seed-0 checkpoint。完整 1024-token 输出与冻结参考逐位一致。三次配对分别改善 0.1663、0.1668、0.1674 ms/token。三槽的中位 TPOT 降低 1.812%，进入独立校准与普通 solver 筛选。

| OLMoE B16 几何 | T1 (s) | T1024 (s) | TPOT (ms/token) |
|---|---:|---:|---:|
| 两槽正式对照 | 0.543603 | 9.971584 | 9.216013 |
| 三槽共享 ring | 0.532354 | 9.789494 | 9.049012 |

**Verified** · 该表使用同卡筛选回执。正式 TPOT 仍取独立计时数据。校准作业为 `353235`。

**Verified** · Llama-3B B16 作业 `353244` 通过完整 teacher forcing、step/loop 和冻结 token 检查。作业 `353245` 的同卡完整请求把 TPOT 从 6.193537 降至 6.048962 ms/token，三次配对均改善，冻结 checkpoint 和输出一致。T1 分别为 0.179948 s 和 0.178665 s，T1024 分别为 6.515936 s 和 6.366753 s。该筛选收益为 2.334%。正式结果由校准后普通 solver 和独立计时确认。

**Verified** · 登录节点核对了真实 packing 的三种末 tile。Llama-3B TN96 生成 171 个 gate task，32 个补齐权重行均为零，原行宽保留。CPU 成本回归核对了 171 个 task 及对应的 100,859,904 B 物理权重流量。SM80 部分 tile 宏禁用时，旧 GEMM、epilogue 与 program 的源内容逐字节一致。启用时，三个 header 与已筛选的部分 tile 路径一致。原价格仍通过完整 tile 的旧源哈希核验，部分 tile 使用独立的 exact service 键。

**Verified** · Qwen30 B1 作业 `353202_0` 的四槽共享方案和三槽对照输出一致。TPOT 从 6.960826 降至 6.941983 ms/token，差异 0.271%，保留原方案。B8 原作业 `353202_1` 在主机 cgroup 达到 62,800 MiB 上限后被停止。重试 `353250_1` 使用 6 个 CPU 和 94,200 MiB，保留同一单 GPU、权重和两份二进制。B4 在进入运行前将主机容量调到同一档。 [Slurm CPU 与内存资源参数](https://slurm.schedmd.com/sbatch.html)

**Verified** · 首次 exact TN96 校准 `353261` 的插桩扰动 6.45%，该项校准作废。后续 `353278` 只对每 16 个 CTA 中的一个采集角色时序。两臂的 uninstrumented 和 traced span 均为 65.536 µs，输出逐位一致，裁掉补齐列后的有效数据也逐位一致。实测价格进入普通 solver。作业 `353300` 的 CTG replay 将 TN96 排在 TN128 之前（5.646557 对 5.654340 ms），同卡完整请求为 6.017032 对 6.173263 ms/token，已选中 TN96。

**Verified** · TN160 三槽先通过 Llama-8B、Mistral B16 的完整正确性检查，作业 `353273_0` 和 `353273_1`。首次性能提交在读取正确性回执的调度器映射时退出，复用已通过的正确性回执，改为在主机侧冻结映射。最终 `353297_0` 的 Llama-8B 为 11.895914 → 11.820239 ms/token，收益 0.636%，进入 exact 校准。`353297_1` 的 Mistral 为 11.448441 → 11.408474 ms/token，收益 0.349%，保留当前正式坐标。两组冻结输出一致。

**Verified** · Qwen30 B2 的四槽共享 ring 由 `353202_2` 验证，T1 为 0.409020 → 0.408982 s，T1024 为 10.959248 → 10.710628 s，TPOT 为 10.313029 → 10.070036 ms/token。三次配对均改善，输出一致。初次每 32 tile 采样的局部 MMA 服务低于全局时钟分辨率，回执未进入价格表。`353298` 改为每 8 tile 采样后，Gu/Dn 的每 tile MMA 价格为 384.0/409.6 ns，随后提交普通 solver。另一个 Qwen30 B1 的 216 CTA 尝试在登录节点被工作区核验拦截。D128/Q8、两个 64-key slot 的 compact arena 仍需 88,064 B，超过该目标的双驻留预算，没有消耗 GPU allocation。

**Verified** · OLMoE B16 三槽在 `353235` 中采集了 216 CTA、81,920 B 的 exact service，所有 108 个 SM 有两项活动同时驻留，输出逐位一致。`353254` 的普通 solver 同卡筛选中，两槽和三槽为 9.216457 与 9.045142 ms/token，收益 1.859%，三次配对均改善且输出一致。正式二进制已通过作业 `353271` 的五次独立正确性验收。

## 正式结果

**Verified** · OLMoE B16 通过了五次新正确性进程，独立计时作业 353323 随后给出 T1 0.531104 s、T1024 9.763991 s 和 TPOT 9.025305 ms/token，较正式旧值 9.185485 ms/token 降低 1.744%。新 decode SHA-256 为 2cb07632d481ed86a2122b7778224c65d0d44284b226d2960c22734bb7faad5a。冻结 baseline 逐字节一致。

**Verified** · 40 格目前为 31 胜、3 平、6 负，几何平均加速比 1.099106，带宽效率的几何平均为 0.692953。其余 39 格正式产物和 TPOT 沿用上一份矩阵。 [40 格正式 TPOT](A800_llm.csv) · [绝对执行时间](A800_execution_times.md) · [带宽效率](A800_SM80_R3_residual_efficiency.csv)
