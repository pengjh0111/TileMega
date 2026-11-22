# TileMega A800 SM80 执行审计和后续复测

**Verified** · 以 `prompt.md` 的 R0–R8 为检查范围。正式矩阵使用 8 个模型、5 个 batch、BF16 TP1、64 个输入 token 和 1024 个输出 token。基线冻结为 180 个有效格和 20 个 N/A。下表区分已采纳的生产改动、完成对照的实验和仍待验证的工作。此前发布的完整结果和各格执行时间在 [正式矩阵](A800_SM80_R3.md) 和 [执行时间](A800_execution_times.md)。

| 轮次 | 已完成 | 后续 |
|---|---|---|
| R0 | 核对 Llama-1B B16、Llama-3B B16 和 OLMoE B8。小 Llama 的差异在 0.50% 阈值内，大 Llama 和 OLMoE 由后续轮次正式晋级。 | 当前对照已结束。 |
| R1 | 新增 321 项 MMA 校准，键数从 92 增至 413。全部 40 格已离线审计，多驻留价格按实际几何匹配。 | 新几何仍需对应的服务价格。 |
| R2 | Llama-3B B16 采纳 TN256 和 down 三槽。Llama-8B、Mistral B16 采纳 216 CTA 的 TN224。原 B8 TN256 对照更慢。 | B16 gate/up 复制分组已补测，保留原配。 |
| R3 | 证实 128 个残差任务在 216 CTA 上只有 20 个 SM 同时执行两项工作。更窄 TN16 虽增加覆盖，但实测更慢。OLMoE B8/B16 采用 216 CTA。 | Llama-8B、Mistral B1/B2/B4/B8 和 Llama-3B B16 的双驻留方案已正式采纳。 |
| R4 | 完成 MoE padding、attention mask 和 dense live-row 对照。正式 body 延用已有选择。 | 五个 residual 几何的精确 epilogue 价格已进入 solver。 |
| R5 | activation issue、wave tail、register stages 和 fresh KV 进入普通价格路径。经过同卡 trace 补价，Qwen B16 的 segments/copy8 排序和实测方向一致，两者仍在平局范围内。 | 新增价格已通过正式坐标重放。DRAM 硬件计数器权限不足，实际共享带宽仍未直接测得。 |
| R6 | 有界 L2 prefetch 在两组几何的端到端对照均在平局阈值内。 | 实验选项关闭。 |
| R7 | Qwen30 B16 采纳共享 arena 的四槽融合专家。行基址和 packed token route 的整请求差异在阈值内。 | 四槽为当前正式配置。 |
| R8 | 生产 ABI 为 128 consumer 和 32 producer，保留当前同步合同。 | 此轮未启动角色 ABI 改造。更换参与范围需要重新校准和验证。 |

## 当前效率与缺口

**Verified** · R3 正式发布时几何平均为 1.078081，按 0.50% 阈值为 23 胜、3 平、14 负。初始值为 1.070658。B16 的 Llama-3B、Llama-8B 和 Mistral 依次达到 0.7173、0.8079、0.7982，目标均为 0.83。以相同逻辑流量和 1685.62 GB/s 的带宽估算，三格分别还需约 13.6%、2.7% 和 3.8% 的 TPOT 降幅。

**Verified** · Qwen30 B16 的 L1-only prefill 仍使 T1 达到 2686.0 ms，SGLang 为 67.2 ms。该项影响 TTFT 和完整请求延迟。本轮验收指标 TPOT 从两种请求的执行时间差计算。

## R5 价格及同卡证据

**Verified plan** · 对相同的生产坐标保留原价格，用 trace 区分各自的 router 启动、专家和 combine 尾部。完整 token 和 checkpoint 必须相同，仪器扰动上限为 5%。B8 的独立筛选采用 0.50% 阈值、每个配置一个进程和三次进程内计时。推送前以正式选型、正确性及独立时延结果为准。

**Verified** · 所有 40 个正式坐标通过实际导入 CTG 重新评分。预测相对正式 TPOT 的绝对误差中位数为 6.436%。该统计使用冻结的完整矩阵。MMA 留一验证是单独的 371 项测试，误差中位数为 14.902%。

| Qwen30 B16 对照 | route startup (µs) | combine 尾部 (µs) | fused stage (µs) | trace 扰动 | 样本 |
|---|---:|---:|---:|---:|---:|
| copy8 | 16.384 | 3.072 | 477.184 | 2.408% | 768 |
| 默认 segments | 16.384 | 3.072 | 478.208 | 2.424% | 768 |

**Verified** · 作业 `352603_0` 在同一张 A800 上采集了两组 trace，复用了完全相同的 seed-0 checkpoint。两组都和各自的未插桩进程产生相同的完整输出，两个未插桩进程的输出也一致。原 registry 只有 copy8 使用实测 startup 和 tail。补入 segments 的实测项后，segments/copy8 的 CTG 预测由 29.289856/29.565633 ms 变为约 29.5707/29.5656 ms。对应同卡正式协议筛选值为 28.062879/27.978402 ms/token，两者在 0.50% 范围内。已采纳四槽坐标仍排在两者前面。

**Verified** · 作业 `352577_0` 能初始化并启用 CUPTI PM Sampling。作业 `352601_0` 使用同一 CUDA 环境，调用 `cuptiPmSamplingGetCounterAvailability` 返回 `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`。目前 CTG 对共享区间继续使用逐任务带宽模型。

**Verified** · 更细的 SM 时钟探针区分 residual side 等待和计算。作业 `352585_0` 的部分短 O 形状超过 5% 扰动。去掉 inner-loop 计时后，作业 `352629_0` 的五个确切几何全部通过 5% 上限，plain 和 traced 输出逐位一致。宏关闭时，比对编译器实际看到的 tokens，验证了原默认、copy-group 和 live-row 三种编译方式。

| Dense B16 residual 几何 | 完整 epilogue (ns/task) | residual ready 后 (ns/task) | 扰动 |
|---|---:|---:|---:|
| Llama-8B O 216 CTA 3 slots | 182.882 | 126.480 | 0.000% |
| Llama-8B down 216 CTA 3 slots | 179.656 | 124.234 | 1.111% |
| Llama-3B O 108 CTA 2 slots | 184.382 | 129.174 | 0.000% |
| Llama-3B O 108 CTA 3 slots | 179.404 | 123.366 | 0.000% |
| Llama-3B down 108 CTA 3 slots | 195.454 | 126.869 | 2.500% |

**Verified** · 作业 `352576_0` 在现有几何补测了更深 O/down ring。两组主导 residual 的活动 CTA 覆盖不变。Llama-8B 的三/四/五槽 down 分别为 92.160/94.208/91.136 µs，五槽比三槽快 1.024 µs，接近探针时间分辨率。沿用正式配置。

**Verified** · 使用新增校准和当前 40 个正式坐标重新导入 CTG 后，预测相对正式 TPOT 的绝对误差中位数为 5.398%。

## Dense 216 CTA 同卡筛选

**Verified** · 每对对照在同一个 allocation 内按原协议执行。表中的固定坐标用于定位收益。两个 B8 对照在全部三次计时中均改进，完整输出与原产物一致。两个 B4 对照从零起始索引 12 和 18 开始出现 token 差异。随后保留原 O/down 的 K 分组，重新检查输出。

| 模型与 batch | Slurm 作业 | Control T1 / T1024 (s) | Trial T1 / T1024 (s) | TPOT control → trial (ms/token) | 完整输出 |
|---|---|---:|---:|---:|---|
| Llama-8B B4 | `352620_1` | 0.122937 / 12.326202 | 0.123023 / 10.982421 | 11.928900 → 10.615247 | 有差异 |
| Llama-8B B8 | `352620_0` | 0.240269 / 12.595349 | 0.240378 / 11.212983 | 12.077302 → 10.725910 | 一致 |
| Mistral B4 | `352620_3` | 0.122550 / 11.893658 | 0.122610 / 10.561406 | 11.506460 → 10.204102 | 有差异 |
| Mistral B8 | `352620_2` | 0.239769 / 12.170469 | 0.239618 / 10.752452 | 11.662464 → 10.276476 | 一致 |

**Inferred** · 216 CTA 在 B8 的收益很可能与使用较小的 operand staging 和 compact attention arena 有关。现有 compact 屏障探针使用 128 个 consumer、79,872 B 的 arena，并在每个 SM 同时执行两个 CTA。模型的 batch 决定 attention task 的数量，单个 CTA 的屏障路径按原 D128/Q4/no-norm 几何计价。

**Verified** · 保留原 O/down 分组后，作业 `352669` 的两组 B4 对照与冻结输出完全一致。Llama-8B 的 TPOT 为 11.953857 → 10.887819 ms/token，Mistral 为 11.504108 → 10.453833 ms/token，三个计时对均改进。由此将变更缩小到保留残差归约次序的双驻留方案。

**Verified** · Llama-3B B16 在作业 `352673_0` 对照复用相同的输出与 checkpoint。保持原 K 分组并使用 compact D128/Q3 后，筛选 TPOT 为 6.210975 → 6.168481 ms/token。作业 `352688_0` 补测了 D128/Q3 的 128-consumer 屏障，在 79,872 B 的 arena 中令全部 108 个 SM 同时运行两个 CTA，1,944 次样本的中位数为 25.0 ns。Q4 校准保留原回执，Q3 使用独立几何键。

**Verified** · 作业 `352671_0` 比较 B16 gate/up 的 0、4、8 复制分组。大 dense 的 216 CTA/TN224 使用分组后，未插桩 span 由 148.480 µs 增至 151.552/154.624 µs。Llama-3B TN256 的三者为 72.704/77.824/71.680 µs，8 组差异为一档时钟分辨率。未据此选择新分组。

**Verified** · 旧校准目标使用随源代码提供的预处理等价证明加载。用新代码评估原三套 Llama-8B 和 Qwen 坐标，原 additive 与 replay 价格保持逐值一致。新目标把正式 residual geometry 的高分辨率价格与原 mainloop 价格分开，保留旧的原始回执。Linux 登录节点构建通过 native prices、native flow 和 native pipeline 单元测试。

**Verified** · B1/B2 的第一组双驻留对照把原 8/4 个 attention chunks 改成 1，完整输出出现差异。作业 `352772` 保留原 attention chunks 和所有 GEMM 坐标，仅改变 CTA 驻留数量和共享 arena、把 attention ring 改为两槽。四个对照均恢复完整 token 一致性，所有计时对均改进。

| 模型与 batch | Slurm 作业 | 同卡 TPOT control → trial (ms/token) |
|---|---|---:|
| Llama-8B B1 | `352772_1` | 11.282512 → 10.494569 |
| Llama-8B B2 | `352772_0` | 11.350984 → 10.502735 |
| Mistral B1 | `352772_3` | 11.036291 → 10.114905 |
| Mistral B2 | `352772_2` | 10.909548 → 10.101498 |

**Verified** · 普通 solver 在 Llama-8B B1 的 216 CTA 坐标缺三个 exact service 项。作业 `352845_0` 补齐 O、down 和 gate/up 的 live-row/copy8 几何。三个探针输出逐位一致，最大扰动 3.704%。已有 MMA 几何与正确定价继续复用。

## 正式晋级

**Verified** · 先做普通 CTG 搜索和同卡测量，再重建相同指令的正式产物。每个候选通过五个独立正确性进程，其中一个做完整 teacher-forced 检查。最后的独立计时仍是一个进程，包含进程内预热和三次计时。

| 模型与 batch | 正确性 | 独立计时 | T1 (s) | T1024 (s) | 正式 TPOT 旧 → 新 (ms/token) |
|---|---|---|---:|---:|---:|
| Llama-3.1-8B B1 | `352906_0` 5/5 | `352941_0` | 0.041409 | 10.797720 | 11.241003 → 10.514478 |
| Llama-3.1-8B B2 | `352913_0` 5/5 | `352947_0` | 0.071683 | 10.806937 | 11.373559 → 10.493895 |
| Llama-3.1-8B B4 | `352766_0` 5/5 | `352849_0` | 0.123312 | 11.210784 | 11.955695 → 10.838193 |
| Llama-3.1-8B B8 | `352734_0` 5/5 | `352799_0` | 0.240106 | 11.199055 | 12.028446 → 10.712561 |
| Llama-3.2-3B B16 | `352843_0` 5/5 | `352874_0` | 0.180478 | 6.481073 | 6.188437 → 6.158939 |
| Mistral-7B-v0.3 B1 | `352928_0` 5/5 | `352960_0` | 0.040971 | 10.373222 | 11.025912 → 10.099952 |
| Mistral-7B-v0.3 B2 | `352917_0` 5/5 | `352954_0` | 0.071125 | 10.435792 | 10.935394 → 10.131639 |
| Mistral-7B-v0.3 B4 | `352767_0` 5/5 | `352844_0` | 0.122462 | 10.817838 | 11.496108 → 10.454913 |
| Mistral-7B-v0.3 B8 | `352735_0` 5/5 | `352787_0` | 0.239616 | 10.740292 | 11.688645 → 10.264590 |

**Verified** · 本轮正式采纳 9 个坐标。全部 40 格保留相同模型配置、共享权重与 prompt，冻结基线逐字节一致。

| 40 格指标 | R3 发布时 | 本轮发布后 |
|---|---:|---:|
| 最快冻结 baseline TPOT / TileMega TPOT 几何平均 | 1.078081 | 1.098623 |
| TileMega HBM roofline 效率几何平均 | 0.679698 | 0.692649 |
| 0.50% 阈值下的胜 / 平 / 负 | 23 / 3 / 14 | 31 / 3 / 6 |

**Verified** · B16 效率沿用 1685.62 GB/s HBM 基准。Qwen 和 OLMoE 的逻辑流量沿用匹配的路由样本。

| 重点 dense B16 | 正式 TPOT (ms/token) | Roofline 效率 | 目标 |
|---|---:|---:|---:|
| Llama-3.2-3B | 6.158939 | 0.7207 | 0.8300 |
| Llama-3.1-8B | 11.908982 | 0.8079 | 0.8300 |
| Mistral-7B-v0.3 | 11.472326 | 0.7982 | 0.8300 |

**Verified** · 仍落后于最快冻结 baseline 的格子分布为 olmoe_1b_7b 1 格、qwen3_30b_a3b 5 格。整矩阵几何平均为 1.098623，目标区间为 1.12–1.18。

**Inferred** · A800 最快 baseline 的 roofline 效率约 0.6305，H800 为 0.5309。H800 上的 TMA 和 cluster 路径仍带来额外硬件收益。本轮恢复了更多 dense 格的余量，三个重点 B16 距 0.83 仍有差距。 [H800 正式结果](SM90R6_hw_bound.csv)

当前完整数据见 [40 格正式 TPOT](A800_llm.csv) · [绝对执行时间](A800_execution_times.md) · [40 格 roofline 效率](A800_SM80_R3_followup_efficiency.csv) · [Slurm 作业台账](A800_SM80_R3_followup_jobs.csv)。

原始回执、二进制和 Slurm 控制器位于 A800 下的 `runs/a800/optimization/r4-closeout/`。作业 `352629_0` 的独立工作树原始记录在 `TileMega-a800-r4-epilogue-only/runs/a800/calibration/r2-supplement-service/352629/`。
