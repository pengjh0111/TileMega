# T0：先修测量，再比较 residency / spill

✅ 4090 两状态正确性 400/400；四臂各 25 轮已完成。基线 e305a9f。
⚠️ 5090 只提供人工脚本，未运行。本结果不支持“1 → 2 CTA/SM”的解释。

## 测量约定（实验前确定）

`Benchmark.cuh` 为 ModelHarness 与 GeneratedLlamaRuntime 共用入口。
`TILEMEGA_WARMUP` 默认 5，`TILEMEGA_REPEAT` 默认 11；warmup 不创建计时事件，
每次计时仅覆盖一次前向的 kernel launch(s)，报告样本中位数。
环境值非法时拒绝，不悄悄修正。编译宏 `TILEMEGA_COLD_START_TIMING=1` 固定为
warmup=0、repeat=1，保留历史单次模式。此前本项目的 L2 时间均属于单次模式，
本轮不能用它们替代稳态时间或跨模式直接作收益比例。

每个样本在计时外恢复相同输入和事件初态，是独立前向，不是自回归生成。
这会影响数据缓存状态，必须将本测量称为“重置输入后的 warmup/repeat 前向”，
不能称为纯热缓存或 serving 吞吐。既有跨迭代检查仍调用单次 LaunchL2，并保留
从 iteration=0 到 1、2… 的单调事件语义；benchmark 不包裹该检查。

`TILEMEGA_MIN_BLOCKS_PER_SM` 默认 1，两个 runtime 的 launch_bounds 共用同一
定义；=2 只改变编译器寄存器/溢出权衡，不删除 TaskSmem 的任何字段或任务。
所有旧事件路径优化宏保持关闭。

`E2E_RESOURCE` 的 smem/occupancy_smem 是 L2 查询用的动态 shared 字节，
task_smem 是 sizeof(TaskSmem)，static_smem 单列。reg 来自 cudaFuncGetAttributes，
脚本要求与该二进制 L2 的 ptxas 数值一致；L1/L0.5 寄存器和各 kernel 溢出字节
同时保留。ctas_per_sm/grid 是公共 launch 的驻留约束；l1_ctas/l2_ctas 单独列出，
防止把另一个 kernel 造成的 grid 限制错归因为 F-40 失效。

F-40 按实际 block/warp 与 TargetSpec 预算计算，各 warp 的寄存器分配按原规则
向上对齐 256。不能把 4090 的寄存器阈值 128 套到 128-thread kernel。
`min_blocks=2 && F40=2 && l2_ctas=1` 时脚本立即停止，保留日志待解释。

## 实际资源与前提修正

两个模型、两个 seq、两种 launch_bounds 的 full arm 均为 block=128、
L2 registers=212、TaskSmem=occupancy_smem=24576 B、static_smem=0、
ctas_per_sm=2、grid=256。L1 registers=208，L0.5 registers=206。
完整编译器日志在 `raw/ptxas/`；各 kernel 的 spill stores/loads 均为 0 B。
neither arm 的 L2 registers=210，仍是 2 CTA/SM；不能省略逐臂资源记录。

TargetSpec 预算为 registers=65536、smem=102400 B、threads=1536、warp=32。
F-40 给出 register/smem/thread 三项分别为 2/4/12，绑定于寄存器，与查询一致。
用户给出的 block=256、REG=168、SHM=49536 B 不是此次 BF16 生成实例。
`MIN_BLOCKS_PER_SM=2` 在这里没有制造额外驻留，也没有制造 spill；不得改变线程数
或删减 TaskSmem 来满足该前提。两状态 SASS 并不相同，资源相同不等于机器码相同。

对 T3：当前实例的两 CTA shared 预算为 51200 B/CTA，距 24576 B 尚有 26624 B，
不是 1152 B。融合仍须按合并后的实际 shared/static/register 需求重新验证 residency。

## 时间与四臂（毫秒）

下表为 25 个全新进程各自 warmup=5、repeat=11 中位数的中位数。
原始逐轮资源、时间见 `raw/attrib.tsv`，配对 bootstrap 95% CI 和 Wilcoxon 见
`report/paired.tsv`。跨轮时钟波动明显，不能用未配对的中位数差替代配对统计。

| 模型/seq | min blocks | L0.5 | L1 | L2 | notify | wait | barrier | loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gqa2/4 | 1 | .376832 | .375808 | .411648 | .054496 | .029696 | .041984 | −.006144 |
| gqa2/4 | 2 | .376800 | .374880 | .411808 | .051200 | .032768 | .041056 | −.007040 |
| gqa2/128 | 1 | .524288 | .523264 | .567104 | .064416 | .026624 | .038272 | −.011264 |
| gqa2/128 | 2 | .561152 | .564416 | .606240 | .066688 | .026464 | .039136 | −.012288 |
| mha4/4 | 1 | .793504 | .804032 | .869376 | .112320 | .044032 | .075968 | −.013312 |
| mha4/4 | 2 | .793568 | .805888 | .870400 | .109632 | .048928 | .077792 | −.016192 |
| mha4/128 | 1 | 1.108992 | 1.123520 | 1.217280 | .160768 | .044224 | .075904 | −.025792 |
| mha4/128 | 2 | 1.090560 | 1.113088 | 1.204064 | .160576 | .044032 | .083232 | −.026624 |

四臂闭合误差逐轮为 0。L2 的配对差（=2 减 =1）分别为：

| 模型/seq | 中位差 ms | 配对 95% CI | Wilcoxon p |
|---|---:|---|---:|
| gqa2/4 | .000160 | [0, .001120] | .07622 |
| gqa2/128 | .000192 | [−.000064, .000864] | .69644 |
| mha4/4 | .001024 | [0, .002048] | .03956 |
| mha4/128 | −.003072 | [−.028672, .014336] | .46752 |

没有可归因于增加 occupancy 的收益；默认保留 1，开关保留。mha4/4 的小幅变差如实
记录，不归因于 spill（日志为零）。正确性为每模型 × seq{4,128} × 两状态各 50/50，
总计 400/400，见 `raw/correctness.tsv`。这些是新测量，未复用历史冷启动时间。
