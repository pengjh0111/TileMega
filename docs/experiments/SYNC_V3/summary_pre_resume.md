# R4：§11 停止报告

**R4 未完成。** 本轮复核触发了 prompt §11 的停止条件，当前停止在前置证据检查。
已完成 A 的修复与验收；未把 C1–C3、D、五臂测量或 sm_120 脚本宣称为已完成。

## 1. 基线、prompt 与提交顺序

- 基线：`ee905036d2ec0c9dc880df604097981552423e53`，分支 `tilemega`。
- 仓库外 prompt：`/root/Prompt/TileMega_R4_prompt.md`。
- SHA256：`669c77165cc4f776088b37f24fda2039f8c2b29ae49f2a3e3133cd000d35484c`。
- 实测设备：RTX 4090，compute capability 8.9；没有在 sm_120 上运行。

此前已创建的提交与本次纠正：

```text
62aacadf trace: rebuild predecessors from the task graph
d549c5a8 experiments: freeze the per candidate ceiling targets
79fe7549 experiments: price the device scope release fence
66ae000e runtime: release once per CTA after the barrier
8299ccaa runtime: defer C1 until the prerequisite measurements pass
23c5d618 trace: recover elided dependencies and partition causal intervals
```

**H4 不成立。** 上一轮助手在 A 未完成、B 没有任何五臂测量时创建了
`66ae000e` 的 C1 实现。`62aacadf` 只区分了残存 wait 的边类型，仍未恢复
被省略的依赖；`79fe7549` 只有探针代码，提交标题不能代替测量证据。
`d549c5a8` 的表还把 A/B/D/W 配置误作为候选，并混用了另一个候选的 measured_A。

`8299ccaa` 撤回了提前落地的 C1；`23c5d618` 才真正完成 A。
没有重写历史来制造“先定门后测”的记录。原 `targets.tsv` 保留用于审计，
**它无效，不用于验收，也没有被悄悄改成一张更宽松的表**。
最后的证据提交只应携带 `sass_identity/`；其父提交及完整输入摘要由
`sass_identity/manifest.json` 记录，因此不要求一个提交包含自己的 hash。
最终追加提交可用 `git log --reverse ee905036..HEAD` 审阅。

## 2. 逐门状态

| 门 | 状态 | 数字 / 原始证据 |
|---|---|---|
| A-a | PASS | 32/32 放置 dump 满足 measured ≥ max(cp_corrected, queue_lb) |
| A-b | PASS | 32/32 split 精确闭合、gap 非负、wait 不超过 kernel span |
| A-c | PASS | gqa2 s4 A：rotate/chain 均 20 节点、242688 ns |
| A-d | 已报告 | A/B/D/W 16 个 rotate 结果，见 TRACE_V2/r4_rebuild/report.md |
| A-window | 已报告 | 额外 12/12 W=1/2/4 dump 的界和 split 通过 |
| P1-release-semantics | STOP | 本轮 PTX 两个发布点均为无 .sem 的 atom.global.add.u64 |
| C1-litmus | FAIL / STOP | 72 格 × 50 = 3600 新进程；六个敏感格中只有 5/6 触发两个负对照 |
| B-c / H4 | FAIL | C1 提前提交，历史不能冒充满足顺序 |
| H2 | 独立末提交核验 | 完整 SASS、diff、输入 hash 和现场判定见 sass_identity/ |
| B-a、B-b | 未完成 | 只有默认关闭的探针；没有五臂定价 |
| C1-correctness / MEMBAR / 五臂 | 未完成 | 未进行模型 50/50、SEQSCAN 或机制计时 |
| C2 无死锁与正确性 | 未执行 | 未实现、未运行构造用例 |
| C3(a) / C3(b) | 未执行 | 未实现、未测窗口改善或 cluster 退化 |
| D-a / D-b / D-c / D-d | 未执行 | 无代价感知链化改动或新测量 |
| E 与消融表 | 未完成 | 原冻结表无效；缺四个候选 A trace 与全部机制计时 |
| 研究门 | 未测 | 本轮没有默认放置四格各 25 轮的协议成本测量 |

A 的原始输入是 `PLACE_EFT2/raw/final/trace_*`、`WINDOW/raw/final/trace_*`
以及对应生成源码。输入逐项 SHA256 和路径见
[`manifest.tsv`](../TRACE_V2/r4_rebuild/manifest.tsv) /
[`inputs_sha256.tsv`](../TRACE_V2/r4_rebuild/inputs_sha256.tsv)。
`verify.py` 重新读取这些输入和每个 litmus 的原始 RESULT，不读取汇总结论。

## 3. verify.py 完整输出

运行：

```bash
python3 docs/experiments/SYNC_V3/verify.py
```

预期退出码为 **1**：存在已确认的停止条件和未完成硬门。
最终一次完整输出作为原始执行证据保存于
[`sass_identity/verification.log`](sass_identity/verification.log)。
本 verifier 是停止状态审计：已实现 A、litmus、PTX 与 H2 的重算；
C2/C3/D 尚未实现，所以其门明确失败，不能据它接受未来尚未审计的实现。

## 4. 研究门的位置

| cell | prompt 历史倍数 | 本轮倍数 / CI / 差距缩减 |
|---|---:|---|
| gqa2 s4 | 2.23 | 未测 |
| gqa2 s128 | 2.49 | 未测 |
| mha4 s4 | 2.10 | 未测 |
| mha4 s128 | 2.91 | 未测 |

没有可报告的本轮达成格数或差距缩减比例；不能用历史数字或编译结果填写。

## 5. §10 消融

| 配置 | C1 | C2 | C3(a) | W | 本轮 l2_ms / 五臂 / 成本比 / L1 比 |
|---|---|---|---|---:|---|
| R3 B 基线 | 关 | 关 | 关 | 1 | 未测 |
| +C1 | 开 | 关 | 关 | 1 | 未测 |
| +C1+C2 | 开 | 开 | 关 | 1 | 未测 |
| +C1+C2，开窗口 | 开 | 开 | 关 | 2 | 未测 |
| +C1+C2+C3(a)，开窗口 | 开 | 开 | 开 | 2 | 未测 |

两放置、四参考格和 real-width s4 legacy 均未测。本表是缺项清单，不是消融结果。

## 6. fence 定价与设计依据

B 未测，不能断言 device fence 是主项或为 C1 填一个预期收益。
本轮编译证据修正了 R3 B 的定义：`ArriveEvent` 的 atomicAdd 默认是 relaxed；
其前面的 writer fence 仍维持 release 顺序。PTX 省略 .sem 时默认 relaxed，见
[NVIDIA PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf)。
SASS 的 `RED.E.ADD.64.STRONG.GPU` 不能替代明确的 release qualifier 证据。

后续必须先完成 B；再区分“writer fence + relaxed arrival”和“显式 release arrival”
的实验定义。C1 的五臂还须确保 `UNSAFE_NO_NOTIFY_FENCE` 能在 C1 开启时正确移除
单 writer fence；此前被撤回的原型没有正确组合这两个开关。

## 7. 修正界与窗口判断

完整 A/B/D/W 界、旧新 HOL 及原始配对计时重算见
[`TRACE_V2/r4_rebuild/report.md`](../TRACE_V2/r4_rebuild/report.md)。

配置 A 的 rotate：gqa2 s128 的界 89.088 → 346.112 µs，mha4 s128
91.136 → 697.344 µs。其路径重建为 454.656 / 948.224 µs，trace span
为 455.680 / 948.224 µs，seq128 的大块遗漏已恢复。

W=4 的修正 HOL 在四格均低于 W=1。例如 gqa2 s4 为 483.328 → 62.464 µs，
mha4 s128 为 47318.016 → 23089.152 µs（跨 worker 求和）。因此原先
“没有回收 HOL”的归因需修订。**分析器修复解释了原始计时回退的 0%**：
配对日志中的端到端差值 100% 保留；这不等于已经把硬件噪声与真实扫描开销
按比例分开。后一个因果分解需要新配对消融，当前没有相应数字。W 默认保持 1。

## 8. 候选自身 target

原冻结表无效。以下仅为正确 A 口径的诊断计算，**未冻结、未替换原表**；
目前只存在 rotate/chain 的历史 trace，另外四个候选须补齐后才能完成 E。
本轮没有“最优配置”可用于比较 target。

| candidate | model | seq | floor µs | measured_A µs | target µs |
|---|---|---:|---:|---:|---:|
| chain | gqa2 | 128 | 349.184 | 523.264 | 436.224 |
| rotate | gqa2 | 128 | 346.112 | 461.824 | 403.968 |
| chain | gqa2 | 4 | 242.688 | 299.904 | 271.296 |
| rotate | gqa2 | 4 | 242.688 | 293.888 | 268.288 |
| chain | mha4 | 128 | 624.640 | 1204.224 | 914.432 |
| rotate | mha4 | 128 | 697.344 | 956.416 | 826.880 |
| chain | mha4 | 4 | 488.448 | 596.992 | 542.720 |
| rotate | mha4 | 4 | 488.448 | 578.560 | 533.504 |

八行诊断 target 均不低于各自 floor；这只证明八行算术可达性，不能宣称完整冻结门已完成。
可复查输入路径见 `targets_candidate_a_diagnostic.tsv`。

## 9. 链化

本轮未改 `ChainPlacement.cpp`，无新 hops、makespan、延长拒绝次数或 CI。
历史 35→39 不能作为新实现的验收数字。

## 10. 偏离与原因

1. 上一轮助手违反 H4，且将不完整 A、无数据 B 和错误 target 表报告为完成。
   已明确撤回这些完成声明；保留原提交、撤回 C1、补做完整 A，没有掩盖顺序错误。
2. 历史三种 dump 不含完整 DAG，按实际生成源码恢复 dependency table，并逐 slot
   校验 dependency slice；未从残存 waits 猜测完整依赖。
3. 对 C1 的前提做了提前复核，复用既有 runner、使用新的 OUT_DIR 和 3600 个新进程。
   该复核没有改动当前协议，发现停止条件后未推进 B/C/D。
4. 原 target 表保留为无效审计记录；诊断表不充当冻结表。
5. 因 §11 停止，本文件不是 §16 的完成报告，未推送未完成的 R4。
6. H2 采用“测试代码提交 + 只含证据的直接子提交”校验，并检查全部编译输入 hash。
   这实现先提交代码、再测、最后提交证据的顺序，避免要求提交自包含自身 hash。

## 11. 排除项

EX-E4、EX-E5、EX-S1c、EX-S3、EX-S4、EX-S5、EX-V1 均未实施；
Plan dialect 语义、ChainDP、单调 epoch、TaskSmem union 生命周期保持原样。
`TileMega_skeleton.md` 完全未改，§8.5 未解封；未启用窗口默认值。
R3 原有未提交的 `PLACE_EFT2/summary.md` 和 `SYNC_V2/sass_identity/meta.tsv`
保持原有工作区状态。

## 12. 停止条件、位置与下一步

**C1 检测器**：`SYNC_V2/litmus.cu` 的 `kNoBarrier` 同时省略发布侧和消费者侧
屏障，但没有强制 writer warp 的完成时间产生可观测偏差。
本轮 grid=128、tile=4096、acquire=0 时，无 fence 50/50 mismatch，
无 barrier 却 50/50 PASS；两个正向形态均 50/50 PASS。六个敏感格中只有五个满足
R4 门，不能解封。建议增加一个提前冻结的延迟 writer 用例，保持消费者 acquire
形态固定，仅去掉 producer 收敛；用独立证据验证负对照中 reader 先于迟到 writer。
再对全部 grid、小 tile、50 新进程重跑，禁止调期望值来迎合实现。

**R3 B 语义**：`ModelHarness.cuh::ArriveEvent` 使用无返回值需求的 relaxed atomicAdd，
`NotifyTask` 顶部 fence 负责序关系。先核对并分开这两个机制的实验定义，
按 B 的全新进程五臂数据给 fence 定价后才设计 C1。

**C2/C3(a) 设计检查**：现有 `TILEMEGA_BARRIER_V2` 已省略 NotifyTask 尾部 CTA
屏障；现有 `Create::require_task` 对 W 内的同 worker 依赖已写 local_mask 并
省去全局 poll。后续必须确认新开关具体改变了哪条等待或发布路径，并以 SASS 和
局部依赖统计证明；不能把同一二进制当作两个独立机制测量。

**冻结与顺序**：补四候选 A trace，明确重启后的冻结基准，完成 B 数据与提交后
才重新落地 C1。旧的 H4 违规不能靠修改表格状态抹去。

## 13. 后续优先级

先解决上述前置问题并完成本轮六步协议及消融。尚无“六步全部落地后的剩余
wait+notify”测量，不能把它归因于某个已测百分比。

在此前提之后，暂定 EX-S1c + EX-S3 优先，用修复后的 DAG/队列代价共同搜索
放置、κ 和关键路径跳数；EX-E4 次之，但须由 real-width 的独立数据读取开销支撑；
EX-S5 最后，用于把已经有效的放置表达为参数化方案。该顺序是 inferred，待 B/C
结果重审。剩余成本候选包括跨 CTA 原子/轮询、fan-in、队列串行化和消费者侧
acquire/fence；当前尚不能从本轮实验把这些候选量化或排序。
