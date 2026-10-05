# 自动接续与实现边界（最终带限制封存）

verified（CPU）：不依赖实测反馈的 E2c–E6 脚本、测试定义和自动收集器已实现。
本节开工时只描述自动化边界；最终 GPU 验收与未通过项见 summary.md，不把脚本完成等同于全协议通过。
保留本轮在 sm89 Phase D 未完成时启动的记录与 vLLM 0.29.0；现已合并最终 `9aebaf655`。
`baseline_alignment.json` 验证上游设备/编译器源码相对检查点未变、本机二进制指纹和九节标定印章仍匹配、SL-5 特征相同；不覆盖已测二进制。

## 执行关系

历史接续阶段：r4 队列不停止、不清空状态，也不修改已生成的实验二进制。
`pipeline.py register` 原子发布新的 r5 队列，直接接上原调度器：

```text
E3 fixed/trace + R13F → E2b smoke → catalog（冻结 SO/manifest/serving sidecar SHA）
                                         ├→ E2c bulk / PDL / L1-loop / seed+FX21
                                         └→ CPU codegen 审计
E2c 终态 → E4a → E4b → E4c → E4d → 一次有界金丝雀替代 → E4f
E4f 终态 → E5a → E5b → E5c → E5d → E6 S1–S10/报告草稿
```

75 退出码仍由原守卫/调度器按原冷却规则处理；所有 GPU 测量必须持共享锁并通过守卫。
E4 每组四格各三轮、旋转臂顺序。E4c 的 PDL 臂及对照全部 step-events=0；其他组默认 1。
TM 显式 mode/decode-loop/prefill-mode L1；R13F 使用选择器的 auto sidecar，缺少它不能静默退回默认。
vLLM 保持固定版本；仅使用本机已存在的 CUDA 13 编译环境配合 torch cu130，TM 编译器仍是 CUDA 12.8。
E4e 与 E5e 为规格可选项，本队列省略；MB-1d 的 cluster 首次执行数据仍保留。

## 依赖与失败不混淆

| 条件 | 自动动作 |
|---|---|
| 固定几何构建失败 / 必需 prefill 缺失 | 该臂 unavailable，记录原因；不改几何默认值、不阻断无关臂 |
| 对应 mode/loop 冒烟未安全执行 | 仅该执行方式不可计时；其他已执行方式继续 |
| 同步组中的部分候选不存在 | 参考存在时检验其余候选，记录 covered/missing labels；不冒充完整覆盖 |
| 数值不一致 | 保留 mismatches、C-1/C-2、50-process pass_rate；不把全部性能任务取消 |
| 50 进程执行完成但少于 50/50 正确 | complete 表示采集完，正确性 PASS 为另一个字段；性能必须附此限定 |
| 对应同步路径崩溃/超时，50 进程未完成 | 对应路径不能自动放行到性能任务；无关路径继续 |
| 设备无 owner 但持续饱和 | 保持原阈值，停留/报错；不 reset、不放宽守卫、不删除失败 |
| 单次退出后利用率仍显示忙 | 按原六次/5 秒采样区分正常采样滞后与持续故障，不凭一个采样宣布硬件坏 |
| 标记的 canary | 整个配对轮次可重跑一次，保留原轮次；不按最快值挑选结果 |
| 两机模型 SHA 未齐、vLLM 版本不同 | 源码与最终结果已对齐；比较保留模型未核实、0.29/0.30 版本及 sm89 原金丝雀的限定 |

所有 paired effect 仅使用同一 E4 组、同一格、相同三个轮次。
机制相对变化使用配对轮次两臂 TPOT 中位数之比；TM/vLLM E2E 加速比使用逐轮比值的中位数，TPOT 加速比另列，避免与 R13 T10 混口径。
不同几何的 greedy token 不强行逐位一致；C-2 只比较预期数值一致的机制臂；C-1 使用原 hf_check 口径。
PDL/loop 请求不能回退后假装成功；二进制/执行器 sidecar 变化则拒绝运行。

## 工具与证据

- `protocol.py`：每组至少 50 个唯一新 PID；prefill L1、64 token、step-events=0；纯数值失败继续收满 50。
- `pipeline.py`：manifest/可用性/依赖、固定 R13 anchor、正确性、trace、codegen；没有求解器/代价模型调整。
- `report.py`：S1–S10、配对样本/范围、不可用数据、sm89 检查点、候选选择、阶段/步骤/页账本、预测核对。
- `summary.generated.md`：E6 完成后的报告草稿，最终验收再整理为 summary.md 并本地提交；不自动 push。
- `implementation_status.json`：区分实现完成、最终证据审阅完成与 full_protocol_pass；不将可用项收齐写成全部通过。

任何未来运行暴露的允许范围内修正仍须单独 commit；触及同步要给受影响的新二进制追加 50 新进程证据。
当前没有此类同步代码修正，故 E2c 第五组在报告里为不适用，不凭空增加实验。
工程时间上限与禁止调优/修改求解器、代价模型继续有效。

## 最终上游验收工具回放

`acceptance.py` 记录最终基线、目标印章、二进制 SHA、配置与已完成阶段证据。
原 trigger=0 静态误判用最终控制流检查器回放已有 PTX，8/8 通过；旧失败保留。
七轮 vLLM 的 TM-loop 后置条件误拒，用原命令、metrics 和 clean guard CPU 回放；原轮 JSON 不变。
合并期间 Qwen3 B16 第 1 轮入口读到临时冲突标记，未启动模型测量；只补这个缺失轮到独立 r6 输出，保留原失败。
补测经原调度器/守卫，只有 clean guard 才接纳；pending E4b 自动依赖它，不增加实验格子或选择最快值。
固定分页 B16 八臂及两模型 R13F 因 StageFlowModel 的负流量检查失败，记录 unavailable/R14 方案，不改求解器、不做无输入变化的重试。

## E4/E5 验收与页 trace 补齐

verified：原 r5 74 节点全部终态，70 done/4 failed；r6 恢复缺失锚定轮已 done。
`closure.py audit` 独立核对 195/195 可用测量与 102 unavailable，C-2 非自身比较 120/120、两个组 100 不同新 PID。
这不是 297/297 全域通过；五个单测失败、B0h Llama B1 C-1 失败、分页协议缺失仍保留。
cluster 微基准仅 2/12 点执行，10 点因 foreground grid 整除条件 unsupported；不新增合法 grid 实验。

验收发现 PR 的 TR-1/多步 trace 漏加 PAGE_TRACE；原 E4/E5/E6 已先归档，不覆盖旧 trace。
`closure.py register` 发布六个节点：两格 PR 原源码 trace 重编（最多二并行）→四个 PR E5a/E5c 重采集→E6 更新。
使用独立 r7 输出，clean outer guard 后才替代对应 trace；B0 trace 与全部性能轮次不重复。
PAGE_TRACE 与 STAGE/STEP 同时开启，不额外给 TR-1 增加 TRACE_V2 插桩；不改生成几何、求解器或同步源码。
大型表通过压缩原始证据归档保存，索引见 `results/README.md`；本次 CPU 回归 35/35 与 40/40。
verified：r7 六节点全部 done/exit0，四项 trace 的 outer guard clean；原生成源码/几何未变，只有三项预定 trace flags。
`review.py inspect` 核对 27 原冻结二进制与 measurements/S4/S5/S9/S10 SHA 不变；没有重复跑性能矩阵。
离线页账本纠正已做 CPU 38/38 与 40/40 回归，native fingerprint/九节印章匹配；不改 GPU 同步源码。
TR-1 页导出只留 past1000，16 launch 累计等待归一化；past64/575 阶段记录完整，但页明细缺失如实记入最终报告。
原始 r7 与修正前账本独立归档，最终 S6/S7/cross-arch 表和 review/completion 另封存；不覆盖旧证据。
29 个 SO/sidecar/生成源码打包到共享存储 junhuipeng 目录，模型权重也已持久化，SHA 索引随报告提交。
确认全部已注册节点终态且无子进程后，空闲调度器 PID4488 用 SIGTERM 正常退出；没有 reset/清空历史状态。
本轮没有剩余队列；以 195/195 可用记录、102 unavailable、已知单测/C-1/页协议缺口带限制结项。
R14 只写方案，不追加实验或修改 solver；全部仅本地 commit，不 push。
