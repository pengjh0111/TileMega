# R9 阶段性代码审查 — 测试中

这是用户要求的中途审查推送，不是 R9 最终验收。SV-1 至 SV-7 的实现已经提交；完整规格符合性、真实模型性能与资源复核仍在测试中。已声明的规格扩展与降级见 [deviations.md](deviations.md)，不能将“有实现”解释为“全部门通过”。

快照时间：`2026-09-25T04:11:10.700740+00:00`。代码基线：`197cd66da36c4fe8929698135d62dcb032b4e91f`。本次进度证据的源 HEAD：`e376b5edacbeebcd30f1980d9840fce4b9d39e86`。Prompt SHA256：`6779512fd12cde3dd13800a52a330d8843e457cdfa6e766a505296750eb5ece1`。

## 当前结果

- **Verified：SV-0 已完成。** 两锚定模型 × seq 1/4/16/64，八个 legacy 对照各 10/10 内部逐位一致，三档时间及求解阶段分解已归档。
- **Verified：独立检查已完成。** 最新 CPU 专项 8/8（包括串并行搜索等价），独立结构/构建命令检查 14/14，GPU 测试调度检查 4/4；参考模型回归已测 120/120。见 `handoff/` 与 `reference/` 原始日志。
- **测试中：32 个真实 skeleton 臂均未完整完成。** 五个搜索运行、27 个排队；全部实测矩阵为 8/40，完成者均是 legacy。G-8 尚未判定，缺样本产生的 FAIL 不是实测失败八格。
- **待搜索结果：** top-5 实际驻留复核/必要的重解及完整模拟、top-3 GPU 实测、所选配置的多变体映射与 Oracle 审计、交错率、attention 归因及全部性能比较。已有后台脚本串接。
- **最终收尾待办：** 审查全部门的结果，定位失败，补最终 FINDINGS/TODO/STATUS/summary，最后再提交推送。本次不做该完成声明。

| 活跃搜索臂 | 已记录的不同全模型配置 | 完成的坐标扫描数 |
| --- | ---: | ---: |
| llama_s1/skeleton-k8 | 2903 | 2 |
| llama_s4/skeleton-k4 | 1706 | 1 |
| llama_s4/skeleton-k8 | 1652 | 1 |
| llama_s4/skeleton-k16 | 1649 | 1 |
| llama_s4/skeleton-kW | 1646 | 1 |

每臂六个 GEMM 语义类、每类 1,218 个候选，最多三轮坐标下降；上表各臂尚在第一轮。逐行原始搜索快照及 SHA256 位于 `review_snapshot/`，不会随着后台计算覆盖。实时产物仍在 `matrix/`；本次没有改变测试进程、搜索域、并发度或计时命令。

## 代码审查入口

| 对应要求 | 主要代码 |
| --- | --- |
| SV-1 单次语义导入、实例化与 SemSig 缓存 | `lib/Frontend/Frontend.cpp`；`lib/Analysis/CouplingCache.cpp` |
| SV-2 算子类与 per-GEMM 变体 | `include/tilemega/Solver/OperatorClasses.h`；`lib/Codegen/Codegen.cpp` |
| SV-3 变体资源缓存与驻留估计 | `include/tilemega/Solver/VariantResourceCache.h`；`tools/tilemega-compile.cpp` |
| SV-4 精确符号 Oracle | `lib/Analysis/SymbolicOracle.cpp`；`lib/Analysis/OracleExpression.h` |
| SV-5 Skeleton 与 IR | `lib/Solver/PlanSkeleton.cpp`；`include/tilemega/Dialect/CouplingGraph/ExecOps.td` |
| SV-6 EST 就绪前沿放置 | `lib/Solver/SkeletonPlacement.cpp` |
| SV-7 坐标下降、驻留枚举与最终模拟 | `lib/Solver/SkeletonSearch.cpp`；`lib/Solver/SkeletonFinalize.cpp` |
| SV-8 实验与全部门检查 | `docs/experiments/SOLVER_V2/{run_matrix,measure,audit_winners,verify}.py` |

活跃完整搜索使用冻结的编译器快照 `5431e69724e7223e48d3e33832111129e9133df7`；后续提交包含已单测的精确关系缓存优化、实验调度/验证和文档。不能把未重跑的 CPU 搜索耗时归给推送 HEAD 的全部优化；该边界也在偏离台账中披露。为保持实验不变，本次不替换运行中的编译器。

## 2,831 个配置具体是什么

`ConfigKey` 把六个 GEMM 语义类的 `(tile_m,tile_n,tile_k,stages,split_k)` 串成一个全模型配置。`seen` 按整个串去重；每条 EVALUATE 表示一次已经返回的评价（一般也可能含拒绝记录，须查看 error 列）。编号从零开始，编号 2830 是第 2,831 条。

这不是六类候选的笛卡尔积穷举。坐标下降每次固定其余五类，只扫一类，选完再扫下一类；重复组合复用已有评价。一次全模型配置评价又会枚举 `residency=1..R_max`，因此实际求解的 Plan 数通常多于 EVALUATE 行数。

每个有效组合按顺序执行：变体资源估计 → 该组粒度实例化与缓存耦合 → 对各驻留度计算 W → 符号 Oracle / Skeleton → EST 就绪调度输出 worker 与 slot → 取最小 Level 2 makespan 作为组合分数。低分表示代价模型预测的全模型完成时间更短，不是 GPU 实测时间。模拟器不参与每次外层打分；坐标下降结束后才对 top-5 做真实驻留复核与模拟，选 top-3 测量并以真实时间最终选优。

`skeleton-k8` 的 8 是 Skeleton 铺开宽度的全局系数 `k_base`，不是 tile K=8、split-K=8，也不是只评估八个 worker；具体宽度随每个 task space 的相对负载缩放，还并入最多两个关键前驱的 worker。
