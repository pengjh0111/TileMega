# T0–T4 可续接状态（基线 e305a9f）

本文件是进度记录，**不是整轮完成报告**。不覆盖历史负结果，不降低验收阈值。

| 项目 | 已完成与证据 | 尚未完成 |
|---|---|---|
| T0 | warmup/repeat、逐臂资源、min-blocks 开关；4090 400/400，四臂 25 轮；OCCUPANCY/result.md | 5090 仅人工脚本，未运行 |
| T1 | 直接 CG 输入、角色/别名绑定；2154 输入侧位模式；有限 S=1..16 DP 32/32 点对照 | 四指标完整进入价格计算、一般符号转移/交点、完整 SOLVER alignment 对照；输入 gate 不等于完整 gate |
| T2.a | FP32 partials 默认启用、旧路径开关；两模型 × 五 split × 50=500/500；流量增量及 FP32 CPU 2154 位模式 | FP32 GPU 新 50 进程回归；全部 shape 数值域；combine 新速率实测 |
| T2.b | golden 来源与三条路线评估，判据不变 | 独立同序逐层诊断、FP32 中转实现与真实成本、973M 通过 |
| T2.c | 原二进制 308/308 分类为数值失败，逐项日志与哈希 | 不能将一次分类重放冒充 50 进程同步验证 |
| T2.d | 参考模型大 split 失败消失 | 4×4096 最优配置重跑、全域可行性结论；不宣布条件 9 全面关闭 |
| T3 | skeleton 先加入第六操作 Fuse；按 T0 修正 residency 预算 | 区间 DP、两条真实融合、50 进程、FUSION/run_sm120.sh |
| T4 | band/wavefront 离线六例合法且 locality 正结果，负载失衡同时报告 | T4.2 真实 CG→runtime task 投影与接入、I3/GPU 验证、PLACE/run_sm120.sh |

## 继续时的关键约束

- 当前 BF16 实例是 128 threads / 212 registers / 24576 B shared / 2 CTA/SM，
  不套用题述的 256-thread 实例。T0 编译清单固定在 partial 修复前，不混用源版本。
- 两级原子扇入与“TC 永不获胜”两条方向仍被否定，不重启。
- FP32 partials 同时保留 Linear→BF16→residual 舍入边界；默认运行时宏与
  CostModelOptions 同步为 true。旧流量模型用 --bf16-partials-baseline。
- T1 选择 (b) 是由于 SDCM 的 sqrt/exp 和 IEEE 重复加法不属于现有 QP 表示。
  有限整数枚举不冒充一般符号交点；四个 coupling 指标当前仍未完整定价。
- PARAMETRIC 工具有 isl context 引用残留警告，尚待定位，不因 exit=0 忽略。
- 生产 placement 不能把 18 算子探针的固定 ReferenceModel 硬塞进 30/60 stage
  模型；必须处理 attention 等逻辑 task 到 runtime stage 与 split rewrite 的投影。
- 已写的 sm_120 脚本只限 OCCUPANCY、BF16，均未运行。缺少的两份不能用空脚本
  或写死结果占位。
- `third_party/barvinok` 原有未跟踪内容保留，未作清理，不纳入本轮提交。
