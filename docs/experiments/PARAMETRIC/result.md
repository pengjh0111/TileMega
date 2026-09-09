# T1：参数化消费路径——实现前设计

⚠️ 输入侧 2154/2154 位级一致性及有限域 DP 已验证；完整代价消费与一般符号 DP 尚未通过。

## max 之外的表示限制与选择

检查 `CostModel.cpp:270` 发现，现有 θ 相关代价不止是拟多项式和 max：
LiveFootprintBytes 进入 SDCM，CacheHitProbability 包含 sqrt 和 NormalTail，后者
在 `:31` 包含 exp 与有理近似。越过 L2 knee 后，这个函数一般不是分段拟多项式。
此外 `GemmStageNs`/`NonGemmStageNs` 按 wave 顺序重复 FP64 加法；把它改成
“波数 × 单波代价”即便在实数上等价，也不能保证 double 位模式一致。

因此本轮选择 **(b)**：CG 中的四个 MetricAttr、task extent、task count 和 θ
绑定保留符号形式；参数化状态转移只在给定 θ 后执行既有 FP64 操作序列，
不把校准速率改成新值，不将 exp/sqrt 伪装成 QuasiPolynomial，也不改变
FromGeneratedCuda 的语义。Residency 继续链外 pin。

这不是声称 (a) 永远做不到：可扩充为带超越函数与精确 IEEE 运算语义的表达式
及 guard 求解器，但它已不是目前的 isl_pw_qpolynomial。纯 (a) 想在所有参数域
保留旧模型位级行为，目前的表示不足。不能以参考尺寸恰好全驻留 L2 为由把
cache hit 填成常量 1，让特定 2154 个点绕过一般性问题。

## (b) 对 P5.1 的明确限制

它提供可代入 θ 的模型、状态转移和有参数域的解结构；**不提供**各 lane 的
闭式交点、跨任意大参数域的最优性证明或全局拟多项式代价。分段求交点和
相邻 <10% 合并仍是未实现项。若在有限整数域验证各段，必须公开域和求解方法，
不能把逐点枚举或抽样外推冒充一般符号交点求解。

## 一致性闸门设计

CG 路径必须真正解析 task 与 coupling MetricAttr，并校验其角色与坐标域；
从具体 .cu 读回的路径仅用作独立对照，不能在参数化路径中调用它来制造相等。
在 θ 代入后比较每项 CostBreakdown 的 double 位模式，以及 DP 的完整配置序列、
固定 residency 与总代价。1077 FP32 配置 × 2 模型逐格比较，任何不等先停止并
给出第一处依赖量/运算差异，不改容差。

尚待实现和验证：完整代价消费、一般符号 DP、完整 2154 闸门。

## 输入路径的首个闸门

✅ `ModelDescription::FromCouplingGraph` 已直接读取 verified CG 的 model_plan、
task→stage 映射和四种 MetricAttr；不调用 FromGeneratedCuda。ModelDims 可用
命名参数表示 seq/past，SubstituteParams 通过 QuasiPolynomial 求值解析维度。
未绑定的模型不能评估 LiveFootprintBytes。

✅ `run_input_gate.sh` 对两份独立原始 FP32 export 重新 import CG，绑定 S=4、past=3，
与历史生成 .cu 读入的模型比较所有 CostBreakdown double 位模式及 stage_count，
2154/2154 相等。每个 ORACLE 原表实际有 1080 行，其中 3 个历史 RUNFAIL；按既有
1077 个 PASS 配置定义比较，不伪造失败项的 occupancy。首次检查把 1080 行均当
PASS 而拒绝输入，已修正为明确记录排除数，没有改任何数值容差。

⚠️ 这是**输入侧**闸门，不是 T1 整体已完成：四个 coupling 指标已保留并可代入，
但还未完整进入价格计算；下述有限域 DP 不代替完整的 P5.1 参数化求解。
不能仅凭这份相等结果声称代价模型已消费全部拟多项式。

弯路：初版只绑定用户参数 S/past，实际 CG 参数名却是 s11/s14，且存在别名。
输入侧代价仍能相等，因为 coupling 指标尚未参与价格计算。这正说明该闸门不能
单独证明消费链路正确。现由 Frontend 显式记录 dimension_roles，绑定时同时处理
真实名称和 symbol_aliases；缺少角色的旧 CG 要从原 export 重新导入，不能猜名称。
另有 isl context 引用残留警告，尚需定位；未把 exit=0 解释为无资源生命周期问题。

✅ FP32 partial-storage 流量选项前后同样完成 2154/2154 全 CostBreakdown 位模式
相等（独立打印 FP32_PARTIAL_COST_GATE）；它是 T2 的 CPU 回归，不替代本节未完成项。

## 有限整数域 DP（方案 b，明确限制）

✅ `ChainDP::SolveFiniteParameter` 在用户显式提供的闭整数域逐点绑定 θ，调用原 DP；
Residency 仍在原 DP 链外 pin。只把配置和 residency 完全相同的相邻点收为一段，
不做 <10% 合并，也不抽样推断区间。每段保留所有点的 IEEE 代价，不伪装成常数
拟多项式。时间/存储都随域长度增长，不声称适用于无界域。

对完整 1077 候选集，两个模型在 S=[1,16]、past=3 的 uniform 模式各得到 1 段：
tile=16×64×16，stages=2，split=16，residency=2；每个 GEMM 的配置见
`finite_dp.tsv`。32/32 点与独立具体维度的历史输入 DP 选择、residency、总代价位模式
一致。S=4 上 uniform/per-operator-split/per-operator 三种模式也全部一致。
这不是重新执行 SOLVER 的 alignment pruning 全套验收，也不是 BF16 最优性结论。

独立编译开关：TILEMEGA_PARAMETRIC_INPUT、TILEMEGA_FINITE_PARAMETER_DP（默认 1）；
关闭时对应新 API 明确拒绝，不退回生成 .cu reader 或具体常量。旧 API 语义不变。
