# T3：Fusion 参数化——实施前设计

## Round 5 当前进度

最新补做：mixed task 阶段价格已实现，4308 GEMM 位模式组及14张scalar
历史表回归通过，见[task_prices_streamed/result.md](task_prices_streamed/result.md)。
独立L-task写回pass已在两模型验证，新增融合op、删除内部边并重建外部耦合，
1890/1890新图守恒格通过，见[rewrite_complete/result.md](rewrite_complete/result.md)。
尚未接区间DP决策、融合runtime投影和GPU；以下历史未实现描述以本段为最新状态。

✅ 逐 task 定价入口 `CostModel::TaskInstanceNs` 与整 stage 入口共用
`TaskCostImpl`；`FusionRecomputeNs` 按 producer 的精确 fanout 和坐标逐项收费，
不以平均 fanout 或整 stage 成本代替。真实 scalar 输入测试 fanout=2 收一次、
fanout=1 收零，18 个错误分支 before/after 均为 0。
`COST_MODEL/instance_price_gate` 重跑 4308 配置组，两个 stage 入口各
904680 次 double 位比较全部相等；新增 seq4 的逐 task 波次组装检查也通过。
这尚不是融合后的 mixed task 完整定价或区间 DP。

✅ `lib/Analysis/FusionAccess.cpp` 保留消费者索引的逐 tensor 中间 tile，
`lib/Solver/FusionResources.cpp` 从精确物理集合计算 peak bytes，再取
`max(producer_scratch, consumer_scratch) + peak`；寄存器取两阶段 max。
`FusionCtasPerSm` 使用 TargetSpec 预算、调用方的寄存器分配粒度及静态 shared，
不把不合法配置夹成 1。旧 ChainDP 路径保持不变，避免改动历史位一致锚点。
这仍需融合 kernel 的 tier-3 编译资源复核，不能把 max 寄存器声明当实测。

✅ `test/unit/fusion_access_test.cpp` 验证 max 而非 sum、外部写回、fanout、
shared 台阶（51200 B 动态加 1 B 静态：2→1）、单 CTA 超预算返回 0；
9 个错误分支分别检查引用数不变，`ISL_CONTEXT remaining=0`。

✅ `lib/Solver/TaskModel.cpp:DeriveModelTaskAccesses` 复用生产语义的物理元素访问，
scalar 使用已有 runtime ownership；完整 element_reads 替代矩形读集，名义
collective 工作量不进入融合访存分析。`task_element_work_test` 对两份生产
export 执行，gqa2/mha4 semantic roundtrip 34/68，错误分支合计 13，零残留。

⚠️ 这些仍不是融合完成。混合task定价和独立L-task写回已补做，
区间DP、事件重投影与两条真实GPU融合待实现。`TaskInstanceNs`用于单task
重算，不能拿整个stage的`TaskCostNs`替代。

⚠️ `run_sm120.sh` 已替换占位逻辑，调用 `../run_schedule_sm120.py`。
用法：`run_sm120.sh --manifest <frozen-builds.json> --out <new-directory>`。
runner 核对实际 sm_120 GPU、BF16 全比较矩阵、source/binary/ptxas/prediction
哈希，50 轮完整状态轮转，warmup=5/repeat=11，输出全部 resource/schedule
记录与预测并在正确性失败时停止。两条 CPU 单测与 shell 语法检查通过。
未运行 sm_120；真实融合构建/预测 manifest 尚未产出，因此 B1.5 未整体验收。

⚠️ 尚未实现区间 DP 或两条手工融合 kernel；不是已验证结果。
形式化已先写入 skeleton §2.3（第六个 CG 操作 Fuse）。

✅ 本次补做：`FusedTaskInput` 从写回的 `phase_semantics`、`phase_granularities`、
`phase_stages` 和 `phase_maps` 重建阶段输入；它重新计算物理读写集合并与写回
属性做双向集合检查，外部消费者/导出张量会强制保留生产者写回。实际
RoPE→KV 与已有 GEMM→add 两类候选（gqa2/mha4 共 4 格）写回前后六个
double 价格字段逐位相等，10 个非法输入分支均拒绝且 `ISL_CONTEXT remaining=0`。
混合算术审计确认 GEMM 保持 MMA、add/非 GEMM 保持 SIMT；这是 CPU 功能验证，
不是 GPU 融合执行验收。`SumAlong(C)` 使用 barvinok 的精确关系 fiber 求和，
不采样参数，也不把阶段坐标依赖强行当常量。

本次验证：portable CTest 39/39 通过；`fusion_rewrite_test` 的融合图仍为
1890/1890 边守恒、16 个拒绝错误分支，`fusion_written_price_test` 为
4/4、10 个拒绝分支、零 isl 残留；五个 target audit 均 0 failures，policy
通过。阶段粒度/stage 元数据的缺失、错长和负 stage 均有独立拒绝测试。

本轮仅链上相邻的单生产者融合，不跨分支。合法性需要 tile 投影匹配，外部仍需
中间值时不能删除写回。收益分别是内部边同步与中间 global 流量；代价分别是
生产者 tile 限制、整个融合段的 live shared/register 预算及 task 数收缩后的
wave tail。GEMM→RMSNorm 要整行，不得保持原来多个 N tile 的并行度却只减事件。

Residency 仍是整个 kernel 的外层固定参数，内层只考虑能达到这个层级的区间；
融合段不能独立选另一 occupancy。最终结果还须以该段的实际寄存器编译信息校验。

## T0 依赖

✅ 当前受测 BF16 模型的资源记录与题述的 256-thread 实例不同：128 threads、
24576 B TaskSmem、L2 212 registers，在 min_blocks=1/2 均实际为 2 CTA/SM。
不使用“当前必为 1 CTA，所以 smem 恒免费”或“只有 1152 B 余量”的假设定型。
以 TargetSpec 的 102400 B/SM 及该实例为例，固定 2 CTA 的动态 shared 余量是
51200−24576=26624 B（还须扣静态 shared），只对此实例成立。

✅ 两状态 400/400 correctness、T0 四臂 25 轮已完成。⚠️ 它尚未验证
GEMM→add 或 GEMM→RMSNorm 的代价预测。若融合不跨预算台阶，residency 增量
成本可为 0，但约束必须保留；实际融合增加寄存器需求时也不能沿用 212。
