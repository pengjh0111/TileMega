# T3：Fusion 参数化——实施前设计

⚠️ 尚未实现区间 DP 或两条手工融合 kernel；不是已验证结果。
形式化已先写入 skeleton §2.3（第六个 CG 操作 Fuse）。

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
