# R2-F: Does tileiras Auto-Pipeline Across Persistent-Loop Iterations?

## 范围说明 ⚠️

Round 1/2 现有的所有 kernel（`spin_wait_tokenchain` 及其变体）都是纯 elementwise
load+reduce（求和 checksum），**不包含任何 MMA/tensor-core 矩阵乘法**。因此本实验只能回答
"tileiras 是否在 load→compute 之间做跨迭代的软件流水线/双缓冲"这个结构性问题，
**无法**测量真实 TFLOPS-vs-peak（那需要一个全新的、专门写的 tile-level matmul kernel，
不在现有 round 1/2 产出范围内）。这一子问题标记为 ❌ 未完成，如需回答必须新增一个
K-dim 累加的 matmul kernel 实验（超出本轮时间预算）。`optimization_hints` 对资源占用的
影响已经在 round 1 E5 里验证过（REG=228 不随 occupancy/CGA hint 变化），此处不重复。

## 方法

对 `spin_wait_tokenchain.cubin`（consumer 分支的 256 次 `%chunk2` reduce-for 循环）做
`cuobjdump -sass` 全量反汇编（`/tmp/tokenchain_full.sass`，3321 行），统计：
- 所有 `BRA` 指令的目标地址，区分“前向跳转”（if/else 结构）和“后向跳转”（真正的 runtime 循环）。
- `LDG.E.STRONG.GPU`／`FADD`／`SHFL.BFLY`／`BAR.SYNC.DEFER_BLOCKING` 的出现次数和相对顺序。

## 发现 1：consumer 的 256 次 reduce 循环确实被完全展开（无 runtime 回边）✅ verified

全文件仅有的 `BRA` 回边（目标地址小于自身地址，真正构成循环）：

```
/*0230*/  @P0 BRA 0x1f0     # 自旋等待循环（consumer, 单条件轮询）
/*65b0*/  @P0 BRA 0x6260    # 见发现 3（producer 的数据生成循环）
```

consumer 的 reduce-for 循环体所在区间（`0x260` ~ `0x61c0`，约 2400 条指令）内**没有任何
后向 `BRA`**——即这段代码是从 `0x260` 一路直线执行到 `0x61c0`，是 256 次循环体被静态
展开（unroll）拼接而成的直线代码，而不是一个真正跑 256 次的 runtime 循环。这与 R2-A
result.md §3 的描述一致：`BAR.SYNC.DEFER_BLOCKING` 在这段区间里等间隔出现 112 次
（`grep -c BAR.SYNC` 结果），与展开的迭代边界吻合。

## 发现 2（新，之前轮次未记录）：自旋等待循环体内部本身也有一条 `BAR.SYNC.DEFER_BLOCKING` ✅ verified

反汇编自旋循环本体（偏移 `0x1f0`~`0x230`，`spin_wait_tokenchain.cubin`）：

```
/*01f0*/  BAR.SYNC.DEFER_BLOCKING 0x1, 0x80 ;   <- 每次轮询迭代都执行的块内硬件屏障
/*0200*/  LDG.E.STRONG.GPU R4, desc[UR8][R2.64] ;
/*0210*/  CCTL.IVALL ;                          <- acquire 序的缓存失效（baseline 版本）
/*0220*/  ISETP.NE.AND P0, PT, R4, 0x1, PT ;
/*0230*/  @P0 BRA 0x1f0 ;                        <- 回边
```

**即：自旋等待的每一次轮询，都要先让该 tile block 的全部 256 个线程在
`BAR.SYNC.DEFER_BLOCKING` 上会合，然后才发起一次 flag 读取。** 这不是 round 1/round 2-A
之前任何文档提到过的细节。含义：
- Tile IR 把"tile block"当作 256 线程 SPMD 执行、在控制流合并点插入硬件屏障做收敛
  （tileiras 代码生成策略），而不是只让 1 个线程轮询、其余 255 个线程等待。
- 这意味着自旋等待循环远不是"几条指令的紧凑忙等"，而是**每次迭代都要等 256 个线程里最慢的
  warp 到达屏障**——如果同一 block 内有任何 warp 因为占用/调度原因迟迟不能被
  issue（例如 grid=170 时 SM 占用率达到 100%、零富余度的场景下，一个 block 的部分 warp
  可能要等待其他 block 让出资源），这个屏障本身就可能成为额外的、之前从未被识别过的
  卡顿/死锁风险点。R2-A 定位的卡死 PC（`0x6190`，consumer reduce 收尾的 `LDS`）
  紧跟在另一条 `BAR.SYNC.DEFER_BLOCKING` 之后（R2-A result.md 66-68 行），与这里发现的
  "BAR.SYNC 密集分布在 consumer 路径"的模式一致——**卡死很可能与块内屏障的到达/收敛机制
  有关，而不是自旋轮询本身的内存序问题**。这为"R2-B（改用 relaxed 轮询）没能修复挂起"
  提供了一个结构性的解释方向：R2-B 只去掉了轮询循环里的 `CCTL.IVALL`（缓存失效），
  **没有**去掉 `BAR.SYNC.DEFER_BLOCKING`（block 内的实际硬件屏障仍然每次迭代都执行）。
  标记为 ⚠️ documented-but-unverified：尚未用 cuda-gdb 逐 warp 验证卡死的具体线程是否
  正卡在某条 `BAR.SYNC` 指令本身的到达计数上（R2-A 观测到的卡死 PC 是 BAR.SYNC *之后*
  的 LDS，不是 BAR.SYNC 指令本身），所以这是一个有强支持证据、但未做最终定点验证的假说。

## 发现 3：producer 的数据生成循环是真正的 runtime 循环（2x 展开），consumer 是完全展开 ✅ verified

producer 分支（`is_producer` 为真时跳转到的 `0x61d0` 之后）的数据生成循环：

```
/*64c0*/  IADD R15, R15, 0x2 ;                 # 每次迭代步进 2（对应源码 256 chunks，
/*65a0*/  ISETP.NE.AND P0, PT, R15, 0x100, PT ;#  即 2x 展开后跑 128 次真实循环）
/*65b0*/  @P0 BRA 0x6260 ;                     # 回边
```

即 producer 的 256-chunk 写入循环被 tileiras 以展开因子 2 编译成 **128 次真实 runtime
循环**，而不是像 consumer 一样完全静态展开。同一个 kernel 里，tileiras 对 producer/consumer
两条分支的循环展开策略不一致（producer 部分展开，consumer 完全展开）。这是 round 1 未记录
的新事实，说明 tileiras 的展开决策不是一个全局固定阈值，可能与循环体大小、寄存器压力等因素
相关（具体决策依据未探究，标记 ❌ speculation 到此为止）。

## 发现 4：consumer 内存加载呈现"先大批量 LDG，后集中 reduce"的模式 ⚠️ documented-but-unverified（跨迭代流水线的解释存疑）

统计 `/tmp/tokenchain_full.sass`：

```
第一条 LDG 在第 71 行（偏移 0x200，自旋循环里那条），第一条 FADD 在第 973 行（偏移 0x1e30）
第一条 FADD 之前一共出现了 209 条 LDG（全文件总共只有 225 条 LDG）
全文件 FADD 共 420 条，SHFL.BFLY 共 196 条
```

即：绝大多数（209/225 ≈ 93%）的 `LDG` 集中在代码最前段一次性发出，随后才开始出现
`FADD`/`SHFL.BFLY`（warp 内蝶式规约）+ `STS`/`LDS`（跨 warp 规约）交替的收尾序列。
这**不是**一个"每次迭代 load 一点、立刻 reduce 一点"的紧耦合模式，更像是把很大一批加载
提前集中发出（提高内存级并行度/隐藏访存延迟），之后再做批量规约。

**未能确认**的是：这 209 条 LDG 是否精确对应"未来若干次循环迭代"的数据（即经典意义上的
软件流水线/双缓冲：预取 iteration i+1..i+k 的数据、同时计算 iteration i），还是仅仅是
单次 `load_view_tko` 把一个 1024 元素 tile 拆成的加载指令、以及寄存器分配阶段把多个逻辑
chunk 的数据保留在不同物理寄存器里所致的表面现象。要精确回答需要对每条 LDG 的目标地址
（偏移量）做完整逐条反推，映射回具体的 `%chunk2` 迭代号——这部分工作量较大，在本轮时间预算内
未完成，故标记为 ⚠️ documented-but-unverified，不作为"tileiras 实现了跨迭代软件流水线"的
确定性结论。

## 结论（R2-F 核心问题的回答）

- ✅ **有真实的跨迭代加载批处理/预取现象**（大批量 LDG 领先于 FADD/reduce 发出），
  说明 tileiras/ptxas 不是逐迭代严格串行发出"1 load→1 compute"。
- ⚠️ **是否构成经典意义的"手动/编译器双缓冲流水线"未定点验证**——现有证据不足以精确回答
  "iteration i+1 的 load 是否被显式提前到 iteration i 的 compute 之前"这个具体问题，
  只能确认"存在大规模的加载先行现象"。
- ❌ **真实 TFLOPS/peak 未测量**——现有实验里没有 MMA/matmul kernel，此项无法在本轮完成，
  需要专门新增实验（不在本轮时间预算内）。
- ❌ **手动 2 次迭代 unroll 对比测试未完成**。
- ✅ **新发现（意外收获）**：自旋等待循环体本身在 SASS 里包含一条真正的块内硬件屏障
  `BAR.SYNC.DEFER_BLOCKING`，每次轮询都执行；这为解释"R2-B 的 relaxed+acquire 轮询为何
  没能修复挂起"提供了一个新的、有 SASS 证据支持但未做最终 cuda-gdb 定点验证的结构性假说
  ——即挂起可能与 256 线程的块内屏障收敛机制在高 SM 占用下的行为有关，而不仅仅是
  round 1 假设的跨 block flag 可见性问题。

## 状态：R2-F 部分完成（核心结构性问题已回答，TFLOPS/手动 unroll 对比未完成，明确标记为 pending）
