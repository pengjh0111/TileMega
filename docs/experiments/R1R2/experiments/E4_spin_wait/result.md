# E4 结果（⭐ 最高优先级）：跨 tile block 生产者/消费者自旋等待

## 结论速览（重要：结论在实验过程中被推翻/修正过一次，完整过程都保留在下面，不做事后美化）

调查分三个阶段，结论逐步演变，**最终结论是"⚠️ 受限可行，但有两个都很致命的陷阱"**，而不是最初看到的
"❌ 完全不可行"，也不是天真乐观的"✅ 能用"：

1. **第一版写法（只靠 `acquire`/`release` ordering attribute，不显式串 token）**：
   verifier/lowering/tileiras 全过；但 SASS 证据显示消费者的整个自旋循环被编译器**完全消除**，
   1000/1000 次实机运行 100% 读到脏数据。—— ❌ 这种写法不可行。
2. **修复后（把 token 显式作为 `loop iter_values` 的循环携带值，在每次迭代之间强制一条 SSA
   依赖链）**：SASS 里出现了真正的、会重复执行的自旋循环（`LDG`+`CCTL.IVALL`+`ISETP`+条件回跳），
   1000/1000 次实机运行全部通过（在合理的 float32 舍入误差范围内）。—— ✅ 这种写法在 grid 较小时确实可行，
   且是本次调研发现的、**唯一**能让自旋等待真正生效的写法。
3. **但**：用这个"修复后"能正确工作的 kernel 测试"grid 超过可同时驻留的 tile block 数"时，
   在 grid=120～170 之间（远低于 SM 数 170，也远低于 driver 自己报告的
   `cuOccupancyMaxActiveBlocksPerMultiprocessor` 算出的理论上限 1360）**必现死锁**（30 秒内不返回）。
   更严重的是：改用专门为此设计的 `cuLaunchCooperativeKernel`（它本应该在 grid 超过可行并发数时
   直接在启动时报错拒绝，而不是死锁），**它接受了 grid=170 的启动请求（因为 170 << 1360），
   但实际执行依然死锁**。—— ❌ 目前没有找到任何一种启动方式能安全地把这类自旋等待 kernel 扩展到
   中大型 grid。

三段式标注：
- 第一版写法自旋循环被消除、100% 数据竞争：✅ 已验证（SASS 直接证据 + 1000/1000 次实机复现）
- token 显式串联可以让自旋循环真正生效、correctness 恢复：✅ 已验证（SASS 直接证据 + 1000/1000 次实机通过）
- grid 变大后死锁（即使用 cooperative launch）：✅ 已验证（直接用 `timeout` 观测到 30s+ 挂起，
  且排除了"GPU 本身卡死"——每次挂起后新建 context 都能立刻正常跑小 grid，见下文 GPU 健康检查记录）
- 死锁的根本机制（是 tileiras 生成的 cubin 没有正确声明 cooperative-launch 所需的 metadata，
  还是 CUDA driver 对这类 cubin 的 occupancy 计算本身就不准，还是其他原因）：❌ 推测/不确定，
  没有条件看 tileiras/driver 闭源部分的实现，如实标注为未能查明根因，只报告"确实会挂"这个可复现现象。

## 文件

- `spin_wait_test.mlir` — 核心实验文件：单 entry，按 `get_tile_block_id` 的 x 分量分角色：
  block 0 = 生产者（256 次 chunk store 写满 262144 个 f32，再用
  `atomic_rmw_tko release device` 对 flag 做 xchg），block 1（及以后所有非 0 block）= 消费者
  （`loop { load_ptr_tko acquire device %flag; if eq 1 { break } continue }` 自旋等待后，
  重新读取整个 buffer 算 checksum 写出）。
- `host_test.cpp` / `host_test` — 反复启动 grid=2 的 kernel，每次清零 flag/checksum 并用
  `0xdeadbeef` 毒化 data buffer，比较 checksum 与 CPU 端参考值 `sum_{i=0}^{262143} float(i)`。
- `variants/spin_wait_relaxed.mlir`、`variants/spin_wait_tlblk.mlir` — 对照组：分别把
  `acquire`→`relaxed`，`device`→`tl_blk`（scope），验证是否是 ordering/scope 的具体取值导致问题。
- `variants/host_test_variant.cpp` — 通用化的 host 驱动，可对任意 cubin/entry 名跑同样的
  重复启动 + checksum 校验。
- `spin_wait_test.sass` — `cuobjdump -sass` 完整反汇编（2201 行）。

## 1. Verifier

```
$ /data/cuda-tile/build/bin/cuda-tile-opt spin_wait_test.mlir
```
退出码 0（见 `verify_step1.log`）。中途唯一一次真实的 verifier 报错、以及其修复方式记录如下（有实际价值，
补充了 E1 notes 里没写清楚的一条约束）：

```
spin_wait_test.mlir:59:34: error: 'cuda_tile.load_view_tko' op memory scope is required for acquire load
```
原因：`load_view_tko acquire %data_pview2[...]`（消费者读 checksum 循环里）没写 scope。
`weak`/`relaxed` 可以不写 scope，但 `acquire`（以及推测 `release`/`acq_rel`）**必须**显式写
`memory_scope`，否则 verifier 直接拒绝——这是一条被真实报错验证过的、E1 notes 里 Q5 没有明确写出的补充约束。
修复：加上 `device` 关键字后一次通过。

## 2. Lowering 到 tilebc / 3. tileiras 汇编成 cubin

```
$ /data/cuda-tile/build/bin/cuda-tile-translate spin_wait_test.mlir \
    --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module \
    -o spin_wait_test.tilebc          # exit=0, 647 字节 (translate.log 为空)
$ /usr/local/cuda-13.1/bin/tileiras --gpu-name sm_120 spin_wait_test.tilebc \
    -o spin_wait_test.cubin           # exit=0, 37640 字节 (tileiras.log 为空)
```
两步均无任何报错/警告。`cuobjdump --dump-resource-usage`：
```
Function spin_wait_test:
 REG:126 STACK:0 SHARED:1040 LOCAL:0 CONSTANT[0]:920 TEXTURE:0 SURFACE:0 SAMPLER:0
```

**结论：跨 tile-block 的 release/acquire 原子操作在 verifier 和 tileiras 层面完全被接受，
没有出现任何"跨 block 同步不支持"之类的编译期报错。** 这本身是个中性/正面信号——问题不出在
"能不能表达"，而出在"表达了之后编译器有没有忠实实现语义"。

## 4. SASS 级证据：自旋等待循环被完全消除

用 `cuobjdump -sass spin_wait_test.cubin > spin_wait_test.sass`（2201 行）逐条排查。

**方法**：flag 指针是 kernel 的第 2 个参数，在常量 bank 里的偏移是 `c[0x0][0x388]`
（第 1 个参数 data 指针是 `c[0x0][0x380]`，可以在生产者和消费者两条路径的开头都各自独立看到一次
`LDC.64 R2, c[0x0][0x380]`，互相印证偏移量分配是对的）。**只要消费者真的执行了自旋等待循环，
就必须在其代码路径里出现至少一次 `LDC.64 ⋯, c[0x0][0x388]` 把 flag 指针加载进寄存器**（不可能凭空
产生指向 flag 的地址）。

```
$ grep -n '0x388\]' spin_wait_test.sass
469:        /*0e70*/                   LDC.64 R2, c[0x0][0x388] ;
```
**全文件唯一一次**出现 `c[0x0][0x388]`，位置在生产者的 `atomic_rmw_tko release` 处
（紧邻 `MEMBAR.ALL.GPU` / `ERRBAR` / `CGAERRBAR` 三条真实硬件 fence 指令之后，
`ATOMG.E.EXCH.STRONG.GPU` 之前——这部分是正确、已验证的：RELEASE 语义确实编译成了真实的 GPU-wide fence）。

**消费者代码路径（block-id 分支目标 `0xf40` 开始）完全没有引用 `0x388`**：
```
/*00d0*/  BRA.U UP0, 0xf40 ;                 <- 生产者(0)/消费者(else) 的真实分支
...
/*0f40*/  S2R R0, SR_TID.X ;
/*0f50*/  LDC.64 R2, c[0x0][0x380] ;         <- 直接加载 data 指针（不是 flag！）
/*0f60*/  IMAD.WIDE.U32 R2, R0, 0x4, R2 ;
/*0f70*/  LDG.E.STRONG.GPU R121, desc[UR8][R2.64] ;   <- 直接开始读 data
/*0f80*/  CCTL.IVALL ;
... (后面是 136 条 LDG.E.STRONG.GPU + CCTL.IVALL 交替，即 checksum 循环体，被展开/软件流水线化)
```
即：消费者从进入自己的代码路径开始，**第一条内存相关指令就是直接加载 data 指针并开始读数据**，
中间没有任何针对 flag 的 `LDG`、没有 `ISETP` 比较、没有条件跳回自身的自旋分支。全文件搜索
`ISETP`（7 处）和所有 `BRA`（6 处，见下）确认：唯一涉及 flag 相关寄存器 R2 的比较/分支就是
生产者内部循环退出条件（`0e30`/`0e40`/`0e60` 附近，判断是否还有 chunk 要写)，
以及消费者 checksum 循环的展开控制流；**没有任何一条 SASS 分支的目标地址回跳到一段
"读 flag→比较→跳回"的独立小循环体**。

（另有 18 处字符串 `388` 命中，全部是不相关指令编码巧合命中十六进制注释里的字节，如
`@!P2 STS [R5], R18 ; /* 0x000000120500a388 */`——已逐条核实与 `c[0x0][0x388]` 常量 bank
访问无关，只是编码巧合。）

**结论：✅ 已验证——消费者 `loop { load_ptr_tko acquire device %flag; if eq1 {break} continue }`
整个循环结构（包括对 flag 的加载本身）在最终 SASS 里完全不存在，被编译器彻底消除。**

## 5. 实机运行验证：100% 可复现的数据竞争

```
$ ./host_test 1000
[cuFuncGetAttribute] spin_wait_test: REGS=126 SHARED_BYTES=16 LOCAL_BYTES=0 MAX_THREADS_PER_BLOCK=512
Reference checksum = 34359607296.000000
[iter 0] MISMATCH: checksum=34359672832.000000 ref=34359607296.000000 err=65536
[iter 1] MISMATCH: checksum=-600945908683762565120.000000 ref=34359607296.000000 err=6.00946e+20
[iter 2] MISMATCH: checksum=-600945908683762565120.000000 ref=34359607296.000000 err=6.00946e+20
... (省略，1000 次里每一次都是 MISMATCH)
[iter 999] MISMATCH: checksum=-600945908683762565120.000000 ref=34359607296.000000 err=6.00946e+20

=== SUMMARY over 1000 iterations ===
PASS=0 FAIL=1000
observed checksum range: [-600945908683762565120.000000, 34359672832.000000]  (reference=34359607296.000000)
```
完整原始输出见 `host_test_1000.log`。

**1000/1000 次全部失败**，其中绝大多数迭代的 checksum 是一个巨大的负数
`-6.00946e20`——这正是把 `0xdeadbeef` 这个"毒化"bit pattern 当 float32 解释后累加 262144 次
的结果量级，**直接证明消费者读到的是生产者根本还没来得及写入的"毒化"初始值**，
而不是"读到了旧的正确值"这种更温和的乱序问题。极少数迭代（如 iter 0）误差只有 65536，
说明生产者恰好写完了绝大部分 chunk、只差最后几个还没写完时消费者就已经读完了——
这与"生产者写和消费者读之间完全没有同步、纯靠运气的执行时序重叠"这个结论完全吻合。

## 6. 排除"是不是我少加了某个属性"

检索了 `/data/cuda-tile/include/**/*.td` 中所有 `.td` 定义，没有找到任何
`volatile` / `no_optimize` / `noalias` / 阻止循环不变量提升的属性或独立 fence op
（与 E1 Q7 的结论一致：cuda-tile 没有独立的 Fence/Barrier op）。`LoadPtrTkoOp` 的官方文档注释
（`/data/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td:2333` 起）原文写的是：

> "Token-ordered operations are not constrained by program order. The compiler may reorder them
> (i.e. place them earlier or later in program order) unless further constrained by tokens."

这段文字只承诺"不加 token 时可能被重排"，**没有显式承诺"不会被连同其所在循环一起整体消除"**。
本实验的 SASS 证据说明实际行为比文档字面描述更激进：不只是重排，而是整个自旋循环连同其内部
唯一的内存操作被判定为死代码后一并删除。这看起来是 tileiras 后端对"跨 tile-block 可变内存"
的别名分析没有把"另一个 tile-block 分支里对同一地址的写"当作会让这次 `load` 在多次执行之间
产生不同值的依据——但这只是我们从现象反推的假设（❌ 推测，未看到 tileiras 后端源码，无法坐实
具体是哪个 pass 做的）。

## 7. Ordering / Scope 变体对照（是否换成"更弱"或"更窄 scope"会有本质不同？）

- `variants/spin_wait_relaxed.mlir`：把消费者的 `acquire` 换成 `relaxed`（同样是 device scope）。
- `variants/spin_wait_tlblk.mlir`：把生产者的 `release`/消费者的 `acquire` 的 scope 都从
  `device` 换成 `tl_blk`（语义上这本来就是错的——tl_blk 只保证同一个 tile block 内部的可见性，
  跨 block 用 tl_blk 本来就不该能同步成功；这里测试的是"用错 scope 编译器是否至少能给点提示"）。

两者 verifier / translate / tileiras 全部照常通过（exit=0，无任何警告）：
```
$ cd variants && for f in spin_wait_relaxed spin_wait_tlblk; do ... done
verify exit=0 / translate exit=0 / tileiras exit=0     (对两个变体都是)
```
实机运行（各 200 次）：
```
[spin_wait_relaxed] over 200 iterations: PASS=0 FAIL=200  observed range=[-600945908683762565120.000000, 34359672832.000000] ref=34359607296.000000
[spin_wait_tlblk]    over 200 iterations: PASS=0 FAIL=200  observed range=[-600945908683762565120.000000, 34359672832.000000] ref=34359607296.000000
```
**观测到的 checksum 取值范围与"正确"的 acquire+device 版本完全一致**（同样是
`[-6.00946e20, 3.4359672832e10]`）——这是很强的间接证据：说明 ordering（acquire vs relaxed）
和 scope（device vs tl_blk）这两个属性的具体取值，**在当前 tileiras 后端下对这段代码的最终生成结果
没有任何可观测影响**（自旋循环反正都被删了，换什么 ordering/scope 都一样删）。也就是说，
**用错误的、语义上根本不该用于跨-block 同步的 `tl_blk` scope，编译器既不报错也不产生任何不同的
（错误的）行为——因为两种情况下真正决定运行时行为的自旋循环压根不存在。**

## 8. 大 grid（10×SM）前向进展测试（第一版，自旋已被消除的 kernel）

GPU 有 170 个 SM（`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` 查得），
用 grid=1700（10×170）启动同一个（已确认自旋被消除的）`spin_wait_test` kernel：
```
launching grid=1700 ...
kernel completed without hang.
checksum(from block-1)=34359672832.000000
```
没有出现 hang。但这个结果的价值当时就需要降级说明：由于自旋等待本来就已经被编译器整体删除，
这个测试测的其实不是"真自旋等待在大 grid 下是否死锁"，而只是"一个不做任何等待的 kernel 能不能跑完"。
—— 真正有意义的大 grid 测试必须用第 9 节里"自旋等待真正生效"的版本重新做一遍，见下文。

## 9. 修复：用 token 显式串联循环迭代，让自旋等待真正在 SASS 里生效

第 4/5 节确认了"只靠 `acquire`/`release` ordering attribute"不足以阻止编译器把整个自旋循环消除。
`LoadPtrTkoOp` 的文档（`/data/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td:2333` 起）提到
"unless further constrained by tokens"——这暗示了另一条路：把 token 也做成 `loop iter_values` 的
**循环携带值**，让每次迭代的 `load_ptr_tko` 都真实地读取上一次迭代产出的 token（构造一条无法被
消除的 SSA def-use 链），而不是像原版那样每次迭代都重新 `reshape %flag` 后传入一个"看起来一样"、
在编译器眼里彼此独立、可任意合并/消除的调用。

新文件：`variants/spin_wait_tokenchain.mlir`，消费者部分核心改动：
```mlir
%init_tok = make_token : token
%final_tok = loop iter_values(%tok = %init_tok) : token -> token {
  %val, %ltok = load_ptr_tko acquire device %flag_1c token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
  %val_s = reshape %val : tile<1xi32> -> tile<i32>
  %c1_check = constant <i32: 1> : tile<i32>
  %is_set = cmpi equal %val_s, %c1_check, signed : tile<i32> -> tile<i1>
  if %is_set {
    break %ltok : token
  }
  continue %ltok : token
}
```
（随后 checksum 循环里的 `load_view_tko acquire device %data_pview2[%chunk2] token=%final_tok` 也显式
依赖这个最终 token，确保 happens-before 关系从"自旋等到 flag=1"一路传递到"读取 data"。）

verifier/translate/tileiras 全部一次通过（`tokenchain.verify.log`/`tokenchain.translate.log`/
`tokenchain.tileiras.log` 均为 0 报错）。

**SASS 证据：这次真的出现了一个会重复执行的自旋循环**（`spin_wait_tokenchain.sass`）：
```
/*01e0*/    LDC.64 R2, c[0x0][0x388] ;                 <- 取 flag 指针（本版本里 c[0x0][0x388] 出现了 2 次，
                                                           另一次在生产者的 release atomic 里，两处地址不同，
                                                           互不冲突）
/*01f0*/    BAR.SYNC.DEFER_BLOCKING 0x1, 0x80 ;         <- 循环头
/*0200*/    LDG.E.STRONG.GPU R4, desc[UR8][R2.64] ;     <- 读 flag 的值（每次迭代都真实执行的 load）
/*0210*/    CCTL.IVALL ;                                <- 缓存失效，强制下次读到最新值（ACQUIRE 语义的硬件体现）
/*0220*/    ISETP.NE.AND P0, PT, R4, 0x1, PT ;           <- 与 1 比较
/*0230*/    @P0 BRA 0x1f0 ;                              <- 不等于 1 就条件跳回 0x1f0（真·向后跳转，回边地址 0x1f0 < 0x230）
```
`0x230` 处的 `@P0 BRA 0x1f0` 是一次目标地址小于自身地址的条件分支，即真正的循环回边——这与第 4 节里
"全文件搜不到任何回跳到 flag 读取代码"的情况形成鲜明对比，直接证明这次自旋循环没有被消除。

**实机验证（更正了容差：float32 在这个量级下 parallel/tree reduction 相比 CPU 顺序求和天然有
~16 ULP 级别的舍入差异，用绝对误差 <1.0 的容差在这里过严，改用相对误差 <1e-5）**：
```
$ ./host_test_tokenchain spin_wait_tokenchain.cubin spin_wait_tokenchain 1000
[spin_wait_tokenchain] over 1000 iterations: PASS=1000 FAIL=0 observed_range=[34359672832.000000,34359672832.000000] ref=34359607296.000000
```
**1000/1000 全部通过**，而且 1000 次运行返回的 checksum 值完全一致（都是 34359672832.0，
与参考值 34359607296.0 之间的相对误差 ≈1.9e-6，量级上正好对应 float32 归约顺序不同带来的舍入差，
不是数据损坏）。"每次都返回完全相同的值"本身就是强有力的旁证：如果还存在竞争，1000 次独立运行
不可能每次都精确落在同一个值上。

**结论（更正）：只要显式把 token 做成 loop 的循环携带值、构造真实的跨迭代 SSA 依赖链，
cuda-tile 的自旋等待模式在小 grid 下是真实、正确、可 100% 复现工作的（✅ 已验证）。
仅靠 memory_ordering_semantics 属性本身（不做 token 链）是不够的，这是一个容易踩、而且没有
verifier/编译期报错提示的陷阱（❌ 陷阱已验证存在；⚠️ 这是否是 tileiras 的已知限制还是设计如此，
未找到官方说明，不确定）。**

## 10. 新问题：grid 变大后必现死锁——即使自旋等待本身已经"修好"

用修好的 `spin_wait_tokenchain.cubin` 重新做大 grid 测试（`variants/host_test_biggrid_tokenchain.cpp`，
每次 `timeout N` 包裹防止真死锁卡住整个会话）：

| grid | 结果 |
|---|---|
| 4 / 10 / 32 / 50 / 80 | ✅ 正常完成，checksum 与预期一致 |
| 120 / 150 / 160 / 170 / 1700 | ❌ 挂起，`timeout 15`~`timeout 30` 均触发（exit=124，即被 `timeout` 强制杀死，30 秒内 `cuCtxSynchronize()` 未返回）|

原始命令与输出（节选）：
```
=== grid=80 ===
launching grid=80 ...
kernel completed without hang.
checksum(from block-1, other consumer blocks not checked)=34359672832.000000
exit=0
=== grid=120 ===
launching grid=120 ...
exit=124
=== grid=170 ===
launching grid=170 ...
exit=124
```
死锁阈值在 80～120 之间（未精确二分到单个 grid 值，因为定位大致区间已经足以支撑结论，进一步的
精确定位对报告结论没有增量价值）。**关键点：170 远小于 GPU 的 SM 数量 170——不对，170 恰好等于
SM 数——但即便 grid = SM 数，依然会挂**，说明"每个 SM 摊到恰好 1 个 block"这种朴素假设并不能保证
生产者 block 会被安排到某个 SM 上与消费者们同时运行；哪个 block 被分配到哪个 SM、何时被分配，
在没有额外保证的普通 `cuLaunchKernel` 下是完全不受控的。

**用专门为此设计的 `cuLaunchCooperativeKernel` 结果如何？** 这个 API 的设计目的正是"要么保证全部
block 同时驻留，要么在 launch 时就直接报错拒绝，绝不允许出现这种静默死锁"。实测（
`variants/host_test_coop.cpp`）：
```
$ cuOccupancyMaxActiveBlocksPerMultiprocessor(...)  ->  err=0 numBlocksPerSm=8  (SM count=170, 理论 max coop grid = 1360)
attempting cuLaunchCooperativeKernel with grid=170 ...
cuLaunchCooperativeKernel ACCEPTED the launch, now syncing...
exit=124   <- 30 秒后被 timeout 杀死，即：依然死锁
```
**driver 自己的 occupancy 计算器认为每个 SM 能同时驻留 8 个这样的 block（理论 grid 上限 1360），
170 远低于这个上限，因此 `cuLaunchCooperativeKernel` 正常接受了这次启动请求——但实际执行依然
死锁。** 也就是说，CUDA 官方为解决这一类问题准备的机制，在这个 tileiras 编译产物上并没有起到
应有的保护作用。

**GPU 健康性核实（排除"是 GPU 本身卡死，不是这个 kernel 的问题"这种可能性）**：每次触发死锁、
用 `timeout` 杀掉宿主进程后，都立刻用一个全新的 CUDA context 跑一次 grid=2～5 的小规模验证，
每次都能正常、快速地完成并给出正确结果（同时 `nvidia-smi` 显示的 GPU 利用率也会在死锁进程被杀死
后的几秒内从 100% 回落到 0～2%）——这确认了 GPU 硬件本身没有损坏或永久卡死，问题确实是"这个
kernel/这种启动方式在大 grid 下会构成一个真实的死锁"，而不是环境损坏的假象。

**根因（未查明，如实标注为不确定）**：可能是（a）tileiras 生成的 cubin 没有正确设置
cooperative-launch 所需的某些 metadata，导致 driver 端 `cuOccupancyMaxActiveBlocksPerMultiprocessor`
的计算对这类 cubin 失真（算出的 8 blocks/SM 在实际调度中并不成立）；也可能是（b）我的 role-dispatch
逻辑（`get_tile_block_id` 结合看起来与 CGA/cluster 相关的浮点乘法取整技巧，见 SASS 里
`SR_CgaSize`/`UI2FP`/`UFMUL` 那一串指令）在大 grid 下有还没被发现的正确性问题，导致生产者角色被
错误分配。由于没有 tileiras/driver 闭源部分的源码或调试手段，**这两种可能性都没有被坐实，
只能报告"确实会挂"这个可复现现象本身（❌ 推测，未验证根因）**。

## 结论对 megakernel 可行性的意义

E4 的结论比最初看到的"❌ 自旋等待完全不可行"要更复杂、也更有价值：

1. **cuda-tile 提供的跨 tile-block 同步原语（acquire/release + device scope 的 atomic/load）
   本身是可以正确工作的（✅ 已验证）**——但前提是必须采用一种不太直观、文档里没有明确要求的写法：
   把 token 显式串成 loop 的循环携带值。仅靠 ordering attribute 本身（不做 token 链）会被编译器
   悄无声息地把整个自旋循环消除，产生 100% 可复现、且编译期没有任何报错或警告的静默数据竞争
   （这是本报告认为最容易被忽视、后果最严重的一类陷阱：编译器全程"看起来很正常"）。
2. **即使用了正确写法，自旋等待型的 megakernel 设计仍然有一个更根本、且当前没有已知解法的障碍：
   grid 大小超过某个（远低于 SM 数、也远低于 driver 自称的 occupancy 上限的）阈值后必然死锁**，
   而且 NVIDIA 官方为此设计的 `cuLaunchCooperativeKernel` 保护机制在本次实测环境下没有起作用。
   这意味着**在当前工具链版本下，"persistent kernel + 少量固定数量 tile block 常驻 + 自旋等待做
   跨 block 同步"这一 megakernel 核心设计模式，只能在一个很小、且没有可靠方法预先算出上界的 grid
   规模内使用**——对于 Mirage MPK / Event Tensor 论文里设想的、要用一个 kernel 吃满整张 GPU 的
   occupancy 的场景，这是一个目前找不到绕过办法的实质性阻塞项。
3. 三段式标注总结：token 链修复自旋等待 ✅ 已验证可行；grid 超阈值死锁（含 cooperative launch
   失效）✅ 已验证存在；死锁根本机制 ❌ 未查明，仅报告现象。

## 更正说明（见 E5）

**E5（`../E5_coexistence/result.md` §2）进一步证明，上面 §10 报告的"grid=80 安全、120 起必现死锁"
这个具体数值边界并不是一个稳定的、由 grid 大小唯一决定的阈值**，而是对 kernel 启动前 GPU
的运行时状态（很可能是时钟/功耗状态）高度敏感的一个竞态条件：同一个 `spin_wait_tokenchain.cubin`、
同一个 grid=80，在 host 端多了一次不相关的 1MB 预热 `cuMemsetD32` 时稳定成功（3/3），
去掉这次预热后不仅 grid=80 会挂，**连 grid=30 都会挂（且结果不稳定，多次运行时有时成功有时挂起）**。
用纯粹的宿主端延时（不做任何 GPU 操作）不能复现预热的效果。也就是说 §10 里"80 以下安全"的说法
需要按 E5 的结论理解为**"在那次实验恰好使用的 harness 时序下观测到 80 以下没有触发"，而不是
"grid≤80 就能保证安全"**——这是一个更严重的结论：无法通过选择足够小的 grid 来可靠规避这个问题。
详见 `../E5_coexistence/result.md`。
