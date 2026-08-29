# R2-D: Occupancy Audit — REG=228 vs Driver's "8 blocks/SM" Claim

## 目的

Round 1 (E5) 报告 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 对 `spin_wait_tokenchain.cubin`
(REG=228) 始终返回 **8 blocks/SM**（无论 occupancy/CGA hint 取值），但在 sm_120
(64K regs/SM, 170 SM) 上，REG=228 × 8 = 1824 regs/block-worth，这在数学上不可能：
如果每个 block 需要 REQNTID(=256) 个线程，仅寄存器一项就需要 228×256=58368
regs/block，65536/58368=1.12 → 每 SM 最多 1 个 block，不可能是 8。round 1 没有解开这个矛盾。

## 关键发现 1：round 1 的 occupancy 查询用错了 blockSize ✅ verified

`cuobjdump -elf spin_wait_tokenchain.cubin`：

```
Attribute: EIATTR_REGCOUNT   register count: 228
Attribute: EIATTR_REQNTID    0x100 0x1 0x1        # = (256, 1, 1)
```

`occ_probe.cpp` 用 `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)` 独立确认：

```
MAX_THREADS_PER_BLOCK = 256
NUM_REGS = 228
SHARED_SIZE_BYTES = 160
LOCAL_SIZE_BYTES = 0
SM_COUNT = 170
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=1)   -> numBlocksPerSm=8 (max coop grid=1360)
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=32)  -> numBlocksPerSm=8 (max coop grid=1360)
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=128) -> numBlocksPerSm=2 (max coop grid=340)
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=256) -> numBlocksPerSm=1 (max coop grid=170)
```

即：round 1 (E4 `host_test_coop.cpp` line 15, E5 `host_test_occ_v2.cpp` 等) 的所有 occupancy
诊断代码全部用 `cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, 1, 0)` —— **blockSize=1**，
这与该 cubin 真正被强制执行的 `blockDim=256`（由 `EIATTR_REQNTID` 硬编码，与 host 传给
`cuLaunchKernel`/`cuLaunchCooperativeKernel` 的 blockDim 参数无关）不符。

用 blockSize=1（或 32，二者结果相同——推测是寄存器按 warp 粒度分配，≤32 线程时按 1 个 warp
计算）查询会把资源占用低估 256 倍，从而把 numBlocksPerSm 算成 8（对应“1个warp的寄存器用量”
能塞几份），max grid 算成 1360。用正确的 blockSize=256 查询，得到 numBlocksPerSm=1，
max grid=170 —— **与经验观测到的 hang 边界（grid=170 附近开始必然 hang）完全吻合**，
且与手算完全一致（65536/(228×256)=1.12→floor to 1）。

**结论：round 1 说的“driver 对 tileiras cubin 的 occupancy 计算失真”表述不准确。driver 的算法
本身没问题（见下面对照组）；出问题的是 round 1 所有 harness 查询/使用的 blockSize 输入
(=1)，与该 kernel 真实强制的线程数 (=256) 不匹配。**

## 关键发现 2：这不是 tileiras cubin 独有的问题 —— nvcc 对照组验证 ✅ verified

`reg_pressure_kernel_v4.cu`：普通 nvcc kernel，`-maxrregcount=255` 编译，靠一条 220 元素的
非线性递推链 + 后续大量使用防止寄存器复用，把 `cuobjdump -elf` 报告的
`EIATTR_REGCOUNT` 做到 **218**（与 tileiras 的 228 同一量级，作为受控对照，不追求完全相等）：

```
$ cuobjdump -elf reg_pressure_kernel_v4.cubin | grep -A2 EIATTR_REGCOUNT
Attribute: EIATTR_REGCOUNT   register count: 218
```

对这个**普通 nvcc cubin**做同样的 occupancy 扫描（`occ_probe reg_pressure_kernel_v4.cubin
reg_pressure_kernel`）：

```
MAX_THREADS_PER_BLOCK = 256
NUM_REGS = 218
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=1)   -> numBlocksPerSm=8 (max coop grid=1360)
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=32)  -> numBlocksPerSm=8 (max coop grid=1360)
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=128) -> numBlocksPerSm=2 (max coop grid=340)
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=256) -> numBlocksPerSm=1 (max coop grid=170)
```

**与 tileiras cubin 的模式完全一致**（blockSize=1/32→8, 128→2, 256→1），且 blockSize=256
时的手算 (65536/(218×256)=1.17→1) 同样吻合。这证明：driver 的 occupancy 计算逻辑对普通
nvcc cubin 和 tileiras cubin 一视同仁、算法正确；不存在"driver 专门在 tileiras cubin 上算错"
的情况。round 1 观察到的怪异 8 blocks/SM，纯粹是所有 harness 用 blockSize=1 (对应
host launch 一直用的 `blockDim=(1,1,1)` 习惯) 去查询一个真实需要 256 线程/block 的 kernel
所致的**输入错误**，不是 driver bug，也不是 tileiras 生成的 cubin 元数据有问题。

对照组文件：`reg_pressure_kernel.cu`（v1，naive，只有12 regs）、`reg_pressure_kernel_v2.cu`
（220元素非线性递推，-O3 default 40 regs / -maxrregcount=255 得 118 regs）、
`reg_pressure_kernel_v4.cu`（420元素，-maxrregcount=255 得 218 regs，最终对照组）。

## 关键发现 3：因果链——为什么 `cuLaunchCooperativeKernel` 放行了 grid=170（乃至更大） ✅ verified

`host_test_coop.cpp`（round 1 E4 用于测试 cooperative-launch 准入的 harness）逐字确认：

```cpp
// line 15:
CUresult occErr = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fn, 1, 0);
                                                                                  ^^^ blockSize=1
// line 31:
CUresult err = cuLaunchCooperativeKernel(fn, gridSize,1,1, 1,1,1, 0, 0, args);
                                                            ^^^^^ blockDim=(1,1,1)
```

`cuLaunchCooperativeKernel` 自己的准入检查内部机制上等价于用调用者传入的 blockDim
去做与 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 相同的资源核算，再乘以 SM 数来判断
`gridDim` 是否超过硬件可同时驻留的 block 数。由于 host 传的 blockDim 是 (1,1,1) 而不是
kernel 真实强制的 (256,1,1)，这个准入检查把可同时驻留的 block 数算成 8×170=1360，
远超真实值 170，因此 **grid=170（乃至理论上到 1360 之前的任何 grid）都会被"放行"，
即便硬件实际只能同时驻留 170 个 block**。这就是 round 1 E4/E5 观察到
"cuLaunchCooperativeKernel 接受了 grid=170 的请求" 的直接原因。

**用正确的 blockDim=256 重新验证这个因果链**（新写的 `host_test_coop_correct.cpp`，
query 和实际 launch 都用 `cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)`=256）：

```
=== grid=170 ===
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=256) -> numBlocksPerSm=1 (max coop grid=170)
cuLaunchCooperativeKernel ACCEPTED the launch, now syncing...
rc=124   # ← 仍然 TIMEOUT/HANG！

=== grid=171 ===
cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=256) -> numBlocksPerSm=1 (max coop grid=170)
cuLaunchCooperativeKernel REJECTED at launch time: too many blocks in cooperative launch
rc=0

=== grid=340 ===
cuLaunchCooperativeKernel REJECTED at launch time: too many blocks in cooperative launch
rc=0
```

即：用正确 blockDim 之后，driver 的准入检查这次**正确地**在 grid=171/340 时主动拒绝
（"too many blocks in cooperative launch"），证明准入逻辑本身没问题、且与我们的手算/
occ_probe 结果自洽。**但 grid=170 本身（准入检查认定的"刚好能塞满"的边界值）依然会 hang**
（rc=124，等同 round 1 观察到的挂起）。

## 关键结论：REQNTID/blockDim mismatch 是真实的方法论 bug，但不是 hang 的根因 ⚠️ 重要澄清

这个实验把 round 1 遗留的"occupancy 数字对不上"之谜解开了，但同时也**收窄**了它能解释的范围：

1. ✅ **已解释**：round 1 为什么在 E5 里一直看到"8 blocks/SM"这个数字，以及为什么
   `cuLaunchCooperativeKernel` 会放行 grid=170（乃至更大）而不是在启动前就拒绝——
   全部源于 occupancy 查询和实际 launch 调用统一使用了错误的 blockDim=(1,1,1)，
   而不是 kernel 真实被 REQNTID 强制的 blockDim=(256,1,1)。这不是 driver 的 bug
   （nvcc 对照组证明 driver 对同等寄存器压力的普通 cubin 算法一致、正确），是round 1
   所有 harness 的调用错误。
2. ❌ **没有解释 hang 本身**：修正这个 mismatch（用正确的 blockDim=256 发起
   cooperative launch，此时准入检查确认 grid=170 恰好是硬件真实容量的边界，
   grid=171 会被正确拒绝）之后，**grid=170 依旧 hang**。所以 blockDim mismatch
   只解释了"为什么这么大的 grid 一开始就没被拒绝"，不解释"为什么真正跑到硬件容量
   边界时会卡死"。后者与 R2-A 定位的、发生在 spin-wait 循环之外的、post-spin
   reduce/barrier 代码里的挂起是同一个问题，根因仍未完全查明（见 R2-A §5-6,
   以及下面 R2-B 的进一步证据）。

## 涉及文件

- `occ_probe.cpp` — 通用 occupancy 诊断 harness（cuFuncGetAttribute + 4 档 blockSize 扫描）。
- `reg_pressure_kernel.cu` / `_v2.cu` / `_v4.cu` — nvcc 对照组 kernel 源码（迭代找到 REG≈228 的配置）。
- `reg_pressure_kernel_v4.cubin` — 最终对照组 cubin，REG=218。
- `host_test_coop_correct.cpp` — 用正确 blockDim=256 复现 cooperative-launch 准入 + 挂起测试。

## 状态：R2-D 完成 ✅
