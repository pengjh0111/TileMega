# E6 结果：最小 megakernel（一次 kernel 启动内的异构任务图 + 真实跨 block 依赖 + 独立任务）

## 结论速览

**✅ 完全成功（在 E4/E5 已确认安全的小 grid 范围内）。** 把 E2（按 `get_tile_block_id` 做异构分派）、
E4（token 链修复过的、真正生效的跨 block 自旋等待）组合进同一个 entry、同一次 `cuLaunchKernel`
调用里，构造出一个含 3 种角色 tile block 的最小任务图：

- block 0（生产者）：`C = A + B`（262144 元素，分 256 个 1024 大小的 chunk），每个 chunk 用
  token 链的 `atomic_rmw_tko release device ... add` 给全局计数器 `progress` 加 1，全部写完后
  `atomic_rmw_tko release device %flag, xchg, 1`。
- block 1（消费者，依赖 block 0）：token 链自旋等待 `flag`（E4 里唯一验证有效的写法），
  等到后 `D = C * 2.0`（同样分 256 个 chunk）。
- block 2（独立任务，**不**等待 `flag`）：只做一次性（非循环，不受 E4 那个"循环消除"陷阱影响）的
  `progress` 计数器采样，写入 `block2_sample`，然后做完全独立的 `E = X * 3.0`（1024 元素）。

verifier / lowering / tileiras **全部一次通过**，200/200 次实机运行 **correctness 全部正确**
（C/D/E 三个输出数组，每次都与 CPU 参考值逐元素比对），且 **200/200 次里 `block2_sample` 恒为 0**——
即 block 2 每次都是在生产者完成第一个 chunk（`progress` 从 0 变成 1）之前就已经读到了计数器，
这是"block 2 没有被 block 0/1 之间的同步阻塞、三个角色在同一次 launch 内确实并发执行"的直接证据
（尽管证据强度有限，见下文"局限性"说明）。

## 文件

- `minimal_megakernel.mlir` — 核心实验文件，单 entry 三角色分派。
- `host_test.cpp` / `host_test` — 200 次重复启动 + 数据 poison + correctness 全量比对 +
  `block2_sample`/`progress` 采集。

## 1. Verifier（一次通过，含两处需要注意的语法点，供后续实验参考）

```
$ /data/cuda-tile/build/bin/cuda-tile-opt minimal_megakernel.mlir
```
exit=0（`verify.log`）。过程中两处语法教训（写在这里因为在 E1-E5 都没有踩到过、是本实验新确认的）：

1. **`loop` 的 `iter_values` 声明多个循环携带值时，所有 `(变量=初值)` 对都写在同一个
   `iter_values(...)` 括号内，用逗号分隔**（不是像我最初以为的那样，token 和其它类型分开
   `iter_values(%tok=...)  (%chunk=...)` 两组括号）——正确写法示例（已被 verifier 接受）：
   ```
   loop iter_values(%tok = %init_tok_p, %chunk = %c0_i32) : token, tile<i32> -> token { ... }
   ```
2. **`break` 的操作数类型必须匹配 loop 声明里 `->` 之后的"结果类型列表"，而不是 `iter_values`
   本身的类型列表**——`continue` 必须匹配 `iter_values` 的类型（因为 continue 是回到循环头，
   要交回下一轮的循环携带值），但 `break` 是跳出循环、产出的是循环表达式的最终结果，
   两者类型列表可以不同（本例中 `iter_values` 是 `(token, i32)` 两个值，但循环只需要把最终 token
   带出去，所以声明的结果类型只有 `token` 一个，`break %tok : token` 只带 1 个值，
   `continue %tok, %chunk : token, tile<i32>` 带 2 个值）。最初写 `break %tok, %chunk : token, tile<i32>`
   被 verifier 正确拒绝：
   ```
   error: 'cuda_tile.break' op operand types must correspond to the parent loop result types:
   ('!cuda_tile.token', '!cuda_tile.tile<i32>') vs ('!cuda_tile.token')
   ```
   这条报错信息本身是清楚、可操作的，修复后一次通过。

## 2. Lowering + tileiras

```
$ cuda-tile-translate minimal_megakernel.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o minimal_megakernel.tilebc   # exit=0
$ tileiras --gpu-name sm_120 minimal_megakernel.tilebc -o minimal_megakernel.cubin                                                              # exit=0
```
两步均无警告（`translate.log`/`tileiras.log` 为空）。资源使用：
```
Function minimal_megakernel:
 REG:36 STACK:0 SHARED:1028 LOCAL:0 CONSTANT[0]:968
```

## 3. 实机正确性 + 并发证据

```
$ ./host_test 200
[trial 0] ok=1 block2_sample(progress counter at time block2 started)=0 final_progress=256
[trial 1] ok=1 block2_sample(progress counter at time block2 started)=0 final_progress=256
... (省略)
=== SUMMARY over 200 trials ===
PASS=200 FAIL=0 block2_sample range=[0,0]
```
完整原始输出见 `host_test_200.log`。**200/200 次全部通过**：
- `C`（262144 元素）、`D`（262144 元素）、`E`（1024 元素）三个输出数组每次都与 CPU 参考值精确匹配
  （容差 1e-4，本实验用的是简单加/乘，不涉及归约，不存在 E4 里讨论过的并行归约舍入误差问题）。
- `block2_sample` 200 次全部是 0——说明 block 2 在读计数器时，生产者的 256 次 chunk 循环连第 1 次
  自增都还没发生。这是本次调研中能给出的、关于"同一次 launch 内多个角色确实并发推进"的最直接
  可复现证据。

## 局限性（如实标注，不夸大结论）

1. **`block2_sample` 恒为 0，而不是分布在 `[0,256)` 之间的不同值**，这弱化了"证明了细粒度时间重叠"
   这个说法的说服力——更准确的描述是：**它证明了"block 2 不需要等待 block 0/1 完成即可开始执行并
   完成"（这一点是确凿的，因为如果 block 2 是在 block 0 完成之后才被调度执行，`progress` 就应该已经
   是 256 而不是 0）**，但不能进一步量化"重叠了多少"（因为 cuda-tile 没有提供任何设备端计时原语——
   已检索 `Ops.td`，不存在 `clock`/`globaltimer` 相关 op，这一点见 E1 也未发现——所以拿不到更细粒度
   的时间戳来量化重叠程度）。样本值恒为 0 的另一种解释是"block 2 的工作量远小于 block 0 的第一个
   chunk"（block 2 只做 1 个 1024 元素的 tile 乘法，block 0 的循环第一次迭代要做 2 次 load + 1 次
   add + 1 次 store + 1 次 atomic，量级相近，谁先谁后本就没有必然顺序），这本身也是"两者并发推进、
   没有人为阻塞"的合理结果，与"block 2 被 megakernel 设计正确地免除了同步等待"这一结论一致，
   只是不能过度解读为"block 2 在 producer 工作到一半时才采样"这种更强的说法。
2. grid 固定为 3（远小于 E5 里观测到开始出现 flaky 死锁的 grid=30），这是**刻意的保守选择**，
   直接沿用 E4 grid=2、1000/1000 次通过的已验证安全区间往上加了 1。**本实验不能，也没有尝试去
   验证更大规模（比如几十到上百个并发任务角色）的 megakernel 任务图是否同样可行**——按 E4/E5
   的结论，这几乎肯定会撞上那个未查明根因、且对宿主端时序敏感的死锁问题，所以本实验的"完全成功"
   结论**只适用于 grid 很小（个位数量级）的场景**，不能外推到"megakernel 可以用少量常驻 block
   吃满一整个 170-SM 的 GPU"这个更大的目标。

## 结论对 megakernel 可行性的意义

在把 grid 严格控制在 E4/E5 已确认安全的小范围内的前提下，**cuda-tile 完全能够表达 Mirage
MPK / Event Tensor 论文设想的核心模式**：单次 kernel 启动内，不同 tile block 承担不同角色
（异构计算 + 生产者/消费者依赖 + 独立并发任务），依赖关系通过 token 链 + acquire/release 原子操作
真实生效，独立任务不会被无关的同步开销拖慢。**唯一、但极其关键的限制**是 E4/E5 已经确认的：
这一切只在一个不确定、且当前无法可靠预先判断安全边界的小 grid 范围内成立——这意味着 Mirage MPK
论文里"用一个 kernel 吃满全部 SM"的场景，在当前 cuda-tile/tileiras 工具链版本下**没有已知的
安全实现方式**，即便任务图本身的表达能力（本实验已验证）是足够的。
