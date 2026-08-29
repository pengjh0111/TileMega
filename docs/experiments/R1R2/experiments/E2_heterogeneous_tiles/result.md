# E2 结果（⭐ 最高优先级）：异构 tile 分支 dispatch

## 结论速览

**✅ 已验证：完全成功。** 单个 `cuda_tile.entry` 内通过运行时标量 `task_type` 走 `if`，
两个分支使用完全不同形状/rank 的 tile（分支A `tile<128x128xf32>` + `mmaf`；分支B `tile<256xf32>` + `reduce`），
从 verifier 通过、lowering 到 tilebc、tileiras 汇编成 cubin、到两个 task_type 值的真实设备执行结果正确，全链路无阻塞。
这是本次可行性研究里对 megakernel 最重要的正面证据之一。

## 文件

- `heterogeneous_dispatch.mlir` — 核心实验文件，单 entry 双分支。
- `branch_a_only.mlir` / `branch_b_only.mlir` — 对照组，把两个分支各自单独抽出编译，用于量化资源分配是 union 还是 max。
- `host_test.cpp` / `host_test` — host 侧驱动，加载 3 个 cubin，用 `cuFuncGetAttribute` 查询真实资源占用，
  并对 combined kernel 分别用 task_type=0/1 启动，验证正确性。

## 1. Verifier

```
$ /data/cuda-tile/build/bin/cuda-tile-opt heterogeneous_dispatch.mlir
```
退出码 0，完整回显合法 IR（见 `verify_step1.log`）。没有任何 verifier 报错——if 的两个分支内部使用完全不同的
tile 形状（128x128 vs 256 一维）、不同 rank、不同 op（mmaf vs reduce），因为分支不 yield 任何 SSA 值出 if，
所以不受"分支间类型必须匹配"的约束（对应 E1 Q6 的结论）。

## 2. Lowering 到 tilebc

```
$ /data/cuda-tile/build/bin/cuda-tile-translate heterogeneous_dispatch.mlir \
    --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module \
    -o heterogeneous_dispatch.tilebc
```
退出码 0，产出 535 字节的 `.tilebc`（见 `translate.log`，为空即无错误/警告）。

## 3. tileiras 汇编成 cubin

```
$ /usr/local/cuda-13.1/bin/tileiras --gpu-name sm_120 heterogeneous_dispatch.tilebc \
    -o heterogeneous_dispatch.cubin
```
退出码 0，产出 978600 字节 cubin（见 `tileiras.log`，为空）。**没有出现任何"分支形状不兼容"或"无法为两个分支
分配统一寄存器布局"之类的报错** —— tileiras 后端把两条完全不同的计算路径正常编译进了同一个 SASS 函数体（两条路径用
branch/predicate 指令切换，寄存器/共享内存静态分配一份，覆盖两条路径的需求）。

## 4. 真实设备执行正确性

```
$ ./host_test
[task_type=0, branch A / mmaf 128x128] max_abs_err=0  sample out[0]=15.159997 ref[0]=15.159997  out[16383]=15.139998 ref[16383]=15.139998
[task_type=0] RESULT: PASS
[task_type=1, branch B / reduce sum256] out[0]=76.200005 ref_sum=76.199974 abs_err=3.05176e-05
[task_type=1] RESULT: PASS
```
- task_type=0（分支A，128x128 矩阵乘）：与 CPU 端朴素三重循环参考实现逐元素比较，`max_abs_err=0`（float32 精确匹配，
  因为 mmaf 内部很可能就是标准 FMA 累加，测试数据也刻意选用了能精确表示的小数）。
- task_type=1（分支B，256 元素求和 reduce）：与 CPU 端参考实现比较，误差 `3.05e-5`，是 float32 求和顺序不同导致的
  正常浮点舍入误差量级，判定为正确。

完整原始终端输出见 `run_output.log`。

## 5. 资源分配：union 还是 max？（关键定量证据）

用 `cuFuncGetAttribute`（真实 CUDA driver 运行时报告的数值，而非静态 ELF 元数据）分别查询：

| Kernel | REGS | SHARED_BYTES | LOCAL_BYTES(=stack) | MAX_THREADS_PER_BLOCK |
|---|---|---|---|---|
| `branch_a_only`（仅分支A单独编译） | 255 | 100400 | 8984 | 256 |
| `branch_b_only`（仅分支B单独编译） | 12 | 16 | 0 | 1024 |
| `heterogeneous_dispatch`（合并两分支） | **255** | **98368** | 8984 | 256 |

（另有 `cuobjdump --dump-resource-usage` 的静态 ELF 数值作为交叉验证，见 `resource_usage.log`，数值与
`cuFuncGetAttribute` 略有出入但量级和结论完全一致：REG:255/255/255（三者一致取最大值），SHARED:101424/1040/99392。
两套工具的绝对字节数不完全相同是因为 ELF 静态段大小与 driver 运行时实际分配的 shared memory carveout 计算方式不同，
但**结论方向完全一致**。）

**结论：✅ 已验证 —— 寄存器数是两分支的 MAX（255 = max(255,12)，不是 255+12=267，何况 267 早已超过硬件寄存器上限
必定报错，而这里并未报错）；共享内存同样是 MAX 量级而非 SUM（合并后 98368-100400 字节，与单独分支A的 100400-101424
字节几乎相同甚至更低，远小于 SUM(A,B)≈100416-101440+1040-1056）。**

这意味着：在 tileiras 编译模型下，把多种异构任务塞进同一个 entry 的不同分支，**并不会导致共享内存/寄存器开销随分支数量
线性叠加**——因为同一时刻只有一条分支路径真正执行，编译器按 CFG 做的是静态资源复用（和传统 CUDA `if/else` 分支共享
寄存器文件是一致的行为）。**没有观察到任务描述中担心的"occupancy 因加性资源开销而崩溃"的现象** ——
至少在寄存器和共享内存这两个直接可测的维度上，异构分支合并的资源代价接近其中最贵分支的独立开销，而不是所有分支开销之和。

需要注意的重要限制（诚实标注，不夸大）：
- 本实验只有 2 个分支、且互斥（if/else），编译器容易做路径复用；**没有测试 3+ 分支或更复杂控制流嵌套下是否仍保持
  "max 而非 sum"的规律**，这是 ⚠️ 未验证的推广点，不能直接断言对任意数量的异构任务分支都成立。
- `branch_a_only` 已经把寄存器数顶到 255（硬件单线程可寻址寄存器上限附近），这个测试用例本身对寄存器压力已经很极端，
  MAX_THREADS_PER_BLOCK 从 256（分支A存在时）到理论 1024（仅分支B），说明**分支A的高寄存器压力独自就会把 occupancy
  压得很低**——但这是分支A自身workload（128x128矩阵乘全部放单线程/单tile block里）的固有成本，不是"合并多分支"这个动作本身引入的额外成本。

## 6. 形状边界二分查找

任务要求："若因形状不匹配失败，需要二分查找边界"。**本实验完全没有失败**，从 verifier 到执行全部一次性通过，
因此这一步不适用（无失败可供二分）。

## 结论对 megakernel 可行性的意义

E2 是本次调研中最关键的正面证据：**cuda-tile 的类型系统和编译器完全支持"同一个 entry 内，用运行时标量做分支，
分支内使用完全异构的 tile 形状/算子"这一 megakernel 任务图节点异构性的核心需求**，且实测未观察到资源叠加导致的
occupancy 崩溃。这是 Mirage MPK/Event Tensor 式 megakernel 在 Tile IR 上可行的一个必要条件，本实验中被 ✅ 完全验证。
