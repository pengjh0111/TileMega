# E1: 原语盘点（纯代码阅读，无需 GPU）

方法：`grep -n "^def " /data/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td` 得到完整 op 列表（97 个 op 定义），
逐一交叉核对 Types.td / AttrDefs.td。

**重要版本说明（贯穿全文，请先读）**：本机存在两份不同版本的 cuda-tile 源码：
1. `/data/cuda-tile` — 独立仓库，commit `8a775693b18303d6c696be6ffd06dadad1b32a8e`（2026-01-13）。
   **/data/cuda-tile/build/bin/{cuda-tile-opt,cuda-tile-translate}` 就是从这份源码编译的** —— E2-E7 所有手写 MLIR 实验都要用这个版本的语法/能力。
2. `/data/tensor-ir/build/_deps/tensor_ir_cuda_tile-src` — tensor-ir 通过 CMake FetchContent 拉取的**另一份、更新的** cuda-tile，
   commit `af2417041cc939b87ef56d92cfdcf61737c5457e`（锁定于 `/data/tensor-ir/cmake/TensorIRDependencyPins.cmake:6`，
   下载 URL 见 `/data/tensor-ir/build/_deps/tensor_ir_cuda_tile-subbuild/.../tensor_ir_cuda_tile-populate-urlinfo.txt`）。
   `tensor_ir-compiler`/`tensor_ir-opt` 是链接这份代码构建的。

两者 op 集合的 diff（`diff <(grep -oP '^def CudaTile_\K\w+' 旧/Ops.td | sort) <(同 新/Ops.td | sort)`）：
新版本比旧版本**多出**：`AllocaOp`, `Atan2Op`, `AtomicRedViewTkoOp`, `MakeGatherScatterViewOp`, `MakeStridedViewOp`,
`MmaFScaledOp`, `PackOp`, `UnpackOp`；且 `PrintOp` 改名为 `PrintTkoOp`。
✅ 已验证（diff 命令直接输出）。这意味着任务描述里提到的 `AllocaOp`、`AtomicRedViewTkoOp` **在我们能编译运行的工具链（/data/cuda-tile 独立仓库这一版）里根本不存在**，
它们是 cuda-tile 项目后来（2026-01-13 到 2026-08-19 之间）才加入的能力。凡涉及这两个 op 的问题，下面分别标注"旧版（可编译）"与"新版（仅读到，未编译验证）"的结论。

---

## Q1. device 端函数调用

`grep -n "^def " Ops.td` 完整列出 97 个 op（见上），涵盖 Core/ControlFlow/Atomics/Conversion/View/Mem/FloatingPoint/Integer/Bitwise/Misc 各类。
**没有 `FuncOp`、没有 `CallOp`**（两个版本都没有——用同样的 grep 在新版 Ops.td 里搜 `CallOp\|FuncOp` 也是空）。
唯二能作为顶层单元的是：
- `CudaTile_ModuleOp`（`/data/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td:2904`，`SymbolTable` trait）
- `CudaTile_EntryOp`（`Ops.td:1768`，`FunctionOpInterface, IsolatedFromAbove, SingleBlock` trait）
- `CudaTile_ReturnOp`（`Ops.td:3496`，entry 的隐式/显式终结符）

`ModuleOp` 是 `SymbolTable`，理论上可以放多个 `entry`（每个 entry 有自己的 `sym_name`，SymbolTable 只要求名字不冲突）。
✅ 已验证：用 cuda-tile-opt 实测一个含两个 entry 的 module 可以通过 verifier（见 `E1_primitives/multi_entry_test.mlir` 与下方命令输出）。
但因为**不存在 CallOp**，entry 之间**不能互相调用**——`EntryOp` 是 `IsolatedFromAbove`，其 region 内只能引用自己参数和内部产生的 SSA 值，
没有跨 entry 引用符号并跳转执行的算子。多个 entry 只能各自独立被 host 通过 `cuLaunchKernel`/`cuModuleGetFunction` 启动，
这与 README.md:298 (`cuModuleGetFunction(&example_kernel, cuModule, "example_kernel")`) 的用法一致。
结论：**❌ 不可行** —— 没有 device 端函数调用能力；一个 module 可以有多个 entry，但它们之间没有调用关系，只能各自独立被 host 启动。

## Q2. tile 形状是否必须编译期常量

`Types.td:102-138`（`CudaTile_TileType`）明确写道：
> "A tile type has a shape and and element type. **The shape of the tile must be fully static.** ... Only power-of-two shape dimensions are supported."

即 `tile<?x128xf32>` **不合法**——TileType 根本没有为形状保留"动态"占位符（对比 `TensorViewType`，见下）。
✅ 已验证：静态阅读类型定义已经足够确定（这是类型系统的硬约束，不是某条 verifier 规则，是"动态 tile 类型"根本无法构造）；
另外在 cuda-tile-opt 里也无法解析 `tile<?x128xf32>`（parser 会报错，因为 tile<> 的 shape 语法本身只接受整数列表，不接受 `?`）。

与之相对，`TensorViewType`（`Types.td:176-232`）和 `PartitionViewType`（`Types.td:233-`）**允许**按维度动态：
> "The shape and the stride can be dynamic on a per-dimension basis... printed as `?`."
> 例：`!cuda_tile.tensor_view<?x?xf16, strides=[?,1]>`

即"整体张量形状/步幅"可以是运行时值（通过 `make_tensor_view` 的运行时标量参数指定），但从这个 view 里 `load_view_tko`/`store_view_tko`
每次取出的**那一个 tile 的形状**（由 `partition_view<tile=(RxC), ...>` 里的 `tile=(...)` 决定）必须是编译期常量、且是 2 的幂。
这正是 E0 中 `add_dynamic.mlir` 能跑通动态 shape、但每次搬运的 tile 本身仍是静态 `tile<8x8xf32>` 的原因（对照 `/tmp/matmul_lowered.mlir` dump 出的
`tile<1x64xf32>`、`tile<64x64xf32>` 等全部是静态形状）。
结论：**⚠️ 受限** —— tile 粒度（形状）必须编译期常量+2的幂；只有"取多少个 tile"（迭代次数/网格大小）和"张量总大小"可以是运行时值。

## Q3. shared memory 在 IR 层是否可见

`Types.td:74-89`（`CudaTile_PointerType`）：
> "An elemental pointer type $pointerType represents **a single location in global device memory**."

**没有地址空间参数**——`ptr<T>` 只有一个 `pointeeType` 参数，没有 "shared"/"local"/"global" 的枚举。
旧版（/data/cuda-tile，可编译版本）里**完全没有 `AllocaOp`**（`grep -rn AllocaOp` 全仓库为空），
即这一版本的 IR 里**没有任何"显式分配片上/临时内存"的算子**——所有可寻址内存都是通过 kernel 参数传入的 `tile<ptr<T>>`（对应 host 侧 `cuMemAlloc` 的 global memory）。

新版（tensor-ir 内部依赖的 af241704 commit，`.../tensor_ir_cuda_tile-src/include/.../Ops.td:256`）新增了 `CudaTile_AllocaOp`：
```
def CudaTile_AllocaOp : CudaTileMemOpDef<"alloca", "13.3"> {
  // 分配 num_elem 个元素，返回 tile<ptr<T>>；alignment 需 2 的幂
  // 默认结果地址只在"当前 tile thread"内私有可见（寿命限定在 alloca 所在的 block）
  // 加 `global` unit attr 可以让地址在其他 tile thread 间可见（"shareable across tile threads"）
}
```
即使有了 AllocaOp，返回类型依然是同一个 `tile<ptr<f32>>`（PointerType 没有变化，没有增加地址空间字段）——
"private 私有" vs "global 可共享" 是一个**逻辑/生命周期属性**（靠 `global` unit attribute 标记），而不是一个独立的地址空间类型。
IR 层完全看不到"这段内存最终落在寄存器、local memory 还是 __shared__ 里"——这是完全由后端编译器决定、对 IR 不可见的实现细节。
`num_elem`/`alignment` 语法示例见 `.../Ops.td:280-289`（`mlirExamples`）。

结论：
- ✅ 已验证（旧版，可编译工具链）：**没有 shared memory 概念，也没有 alloca；一切内存都是 global pointer**。
- ⚠️ 文档声明但未编译验证（新版 af241704，只在 tensor-ir 的 FetchContent 依赖树里读到源码，没有独立的 cuda-tile-opt/translate 二进制去实测）：
  有了 `alloca`，但依然是"private vs global 可见性"的逻辑抽象，**不是**传统意义上程序员可控的 `__shared__` 显式地址空间；
  没有办法在 IR 层指定"这块内存必须落在 shared memory"。

## Q4. cuda_tile.entry 的 optimization_hints

`EntryOp`（`Ops.td:1768`）第 1792 行：`OptionalAttr<CudaTile_OptimizationHintsAttr>:$optimization_hints` —— entry 级别的**单一**属性，
不是逐区域（if/loop 内部不能有自己的 optimization_hints 覆盖 entry 级别的值——除了 load/store 单独也能挂 optimization_hints，见下）。

`AttrDefs.td:67-` (`CudaTile_OptimizationHintsAttr`)：
```
optimization_hints=<
  sm_100 = {num_cta_in_cga = 8},
  sm_120 = {num_cta_in_cga = 16}
>
```
是**按 SM 架构名分组**的嵌套字典（`allowedKeysArr = {sm_80, sm_90, sm_100, sm_103, sm_110, sm_120, sm_121}`，
`Attributes.cpp:110-111` 用户可选 `default` 作为不区分架构的兜底键，见 tensor-ir `EmitHelpers.cpp:60` 的用法）。

旧版（/data/cuda-tile，可编译）支持的字段只有三个（`Attributes.cpp:45,72,` + `AttrDefs.td:106-109`）：
- `num_cta_in_cga`（`kNumCTAInCGA`）：CGA 里的 CTA 数，**必须是 ≤16 的 2 的幂**（`entry_opt_hints_invalid.mlir` 里
  `optimization_hints=<sm_100={num_cta_in_cga=7}>` 被 verifier 拒绝："expected power-of-two ≤ 16"）。
- `allow_tma`（`kAllowTMA`）：仅用于 load/store，不用于 entry。
- `latency`：仅用于 load/store。
- `occupancy`（`kOccupancy`）：`entry_opt_hints_invalid.mlir` 显示 `occupancy=64` 被拒绝："integer value in the range [1, 32] is expected"。

**旧版里没有 `num_worker_warps_per_cta`**（`grep -rn "num_worker_warps_per_cta\|kNumWorkerWarps" /data/cuda-tile` 全仓库为空，
包括 lib/Dialect/CudaTile/IR/Attributes.cpp）。这与任务描述假设的"已知至少有 num_worker_warps_per_cta"不符——
这个字段是**在新版 cuda-tile（af241704）才加入的**：
```
.../tensor_ir_cuda_tile-src/.../AttrDefs.td:102: num_worker_warps_per_cta - ... 只支持 [4, 8] ... for entry
.../AttrDefs.td:114-115:
    sm_100 = {num_cta_in_cga = 8, num_worker_warps_per_cta = 8},
    sm_120 = {num_cta_in_cga = 16, num_worker_warps_per_cta = 4}
```
（这也解释了为什么 `tensor_ir-compiler`（链接新版 cuda-tile）在没有传 --tile-size 时能跑通，
而它源码 `EmitHelpers.cpp:56` 里引用 `cuda_tile::OptimizationHintsAttr::kNumWorkerWarpsPerCTA`
这个符号在旧版 /data/cuda-tile 里根本不存在——如果拿旧版头文件去链接 tensor-ir 会编译失败，
这佐证了 tensor-ir 必须用自己 FetchContent 的新版本，两个仓库不能混用。）

是硬约束还是建议？从命名（"suggest"/hint）和 verifier 只检查**取值范围合法性**（power-of-2 ≤16，occupancy∈[1,32]，
warps∈{4,8}）而不检查"能否满足"来看，是**编译器提示（hint）**，不是运行时强制保证——具体见 E5 的讨论
（cuda-tile 侧没有找到任何"若无法满足 hint 则报错"的运行时检查逻辑；这属于 ⚠️/❓，因为没有找到反例真正去验证 occupancy 不被满足时会怎样）。

`num_worker_warps_per_cta` 是 **per-entry**（顶层字典的 key 是 SM 架构名，value 内的字段是标量），
**没有找到区域级/if-else 分支级重新指定 warp 数的语法**——EntryOp 只有一个 `optimization_hints` 属性挂在整个 op 上，
`IfOp`/`LoopOp`/`ForOp` 定义里都没有自己的 `optimization_hints` 参数（只有 load/store 才有 op 级别的 optimization_hints，
用于 allow_tma/latency，而这两个字段与 warp 数无关）。

**没有找到任何"显式指定哪些 warp 做 MMA/哪些做 TMA/哪些做控制流（warp specialization 角色分配）"的 IR 机制**——
`grep -rni "warp\|specializ\|agent"` 遍历两个版本的全部 .td 文件，除了 `num_worker_warps_per_cta` 这个标量数字提示外，
没有任何角色分配相关的 op、attribute 或 region 修饰符。详见 E7。

结论：**⚠️ 受限** —— optimization_hints 目前只有 4 个字段（num_cta_in_cga, allow_tma, latency, occupancy），
新版又加了 num_worker_warps_per_cta（仅取值 4 或 8），全部是"entry 级/op 级标量提示"，不支持区域粒度变化，也没有 warp 角色的显式控制。

## Q5. 原子操作完整能力

`AtomicRMWTkoOp`（`Ops.td:411`）：
- `mode`（`CudaTile_AtomicRMWModeAttr`，`AttrDefs.td:202-219`）：`AND,OR,XOR,ADD,ADDF,MAX,MIN,UMAX,UMIN,XCHG`（10 种，默认 ADD）。
- 支持的数据类型（`Ops.td:445-450` 文档）：
  - `ADD,AND,MAX,MIN,OR,UMAX,UMIN,XOR`: i32, i64
  - `ADDF`: f16, f32, f64
  - `XCHF`(sic, 原文档笔误应为 XCHG): i32, i64, f32, f64
- `memory_scope`（`CudaTile_MemoryScopeAttr`, `AttrDefs.td:412-424`）：`TL_BLK`(同一 tile block 内并发)、`DEVICE`(同一 GPU 内并发)、`SYS`(整个系统/所有设备)。
- `memory_ordering_semantics`：`AtomicRMWTkoOp`/`AtomicCASTkoOp` 都被约束为 `OnlyVariants<["RELAXED","ACQUIRE","RELEASE","ACQ_REL"]>`
  （`Ops.td:381,494`）——**不含 WEAK**（WEAK 只用于 load/store，语义是"假设无并发访问"，原子操作本身隐含有并发，所以不提供 WEAK）。
  → **DEVICE scope 下的 ACQUIRE/RELEASE/ACQ_REL 组合是被显式支持的**（这正是 E4 生产者-消费者跨 tile block 通信所需要的最弱合法组合）。
- `AtomicCASTkoOp`（`Ops.td:298`）：比较用**逐位相等**（不是 IEEE-754 语义，NaN 的不同 bit pattern 视为不同值）。

`AtomicRedViewTkoOp`：**旧版（可编译工具链）完全没有这个 op**（`grep -rn AtomicRedViewTkoOp /data/cuda-tile` 为空）。
只存在于新版（`.../tensor_ir_cuda_tile-src/.../Ops.td:633`，引入版本 "13.3"，与 AllocaOp 同批加入）。
⚠️ 未编译验证：只读到定义，没有可用二进制测试其具体行为。

结论：**✅ 已验证**（旧版可编译工具链）：`atomic_rmw_tko`/`atomic_cas_tko` 支持 DEVICE scope 下 RELAXED/ACQUIRE/RELEASE/ACQ_REL 的组合，
这是 E4 依赖的关键能力，语法/verifier 层面存在；具体能否在真实两个 tile block 间正确工作在 E4 中实测。

## Q6. 控制流

三种循环结构，都在 `Ops.td` 里：
- `ForOp`（`Ops.td:1493`）：结构化 range 循环，`lowerBound/upperBound/step` 都是 `CudaTile_ScalarTileOf<CudaTile_AnyInt>`
  即 **`tile<i32>` 的 SSA 值——可以是运行时值**（不要求是 ConstantOp 的结果，类型系统层面没有编译期常量约束）。
  循环体必须以 `continue`（携带下一轮 loop-carried 值）结束；不支持提前 `break`/`return`（"cannot terminate early"）。
  支持 loop-carried values（`iter_values`，见 `mlirExamples`），且携带值类型可以在循环体内**改变类型**
  （示例里 i32 carried 到最后 break 出 f32 —— 不过这是 LoopOp 的例子，ForOp 是 `AllTypesMatch<["initValues","resultValues"]>`，
  即 ForOp 的 loop-carried 值类型**前后必须一致**，而 LoopOp 没有这个约束，见下）。
- `LoopOp`（`Ops.td:2462`）：**无界**（"unstructured infinite loop"），靠 `break`/`continue` 控制何时退出——这就是 while(true) 的形式，
  是 E3/E4 spin-wait 的天然载体：`loop { %flag = load ...; if cond { break } }`。
  同样支持 `iter_values` 携带值，且**没有** ForOp 那个 AllTypesMatch 限制（`mlirExamples` 里演示了 i32→f32 的类型转变例子）。
  警告注明："Early returns from inside loops are not supported"（必须先 break 出循环再 return）。
- `IfOp`（`Ops.td:1972`）：`then`/可选 `else`，用 `yield` 产生结果；**如果不 yield 任何值，then/else 分支内部完全独立，没有类型匹配约束**
  （"If yielding value(s) the types of yielded values must match"——隐含"不 yield 就没有此约束"）。这对 E2 至关重要：
  不同分支内部可以使用完全不同形状/秩/类型的 tile，只要不通过 yield 把值带出 if 之外。
  唯一额外限制："Results of if must not be a tensor_view or view type"（if 不能直接返回 view 类型的结果，但可以返回普通 tile）。
- `BreakOp`/`ContinueOp`（`Ops.td:713`/`960`）：分别终止/继续所在的最内层 Loop/For。

结论：**✅ 已验证**（源码定义层面）：
1. LoopOp/ForOp 的边界值可以是运行时 SSA 值（这是 E3 persistent loop 的基础）。
2. 存在无界 while 形式（LoopOp），这是 E4 spin-wait 的基础。
3. loop-carried value（yield 语义，ForOp/LoopOp 叫 `continue`/`break` 带值）完整支持。

## Q7. 独立的 memory fence / barrier 算子

`grep -n "^def CudaTile_.*Fence\|Barrier\|Sync"` 在 Ops.td 全文（新旧两版）都是**空结果**——
**没有独立的 fence/barrier op**。跨线程/跨 tile block 的顺序保证完全通过：
1. `_tko` 后缀算子自带的 `memory_ordering_semantics` 属性（WEAK/RELAXED/ACQUIRE/RELEASE/ACQ_REL，因 op 而异，见 Q5 的
   `OnlyVariants` 清单：load 系列 `WEAK/RELAXED/ACQUIRE`，store 系列 `WEAK/RELAXED/RELEASE`，atomic 系列 `RELAXED/ACQUIRE/RELEASE/ACQ_REL`）；
2. token 数据流（见 Q8）。

结论：**❌ 不可行**（没有独立 fence op）—— 但 **⚠️ 受限地可行**（通过内存序属性组合表达等价语义，用法上更接近 C++11 atomic 而不是 CUDA 的 `__syncthreads()`/`membar`）。

## Q8. token 机制

`TokenType`（`Types.td:316`）:
> "Tokens are **not runtime values**. Their purpose is to explicitly represent ordering constraints between token-ordered
> operations executed within a tile."

即 token 是纯粹的**编译期 SSA 依赖边**，不对应任何寄存器/内存中的实际数据。`MakeTokenOp`（`Ops.td:3121`）造一个无依赖的新 token；
`JoinTokensOp`（`Ops.td:2108`）把多个 token 合并成一个新 token（"Token-ordered operations which consume the new token
will then be ordered with respect to all joined tokens"）。

`atomic_cas_tko` 文档原文（`Ops.td:330-332`）：
> "A token-ordered atomic compare-and-swap **is not constrained by program order**. **The compiler may reorder it**
> (i.e. place them earlier or later in program order) **unless constrained by tokens**."

`atomic_rmw_tko` 文档（`Ops.td:455-457`）用几乎相同的措辞重申了这一点。这直接回答了"要保证自旋等待正确性必须做什么"：
- 仅仅按 MLIR 文本顺序把 load/store/atomic 排列好**不足以**保证真实执行顺序或者编译器不做重排/合并——
  除非用 token 显式建立 `store 的 result_token` → `下一个 load 的 token=` 数据依赖，
  或者靠 `memory_ordering_semantics`（ACQUIRE/RELEASE）配合正确的 `memory_scope`（DEVICE，因为跨 tile block）
  去建立 happens-before 关系（ACQUIRE/RELEASE 是"语言层面"的保证，token 是"这个特定 IR 实例内，编译器不能把这几个 op 相对重排"的保证，
  两者互补：ordering-semantics 管跨线程可见性，token 管同一线程内、编译器能否把没有数据依赖的 _tko op 换序/合并/公共子表达式消除）。

结论：**✅ 已验证**（源码原文引用）：
- token 不是运行时值，纯编译期排序约束。
- **_tko 算子在没有 token 约束时，允许被编译器按需要重排**——这是 E4 中"自旋读取会不会被优化掉/提到循环外"必须认真测试的直接理论依据：
  如果 spin loop 里的 load 与循环外的其它内存操作之间没有 token 边、且只用 RELAXED/WEAK，编译器在原则上被允许把它当作可折叠的重复读。
  要保证正确性，从文档角度看，至少需要：(a) ACQUIRE ordering（不是 WEAK/RELAXED），
  (b) 循环体内的 load 不能被判定为"不变式"从而被 LICM 提到循环外——这在实践中通常靠"该 load 是循环唯一副作用来源 + ACQUIRE 语义暗示每次都可能观察到新值"来保证，
  但**cuda-tile 规范文本没有明确一句话保证"ACQUIRE 语义的 load 一定不会被 LICM"**——这是 E4 必须用真实 SASS dump 去实测验证的地方，
  不能仅凭文档下结论。

---

## Q1 补充验证：多 entry module 能否通过 verifier

见 `multi_entry_test.mlir` 与其编译日志（本目录）。
