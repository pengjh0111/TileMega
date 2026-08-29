# E7 结果：warp-specialization 可控性调查

## 结论速览

E7 的核心问题（"能否显式控制哪些 warp 做 MMA / 哪些做 TMA / 哪些做控制流，即 warp 角色分配"）
在 E1 阶段的源码检索已经给出了否定答案：**cuda-tile 的 IR 层面没有任何 warp 角色分配的显式机制**，
唯一相关的旋钮是 `optimization_hints.num_worker_warps_per_cta`（一个 entry 级标量，只能取 4 或 8），
且**这个字段只存在于 tensor-ir 通过 FetchContent 拉取的新版 cuda-tile 里，当前实际可独立编译、
本次调研全程使用的旧版 `/data/cuda-tile`（commit `8a775693`）里完全没有这个字段**。

本实验（E7）在 E1 的基础上做了两件新的、直接可验证的事：

1. **重新用当前实际在用的工具链（`/data/cuda-tile/build/bin/cuda-tile-opt`）直接尝试写
   `num_worker_warps_per_cta`，拿到一手的、精确的报错文本**（不是转述 E1 notes 里的旧记录）。
2. **检查是否有任何可用工具能实际测试新版 cuda-tile 里这个字段的行为**——结论是没有：
   新版 cuda-tile 在 tensor-ir 的构建产物里**没有构建出独立的 `cuda-tile-opt` 可执行文件**
   （只构建出了 `cuda-tile-tblgen`），而 `tensor_ir-opt` 本身**没有注册 `cuda_tile` 方言**
   （`--help` 显示 `Available Dialects: arith, builtin, func, nv_tensor_ir`，没有 `cuda_tile`），
   所以**没有任何一个当前可用的二进制能直接解析/验证带 `num_worker_warps_per_cta` 的 cuda_tile
   文本 IR**——这一点在 E1 阶段是通过 grep 源码推断的，本实验把它坐实为"确认没有可用工具"这一
   更强的结论。

三段式标注：
- IR 层没有 warp 角色分配机制（load/store 之外没有任何"warp 粒度"控制）：✅ 已验证（E1 阶段的
  `.td` 全文检索结果，本实验未推翻）
- 旧版（当前实际使用的）cuda-tile 完全不支持 `num_worker_warps_per_cta`：✅ 已验证
  （本实验用当前工具链实测复现了报错）
- 新版 cuda-tile 里这个字段实际运行时行为如何（是否真的能控制 warp 数、对性能/正确性有何影响）：
  ❌ 未能验证——没有可用的二进制工具能测试，如实标注为"无法验证"而非编造结论。

## 文件

- `entry_num_worker_warps.mlir` — 尝试在当前工具链下使用 `num_worker_warps_per_cta` 的最小复现文件。

## 1. 实测：当前工具链拒绝 `num_worker_warps_per_cta`

```mlir
cuda_tile.module @cuda_tile_module {
  entry @test_warps(%arg0: tile<ptr<f32>>) optimization_hints=<sm_120 = {num_cta_in_cga = 2, num_worker_warps_per_cta = 4}> {
    return
  }
}
```
```
$ /data/cuda-tile/build/bin/cuda-tile-opt entry_num_worker_warps.mlir
entry_num_worker_warps.mlir:5:123: error: custom op 'cuda_tile.entry' unknown param num_worker_warps_per_cta for sm_120
  entry @test_warps(%arg0: tile<ptr<f32>>) optimization_hints=<sm_120 = {num_cta_in_cga = 2, num_worker_warps_per_cta = 4}> {
                                                                                                                          ^
exit=1
```
`num_cta_in_cga=2` 部分被正常解析（说明其它字段解析没问题，是 `num_worker_warps_per_cta`
这个具体的 key 名不被识别），报错信息明确、可操作（"unknown param ... for sm_120"）。
这与 E1 notes.md（`/data/tensor_ir_test/megakernel_feasibility/experiments/E1_primitives/notes.md:107-108`）
"全仓库为空" 的 grep 结论完全吻合，本实验补充了一手的运行时报错证据。

## 2. 排查：有没有任何工具能测试新版 cuda-tile 里这个字段

### 2.1 新版 cuda-tile 没有构建出独立的 `cuda-tile-opt`

```
$ find /data/tensor-ir/build/_deps/tensor_ir_cuda_tile-build -name "cuda-tile-opt" -o -name "cuda-tile-tblgen"
/data/tensor-ir/build/_deps/tensor_ir_cuda_tile-build/tools/cuda-tile-opt        <- 只是构建目录（CMakeFiles/cmake_install.cmake），没有可执行文件
/data/tensor-ir/build/_deps/tensor_ir_cuda_tile-build/tools/cuda-tile-tblgen
$ ls /data/tensor-ir/build/_deps/tensor_ir_cuda_tile-build/bin/
cuda-tile-tblgen   <- 只有这一个二进制被实际构建出来
```
`tools/cuda-tile-opt/` 目录下只有 CMake 的中间产物，没有可执行文件——说明 tensor-ir 的构建脚本
只构建了新版 cuda-tile 所需的最小子集（library + tblgen，用于生成 tensor-ir 自己需要的头文件/表），
没有构建它自己的 `cuda-tile-opt` 命令行工具。这与之前（E1 阶段）"新版 cuda-tile 不能独立编译"的
结论一致，本实验进一步确认"即使作为 tensor-ir 的子依赖被构建，也没有产出可用于直接测试的 opt 工具"。

### 2.2 `tensor_ir-opt` 不认识 `cuda_tile` 方言

```
$ /data/tensor-ir/build/bin/tensor_ir-opt --help | head -5
OVERVIEW: Tensor IR optimizer test driver
Available Dialects: arith, builtin, func, nv_tensor_ir
```
只注册了 4 个方言，**没有 `cuda_tile`**——所以无法直接喂一段 `cuda_tile.entry ... optimization_hints=<... num_worker_warps_per_cta=4>` 的文本 IR 给它验证。`nv_tensor_ir` 到 `cuda_tile` 的 lowering
显然存在（tensor-ir 最终要生成 cuda-tile bytecode），但没有以"接受 cuda_tile 文本 IR 并验证"这种
形式暴露给命令行工具，本次调研没有找到绕过这一限制的办法。

## 3. 结论对 megakernel 可行性的意义（综合 E1 + E7）

1. **当前实际可用、本次调研全程使用的 cuda-tile 版本，在 IR 层完全没有 warp-specialization
   的控制手段**（无角色分配 op/attribute，`num_worker_warps_per_cta` 字段不存在）——
   这意味着 Mirage MPK / Event Tensor 论文里"部分 warp 做计算、部分 warp 做数据搬运/事件调度"
   这种细粒度 warp 角色划分，**在这个版本的 cuda-tile 上没有已知的直接表达方式**（✅ 已验证：
   本次工具链下不支持）。
2. 更新版本的 cuda-tile（tensor-ir 依赖的那个）声明了一个 `num_worker_warps_per_cta`（取值仅
   4 或 8）的 entry 级标量提示，**但本次调研没有可用工具能验证它的真实运行时效果**——如实标注为
   ❌ 未能验证，不代表它一定有效或一定无效，只是当前环境下无法判断。若未来需要确认这一点，
   需要找到（或自行构建）新版 cuda-tile 的独立 `cuda-tile-opt`/`tileiras` 工具链。
