# E3 结果：runtime `total_tiles` 的 persistent-loop（grid-stride loop）

## 结论速览

**✅ 完全成功。** cuda-tile 的 `for` 循环（`ForOp`）允许下界/上界/步长全部是运行时 SSA 值
（不要求编译期常量），这一点在 E1 Q6 的源码阅读阶段已经确认，本实验在实际编译+运行链路上
把它坐实：用 `get_tile_block_id`（起始偏移）/ `get_num_tile_blocks`（步长）/ 一个运行时 kernel
参数 `total_tiles`（上界）三者构造出标准的 CUDA "grid-stride loop"，一次性通过 verifier /
lowering / tileiras 汇编，实机运行 100 万个元素 0 处不匹配，且性能随 grid 大小变化的趋势符合预期。

## 文件

- `persistent_loop.mlir` — persistent 版本：`entry @persistent_loop(%total_tiles: tile<i32>, %out: tile<ptr<f32>>)`，
  用 `for %tile_idx in (%bx to %total_tiles, step %gx) : tile<i32> { ... }` 实现 grid-stride loop，
  每次迭代把 `float(tile_idx)*2.0` 写到 `out[tile_idx]`。
- `non_persistent.mlir` — 对照组：`entry @non_persistent(%out: tile<ptr<f32>>)`，无循环，
  每个 tile block 只处理 `out[blockId_x]` 这一个元素，需要用 `grid = 元素总数` 启动。
- `host_test.cpp` / `host_test` — 正确性校验（100 万元素全比对）+ 用 `cuEvent` 做的启动耗时对比。

## 1. Verifier（两个文件都是首次尝试即通过）

```
$ /data/cuda-tile/build/bin/cuda-tile-opt persistent_loop.mlir
```
退出码 0，规范化打印结果（节选，确认 for 的三个操作数都被解析为运行时 SSA 值而非常量）：
```
for %loopIdx in (%blockId_x to %arg0, step %gridSize_x) : tile<i32> {
  ...
  store_view_tko weak %1, %pview[%loopIdx] : ...
}
```
`non_persistent.mlir` 同样一次通过（见 `verify_np.log`）。

## 2/3. Lowering + tileiras 汇编

```
$ cuda-tile-translate persistent_loop.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o persistent_loop.tilebc   # exit=0
$ tileiras --gpu-name sm_120 persistent_loop.tilebc -o persistent_loop.cubin                                                              # exit=0
```
两步均无报错/警告（`translate.log`/`tileiras.log` 均为空）。`cuobjdump --dump-resource-usage`：
```
Function persistent_loop:
 REG:19 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:912
```
`non_persistent.mlir` 同样两步都 exit=0（见 `translate_np.log`/`tileiras_np.log`）。

## 4. 实机正确性

```
$ ./host_test
SM count = 170
[persistent, grid=170, total_tiles=1000000] mismatches=0 / 1000000
[non-persistent, grid=1000000] mismatches=0 / 1000000
```
两种写法（170 个 block 每个处理约 5882 个 tile / 100 万个 block 每个处理 1 个 tile）都 100% 正确，
完整原始输出见 `host_test.log`。

## 5. 性能趋势（非严格 benchmark，只是量级层面的健全性检查）

```
[timing] persistent (grid=170, total_tiles=1000000): 0.1104 ms / launch
[timing] non-persistent (grid=1000000): 0.4735 ms / launch
[timing] persistent (grid=10000, total_tiles=1000000, ~100 tiles/block): 0.0131 ms / launch
```
- persistent(grid=170) 比 non-persistent(grid=1000000) 快约 4.3 倍，方向上符合"减少 block 启动数量、
  用循环摊薄启动开销"这一 persistent-kernel 的核心动机。
- persistent(grid=10000) 反而比 persistent(grid=170) 快接近 8.4 倍——这说明单纯"grid 越小越好"是
  错误直觉：170 个 block 对 170 个 SM 意味着朴素假设下每个 SM 只放 1 个 block，如果硬件实际能在
  每个 SM 上同时驻留更多 block（E4 里查到的 `cuOccupancyMaxActiveBlocksPerMultiprocessor` 对另一个
  kernel给出的典型值是 8 blocks/SM），170 个 block 就远不足以充分利用 GPU 的并行能力，
  而 grid=10000 能更好地填满硬件的并发槽位、同时仍然享受到"每个 block 处理 100 个 tile 摊薄开销"
  的好处。**这不是一次严谨的 benchmark（没有 warm-up、没有多次独立测量取方差等），只作为"确实
  可以正常运行且性能量级合理"的健全性证据，不作为精确的性能结论使用。**

## 结论对 megakernel 可行性的意义

runtime 可变的 `total_tiles` + grid-stride persistent loop 这一 megakernel 的基础结构，
在 cuda-tile 上是**直接、无障碍可行的（✅ 已验证）**：不需要任何变通写法，`for` 循环的
运行时上界/步长这一特性开箱即用。唯一需要牢记的是（与 E4 死锁发现相呼应）：persistent kernel
的 grid 大小选择需要考虑真实可并发驻留的 block 数，而不是简单地等于 SM 数量——过小的 grid
可能显著欠用硬件并行度。
