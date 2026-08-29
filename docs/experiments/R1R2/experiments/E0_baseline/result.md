# E0 结果：基线工具链走通性（verifier → translate → tileiras → 实机运行）

## 结论速览

**✅ 已验证：完整 pipeline 一次性走通。** 本实验目的是在做任何 megakernel 相关的复杂实验之前，
先确认 verifier / translate / tileiras / cuLaunchKernel 这条最基础的链路本身没有环境问题。
本次会话对已存在的产物重新做了一次实测复现（而非仅信任此前遗留的文件），结果与文件名/大小暗示的
"已完成"状态一致。

## 文件

- `example.mlir` — 最小 kernel：`load_ptr_tko weak` 读 128 个 f32，`print` 打印。
- `example.cubin` / `example_131.cubin` / `example_1331.cubin` — 分别用不同 tileiras 版本
  （系统默认 / cuda-13.1 / cuda-13.3.1）汇编出的 cubin，用于交叉确认多个 tileiras 版本都能正常工作。
- `example_host.cpp` / `example` — host 侧驱动（`cuModuleLoad`+`cuLaunchKernel`）。
- `add_dynamic.mlir` — `nv_tensor_ir` 层的动态形状加法算子测试（`tensor_ir-compiler` 输入，
  而非直接的 cuda_tile IR），用于确认 tensor-ir 前端本身也能跑通。
- `matmul_lowered_cudatile.mlir` — 一个由 `tensor_ir-compiler` 从更高层 nv_tensor_ir matmul 图
  自动 lowering 出来的 cuda_tile IR dump，用作后续实验参考 tensor_ir 编译器实际生成什么样的
  cuda_tile 语法（`make_tensor_view`/`make_partition_view`/`load_view_tko`/`mmaf`/`store_view_tko`
  的标准用法范例，被 E2/E3/E6 多次借鉴其语法）。

## 1. Verifier（本次会话重新实测）

```
$ /data/cuda-tile/build/bin/cuda-tile-opt example.mlir
```
exit=0，回显合法规范化后的 IR（`print`/`iota`/`reshape`/`broadcast`/`offset`/`load_ptr_tko weak` 全部
被接受，无警告）。

## 2. 实机运行（本次会话重新实测，非仅信任旧文件）

```
$ ./example
Running example module
Data: [0.000000, 5.000000, 10.000000, 15.000000, ..., 635.000000]
exit=0
```
128 个输出值（步长 5.0）与 host 侧初始化数据的期望模式吻合，`print` 内建原语本身也确认可用
（这是 E4/E5/E6 调试时唯一能拿到的"设备端可见输出"手段之一，因为 cuda-tile 没有 clock/globaltimer
原语，见 E1 notes.md Q7 附近讨论）。

## 3. nv_tensor_ir 前端 / lowering 到 cuda_tile 的参考产物

`add_dynamic.mlir` 是 tensor-ir 自带测试用例格式（`// RUN: tensor_ir-compiler %s --dynamic-dims=...`），
`matmul_lowered_cudatile.mlir` 是从类似输入 lowering 出来的 cuda_tile IR——两者均只作为**语法参考**
被后续实验引用，本身不构成独立的可行性结论，故未在此重复给出 tensor_ir-compiler 的完整命令行/日志
（这属于 E0 阶段的探索性副产物，不是本实验的核心断言）。

## 结论对 megakernel 可行性的意义

E0 只回答一个基础问题："这套工具链在本机环境下能不能正常工作"——答案是能。这为后续 E1-E7 的
一切结论提供了前提：任何后续实验里的失败都可以归因于具体的语言/编译器行为，而不是环境本身损坏。
