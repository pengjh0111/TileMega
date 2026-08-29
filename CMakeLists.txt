# ==============================================================================
# TileMega — 在 CUDA Tile IR / TensorIR 栈上把 megakernel 的任务划分、事件生成与
# 粒度选择变成可求解的编译问题。
#
# 构建模型（见 TILEMEGA_SKELETON.md §4）：
#   本工程是顶层，third_party/tensor-ir 以 add_subdirectory 方式嵌入。
#   tensor-ir 负责拉取并构建 LLVM 与 CUDA Tile，因此整个构建树里
#   **只有一份 LLVM、一份 cuda-tile**，从构造上消除风险 R10。
#   third_party/cuda-tile 这个 submodule 不参与构建，它是"钉子"与可读副本：
#   下面的一致性断言会强制它与 tensor-ir 实际拉取的版本相同。
# ==============================================================================
cmake_minimum_required(VERSION 3.24)

project(TileMega VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ------------------------------------------------------------------------------
# 选项
# ------------------------------------------------------------------------------
option(TILEMEGA_BUILD_TESTS      "Build TileMega tests"                    ON)
option(TILEMEGA_ENABLE_ISL       "Build the ISL/barvinok analysis layer"   ON)
option(TILEMEGA_ENABLE_GPU_TESTS "Enable tests that require a live GPU"    ON)

# ------------------------------------------------------------------------------
# 依赖版本一致性断言 —— 永久关闭风险 R10
#
# R1 调研环境里同时存在两份 commit 相差 6 个月的 cuda-tile（op 集不同），
# 是一个隐性坑。这里把它变成配置期的硬错误。
# ------------------------------------------------------------------------------
include(TileMegaVersionGuard)
tilemega_assert_dependency_pins()

# ------------------------------------------------------------------------------
# TensorIR（连带 LLVM/MLIR 与 CUDA Tile）
# ------------------------------------------------------------------------------
set(TENSOR_IR_DOWNLOAD_LLVM ON CACHE BOOL "" FORCE)
set(TENSOR_IR_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/tensor-ir EXCLUDE_FROM_ALL)

# tensor-ir 已经把 MLIR/LLVM 的 cmake 模块路径推进 CMAKE_MODULE_PATH，
# 但那是在它的目录作用域里。这里重新拿一次供本工程的 add_mlir_* 使用。
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# tensor-ir / cuda-tile 的头文件路径不会自动传播，见该模块的说明。
include(FetchContent)
include(TileMegaDeps)
tilemega_collect_dependency_includes(TILEMEGA_DEP_INCLUDE_DIRS)

# ------------------------------------------------------------------------------
# ISL / barvinok
# ------------------------------------------------------------------------------
if(TILEMEGA_ENABLE_ISL)
  include(TileMegaBarvinok)
  tilemega_find_barvinok()
endif()

# ------------------------------------------------------------------------------
# CUDA（Runtime 层与 GPU 测试用；不用 CUDA 语言，只用 driver/runtime API）
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "TileMega: CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")

# MLIR/LLVM 是通过 FetchContent 在本构建树里现建的，它们的头文件路径同样
# 只存在于自己的目录作用域，这里显式引入。
include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${TILEMEGA_DEP_INCLUDE_DIRS})

add_subdirectory(lib)
add_subdirectory(tools)

if(TILEMEGA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

message(STATUS "")
message(STATUS "==== TileMega configuration summary ====")
message(STATUS "  cuda-tile pin  : ${TILEMEGA_CUDA_TILE_PIN}")
message(STATUS "  LLVM pin       : ${TILEMEGA_LLVM_PIN}")
message(STATUS "  ISL layer      : ${TILEMEGA_ENABLE_ISL}")
message(STATUS "  GPU tests      : ${TILEMEGA_ENABLE_GPU_TESTS}")
message(STATUS "========================================")
