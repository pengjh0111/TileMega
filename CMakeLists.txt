# ==============================================================================
# TileMega -- turning megakernel task partitioning, event generation and
# granularity selection into a solvable compilation problem on the CUDA Tile IR
# / TensorIR stack.
#
# Build model (see TILEMEGA_SKELETON.md section 4):
#   This project is the top level; third_party/tensor-ir is embedded via
#   add_subdirectory. tensor-ir owns fetching and building LLVM and CUDA Tile,
#   so the build tree contains exactly ONE LLVM and ONE cuda-tile, which
#   eliminates risk R10 by construction.
#   The third_party/cuda-tile submodule does not take part in the build; it is
#   the version pin and a readable copy. The consistency assertions below force
#   it to match the revision tensor-ir actually fetches.
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
# Options
# ------------------------------------------------------------------------------
option(TILEMEGA_BUILD_TESTS      "Build TileMega tests"                    ON)
option(TILEMEGA_ENABLE_ISL       "Build the ISL/barvinok analysis layer"   ON)
option(TILEMEGA_ENABLE_GPU_TESTS "Enable tests that require a live GPU"    ON)

# ------------------------------------------------------------------------------
# Dependency version consistency assertions -- closes risk R10 permanently
#
# The R1 study environment held two cuda-tile checkouts six months apart with
# differing op sets, which was a latent trap. Here that becomes a hard
# configure-time error.
# ------------------------------------------------------------------------------
include(TileMegaVersionGuard)
tilemega_assert_dependency_pins()

# ------------------------------------------------------------------------------
# TensorIR (which brings LLVM/MLIR and CUDA Tile with it)
# ------------------------------------------------------------------------------
set(TENSOR_IR_DOWNLOAD_LLVM ON CACHE BOOL "" FORCE)
set(TENSOR_IR_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/tensor-ir EXCLUDE_FROM_ALL)

# tensor-ir already appended the MLIR/LLVM cmake module paths to
# CMAKE_MODULE_PATH, but only inside its own directory scope. Fetch them again
# here so this project's add_mlir_* calls can find them.
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include paths for tensor-ir / cuda-tile do not propagate; see that module.
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
# CUDA (for the Runtime layer and GPU tests; no CUDA language is enabled, only
# the driver/runtime API is used)
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "TileMega: CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")

# MLIR/LLVM are built inside this tree via FetchContent, and their include paths
# likewise exist only in their own directory scope, so pull them in explicitly.
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
