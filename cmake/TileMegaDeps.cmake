# ==============================================================================
# Expose the include search paths of our dependencies.
#
# Why this module is needed: tensor-ir and cuda-tile both declare their include
# paths with DIRECTORY-scoped include_directories(), which does not propagate
# through a target's INTERFACE_INCLUDE_DIRECTORIES. So even a TileMega target
# that links NVTensorIR::Dialect / CudaTileDialect cannot see their headers.
#
# This collects the four paths (two projects x source and generated) into one
# variable. The generated directories (*.inc) must be included as well, or the
# tablegen outputs are not found.
# ==============================================================================
include_guard(GLOBAL)

function(tilemega_collect_dependency_includes out_var)
  set(_dirs)

  # --- tensor-ir ------------------------------------------------------------
  list(APPEND _dirs
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensor-ir/include"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/tensor-ir/include")

  # --- cuda-tile: fetched by tensor-ir through FetchContent. Ask FetchContent
  #     for the paths rather than hard-coding directory names under _deps.
  FetchContent_GetProperties(tensor_ir_cuda_tile)
  if(NOT tensor_ir_cuda_tile_SOURCE_DIR)
    message(FATAL_ERROR
      "TileMega: 拿不到 cuda-tile 的 FetchContent 路径。"
      "上游可能改了 content 名字（原为 tensor_ir_cuda_tile）。")
  endif()
  list(APPEND _dirs
    "${tensor_ir_cuda_tile_SOURCE_DIR}/include"
    "${tensor_ir_cuda_tile_BINARY_DIR}/include")

  foreach(d IN LISTS _dirs)
    if(NOT EXISTS "${d}")
      message(FATAL_ERROR "TileMega: 依赖头文件目录不存在: ${d}")
    endif()
  endforeach()

  set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()
