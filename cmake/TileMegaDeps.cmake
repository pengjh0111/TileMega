# ==============================================================================
# 暴露依赖的头文件搜索路径。
#
# 为什么需要这个模块：tensor-ir 与 cuda-tile 都用**目录作用域**的
# include_directories() 声明自己的头文件路径，这种写法不会通过 target 的
# INTERFACE_INCLUDE_DIRECTORIES 传播出去。所以 TileMega 这边的 target 即使
# 链接了 NVTensorIR::Dialect / CudaTileDialect，也看不到它们的头文件。
#
# 这里把四个路径（两个工程 × 源码/生成各一）统一收集成一个变量。
# 生成目录（*.inc）必须一并加入，否则 tablegen 产物找不到。
# ==============================================================================
include_guard(GLOBAL)

function(tilemega_collect_dependency_includes out_var)
  set(_dirs)

  # --- tensor-ir ------------------------------------------------------------
  list(APPEND _dirs
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensor-ir/include"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party/tensor-ir/include")

  # --- cuda-tile：由 tensor-ir 通过 FetchContent 拉取，问 FetchContent 要路径，
  #     不要硬编码 _deps 下的目录名。
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
