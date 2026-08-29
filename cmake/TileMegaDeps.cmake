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
      "TileMega: cannot obtain cuda-tile's FetchContent paths. "
      "Upstream may have renamed the content (it was tensor_ir_cuda_tile).")
  endif()
  list(APPEND _dirs
    "${tensor_ir_cuda_tile_SOURCE_DIR}/include"
    "${tensor_ir_cuda_tile_BINARY_DIR}/include")

  foreach(d IN LISTS _dirs)
    if(NOT EXISTS "${d}")
      message(FATAL_ERROR "TileMega: dependency include directory missing: ${d}")
    endif()
  endforeach()

  set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()
