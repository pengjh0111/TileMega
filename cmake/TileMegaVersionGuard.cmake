# ==============================================================================
# Dependency version guard
#
# Background (TILEMEGA_SKELETON.md, risk R10): in the R1 study environment the
# hand-written MLIR experiments used /data/cuda-tile @ 8a775693 (2026-01) while
# tensor-ir pulled af241704 (2026-07) through FetchContent. The two had
# different op and attribute sets, a latent trap running through the whole
# study report.
#
# TileMega's build model gives tensor-ir exclusive ownership of fetching
# cuda-tile, so the build tree naturally holds only one copy. This module
# verifies that the third_party/cuda-tile "pin submodule" really does match the
# commit tensor-ir will fetch, and that cuda-tile and tensor-ir agree on the
# LLVM revision (tensor-ir's pin file states explicitly that the two must be
# updated together).
#
# Any mismatch is a configure-time FATAL_ERROR rather than a runtime surprise.
# ==============================================================================
include_guard(GLOBAL)

# Extract the 40-character SHA of a pin variable from a CMake file.
# Note: the pin file puts the value on the line AFTER the variable name, so the
# whole file must be read; file(STRINGS ... REGEX) matches line by line and
# would never see it.
function(_tilemega_read_pin file var_name out_var)
  if(NOT EXISTS "${file}")
    message(FATAL_ERROR
      "TileMega: 找不到 ${file}。\n"
      "  submodule 可能没有初始化，执行：git submodule update --init --recursive")
  endif()
  file(READ "${file}" _content)
  # CMake regular expressions do not support {n} repetition, so match with +
  # and check the length separately. Both spellings must be handled:
  # tensor-ir's pin is quoted and on the next line, cuda-tile's is unquoted.
  if(NOT _content MATCHES "${var_name}[ \t\r\n]*\"?([0-9a-f]+)")
    message(FATAL_ERROR
      "TileMega: 在 ${file} 中未能抽出 ${var_name} 的 SHA。\n"
      "  上游可能改了 pin 文件的格式，需要同步更新本守卫。")
  endif()
  set(_sha "${CMAKE_MATCH_1}")
  string(LENGTH "${_sha}" _len)
  if(NOT _len EQUAL 40)
    message(FATAL_ERROR
      "TileMega: ${file} 里 ${var_name} 抽到的不是 40 位 SHA：'${_sha}'")
  endif()
  set(${out_var} "${_sha}" PARENT_SCOPE)
endfunction()

function(tilemega_assert_dependency_pins)
  set(_root "${CMAKE_CURRENT_SOURCE_DIR}")
  set(_ti   "${_root}/third_party/tensor-ir")
  set(_ct   "${_root}/third_party/cuda-tile")

  # --- pins declared by tensor-ir -------------------------------------------
  _tilemega_read_pin("${_ti}/cmake/TensorIRDependencyPins.cmake"
                     "TENSOR_IR_PINNED_CUDA_TILE_COMMIT" _ti_cuda_tile)
  _tilemega_read_pin("${_ti}/cmake/TensorIRDependencyPins.cmake"
                     "TENSOR_IR_PINNED_LLVM_COMMIT" _ti_llvm)

  # --- commit the cuda-tile submodule is actually checked out at ------------
  find_package(Git QUIET)
  if(NOT Git_FOUND)
    message(WARNING "TileMega: 找不到 git，跳过 cuda-tile submodule 版本校验")
    set(_ct_head "${_ti_cuda_tile}")
  else()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${_ct}" rev-parse HEAD
      OUTPUT_VARIABLE _ct_head OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR
        "TileMega: 无法读取 ${_ct} 的 HEAD。执行 git submodule update --init --recursive")
    endif()
  endif()

  # --- the LLVM revision cuda-tile pins for itself --------------------------
  _tilemega_read_pin("${_ct}/cmake/IncludeLLVM.cmake"
                     "LLVM_BUILD_COMMIT_HASH" _ct_llvm)

  # --- assertion 1: the two cuda-tile copies must be the same commit --------
  if(NOT _ct_head STREQUAL _ti_cuda_tile)
    message(FATAL_ERROR
      "TileMega: cuda-tile 版本不一致（风险 R10）。\n"
      "  third_party/cuda-tile HEAD          = ${_ct_head}\n"
      "  tensor-ir 将要拉取的 cuda-tile      = ${_ti_cuda_tile}\n"
      "  这两份的 op/属性集合可能不同，R1 调研已在此踩过坑。\n"
      "  修复：git -C third_party/cuda-tile checkout ${_ti_cuda_tile}\n"
      "  若确实要升级 cuda-tile，必须同时升级 tensor-ir 的 pin 文件并重测。")
  endif()

  # --- assertion 2: cuda-tile and tensor-ir must agree on LLVM --------------
  # From tensor-ir's own pin file:
  #   "Update the CUDA Tile and LLVM revisions together.
  #    The LLVM revision must match the compatibility pin for the selected
  #    CUDA Tile revision."
  if(NOT _ct_llvm STREQUAL _ti_llvm)
    message(FATAL_ERROR
      "TileMega: LLVM pin 不一致。\n"
      "  cuda-tile @ ${_ct_head} 要求 LLVM = ${_ct_llvm}\n"
      "  tensor-ir 声明的 LLVM          = ${_ti_llvm}\n"
      "  这两个必须成对更新，否则 cuda-tile 大概率编不过。")
  endif()

  set(TILEMEGA_CUDA_TILE_PIN "${_ct_head}" PARENT_SCOPE)
  set(TILEMEGA_LLVM_PIN      "${_ti_llvm}" PARENT_SCOPE)
  message(STATUS "TileMega: 依赖 pin 校验通过 "
                 "(cuda-tile ${_ct_head}, LLVM ${_ti_llvm})")
endfunction()
