# ==============================================================================
# 依赖版本一致性守卫
#
# 背景（TILEMEGA_SKELETON.md 风险 R10）：R1 调研环境里，手写 MLIR 实验用的是
# /data/cuda-tile @ 8a775693（2026-01），而 tensor-ir 通过 FetchContent 拉取的是
# af241704（2026-07）——两份 op/属性集合不同，是贯穿整份调研报告的隐性坑。
#
# TileMega 的构建模型让 tensor-ir 独占 cuda-tile 的拉取权，所以构建树里天然只有
# 一份。这个模块负责校验：third_party/cuda-tile 这个"钉子 submodule"确实和
# tensor-ir 实际会拉的 commit 一致，以及 cuda-tile 与 tensor-ir 对 LLVM 的
# 钉法一致（tensor-ir 的 pin 文件明确要求两者必须成对更新）。
#
# 任何一项不符 —— 配置期直接 FATAL_ERROR，而不是等到运行时踩坑。
# ==============================================================================
include_guard(GLOBAL)

# 从 CMake 文件中抽取一个 pin 变量的 40 位 SHA。
# 注意：pin 文件把值写在变量名的**下一行**，所以必须整文件读，
# 不能用 file(STRINGS ... REGEX)（那是逐行匹配的）。
function(_tilemega_read_pin file var_name out_var)
  if(NOT EXISTS "${file}")
    message(FATAL_ERROR
      "TileMega: 找不到 ${file}。\n"
      "  submodule 可能没有初始化，执行：git submodule update --init --recursive")
  endif()
  file(READ "${file}" _content)
  # CMake 的正则不支持 {n} 重复，只能匹配 + 再单独校验长度。
  # 两种写法都要吃：tensor-ir 的 pin 带引号且换行，cuda-tile 的不带引号。
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

  # --- tensor-ir 声明的 pin --------------------------------------------------
  _tilemega_read_pin("${_ti}/cmake/TensorIRDependencyPins.cmake"
                     "TENSOR_IR_PINNED_CUDA_TILE_COMMIT" _ti_cuda_tile)
  _tilemega_read_pin("${_ti}/cmake/TensorIRDependencyPins.cmake"
                     "TENSOR_IR_PINNED_LLVM_COMMIT" _ti_llvm)

  # --- cuda-tile submodule 实际 checkout 的 commit ---------------------------
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

  # --- cuda-tile 自己钉的 LLVM ----------------------------------------------
  _tilemega_read_pin("${_ct}/cmake/IncludeLLVM.cmake"
                     "LLVM_BUILD_COMMIT_HASH" _ct_llvm)

  # --- 断言 1：两份 cuda-tile 必须同 commit ---------------------------------
  if(NOT _ct_head STREQUAL _ti_cuda_tile)
    message(FATAL_ERROR
      "TileMega: cuda-tile 版本不一致（风险 R10）。\n"
      "  third_party/cuda-tile HEAD          = ${_ct_head}\n"
      "  tensor-ir 将要拉取的 cuda-tile      = ${_ti_cuda_tile}\n"
      "  这两份的 op/属性集合可能不同，R1 调研已在此踩过坑。\n"
      "  修复：git -C third_party/cuda-tile checkout ${_ti_cuda_tile}\n"
      "  若确实要升级 cuda-tile，必须同时升级 tensor-ir 的 pin 文件并重测。")
  endif()

  # --- 断言 2：cuda-tile 与 tensor-ir 对 LLVM 的钉法必须一致 ------------------
  # tensor-ir 的 pin 文件原话：
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
