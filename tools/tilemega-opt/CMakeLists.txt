# tilemega-opt：注册 nv_tensor_ir + cuda_tile 两个 dialect。
#
# 注意（骨架 §4）：上游 tensor_ir-opt 虽然链接了 cuda-tile 作为库，但它的
# registerDialects() 没有注册 cuda_tile dialect，读不了含 cuda_tile op 的 IR。
# TileMega 的 Codegen 输出正是两者混合的 IR，所以必须自建这个驱动。
set(LLVM_LINK_COMPONENTS Core Support AsmParser)

set(TILEMEGA_OPT_LIBS
  # TensorIR 侧
  NVTensorIR::Registration
  NVTensorIR::Dialect
  NVTensorIR::Transform
  NVTensorIR::CudaTilePipelines
  NVTensorIR::ToCudaTileConversion
  # CUDA Tile 侧 —— 上游 tensor_ir-opt 缺的就是这两个
  CudaTileDialect
  CudaTileTransforms
  # MLIR core
  MLIRIR
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
)

add_mlir_tool(tilemega-opt tilemega-opt.cpp DEPENDS ${TILEMEGA_OPT_LIBS})
target_link_libraries(tilemega-opt PRIVATE ${TILEMEGA_OPT_LIBS})
target_include_directories(tilemega-opt PRIVATE ${TILEMEGA_DEP_INCLUDE_DIRS})
llvm_update_compile_flags(tilemega-opt)
mlir_check_all_link_libraries(tilemega-opt)
