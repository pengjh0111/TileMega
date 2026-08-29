# tilemega-opt: registers both the nv_tensor_ir and cuda_tile dialects.
#
# Note (skeleton section 4): upstream tensor_ir-opt links cuda-tile as a
# library, but its registerDialects() does not register the cuda_tile dialect,
# so it cannot read IR containing cuda_tile ops. TileMega's codegen emits
# exactly that mixture, hence this separate driver.
set(LLVM_LINK_COMPONENTS Core Support AsmParser)

set(TILEMEGA_OPT_LIBS
  # TensorIR side
  NVTensorIR::Registration
  NVTensorIR::Dialect
  NVTensorIR::Transform
  NVTensorIR::CudaTilePipelines
  NVTensorIR::ToCudaTileConversion
  # CUDA Tile side -- these two are what upstream tensor_ir-opt is missing
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
