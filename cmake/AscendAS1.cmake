project(TileMegaAscendAS1 VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TILEMEGA_ENABLE_MLIR ON)
set(TILEMEGA_ENABLE_ISL ON)
set(TILEMEGA_TARGET_ARCH ascend_3510)
set(AS1_WORK "/home/d50064638/junhuip/tilemega-as1" CACHE PATH "Personal AS-1 workspace")
set(AS1_HOST_PREFIX "${AS1_WORK}/envs/host" CACHE PATH "Private polyhedral dependencies")
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")
add_library(tilemega_isl_stack INTERFACE)
foreach(dep barvinok isl polylibgmp ntl gmpxx gmp)
  find_library(AS1_${dep}_LIB NAMES ${dep} PATHS "${AS1_HOST_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(tilemega_isl_stack INTERFACE "${AS1_${dep}_LIB}")
endforeach()
target_include_directories(tilemega_isl_stack SYSTEM INTERFACE "${AS1_HOST_PREFIX}/include")
add_library(tilemega::isl_stack ALIAS tilemega_isl_stack)
set_source_files_properties(lib/Analysis/ISLContext.cpp PROPERTIES
  INCLUDE_DIRECTORIES "${AS1_WORK}/third_party/isl-0.28;${AS1_WORK}/build/isl")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/tilemega/Dialect/CouplingGraph")
set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/CGDialect.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGDialect.h.inc
  -gen-dialect-decls -dialect=tmcg -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGDialect.cpp.inc
  -gen-dialect-defs -dialect=tmcg -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/ExecDialect.h.inc
  -gen-dialect-decls -dialect=tmexec -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/ExecDialect.cpp.inc
  -gen-dialect-defs -dialect=tmexec -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/CGAttrs.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGAttrs.h.inc
  -gen-attrdef-decls -attrdefs-dialect=tmcg -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGAttrs.cpp.inc
  -gen-attrdef-defs -attrdefs-dialect=tmcg -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/CGOps.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGOps.h.inc
  -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGOps.cpp.inc
  -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/ExecOps.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/ExecOps.h.inc
  -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/ExecOps.cpp.inc
  -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_public_tablegen_target(TileMegaCGIncGen)

add_library(tilemega_core STATIC
  lib/Solver/AscendSchedule.cpp
  lib/Solver/AscendPlan.cpp
  lib/Codegen/Ascend/AscendCodegen.cpp
  lib/Codegen/Ascend/AscendGraphCodegen.cpp
  lib/Codegen/Ascend/AscendEncoderCodegen.cpp
  lib/Support/Json.cpp
  lib/Target/Ascend/TargetSpec.cpp
  lib/Target/Ascend/Track.cpp
  lib/Frontend/Frontend.cpp
  lib/Frontend/GraphPattern.cpp
  lib/Frontend/ModelPlan.cpp
  lib/Frontend/DmDescriptorCodec.cpp
  lib/Frontend/SemanticLifting.cpp
  lib/Frontend/DnnSemanticLifting.cpp
  lib/Frontend/DnnStorage.cpp
  lib/Frontend/DnnModelPlan.cpp
  lib/Frontend/DnnDeferredNorm.cpp
  lib/Frontend/DnnResourceProbe.cpp
  lib/Frontend/DnnDwPwFusion.cpp
  lib/Frontend/MoeRegionPattern.cpp
  lib/Frontend/MoeRegionPlan.cpp
  lib/Frontend/MoeSemanticLifting.cpp
  lib/Frontend/ServingSemanticLifting.cpp
  lib/Analysis/SemanticCodec.cpp
  lib/Dialect/CouplingGraph/CGDialect.cpp
  lib/Dialect/CouplingGraph/PlacementPlan.cpp
  lib/Dialect/CouplingGraph/HandoffAccess.cpp
  lib/Dialect/CouplingGraph/HandoffPass.cpp
  lib/Dialect/CouplingGraph/FusionPass.cpp
  lib/Dialect/CouplingGraph/CGContract.cpp
  lib/Analysis/ClosedForm.cpp
  lib/Analysis/AffineRelation.cpp
  lib/Analysis/ISLContext.cpp
  lib/Analysis/QuasiPolynomial.cpp
  lib/Analysis/CouplingRelation.cpp
  lib/Analysis/Analysis.cpp
  lib/Analysis/AccessRelation.cpp
  lib/Analysis/SymbolicOracle.cpp
  lib/Analysis/CouplingCache.cpp
  lib/Analysis/Semantics.cpp
  lib/Analysis/TaskInstantiation.cpp
  lib/Analysis/TaskStorage.cpp
  lib/Analysis/CouplingDerivation.cpp
  lib/Analysis/DependencyForm.cpp
  lib/Analysis/ReferenceModels.cpp
  lib/Solver/ClusterLabeling.cpp
  lib/Solver/BackendCostQuery.cpp
  lib/Solver/CandidateGenerator.cpp
  lib/Solver/ImplementationContract.cpp
  lib/Solver/AlignmentPropagation.cpp
  lib/Solver/ChainDP.cpp
  lib/Solver/FusionIntervalDP.cpp
  lib/Solver/SymbolicChainDP.cpp
  lib/Solver/CouplingInterfaceDP.cpp
  lib/Solver/ScalarTaskWork.cpp
  lib/Solver/AttentionWork.cpp
  lib/Solver/ListScheduler.cpp
  lib/Solver/ExecutionSimulator.cpp
  lib/Solver/SkeletonSearch.cpp
  lib/Solver/DnnStructureSearch.cpp
  lib/Solver/SkeletonFinalize.cpp
  lib/Solver/SkeletonPlacement.cpp
  lib/Solver/PlanSkeleton.cpp
  lib/Solver/EftPlacement.cpp
  lib/Solver/ChainPlacement.cpp
  lib/Solver/PlanMaterialize.cpp
  lib/Solver/VariantSchedule.cpp
  lib/Solver/CostModel.cpp
  lib/Solver/SymbolicTaskCost.cpp
  lib/Solver/FusionResources.cpp
  lib/Solver/BalancedPlacement.cpp
  lib/Solver/LaneIntersections.cpp
  lib/Codegen/RuntimeTaskGraph.cpp
  lib/Codegen/RuntimeDependencyCodec.cpp
  lib/Solver/CacheServiceCurve.cpp
  lib/Solver/ModelDescription.cpp
  lib/Solver/MemoryPlan.cpp
  lib/Solver/TaskModel.cpp
  lib/Solver/BindingRequestTraffic.cpp
  lib/Solver/FusedTaskInput.cpp
  lib/Solver/RuntimeProjection.cpp
  lib/Solver/HandoffRuntimeProjection.cpp
  lib/Solver/FusedRuntimeProjection.cpp
  lib/Codegen/ExactRuntimeTaskGraph.cpp
  lib/Analysis/OpArithmetic.cpp
  lib/Analysis/TaskWork.cpp
  lib/Analysis/TaskElementRelation.cpp
  lib/Analysis/StorageHazards.cpp
  lib/Analysis/DependencyTable.cpp
  lib/Analysis/BoundDependencyForm.cpp
  lib/Analysis/CountedDependencyForm.cpp
  lib/Dialect/CouplingGraph/BoundDependency.cpp
  lib/Dialect/CouplingGraph/CountedDependency.cpp
  lib/Analysis/VirtualTaskBinding.cpp
  lib/Analysis/ExactTaskWork.cpp
  lib/Analysis/DramFloor.cpp
  lib/Solver/TaskPriceParts.cpp
  lib/Solver/PiecePricing.cpp
  lib/Solver/MoeTaskPricing.cpp
  lib/Solver/DramFluid.cpp
  lib/Solver/ModelDramFloor.cpp
  lib/Solver/MoeDramBindings.cpp
  lib/Solver/MoeStructureSearch.cpp
  lib/Solver/DmReductionFlow.cpp
  lib/Solver/FlowPreparation.cpp
  lib/Solver/StageFlowModel.cpp
  lib/Solver/FluidExecutionSimulator.cpp
  lib/Analysis/FusionAccess.cpp
  lib/Analysis/HandoffAccess.cpp
  lib/Codegen/Codegen.cpp
  lib/Codegen/ServingPages.cpp
)
add_library(TileMega::tilemega ALIAS tilemega_core)
add_dependencies(tilemega_core TileMegaCGIncGen)
target_compile_features(tilemega_core PUBLIC cxx_std_17)
target_include_directories(tilemega_core PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(tilemega_core SYSTEM PRIVATE "$ENV{ASCEND_HOME_PATH}/include"
  "$ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc"
  "$ENV{ASCEND_HOME_PATH}/x86_64-linux/pkg_inc/runtime")
target_link_directories(tilemega_core PUBLIC "$ENV{ASCEND_HOME_PATH}/lib64")
target_link_libraries(tilemega_core PUBLIC MLIRIR MLIRSupport MLIRPass tilemega::isl_stack ascendcl runtime platform dl)
target_compile_definitions(tilemega_core PRIVATE
  TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  TILEMEGA_ASCEND_AS1=1 TILEMEGA_PHYSICAL_WAIT_DOMAIN=1
  TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION=1 TILEMEGA_EARLY_QP_BINDING=1
  TILEMEGA_BALANCED_QP_SUM=1 TILEMEGA_OP_ARITHMETIC=1 TILEMEGA_TASK_WORK=1
  TILEMEGA_EXACT_ELEMENT_WORK=1 TILEMEGA_SEMANTIC_COST_INPUT=1
  TILEMEGA_DERIVED_TASK_COST=1 TILEMEGA_COMPLETE_GEMM_READS=1
  TILEMEGA_COMPLETE_NORMALIZATION_READS=1)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega_core PRIVATE -fno-rtti)
endif()
add_executable(as1_track_contracts EXCLUDE_FROM_ALL test/unit/ascend_track_test.cpp)
target_link_libraries(as1_track_contracts PRIVATE tilemega_core)
target_compile_options(as1_track_contracts PRIVATE -UNDEBUG)
add_executable(as1_isl_crosslink EXCLUDE_FROM_ALL test/unit/isl_crosslink_test.cpp)
target_link_libraries(as1_isl_crosslink PRIVATE tilemega_core MLIRPresburger)
add_executable(as1_compile_model EXCLUDE_FROM_ALL tools/as1_compile_model.cpp)
target_compile_definitions(as1_compile_model PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(as1_compile_model PRIVATE tilemega_core)
add_executable(as2_compile_llm EXCLUDE_FROM_ALL tools/as2_compile_llm.cpp)
target_compile_definitions(as2_compile_llm PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(as2_compile_llm PRIVATE tilemega_core)
add_executable(as2_rank_gemm EXCLUDE_FROM_ALL tools/as2_rank_gemm.cpp)
target_compile_definitions(as2_rank_gemm PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(as2_rank_gemm PRIVATE tilemega_core)
add_executable(as2_schedule_contracts EXCLUDE_FROM_ALL test/unit/ascend_schedule_test.cpp)
target_compile_options(as2_schedule_contracts PRIVATE -UNDEBUG)
target_compile_definitions(as2_schedule_contracts PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(as2_schedule_contracts PRIVATE tilemega_core)
add_executable(as1_target_contracts EXCLUDE_FROM_ALL test/unit/ascend_target_test.cpp)
target_compile_definitions(as1_target_contracts PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(as1_target_contracts PRIVATE -UNDEBUG)
target_link_libraries(as1_target_contracts PRIVATE tilemega_core)
