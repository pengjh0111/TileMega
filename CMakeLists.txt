cmake_minimum_required(VERSION 3.22)

# Some CUDA container images expose versioned /usr/local/cuda-* directories
# without adding nvcc to PATH. Discover the compiler before enabling CUDA;
# CUDACXX and an explicit CMAKE_CUDA_COMPILER still take precedence.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  find_program(_tilemega_nvcc nvcc
    HINTS /usr/local/cuda/bin
    PATHS /usr/local
    PATH_SUFFIXES cuda/bin cuda-12.8/bin cuda-12.7/bin cuda-12.6/bin)
  if(_tilemega_nvcc)
    set(CMAKE_CUDA_COMPILER "${_tilemega_nvcc}" CACHE FILEPATH "CUDA compiler")
  endif()
endif()
project(TileMega VERSION 0.1.0 LANGUAGES C CXX CUDA)

option(TILEMEGA_BUILD_TESTS "Build unit and integration tests" ON)
option(TILEMEGA_BUILD_VERIFY "Build verification experiments" ON)
option(TILEMEGA_PHYSICAL_WAIT_DOMAIN
  "Count and persist couplings on physical producer domains" ON)
option(TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION "Build symbolic runtime queue counts" ON)
option(TILEMEGA_EARLY_QP_BINDING "Specialize bound QP parameter tokens before parsing" ON)
option(TILEMEGA_BALANCED_QP_SUM "Deduplicate and balance exact QP additions" ON)
option(TILEMEGA_OP_ARITHMETIC "Enable audited operator arithmetic declarations" ON)
option(TILEMEGA_TASK_WORK "Derive physical and nominal work from access maps" ON)
option(TILEMEGA_EXACT_ELEMENT_WORK "Count exact predicated element reads" ON)
option(TILEMEGA_SEMANTIC_COST_INPUT "Carry validated semantic accesses through CG" ON)
option(TILEMEGA_DERIVED_TASK_COST "Enable explicit access-derived task pricing" ON)
set_source_files_properties(lib/Frontend/Frontend.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_SEMANTIC_COST_INPUT=$<BOOL:${TILEMEGA_SEMANTIC_COST_INPUT}>")
set_source_files_properties(lib/Solver/CostModel.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_DERIVED_TASK_COST=$<BOOL:${TILEMEGA_DERIVED_TASK_COST}>")
option(TILEMEGA_COMPLETE_GEMM_READS "Include external GEMM weights in semantic reads" ON)
option(TILEMEGA_COMPLETE_NORMALIZATION_READS "Include normalization scale reads" ON)
set_source_files_properties(lib/Analysis/TaskWork.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_TASK_WORK=$<BOOL:${TILEMEGA_TASK_WORK}>;TILEMEGA_EXACT_ELEMENT_WORK=$<BOOL:${TILEMEGA_EXACT_ELEMENT_WORK}>")
set_source_files_properties(lib/Analysis/ReferenceModels.cpp lib/Frontend/SemanticLifting.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_COMPLETE_GEMM_READS=$<BOOL:${TILEMEGA_COMPLETE_GEMM_READS}>;TILEMEGA_COMPLETE_NORMALIZATION_READS=$<BOOL:${TILEMEGA_COMPLETE_NORMALIZATION_READS}>")
set_source_files_properties(lib/Analysis/OpArithmetic.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_OP_ARITHMETIC=$<BOOL:${TILEMEGA_OP_ARITHMETIC}>")
set_source_files_properties(lib/Analysis/QuasiPolynomial.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_EARLY_QP_BINDING=$<BOOL:${TILEMEGA_EARLY_QP_BINDING}>;TILEMEGA_BALANCED_QP_SUM=$<BOOL:${TILEMEGA_BALANCED_QP_SUM}>")
set_source_files_properties(lib/Solver/RuntimeProjection.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION=$<BOOL:${TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION}>")
set_source_files_properties(lib/Analysis/CouplingDerivation.cpp PROPERTIES
  COMPILE_DEFINITIONS "TILEMEGA_PHYSICAL_WAIT_DOMAIN=$<BOOL:${TILEMEGA_PHYSICAL_WAIT_DOMAIN}>")
option(TILEMEGA_ENABLE_MLIR "Build the required MLIR Coupling Graph contract" ON)
option(TILEMEGA_ENABLE_ISL
  "Build the required isl/barvinok coupling-derivation solver (Phase 3)" ON)
set(TILEMEGA_TARGET_ARCH "auto" CACHE STRING "sm_80|sm_89|sm_90|sm_100|sm_120|auto")
set_property(CACHE TILEMEGA_TARGET_ARCH PROPERTY STRINGS auto sm_80 sm_89 sm_90 sm_100 sm_120)
set(TILEMEGA_TARGET_CONFIG "" CACHE PATH "target json override")

if(NOT TILEMEGA_TARGET_ARCH STREQUAL "auto")
  if(NOT TILEMEGA_TARGET_ARCH MATCHES "^sm_(80|89|90|100|120)$")
    message(FATAL_ERROR "Unsupported TILEMEGA_TARGET_ARCH=${TILEMEGA_TARGET_ARCH}")
  endif()
  string(REGEX REPLACE "^sm_" "" _tilemega_cuda_arch "${TILEMEGA_TARGET_ARCH}")
  if(TILEMEGA_TARGET_ARCH STREQUAL "sm_90" OR TILEMEGA_TARGET_ARCH STREQUAL "sm_100")
    set(_tilemega_cuda_arch "${_tilemega_cuda_arch}a")
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "" FORCE)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${_tilemega_cuda_arch},code=sm_${_tilemega_cuda_arch}>)
  else()
    set(CMAKE_CUDA_ARCHITECTURES "${_tilemega_cuda_arch}" CACHE STRING "" FORCE)
  endif()
endif()

find_package(CUDAToolkit REQUIRED)
configure_file(cmake/BuildConfig.h.in include/tilemega/Support/BuildConfig.h @ONLY)

function(tilemega_configure_cuda_target target)
  if(TILEMEGA_TARGET_ARCH STREQUAL "auto")
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES OFF)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=native>)
  endif()
endfunction()

if(NOT TILEMEGA_ENABLE_MLIR)
  message(FATAL_ERROR
    "TILEMEGA_ENABLE_MLIR=OFF is no longer supported: the Coupling Graph "
    "dialect is the sole L4-L1 contract. Build MLIR and configure with "
    "-DMLIR_DIR=/path/to/lib/cmake/mlir (see docs/BUILD_MLIR.md).")
endif()
find_package(MLIR QUIET CONFIG)
if(NOT MLIR_FOUND)
  message(FATAL_ERROR
    "MLIR is required for the TileMega Coupling Graph dialect. Configure with "
    "-DMLIR_DIR=/path/to/lib/cmake/mlir. The pinned source/build recipe and "
    "commit are recorded in docs/BUILD_MLIR.md.")
endif()
message(STATUS "Using MLIR ${MLIR_PACKAGE_VERSION} from ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if(NOT TILEMEGA_ENABLE_ISL)
  message(FATAL_ERROR
    "TILEMEGA_ENABLE_ISL=OFF is no longer supported: isl/barvinok is the "
    "Analysis layer's sole coupling-derivation solver (skeleton principle "
    "3, docs/DEPENDENCIES.md). Build isl/polylib/barvinok (see "
    "docs/DEPENDENCIES.md) and reconfigure.")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ISL)
set_source_files_properties(lib/Analysis/ISLContext.cpp PROPERTIES
  INCLUDE_DIRECTORIES "${_tilemega_isl_src};${TILEMEGA_ISL_BUILD_DIR}")

# R8 BE-7: two dialects out of one -- `tmcg` for the graph's structure and
# `tmexec` for the decisions solved on it. The attributes stay with `tmcg`,
# which is where the structure they describe lives.
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
  lib/Support/Json.cpp
  lib/Target/TargetSpec.cpp
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
  lib/Solver/Sm90DensePlan.cpp
  lib/Solver/Sm90DenseSearchSpace.cpp
  lib/Solver/Sm90DenseCostModel.cpp
  lib/Solver/Sm90NativeConstants.cpp
  lib/Solver/Sm90DenseRegisterContract.cpp
  lib/Solver/Sm90DenseSearch.cpp
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
  lib/Runtime/EventBuffer.cpp
  lib/Runtime/MegakernelRuntime.cpp)
add_library(TileMega::tilemega ALIAS tilemega_core)
set_target_properties(tilemega_core PROPERTIES OUTPUT_NAME tilemega)
target_compile_features(tilemega_core PUBLIC cxx_std_17)
target_include_directories(tilemega_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/test>)
add_dependencies(tilemega_core TileMegaCGIncGen)
target_link_libraries(tilemega_core PUBLIC CUDA::cudart CUDA::cuda_driver MLIRIR MLIRSupport MLIRPass
  tilemega::isl_stack)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega_core PRIVATE -fno-rtti)
endif()

add_executable(tilemega-opt tools/tilemega-opt.cpp)
add_dependencies(tilemega-opt TileMegaCGIncGen)
target_link_libraries(tilemega-opt PRIVATE TileMega::tilemega MLIROptLib)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega-opt PRIVATE -fno-rtti)
endif()
set_target_properties(tilemega-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# The microbenchmarks live in their own library: GemmCalibration.cu
# instantiates six CUTLASS collectives, and putting that in `tilemega` would
# make every build of the core library pay for it.
add_library(tilemega_calibration STATIC
  lib/Target/Calibration.cu
  lib/Target/InflightCalibration.cu
  lib/Target/PrefetchCalibration.cu
  lib/Target/ServingTaskCalibration.cu
  lib/Target/ServingPagedCalibration.cu
  lib/Target/ServingSyncCalibration.cu
  lib/Target/HopperCalibration.cu
  lib/Target/GemmCalibration.cu)
target_link_libraries(tilemega_calibration PUBLIC TileMega::tilemega)
set_target_properties(tilemega_calibration PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(tilemega_calibration PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# Calibration measures the machine it runs on, so it is always built for the
# device present rather than the project-wide default architecture.
if(TILEMEGA_TARGET_ARCH STREQUAL "auto")
  set_property(TARGET tilemega_calibration PROPERTY CUDA_ARCHITECTURES OFF)
  target_compile_options(tilemega_calibration PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=native>)
endif()

# Independent loading probes share only architecture and async primitives.
add_executable(tilemega-loadbench EXCLUDE_FROM_ALL tools/experimental/loadbench/main.cu)
target_include_directories(tilemega-loadbench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include)
target_link_libraries(tilemega-loadbench PRIVATE CUDA::cudart)
set_target_properties(tilemega-loadbench PROPERTIES CUDA_STANDARD 17)
target_compile_options(tilemega-loadbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
tilemega_configure_cuda_target(tilemega-loadbench)

add_test(NAME scalar_cost_errors COMMAND tilemega probe scalar-error-probe "${CMAKE_SOURCE_DIR}")

add_test(NAME coupling_interface_gqa2 COMMAND tilemega probe interface-probe "${CMAKE_SOURCE_DIR}" gqa2)
add_test(NAME coupling_interface_mha4 COMMAND tilemega probe interface-probe "${CMAKE_SOURCE_DIR}" mha4)

# EX-S1's driver lives with its experiment rather than in tools/, which R2's
# scope fence excludes; it needs the frontend to price tasks, so it is a target
# here instead of an nvcc one-off inside the runner.

add_custom_target(check-policy
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckPolicy.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Checking architecture-dispatch and hardware-constant policy")

if(TILEMEGA_BUILD_TESTS)
  enable_testing()
  add_executable(serving_task_order_test test/unit/serving_task_order_test.cpp)
  target_include_directories(serving_task_order_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(serving_task_order_test PRIVATE -UNDEBUG)
  add_test(NAME serving_task_order COMMAND serving_task_order_test)
  add_executable(serving_shared_memory_budget_test test/unit/shared_memory_budget_test.cpp)
  target_include_directories(serving_shared_memory_budget_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(serving_shared_memory_budget_test PRIVATE cxx_std_17)
  add_test(NAME serving_shared_memory_budget COMMAND serving_shared_memory_budget_test)
  add_executable(conv_weight_extent_test test/unit/conv_weight_extent_test.cpp)
  target_link_libraries(conv_weight_extent_test PRIVATE TileMega::tilemega)
  target_compile_options(conv_weight_extent_test PRIVATE -UNDEBUG)
  add_test(NAME conv_weight_extent COMMAND conv_weight_extent_test)
  add_executable(serving_residency_budget_test test/unit/serving_residency_budget_test.cpp)
  target_link_libraries(serving_residency_budget_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_residency_budget_test PRIVATE -UNDEBUG)
  add_test(NAME serving_residency_budget COMMAND serving_residency_budget_test)
  add_executable(cluster_split_plan_test test/unit/cluster_split_plan_test.cpp)
  target_link_libraries(cluster_split_plan_test PRIVATE TileMega::tilemega)
  target_compile_options(cluster_split_plan_test PRIVATE -UNDEBUG)
  add_test(NAME cluster_split_plan COMMAND cluster_split_plan_test)
  add_executable(dependency_interval_pool_test test/unit/dependency_interval_pool_test.cpp)
  target_link_libraries(dependency_interval_pool_test PRIVATE TileMega::tilemega)
  target_compile_options(dependency_interval_pool_test PRIVATE -UNDEBUG)
  add_test(NAME dependency_interval_pool COMMAND dependency_interval_pool_test)
  add_executable(cluster_placement_test test/unit/cluster_placement_test.cpp)
  target_include_directories(cluster_placement_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(cluster_placement_test PRIVATE -UNDEBUG)
  add_test(NAME cluster_placement COMMAND cluster_placement_test)
  add_executable(serving_implementation_test test/unit/serving_implementation_test.cpp)
  target_link_libraries(serving_implementation_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_implementation_test PRIVATE -UNDEBUG)
  add_test(NAME serving_implementation COMMAND serving_implementation_test)
  add_executable(serving_seed_geometry_test test/unit/serving_seed_geometry_test.cpp)
  target_link_libraries(serving_seed_geometry_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_seed_geometry_test PRIVATE -UNDEBUG)
  add_test(NAME serving_seed_geometry COMMAND serving_seed_geometry_test)
  add_executable(serving_prefetch_policy_test test/unit/serving_prefetch_policy_test.cpp)
  target_link_libraries(serving_prefetch_policy_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_prefetch_policy_test PRIVATE -UNDEBUG)
  add_test(NAME serving_prefetch_policy COMMAND serving_prefetch_policy_test)
  add_executable(dm_linear_address_test test/unit/dm_linear_address_test.cpp)
  target_include_directories(dm_linear_address_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(dm_linear_address_test PRIVATE TILEMEGA_DM_LINEAR_ADDRESS=1)
  target_compile_options(dm_linear_address_test PRIVATE -UNDEBUG)
  add_test(NAME dm_linear_address COMMAND dm_linear_address_test)
  add_executable(attention_partition_test test/unit/attention_partition_test.cpp)
  target_include_directories(attention_partition_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(attention_partition_test PRIVATE -UNDEBUG)
  add_test(NAME attention_partition COMMAND attention_partition_test)
  add_executable(dm_reduction_plan_test test/unit/dm_reduction_plan_test.cpp)
  target_link_libraries(dm_reduction_plan_test PRIVATE TileMega::tilemega CUDA::cudart)
  target_compile_options(dm_reduction_plan_test PRIVATE -UNDEBUG)
  add_test(NAME dm_reduction_plan COMMAND dm_reduction_plan_test)
  add_executable(moe_runtime_operands_test test/unit/moe_runtime_operands_test.cpp)
  target_link_libraries(moe_runtime_operands_test PRIVATE TileMega::tilemega CUDA::cudart)
  target_compile_options(moe_runtime_operands_test PRIVATE -UNDEBUG)
  add_test(NAME moe_runtime_operands COMMAND moe_runtime_operands_test)
  add_executable(tilemega-unit test/unit/runner.cpp
    test/unit/target_spec_test.cpp
    test/unit/coupling_types_test.cpp
    test/unit/table27_test.cpp
    test/unit/semantics_test.cpp
    test/unit/dram_floor_test.cpp
    test/unit/regime_a_price_test.cpp
    test/unit/stage_flow_test.cpp
    test/unit/attention_page_layout_test.cpp
    test/unit/serving_attention_legality_test.cpp
    test/unit/flow_runtime_release_test.cpp
    test/unit/incidence_test.cpp
    test/unit/containment_test.cpp
    test/unit/event_synthesis_test.cpp
    test/unit/isl_relation_test.cpp
    test/unit/runtime_projection_test.cpp
    test/unit/relation_bounds_test.cpp
    test/unit/task_element_work_test.cpp
    test/unit/window_task_access_test.cpp
    test/unit/exact_task_metadata_test.cpp
    test/unit/dynamic_task_cursor_test.cpp
    test/unit/moe_opaque_control_test.cpp
    test/unit/storage_hazards_test.cpp
    test/unit/bound_dependency_form_test.cpp
    test/unit/capacity_form_test.cpp
    test/unit/virtual_task_binding_test.cpp
    test/unit/runtime_dependency_table_test.cpp
    test/unit/bound_dependency_codegen_test.cpp
    test/unit/dm_page_layout_test.cpp
    test/unit/moe_binding_test.cpp
    test/unit/graph_pattern_test.cpp
    test/unit/model_plan_order_test.cpp
    test/unit/layout_bridge_test.cpp
    test/unit/cg_attr_test.cpp
    test/unit/backend_query_test.cpp
    test/unit/chain_dp_test.cpp
    test/unit/cache_service_curve_test.cpp
    test/unit/fusion_access_test.cpp
    test/unit/handoff_runtime_projection_test.cpp
    test/unit/handoff_ir_test.cpp
    test/unit/handoff_access_test.cpp
    test/unit/lane_intersections_test.cpp
    test/unit/fusion_rewrite_test.cpp
    test/unit/stage_kappa_test.cpp
    test/unit/pipeline_sigma_test.cpp
    test/unit/fusion_written_price_test.cpp
    test/unit/fused_runtime_projection_test.cpp
    test/unit/exact_runtime_task_graph_test.cpp
    test/unit/fusion_batch_rewrite_test.cpp
    test/unit/fiber_sum_test.cpp
    test/unit/list_scheduler_test.cpp
    test/unit/plan_contract_test.cpp
    test/unit/execution_simulator_test.cpp
    test/unit/eft_placement_test.cpp
    test/unit/chain_placement_test.cpp
    test/unit/plan_table_test.cpp
    test/unit/cluster_labeling_test.cpp
    test/unit/alignment_propagation_test.cpp
    test/unit/impl_contract_test.cpp
    test/unit/serving_pruning_test.cpp
    test/unit/serving_lag_test.cpp
    test/unit/serving_task_index_test.cpp
    test/unit/serving_model_plan_test.cpp
    test/unit/serving_token_sets_test.cpp
    test/unit/serving_import_test.cpp
    test/unit/forward_frontend_test.cpp
    test/unit/frontend_import_test.cpp
    test/unit/dm_descriptor_test.cpp
    test/unit/skeleton_placement_test.cpp
    test/unit/isolated_evaluation_test.cpp
    test/unit/skeleton_search_isolation_test.cpp
    test/unit/plan_skeleton_test.cpp
    test/unit/variant_resource_test.cpp
    test/unit/symbolic_oracle_test.cpp
    test/unit/operator_classes_test.cpp
    test/unit/moe_routing_profile_test.cpp
    test/unit/moe_task_pricing_test.cpp
    test/unit/moe_dram_bindings_test.cpp
    test/unit/moe_structure_search_test.cpp
    test/unit/moe_group_profile_test.cpp
    test/unit/dm_gemm_traits_test.cpp
    test/unit/dm_gemm_candidates_test.cpp
    test/unit/conv_iteration_test.cpp
    test/unit/conv_reduction_partition_test.cpp
    test/unit/conv_reduction_cg_test.cpp
    test/unit/dnn_semantic_lifting_test.cpp
    test/unit/dnn_epilogue_semantics_test.cpp
    test/unit/dnn_model_plan_test.cpp
    test/unit/dnn_deferred_ln_test.cpp
    test/unit/dnn_dwpw_fusion_test.cpp
    test/unit/dnn_memory_codegen_test.cpp
    test/unit/moe_region_pattern_test.cpp
    test/unit/moe_region_plan_test.cpp
    test/unit/moe_semantic_lifting_test.cpp
    test/unit/moe_region_codegen_test.cpp
    test/unit/task_storage_test.cpp
    test/unit/dm_counted_fusion_test.cpp
    test/unit/dm_gemm_class_domain_test.cpp
    test/unit/dm_forward_search_test.cpp
    test/unit/dm_physical_footprint_test.cpp
    test/unit/dnn_structure_search_test.cpp
    test/unit/task_arithmetic_domain_test.cpp
    test/unit/dm_store_geometry_test.cpp
    test/unit/dm_exact_fusion_test.cpp
    test/unit/memory_plan_test.cpp
    test/unit/dm_fused_dependencies_test.cpp
    test/unit/dm_dram_images_test.cpp
    test/unit/binding_request_traffic_test.cpp
    test/unit/binding_request_pricing_test.cpp
    test/unit/typed_affine_pricing_test.cpp
    test/unit/dm_virtual_gemm_partition_test.cpp
    test/unit/dm_semantic_signature_test.cpp
    test/unit/dm_virtual_frontend_test.cpp
    test/unit/dm_piece_pricing_test.cpp
    test/unit/dm_target_memory_test.cpp
    test/unit/dm_page_resources_test.cpp
    test/unit/counted_dependency_cg_test.cpp
    test/unit/runtime_dependency_codec_test.cpp
    test/unit/counted_threshold_test.cpp
    test/unit/counted_write_test.cpp
    test/unit/coupling_cache_test.cpp
    test/unit/semantic_lifting_test.cpp
    test/unit/embedding_plan_test.cpp
    test/unit/wiring_coupling_test.cpp
    test/unit/degradation_test.cpp
)
  target_link_libraries(tilemega-unit PRIVATE TileMega::tilemega MLIRParser)
  add_dependencies(tilemega-unit TileMegaCGIncGen)
  target_compile_definitions(tilemega-unit PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(tilemega-unit PRIVATE -UNDEBUG)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(tilemega-unit PRIVATE -fno-rtti)
  endif()

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  # The lit driver must never be located through a path inside a third_party
  # submodule: such a path is not part of the submodule's tracked content, so it
  # only resolves on a machine where someone created it by hand. LLVMConfig
  # exports LLVM_BUILD_MAIN_SRC_DIR, which points at the LLVM source tree that
  # produced the MLIR package found above; derive the driver from that instead.
  find_file(TILEMEGA_LIT_DRIVER lit.py
    HINTS
      "${LLVM_BUILD_MAIN_SRC_DIR}/utils/lit"
      "${LLVM_TOOLS_BINARY_DIR}/../utils/lit")
  if(NOT TILEMEGA_LIT_DRIVER)
    message(FATAL_ERROR
      "Could not locate lit.py from the MLIR package at ${MLIR_DIR}. Pass "
      "-DTILEMEGA_LIT_DRIVER=/path/to/llvm/utils/lit/lit.py, or configure "
      "against an MLIR package whose LLVMConfig exports LLVM_BUILD_MAIN_SRC_DIR "
      "(see docs/BUILD_MLIR.md).")
  endif()
  configure_file(test/lit.site.cfg.py.in test/lit.site.cfg.py @ONLY)
  add_custom_target(check-cg-lit
    COMMAND ${Python3_EXECUTABLE} ${TILEMEGA_LIT_DRIVER}
      -sv ${CMAKE_CURRENT_BINARY_DIR}/test
    DEPENDS tilemega-opt
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  # Also a ctest case, so a plain `ctest` run covers the dialect's textual
  # form. lit is given the same directory the custom target uses; ctest does
  # not build tilemega-opt, so the suite fails loudly rather than silently
  # skipping when the tool is stale.
  add_test(NAME cg_lit
    COMMAND ${Python3_EXECUTABLE} ${TILEMEGA_LIT_DRIVER}
      -sv ${CMAKE_CURRENT_BINARY_DIR}/test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_test(NAME target_spec COMMAND tilemega-unit target_spec)
  add_test(NAME target_audit COMMAND tilemega audit target)

  add_test(NAME coupling_types COMMAND tilemega-unit coupling_types)

  add_test(NAME table27 COMMAND tilemega-unit table27)

  add_test(NAME stage_flow COMMAND tilemega-unit stage_flow)
  add_test(NAME attention_page_layout COMMAND tilemega-unit attention_page_layout)
  add_test(NAME serving_attention_legality COMMAND tilemega-unit serving_attention_legality)

  add_test(NAME flow_runtime_release COMMAND tilemega-unit flow_runtime_release)

  add_test(NAME regime_a_price COMMAND tilemega-unit regime_a_price
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/targets/solver_r9b.json)

  add_test(NAME dram_floor COMMAND tilemega-unit dram_floor)

  add_test(NAME semantics COMMAND tilemega-unit semantics)

  add_test(NAME incidence_identity COMMAND tilemega-unit incidence_identity)
  add_test(NAME imported_incidence_identity
    COMMAND tilemega probe event-cost ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(NAME containment COMMAND tilemega-unit containment)

  add_test(NAME event_synthesis COMMAND tilemega-unit event_synthesis)

  add_test(NAME runtime_projection COMMAND tilemega-unit runtime_projection)

  add_test(NAME relation_bounds COMMAND tilemega-unit relation_bounds)

  add_test(NAME graph_pattern COMMAND tilemega-unit graph_pattern)
  add_test(NAME window_task_access COMMAND tilemega-unit window_task_access)
  add_test(NAME exact_task_metadata COMMAND tilemega-unit exact_task_metadata)
  add_test(NAME storage_hazards COMMAND tilemega-unit storage_hazards)
  add_test(NAME bound_dependency_form COMMAND tilemega-unit bound_dependency_form)
  add_test(NAME capacity_form COMMAND tilemega-unit capacity_form)
  add_test(NAME virtual_task_binding COMMAND tilemega-unit virtual_task_binding)
  add_test(NAME runtime_dependency_table COMMAND tilemega-unit runtime_dependency_table)
  add_test(NAME bound_dependency_codegen COMMAND tilemega-unit bound_dependency_codegen)
  add_test(NAME dm_page_layout COMMAND tilemega-unit dm_page_layout)
  add_test(NAME moe_binding COMMAND tilemega-unit moe_binding)

  add_test(NAME task_element_work COMMAND tilemega-unit task_element_work
    ${CMAKE_SOURCE_DIR}/test/fixtures/exports/seqscan/gqa2.json
    ${CMAKE_SOURCE_DIR}/test/fixtures/exports/seqscan/mha4.json)
  add_test(NAME op_arithmetic_audit COMMAND tilemega audit op)
  add_test(NAME task_work_domains COMMAND tilemega probe task-work-probe
    "${CMAKE_SOURCE_DIR}/test/fixtures/exports/seqscan/gqa2.json"
    "${CMAKE_SOURCE_DIR}/test/fixtures/exports/seqscan/mha4.json")

  add_test(NAME isl_relation COMMAND tilemega-unit isl_relation)

  add_test(NAME layout_bridge COMMAND tilemega-unit layout_bridge)

  add_test(NAME cg_attr_roundtrip COMMAND tilemega-unit cg_attr_roundtrip)

  add_test(NAME backend_query COMMAND tilemega-unit backend_query)

  add_test(NAME cache_service_curve COMMAND tilemega-unit cache_service_curve)

  add_test(NAME fusion_access COMMAND tilemega-unit fusion_access)
  add_test(NAME handoff_runtime_projection COMMAND tilemega-unit handoff_runtime_projection)
  add_test(NAME handoff_ir COMMAND tilemega-unit handoff_ir)
  add_test(NAME serving_argmax_tile_import COMMAND tilemega-unit handoff_ir
    argmax_tile_import ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/serving_split.json)
  foreach(split_case IN ITEMS nonpaged_attention nonpaged_combine four one off escape)
    add_test(NAME handoff_split_${split_case} COMMAND tilemega-unit handoff_ir
      splitk_import ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/serving_split.json
      ${split_case} ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/targets/serving_r12b.json)
  endforeach()
  add_test(NAME handoff_access COMMAND tilemega-unit handoff_access)
  foreach(split_case IN ITEMS four one off escape)
    add_test(NAME handoff_nonpaged_split_${split_case} COMMAND tilemega-unit handoff_ir
      nonpaged_splitk_import ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/serving_split.json
      ${split_case} ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/targets/serving_r12b.json)
  endforeach()

  add_test(NAME lane_intersections COMMAND tilemega-unit lane_intersections)

  add_test(NAME fusion_rewrite COMMAND tilemega-unit fusion_rewrite)

  add_test(NAME stage_kappa COMMAND tilemega-unit stage_kappa)

  add_test(NAME pipeline_sigma COMMAND tilemega-unit pipeline_sigma)

  add_test(NAME fusion_written_price COMMAND tilemega-unit fusion_written_price)

  add_test(NAME fused_runtime_projection COMMAND tilemega-unit fused_runtime_projection)

  add_test(NAME exact_runtime_task_graph COMMAND tilemega-unit exact_runtime_task_graph)

  add_test(NAME fusion_batch_rewrite COMMAND tilemega-unit fusion_batch_rewrite)

  add_test(NAME fiber_sum COMMAND tilemega-unit fiber_sum)
  add_test(NAME attention_work COMMAND tilemega probe attention-work "${CMAKE_CURRENT_SOURCE_DIR}")

  add_test(NAME chain_dp COMMAND tilemega-unit chain_dp)

  add_test(NAME list_scheduler COMMAND tilemega-unit list_scheduler)

  add_test(NAME plan_contract COMMAND tilemega-unit plan_contract)

  add_test(NAME execution_simulator COMMAND tilemega-unit execution_simulator)

  add_test(NAME eft_placement COMMAND tilemega-unit eft_placement)

  add_test(NAME chain_placement COMMAND tilemega-unit chain_placement)

  add_test(NAME plan_table COMMAND tilemega-unit plan_table)

  add_test(NAME cluster_labeling COMMAND tilemega-unit cluster_labeling)

  add_test(NAME alignment_propagation COMMAND tilemega-unit alignment_propagation)

  add_test(NAME impl_contract COMMAND tilemega-unit impl_contract)

  add_executable(task_body_contract_test test/unit/task_body_contract_test.cu)
  add_executable(dm_conv_operands_test test/unit/dm_conv_operands_test.cu)
  target_link_libraries(dm_conv_operands_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_conv_operands_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_conv_operands_test)
  add_test(NAME dm_conv_operands COMMAND dm_conv_operands_test)
  add_executable(dm_conv_task_test test/unit/dm_conv_task_test.cu)
  target_link_libraries(dm_conv_task_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_conv_task_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_conv_task_test)
  add_test(NAME dm_conv_task COMMAND dm_conv_task_test)

  add_executable(dm_dnn_runtime_test test/unit/dm_dnn_runtime_test.cu)
  target_link_libraries(dm_dnn_runtime_test PRIVATE TileMega::tilemega)
  target_compile_definitions(dm_dnn_runtime_test PRIVATE TILEMEGA_DM_REDUCTION_DEPTH=1)
  target_compile_options(dm_dnn_runtime_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG;--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_dnn_runtime_test)
  add_test(NAME dm_dnn_runtime COMMAND dm_dnn_runtime_test)
  add_executable(dm_pool_depth_paged_test test/unit/dm_dnn_runtime_test.cu)
  target_link_libraries(dm_pool_depth_paged_test PRIVATE TileMega::tilemega)
  target_compile_definitions(dm_pool_depth_paged_test PRIVATE
    DM_TEST_PAGED=1 TILEMEGA_DM_REDUCTION_DEPTH=1)
  target_compile_options(dm_pool_depth_paged_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_pool_depth_paged_test)
  add_test(NAME dm_pool_depth_paged COMMAND dm_pool_depth_paged_test)
  add_executable(dm_norm_embedding_layout_test test/unit/dm_norm_embedding_layout_test.cu)
  target_link_libraries(dm_norm_embedding_layout_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_norm_embedding_layout_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_norm_embedding_layout_test)
  add_test(NAME dm_norm_embedding_layout COMMAND dm_norm_embedding_layout_test)
  add_executable(dm_pool_global_test test/unit/dm_pool_global_test.cu)
  target_link_libraries(dm_pool_global_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_pool_global_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_pool_global_test)
  add_test(NAME dm_pool_global COMMAND dm_pool_global_test)
  add_executable(dm_depthwise_test test/unit/dm_depthwise_test.cu)
  target_link_libraries(dm_depthwise_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_depthwise_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_depthwise_test)
  add_test(NAME dm_depthwise COMMAND dm_depthwise_test)
  add_executable(dm_dwpw_fused_test test/unit/dm_dwpw_fused_test.cu)
  target_link_libraries(dm_dwpw_fused_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_dwpw_fused_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG;-Xptxas=-v,-warn-spills>)
  tilemega_configure_cuda_target(dm_dwpw_fused_test)
  add_test(NAME dm_dwpw_fused COMMAND dm_dwpw_fused_test)
  add_executable(dm_encoder_attention_test test/unit/dm_encoder_attention_test.cu)
  target_link_libraries(dm_encoder_attention_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_encoder_attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_encoder_attention_test)
  add_test(NAME dm_encoder_attention COMMAND dm_encoder_attention_test)
  add_executable(dm_descriptor_device_test test/unit/dm_descriptor_device_test.cu)
  target_link_libraries(dm_descriptor_device_test PRIVATE TileMega::tilemega)
  target_compile_definitions(dm_descriptor_device_test PRIVATE TILEMEGA_DM_SUPPORT=1)
  target_compile_options(dm_descriptor_device_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_descriptor_device_test)
  add_test(NAME dm_descriptor_device COMMAND dm_descriptor_device_test)
  add_executable(dm_epilogue_value_test test/unit/dm_epilogue_value_test.cu)
  target_link_libraries(dm_epilogue_value_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_epilogue_value_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_epilogue_value_test)
  add_test(NAME dm_epilogue_value COMMAND dm_epilogue_value_test
    "${CMAKE_SOURCE_DIR}/test/fixtures/dnn/epilogue_values.bin")
  add_executable(dm_epilogue_tile_test test/unit/dm_epilogue_tile_test.cu)
  target_link_libraries(dm_epilogue_tile_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_epilogue_tile_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_epilogue_tile_test)
  add_test(NAME dm_epilogue_tile COMMAND dm_epilogue_tile_test
    "${CMAKE_SOURCE_DIR}/test/fixtures/dnn/epilogue_tiles.bin")
  add_executable(dm_gemm_chain_test test/unit/dm_gemm_chain_test.cu)
  target_link_libraries(dm_gemm_chain_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_gemm_chain_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_gemm_chain_test)
  add_test(NAME dm_gemm_chain COMMAND dm_gemm_chain_test
    "${CMAKE_SOURCE_DIR}/test/fixtures/dnn/gemm_chains.bin")
  add_executable(forward_runtime_test test/unit/forward_runtime_test.cu)
  target_link_libraries(forward_runtime_test PRIVATE TileMega::tilemega)
  target_compile_options(forward_runtime_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(forward_runtime_test)
  add_test(NAME forward_runtime COMMAND forward_runtime_test)
  add_executable(dm_counted_stage_threshold_test test/unit/dm_counted_stage_threshold_test.cu)
  target_link_libraries(dm_counted_stage_threshold_test PRIVATE TileMega::tilemega)
  target_compile_definitions(dm_counted_stage_threshold_test PRIVATE TILEMEGA_EVENT_KAPPA=1)
  target_compile_options(dm_counted_stage_threshold_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(dm_counted_stage_threshold_test)
  add_test(NAME dm_counted_stage_threshold COMMAND dm_counted_stage_threshold_test)
  foreach(phase IN ITEMS 0 2)
    add_executable(dm_paged_multipage_phase${phase}_test
      test/unit/dm_paged_multipage_phase_test.cu)
    target_link_libraries(dm_paged_multipage_phase${phase}_test PRIVATE TileMega::tilemega)
    target_compile_definitions(dm_paged_multipage_phase${phase}_test PRIVATE DM_TEST_PHASE=${phase})
    target_compile_options(dm_paged_multipage_phase${phase}_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
    tilemega_configure_cuda_target(dm_paged_multipage_phase${phase}_test)
    add_test(NAME dm_paged_multipage_phase${phase}
      COMMAND dm_paged_multipage_phase${phase}_test)
  endforeach()
  foreach(mode IN ITEMS 0 1 2 3 4)
    add_executable(dm_qperkv8_attention_mode${mode}_test test/unit/dm_qperkv8_attention_test.cu)
    target_link_libraries(dm_qperkv8_attention_mode${mode}_test PRIVATE TileMega::tilemega)
    target_compile_definitions(dm_qperkv8_attention_mode${mode}_test PRIVATE DM_TEST_ATTN_MODE=${mode} TILEMEGA_DM_SUPPORT=1)
    target_compile_options(dm_qperkv8_attention_mode${mode}_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
    tilemega_configure_cuda_target(dm_qperkv8_attention_mode${mode}_test)
    add_test(NAME dm_qperkv8_attention_mode${mode} COMMAND dm_qperkv8_attention_mode${mode}_test)
  endforeach()
  add_executable(dm_qperkv8_prefill_test test/unit/dm_qperkv8_prefill_test.cu)
  target_link_libraries(dm_qperkv8_prefill_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_qperkv8_prefill_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_qperkv8_prefill_test)
  add_test(NAME dm_qperkv8_prefill COMMAND dm_qperkv8_prefill_test)
  foreach(wrapper RANGE 0 1)
    add_executable(dm_moe_routing_combine_${wrapper}_test test/unit/dm_moe_routing_combine_test.cu)
    target_link_libraries(dm_moe_routing_combine_${wrapper}_test PRIVATE TileMega::tilemega)
    target_compile_definitions(dm_moe_routing_combine_${wrapper}_test PRIVATE DM_TEST_STAGE_WRAPPERS=${wrapper})
    target_compile_options(dm_moe_routing_combine_${wrapper}_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
    tilemega_configure_cuda_target(dm_moe_routing_combine_${wrapper}_test)
    add_test(NAME dm_moe_routing_combine_${wrapper} COMMAND dm_moe_routing_combine_${wrapper}_test)
  endforeach()
  add_executable(dm_moe_rms_stats_test test/unit/dm_moe_rms_stats_test.cu)
  target_link_libraries(dm_moe_rms_stats_test PRIVATE TileMega::tilemega)
  target_compile_options(dm_moe_rms_stats_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
  tilemega_configure_cuda_target(dm_moe_rms_stats_test)
  add_test(NAME dm_moe_rms_stats COMMAND dm_moe_rms_stats_test)
  target_link_libraries(task_body_contract_test PRIVATE TileMega::tilemega)
  add_test(NAME task_body_contract COMMAND task_body_contract_test)

  add_executable(bf16_task_body_compile_test
    test/unit/bf16_task_body_compile_test.cu)
  target_link_libraries(bf16_task_body_compile_test PRIVATE TileMega::tilemega)
  target_compile_options(bf16_task_body_compile_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(bf16_task_body_compile_test)
  add_test(NAME bf16_task_body_compile COMMAND bf16_task_body_compile_test)
  add_executable(serving_gemm_config_test
    test/unit/serving_gemm_config_test.cu)
  target_link_libraries(serving_gemm_config_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_gemm_config_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_gemm_config_test)
  add_test(NAME serving_gemm_config COMMAND serving_gemm_config_test)

  add_test(NAME serving_pruning COMMAND tilemega-unit serving_pruning)
  add_test(NAME serving_lag COMMAND tilemega-unit serving_lag)

  add_test(NAME serving_task_index COMMAND tilemega-unit serving_task_index)
  # CUDA numerical probe, run explicitly on a device just like attention_phase.
  add_executable(serving_epilogue_test test/unit/serving_epilogue_test.cu)
  target_link_libraries(serving_epilogue_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_epilogue_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_epilogue_test)
  add_test(NAME serving_epilogue COMMAND serving_epilogue_test)
  add_executable(serving_gemm_task_body_test
    test/unit/serving_gemm_task_body_test.cu)
  target_link_libraries(serving_gemm_task_body_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_gemm_task_body_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_gemm_task_body_test)
  add_test(NAME serving_gemm_task_body COMMAND serving_gemm_task_body_test)
  add_executable(serving_rmsnorm_test test/unit/serving_rmsnorm_test.cu)
  target_link_libraries(serving_rmsnorm_test PRIVATE TileMega::tilemega)
  tilemega_configure_cuda_target(serving_rmsnorm_test)
  add_test(NAME serving_rmsnorm COMMAND serving_rmsnorm_test)
  add_executable(serving_state_tasks_test test/unit/serving_state_tasks_test.cu)
  target_link_libraries(serving_state_tasks_test PRIVATE TileMega::tilemega)
  tilemega_configure_cuda_target(serving_state_tasks_test)
  add_test(NAME serving_state_tasks COMMAND serving_state_tasks_test)
  add_executable(serving_combine_test test/unit/serving_combine_test.cu)
  target_link_libraries(serving_combine_test PRIVATE TileMega::tilemega)
  tilemega_configure_cuda_target(serving_combine_test)
  add_test(NAME serving_combine COMMAND serving_combine_test)

  add_test(NAME serving_token_sets COMMAND tilemega-unit serving_token_sets)

  add_executable(serving_harness_compile_test
    test/unit/serving_harness_compile_test.cu)
  target_link_libraries(serving_harness_compile_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_harness_compile_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_harness_compile_test)
  add_test(NAME serving_harness_compile COMMAND serving_harness_compile_test)
  add_executable(serving_attention_mma_test
    test/unit/serving_attention_mma_test.cu)
  target_link_libraries(serving_attention_mma_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_attention_mma_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_attention_mma_test)
  add_test(NAME serving_attention_mma COMMAND serving_attention_mma_test)
  add_executable(serving_fused_attention_test
    test/unit/serving_fused_attention_test.cu)
  target_link_libraries(serving_fused_attention_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_fused_attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_fused_attention_test)
  add_test(NAME serving_fused_attention COMMAND serving_fused_attention_test)
  add_executable(serving_attention_merge_test test/unit/serving_attention_merge_test.cu)
  target_link_libraries(serving_attention_merge_test PRIVATE TileMega::tilemega)
  add_test(NAME serving_attention_merge COMMAND serving_attention_merge_test)

  add_executable(serving_attention_cases_test
    test/unit/serving_attention_cases_test.cu)
  target_link_libraries(serving_attention_cases_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_attention_cases_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_attention_cases_test)
  add_test(NAME serving_attention_cases COMMAND serving_attention_cases_test)
  foreach(test IN ITEMS sm90_gmma_layout attention_storage_layout page_ring paged_gemm attention_layout serving_mma_pipeline serving_pvswap monotonic_last_arriver serving_argmax_rows serving_gemv serving_gemv_access serving_task_profile)
    add_executable(${test}_test test/unit/${test}_test.cu)
    target_link_libraries(${test}_test PRIVATE TileMega::tilemega)
    target_compile_options(${test}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(${test}_test)
    add_test(NAME ${test} COMMAND ${test}_test)
  endforeach()
  if(TILEMEGA_TARGET_ARCH STREQUAL "sm_90")
    add_executable(sm90_dense_boundary_diagnostic_test test/unit/sm90_dense_boundary_diagnostic_test.cpp)
    target_include_directories(sm90_dense_boundary_diagnostic_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(sm90_dense_boundary_diagnostic_test PRIVATE cxx_std_17)
    target_compile_options(sm90_dense_boundary_diagnostic_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_boundary_diagnostic COMMAND sm90_dense_boundary_diagnostic_test)
    add_executable(sm90_dense_epilogue_contract_test test/unit/sm90_dense_epilogue_contract_test.cpp)
    target_include_directories(sm90_dense_epilogue_contract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(sm90_dense_epilogue_contract_test PRIVATE cxx_std_17)
    target_compile_options(sm90_dense_epilogue_contract_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_epilogue_contract COMMAND sm90_dense_epilogue_contract_test)
    add_test(NAME sm90_dense_role_contract COMMAND ${Python3_EXECUTABLE}
      -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python
      -p test_sm90_dense_role_contract.py)
    add_executable(sm90_dense_coordinates_test test/unit/sm90_dense_coordinates_test.cpp)
    target_link_libraries(sm90_dense_coordinates_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_dense_coordinates_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_coordinates COMMAND sm90_dense_coordinates_test)
    add_executable(sm90_dense_search_space_test test/unit/sm90_dense_search_space_test.cpp)
    target_link_libraries(sm90_dense_search_space_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_dense_search_space_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_search_space COMMAND sm90_dense_search_space_test
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/targets/sm_90_h800.json)
    add_executable(sm90_dense_lowering_contract_test test/unit/sm90_dense_lowering_contract_test.cpp)
    target_link_libraries(sm90_dense_lowering_contract_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_dense_lowering_contract_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_lowering_contract COMMAND sm90_dense_lowering_contract_test)
    add_executable(sm90_dense_compile_outcome_test test/unit/sm90_dense_compile_outcome_test.cpp)
    target_include_directories(sm90_dense_compile_outcome_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/commands)
    target_link_libraries(sm90_dense_compile_outcome_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_dense_compile_outcome_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_compile_outcome COMMAND sm90_dense_compile_outcome_test)
    add_executable(sm90_dense_register_contract_test test/unit/sm90_dense_register_contract_test.cpp)
    target_link_libraries(sm90_dense_register_contract_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_dense_register_contract_test PRIVATE -UNDEBUG)
    add_test(NAME sm90_dense_register_contract COMMAND sm90_dense_register_contract_test
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/targets/sm_90_h800.json)
    add_executable(sm90_dense_plan_test test/unit/sm90_dense_plan_test.cpp)
    target_link_libraries(sm90_dense_plan_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_dense_plan_test PRIVATE -UNDEBUG)
    foreach(test IN ITEMS sm90_dense_gemm sm90_dense_last_arriver sm90_dense_attention sm90_dense_token sm90_dense_epilogue)
      add_executable(${test}_test test/unit/${test}_test.cu)
      target_link_libraries(${test}_test PRIVATE TileMega::tilemega)
      target_compile_options(${test}_test PRIVATE -UNDEBUG
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
      tilemega_configure_cuda_target(${test}_test)
      add_test(NAME ${test} COMMAND ${test}_test)
    endforeach()
    add_executable(dm_epilogue_chain_view_test test/unit/dm_epilogue_tile_test.cu)
    target_link_libraries(dm_epilogue_chain_view_test PRIVATE TileMega::tilemega)
    target_compile_definitions(dm_epilogue_chain_view_test PRIVATE TILEMEGA_DM_CHAIN_REFERENCE=1)
    target_compile_options(dm_epilogue_chain_view_test PRIVATE -UNDEBUG
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(dm_epilogue_chain_view_test)
    add_test(NAME dm_epilogue_chain_view COMMAND dm_epilogue_chain_view_test
      ${CMAKE_SOURCE_DIR}/test/fixtures/dnn/epilogue_tiles.bin)
    add_executable(dm_nchw_store_test test/unit/dm_nchw_store_test.cu)
    target_link_libraries(dm_nchw_store_test PRIVATE TileMega::tilemega)
    target_compile_options(dm_nchw_store_test PRIVATE -UNDEBUG
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(dm_nchw_store_test)
    add_test(NAME dm_nchw_store COMMAND dm_nchw_store_test)
    add_executable(dm_dwpw_wgmma_test test/unit/dm_dwpw_wgmma_test.cu)
    target_link_libraries(dm_dwpw_wgmma_test PRIVATE TileMega::tilemega)
    target_compile_options(dm_dwpw_wgmma_test PRIVATE -UNDEBUG
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(dm_dwpw_wgmma_test)
    add_test(NAME dm_dwpw_wgmma COMMAND dm_dwpw_wgmma_test)
    add_executable(dm_depthwise_tma_test test/unit/dm_depthwise_tma_test.cu)
    target_link_libraries(dm_depthwise_tma_test PRIVATE TileMega::tilemega)
    target_compile_options(dm_depthwise_tma_test PRIVATE -UNDEBUG
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(dm_depthwise_tma_test)
    add_test(NAME dm_depthwise_tma COMMAND dm_depthwise_tma_test)
    add_executable(serving_qkv_epilogue_test test/unit/serving_qkv_epilogue_test.cu)
    target_link_libraries(serving_qkv_epilogue_test PRIVATE TileMega::tilemega)
    target_compile_options(serving_qkv_epilogue_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(serving_qkv_epilogue_test)
    add_test(NAME serving_qkv_epilogue COMMAND serving_qkv_epilogue_test)
    add_executable(serving_qkv_prefill_test test/unit/serving_qkv_prefill_test.cu)
    target_link_libraries(serving_qkv_prefill_test PRIVATE TileMega::tilemega)
    target_compile_options(serving_qkv_prefill_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(serving_qkv_prefill_test)
    add_test(NAME serving_qkv_prefill COMMAND serving_qkv_prefill_test)
    add_executable(sm90_prefill_tma_test test/unit/sm90_prefill_tma_test.cu)
    target_link_libraries(sm90_prefill_tma_test PRIVATE TileMega::tilemega CUDA::cuda_driver)
    target_compile_options(sm90_prefill_tma_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(sm90_prefill_tma_test)
    add_test(NAME sm90_prefill_tma COMMAND sm90_prefill_tma_test)
    add_executable(attention_layout_wgmma_test test/unit/attention_layout_test.cu)
    target_link_libraries(attention_layout_wgmma_test PRIVATE TileMega::tilemega)
    target_compile_definitions(attention_layout_wgmma_test PRIVATE TILEMEGA_ATTENTION_IMPL=4)
    target_compile_options(attention_layout_wgmma_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(attention_layout_wgmma_test)
    add_test(NAME attention_layout_wgmma COMMAND attention_layout_wgmma_test)
    add_executable(serving_gemv_nc_test test/unit/serving_gemv_nc_test.cu)
    target_link_libraries(serving_gemv_nc_test PRIVATE TileMega::tilemega)
    target_compile_options(serving_gemv_nc_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(serving_gemv_nc_test)
    add_test(NAME serving_gemv_nc COMMAND serving_gemv_nc_test)
    add_executable(serving_gemv_bulk_test test/unit/serving_gemv_bulk_test.cu)
    target_link_libraries(serving_gemv_bulk_test PRIVATE TileMega::tilemega)
    target_compile_options(serving_gemv_bulk_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(serving_gemv_bulk_test)
    add_test(NAME serving_gemv_bulk COMMAND serving_gemv_bulk_test)
    foreach(cluster IN ITEMS 2 4)
      add_executable(serving_cluster_launch_${cluster}_test test/unit/serving_cluster_launch_test.cu)
      target_link_libraries(serving_cluster_launch_${cluster}_test PRIVATE TileMega::tilemega)
      target_compile_definitions(serving_cluster_launch_${cluster}_test PRIVATE
        TILEMEGA_SERVING_CLUSTER_DIM=${cluster})
      target_compile_options(serving_cluster_launch_${cluster}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
      tilemega_configure_cuda_target(serving_cluster_launch_${cluster}_test)
      add_test(NAME serving_cluster_launch_${cluster} COMMAND serving_cluster_launch_${cluster}_test)
      foreach(kind IN ITEMS sm90_cluster_mailbox serving_cluster_combine sm90_attention_cluster)
        add_executable(${kind}_${cluster}_test test/unit/${kind}_test.cu)
        target_link_libraries(${kind}_${cluster}_test PRIVATE TileMega::tilemega CUDA::cuda_driver)
        target_compile_definitions(${kind}_${cluster}_test PRIVATE TILEMEGA_SERVING_CLUSTER_DIM=${cluster})
        target_compile_options(${kind}_${cluster}_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
        tilemega_configure_cuda_target(${kind}_${cluster}_test)
        add_test(NAME ${kind}_${cluster} COMMAND ${kind}_${cluster}_test)
      endforeach()
    endforeach()
    foreach(layout IN ITEMS row tiled)
      add_executable(sm90_decode_tma_${layout}_test test/unit/sm90_decode_tma_test.cu)
      target_link_libraries(sm90_decode_tma_${layout}_test PRIVATE TileMega::tilemega CUDA::cuda_driver)
      if(layout STREQUAL "tiled")
        target_compile_definitions(sm90_decode_tma_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
      else()
        target_compile_definitions(sm90_decode_tma_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=0)
      endif()
      target_compile_options(sm90_decode_tma_${layout}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
      tilemega_configure_cuda_target(sm90_decode_tma_${layout}_test)
      add_test(NAME sm90_decode_tma_${layout} COMMAND sm90_decode_tma_${layout}_test)
    endforeach()
    foreach(layout IN ITEMS nonpaged paged)
      add_executable(serving_residency_${layout}_test test/unit/serving_residency_test.cu)
      target_link_libraries(serving_residency_${layout}_test PRIVATE TileMega::tilemega CUDA::cuda_driver)
      if(layout STREQUAL "paged")
        target_compile_definitions(serving_residency_${layout}_test PRIVATE TILEMEGA_PAGED=1)
      else()
        target_compile_definitions(serving_residency_${layout}_test PRIVATE TILEMEGA_PAGED=0)
      endif()
      target_compile_options(serving_residency_${layout}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
      tilemega_configure_cuda_target(serving_residency_${layout}_test)
      add_test(NAME serving_residency_${layout} COMMAND serving_residency_${layout}_test)
    endforeach()
    add_executable(sm90_async_proxy_test test/unit/sm90_async_proxy_test.cu)
    target_link_libraries(sm90_async_proxy_test PRIVATE TileMega::tilemega CUDA::cuda_driver)
    target_compile_options(sm90_async_proxy_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(sm90_async_proxy_test)
    add_test(NAME sm90_async_proxy COMMAND sm90_async_proxy_test)
    add_executable(sm90_barrier_lifecycle_test test/unit/sm90_barrier_lifecycle_test.cu)
    target_link_libraries(sm90_barrier_lifecycle_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_barrier_lifecycle_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(sm90_barrier_lifecycle_test)
    add_test(NAME sm90_barrier_lifecycle COMMAND sm90_barrier_lifecycle_test)
    foreach(layout IN ITEMS row tiled)
      add_executable(sm90_moe_bulk_gmma_${layout}_test test/unit/sm90_moe_bulk_gmma_test.cu)
      target_link_libraries(sm90_moe_bulk_gmma_${layout}_test PRIVATE TileMega::tilemega)
      target_compile_options(sm90_moe_bulk_gmma_${layout}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
      if(layout STREQUAL "tiled")
        target_compile_definitions(sm90_moe_bulk_gmma_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
      endif()
      tilemega_configure_cuda_target(sm90_moe_bulk_gmma_${layout}_test)
      add_test(NAME sm90_moe_bulk_gmma_${layout} COMMAND sm90_moe_bulk_gmma_${layout}_test)
    endforeach()
    add_executable(sm90_encoder_attention_test test/unit/sm90_encoder_attention_test.cu)
    target_link_libraries(sm90_encoder_attention_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_encoder_attention_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(sm90_encoder_attention_test)
    add_test(NAME sm90_encoder_attention COMMAND sm90_encoder_attention_test)
    add_executable(sm90_prefill_attention_test test/unit/sm90_prefill_attention_test.cu)
    target_link_libraries(sm90_prefill_attention_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_prefill_attention_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(sm90_prefill_attention_test)
    add_test(NAME sm90_prefill_attention COMMAND sm90_prefill_attention_test)
    foreach(layout IN ITEMS row tiled)
      add_executable(sm90_conv_im2col_${layout}_test test/unit/sm90_conv_im2col_test.cu)
      target_link_libraries(sm90_conv_im2col_${layout}_test PRIVATE TileMega::tilemega)
      target_compile_definitions(sm90_conv_im2col_${layout}_test PRIVATE TILEMEGA_DM_SUPPORT=1)
      target_compile_options(sm90_conv_im2col_${layout}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
      if(layout STREQUAL "tiled")
        target_compile_definitions(sm90_conv_im2col_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
      endif()
      tilemega_configure_cuda_target(sm90_conv_im2col_${layout}_test)
      add_test(NAME sm90_conv_im2col_${layout} COMMAND sm90_conv_im2col_${layout}_test)
    endforeach()
    foreach(layout IN ITEMS row tiled)
      add_executable(sm90_forward_gmma_${layout}_test test/unit/sm90_forward_gmma_test.cu)
      target_link_libraries(sm90_forward_gmma_${layout}_test PRIVATE TileMega::tilemega)
      target_compile_options(sm90_forward_gmma_${layout}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      if(layout STREQUAL "tiled")
        target_compile_definitions(sm90_forward_gmma_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
      endif()
      tilemega_configure_cuda_target(sm90_forward_gmma_${layout}_test)
      add_test(NAME sm90_forward_gmma_${layout} COMMAND sm90_forward_gmma_${layout}_test)
    endforeach()
    add_executable(sm90_forward_moe_test test/unit/sm90_forward_moe_test.cu)
    target_link_libraries(sm90_forward_moe_test PRIVATE TileMega::tilemega)
    target_compile_options(sm90_forward_moe_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(sm90_forward_moe_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
    tilemega_configure_cuda_target(sm90_forward_moe_test)
    add_test(NAME sm90_forward_moe COMMAND sm90_forward_moe_test)
    foreach(layout IN ITEMS row tiled)
      foreach(kind IN ITEMS gmma decode_tma)
        add_executable(sm90_${kind}_m8_${layout}_test test/unit/sm90_${kind}_m8_test.cu)
        target_link_libraries(sm90_${kind}_m8_${layout}_test PRIVATE TileMega::tilemega CUDA::cuda_driver)
        target_compile_options(sm90_${kind}_m8_${layout}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
        if(layout STREQUAL "tiled")
          target_compile_definitions(sm90_${kind}_m8_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
        endif()
        tilemega_configure_cuda_target(sm90_${kind}_m8_${layout}_test)
        add_test(NAME sm90_${kind}_m8_${layout} COMMAND sm90_${kind}_m8_${layout}_test)
      endforeach()
    endforeach()
    foreach(layout IN ITEMS row tiled wide)
      add_executable(sm90_gmma_${layout}_test test/unit/sm90_gmma_test.cu)
      target_link_libraries(sm90_gmma_${layout}_test PRIVATE TileMega::tilemega)
      target_compile_options(sm90_gmma_${layout}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      if(layout STREQUAL "tiled")
        target_compile_definitions(sm90_gmma_${layout}_test PRIVATE TILEMEGA_NONPAGED_TILED=1)
      elseif(layout STREQUAL "wide")
        target_compile_definitions(sm90_gmma_${layout}_test PRIVATE TILEMEGA_SM90_WIDE_TEST=1)
      endif()
      tilemega_configure_cuda_target(sm90_gmma_${layout}_test)
      add_test(NAME sm90_gmma_${layout} COMMAND sm90_gmma_${layout}_test)
    endforeach()
  endif()
  add_executable(attention_layout_pvswap_test test/unit/attention_layout_test.cu)
  target_link_libraries(attention_layout_pvswap_test PRIVATE TileMega::tilemega)
  target_compile_definitions(attention_layout_pvswap_test PRIVATE TILEMEGA_ATTENTION_PVSWAP=1)
  target_compile_options(attention_layout_pvswap_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(attention_layout_pvswap_test)
  add_test(NAME attention_layout_pvswap COMMAND attention_layout_pvswap_test)
  foreach(impl IN ITEMS pvswap2 simt)
    if(impl STREQUAL "pvswap2")
      set(impl_id 2)
    else()
      set(impl_id 3)
    endif()
    add_executable(attention_layout_${impl}_test test/unit/attention_layout_test.cu)
    target_link_libraries(attention_layout_${impl}_test PRIVATE TileMega::tilemega)
    target_compile_definitions(attention_layout_${impl}_test PRIVATE TILEMEGA_ATTENTION_IMPL=${impl_id})
    target_compile_options(attention_layout_${impl}_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    tilemega_configure_cuda_target(attention_layout_${impl}_test)
    add_test(NAME attention_layout_${impl} COMMAND attention_layout_${impl}_test)
  endforeach()
  add_test(NAME attention_pvswap2_bitwise COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/python/attention_variant_equivalence.py
    $<TARGET_FILE:attention_layout_pvswap_test> $<TARGET_FILE:attention_layout_pvswap2_test>)
  foreach(test IN ITEMS dm_epoch_last_arriver dm_counted_threshold
      dm_nonpaged_handoff dm_nonpaged_attention_handoff dm_small_native_gemm
      dm_binding_stream dm_binding_pages dm_moe_counted_publish)
    add_executable(${test}_test test/unit/${test}_test.cu)
    target_link_libraries(${test}_test PRIVATE TileMega::tilemega)
    target_compile_options(${test}_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-UNDEBUG>)
    tilemega_configure_cuda_target(${test}_test)
    add_test(NAME ${test} COMMAND ${test}_test)
  endforeach()
  add_executable(norm_prologue_gemm_test test/unit/norm_prologue_gemm_test.cu)
  target_link_libraries(norm_prologue_gemm_test PRIVATE TileMega::tilemega)
  target_compile_options(norm_prologue_gemm_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(norm_prologue_gemm_test)
  add_test(NAME norm_prologue_gemm COMMAND norm_prologue_gemm_test)
  add_executable(last_arriver_test test/unit/last_arriver_test.cu)
  target_link_libraries(last_arriver_test PRIVATE TileMega::tilemega)
  target_compile_options(last_arriver_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(last_arriver_test)
  add_test(NAME last_arriver COMMAND last_arriver_test)
  add_executable(serving_watchdog_test test/unit/serving_watchdog_test.cu)
  target_link_libraries(serving_watchdog_test PRIVATE TileMega::tilemega)
  tilemega_configure_cuda_target(serving_watchdog_test)
  add_test(NAME serving_watchdog COMMAND serving_watchdog_test)
  add_executable(direct_handoff_test test/unit/direct_handoff_test.cu)
  target_link_libraries(direct_handoff_test PRIVATE TileMega::tilemega)
  tilemega_configure_cuda_target(direct_handoff_test)
  add_test(NAME direct_handoff COMMAND direct_handoff_test)
  add_executable(independent_attention_test test/unit/serving_attention_cases_test.cu)
  target_link_libraries(independent_attention_test PRIVATE TileMega::tilemega)
  target_compile_definitions(independent_attention_test PRIVATE TILEMEGA_TEST_INDEPENDENT=1)
  target_compile_options(independent_attention_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(independent_attention_test)
  add_test(NAME independent_attention COMMAND independent_attention_test)
  add_executable(paged_attention_test test/unit/serving_attention_cases_test.cu)
  target_link_libraries(paged_attention_test PRIVATE TileMega::tilemega)
  target_compile_definitions(paged_attention_test PRIVATE TILEMEGA_TEST_PAGED=1)
  target_compile_options(paged_attention_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(paged_attention_test)
  add_test(NAME paged_attention COMMAND paged_attention_test)

  add_executable(serving_prefill_attention_test
    test/unit/serving_prefill_attention_test.cu)
  target_link_libraries(serving_prefill_attention_test PRIVATE TileMega::tilemega)
  target_compile_options(serving_prefill_attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_prefill_attention_test)
  add_test(NAME serving_prefill_attention COMMAND serving_prefill_attention_test)
  add_executable(serving_prefill_attention_128_test
    test/unit/serving_prefill_attention_test.cu)
  target_link_libraries(serving_prefill_attention_128_test PRIVATE TileMega::tilemega)
  target_compile_definitions(serving_prefill_attention_128_test PRIVATE TILEMEGA_TEST_PREFILL_SEQ=128)
  target_compile_options(serving_prefill_attention_128_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(serving_prefill_attention_128_test)
  add_test(NAME serving_prefill_attention_128 COMMAND serving_prefill_attention_128_test)
  # Run explicitly: this numerical probe needs a CUDA device, unlike the
  # compile-only TaskBody contracts in the default CTest suite.
  add_executable(attention_phase_test test/unit/attention_phase_test.cu)
  target_link_libraries(attention_phase_test PRIVATE TileMega::tilemega)
  target_compile_options(attention_phase_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_executable(fused_gemm_task_body_test test/unit/fused_gemm_task_body_test.cu)
  target_link_libraries(fused_gemm_task_body_test PRIVATE TileMega::tilemega)
  target_compile_definitions(fused_gemm_task_body_test PRIVATE
    TILEMEGA_MODEL_BF16=1 TILEMEGA_GEMM_TILE_M=32 TILEMEGA_GEMM_TILE_N=128
    TILEMEGA_GEMM_TILE_K=16 TILEMEGA_GEMM_STAGES=3)
  target_compile_options(fused_gemm_task_body_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(fused_gemm_task_body_test)

  add_executable(fused_harness_compile_test test/unit/bf16_task_body_compile_test.cu)
  target_link_libraries(fused_harness_compile_test PRIVATE TileMega::tilemega)
  target_compile_definitions(fused_harness_compile_test PRIVATE TILEMEGA_FUSION_RUNTIME=1)
  target_compile_options(fused_harness_compile_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  tilemega_configure_cuda_target(fused_harness_compile_test)
  add_test(NAME fused_harness_compile COMMAND fused_harness_compile_test)

  add_test(NAME forward_frontend COMMAND tilemega-unit forward_frontend)
  add_test(NAME frontend_import COMMAND tilemega-unit frontend_import)
  add_test(NAME dm_descriptor COMMAND tilemega-unit dm_descriptor)

  add_test(NAME skeleton_placement COMMAND tilemega-unit skeleton_placement)

  add_test(NAME isolated_evaluation COMMAND tilemega-unit isolated_evaluation)

  add_test(NAME skeleton_search_isolation COMMAND tilemega-unit skeleton_search_isolation)
  add_test(NAME serving_search_rejection COMMAND tilemega-unit serving_search_rejection)

  add_test(NAME plan_skeleton COMMAND tilemega-unit plan_skeleton)

  add_test(NAME variant_resource COMMAND tilemega-unit variant_resource)

  add_test(NAME symbolic_oracle COMMAND tilemega-unit symbolic_oracle)

  add_test(NAME operator_classes COMMAND tilemega-unit operator_classes)
  add_test(NAME dm_operator_classes COMMAND tilemega-unit dm_operator_classes)
  add_test(NAME serving_model_domains COMMAND tilemega-unit serving_model_domains)
  add_test(NAME moe_routing_profile COMMAND tilemega-unit moe_routing_profile)
  add_test(NAME moe_task_pricing COMMAND tilemega-unit moe_task_pricing)
  add_test(NAME moe_dram_bindings COMMAND tilemega-unit moe_dram_bindings)
  add_test(NAME moe_structure_search COMMAND tilemega-unit moe_structure_search)
  add_test(NAME moe_group_profile COMMAND tilemega-unit moe_group_profile)
  add_test(NAME dm_gemm_traits COMMAND tilemega-unit dm_gemm_traits)
  add_test(NAME dm_gemm_candidates COMMAND tilemega-unit dm_gemm_candidates)
  add_test(NAME conv_iteration COMMAND tilemega-unit conv_iteration)
  add_test(NAME conv_reduction_partition COMMAND tilemega-unit conv_reduction_partition)
  add_test(NAME conv_reduction_cg COMMAND tilemega-unit conv_reduction_cg)
  add_test(NAME dnn_semantic_lifting COMMAND tilemega-unit dnn_semantic_lifting)
  add_test(NAME dnn_epilogue_semantics COMMAND tilemega-unit dnn_epilogue_semantics)
  add_test(NAME dnn_model_plan COMMAND tilemega-unit dnn_model_plan)
  add_test(NAME moe_region_pattern COMMAND tilemega-unit moe_region_pattern)
  add_test(NAME moe_region_plan COMMAND tilemega-unit moe_region_plan)
  add_test(NAME task_storage COMMAND tilemega-unit task_storage)
  add_test(NAME dm_counted_fusion COMMAND tilemega-unit dm_counted_fusion)
  add_test(NAME dm_gemm_class_domain COMMAND tilemega-unit dm_gemm_class_domain)
  add_test(NAME dynamic_task_cursor COMMAND tilemega-unit dynamic_task_cursor)
  add_test(NAME moe_opaque_control COMMAND tilemega-unit moe_opaque_control)
  add_test(NAME dm_forward_search COMMAND tilemega-unit dm_forward_search)
  add_test(NAME dm_physical_footprint COMMAND tilemega-unit dm_physical_footprint)
  add_test(NAME dnn_structure_search COMMAND tilemega-unit dnn_structure_search)
  add_test(NAME task_arithmetic_domain COMMAND tilemega-unit task_arithmetic_domain)
  add_test(NAME dm_store_geometry COMMAND tilemega-unit dm_store_geometry)
  add_test(NAME dm_exact_fusion COMMAND tilemega-unit dm_exact_fusion)
  add_test(NAME dnn_deferred_ln COMMAND tilemega-unit dnn_deferred_ln)
  add_test(NAME dnn_dwpw_fusion COMMAND tilemega-unit dnn_dwpw_fusion
    "${CMAKE_SOURCE_DIR}/test/fixtures/dnn/dwpw_pair.json")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-fixtures")
  add_test(NAME dnn_memory_codegen COMMAND tilemega-unit dnn_memory_codegen
    --emit "${CMAKE_BINARY_DIR}/test-fixtures/dnn-memory.cu")
  add_test(NAME dnn_memory_search COMMAND tilemega-unit dnn_memory_codegen
    --emit "${CMAKE_BINARY_DIR}/test-fixtures/dnn-memory-search.cu" --search)
  add_test(NAME dm_dispatch_codegen COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_SOURCE_DIR}/test/python/check_dm_dispatch_codegen.py"
    --unit $<TARGET_FILE:tilemega-unit> --compiler "${CMAKE_CXX_COMPILER}"
    --fixture "${CMAKE_SOURCE_DIR}/test/fixtures/dnn/dwpw_pair.json"
    --out "${CMAKE_BINARY_DIR}/test-fixtures/dm-dispatch")
  add_test(NAME dm_fused_dependencies COMMAND tilemega-unit dm_fused_dependencies)
  add_test(NAME dm_dram_images COMMAND tilemega-unit dm_dram_images)
  add_test(NAME binding_request_traffic COMMAND tilemega-unit binding_request_traffic)
  add_test(NAME binding_request_pricing COMMAND tilemega-unit binding_request_pricing)
  add_test(NAME typed_affine_pricing COMMAND tilemega-unit typed_affine_pricing)
  add_test(NAME dm_virtual_gemm_partition COMMAND tilemega-unit dm_virtual_gemm_partition)
  add_test(NAME moe_semantic_lifting COMMAND tilemega-unit moe_semantic_lifting)
  add_test(NAME moe_region_codegen COMMAND tilemega-unit moe_region_codegen)
  add_test(NAME dm_semantic_signature COMMAND tilemega-unit dm_semantic_signature)
  add_test(NAME dm_virtual_frontend COMMAND tilemega-unit dm_virtual_frontend)
  add_test(NAME dm_piece_pricing COMMAND tilemega-unit dm_piece_pricing)
  add_test(NAME dm_target_memory COMMAND tilemega-unit dm_target_memory)
  add_test(NAME dm_page_resources COMMAND tilemega-unit dm_page_resources)
  add_test(NAME counted_dependency_cg COMMAND tilemega-unit counted_dependency_cg)
  add_test(NAME counted_dependency_flow COMMAND tilemega-unit counted_dependency_flow)
  add_test(NAME runtime_dependency_codec COMMAND tilemega-unit runtime_dependency_codec)
  add_test(NAME counted_threshold COMMAND tilemega-unit counted_threshold)
  add_test(NAME counted_write COMMAND tilemega-unit counted_write)

  add_test(NAME coupling_cache COMMAND tilemega-unit coupling_cache)

  add_test(NAME semantic_lifting COMMAND tilemega-unit semantic_lifting)

  add_test(NAME embedding_plan COMMAND tilemega-unit embedding_plan)

  add_test(NAME wiring_coupling COMMAND tilemega-unit wiring_coupling)

  add_test(NAME frontend_degradation COMMAND tilemega-unit frontend_degradation)

  # Part 1 dependency probe (docs/DEPENDENCIES.md): MLIR's own Presburger
  # implementation and isl/barvinok in one binary, so a regression that
  # breaks this link/runtime combination is caught by `ctest`, not
  # rediscovered by hand next time this dependency set changes.
  add_executable(isl_crosslink_test test/unit/isl_crosslink_test.cpp)
  target_link_libraries(isl_crosslink_test PRIVATE MLIRIR MLIRSupport
    MLIRPresburger tilemega::isl_stack)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(isl_crosslink_test PRIVATE -fno-rtti)
  endif()
  add_test(NAME isl_crosslink COMMAND isl_crosslink_test)

  # A Release configuration defines NDEBUG, which compiles every `assert` in
  # these binaries away: the suite then exits 0 without checking anything.
  # Verified here -- semantic_lifting reported four unexpected differences and
  # still passed in build-portable (Release) before this loop existed.
  get_property(_tilemega_dir_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_tilemega_cuda_test IN LISTS _tilemega_dir_targets)
    if(_tilemega_cuda_test MATCHES "_test$")
      get_target_property(_tilemega_sources ${_tilemega_cuda_test} SOURCES)
      foreach(_tilemega_source IN LISTS _tilemega_sources)
        if(_tilemega_source MATCHES "\\.cu$")
          tilemega_configure_cuda_target(${_tilemega_cuda_test})
          break()
        endif()
      endforeach()
    endif()
  endforeach()
  foreach(_tilemega_target IN LISTS _tilemega_dir_targets)
    if(_tilemega_target MATCHES "_test$")
      target_compile_options(${_tilemega_target} PRIVATE -UNDEBUG)
    endif()
  endforeach()
endif()

if(TILEMEGA_BUILD_VERIFY)
  add_subdirectory(tools/experimental/crosscompile)
endif()

add_custom_target(check-tilemega DEPENDS check-policy)
if(TARGET check-cg-lit)
  add_dependencies(check-tilemega check-cg-lit)
endif()
if(TARGET target_spec_test)
  add_dependencies(check-tilemega target_spec_test)
endif()
if(TARGET containment_test)
  add_dependencies(check-tilemega containment_test)
endif()
if(TARGET event_synthesis_test)
  add_dependencies(check-tilemega event_synthesis_test)
endif()
if(TARGET layout_bridge_test)
  add_dependencies(check-tilemega layout_bridge_test)
endif()
if(TARGET table27_test)
  add_dependencies(check-tilemega table27_test)
endif()
if(TARGET coupling_types_test)
  add_dependencies(check-tilemega coupling_types_test)
endif()

if(TARGET tilemega-unit)
  add_dependencies(check-tilemega tilemega-unit)
endif()

option(TILEMEGA_BUILD_EXPERIMENTAL "Build historical experiment tools" OFF)
if(TILEMEGA_BUILD_EXPERIMENTAL)
  add_executable(tilemega-flow-validation tools/experimental/tilemega-flow-validation.cpp)
  target_link_libraries(tilemega-flow-validation PRIVATE TileMega::tilemega MLIRParser)
  target_compile_definitions(tilemega-flow-validation PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(tilemega-flow-validation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-flow-grid tools/experimental/tilemega-flow-grid.cpp)
  target_link_libraries(tilemega-flow-grid PRIVATE TileMega::tilemega MLIRParser)
  target_compile_definitions(tilemega-flow-grid PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(tilemega-flow-grid PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-flow-point tools/experimental/tilemega-flow-point.cpp)
  target_link_libraries(tilemega-flow-point PRIVATE TileMega::tilemega MLIRParser)
  set_target_properties(tilemega-flow-point PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-shape-probe tools/experimental/tilemega-shape-probe.cpp)
  target_link_libraries(tilemega-shape-probe PRIVATE TileMega::tilemega)
  target_compile_definitions(tilemega-shape-probe PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(tilemega-shape-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-symbolic-probe tools/experimental/tilemega-symbolic-probe.cpp)
  target_link_libraries(tilemega-symbolic-probe PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-symbolic-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-affine-probe tools/experimental/tilemega-affine-probe.cpp)
  target_link_libraries(tilemega-affine-probe PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-affine-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-event-price tools/experimental/tilemega-event-price.cpp)
  target_link_libraries(tilemega-event-price PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-event-price PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-fusion-events tools/experimental/tilemega-fusion-events.cpp)
  add_executable(tilemega-fusion-dp tools/experimental/tilemega-fusion-dp.cpp)
  target_link_libraries(tilemega-fusion-dp PRIVATE TileMega::tilemega MLIRParser)
  set_target_properties(tilemega-fusion-dp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-fusion-chain tools/experimental/tilemega-fusion-chain.cpp)
  target_link_libraries(tilemega-fusion-chain PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-fusion-chain PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  target_link_libraries(tilemega-fusion-events PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-fusion-events PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-fusion-probe tools/experimental/tilemega-fusion-probe.cpp)
  add_executable(tilemega-fusion-cost tools/experimental/tilemega-fusion-cost.cpp)
  add_executable(tilemega-symbolic-task-price tools/experimental/tilemega-symbolic-task-price.cpp)
  target_link_libraries(tilemega-symbolic-task-price PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-symbolic-task-price PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  target_link_libraries(tilemega-fusion-cost PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-fusion-cost PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-attention-plan-price tools/experimental/tilemega-attention-plan-price.cpp)
  add_executable(tilemega-attention-dp tools/experimental/tilemega-attention-dp.cpp)
  target_link_libraries(tilemega-attention-dp PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-attention-dp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  target_link_libraries(tilemega-attention-plan-price PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-attention-plan-price PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  target_link_libraries(tilemega-fusion-probe PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-fusion-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
endif()

add_library(tilemega_commands OBJECT
  tools/commands/audit-binary.cpp
  tools/commands/device.cpp
  tools/commands/attention-work.cpp
  tools/commands/calibrate.cpp
  tools/commands/calibration-suite.cpp
  tools/commands/compile.cpp
  tools/commands/native_dense.cpp
  tools/commands/derive.cpp
  tools/commands/dram-floor.cpp
  tools/commands/request-floor.cpp
  tools/commands/event-cost.cpp
  tools/commands/flow-audit.cpp
  tools/commands/import.cpp
  tools/commands/interface-probe.cpp
  tools/commands/occupancy.cpp
  tools/commands/op-audit.cpp
  tools/commands/parametric.cpp
  tools/commands/runtime-projection.cpp
  tools/commands/scalar-error-probe.cpp
  tools/commands/skeleton-audit.cpp
  tools/commands/target-audit.cpp
  tools/commands/task-work-probe.cpp
  tools/commands/wait-policy.cpp
  tools/commands/wiring.cpp
)
add_dependencies(tilemega_commands TileMegaCGIncGen)
target_link_libraries(tilemega_commands PRIVATE TileMega::tilemega tilemega_calibration MLIRParser)
target_compile_definitions(tilemega_commands PRIVATE TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega_commands PRIVATE -fno-rtti)
endif()
add_executable(tilemega tools/tilemega.cpp $<TARGET_OBJECTS:tilemega_commands>)
target_link_libraries(tilemega PRIVATE TileMega::tilemega tilemega_calibration MLIRParser)
set_target_properties(tilemega PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(tilemega_fingerprint
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/tilemega/fingerprint.py
    --root ${CMAKE_CURRENT_SOURCE_DIR}
    --header ${CMAKE_CURRENT_BINARY_DIR}/include/tilemega/Support/BuildFingerprint.h
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/include/tilemega/Support/BuildFingerprint.h
  VERBATIM)
add_dependencies(tilemega tilemega_fingerprint)
add_dependencies(tilemega_commands tilemega_fingerprint)
add_test(NAME bench_framework
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_bench_framework.py)
add_test(NAME bench_queue_check
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_bench_queue_check.py)
add_test(NAME cleanup_plan
  COMMAND ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_cleanup_plan.py)
add_test(NAME bench_audit
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_bench_audit.py)
add_test(NAME artifact_comparison
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_compare_artifacts.py)
add_test(NAME build_identity
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_unified_identity.py)
add_test(NAME build_feature_contract
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_feature_contract.py)
add_test(NAME native_argument_contract
  COMMAND ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_r16_argument_contract.py)
add_test(NAME source_fingerprint
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_fingerprint_sources.py)
add_test(NAME generated_feature_contract
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    "TILEMEGA_BIN=$<TARGET_FILE:tilemega>"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_generated_features.py)
add_test(NAME dnn_bf16_gates
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_dnn_gates.py)

if(TILEMEGA_BUILD_TESTS AND Python3_EXECUTABLE)
  add_test(NAME serving_workload_host COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_workload*.py)
endif()

if(TILEMEGA_BUILD_TESTS AND Python3_EXECUTABLE)
  add_test(NAME sm90r1_trace_causal_parent COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python ${Python3_EXECUTABLE} -m unittest discover
    -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_trace_causal_parent.py)
  foreach(suite model_registry serving_head light_selection moe_full
      moe_deployment moe_routing_check serving_replay dnn_generated_metrics
      dm_scheduler_lock dnn_local_pool dnn_local_replay dnn_check_cli)
    add_test(NAME r15_${suite} COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/test/python -p test_${suite}.py)
  endforeach()
endif()
