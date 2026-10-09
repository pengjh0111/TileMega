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
  set(CMAKE_CUDA_ARCHITECTURES "${_tilemega_cuda_arch}" CACHE STRING "" FORCE)
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
    "-DMLIR_DIR=/path/to/lib/cmake/mlir (see docs/experiments/V_F/result.md).")
endif()
find_package(MLIR QUIET CONFIG)
if(NOT MLIR_FOUND)
  message(FATAL_ERROR
    "MLIR is required for the TileMega Coupling Graph dialect. Configure with "
    "-DMLIR_DIR=/path/to/lib/cmake/mlir. The pinned source/build recipe and "
    "commit are recorded in docs/experiments/V_F/result.md.")
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
  lib/Solver/CacheServiceCurve.cpp
  lib/Solver/ModelDescription.cpp
  lib/Solver/TaskModel.cpp
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
  lib/Dialect/CouplingGraph/BoundDependency.cpp
  lib/Analysis/VirtualTaskBinding.cpp
  lib/Analysis/ExactTaskWork.cpp
  lib/Analysis/DramFloor.cpp
  lib/Solver/TaskPriceParts.cpp
  lib/Solver/PiecePricing.cpp
  lib/Solver/DramFluid.cpp
  lib/Solver/ModelDramFloor.cpp
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
  add_executable(tilemega-unit test/unit/runner.cpp
    test/unit/target_spec_test.cpp
    test/unit/coupling_types_test.cpp
    test/unit/table27_test.cpp
    test/unit/semantics_test.cpp
    test/unit/dram_floor_test.cpp
    test/unit/regime_a_price_test.cpp
    test/unit/stage_flow_test.cpp
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
    test/unit/moe_group_profile_test.cpp
    test/unit/dm_gemm_traits_test.cpp
    test/unit/dm_gemm_candidates_test.cpp
    test/unit/dm_gemm_class_domain_test.cpp
    test/unit/dm_forward_search_test.cpp
    test/unit/task_arithmetic_domain_test.cpp
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

  add_test(NAME flow_runtime_release COMMAND tilemega-unit flow_runtime_release)

  add_test(NAME regime_a_price COMMAND tilemega-unit regime_a_price
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/experiments/SOLVER_R9B/fit/target.json)

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
    ${CMAKE_SOURCE_DIR}/docs/experiments/SEQSCAN/raw/export/gqa2.json
    ${CMAKE_SOURCE_DIR}/docs/experiments/SEQSCAN/raw/export/mha4.json)
  add_test(NAME op_arithmetic_audit COMMAND tilemega audit op)
  add_test(NAME task_work_domains COMMAND tilemega probe task-work-probe
    "${CMAKE_SOURCE_DIR}/docs/experiments/SEQSCAN/raw/export/gqa2.json"
    "${CMAKE_SOURCE_DIR}/docs/experiments/SEQSCAN/raw/export/mha4.json")

  add_test(NAME isl_relation COMMAND tilemega-unit isl_relation)

  add_test(NAME layout_bridge COMMAND tilemega-unit layout_bridge)

  add_test(NAME cg_attr_roundtrip COMMAND tilemega-unit cg_attr_roundtrip)

  add_test(NAME backend_query COMMAND tilemega-unit backend_query)

  add_test(NAME cache_service_curve COMMAND tilemega-unit cache_service_curve)

  add_test(NAME fusion_access COMMAND tilemega-unit fusion_access)
  add_test(NAME handoff_runtime_projection COMMAND tilemega-unit handoff_runtime_projection)
  add_test(NAME handoff_ir COMMAND tilemega-unit handoff_ir)
  foreach(split_case IN ITEMS four one off escape)
    add_test(NAME handoff_split_${split_case} COMMAND tilemega-unit handoff_ir
      splitk_import ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/serving_split.json
      ${split_case} ${CMAKE_CURRENT_SOURCE_DIR}/docs/experiments/SERVING_R12C/target_r12b.json)
  endforeach()
  add_test(NAME handoff_access COMMAND tilemega-unit handoff_access)
  foreach(split_case IN ITEMS four one off escape)
    add_test(NAME handoff_nonpaged_split_${split_case} COMMAND tilemega-unit handoff_ir
      nonpaged_splitk_import ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/serving_split.json
      ${split_case} ${CMAKE_CURRENT_SOURCE_DIR}/docs/experiments/SERVING_R12C/target_r12b.json)
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
  foreach(test IN ITEMS page_ring paged_gemm)
    add_executable(${test}_test test/unit/${test}_test.cu)
    target_link_libraries(${test}_test PRIVATE TileMega::tilemega)
    target_compile_options(${test}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
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

  add_test(NAME plan_skeleton COMMAND tilemega-unit plan_skeleton)

  add_test(NAME variant_resource COMMAND tilemega-unit variant_resource)

  add_test(NAME symbolic_oracle COMMAND tilemega-unit symbolic_oracle)

  add_test(NAME operator_classes COMMAND tilemega-unit operator_classes)
  add_test(NAME dm_operator_classes COMMAND tilemega-unit dm_operator_classes)
  add_test(NAME moe_routing_profile COMMAND tilemega-unit moe_routing_profile)
  add_test(NAME moe_group_profile COMMAND tilemega-unit moe_group_profile)
  add_test(NAME dm_gemm_traits COMMAND tilemega-unit dm_gemm_traits)
  add_test(NAME dm_gemm_candidates COMMAND tilemega-unit dm_gemm_candidates)
  add_test(NAME dm_gemm_class_domain COMMAND tilemega-unit dm_gemm_class_domain)
  add_test(NAME dm_forward_search COMMAND tilemega-unit dm_forward_search)
  add_test(NAME task_arithmetic_domain COMMAND tilemega-unit task_arithmetic_domain)

  add_test(NAME coupling_cache COMMAND tilemega-unit coupling_cache)

  add_test(NAME semantic_lifting COMMAND tilemega-unit semantic_lifting)

  add_test(NAME embedding_plan COMMAND tilemega-unit embedding_plan)

  add_test(NAME wiring_coupling COMMAND tilemega-unit wiring_coupling)

  add_test(NAME frontend_degradation COMMAND tilemega-unit frontend_degradation)

  # Part 1 dependency probe (docs/DEPENDENCIES.md): MLIR's own Presburger
  # implementation and isl/barvinok in one binary, so a regression that
  # breaks this link/runtime combination is caught by `ctest`, not
  # rediscovered by hand next time this dependency set changes.
  add_executable(isl_crosslink_test docs/experiments/P3_ISL/crosslink_probe.cpp)
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
  add_subdirectory(docs/experiments/V_I)
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
  add_executable(tilemega-r9-early docs/experiments/SOLVER_R9B/early_signal.cpp)
  target_link_libraries(tilemega-r9-early PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-r9-early PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-shape-probe tools/experimental/tilemega-shape-probe.cpp)
  target_link_libraries(tilemega-shape-probe PRIVATE TileMega::tilemega)
  target_compile_definitions(tilemega-shape-probe PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(tilemega-shape-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-identity-probe tools/experimental/tilemega-identity-probe.cpp)
  target_link_libraries(tilemega-identity-probe PRIVATE TileMega::tilemega)
  target_compile_definitions(tilemega-identity-probe PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(tilemega-identity-probe PROPERTIES
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
  add_executable(tilemega-task-work-gate tools/experimental/tilemega-task-work-gate.cpp)
  add_executable(tilemega-task-cost-gate tools/experimental/tilemega-task-cost-gate.cpp)
  add_executable(tilemega-scalar-cost-probe tools/experimental/tilemega-scalar-cost-probe.cpp)
  target_link_libraries(tilemega-scalar-cost-probe PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-scalar-cost-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  target_link_libraries(tilemega-task-cost-gate PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-task-cost-gate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-simulate docs/experiments/SIMULATOR/simulate.cpp)
  target_link_libraries(tilemega-simulate PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-simulate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-place-eft docs/experiments/PLACE_EFT/place_eft.cpp)
  target_link_libraries(tilemega-place-eft PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-place-eft PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-place-chain docs/experiments/CHAIN/place_chain.cpp)
  target_link_libraries(tilemega-place-chain PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-place-chain PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  target_link_libraries(tilemega-task-work-gate PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-task-work-gate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
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
  add_executable(tilemega-costmodel tools/experimental/tilemega-costmodel.cpp)
  target_link_libraries(tilemega-costmodel PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-costmodel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-migrate tools/experimental/tilemega-migrate.cpp)
  target_link_libraries(tilemega-migrate PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-migrate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  add_executable(tilemega-solve tools/experimental/tilemega-solve.cpp)
  target_link_libraries(tilemega-solve PRIVATE TileMega::tilemega)
  set_target_properties(tilemega-solve PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
endif()

add_library(tilemega_commands OBJECT
  tools/commands/audit-binary.cpp
  tools/commands/device.cpp
  tools/commands/attention-work.cpp
  tools/commands/calibrate.cpp
  tools/commands/calibration-suite.cpp
  tools/commands/compile.cpp
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
