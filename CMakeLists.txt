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
option(TILEMEGA_ENABLE_MLIR "Build the required MLIR Coupling Graph contract" ON)
option(TILEMEGA_ENABLE_ISL
  "Build the required isl/barvinok coupling-derivation solver (Phase 3)" ON)
set(TILEMEGA_TARGET_ARCH "auto" CACHE STRING "sm_80|sm_89|sm_90|sm_120|auto")
set_property(CACHE TILEMEGA_TARGET_ARCH PROPERTY STRINGS auto sm_80 sm_89 sm_90 sm_120)
set(TILEMEGA_TARGET_CONFIG "" CACHE PATH "target json override")

if(NOT TILEMEGA_TARGET_ARCH STREQUAL "auto")
  if(NOT TILEMEGA_TARGET_ARCH MATCHES "^sm_(80|89|90|120)$")
    message(FATAL_ERROR "Unsupported TILEMEGA_TARGET_ARCH=${TILEMEGA_TARGET_ARCH}")
  endif()
  string(REGEX REPLACE "^sm_" "" _tilemega_cuda_arch "${TILEMEGA_TARGET_ARCH}")
  set(CMAKE_CUDA_ARCHITECTURES "${_tilemega_cuda_arch}" CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

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

set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/CGDialect.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGDialect.h.inc
  -gen-dialect-decls -dialect=tilemega -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGDialect.cpp.inc
  -gen-dialect-defs -dialect=tilemega -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/CGAttrs.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGAttrs.h.inc
  -gen-attrdef-decls -attrdefs-dialect=tilemega -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGAttrs.cpp.inc
  -gen-attrdef-defs -attrdefs-dialect=tilemega -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LLVM_TARGET_DEFINITIONS include/tilemega/Dialect/CouplingGraph/CGOps.td)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGOps.h.inc
  -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(include/tilemega/Dialect/CouplingGraph/CGOps.cpp.inc
  -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_public_tablegen_target(TileMegaCGIncGen)

add_library(tilemega STATIC
  lib/Support/Json.cpp
  lib/Target/TargetSpec.cpp
  lib/Frontend/Frontend.cpp
  lib/Frontend/GraphPattern.cpp
  lib/Frontend/ModelPlan.cpp
  lib/Frontend/SemanticLifting.cpp
  lib/Dialect/CouplingGraph/CGDialect.cpp
  lib/Dialect/CouplingGraph/CGContract.cpp
  lib/Analysis/ClosedForm.cpp
  lib/Analysis/AffineRelation.cpp
  lib/Analysis/ISLContext.cpp
  lib/Analysis/QuasiPolynomial.cpp
  lib/Analysis/CouplingRelation.cpp
  lib/Analysis/Analysis.cpp
  lib/Analysis/AccessRelation.cpp
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
  lib/Solver/ListScheduler.cpp
  lib/Solver/CostModel.cpp
  lib/Solver/ModelDescription.cpp
  lib/Codegen/Codegen.cpp
  lib/Runtime/EventBuffer.cpp
  lib/Runtime/MegakernelRuntime.cpp)
add_library(TileMega::tilemega ALIAS tilemega)
target_compile_features(tilemega PUBLIC cxx_std_17)
target_include_directories(tilemega PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/test>)
add_dependencies(tilemega TileMegaCGIncGen)
target_link_libraries(tilemega PUBLIC CUDA::cudart MLIRIR MLIRSupport
  tilemega::isl_stack)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega PRIVATE -fno-rtti)
endif()

add_executable(tilemega-opt tools/tilemega-opt.cpp)
add_dependencies(tilemega-opt TileMegaCGIncGen)
target_link_libraries(tilemega-opt PRIVATE TileMega::tilemega MLIROptLib)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega-opt PRIVATE -fno-rtti)
endif()
set_target_properties(tilemega-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-import tools/tilemega-import.cpp)
add_dependencies(tilemega-import TileMegaCGIncGen)
target_link_libraries(tilemega-import PRIVATE TileMega::tilemega MLIRIR MLIRSupport)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega-import PRIVATE -fno-rtti)
endif()
set_target_properties(tilemega-import PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-compile tools/tilemega-compile.cpp)
add_dependencies(tilemega-compile TileMegaCGIncGen)
target_link_libraries(tilemega-compile PRIVATE TileMega::tilemega MLIRIR MLIRParser MLIRSupport)
target_compile_definitions(tilemega-compile PRIVATE
  TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilemega-compile PRIVATE -fno-rtti)
endif()
set_target_properties(tilemega-compile PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# The microbenchmarks live in their own library: GemmCalibration.cu
# instantiates six CUTLASS collectives, and putting that in `tilemega` would
# make every build of the core library pay for it.
add_library(tilemega_calibration STATIC
  lib/Target/Calibration.cu
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

add_executable(tilemega-calibrate tools/tilemega-calibrate.cpp)
target_link_libraries(tilemega-calibrate PRIVATE tilemega_calibration)
set_target_properties(tilemega-calibrate PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-derive tools/tilemega-derive.cpp)
target_link_libraries(tilemega-derive PRIVATE TileMega::tilemega)
set_target_properties(tilemega-derive PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-shape-probe tools/tilemega-shape-probe.cpp)
target_link_libraries(tilemega-shape-probe PRIVATE TileMega::tilemega)
target_compile_definitions(tilemega-shape-probe PRIVATE
  TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(tilemega-shape-probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-wiring tools/tilemega-wiring.cpp)
target_link_libraries(tilemega-wiring PRIVATE TileMega::tilemega)
target_compile_definitions(tilemega-wiring PRIVATE
  TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(tilemega-wiring PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-identity-probe tools/tilemega-identity-probe.cpp)
target_link_libraries(tilemega-identity-probe PRIVATE TileMega::tilemega)
target_compile_definitions(tilemega-identity-probe PRIVATE
  TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(tilemega-identity-probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-costmodel tools/tilemega-costmodel.cpp)
target_link_libraries(tilemega-costmodel PRIVATE TileMega::tilemega)
set_target_properties(tilemega-costmodel PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-target-audit tools/tilemega-target-audit.cpp)
target_link_libraries(tilemega-target-audit PRIVATE TileMega::tilemega)
target_compile_definitions(tilemega-target-audit PRIVATE
  TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(tilemega-target-audit PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_executable(tilemega-solve tools/tilemega-solve.cpp)
target_link_libraries(tilemega-solve PRIVATE TileMega::tilemega)
set_target_properties(tilemega-solve PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_custom_target(check-policy
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckPolicy.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Checking architecture-dispatch and hardware-constant policy")

if(TILEMEGA_BUILD_TESTS)
  enable_testing()
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
  add_executable(target_spec_test test/unit/target_spec_test.cpp)
  target_link_libraries(target_spec_test PRIVATE TileMega::tilemega)
  target_compile_definitions(target_spec_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME target_spec COMMAND target_spec_test)
  add_test(NAME target_audit COMMAND tilemega-target-audit)

  add_executable(coupling_types_test test/unit/coupling_types_test.cpp)
  target_link_libraries(coupling_types_test PRIVATE TileMega::tilemega)
  add_test(NAME coupling_types COMMAND coupling_types_test)

  add_executable(table27_test test/unit/table27_test.cpp)
  target_link_libraries(table27_test PRIVATE TileMega::tilemega)
  add_test(NAME table27 COMMAND table27_test)

  add_executable(semantics_test test/unit/semantics_test.cpp)
  target_link_libraries(semantics_test PRIVATE TileMega::tilemega)
  add_test(NAME semantics COMMAND semantics_test)

  add_executable(containment_test test/unit/containment_test.cpp)
  target_link_libraries(containment_test PRIVATE TileMega::tilemega)
  add_test(NAME containment COMMAND containment_test)

  add_executable(event_synthesis_test test/unit/event_synthesis_test.cpp)
  target_link_libraries(event_synthesis_test PRIVATE TileMega::tilemega)
  add_test(NAME event_synthesis COMMAND event_synthesis_test)

  add_executable(isl_relation_test test/unit/isl_relation_test.cpp)
  target_link_libraries(isl_relation_test PRIVATE TileMega::tilemega)
  add_test(NAME isl_relation COMMAND isl_relation_test)

  add_executable(layout_bridge_test test/unit/layout_bridge_test.cpp)
  target_link_libraries(layout_bridge_test PRIVATE TileMega::tilemega)
  add_test(NAME layout_bridge COMMAND layout_bridge_test)

  add_executable(cg_attr_test test/unit/cg_attr_test.cpp)
  add_dependencies(cg_attr_test TileMegaCGIncGen)
  target_link_libraries(cg_attr_test PRIVATE TileMega::tilemega MLIRIR MLIRParser)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(cg_attr_test PRIVATE -fno-rtti)
  endif()
  add_test(NAME cg_attr_roundtrip COMMAND cg_attr_test)

  add_executable(backend_query_test test/unit/backend_query_test.cpp)
  target_link_libraries(backend_query_test PRIVATE TileMega::tilemega)
  target_compile_definitions(backend_query_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME backend_query COMMAND backend_query_test)

  add_executable(chain_dp_test test/unit/chain_dp_test.cpp)
  target_link_libraries(chain_dp_test PRIVATE TileMega::tilemega)
  target_compile_definitions(chain_dp_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME chain_dp COMMAND chain_dp_test)

  add_executable(list_scheduler_test test/unit/list_scheduler_test.cpp)
  target_link_libraries(list_scheduler_test PRIVATE TileMega::tilemega)
  add_test(NAME list_scheduler COMMAND list_scheduler_test)

  add_executable(cluster_labeling_test test/unit/cluster_labeling_test.cpp)
  target_link_libraries(cluster_labeling_test PRIVATE TileMega::tilemega)
  add_test(NAME cluster_labeling COMMAND cluster_labeling_test)

  add_executable(alignment_propagation_test
    test/unit/alignment_propagation_test.cpp)
  target_link_libraries(alignment_propagation_test PRIVATE TileMega::tilemega)
  add_test(NAME alignment_propagation COMMAND alignment_propagation_test)

  add_executable(impl_contract_test test/unit/impl_contract_test.cpp)
  target_link_libraries(impl_contract_test PRIVATE TileMega::tilemega)
  add_test(NAME impl_contract COMMAND impl_contract_test)

  add_executable(task_body_contract_test test/unit/task_body_contract_test.cu)
  target_link_libraries(task_body_contract_test PRIVATE TileMega::tilemega)
  add_test(NAME task_body_contract COMMAND task_body_contract_test)

  add_executable(frontend_import_test test/unit/frontend_import_test.cpp)
  add_dependencies(frontend_import_test TileMegaCGIncGen)
  target_link_libraries(frontend_import_test PRIVATE TileMega::tilemega MLIRIR)
  target_compile_definitions(frontend_import_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(frontend_import_test PRIVATE -fno-rtti)
  endif()
  add_test(NAME frontend_import COMMAND frontend_import_test)

  add_executable(semantic_lifting_test test/unit/semantic_lifting_test.cpp)
  add_dependencies(semantic_lifting_test TileMegaCGIncGen)
  target_link_libraries(semantic_lifting_test PRIVATE TileMega::tilemega MLIRIR)
  target_compile_definitions(semantic_lifting_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(semantic_lifting_test PRIVATE -fno-rtti)
  endif()
  add_test(NAME semantic_lifting COMMAND semantic_lifting_test)

  add_executable(wiring_coupling_test test/unit/wiring_coupling_test.cpp)
  target_link_libraries(wiring_coupling_test PRIVATE TileMega::tilemega)
  target_compile_definitions(wiring_coupling_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME wiring_coupling COMMAND wiring_coupling_test)

  add_executable(degradation_test test/unit/degradation_test.cpp)
  add_dependencies(degradation_test TileMegaCGIncGen)
  target_link_libraries(degradation_test PRIVATE TileMega::tilemega MLIRIR)
  target_compile_definitions(degradation_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(degradation_test PRIVATE -fno-rtti)
  endif()
  add_test(NAME frontend_degradation COMMAND degradation_test)

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
