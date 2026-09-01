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
option(TILEMEGA_ENABLE_ISL "Enable ISL analysis (Phase 3)" OFF)
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
  lib/Target/TargetSpec.cpp
  lib/Frontend/Frontend.cpp
  lib/Dialect/CouplingGraph/CGDialect.cpp
  lib/Analysis/ClosedForm.cpp
  lib/Analysis/AffineRelation.cpp
  lib/Analysis/Analysis.cpp
  lib/Solver/Solver.cpp
  lib/Codegen/Codegen.cpp
  lib/Runtime/EventBuffer.cpp
  lib/Runtime/MegakernelRuntime.cpp)
add_library(TileMega::tilemega ALIAS tilemega)
target_compile_features(tilemega PUBLIC cxx_std_17)
target_include_directories(tilemega PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include>)
add_dependencies(tilemega TileMegaCGIncGen)
target_link_libraries(tilemega PUBLIC CUDA::cudart MLIRIR MLIRSupport)
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

add_executable(tilemega-calibrate tools/tilemega-calibrate.cpp)
target_link_libraries(tilemega-calibrate PRIVATE TileMega::tilemega)
set_target_properties(tilemega-calibrate PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_custom_target(check-policy
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckPolicy.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Checking architecture-dispatch and hardware-constant policy")

if(TILEMEGA_BUILD_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_file(TILEMEGA_LIT_DRIVER lit.py HINTS
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/cutlass_compiler/external/llvm-project/llvm/utils/lit"
    REQUIRED)
  configure_file(test/lit.site.cfg.py.in test/lit.site.cfg.py @ONLY)
  add_custom_target(check-cg-lit
    COMMAND ${Python3_EXECUTABLE} ${TILEMEGA_LIT_DRIVER}
      -sv ${CMAKE_CURRENT_BINARY_DIR}/test
    DEPENDS tilemega-opt
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_executable(target_spec_test test/unit/target_spec_test.cpp)
  target_link_libraries(target_spec_test PRIVATE TileMega::tilemega)
  target_compile_definitions(target_spec_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME target_spec COMMAND target_spec_test)

  add_executable(coupling_types_test test/unit/coupling_types_test.cpp)
  target_link_libraries(coupling_types_test PRIVATE TileMega::tilemega)
  add_test(NAME coupling_types COMMAND coupling_types_test)

  add_executable(cg_attr_test test/unit/cg_attr_test.cpp)
  add_dependencies(cg_attr_test TileMegaCGIncGen)
  target_link_libraries(cg_attr_test PRIVATE TileMega::tilemega MLIRIR)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(cg_attr_test PRIVATE -fno-rtti)
  endif()
  add_test(NAME cg_attr_roundtrip COMMAND cg_attr_test)

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
if(TARGET coupling_types_test)
  add_dependencies(check-tilemega coupling_types_test)
endif()
