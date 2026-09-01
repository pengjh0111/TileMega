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
project(TileMega VERSION 0.1.0 LANGUAGES CXX CUDA)

option(TILEMEGA_BUILD_TESTS "Build unit and integration tests" ON)
option(TILEMEGA_BUILD_VERIFY "Build verification experiments" ON)
option(TILEMEGA_ENABLE_MLIR "Enable MLIR frontend and dialect (Phase 1)" OFF)
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

add_library(tilemega STATIC
  lib/Target/TargetSpec.cpp
  lib/Frontend/Frontend.cpp
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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include>)
target_link_libraries(tilemega PUBLIC CUDA::cudart)

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
  add_executable(target_spec_test test/unit/target_spec_test.cpp)
  target_link_libraries(target_spec_test PRIVATE TileMega::tilemega)
  target_compile_definitions(target_spec_test PRIVATE
    TILEMEGA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME target_spec COMMAND target_spec_test)

  add_executable(coupling_types_test test/unit/coupling_types_test.cpp)
  target_link_libraries(coupling_types_test PRIVATE TileMega::tilemega)
  add_test(NAME coupling_types COMMAND coupling_types_test)
endif()

if(TILEMEGA_BUILD_VERIFY)
  add_subdirectory(docs/experiments/V_I)
endif()

add_custom_target(check-tilemega DEPENDS check-policy)
if(TARGET target_spec_test)
  add_dependencies(check-tilemega target_spec_test)
endif()
if(TARGET coupling_types_test)
  add_dependencies(check-tilemega coupling_types_test)
endif()
