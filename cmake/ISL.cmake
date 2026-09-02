# SPDX-License-Identifier: BSD-3-Clause
# Locates a pre-built isl + polylib + barvinok stack and exposes it as the
# INTERFACE target `tilemega::isl_stack`.
#
# These are autotools projects, not CMake ones, and this file deliberately
# does not drive their build (no ExternalProject_Add): the rest of this repo
# already uses the "build once out-of-tree, point CMake at the result" model
# for MLIR (see docs/BUILD_MLIR.md and the -DMLIR_DIR= convention), and the
# same model is used here for consistency and because autoreconf/libtool
# integration inside ExternalProject_Add is its own source of fragility this
# project does not need. See docs/DEPENDENCIES.md for the exact recipe
# (`third_party/barvinok`'s bundled isl/polylib submodules, `autogen.sh` +
# `configure --with-int=gmp --with-pic` + `make`, one out-of-tree build dir
# per component).
#
# Expected build directories (overridable):
#   TILEMEGA_ISL_BUILD_DIR      (default: ${CMAKE_SOURCE_DIR}/build-isl)
#   TILEMEGA_POLYLIB_BUILD_DIR  (default: ${CMAKE_SOURCE_DIR}/build-polylib)
#   TILEMEGA_BARVINOK_BUILD_DIR (default: ${CMAKE_SOURCE_DIR}/build-barvinok)

set(TILEMEGA_ISL_BUILD_DIR "${CMAKE_SOURCE_DIR}/build-isl" CACHE PATH
  "Out-of-tree isl autotools build directory (see docs/DEPENDENCIES.md)")
set(TILEMEGA_POLYLIB_BUILD_DIR "${CMAKE_SOURCE_DIR}/build-polylib" CACHE PATH
  "Out-of-tree polylib autotools build directory")
set(TILEMEGA_BARVINOK_BUILD_DIR "${CMAKE_SOURCE_DIR}/build-barvinok" CACHE PATH
  "Out-of-tree barvinok autotools build directory")

set(_tilemega_isl_src "${CMAKE_SOURCE_DIR}/third_party/barvinok/isl")
set(_tilemega_polylib_src "${CMAKE_SOURCE_DIR}/third_party/barvinok/polylib")
set(_tilemega_barvinok_src "${CMAKE_SOURCE_DIR}/third_party/barvinok")

find_library(TILEMEGA_ISL_LIBRARY NAMES isl
  PATHS "${TILEMEGA_ISL_BUILD_DIR}/.libs" NO_DEFAULT_PATH)
find_library(TILEMEGA_POLYLIB_LIBRARY NAMES polylibgmp
  PATHS "${TILEMEGA_POLYLIB_BUILD_DIR}/.libs" NO_DEFAULT_PATH)
find_library(TILEMEGA_BARVINOK_LIBRARY NAMES barvinok
  PATHS "${TILEMEGA_BARVINOK_BUILD_DIR}/.libs" NO_DEFAULT_PATH)
find_path(TILEMEGA_ISL_GENERATED_INCLUDE_DIR isl/stdint.h
  PATHS "${TILEMEGA_ISL_BUILD_DIR}/include" NO_DEFAULT_PATH)
find_library(TILEMEGA_NTL_LIBRARY NAMES ntl)
find_library(TILEMEGA_GMP_LIBRARY NAMES gmp)

if(NOT TILEMEGA_ISL_LIBRARY OR NOT TILEMEGA_POLYLIB_LIBRARY OR
   NOT TILEMEGA_BARVINOK_LIBRARY OR NOT TILEMEGA_ISL_GENERATED_INCLUDE_DIR)
  message(FATAL_ERROR
    "TILEMEGA_ENABLE_ISL=ON but the isl/polylib/barvinok build products were "
    "not found under build-isl/build-polylib/build-barvinok (or the "
    "TILEMEGA_*_BUILD_DIR overrides). Build them first -- see "
    "docs/DEPENDENCIES.md for the exact autogen.sh/configure/make recipe -- "
    "then reconfigure.")
endif()
if(NOT TILEMEGA_NTL_LIBRARY OR NOT TILEMEGA_GMP_LIBRARY)
  message(FATAL_ERROR
    "TILEMEGA_ENABLE_ISL=ON needs system libntl and libgmp (barvinok links "
    "both). Install libntl-dev and libgmp-dev.")
endif()

add_library(tilemega_isl_stack INTERFACE)
target_include_directories(tilemega_isl_stack SYSTEM INTERFACE
  "${_tilemega_isl_src}/include"
  "${TILEMEGA_ISL_GENERATED_INCLUDE_DIR}"
  "${_tilemega_polylib_src}/include"
  "${_tilemega_barvinok_src}"
  "${_tilemega_barvinok_src}/barvinok"
  "${TILEMEGA_BARVINOK_BUILD_DIR}")
# Static-link order matters (barvinok -> isl -> polylib -> ntl -> gmp); this
# exact order was validated by docs/experiments/P3_ISL/crosslink_probe.cpp.
target_link_libraries(tilemega_isl_stack INTERFACE
  "${TILEMEGA_BARVINOK_LIBRARY}"
  "${TILEMEGA_ISL_LIBRARY}"
  "${TILEMEGA_POLYLIB_LIBRARY}"
  "${TILEMEGA_NTL_LIBRARY}"
  "${TILEMEGA_GMP_LIBRARY}")
add_library(tilemega::isl_stack ALIAS tilemega_isl_stack)

message(STATUS "ISL stack: isl=${TILEMEGA_ISL_LIBRARY} "
  "polylib=${TILEMEGA_POLYLIB_LIBRARY} barvinok=${TILEMEGA_BARVINOK_LIBRARY}")
