# ==============================================================================
# Locating barvinok / isl
#
# barvinok is an autotools project. Wiring it into the ninja build graph via
# ExternalProject would create ugly ordering dependencies, and it is an almost
# unchanging leaf dependency. So: scripts/build_barvinok.sh builds and installs
# it once into third_party/.install/barvinok, and this module only finds it.
#
# Dependency resolution goes through pkg-config: the installed barvinok.pc
# already carries the authoritative link line
#   -lbarvinok -lpolylibgmp -lisl -lntl -lgmp
# Maintaining that list by hand (polylib / NTL / GMP / pthread) is both easy to
# get wrong and silently stale when upstream changes its dependencies -- the
# first attempt here omitted polylib and produced a wall of undefined
# Vector_Free references.
#
# Interface choice (Phase 0 decision): the analysis layer uses isl's C API with
# a hand-written RAII wrapper, not isl-noexceptions.h. See
# include/tilemega/Analysis/ISLContext.h.
# ==============================================================================
include_guard(GLOBAL)

set(TILEMEGA_BARVINOK_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/.install/barvinok"
    CACHE PATH "barvinok/isl 的安装前缀")

function(tilemega_find_barvinok)
  find_package(PkgConfig REQUIRED)

  set(_pcdir "${TILEMEGA_BARVINOK_PREFIX}/lib/pkgconfig")
  if(NOT EXISTS "${_pcdir}/barvinok.pc")
    message(FATAL_ERROR
      "TileMega: 在 ${TILEMEGA_BARVINOK_PREFIX} 下找不到 barvinok。\n"
      "  先执行：  ./scripts/build_barvinok.sh\n"
      "  或用 -DTILEMEGA_ENABLE_ISL=OFF 暂时关掉分析层。")
  endif()

  # Look only at the copy we installed ourselves, so we do not accidentally
  # pick up a system isl (the host has libisl.so.23, whose version need not
  # match barvinok).
  set(ENV{PKG_CONFIG_PATH} "${_pcdir}")
  pkg_check_modules(TILEMEGA_BARVINOK REQUIRED IMPORTED_TARGET barvinok)
  pkg_check_modules(TILEMEGA_ISL      REQUIRED IMPORTED_TARGET isl)

  add_library(TileMega::isl      ALIAS PkgConfig::TILEMEGA_ISL)
  add_library(TileMega::barvinok ALIAS PkgConfig::TILEMEGA_BARVINOK)

  message(STATUS "TileMega: isl      ${TILEMEGA_ISL_VERSION} "
                 "(${TILEMEGA_ISL_LIBRARY_DIRS})")
  message(STATUS "TileMega: barvinok ${TILEMEGA_BARVINOK_VERSION} "
                 "libs=${TILEMEGA_BARVINOK_LIBRARIES}")
endfunction()
