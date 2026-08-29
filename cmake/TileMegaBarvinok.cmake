# ==============================================================================
# barvinok / isl 定位
#
# barvinok 是 autotools 工程，把它塞进 ninja 构建图（ExternalProject）会带来
# 难看的顺序依赖，而它是个几乎不变的叶子依赖。所以：用 scripts/build_barvinok.sh
# 一次性构建安装到 third_party/.install/barvinok，这里只负责找到它。
#
# 依赖解析走 pkg-config：barvinok 装出来的 barvinok.pc 里已经有权威的链接行
#   -lbarvinok -lpolylibgmp -lisl -lntl -lgmp
# 手工维护这串（polylib / NTL / GMP / pthread）既容易漏，又会在上游改依赖时
# 静默失效——首次尝试就漏了 polylib，报的是一堆 Vector_Free 未定义。
#
# 接口选择（骨架 Phase 0 决策）：analysis 层直接用 isl 的 C API + 自建 RAII
# 封装，不用 isl-noexceptions.h。见 include/tilemega/Analysis/ISLContext.h。
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

  # 只看我们自己装的那份，不要意外命中系统里的 isl
  #（系统有 libisl.so.23，版本未必匹配 barvinok）。
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
