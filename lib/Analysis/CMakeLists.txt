# L3 analysis layer: ISL is used between tasks only (skeleton principle one).
if(NOT TILEMEGA_ENABLE_ISL)
  add_library(TileMegaAnalysis INTERFACE)
  return()
endif()

add_library(TileMegaAnalysis STATIC
  ISLContext.cpp
)
target_link_libraries(TileMegaAnalysis
  PUBLIC  TileMega::isl TileMega::barvinok
  PRIVATE MLIRIR MLIRSupport)
target_include_directories(TileMegaAnalysis
  PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ISLContext uses isl_ctx_last_error_msg, which requires isl >= 0.23.
# The isl bundled with barvinok is 0.28, so this is satisfied.
