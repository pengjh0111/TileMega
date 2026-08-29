# L3 分析层：ISL 只做 task 间（骨架原则一）。
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

# ISLContext 用 isl_ctx_last_error_msg，该接口要求 isl >= 0.23。
# barvinok 自带的 isl 是 0.28，满足。
