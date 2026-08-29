add_executable(tilemega-isl-smoke ISLContextSmokeTest.cpp)
target_link_libraries(tilemega-isl-smoke PRIVATE TileMegaAnalysis)
add_test(NAME isl-smoke COMMAND tilemega-isl-smoke)
