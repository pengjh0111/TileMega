# Depends only on the CUDA driver API, not MLIR, so it stays usable even when
# the compiler has not been built.
add_executable(tilemega-occupancy tilemega-occupancy.cpp)
target_link_libraries(tilemega-occupancy PRIVATE CUDA::cuda_driver)
