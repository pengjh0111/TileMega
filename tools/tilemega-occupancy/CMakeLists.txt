# 只依赖 CUDA driver API，不依赖 MLIR —— 保持它能在没构建编译器时也可用。
add_executable(tilemega-occupancy tilemega-occupancy.cpp)
target_link_libraries(tilemega-occupancy PRIVATE CUDA::cuda_driver)
