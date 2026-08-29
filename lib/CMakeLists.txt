# TileMega 库层。各子层的依赖方向严格单向：
#   Frontend → Analysis → Partition → Codegen → Runtime
# Partition 只能通过 BackendCostQuery 反向触碰 Codegen/Backend（骨架原则二）。
add_subdirectory(Analysis)
add_subdirectory(Partition)
add_subdirectory(Codegen)
add_subdirectory(Runtime)
add_subdirectory(Frontend)
