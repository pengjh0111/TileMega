# TileMega library layers. Dependencies between them are strictly one way:
#   Frontend -> Analysis -> Partition -> Codegen -> Runtime
# Partition may only reach back into Codegen/Backend through BackendCostQuery
# (skeleton principle two).
add_subdirectory(Analysis)
add_subdirectory(Partition)
add_subdirectory(Codegen)
add_subdirectory(Runtime)
add_subdirectory(Frontend)
