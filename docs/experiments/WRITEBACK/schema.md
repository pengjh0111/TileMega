# R6 symbolic placement carrier

Verified by `schema_tests.log`: legacy syntax and symbolic theta functions
round-trip, multivalued parameters are rejected, and grid/domain violations
are fatal. Maps use the existing ISL-backed coupling-map attribute: this is
a representation reuse, not a change to dependency semantics.

`params_map` replaces `params` and returns exactly the mode's parameter arity.
`grid_map` and `resident_limit_map` have the same theta domain and jointly
prove `0 < grid <= resident_limit`. Code generation requires explicit integer
`tilemega.placement_bindings` before consuming a symbolic parameter vector.
The target-specific resident limit must still be checked by the solving pass;
the schema cannot turn a supplied limit into a device-capacity measurement.

Materialized placement stays in the existing module-level CG attribute
`tilemega.placement_table`. This avoids duplicating a model-wide table on
every PlacementOp and preserves the existing Codegen/RuntimeVariantDesc
consumer. Its bound seq/past/grid remain mandatory. Symbolic parameters do
not authorize transplanting a materialized table to another grid.
