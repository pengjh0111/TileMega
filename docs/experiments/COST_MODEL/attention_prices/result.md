# Initial Scalar Evaluation Attempt

This run is intentionally incomplete. gqa2 chunk1/2 passed plan/resource
reconciliation. Chunk4 spent more than six minutes repeatedly parsing a
piecewise polynomial for individual task coordinates inside DP evaluation.
Two debugger stack samples identified `QuasiPolynomial::BindCoordinates`
under `CostModel::TaskCostNs(AttentionPhaseWork)` and `ChainDP::BetweenNs`.
The process was explicitly terminated with SIGTERM to apply exact batch
evaluation; this is neither a GPU hang nor a correctness failure.

The completed rows are retained as price-bit controls for `attention_prices_batch`.
The initial preliminary tool used tile ownership rather than the archived
element ownership; that mismatch was found before this frozen run and fixed.
No comparison against incompatible ownership is claimed.
