# SM90R3 final serving configurations

One file per accepted cell (25 dense + 5 MoE). Each records the solve command, the
solver's selected tmexec coordinates, the source HEAD, and the generated-source and
binary hashes, so the cell rebuilds from a single HEAD. `measured` holds the P-FINAL
result and `correctness` the P-CORR outcome. Dense prefills are the R2 closeout
options rebuilt at this HEAD; the MoE prefill stays the frozen R1 legacy binary whose
expert pages match the native decode.
