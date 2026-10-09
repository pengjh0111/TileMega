Verified on nvcc 12.8.93, `-std=c++17 -O3 --expt-relaxed-constexpr`, sm_89:
the host results of all three sources are `9,10`. The unbraced CUDA call
variant returns `1,10`; the unbraced declaration variant fails compilation
because `next` is undefined in the true instantiation. The explicitly braced
variant returns `9,10` on host and device. The expected values are fixed by
ordinary integer addition and were not changed to accommodate nvcc.

The original sources, build diagnostics, compiler version, complete macro
captures, resource lines, binary hashes and execution observations are sealed
in [CI2_nvcc_constexpr_scope.json](../../results/CI2_nvcc_constexpr_scope.json).
The two unbraced files retain the counterexample and are excluded from the
normal build. This is a scalar compiler observation, with no race or
synchronization claim and no claim about other CUDA versions.

For a manual reproduction, compile each `.cu` with the flags above and run
the resulting executable. Both GPU compilation and execution must use
`flock "$TILEMEGA_GPU_LOCK"`; do not collect timing data from these probes.
An ordinary C++ host comparison can use `c++ -x c++ -std=c++17 -O3`.
Production convolution copy branches use explicit compound blocks. Their
numerical and synchronization checks are separate from this counterexample.
