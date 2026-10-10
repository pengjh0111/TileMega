# R14 closeout integration evidence

R14 frozen source: `97f7a2f1d`; accepted supplemental GPU results bind this source.
Preserved remote: `a3f0dc6d10c368dbbe3073c66cc750aec92a70a3`, including the
user's DNN/MoE merge parent `0e64c56b010488a7229a75ed79814fb865aef197`.
Integration commits: `e28d090ce` (source compatibility merge), `8f4f7a6f0`
(the existing R14 acceptance commit merged without dropping DM findings).

`acceptance.json` records completed host/Python checks, compile-only inputs,
the final compiler source fingerprint, artifact hashes and every archive member.
`checks.tar.xz` retains the original logs, commands, compatibility proof and
the previously untracked GQA host fixture used by the handoff test. Binaries,
object files and generated CUDA source are registered by hash rather than copied.

The 13 R14 branch bodies match their frozen originals byte for byte. Twelve
DM headers match the remote byte for byte; DmServingRuntime adds three optional
host-only introspection exports. TILEMEGA_DM_SUPPORT selects the contract.
Separate Python identity APIs retain both schemas without rewriting artifact IDs.
This temporary duplication avoids conflating independently evolved template ABIs.

The original logs preserve initial integration errors and their successful
rechecks. A supplementary 40-family MoE CPU search did not finish within the
300-second command window; five later optional host checks were not reached.
Only completed checks are counted. No CUDA kernels or performance measurements
were launched after integration, and no new synchronization conclusion is drawn.
