# DM-1 additional GEMM tiles

verified: the host legality/storage contract passes the six registered host
checks and check-policy (`results/CI7_gemm_traits_host.json`). It admits 76
M/N/K geometries at stages >= 2, including 46 additions to serving's original
domain. `ServingBF16ShapeLegal` retains the original LLM domain; the new
contract is explicitly named `DmServingBF16ShapeLegal`.

The new `ServingDmGemm` mainloop is currently under numerical validation;
it is not yet connected to plan candidates or production task dispatch.
`runs/dm1-ci7-small-gemm-families` checks every added geometry on sm_89 and
compiles the narrow-N warp layouts on all five targets. The four 16x16 K
geometries each require 50 fresh processes. Transitive source hashes must
remain unchanged throughout the check.

inferred implementation invariants:

- Shared-memory atoms are 8x16, 8x32 or 8x64, with the corresponding CuTe
  swizzle. Each copy vector carries eight BF16 elements; invalid M/N/K elements
  are zero-filled. Tiled B already contains padded zero elements.
- TN=16 uses four M warps for TM>=64, two M/two N warps for TM=32, or two N/two
  K warps for TM=16. Every case has four computing warps.
- The 16x16 K groups own alternating 16-element reduction blocks. Their C
  fragments write separate FP32 scratch banks, converge, then add the banks
  once. A 64-thread operand layout lets TK=16 represent its actual extent.
- Committed copies prime the bounded ring before the first MMA. The next
  operand fragment is loaded before the current fragment's MMA. A slot is
  overwritten after all readers of its previous contents converge; all
  committed copies finish before scratch reuse.
- Storage is the maximum of staged A/B, FP32 K partials, and the output tile
  plus two FP32 statistics per row. Allocation and solver traits share a
  compile-time equality check. Target resource limits remain separate policy
  inputs.

The standalone oracle compares BF16-rounded outputs with an independently
accumulated FP32 reference using the original DM-1 tolerance. It also compares
row-major and tiled-B FP32 results bitwise, checks split offsets and tails,
and retains output canaries. This is a mainloop check; finite epilogue chains,
page-ring integration, im2col, rowgather, expert access and model gates remain
pending. No performance or calibrated coefficient is reported here.
