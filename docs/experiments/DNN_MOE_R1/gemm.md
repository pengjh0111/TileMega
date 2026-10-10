# DM-1 additional GEMM tiles

verified: the host legality/storage contract passes the six registered host
checks and check-policy (`results/CI7_gemm_traits_host.json`). It admits 76
M/N/K geometries at stages >= 2, including 46 additions to serving's original
domain. `ServingBF16ShapeLegal` retains the original LLM domain; the new
contract is explicitly named `DmServingBF16ShapeLegal`.

verified: `results/CI7_small_gemm.json` seals 54 configurations, 126 builds
and 250/250 fresh processes. Every added geometry passes on sm_89; narrow-N
warp layouts compile on all five targets. The four 16x16 K geometries each
pass 50/50 fresh processes. All transitive source hashes stayed unchanged.
Production task dispatch, operands and skeleton search integration are pending.

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
