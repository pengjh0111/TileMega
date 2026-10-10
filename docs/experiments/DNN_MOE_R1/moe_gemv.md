# Functional MoE GEMV family

verified: `--moe-gemv 1` generates and runs gathered, indirectly bound expert
GEMV in both nonpaged and paged MoE regions. One fixed synthetic T=1 slot input
passes the independent reference and routing checks in L1/L2; the two modes
are bit-identical. Receipts and per-kernel spill/resource records are in
`results/MO_gemv_generated_smoke.json`. Whole-megakernel kernels spill; the thin
GEMV compile probes have zero spill stores/loads. This is one-process functional
evidence, not a synchronization pass-rate claim or a performance comparison.

The backend uses four warps with FP32 dot accumulation and deterministic warp
reductions, then the existing finite BF16 epilogue. Up to four live rows retain
the MMA family's physical M padding. Gathered row indices, indirect expert B,
per-token deferred RMSNorm, SwiGLU and row scatter use the existing resolved
binding/epilogue paths. Dense row-major, tiled B and page-ring B are supported;
page release follows convergence of all B readers. Im2col and A-scale are
excluded. Fixed builds select TM16/TN32/TK64/stages2/split1 for eligible GEMMs.

verified: compile-only probes instantiate TN8/16/32 dense and TN32 paged on
sm_80/89/90/100/120 through `arch::Caps`. TN8 is a backend probe; the finite DM
plan epilogue/candidate domain has not been extended to TN8. The initial probe
failed because `CurrentArch` is void during nvcc host parsing; the corrected
probe uses the explicit generated architecture identifier.

verified: predicted structural search now jointly evaluates slot and four group
BM families with MMA/GEMV (10 states), rescoring geometry and using actual
family resource probes. The selected family survives plan rebuilding and is
emitted into CUDA, manifest and identity. Host search/final CUDA generation and
650 occupancy price checks pass. GEMV applies only to eligible conditioned live
rows; remaining rows retain MMA. Scalar dot/reduction/barrier costs use target
rates; their accuracy is inferred, with no invented empirical calibration.

verified: the hybrid GEMV/MMA actual-body sm_89 resource probe compiles with
128 registers, 12,288 shared bytes and 128 threads. The entry has a 288-byte
stack and zero direct spills, but its out-of-line RunDmResolved callee has
296-byte spill stores and 416-byte spill loads. This hybrid probe is not the
zero-spill thin GEMV probe reported above. See
`results/CI7_moe_gemv_family_host.json` for evidence and source identity.
