# Generated depthwise-pointwise fusion

verified: `ApplyDnnDwPwFusion` selects pointwise consumers individually. It
retains the original depthwise materialization when an output or another
consumer observes it. Fused tasks recompute a private BF16 depthwise tile;
they do not write the shared intermediate. SimpleGate/SCA depthwise stages
are excluded from this MobileNet transformation.

verified: the implicit spatial windows in L-sem expand to private bounded
iteration dimensions. Exact read relations match independent enumeration
for stride, dilation, halo, non-divisible linear tiles and split-K. Every
split-K chunk and pointwise N tile reads/recomputes all depthwise channels.
The GEMM reduction still covers only that chunk's pointwise K interval.
Shared-memory legality includes both private depthwise storage and GEMM
workspace, including the paged workspace/page-ring partition.

verified: `results/DN_dwpw_generated_smoke.json` contains the two generated
artifacts and their identities. One fixed seed supplies the same synthetic
weights/input to the untouched exported depthwise→ReLU6→pointwise→ReLU graph.
The nonpaged artifact uses split-K=2. The paged artifact uses its own legal
geometry. Each runs one epoch in L1 and L2, with bitwise equality between
modes, exact equality to BF16 PyTorch, and maximum FP32 error
0.0025449395179748535. This is execution/numerical evidence; no repeated-process
race claim, full-model gate or latency claim follows.

verified: the initial artifact was rejected before execution because forward
ABI counting omitted `kDwPwFused`. That check is repaired. The generic model
cosine checker then rejected all-zero ReLU vectors (cosine=0 even for identical
zeros). Operator fixtures now explicitly select the specified elementwise
BF16 tolerance; model cosine thresholds are unchanged. Both failed receipts
remain in their immutable run directories.

verified: both artifacts compile for sm_80/89/90/100/120. Resource receipts
include stack frames and spills; the canonical L2 nonpaged kernel spills
4/4 store/load bytes, and the page-loop kernel spills 2038/5008 bytes.
No performance conclusion follows from these resources.

verified: private depthwise arithmetic has its own L-sem output domain and
reduction extent. Its SIMT dot and affine-chain work is charged once per
pointwise N tile and split-K chunk, including M tails, separately from the
pointwise MMA lane. Split-K combiners clear this prologue to avoid charging it
again. The generated fixture's host pricing/codec assertions pass.

verified: after exact-analysis repairs, the complete MobileNetV1 fused graph
generates with 13 fused pairs (`runs/dm1-mbv1-fused-host-v18`). Its frozen native
sm_89 artifact executes one fixed synthetic state/input in both L1 and L2,
with bitwise identical results, FP32 minimum cosine 0.9999978542 and maximum
error 0.0005717725. `results/DN_mbv1_full_fused_generated.json` records identity
7c6ba678… and all compiler resource/spill lines. The canonical L2 kernel has
255 registers and aggregate spill stores/loads 280/1016 bytes. Earlier host
timeouts remain recorded. This is a complete-graph execution smoke, not G-DNN.

Pending: general large-graph analysis cost, per-pair joint solver selection
and full MobileNetV2 fusion coverage remain open.
