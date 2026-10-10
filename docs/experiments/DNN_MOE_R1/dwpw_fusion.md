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

Pending: the complete MobileNetV1 fused CUDA emission exceeded a 300-second
host-build limit. General large-graph analysis cost, per-pair joint solver
selection and complete-model fusion coverage remain open.
