# Deferred DNN LayerNorm implementation

verified: the host rewriter removes 24 BERT and 68 NAFNet normalization
stages. Externally observed outputs, row selection, channel-incomplete
consumers and scaled residuals retain explicit normalization. The four NAFNet
normalizations following PixelShuffle remain explicit: their producer column
ownership differs from the shuffled pixel ownership.

inferred: complete channel contraction with unchanged logical rows proves
that every rewritten consumer reads exactly one normalization row. CG lifting
records the raw value and all statistics parts as explicit reads; ordinary
exact dependency derivation handles their synchronization. No whole-graph
barrier is introduced by this transformation.

Weight recipes produce BF16 W'=W*gamma, FP32 u=sum(stored W'), and
FP32 v=W*beta+bias. The consumer applies rstd*(acc-mean*u)+v before its
remaining finite chain. Residual normalization recomputes the raw row's
normalized BF16 value before adding it. Each operation carries its own width
and epsilon, since a residual width can differ from the enclosing GEMM K.
Embedding sum writes complete [row,2] statistics; GEMM producers write
[row,column_tile,2] statistics of their stored BF16 outputs. Statistics use
FP32 sum and square sum; variance is clamped at zero before adding epsilon.

verified: host descriptor round trips, exclusion/access contracts, unequal
residual K, exact statistics lifting and eight CPU recipe checks pass.
The generated three-GEMM numerical fixture is submitted in
`runs/dm1-deferred-ln-generated-native-v1`; GPU, sanitizer, 50-process and
five-architecture results remain pending. These host results do not establish
full-model numerical correctness or a synchronization claim.

Deviation/pending: `--deferred-ln auto|0` currently controls all proved edges
as one structural choice. Per-edge joint solver selection and extra norm-work
pricing remain to be integrated. No timing or speed claim is made.
