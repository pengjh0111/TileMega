# DM-1 implementation constraints

These are **inferred** design constraints to check during CI/DN/MO
implementation. They are not verification results or relaxations of a gate.

## Preserving G-REG

Adding defaulted fields to a device descriptor can still change SASS: array
indexing uses its size, and adding switch cases can change a jump table.
Therefore default-zero values alone are insufficient for §0.5.

Keep the original device descriptor layouts and switch bodies when compiling
the legacy LLM specialization. DM plans need an explicit compile-time feature
specialization for their additional descriptor fields and task dispatch.
Append host planning fields normally, emit their CG attributes only when
present, and emit the DM specialization only for DNN/MoE plans. This must keep
all three supported executors; it is a model specialization, not a reduced
execution mode. Every new build macro belongs in the manifest and identity.

Check `.cu`, every ptxas resource context and SASS against the sealed bank;
neither matching tokens nor matching source is a substitute for the others.
Adding TK16/32 and TN16 candidates must not silently change the legacy LLM
candidate domain. DM model signatures provide the discriminator.

## Export and front-end details

The Phase 0 fixtures retain the inventories taken before serialization.
`export_roundtrip.json` records deserializer-inserted unused tuple getitems.
DNN import must handle tuple projections and ignore provably unused outputs;
it cannot assign them generic executable tasks. BERT's attention-output
linear also has a bmm-plus-bias spelling in this environment.

HF's original output pytree types must be registered before loading BERT's
archive. NAFNet's custom autograd LayerNorm2d remains unchanged upstream.
Only its exported ATen operations are lowered.

## Descriptor and numeric details to resolve explicitly

- Global pooling stores FP32 means, while the tensor-core GEMM A operand is
  BF16. Its load/conversion and BF16 rounding point need an explicit contract.
- The Cin<=4, C'=4 convolution option needs a clear distinction between
  logical channels, physical pixel pitch and copy alignment. The 16-byte
  alignment rule cannot be inferred from a four-BF16-element logical extent.
- SimpleGate pairs distant channels. Depthwise ownership and its access
  relation must represent both channel segments, rather than assuming that
  a contiguous Tc block already owns both halves.
- Side channel sums must split a linear M tile at image boundaries. Arrivals
  count those segments, not merely the number of CTAs.
- Expert row gather, deferred norm and counted arrivals all use the binding's
  original token/rank. Empty virtual tasks publish completion without reading
  a weight page and must agree with the loader's empty predicate.
- Queue completion, artifact sealing, individual numerical checks and a
  complete G-REG/G-DNN/G-MOE result are distinct evidence records.

Keep the stated numerical assertions and gate thresholds unchanged while
resolving these details. Record any necessary implementation deviation and
its reason in `summary.md`.
