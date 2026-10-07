# CI-2 descriptor contract

verified: `dm_descriptor`, `dm_descriptor_device`, `frontend_import` and
`cg_attr_roundtrip` pass (4/4); `check-policy` passes. Descriptor probes compile
for sm_80/89/90/100/120, and the sm_89 executable passes its 20 field checks.
All seven legacy runtime structure sizes equal the unchanged reference
(`results/CI2_descriptor_abi.json`). This is not complete G-REG evidence.

Implemented: shared host/device POD definitions, optional plan fields,
validated CG serialization, CUDA aggregate emission, finite epilogue kind
packs, convolution tables, and ten appended task kinds with tile ownership.
Runtime operand structures carry the descriptors only under
`TILEMEGA_DM_SUPPORT=1`, emitted exclusively for DM plans. Old plan attributes
and generated CUDA omit these fields and the macro.

Invocation metadata now carries geometry, binding/row/scale sources and a
device buffer view; runtime construction binds dense row counts and row
mapping, checks A storage bounds, and owns/releases the uploaded descriptor
tables. The solver CG reader retains the same descriptors. verified: updated
four-test suite passes, and a generated DM CUDA source compiles with the
complete runtime on sm_89. This is compile evidence, not execution of a model.

CI-2 remains open: finite epilogue specifications and tile execution are
implemented, but their generated dispatch into GEMM/combiner invocations is
pending. No complete DNN/MoE plan executes yet. Numerical subcomponent
evidence and exact rounding contracts are recorded in `numerics.md`.
Arithmetic declarations describe the semantic work, and their implementation
flags remain false until the corresponding numerical body gate passes.

The packed CG integer attributes follow declaration order in
`DmDescriptors.h`. The codec checks exact word counts, unsigned bounds, enum
ranges, convolution output geometry, layout containment/alignment, required
binding sources, chain capacities and model buffer references. Codegen
rechecks the contract when reading serialized CG.

- Dense row stride/offset count logical rows. A zero stride means one; the
  operand's separate element pitch locates the resulting physical row.
- Layout identifiers are indices into the buffer table. NHWC shapes use
  N,H,W,C; row-major shapes use their first `rank` axes. Strides count elements.
- RGB C'=4 can retain an eight-element physical pixel pitch for 16-byte
  alignment. Packed channels and pixel pitch are separate quantities.
- Epilogue parameter slots: bias/scale use 0; residual uses tensor 0 and
  optional channel scale 1; deferred RMSNorm uses statistics 0; deferred LN
  uses statistics/u/v in 0/1/2; residual LN uses tensor/statistics/gamma/beta
  in 0/1/2/3. Activation and gate pairing use their typed fields.
- Every operation declares input/output FP32 or BF16 rounding; the chain
  declares final store rounding. Actual model chains and their numerical
  formulas belong to DN-9/MO-6 and must be tested before performance runs.
- Side outputs are separate from the eight-operation main chain. Each of
  row statistics, channel sums, top-k, argmax and split-K partials may appear
  once. Top-k count must be positive.

inferred: preserving legacy structure sizes is necessary for SASS equality;
it is insufficient alone. Full generated CUDA/resource/SASS/token comparison
against the sealed LLM bank remains required at the Phase 1 gate.
