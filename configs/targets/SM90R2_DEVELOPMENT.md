# Frozen SM90R2 development target

`sm_90_h800_sm90r2_development.json` is an explicit development input. It is
not selected by any default, is not a final target, and does not certify E-3 or
E-4. Its companion `.provenance.json` records the source binding and checks.

**Verified:** the target is a byte-for-byte copy of the frozen F114 merged
target: 6,138,726 bytes, SHA256
`f10a3bb2abca5dcb4265d05e0a2b4bf9fd728314cebb392ff0c730bc65281138`.
No observations, coefficients, lookup dimensions, thresholds, header hashes,
or metadata were removed or changed. All 14 top-level sections outside
`sm90_dense_decode` equal `sm_90_h800.json`, preserving the existing H800
MoE, DNN, and prefill inputs. Only the native decode calibration differs.

All calibration values needed by the price reader are embedded in this file.
The preserved historical paths identify measurement evidence; the price reader
does not import their contents. The ignored `runs/` tree is not the sole copy
of the calibration input. Raw measurements and binaries remain there, as
required by the repository evidence convention.

## Frozen source binding

The target retains its original 15 calibrated device-header hashes. They all
match commit `1283c2e6e50cc00e2aa7d0934d74ad76cff3699f`. The frozen F113
source snapshot can be reconstructed from repository commits, without the
original absolute snapshot path or its patch:

1. Create a separate checkout at `1283c2e6e50cc00e2aa7d0934d74ad76cff3699f`.
2. Restore these files from `cae8f39aef3bd16872a73fbbc5c962ec5549a1a3`:
   `include/tilemega/Solver/Sm90DenseCostModel.h`,
   `include/tilemega/Solver/Sm90DenseSearch.h`,
   `lib/Solver/Sm90DenseCostModel.cpp`,
   `lib/Solver/Sm90DenseSearch.cpp`, and `tools/commands/native_dense.cpp`.
3. Initialize dependencies at their gitlinks; CUTLASS is
   `dc45f979ae336a235da1676b311f35efeb30149a`. Use the repository build
   procedure and this target explicitly. This recipe reconstructs source;
   it does not assert a byte-identical compiler binary or final acceptance.

**Verified:** all 1,541 tracked regular files in the original frozen source
snapshot equal that recipe. The original snapshot's absolute CUTLASS symlink
is a local layout detail; the dependency is identified by its commit instead.

This is deliberately a mixed-commit development binding. Six calibrated
device headers differ from preparation HEAD `584b6393e`. Before a final
single-HEAD target can replace it, verify device equivalence for the default
build with the optional boundary diagnostics disabled, then bind the target
to that committed source. Do not substitute current header hashes merely to
pass the compiler's input check. The calibration remains
`initial_overlap_fitted`; independent validation, production-cache
equivalence, and final measured selection remain separate requirements.

## CPU preservation checks

Python checks compare the complete JSON tree, all non-native sections, and
the ordered observation tables after the same low-batch/chunk normalization
used by the C++ reader. The C++ price probe compares every retained ranked
coordinate from all seven F113 development cells against both targets and
their original request predictions. Raw commands and results are retained in
`runs/sm90r2/E4-canonical-target-preparation-v1/`.

No GPU measurement, fit, default change, or final selection is performed by
preparing this target.
