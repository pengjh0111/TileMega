# V-F — symbolic shapes in CuTe IR

Evidence labels: ✅ executed/observed; ⚠️ limited or version-specific;
❌ conjecture.

## Toolchain and tests

✅ The exact LLVM revision from `cutlass_compiler/LLVM_COMMIT`,
`23a60f15f2fcafcf67b95b0a035053579958b732`, was fetched and built with the
bundled MLIR configuration. `cute-opt` and `base-opt` were produced. All 236
CuTe dialect/transform LIT tests and all 106 C++ unit tests passed.

The cold fetch-to-`cute-opt` interval was 9 minutes 18 seconds with 32 build
jobs on this host. The source checkout occupied 3.1 GiB and the build tree
1.5 GiB. A warm configure/build/test rerun took 11 seconds. The first two
shallow fetches failed near completion through the local proxy; a direct exact
commit fetch succeeded (`raw/fetch_attempts.txt`). These are environment costs,
not compiler failures.

## Static and dynamic algebra

| Operation | Static case | Dynamic case after `cute-fold-static` |
|---|---|---|
| `Composition` | ✅ folded to `cute.static` | ✅ legal, retained as `cute.composition` |
| `LogicalDivide` | ✅ folded | ✅ legal, retained as `cute.logical_divide` |
| `ZippedDivide` | ✅ folded | ✅ legal, retained as `cute.zipped_divide` |
| `RightInverse` | ✅ `(4,3):(1,4) -> 12:1` | ❌ verifier rejects dynamic shape |
| `LeftInverse` | ✅ `(4,3):(3,1) -> (3,4):(4,1)` | ❌ rejects dynamic shape; also rejects dynamic stride |
| `Flatten` | ✅ nested modes become flat modes | ✅ expands while preserving dynamic leaves |
| `Coalesce` | ✅ contiguous modes combine | ✅ expands/preserves unresolved dynamic values |
| `CeilDiv` | ✅ folded | ✅ legal, retained as `cute.ceil_div` |
| `ShapeDiv` | ✅ folded | ✅ legal, retained as `cute.shape_div` |

The decisive diagnostic for a dynamic right inverse is:

```text
expects a static-shape input layout, but got
!cute.layout<"(?,3):(1,4)">
```

This is a legality rejection, not merely a missed optimization.

## Hierarchical flattening and affinity

✅ A nested plain layout such as `(3,(4,5)):(8,(1,4))` expands to
`(3,4,5):(8,1,4)`. Dynamic leaves likewise flatten, for example
`(3,(?,5)):(?,(1,?)) -> (3,?,5):(?,1,?)`.

Flattening preserves an affine address map when strides are constants and
dynamic values occur only in the domain extents. Two important exceptions are:

- A dynamic stride makes the address expression `parameter * coordinate`,
  which is not a Presburger affine expression in a single parametric ISL set.
- A composed layout containing a CuTe swizzle remains a
  `composed_layout`; bitwise swizzle semantics do not become a flat affine
  stride map.

Thus the CuTe-to-ISL bridge can accept flat constant-stride layouts and
parameterized domains directly. Dynamic-stride and swizzled forms need either
specialization, layout cancellation before conversion, or a non-affine Tier.

## `pycute` comparison

✅ `pycute` agreed with the dialect on the shared static cases:

- composition: `(2,4):(1,2)`;
- right inverse: `12:1`;
- left inverse: `(3,4):(4,1)`;
- logical divide: `((3,2),(4,2)):((8,24),(1,4))`;
- zipped divide: `((3,4),(2,2)):((8,1),(24,4))`.

It also produced the expected flatten/coalesce/shape-div values. However,
`pycute.typing.Integer` accepts integer values, not symbolic extents as a
public value domain. It is a useful static oracle but not a symbolic analysis
representation.

## Phase 3 choice

Use the **CuTe MLIR dialect as the imported layout representation**, with a
TileMega-owned CuTe-to-Presburger bridge as semantic authority. This choice is
supported by the passing test suite, existing operation vocabulary, direct
MLIR integration, dynamic divide/composition representation, and agreement
with `pycute` on static cases. Do not use this dialect for code generation.

For `C = W^-1 o R`, use the following rule:

1. Normally `g` fixes the intra-tile `W`; specialize it and use the dialect's
   static `RightInverse`. Keep symbolic outer extents in the ISL domain.
2. If `W` itself contains an unresolved dynamic extent, invert the relation in
   the Presburger bridge rather than emitting `cute.right_inverse`.
3. If the relation is not injective/affine after cancellation, raise its Tier
   instead of inventing an inverse.

This avoids a custom reimplementation of the full 70-op dialect. If a future
dependency decision forces a standalone implementation, the minimum required
surface is `composition`, left/right inverse, logical/zipped divide,
flatten/coalesce, and ceil/shape-div plus hierarchical shape/stride types and
legality checks—approximately 1–2 KLOC and two to three engineer-weeks with
differential tests. That estimate is ⚠️ planning guidance, not measured work.

## Reproduction

```bash
docs/experiments/V_F/run.sh
cat docs/experiments/V_F/raw/test_summary.txt
cat docs/experiments/V_F/raw/pycute.json
```
