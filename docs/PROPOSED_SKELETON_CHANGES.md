# Proposed changes to `TileMega_skeleton.md`

This file records proposals only. The design document itself remains unchanged.
Evidence labels: ✅ verified; ⚠️ documented or partially verified; ❌ conjecture.

## 1. Definition 4 uses symbolic closed forms

- Location: §2.1, Definition 4; §4.3 attributes.
- Current text: all derived quantities are closed forms in `(theta, g)`, but
  the initial C++ stub stored concrete, differently named scalar fields.
- Proposed text: require `wait`, `fanout`, `volume`, and `count` to have the
  `ClosedForm(theta,g)` type, and require `C` to have a structured,
  reparameterizable relation type. Printable strings are diagnostics only.
- Evidence: ✅ `coupling_types_test` evaluates §2.7 edges 1/7/11 and preserves
  the edge-7 relation structure across `Kc` partitioning (F-14).
- Confidence: high.

## 2. Add invariant I3 for streaming progress

- Location: §2.2 after I2; §8.7.
- Current text: §8.7 recommends launching exactly the occupancy capacity.
- Proposed text: define streaming legality over the realized wait-for graph and
  possible resident CTA subsets. A grid may exceed `resident_limit` when every
  possible resident subset exposes a bounded progress frontier; a bounded edge
  span smaller than capacity is a sufficient special case, not the definition.
- Evidence: ✅ V-A circular grid 1536 completed; V-J backward grid 769 completed
  while grid 1536 hung 20/20 (F-4, F-9).
- Confidence: high for the need; medium for a final sufficient/necessary test.

## 3. Extend TargetSpec

- Location: §3.4 and Phase 2 launch planning.
- Current text: lists CUDA capabilities but has no complete portable target
  schema.
- Proposed text: add SKU identity and provenance, maximum resident blocks/SM,
  cluster occupancy and GPC limits, opt-in shared-memory state, collective
  datatype families, and CUDA/CUTLASS version provenance.
- Evidence: ✅ V-B required a narrow-type SM120 path; ⚠️ V-C needs
  cluster-specific occupancy; H100 SM count varies by SKU (F-13).
- Confidence: high.

## 4. Permit one mainloop adapter per CUTLASS family

- Location: §5.3 TaskBody ABI.
- Current text: presents one SM90-style collective call shape.
- Proposed text: keep TaskContext, storage, scheduling, and result contracts
  architecture-independent, but allow family adapters for SM80 cp.async,
  SM90 TMA/GMMA, and SM120 TMA/MMA orchestration.
- Evidence: ✅ V-B directly called `MainloopSm80CpAsync`; SM80/89 were not
  covered by the tested CollectiveBuilder combination (F-6).
- Confidence: high.

## 5. Split traits pruning from survivor compilation

- Location: §5 candidate generation and §7 Phase 4.
- Current text: mentions traits and nvcc queries without a batching policy.
- Proposed text: instantiate constexpr traits in batches, reject illegal
  storage/tile candidates, then compile survivors in parallel with an explicit
  cache key containing source, target, CUDA, and CUTLASS revisions.
- Evidence: ✅ V-D: 100 traits candidates in 17.6s and 0-byte SHM error;
  V-E: 4.658s/variant, 13.2min projected serial, 3.61x four-process speedup
  (F-7, F-11).
- Confidence: high.

## 6. Correct the §8.1 rationale

- Location: §8.1.
- Current text: conflates all-thread polling with the barrier mismatch.
- Proposed text: all-thread polling is semantically valid but may add atomic
  traffic; the correctness prohibition is a collective barrier inside a
  thread-divergent spin loop. Keep the barrier after convergence.
- Evidence: ✅ all-thread polling passed; `barrier_in_spin` hung 150/150 (F-2).
- Confidence: high.

## 7. Classify backoff as a performance rule

- Location: §8.3.
- Current text: states that spin waits must back off.
- Proposed text: label backoff as a tunable performance rule rather than a
  release/acquire correctness requirement.
- Evidence: ✅ removing backoff alone produced 0/150 mismatches (F-3).
- Confidence: high for correctness on the tested protocol.

## 8. Expand release-side rules

- Location: §8.5.
- Current text: shows one thread fence followed by event publication.
- Proposed text: distinguish single-thread production from cooperative CTA
  production. For cooperative writes, every writer fences, the CTA converges,
  and one thread publishes. Add that large tiles can evict stale lines and are
  weak fence-negative controls.
- Evidence: ✅ no CTA release convergence mismatched 100/100; missing fence
  with reuse mismatched 150/150; tile mismatch fell from 100% at 4096 floats to
  2% at 8192 and 0% at 16384 (F-1, F-10).
- Confidence: high for the protocol and observed cliff; medium for the L1-cache
  mechanism.

## 9. Require an explicit shared-storage union

- Location: §8.6.
- Current text: says TaskBody storage takes the maximum.
- Proposed text: require one explicit union whose lifetime covers dispatch.
  Separate storage objects or escaping pointers remain simultaneously live and
  may sum. Include register allocation in the post-merge occupancy check.
- Evidence: ✅ V-G union emitted 8192B; escaping separate storage emitted
  36864B and reduced occupancy from 6 to 2 CTA/SM (F-8).
- Confidence: high.

## 10. Treat resident_limit as a resource capacity

- Location: §8.7.
- Current text: equates launch grid with `SM count * occupancy`.
- Proposed text: define
  `resident_limit = num_sms * active_blocks_per_sm(kernel, block, smem)` as a
  capacity term. Require `grid <= resident_limit` only when progress analysis
  proves full-grid co-residency necessary; cluster kernels use active-cluster
  occupancy and whole-cluster rounding.
- Evidence: ✅ V-A/V-J distinguish capacity from progress; ⚠️ V-C documents
  the cluster-specific query (F-4, F-9).
- Confidence: high for non-cluster capacity; cluster runtime validation pending.

## 11. Separate shape expressions from guard constraints

- Location: §1 frontend contract, §2 parameter domain, and Phase 1 importer.
- Current text: symbolic shapes are discussed, but the distinction between
  value expressions and executable input guards is not explicit.
- Proposed text: import `ExportedProgram` range constraints and guards into a
  canonical Presburger parameter domain. Use `ClosedForm` for values such as
  `seq + past`; use constraint objects for equality, bounds, and modulo. Do not
  infer semantic identity from generated symbol names.
- Evidence: ✅ V-H emitted four symbols for reused cache dimensions and four
  equality guards; unequal K/V cache lengths were rejected (F-15).
- Confidence: high for the distinction; torch's public guard API is
  version-dependent.

## 12. Define the dynamic layout-inverse fallback

- Location: §3.5 CuTe-to-ISL rules and §7 P3.1 `CuteLayoutBridge`.
- Current text: assumes `W^-1 = right_inverse(W)` without a dynamic-shape
  legality condition.
- Proposed text: use CuTe MLIR as imported layout syntax, but specialize
  intra-tile `W(g)` before dialect inversion. If `W` remains dynamic, invert
  the affine relation in the Presburger bridge; dynamic stride or swizzle that
  cannot be cancelled raises Tier. CuTe IR remains outside codegen.
- Evidence: ✅ 236/236 CuTe tests and 106/106 unit tests passed; dynamic
  `right_inverse` is explicitly rejected while dynamic composition/divides
  remain legal (F-12).
- Confidence: high for the pinned revision.

## 13. Make operand coordinates part of the mainloop-adapter contract

- Location: §5.3 TaskBody ABI and generated parameter tables.
- Current text: identifies operand layouts by C++ types but does not state the
  adapter's logical coordinate convention or parameter passing rule.
- Proposed text: each CUTLASS-family adapter declares its logical A/B
  coordinates, expected strides, residue convention, epilogue ownership, and
  shared-storage requirement. Large invocation tables are passed by device
  pointer, not by-value kernel parameters.
- Evidence: ✅ E2E's wrong B tag caused 6,143 mismatches; the correct logical
  `(N,K)` mapping passed. Pointer passing reduced the per-thread stack from
  2,592 to 32 bytes (F-17).
- Confidence: high.

## 14. Add a frontend-only operator class

- Location: §1.4/Phase 1 operator classification.
- Current text: the seven compute/access categories do not cover graph guards,
  symbolic-size extraction, dtype conversion, and metadata assertions.
- Proposed text: distinguish TaskBody operators from frontend-only
  `guard/meta/layout` nodes. Such nodes may be folded into domains/access maps,
  but must have explicit lowering rules and may not be silently discarded.
- Evidence: ✅ V-H's 30-target whitelist contained seven such targets, and the
  E2E adapter needed their shapes/guards while emitting no separate compute
  body (F-15, F-16).
- Confidence: high for the need; exact taxonomy can evolve in Phase 1.
