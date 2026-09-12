# E1-e / H4: no scheduling decision is left in Codegen

Gate E1-e asks that `lib/Codegen/Codegen.cpp` no longer decide a schedule
(§8.11). Round 1 had `BuildVariantSchedule` there, calling `ListScheduler`
directly; commit `157b5960` moved it to `lib/Solver/VariantSchedule.cpp` and
left only the consumption call.

## Evidence (verified, this session)

```
$ grep -n "ListScheduler\|BuildVariantSchedule\|BalanceTaskPlacement\|Schedule(" \
      lib/Codegen/Codegen.cpp
384:        solver::BuildVariantStageSchedule(variant.dependencies, stages.size());
```

The single remaining hit is a call into `lib/Solver`, not a decision. Lines 382-386 read, in full:

```cpp
    // §8.11: the stage order is a solver decision; codegen only copies it in.
    auto const staged =
        solver::BuildVariantStageSchedule(variant.dependencies, stages.size());
    variant.max_dependency_span = staged.max_dependency_span;
    variant.schedule.clear();
```

Sweeping the rest of the directory finds one further reference, and it is not a
decision either:

```
$ grep -rn "ListScheduler\|BalanceTaskPlacement" lib/Codegen/
lib/Codegen/ExactRuntimeTaskGraph.cpp:5:#include <tilemega/Solver/ListScheduler.h>
lib/Codegen/ExactRuntimeTaskGraph.cpp:45:  solver::ListScheduler{}.Validate(graph.successors,stage_order);
```

`Validate` checks an order it is given — here the identity order, built two lines
above by `std::iota` — and returns safety facts; it chooses nothing. That call is
byte-identical to `6c359e2b`, so it is not something this round introduced.

## The move changed the interface, not the algorithm

The old function mutated a `RuntimeVariantRecord`; the new one takes the
dependency list and returns a `VariantStageSchedule`. Normalizing that rename
(`variant.dependencies` → `dependencies`, `variant.` → `result.`, and the
`solver::`/`codegen::` qualifications, which swap sides with the namespace)
leaves exactly three differences — the signature, the result's declaration and
its return:

```
$ diff <(normalize 6c359e2b:BuildVariantSchedule) lib/Solver/VariantSchedule.cpp:10-43
1,2c1,3
< void BuildVariantSchedule(RuntimeVariantRecord& variant,
<                           std::size_t stage_count) {
---
> VariantStageSchedule BuildVariantStageSchedule(
>     std::vector<codegen::DependencyRecord> const& dependencies,
>     std::size_t stage_count) {
11a13
>   VariantStageSchedule result;
30a33
>   return result;
```

So the stage order itself is unchanged, which is what lets E1-a report an empty
`.cu` diff: had the move altered the order, `kSchedule0` would differ.

The host side of H4 is separate: `ModelHarness.cuh` no longer computes a
placement at all. It resolves a `PlacementMode` and calls
`solver::MaterializePlanPlacement`, so mode 4's balanced heuristic — which round
1 ran in the harness — now runs in `lib/Solver/PlanMaterialize.cpp` and the
harness keeps no branch that could decide differently.
