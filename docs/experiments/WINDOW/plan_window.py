#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Stamp W onto a generated model source.

W reaches the executor from the Plan and from nothing else (§8.11), but
`lib/Codegen/Codegen.cpp` refuses to emit a plan whose window is not
`kPlacementWindowImplemented` and that file is outside R3 H1's may-change list.
The reference sources emit no plan brace at all -- their `RuntimePlanDesc` is
default constructed, which is where today's W = 1 comes from -- so this appends
one carrying mode 0, the legacy grid-stride placement the source already uses.
W is therefore the only thing that differs between the arms.

The four fields between `ownership_flags` and the plan (`attention`,
`resident_only`, `balanced_placement`, `exact_dependencies`) have to be spelled
out to reach the plan positionally.  Their values are not guessed: they are the
ones `tilemega-place-chain` emits for these same two models, checked against
`CHAIN/raw/plan/{gqa2,mha4}_s4_chain.cu`.

    python3 plan_window.py <src.cu> <W> <out.cu> [--in-plan]

By default a variant line that already carries a plan is refused rather than
rewritten: WINDOW's own arms are built from the plan-less reference sources, and
a source that unexpectedly carries a plan there means the wrong file was passed.

`--in-plan` is the opposite case and has to be asked for.  PLACE_EFT2 crosses W
with the placement candidates, and `eft`, `wavefront` and `chain` all travel as
an emitted plan whose `window` is already spelled out; there the 4th positional
field is rewritten in place, so the candidate's (worker, slot) arrays survive and
W is again the only difference between that candidate's arms.
"""
import sys

TAIL = ", nullptr, true, false, nullptr, {0u, nullptr, 0u, %uu, 0u}},"
# `window` is the 4th member of RuntimePlanDesc: mode, params, param_count,
# window.  Positional, because that is how codegen emits the brace.
WINDOW_FIELD = 3


def rewrite_in_plan(line, window):
    """Replace `window` inside a plan brace the emitter already wrote."""
    start = line.rindex("{")
    end = line.index("}", start)
    fields = line[start + 1:end].split(",")
    if len(fields) <= WINDOW_FIELD:
        return None
    current = fields[WINDOW_FIELD].strip()
    # Refuse anything that is not the literal the emitter writes: a named
    # constant here would mean the brace is not the plan and the count above
    # matched something else.
    if not current.endswith("u") or not current[:-1].isdigit():
        return None
    fields[WINDOW_FIELD] = f" {window}u"
    return line[:start + 1] + ",".join(fields) + line[end:]


def main():
    argv = [a for a in sys.argv[1:] if a != "--in-plan"]
    in_plan = "--in-plan" in sys.argv[1:]
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, window, out = argv[0], int(argv[1]), argv[2]
    lines = open(src).read().split("\n")
    stamped = 0
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("{kRuntimeGemms"):
            continue
        if not line.rstrip().endswith("},"):
            continue
        # One brace means the line reaches only as far as `ownership_flags`: a
        # nested brace would be an attention record or a plan already emitted,
        # and appending past either would bind the fields to the wrong members.
        if line.count("{") != 1:
            if not in_plan:
                print(f"{src}:{i + 1}: variant already carries a nested initializer",
                      file=sys.stderr)
                return 1
            rewritten = rewrite_in_plan(line.rstrip(), window)
            if rewritten is None:
                print(f"{src}:{i + 1}: no window field in the trailing brace",
                      file=sys.stderr)
                return 1
            lines[i] = rewritten
            stamped += 1
            continue
        if in_plan:
            print(f"{src}:{i + 1}: variant carries no plan to rewrite",
                  file=sys.stderr)
            return 1
        lines[i] = line.rstrip()[:-2] + TAIL % window
        stamped += 1
    if stamped == 0:
        print(f"{src}: no runtime variant line found", file=sys.stderr)
        return 1
    open(out, "w").write("\n".join(lines))
    print(f"stamped window={window} on {stamped} variant(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
