#!/usr/bin/env bash
# R8 C-b: the dialect split makes "the solver decides, codegen reads" a
# property anyone can check. Codegen must not *construct* an execution op;
# the solver and its write-back passes must not construct a graph op.
#
# Construction in this codebase is either `builder.create<X>` / `OperationState`
# with the op's C++ class, or the op name as a string. Both are searched.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
status=0

echo "== lib/Codegen must not construct tmexec.* =="
# Op names only: `tmexec.solved_*` are module attributes, and reading a
# decision is exactly what codegen is for.
if grep -rnE 'create<(tilemega::)?dialect::(PlacementOp|ImplementationOp|PlanOp)>|"tmexec\.(placement|implementation|plan)"' lib/Codegen/ ; then
  echo "FAIL: lib/Codegen constructs an execution op"; status=1
else
  echo "PASS: no execution op constructed in lib/Codegen"
fi

echo "== lib/Solver and the write-back passes must not construct tmcg.* =="
if grep -rnE 'create<(tilemega::)?dialect::(TileSpaceOp|CouplingOp|EventTensorOp|FusedTaskSpaceOp|GraphOp)>|"tmcg\.[a-z_]+"' \
     lib/Solver/ include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h ; then
  echo "FAIL: the solver constructs a graph op"; status=1
else
  echo "PASS: no graph op constructed in lib/Solver or the write-back pass"
fi

echo "== the two dialects are disjoint in the op tables =="
grep -c 'def CG_' include/tilemega/Dialect/CouplingGraph/CGOps.td | sed 's/^/tmcg ops: /'
grep -c 'def Exec_' include/tilemega/Dialect/CouplingGraph/ExecOps.td | sed 's/^/tmexec ops: /'
exit $status
