// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2.6 and §4.3.  This is the sole contract shared by L4-L1.
include "mlir/IR/OpBase.td"

def CG_Dialect : Dialect {
  let name = "tilemega";
  let cppNamespace = "::tilemega::dialect";
  let summary = "TileMega parameterized Coupling Graph";
  let description = [{
    Task spaces, structured coupling relations, symbolic derived metrics,
    event tensors, and placements used by every compiler layer above CUDA.
  }];
  let useDefaultAttributePrinterParser = 1;
}

class CG_Op<string mnemonic, list<Trait> traits = []>
    : Op<CG_Dialect, mnemonic, traits>;

class CG_Attr<string name, list<Trait> traits = []>
    : AttrDef<CG_Dialect, name, traits>;
