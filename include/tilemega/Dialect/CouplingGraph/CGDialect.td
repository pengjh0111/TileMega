// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2.6 and §4.3.  This is the sole contract shared by L4-L1.
//
// R8 BE-7 splits what was one `tilemega` dialect into two, so that "the
// solver writes decisions, codegen reads them" stops being a convention and
// becomes a property anyone can check by grepping for a dialect prefix:
//
//   tmcg    the structure of a parameterized Coupling Graph
//   tmexec  the execution decisions solved on top of that structure
include "mlir/IR/OpBase.td"

def CG_Dialect : Dialect {
  let name = "tmcg";
  let cppNamespace = "::tilemega::dialect";
  let summary = "TileMega Coupling Graph structure";
  let description = [{
    Tile spaces, structured coupling relations, symbolic derived metrics and
    event tensors: everything that describes the graph itself, with no
    statement about how it is executed.
  }];
  let useDefaultAttributePrinterParser = 1;
}

def Exec_Dialect : Dialect {
  let name = "tmexec";
  let cppNamespace = "::tilemega::dialect";
  let summary = "TileMega execution decisions over a Coupling Graph";
  let description = [{
    Placements, implementations and the plan that holds them -- the output of
    the solver, and the only thing codegen is allowed to read as a decision.
  }];
}

class CG_Op<string mnemonic, list<Trait> traits = []>
    : Op<CG_Dialect, mnemonic, traits>;

class Exec_Op<string mnemonic, list<Trait> traits = []>
    : Op<Exec_Dialect, mnemonic, traits>;

class CG_Attr<string name, list<Trait> traits = []>
    : AttrDef<CG_Dialect, name, traits>;
