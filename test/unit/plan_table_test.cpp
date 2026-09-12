// SPDX-License-Identifier: BSD-3-Clause
// EX-E1/EX-S2: the transport of a materialized Plan.  `eft` is the one mode
// that cannot be evaluated below L2, so (pi, sigma) has to travel from the
// solver through the CG and codegen to the host already solved.  What is pinned
// here is that the two halves -- the mode on every task space and the table on
// the module -- cannot appear apart in either direction, that a table which is
// not a dense (pi, sigma) is refused rather than repaired, and that the theta
// it is pinned to travels with it.
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tilemega;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

/// One task space and one placement op, with `module_attrs` on the module and
/// `plan_attrs` on the placement.  Parsing runs the verifier, so a null result
/// is a rejection.
std::string ModuleText(std::string const& module_attrs,
                       std::string const& plan_attrs) {
  std::string text = "module";
  if (!module_attrs.empty()) text += " attributes {" + module_attrs + "}";
  text +=
      " { tilemega.task_space @t {granularity = {}, "
      "kind = #tilemega.task_kind<\"elementwise\">, stage = 0 : i64, "
      "operator_name = \"test\", write_map = #tilemega.access_map<{}>, "
      "arithmetic = \"add\"}\n"
      "  tilemega.placement @t map = [0] cluster = 1";
  if (!plan_attrs.empty()) text += " {" + plan_attrs + "}";
  text += " }";
  return text;
}

std::string Table(char const* worker, char const* slot, char const* tail) {
  return std::string("tilemega.placement_table = {worker = array<i64: ") +
         worker + ">, slot = array<i64: " + slot + ">, " + tail + "}";
}

char const* kEftPlan =
    "mode = \"eft\", params = array<i64>, window = 1 : i64, policy = \"aot\", "
    "resident_only = true";
char const* kTheta = "seq = 4 : i64, past = 3 : i64, grid = 2 : i64";

}  // namespace

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  // Most parses here are meant to fail, so the diagnostic is captured instead
  // of printed; `rejected` asserts on its text, which is what keeps a test from
  // passing on the wrong refusal.
  std::string diagnostic;
  context.getDiagEngine().registerHandler([&](mlir::Diagnostic& message) {
    diagnostic += message.str();
    return mlir::success();
  });
  auto parse = [&](std::string const& text) {
    diagnostic.clear();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module) std::fprintf(stderr, "  refused: %s\n", diagnostic.c_str());
    return module;
  };
  auto rejected = [&](std::string const& text, char const* because) {
    auto module = parse(text);
    return !module && diagnostic.find(because) != std::string::npos;
  };

  // A well-formed pair verifies, and the table reads back exactly as written:
  // four nodes, two workers, sigma dense on each.
  {
    auto module = parse(ModuleText(Table("0, 1, 0, 1", "0, 0, 1, 1", kTheta),
                                   kEftPlan));
    REQUIRE(static_cast<bool>(module));
    dialect::PlacementTable table;
    std::string error;
    REQUIRE(dialect::ReadPlacementTable(*module, &table, &error));
    REQUIRE(table.worker == std::vector<int>({0, 1, 0, 1}));
    REQUIRE(table.slot == std::vector<int>({0, 0, 1, 1}));
    REQUIRE(table.seq == 4 && table.past == 3 && table.grid == 2);
  }

  // Neither half stands alone.  Without the mode the table would be emitted
  // and never read; without the table the mode names a schedule nothing below
  // the solver can produce.
  REQUIRE(rejected(ModuleText(Table("0, 1, 0, 1", "0, 0, 1, 1", kTheta), ""),
                   "needs placement mode eft"));
  REQUIRE(rejected(ModuleText("", kEftPlan), "needs its materialized table"));

  // A closed-form mode with a table is a contradiction, not a preference: the
  // host would materialize the closed form and silently drop the table.
  REQUIRE(rejected(ModuleText(
      Table("0, 1, 0, 1", "0, 0, 1, 1", kTheta),
      "mode = \"rotate\", params = array<i64>, resident_only = true"),
      "needs placement mode eft, not rotate"));

  // Malformed tables, one failure mode each.
  REQUIRE(rejected(ModuleText(Table("0, 1, 0", "0, 0, 1, 1", kTheta), kEftPlan),
                   "3 pi entries and 4 sigma entries"));
  REQUIRE(rejected(ModuleText(Table("0, 1, 0, 2", "0, 0, 1, 0", kTheta), kEftPlan),
                   "outside its grid"));
  REQUIRE(rejected(ModuleText(Table("0, 1, 0, 1", "0, 0, 0, 1", kTheta), kEftPlan),
                   "uses slot 0 twice"));
  REQUIRE(rejected(ModuleText(Table("0, 1, 0, 1", "0, 0, 1, 1",
                                    "seq = 4 : i64, past = 3 : i64, grid = 0 : i64"),
                              kEftPlan),
                   "names grid 0"));
  REQUIRE(rejected(ModuleText(std::string("tilemega.placement_table = {worker = "
                                          "array<i64>, slot = array<i64>, ") +
                                  kTheta + "}",
                              kEftPlan),
                   "is empty; absent and empty are different"));
  REQUIRE(rejected(ModuleText("tilemega.placement_table = {worker = array<i64: 0>}",
                              kEftPlan),
                   "needs worker and slot as array<i64>"));

  // sigma may interleave stages, which is the whole point of carrying it: node
  // 3 (stage 1 here) takes slot 0 on worker 1 ahead of node 1.
  {
    auto module = parse(ModuleText(Table("0, 1, 0, 1", "0, 1, 1, 0", kTheta),
                                   kEftPlan));
    REQUIRE(static_cast<bool>(module));
    dialect::PlacementTable table;
    std::string error;
    REQUIRE(dialect::ReadPlacementTable(*module, &table, &error));
    REQUIRE(table.slot == std::vector<int>({0, 1, 1, 0}));
  }

  std::printf("ok\n");
  return 0;
}
