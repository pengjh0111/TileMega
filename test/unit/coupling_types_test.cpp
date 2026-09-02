#include <tilemega/Analysis/CouplingDerivation.h>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace tilemega::analysis;

namespace {

void Require(bool condition, char const* expression, int line) {
  if (condition) return;
  std::cerr << "requirement failed at line " << line << ": " << expression
            << '\n';
  std::exit(1);
}

#define REQUIRE(expression) Require((expression), #expression, __LINE__)

AffineExpr V(char const* coordinate) {
  return AffineExpr::Variable(coordinate);
}

CouplingEdge Edge1() {
  ProducerMap norm{{"rmsnorm1"}, {V("m")}, {}};
  return {{"rmsnorm1"}, {"wq_wk_wv"},
          AffineRelation({"m", "n"}, {norm}),
          {ClosedForm::Constant(1), ClosedForm::Constant(48),
           ClosedForm::Symbol("Tm") * ClosedForm::Symbol("H"),
           ClosedForm::Symbol("S").CeilDiv(ClosedForm::Symbol("Tm"))},
          Tier::kAffine, SyncKind::kGlobal};
}

CouplingEdge Edge7() {
  ProducerMap combine{
      {"attn_combine"}, {},
      {{"s", AffineExpr::Variable("m", ClosedForm::Symbol("Tm")),
        ClosedForm::Symbol("Tm")},
       {"hh", AffineExpr::Constant(ClosedForm::Constant(0)),
        ClosedForm::Symbol("n_h")}}};
  return {{"attn_combine"}, {"wo"},
          AffineRelation({"m", "n"}, {combine}, {"Tm", "n_h"}),
          {ClosedForm::Symbol("Tm") * ClosedForm::Symbol("n_h"),
           ClosedForm::Constant(32), ClosedForm::Symbol("Tm"),
           ClosedForm::Symbol("S").CeilDiv(ClosedForm::Symbol("Tm"))},
          Tier::kAffine, SyncKind::kGlobal};
}

CouplingEdge Edge11() {
  ProducerMap gate{{"gate"}, {V("m"), V("n")}, {}};
  ProducerMap up{{"up"}, {V("m"), V("n")}, {}};
  return {{"wgate_wup"}, {"silu_mul"},
          AffineRelation({"m", "n"}, {gate, up}),
          {ClosedForm::Constant(2), ClosedForm::Constant(1),
           ClosedForm::Symbol("Tm") * ClosedForm::Symbol("Tn"),
           ClosedForm::Symbol("S").CeilDiv(ClosedForm::Symbol("Tm"))},
          Tier::kAffine, SyncKind::kCluster};
}

}  // namespace

int main() {
  ParamBinding theta;
  theta.Bind("S", 512).Bind("H", 4096).Bind("n_h", 32);
  ParamBinding g;
  g.Bind("Tm", 128).Bind("Tn", 128);

  auto edge1 = Edge1();
  auto edge7 = Edge7();
  auto edge11 = Edge11();
  REQUIRE(edge1.metrics.wait.Eval(theta, g) == 1);
  REQUIRE(edge7.metrics.wait.Eval(theta, g) == 4096);
  REQUIRE(edge11.metrics.wait.Eval(theta, g) == 2);
  REQUIRE(edge1.C.ToString().find("rmsnorm1(m)") != std::string::npos);
  REQUIRE(edge11.C.Producers().size() == 2);

  // §2.5: split-K is a reparameterization of the already-derived relation.
  // StructureKey is preserved and only Kc is added to the parameter list.
  auto split = edge7;
  split.C = edge7.C.PartitionRange("hh", "kc", "Kc");
  split.metrics.wait = edge7.metrics.wait.CeilDiv(ClosedForm::Symbol("Kc"));
  g.Bind("Kc", 4);
  REQUIRE(edge7.C.SameStructure(split.C));
  REQUIRE(split.C.StructureKey() == edge7.C.StructureKey());
  REQUIRE(split.C.Parameters().size() == edge7.C.Parameters().size() + 1);
  REQUIRE(split.C.Parameters().back() == "Kc");
  REQUIRE(split.metrics.wait.Eval(theta, g) == 1024);
  REQUIRE(split.metrics.wait.ToString() == "ceildiv((Tm * n_h), Kc)");
  REQUIRE(split.C.ToString().find("Kc") != std::string::npos);
  return 0;
}
