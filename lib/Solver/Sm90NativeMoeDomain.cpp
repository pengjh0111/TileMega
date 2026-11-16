// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tilemega::solver {
double Sm90NativeExpectedActiveExperts(int experts,int top_k,int batch) {
  if(experts<1 || top_k<1 || top_k>experts || batch<1)
    throw std::invalid_argument("invalid expected-active expert domain");
  return experts*(1-std::pow(1-double(top_k)/experts,batch));
}
int Sm90NativeExpectedExpertTasks(int experts,int top_k,int batch,int tiles) {
  if(tiles<1)throw std::invalid_argument("expert task domain needs positive tile count");
  double expected=Sm90NativeExpectedActiveExperts(experts,top_k,batch)*tiles;
  if(!std::isfinite(expected) || expected>std::numeric_limits<int>::max())
    throw std::invalid_argument("expected expert task count exceeds integer domain");
  return int(std::llround(expected));
}
Sm90NativeMoeTaskDomain Sm90NativeMoeDomain(int experts,int top_k,int batch,int tiles) {
  Sm90NativeMoeTaskDomain result;
  result.expected_active=Sm90NativeExpectedActiveExperts(experts,top_k,batch);
  result.tasks=Sm90NativeExpectedExpertTasks(experts,top_k,batch,tiles);result.tiles=tiles;
  result.ownership=analysis::CouplingRelation::FromIslText(
      "{ [t] -> [e,j] : t=e*"+std::to_string(tiles)+"+j and 0<=t<"+
      std::to_string(result.tasks)+" and e>=0 and 0<=j<"+std::to_string(tiles)+" }");
  return result;
}
json::Value Sm90NativeMoeTaskDomain::ToJson() const {
  return json::Object{{"approximation",kSm90NativeMoeDomainApproximation},
      {"expected_active_experts",expected_active},{"tiles_per_expert",tiles},
      {"expected_tasks",expected_active*tiles},{"rounded_tasks",tasks},
      {"ownership",ownership.ToString()}};
}
analysis::CouplingRelation Sm90NativeMoeExpertCoupling(
    Sm90NativeMoeTaskDomain const& producer,Sm90NativeMoeTaskDomain const& consumer) {
  auto slot=analysis::CouplingRelation::FromIslText("{ [e,j] -> [e] }");
  // Down reads the complete hidden row of its own active expert slot.
  return producer.ownership.ApplyRange(slot).ApplyRange(
      consumer.ownership.ApplyRange(slot).Reverse());
}
analysis::CouplingRelation Sm90NativeMoeCombineCoupling(Sm90NativeMoeTaskDomain const& down) {
  // Combine j reads every expert partial for column tile j.
  return down.ownership.ApplyRange(analysis::CouplingRelation::FromIslText("{ [e,j] -> [j] }"));
}
} // namespace tilemega::solver
