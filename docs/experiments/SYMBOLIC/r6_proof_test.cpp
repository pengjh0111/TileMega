// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ParametricPlacement.h>
#include <iostream>
int main() {
  tilemega::analysis::IslContext context;
  using tilemega::analysis::CouplingRelation;
  auto map=[](char const* s){return CouplingRelation::FromIslText(s);};
  tilemega::solver::ParametricPlacement p;
  p.tasks=map("[S] -> { [] -> [s,t] : 1<=S<=128 and 0<=s<2 and 0<=t<S }");
  p.dependencies=map("[S] -> { [1,t] -> [0,t] : 1<=S<=128 and 0<=t<S }");
  p.pi_sigma=map("[S] -> { [0,t] -> [w,q] : 1<=S<=128 and 0<=t<S and w=t%4 and q=floord(t,4); [1,t] -> [w,q] : 1<=S<=128 and 0<=t<S and w=(S+t)%4 and q=floord(S+t,4) }");
  p.rank=map("[S] -> { [s,t] -> [s,t] : 1<=S<=128 and 0<=s<2 and 0<=t<S }");
  p.grid_limit=map("[S] -> { [] -> [4,8] : 1<=S<=128 }");
  auto proof=tilemega::solver::ProveParametricPlacement(p);
  std::cout << proof.total << proof.bijective << proof.dense << proof.acyclic << proof.resident << ' ' << proof.error << '\n';
  if (!proof.passed()) return 1;
  p.dependencies=map("[S] -> { [1,t] -> [0,t]; [0,t] -> [1,t] : 1<=S<=128 and 0<=t<S }");
  proof=tilemega::solver::ProveParametricPlacement(p);
  if (proof.acyclic) return 2;
  std::cout << "SYMBOLIC_PROOF positive_and_cycle_negative PASS\n";
}
