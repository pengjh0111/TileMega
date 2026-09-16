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
  auto finite=p;tilemega::analysis::ParamBinding theta;theta.Bind("S",8);
  finite.tasks=finite.tasks.BindParams(theta);finite.dependencies=finite.dependencies.BindParams(theta);
  finite.pi_sigma=finite.pi_sigma.BindParams(theta);finite.rank=finite.rank.BindParams(theta);finite.grid_limit=finite.grid_limit.BindParams(theta);
  auto bound=tilemega::solver::ProveParametricPlacement(finite,nullptr,4);
  if(!bound.passed())return 3;
  auto holes=finite;holes.pi_sigma=holes.pi_sigma.ApplyRange(map("{ [w,q] -> [w,2*q] }"));
  auto rejected=tilemega::solver::ProveParametricPlacement(holes,nullptr,4);
  if(rejected.dense || rejected.passed())return 4;
  if(tilemega::solver::ProveParametricPlacement(finite,nullptr,5).passed())return 5;
  std::cout<<"FINITE_PROOF total_image_positive_holes_negative_grid_negative PASS\n";
  p.dependencies=map("[S] -> { [1,t] -> [0,t]; [0,t] -> [1,t] : 1<=S<=128 and 0<=t<S }");
  proof=tilemega::solver::ProveParametricPlacement(p);
  if (proof.acyclic) return 2;
  std::cout << "SYMBOLIC_PROOF positive_and_cycle_negative PASS\n";
}
