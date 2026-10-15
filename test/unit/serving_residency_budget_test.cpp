// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PageLayout.h>
#include <cassert>
#include <cstdio>
int main() {
  using tilemega::solver::PageLayout;
  // Deliberately small synthetic devices prove the calculation is not tied to
  // H800's capacities. The real kernel occupancy is checked separately.
  for(int shared:{98304,167936,233472}) {
    tilemega::TargetSpec target;target.res.max_smem_per_sm=shared;
    target.res.max_dynamic_smem_per_cta=shared-1024;
    auto unchanged=PageLayout::ResidencyBudget(target,0);
    auto one=PageLayout::ResidencyBudget(target,1);
    auto two=PageLayout::ResidencyBudget(target,2);
    assert(unchanged.res.max_dynamic_smem_per_cta==target.res.max_dynamic_smem_per_cta);
    assert(one.res.max_dynamic_smem_per_cta==target.res.max_dynamic_smem_per_cta);
    assert(two.res.max_dynamic_smem_per_cta==shared/2);
    auto layout=PageLayout::Build(two,16384,8192,8192,0,0,1024);
    assert(layout.pages>=1 && layout.shared_bytes<=shared/2);
    assert(target.res.max_dynamic_smem_per_cta==shared-1024);
  }
  bool rejected=false;
  try { tilemega::TargetSpec target;PageLayout::ResidencyBudget(target,3); }
  catch(std::invalid_argument const&) { rejected=true; }
  assert(rejected);
  rejected=false;
  try { tilemega::TargetSpec target;PageLayout::ResidencyBudget(target,2); }
  catch(std::invalid_argument const&) { rejected=true; }
  assert(rejected);
  std::puts("PASS serving residency target budgets and invalid-request gates");
}
