// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <iostream>

int main(int argc,char** argv) {
  if(argc!=3)return 2;
  auto target=tilemega::json::ParseFile(argv[1]);
  auto stale=tilemega::solver::Sm90NativeConstants::UnreachableAmpereProfiles(target,argv[2]);
  for(auto const& profile:stale)std::cerr<<profile<<'\n';
  std::cout<<stale.size()<<" unreachable profiles\n";
  return stale.empty()?0:1;
}
