// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/GraphPattern.h>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace tilemega::frontend;
  std::vector<FxNodeRecord> graph(80);
  for (int i=0;i<80;++i) {
    auto& node=graph[i];node.name="n"+std::to_string(i);
    node.op="call_function";node.target="aten.add.Tensor";
    if (i>0) node.inputs.push_back(graph[i-1].name);
    if (i>4 && i%3==0) node.inputs.push_back(graph[i-4].name);
    if (i%7==0) node.inputs.push_back("external");
  }
  PatternMatcher matcher(graph,{});
  int checks=0;
  for (int repeat=0;repeat<3;++repeat) for (int v=0;v<80;++v) {
    std::unordered_set<std::string> reachable;
    std::vector<std::string> pending{graph[v].name};
    while (!pending.empty()) {
      auto name=pending.back();pending.pop_back();
      if (!reachable.insert(name).second) continue;
      for (auto const& node:graph) if (node.name==name)
        pending.insert(pending.end(),node.inputs.begin(),node.inputs.end());
    }
    for (int a=-2;a<80;++a) {
      auto name=a==-2 ? "absent" : a==-1 ? "external" : graph[a].name;
      if (matcher.DependsOn(graph[v].name,name)!=(reachable.count(name)!=0))
        throw std::runtime_error("cached reachability differs from traversal");
      ++checks;
    }
  }
  if (!matcher.DependsOn("unknown","unknown") || matcher.DependsOn("unknown","n0"))
    throw std::runtime_error("unknown-node identity changed");
  std::cout << "PATTERN_REACHABILITY checks=" << checks << " cached_equals_traversal=PASS\n";
}
