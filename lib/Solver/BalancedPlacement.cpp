// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/BalancedPlacement.h>
#include <tilemega/Solver/ListScheduler.h>
#include <algorithm>
#include <stdexcept>

namespace tilemega::solver {
TaskPlacement BalanceTaskPlacement(std::vector<std::vector<int>> const& successors,
    std::vector<int> const& order, std::vector<int> const& preferred_worker,
    int workers, int max_queue) {
  ListScheduler validator;
  validator.Validate(successors,order);
  if (workers<=0 || max_queue<=0 || preferred_worker.size()!=successors.size() ||
      static_cast<long long>(workers)*max_queue<static_cast<long long>(successors.size()))
    throw std::invalid_argument("balanced placement has insufficient queue capacity");
  auto unique=successors;
  std::vector<std::vector<int>> incoming(successors.size());
  for (std::size_t p=0;p<unique.size();++p) {
    auto& edges=unique[p];
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    for (int c:edges) incoming[c].push_back(p);
  }
  TaskPlacement out;
  out.worker.assign(successors.size(),-1);
  std::vector<int> lengths(workers),last(workers,-1);
  auto queue_dag=unique;
  for (int task:order) {
    int preferred=preferred_worker[task];
    if (preferred<0 || preferred>=workers)
      throw std::invalid_argument("preferred placement worker outside grid");
    std::vector<int> affinity(workers);
    for (int p:incoming[task]) ++affinity[out.worker[p]];
    int chosen=-1;
    for (int w=0;w<workers;++w) {
      if (lengths[w]>=max_queue) continue;
      if (chosen<0 || affinity[w]>affinity[chosen] ||
          (affinity[w]==affinity[chosen] && (lengths[w]<lengths[chosen] ||
              (lengths[w]==lengths[chosen] && w==preferred)))) chosen=w;
    }
    if (chosen<0) throw std::runtime_error("balanced placement exhausted queue capacity");
    out.worker[task]=chosen;
    ++lengths[chosen];
    if (last[chosen]>=0) queue_dag[last[chosen]].push_back(task);
    last[chosen]=task;
  }
  validator.Validate(queue_dag,order);
  out.max_queue=*std::max_element(lengths.begin(),lengths.end());
  for (std::size_t p=0;p<unique.size();++p) {
    bool local=!unique[p].empty();
    for (int c:unique[p]) {
      bool same=out.worker[p]==out.worker[c];
      out.same_worker_edges+=same;
      local=local && same;
      out.max_worker_span=std::max(out.max_worker_span,out.worker[p]-out.worker[c]);
    }
    out.fence_free_producers+=local;
  }
  return out;
}
}  // namespace tilemega::solver
