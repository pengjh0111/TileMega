// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <vector>
#include <map>
#include <queue>
namespace tilemega::solver {
// One group contains identical consumers. Bytes and cap are per consumer;
// water filling accounts for multiplicity, never multiplies device bandwidth.
class DramFluidServer {
 public:
  explicit DramFluidServer(double bytes_per_ns);
  int Add(double bytes,double cap,int count=1);
  double Next();
  std::vector<int> Advance(double delta_ns);
  bool Empty() const;
  double Delivered() const { return delivered_; }
 private:
  struct Group {int count=0;};
  struct Cap {
    double service=0,rate=0;
    long count=0;
    std::priority_queue<std::pair<double,int>,std::vector<std::pair<double,int>>,
        std::greater<std::pair<double,int>>> completions;
  };
  double bandwidth_,delivered_=0;
  std::vector<Group> groups_;
  std::map<double,Cap> caps_;
  bool rates_dirty_=false;
  void Allocate();
};
} // namespace tilemega::solver
