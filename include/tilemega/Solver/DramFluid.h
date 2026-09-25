// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <vector>
namespace tilemega::solver {
// One group contains identical consumers. Bytes and cap are per consumer;
// water filling accounts for multiplicity, never multiplies device bandwidth.
class DramFluidServer {
 public:
  explicit DramFluidServer(double bytes_per_ns);
  int Add(double bytes,double cap,int count=1);
  double Next() const;
  std::vector<int> Advance(double delta_ns);
  bool Empty() const;
  double Delivered() const { return delivered_; }
 private:
  struct Group {double left=0,cap=0,rate=0;int count=0;};
  double bandwidth_,delivered_=0;
  std::vector<Group> groups_;
  std::vector<int> active_;
  void Allocate();
};
} // namespace tilemega::solver
