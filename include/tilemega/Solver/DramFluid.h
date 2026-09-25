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

// Opt-in serving model.  A group carries identical tasks with q bytes in
// flight each.  The device rate is F(sum(count*q)); capped weighted water
// filling redistributes any rate a task cannot consume.
class InflightDramServer {
 public:
  InflightDramServer(double peak_gbps,std::vector<double> inflight_bytes,
      std::vector<double> inflight_gbps,std::vector<double> cta_bytes,
      std::vector<double> cta_gbps);
  int Add(double bytes,double cap,double q,int count=1);
  double Next();
  std::vector<int> Advance(double delta_ns);
  bool Empty() const { return active_.empty(); }
  double Delivered() const { return delivered_; }
  double DeviceRate() const;
 private:
  struct Group {double remaining=0,cap=0,q=0,rate=0;int count=0;};
  double peak_,delivered_=0;
  std::vector<double> inflight_bytes_,inflight_gbps_,cta_bytes_,cta_gbps_;
  std::vector<Group> groups_;
  std::vector<int> active_;
  bool rates_dirty_=false;
  void Allocate();
};
} // namespace tilemega::solver
