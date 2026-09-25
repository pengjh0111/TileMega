// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DramFluid.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tilemega::solver {
DramFluidServer::DramFluidServer(double rate):bandwidth_(rate){if(!(rate>0))throw std::invalid_argument("invalid DRAM bandwidth");}
int DramFluidServer::Add(double bytes,double cap,int count) {
  if(!(bytes>0 && cap>0 && count>0) || !std::isfinite(bytes) || !std::isfinite(cap))throw std::invalid_argument("invalid fluid group");
  int id=groups_.size();groups_.push_back({count});
  auto& state=caps_[cap];state.completions.emplace(state.service+bytes,id);state.count+=count;
  rates_dirty_=true;return id;
}
void DramFluidServer::Allocate() {
  if(!rates_dirty_)return;
  long consumers=0;for(auto const& [cap,state]:caps_)consumers+=state.count;
  double remaining=bandwidth_;
  for(auto& [cap,state]:caps_) {
    state.rate=std::min(cap,std::max(0.,remaining)/consumers);
    remaining-=state.rate*state.count;consumers-=state.count;
  }
  rates_dirty_=false;
}
double DramFluidServer::Next() {
  Allocate();double next=std::numeric_limits<double>::infinity();
  for(auto const& [cap,state]:caps_)if(state.rate>0)
    next=std::min(next,std::max(0.,state.completions.top().first-state.service)/state.rate);
  return next;
}
std::vector<int> DramFluidServer::Advance(double dt) {
  if(dt<0 || !std::isfinite(dt))throw std::invalid_argument("invalid fluid time increment");
  Allocate();std::vector<int> done;
  // Equal caps share a service clock. Cohorts enter with a finish threshold;
  // advancing time touches cap classes, not every active task/cohort.
  for(auto it=caps_.begin();it!=caps_.end();) {
    auto& state=it->second;double service=state.rate*dt;
    state.service+=service;delivered_+=service*state.count;
    while(!state.completions.empty() && state.completions.top().first-state.service<=1e-6) {
      auto [threshold,id]=state.completions.top();state.completions.pop();
      delivered_+=(threshold-state.service)*groups_[id].count;
      state.count-=groups_[id].count;done.push_back(id);rates_dirty_=true;
    }
    if(state.count==0)it=caps_.erase(it);else ++it;
  }
  // Preserve deterministic arrival order across cap classes.
  std::sort(done.begin(),done.end());return done;
}
bool DramFluidServer::Empty() const {return caps_.empty();}
} // namespace tilemega::solver
