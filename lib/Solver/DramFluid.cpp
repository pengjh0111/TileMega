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

namespace {
double CurveAt(std::vector<double> const& x,std::vector<double> const& y,double value) {
  if(x.size()!=y.size() || x.empty() || !(value>0))throw std::invalid_argument("invalid in-flight curve");
  if(value<=x.front())return y.front();
  if(value>=x.back())return y.back();
  auto upper=std::upper_bound(x.begin(),x.end(),value);
  auto i=std::size_t(upper-x.begin());
  double lo=std::log(x[i-1]),hi=std::log(x[i]);
  return y[i-1]+(y[i]-y[i-1])*(std::log(value)-lo)/(hi-lo);
}
void CheckCurve(std::vector<double> const& x,std::vector<double> const& y) {
  if(x.empty() || x.size()!=y.size())throw std::invalid_argument("missing in-flight calibration");
  for(std::size_t i=0;i<x.size();++i) {
    if(!(x[i]>0 && y[i]>0) || !std::isfinite(x[i]) || !std::isfinite(y[i]) ||
       (i && (x[i]<=x[i-1] || y[i]<y[i-1])))
      throw std::invalid_argument("in-flight curve must be positive and monotone");
  }
}
}
InflightDramServer::InflightDramServer(double peak_gbps,
    std::vector<double> inflight_bytes,std::vector<double> inflight_gbps,
    std::vector<double> cta_bytes,std::vector<double> cta_gbps)
    :peak_(peak_gbps),inflight_bytes_(std::move(inflight_bytes)),
     inflight_gbps_(std::move(inflight_gbps)),cta_bytes_(std::move(cta_bytes)),
     cta_gbps_(std::move(cta_gbps)) {
  if(!(peak_>0))throw std::invalid_argument("invalid DRAM peak");
  CheckCurve(inflight_bytes_,inflight_gbps_);CheckCurve(cta_bytes_,cta_gbps_);
}
int InflightDramServer::Add(double bytes,double cap,double q,int count) {
  if(!(bytes>0 && cap>0 && q>0 && count>0) || !std::isfinite(bytes) ||
     !std::isfinite(cap) || !std::isfinite(q))throw std::invalid_argument("invalid in-flight task");
  int id=groups_.size();
  groups_.push_back({bytes,std::min(cap,CurveAt(cta_bytes_,cta_gbps_,q)),q,0,count});
  active_.push_back(id);rates_dirty_=true;return id;
}
double InflightDramServer::DeviceRate() const {
  double sum=0;for(int id:active_)sum+=groups_[id].q*groups_[id].count;
  return active_.empty()?0:std::min(peak_,CurveAt(inflight_bytes_,inflight_gbps_,sum));
}
void InflightDramServer::Allocate() {
  if(!rates_dirty_)return;
  double available=DeviceRate(),weight=0;
  for(int id:active_)weight+=groups_[id].q*groups_[id].count;
  auto sorted=active_;
  std::sort(sorted.begin(),sorted.end(),[&](int a,int b){
    auto const& x=groups_[a];auto const& y=groups_[b];
    double lhs=x.cap/x.q,rhs=y.cap/y.q;
    return lhs==rhs?a<b:lhs<rhs;
  });
  for(int id:sorted) {
    auto& g=groups_[id];
    double proportional=weight>0?available*g.q/weight:0;
    g.rate=std::min(g.cap,std::max(0.,proportional));
    available-=g.rate*g.count;weight-=g.q*g.count;
  }
  rates_dirty_=false;
}
double InflightDramServer::Next() {
  Allocate();double next=std::numeric_limits<double>::infinity();
  for(int id:active_)if(groups_[id].rate>0)
    next=std::min(next,groups_[id].remaining/groups_[id].rate);
  return next;
}
std::vector<int> InflightDramServer::Advance(double dt) {
  if(dt<0 || !std::isfinite(dt))throw std::invalid_argument("invalid in-flight time increment");
  Allocate();std::vector<int> done;
  for(int id:active_) {
    auto& g=groups_[id];double sent=std::min(g.remaining,g.rate*dt);
    g.remaining-=sent;delivered_+=sent*g.count;
    if(g.remaining<=1e-6) {delivered_+=g.remaining*g.count;g.remaining=0;done.push_back(id);}
  }
  if(!done.empty()) {
    active_.erase(std::remove_if(active_.begin(),active_.end(),[&](int id){return groups_[id].remaining==0;}),active_.end());
    rates_dirty_=true;
  }
  return done;
}
} // namespace tilemega::solver
