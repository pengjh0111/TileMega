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
  Sync();
  Key key{std::min(cap,CurveAt(cta_bytes_,cta_gbps_,q)),q};
  int id=groups_.size();groups_.push_back({key,count});
  auto& state=classes_[key];
  state.completions.emplace(state.service+bytes,id);
  state.count+=count;active_count_+=count;rates_dirty_=true;return id;
}
double InflightDramServer::DeviceRate() const {
  double sum=0;for(auto const& [key,state]:classes_)sum+=key.second*state.count;
  return classes_.empty()?0:std::min(peak_,CurveAt(inflight_bytes_,inflight_gbps_,sum));
}
void InflightDramServer::Sync() {
  double dt=clock_-synced_at_;
  if(dt==0)return;
  for(auto& [key,state]:classes_)state.service+=state.rate*dt;
  synced_at_=clock_;
}
void InflightDramServer::Allocate() {
  if(!rates_dirty_)return;
  Sync();
  double available=DeviceRate(),weight=0;
  for(auto const& [key,state]:classes_)weight+=key.second*state.count;
  std::vector<decltype(classes_)::iterator> sorted;
  sorted.reserve(classes_.size());
  for(auto it=classes_.begin();it!=classes_.end();++it)sorted.push_back(it);
  std::sort(sorted.begin(),sorted.end(),[](auto a,auto b){
    double lhs=a->first.first/a->first.second;
    double rhs=b->first.first/b->first.second;
    return lhs==rhs?a->first<b->first:lhs<rhs;
  });
  for(auto it:sorted) {
    auto const& key=it->first;auto& state=it->second;
    double proportional=weight>0?available*key.second/weight:0;
    state.rate=std::min(key.first,std::max(0.,proportional));
    available-=state.rate*state.count;weight-=key.second*state.count;
  }
  total_rate_=0;
  next_due_=std::numeric_limits<double>::infinity();
  for(auto const& [key,state]:classes_)if(state.rate>0) {
    total_rate_+=state.rate*state.count;
    next_due_=std::min(next_due_,clock_+
        std::max(0.,state.completions.top().first-state.service)/state.rate);
  }
  rates_dirty_=false;
}
double InflightDramServer::Next() {
  Allocate();return std::max(0.,next_due_-clock_);
}
std::vector<int> InflightDramServer::Advance(double dt) {
  if(dt<0 || !std::isfinite(dt))throw std::invalid_argument("invalid in-flight time increment");
  Allocate();
  clock_+=dt;
  delivered_+=total_rate_*dt;
  // Most flow events are task readiness, not DRAM completion.  Keep one
  // absolute completion clock and settle class service only on a change in
  // membership or allocation; intermediate events are constant time.
  if(clock_+1e-6<next_due_)return {};
  Sync();std::vector<int> done;
  for(auto it=classes_.begin();it!=classes_.end();) {
    auto& state=it->second;
    // At a large absolute timestamp, a remaining interval shorter than one
    // clock ULP rounds the due time back to clock_. Complete that cohort
    // instead of returning zero-time events forever. The tolerance is at
    // most two representable time steps of service, not a fixed byte fudge.
    double const ulp=std::nextafter(clock_,
        std::numeric_limits<double>::infinity())-clock_;
    double const completion_tolerance=std::max(1e-6,2*ulp*state.rate);
    while(!state.completions.empty() &&
          state.completions.top().first-state.service<=completion_tolerance) {
      auto [threshold,id]=state.completions.top();state.completions.pop();
      delivered_+=(threshold-state.service)*groups_[id].count;
      state.count-=groups_[id].count;active_count_-=groups_[id].count;
      done.push_back(id);rates_dirty_=true;
    }
    if(state.count==0)it=classes_.erase(it);else ++it;
  }
  std::sort(done.begin(),done.end());
  return done;
}
} // namespace tilemega::solver
