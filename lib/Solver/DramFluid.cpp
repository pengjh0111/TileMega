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
  int id=groups_.size();groups_.push_back({bytes,cap,0,count});active_.push_back(id);cap_counts_[cap]+=count;rates_dirty_=true;return id;
}
void DramFluidServer::Allocate() {
  if(!rates_dirty_)return;
  long consumers=0;for(auto const& [cap,count]:cap_counts_)consumers+=count;
  double remaining=bandwidth_;std::map<double,double> rates;
  for(auto const& [cap,count]:cap_counts_) {
    double rate=std::min(cap,std::max(0.,remaining)/consumers);
    rates.emplace(cap,rate);remaining-=rate*count;consumers-=count;
  }
  for(int i:active_)groups_[i].rate=rates.at(groups_[i].cap);
  rates_dirty_=false;
}
double DramFluidServer::Next() {
  Allocate();double next=std::numeric_limits<double>::infinity();for(int i:active_){auto const& g=groups_[i];if(g.rate>0)next=std::min(next,g.left/g.rate);}return next;
}
std::vector<int> DramFluidServer::Advance(double dt) {
  if(dt<0 || !std::isfinite(dt))throw std::invalid_argument("invalid fluid time increment");
  Allocate();std::vector<int> done;for(int i:active_){auto& g=groups_[i];
    double sent=std::min(g.left,g.rate*dt);g.left-=sent;delivered_+=sent*g.count;
    if(g.left<=1e-6){delivered_+=g.left*g.count;done.push_back(i);auto found=cap_counts_.find(g.cap);found->second-=g.count;if(!found->second)cap_counts_.erase(found);g.left=0;g.count=0;rates_dirty_=true;}}
  active_.erase(std::remove_if(active_.begin(),active_.end(),[&](int i){return groups_[i].count==0;}),active_.end());
  return done;
}
bool DramFluidServer::Empty() const {return active_.empty();}
} // namespace tilemega::solver
