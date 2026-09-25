// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <chrono>
#include <iostream>

using namespace tilemega::analysis;
int main() {
  IslContext context;
  auto start=std::chrono::steady_clock::now();
  auto graph=LlamaDecoderLayer(DecoderShape{});
  int cases=0;
  for(int seq:{1,4,16,64})for(int tm:{32,64,128}) {
    auto known=DecoderShape::Table27Theta();
    for(auto const& [name,value]:DecoderShape::Table27G().values)known.Bind(name,value);
    known.Bind("Tm",tm).Bind("Tn",16).Bind("S",seq).Bind("past",3).Bind("L_s",seq+3);
    for(auto const& edge:CouplingDerivation{}.Derive(graph,known)) {
      auto const* producer=graph.Find(edge.src.name);
      auto const* consumer=graph.Find(edge.dst.name);
      auto window=FitWaitWindow(edge,*producer,*consumer,known);
      std::cout<<seq<<'\t'<<tm<<'\t'<<edge.src.name<<'\t'<<edge.dst.name<<'\t'<<window.ToString()<<'\n';
      ++cases;
    }
  }
  std::cerr<<"WAIT_WINDOWS cases="<<cases<<" wall_ms="
      <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
}
