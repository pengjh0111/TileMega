// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <cassert>
#include <iostream>

using namespace tilemega;
using namespace tilemega::solver;
using namespace tilemega::backend;

template<class F> void Reject(F&& action) {
  bool rejected=false;
  try {action();}catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
}

int main() {
  Sm90ForwardCoordinates c;c.abi=Sm90ForwardAbi::kF3;
  Sm90ForwardStage stage;stage.tile_m=128;stage.tile_n=64;stage.slots=6;
  c.stages={stage};
  auto old=c.ToJson();old.Set("schema",3);
  auto loaded=Sm90ForwardCoordinates::FromJson(old);
  assert(loaded.stages[0].tk==64 && loaded.stages[0].mc==0 &&
      loaded.stages[0].store==Sm90ForwardStore::kReg);
  c.stages[0].tk=128;c.stages[0].slots=4;
  auto encoded=c.ToJson();
  assert(encoded.At("schema").AsNumber("schema")==4);
  loaded=Sm90ForwardCoordinates::FromJson(json::Parse(encoded.Dump(0)));
  assert(loaded.stages[0].tk==128 && loaded.stages[0].slots==4);
  c.stages[0].tk=96;Reject([&]{Sm90ForwardCoordinates::FromJson(c.ToJson());});
  c.stages[0].tk=64;c.stages[0].mc=2;
  Reject([&]{Sm90ForwardCoordinates::FromJson(c.ToJson());});
  c.stages[0].mc=0;c.stages[0].tile_n=32;c.stages[0].store=Sm90ForwardStore::kTma;
  Reject([&]{Sm90ForwardCoordinates::FromJson(c.ToJson());});
  Sm90ForwardCoordinates legacy;legacy.stages={stage};
  auto v=legacy.ToJson();assert(v.At("schema").AsNumber("schema")==3);
  assert(Sm90ForwardCoordinates::FromJson(v).abi==Sm90ForwardAbi::kLegacy);
  constexpr auto arena=Sm90F3MakeGemmLayout(128,64,4,false,0,false,0,false,0,0,64);
  constexpr auto span=Sm90F3MakeGemmLayout(128,64,4,false,0,false,0,false,0,0,128);
  static_assert(!arena.Spanning() && span.Spanning());
  static_assert(span.slot_bytes==2*arena.slot_bytes && span.end==196608);
  static_assert(Sm90F3Budget::Fits(span.end));
  constexpr auto stored=Sm90F3MakeGemmLayout(128,64,4,false,0,false,0,false,0,0,64,1,true);
  static_assert(stored.end==arena.end+32768 && stored.store_bytes==32768);
  static_assert(stored.store_base>=stored.side_base+stored.side_bytes);
  constexpr auto residual=Sm90F3MakeGemmLayout(128,64,2,false,0,true,2,false,0,0);
  constexpr auto reused=Sm90F3MakeGemmLayout(128,64,2,false,0,true,2,false,0,0,64,1,true);
  static_assert(reused.end==residual.end && reused.store_bytes==0);
  std::cout<<"F3 schema compatibility, slot widths and arena boundaries PASS\n";
}
