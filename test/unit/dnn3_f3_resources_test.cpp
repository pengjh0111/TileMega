// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <cassert>
#include <iostream>
using namespace tilemega;
using namespace tilemega::solver;

int main() {
  for(auto chain:{Sm90F3ChainClass::kBiasResAct,Sm90F3ChainClass::kDeferredLn,
                  Sm90F3ChainClass::kDeferredLnGelu,Sm90F3ChainClass::kResidualLnRowStats})
    assert(!Sm90F3TileFeasible(256,128,chain));
  assert(Sm90F3TileFeasible(256,128,Sm90F3ChainClass::kPlain));
  assert(Sm90F3TileFeasible(256,128,Sm90F3ChainClass::kBiasAct));
  TargetSpec target;target.res.num_sms=132;
  target.res.max_dynamic_smem_per_cta=232448;
  target.res.max_smem_per_sm=233472;
  Sm90ForwardShape s;Sm90ForwardStage c;
  s.body=c.body=Sm90ForwardBody::kGemm;s.m=128;s.n=2304;s.k=768;
  c.tile_m=128;c.tile_n=64;c.slots=4;c.split_k=3;
  s.dnn1_geometry=json::Object{{"f3_vectors",2},{"f3_residual",false},
      {"needs_stats",true},{"has_channel_pool",false}};
  auto resource=QuerySm90ForwardF3Resources(s,c,target,1);
  assert(resource.feasible && !resource.exclusive);
  assert(resource.task_count==108 && resource.slot_bytes==24576);
  assert(resource.shared_bytes==102400 && resource.arena_bytes==114688);
  assert(!QuerySm90ForwardF3Resources(s,c,target,2).feasible);
  auto reduced=target;reduced.res.max_dynamic_smem_per_cta=227328;
  assert(!QuerySm90ForwardF3Resources(s,c,reduced,1).feasible);
  c.slots=8;
  resource=QuerySm90ForwardF3Resources(s,c,target,1);
  assert(resource.feasible && resource.exclusive);
  c.tile_m=256;c.tile_n=128;
  assert(!QuerySm90ForwardF3Resources(s,c,target,1).feasible);

  s.mode=c.mode=Sm90ForwardMode::kSwapAB;
  s.m=16;s.n=1000;s.k=512;c.tile_m=128;c.tile_n=128;c.split_k=8;c.slots=6;
  s.dnn1_geometry.Set("f3_vectors",1);s.dnn1_geometry.Set("needs_stats",false);
  resource=QuerySm90ForwardF3Resources(s,c,target,1);
  assert(resource.feasible && resource.task_count==64);
  assert(resource.slot_bytes==18432 && resource.shared_bytes==113664);
  s.m=1;c.slots=2;
  resource=QuerySm90ForwardF3Resources(s,c,target,1);
  assert(resource.slot_bytes==17408 && resource.shared_bytes==37888);
  s.m=17;assert(!QuerySm90ForwardF3Resources(s,c,target,1).feasible);

  s.mode=c.mode=Sm90ForwardMode::kNormal;
  s.m=784;s.n=2048;s.k=512;c.tile_m=128;c.tile_n=64;c.split_k=1;c.slots=2;
  s.dnn1_geometry.Set("f3_vectors",2);
  s.dnn1_geometry.Set("has_channel_pool",true);s.dnn1_geometry.Set("pool_rows",49);
  resource=QuerySm90ForwardF3Resources(s,c,target,1);
  assert(resource.feasible && resource.shared_bytes==58368);
  s.dnn1_geometry.Set("pool_rows",196);
  assert(QuerySm90ForwardF3Resources(s,c,target,1).shared_bytes==56320);

  c=Sm90ForwardStage{};s=Sm90ForwardShape{};
  s.body=c.body=Sm90ForwardBody::kLayout;
  s.dnn1_geometry=json::Object{{"batch",16},{"width",224},{"height",224}};
  assert(QuerySm90ForwardF3Resources(s,c,target,1).task_count==392);
  s.body=c.body=Sm90ForwardBody::kPool;
  s.dnn1_geometry=json::Object{{"batch",1},{"output_width",112},
      {"output_height",112},{"channels",64}};
  assert(QuerySm90ForwardF3Resources(s,c,target,1).task_count==392);
  s.body=c.body=Sm90ForwardBody::kGlobalPool;
  s.dnn1_geometry=json::Object{{"batch",16},{"channels",512}};
  assert(QuerySm90ForwardF3Resources(s,c,target,1).task_count==32);
  for(auto body:{Sm90ForwardBody::kEmbedding,Sm90ForwardBody::kLayerNorm}) {
    s.body=c.body=body;s.dnn1_geometry=json::Object{{"rows",2048}};
    assert(QuerySm90ForwardF3Resources(s,c,target,1).task_count==256);
  }
  s.body=c.body=Sm90ForwardBody::kEncoderAttention;
  s.dnn1_geometry=json::Object{{"batch",16},{"heads",12}};
  resource=QuerySm90ForwardF3Resources(s,c,target,1);
  assert(resource.task_count==192 && resource.shared_bytes==100352);
  Sm90ForwardSearchSpace space;space.target=target;space.abi=Sm90ForwardAbi::kF3;
  s=Sm90ForwardShape{};s.m=512;s.n=64;s.k=768;
  s.dnn1_geometry=json::Object{{"epilogue_steps",json::Array{}},{"has_row_stats",false}};
  space.shapes={s};auto candidates=space.Candidates(0,1);
  // Eleven tile shapes, six divisors of kIter=12, seven slot counts.
  assert(candidates.size()==462 && space.Candidates(0,2).empty());
  bool slots5=false,slots7=false,tm256=false,wide_tail=false;
  for(auto const& option:candidates) {
    slots5|=option.slots==5;slots7|=option.slots==7;
    tm256|=option.tile_m==256;wide_tail|=option.tile_n>s.n;
  }
  assert(slots5 && slots7 && tm256 && wide_tail);
  space.shapes[0].m=128;
  assert(space.Candidates(0,1).size()==336);
  space.shapes[0].mode=Sm90ForwardMode::kSwapAB;
  assert(space.Candidates(0,1).size()==42);
  space.shapes[0].body=Sm90ForwardBody::kEncoderAttention;
  space.shapes[0].mode=Sm90ForwardMode::kNormal;
  candidates=space.Candidates(0,1);
  assert(candidates.size()==1 && candidates[0].tile_m==128);
  std::cout<<"F3 resource geometry and production work-item counts PASS\n";
}
