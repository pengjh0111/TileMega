// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Backend/Sm90DenseAttentionPartition.h>
#include <cassert>
#include <vector>
template<int S> void check() {
 for(int past=64;past<=1086;++past)for(int penalty:{0,8,16}) {
  std::vector<int> seen(past+1);int maximum=0,owner=0;
  for(int s=0;s<S;++s) {
   auto r=tilemega::backend::Sm90DenseAttentionPartition<S>::Range(past,s,penalty);
   assert(r.begin>=0 && r.end>=r.begin && r.end<=past+1);
   maximum=std::max(maximum,r.end-r.begin);owner+=r.Contains(past);
   for(int p=r.begin;p<r.end;++p)++seen[p];
  }
  assert(owner==1);
  for(int count:seen)assert(count==1);
  assert(maximum<=tilemega::backend::Sm90DenseAttentionPartition<S>::MaxKeys(1088,penalty));
 }
}
void CheckPartitions() {check<1>();check<2>();check<4>();check<8>();check<16>();check<32>();check<64>();}

int main() {
 CheckPartitions();
 using namespace tilemega::solver;
 Sm90DenseShapeTable nk={{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{151936,1024}}};
 Sm90DenseCoordinates c;
 for(int i=0;i<5;++i) {auto& g=c.gemms[i];g.tile_n=i==4?128:nk[i][0]/128;g.slot_bytes=2*g.tile_n*64;}
 c.swiglu_u=8;c.attention.chunks=16;c.weight_stream=true;c.layer_loop=true;c.embed_la=true;
 c.Validate(nk,1);auto json=c.ToJson();auto d=Sm90DenseCoordinates::FromJson(json);d.Validate(nk,1);
 assert(d.gemms[2].tile_n==48 && d.gemms[1].tile_n==8 && d.weight_stream && d.layer_loop && d.embed_la);
 auto independent=c;independent.argmax_stage=true;
 auto restored=Sm90DenseCoordinates::FromJson(independent.ToJson());
 restored.Validate(nk,1);assert(restored.argmax_stage && restored.tp==1);
 assert(c.ToJson().At("schema").AsNumber("schema")==6);
 auto merge=c;merge.attention.merge_stage=true;
 auto merge_restored=Sm90DenseCoordinates::FromJson(merge.ToJson());
 merge_restored.Validate(nk,1);
 assert(merge_restored.attention.merge_stage && !merge_restored.argmax_stage);
 assert(merge_restored.ToJson().Dump()==merge.ToJson().Dump());
 // D5/D6 rules: full residency, the first-task stream and full-width attention.
 assert(d.gemms[0].FullyResident(1024)==(d.gemms[0].slots==16) && d.gemms[0].StreamBytes(1024)==3*2*32*64);
 assert(d.attention.FullyResident(575) && d.attention.StreamBytes(128)==3*4*64*128);
 assert(!d.attention.FullWidth(128,2,Sm90DenseArena(227*1024)) && d.attention.FullWidth(128,2,Sm90DenseArena(113*1024-1024)));
 for(auto key:{"weight_stream","layer_loop","emb","residency"}) {
  bool refused=false;
  tilemega::json::Object pruned;for(auto const& item:json.AsObject("coordinates"))if(item.first!=key)pruned.push_back(item);
  try {Sm90DenseCoordinates::FromJson(tilemega::json::Value(pruned));}catch(std::exception const&) {refused=true;}
  assert(refused);
 }
 json.Set("split_k",1);bool rejected=false;
 try { Sm90DenseCoordinates::FromJson(json); }catch(std::invalid_argument const&) {rejected=true;}
 assert(rejected);
 for(int batch:{1,2,4,8,16}) {
  auto exact=Sm90DenseCoordinates::Exact(nk,batch,200*1024);
  assert(exact.attention.chunks==16/batch && exact.swiglu_u==8 && !exact.weight_stream);
  for(unsigned i=0;i<4;++i)assert(nk[i][0]/exact.gemms[i].tile_n==128);
 }
 bool too_small=false;
 try {Sm90DenseCoordinates::Exact(nk,1,1024);}
 catch(std::invalid_argument const&) {too_small=true;}
 assert(too_small);
 // A1: an unfused MoE coordinate used to build and then read a schedule slot
 // the router stopped writing at E2. Validate() must refuse it, including the
 // schema-5 form REF-R3's cells are written in, which loads expert_fuse false.
 Sm90DenseShapeTable moe_nk={{{6144,2048},{2048,2048},{0,0},{0,0},
     {50304,2048},{64,2048},{2048,2048},{2048,1024}}};
 Sm90DenseCoordinates m;m.moe=true;
 for(unsigned i:m.Active()) {auto& g=m.gemms[i];g.tile_n=32;g.slot_bytes=2*g.tile_n*64;}
 m.gemms[4].tile_n=128;m.gemms[4].slot_bytes=2*128*64;
 m.swiglu_u=8;m.attention.chunks=16;m.weight_stream=true;m.layer_loop=true;m.embed_la=true;
 for(auto route:{Sm90DenseCoordinates::Route::kLastArriver,Sm90DenseCoordinates::Route::kStage}) {
  auto global=m;global.expert_fuse=true;global.route=route;
  global.Validate(moe_nk,1);
  assert(global.ToJson().At("schema").AsNumber("schema")==9);
  auto loaded=Sm90DenseCoordinates::FromJson(global.ToJson());
  assert(loaded.route==route && loaded.ToJson().Dump()==global.ToJson().Dump());
  auto separate=global;separate.expert_fuse=false;separate.ValidateSchedule(moe_nk,1,128,1,1088,232448);
  global.moe=false;bool invalid=false;
  try {global.Validate(nk,1);}catch(std::invalid_argument const&) {invalid=true;}
  assert(invalid);
 }

 m.expert_fuse=true;m.gemms[7].slots=m.gemms[6].slots;
 bool unfused_refused=false;
 auto unfused=m;unfused.expert_fuse=false;
 try {unfused.Validate(moe_nk,1);}
 catch(std::invalid_argument const&) {unfused_refused=true;}
 assert(unfused_refused);
}
