// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseWorkGraph.h>
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <cassert>
#include <set>

using namespace tilemega;
using namespace tilemega::solver;
namespace {
template<class F> void Reject(F f) {
  bool rejected=false;try {f();}catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
}
void Conservation(Sm90DenseWorkGraph const& graph,Sm90DenseWorkload const& w) {
  assert(graph.stages.size()==unsigned(5*w.layers+3));
  assert(graph.stages.front().kind=="embedding" && graph.stages.back().kind=="argmax");
  for(auto const& stage:graph.stages) {
    std::set<int> tasks;
    std::int64_t weights=0,activation=0,kv=0,current=0;
    for(auto const& cta:stage.ctas) {
      int sequence=0;
      for(auto const& task:cta.tasks) {
        assert(task.task%graph.grid==cta.cta && tasks.insert(task.task).second);
        for(auto const& b:task.blocks) {
          assert(b.sequence==sequence++ && b.slot==b.sequence%stage.ring_slots);
          assert(b.previous_reader==(b.sequence<stage.ring_slots?-1:b.sequence-stage.ring_slots));
          assert(b.prefix_bytes>=0 && b.prefix_bytes<=b.weight_bytes+b.kv_bytes);
          assert(b.activation_arrivals==32 && b.empty_arrivals==128);
          weights+=b.weight_bytes;activation+=b.activation_bytes;
          kv+=b.kv_bytes;current+=b.current_rows;
          if(&task!=&cta.tasks.front())assert(!b.prefix_bytes);
        }
      }
    }
    assert(tasks.size()==unsigned(stage.tasks));
    auto family=std::find(kSm90DenseFamilies.begin(),kSm90DenseFamilies.end(),stage.kind);
    if(family!=kSm90DenseFamilies.end()) {
      unsigned i=family-kSm90DenseFamilies.begin();
      assert(weights==2ll*w.shapes[i][0]*w.shapes[i][1]);
      assert(activation==2ll*w.batch*w.shapes[i][1]*stage.tasks);
      assert(!kv && !current);
    }else if(stage.kind=="attention") {
      assert(kv==4ll*w.batch*8*graph.past*w.head_dim);
      assert(current==w.batch*8 && !weights && !activation);
    }
  }
}
}

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  auto target=TargetSpec::FromJson(argv[1]);
  std::vector<Sm90DenseWorkload> models={
      {{{{4096,1024},{1024,2048},{6144,1024},{1024,3072},{151936,1024}}},1,2,128,2,true},
      {{{{4096,2048},{2048,2048},{12288,2048},{2048,6144},{151936,2048}}},1,2,128,2,true},
      {{{{3072,2048},{2048,2048},{16384,2048},{2048,8192},{128256,2048}}},1,2,64,4,false},
      {{{{5120,3072},{3072,3072},{16384,3072},{3072,8192},{128256,3072}}},1,2,128,3,false},
      {{{{6144,4096},{4096,4096},{28672,4096},{4096,14336},{128256,4096}}},1,2,128,4,false}};
  for(auto work:models)for(int batch:{1,2,4,8,16}) {
    work.batch=batch;
    auto coordinates=Sm90DenseCoordinates::Exact(work.shapes,batch,target.res.max_dynamic_smem_per_cta);
    for(int past:{64,575,1086}) {
      auto graph=BuildSm90DenseWorkGraph(work,coordinates,target,past);
      Conservation(graph,work);
      assert(graph.stages[1].LocalTaskHistogram().at(0)==4);
      // The lm_head's local ring does not restart at task boundaries.
      auto const& head=graph.stages[graph.stages.size()-2].ctas.front();
      int iterations=work.shapes[4][1]/coordinates.gemms[4].TileK();
      assert(head.tasks.size()>1 && head.tasks[1].blocks.front().sequence==iterations);
      for(auto const& stage:graph.stages)if(stage.kind=="attention") {
        for(auto const& cta:stage.ctas)for(auto const& task:cta.tasks)
          for(auto const& b:task.blocks) {
            assert(b.expected_transactions==(b.history_rows?1:0));
            assert(b.compute_rows==(batch<=4?b.valid_rows:coordinates.attention.block_keys));
          }
      }
    }
  }
  auto work=models.front();
  auto c=Sm90DenseCoordinates::Exact(work.shapes,1,target.res.max_dynamic_smem_per_cta);
  auto base=BuildSm90DenseWorkGraph(work,c,target);
  Reject([&]{base.RequireResident({});});
  Sm90DenseResidentEvidence evidence{base.grid,base.shared_bytes,1,std::string(64,'1'),std::string(64,'2')};
  base.RequireResident(evidence);
  auto bad=evidence;--bad.dynamic_shared_bytes;Reject([&]{base.RequireResident(bad);});
  auto resident2=c;resident2.residency=2;
  for(auto& g:resident2.gemms) {g.slot_bytes=2*g.tile_n*64;g.slots=2;}
  auto graph2=BuildSm90DenseWorkGraph(work,resident2,target);
  auto weak=evidence;weak.grid=graph2.grid;weak.dynamic_shared_bytes=graph2.shared_bytes;
  Reject([&]{graph2.RequireResident(weak);});

  // NewRowWork can create empty chunks. They still publish and, at low B,
  // consume one barrier generation with no fabricated KV transaction.
  c.attention.chunks=64;c.attention.new_row_work=1024;
  auto empty=BuildSm90DenseWorkGraph(work,c,target,64);Conservation(empty,work);
  int empty_tasks=0;
  for(auto const& cta:empty.stages[2].ctas)for(auto const& task:cta.tasks)
    if(task.blocks.front().compute_rows==0) {
      ++empty_tasks;assert(task.blocks.size()==1);
      assert(task.blocks.front().explicit_bulk_arrivals==1 && !task.blocks.front().kv_bytes);
    }
  assert(empty_tasks>0);
  auto high=models.front();high.batch=16;
  auto high_c=Sm90DenseCoordinates::Exact(high.shapes,16,target.res.max_dynamic_smem_per_cta);
  auto padded=BuildSm90DenseWorkGraph(high,high_c,target,64);
  auto const& tail=padded.stages[2].ctas.front().tasks.front().blocks.back();
  assert(tail.valid_rows==1 && tail.compute_rows==64 && !tail.history_rows && tail.current_rows==1);
  auto invalid=high_c;invalid.attention.block_keys=16;
  Reject([&]{BuildSm90DenseWorkGraph(high,invalid,target);});
  invalid=high_c;invalid.gemms[4].mma=Sm90DenseMma::kWgmma;
  Reject([&]{BuildSm90DenseWorkGraph(high,invalid,target);});
  auto wrong_arch=target;wrong_arch.nvcc_arch="sm_90";
  Reject([&]{BuildSm90DenseWorkGraph(high,high_c,wrong_arch);});

  Sm90DenseSearchSpace space{work.shapes,1,128,2,target};
  auto seeds=space.Seeds();auto prefix=BuildSm90DenseWorkGraph(work,seeds[2],target);
  Conservation(prefix,work);
  assert(prefix.stages[1].prefix_predecessor==0);
  bool traffic=false;
  for(auto const& stage:prefix.stages)for(auto const& cta:stage.ctas)
    for(auto const& task:cta.tasks)for(auto const& b:task.blocks)if(b.prefix_bytes)traffic=true;
  assert(traffic);
  assert(!prefix.activation_reuse && !prefix.launch_inactive_zero_bytes_per_cta);
  assert(prefix.stages[1].inactive_zero_bytes_per_cta>0);
  auto reusable=space.ActivationReuseSeeds();assert(!reusable.empty());
  auto reused=BuildSm90DenseWorkGraph(work,reusable.front(),target);
  assert(reused.activation_reuse && reused.launch_inactive_zero_bytes_per_cta>0);
  for(auto const& stage:reused.stages)assert(!stage.inactive_zero_bytes_per_cta);
  auto json=prefix.ToJson();assert(!json.At("prediction_available").AsBool("prediction"));
  assert(!json.At("l2_hints_guarantee_readiness").AsBool("L2 readiness"));
  // Alias retirement comes from actual payload overlap, not equal offsets.
  auto l2=c;l2.attention.new_row_work=0;l2.attention.chunks=16;
  l2.gemms[0].prefetch_l2_bytes=16384;
  auto hint=BuildSm90DenseWorkGraph(work,l2,target);
  assert(hint.stages[1].incoming_order==Sm90DenseTailOrder::kBeforeProduce);
  l2.gemms[0].preissue_late=true;
  assert(BuildSm90DenseWorkGraph(work,l2,target).stages[1].incoming_order==
      Sm90DenseTailOrder::kAfterProduce);
}
