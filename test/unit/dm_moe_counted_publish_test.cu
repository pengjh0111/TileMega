// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MODEL_BF16 1
#include <tilemega/Codegen/tasks/MoeCountedPublication.cuh>
#include <tilemega/Codegen/executor/CountedDependency.cuh>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <vector>

using namespace tilemega::codegen;
constexpr unsigned Tokens=17,TopK=8,Experts=9,Width=70,ChannelTile=32;
struct Invocation {
  unsigned tile_m=0,tile_n=ChannelTile,tiles_n=3;
  DmGemmAccess access{};
  void const* binding=nullptr;
  void const* rows=nullptr;
};
__host__ __device__ int Value(unsigned token,unsigned rank,unsigned channel,unsigned epoch) {
  return int(1000*token+37*rank+channel+epoch*17);
}
__global__ void Pipeline(Params p,Invocation inv,int* partials,int* output,unsigned epoch) {
  if(!executor::IsCompute())return;
  unsigned subtile=(inv.access.block_rows+inv.tile_m-1)/inv.tile_m;
  unsigned tasks=inv.access.binding_blocks*subtile*inv.tiles_n;
  auto const* blocks=static_cast<MoeBindingRecord const*>(inv.binding);
  auto const* rows=static_cast<MoeBindingRow const*>(inv.rows);
  // Two synthetic workers finish their own producer work before claiming
  // consumer tasks; neither waits while it still owes producer contributions.
  for(unsigned task=blockIdx.x;task<tasks;task+=gridDim.x) {
    unsigned tile_m=task/inv.tiles_n,column=(task%inv.tiles_n)*ChannelTile;
    auto block=blocks[tile_m/subtile];unsigned begin=(tile_m%subtile)*inv.tile_m;
    if(block.valid && begin<block.row_count) {
      unsigned count=min(inv.tile_m,block.row_count-begin);
      for(unsigned item=executor::ComputeThread();item<count*ChannelTile;item+=128) {
        auto row=rows[block.row_begin+begin+item/ChannelTile];unsigned c=column+item%ChannelTile;
        if(c<Width)partials[(row.token*TopK+row.rank)*Width+c]=Value(row.token,row.rank,c,epoch);
      }
    }
    PublishMoeCountedRows(p,0,task,inv);
  }
  executor::ComputeSync();
  auto const& combine=p.stages[1];auto const& dep=p.dependencies[0];
  for(unsigned task=blockIdx.x;task<dep.table_rows;task+=gridDim.x) {
    auto expected=p.counted_thresholds.values[task];
    executor::CountedDependency::Wait(p.counted_dependencies+dep.counted_offset+task,
        std::uint64_t(expected)*(epoch+1));
    unsigned token=(task/3)*combine.group,column=(task%3)*ChannelTile;
    for(unsigned item=executor::ComputeThread();item<combine.group*ChannelTile;item+=128) {
      unsigned t=token+item/ChannelTile,c=column+item%ChannelTile;
      if(t<Tokens && c<Width) {
        int sum=0;for(unsigned rank=0;rank<TopK;++rank)sum+=partials[(t*TopK+rank)*Width+c];
        output[t*Width+c]=sum;
      }
    }
    executor::ComputeSync();
  }
}
template<class T>T* Managed(unsigned count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,sizeof(T)*count)==cudaSuccess);return p;
}
void Check(unsigned bm,unsigned tm,unsigned token_tile,unsigned threads) {
  unsigned capacity=0;assert(MoeVirtualCapacity(Tokens,TopK,Experts,bm,bm!=1,&capacity));
  auto* blocks=Managed<MoeBindingRecord>(capacity);auto* rows=Managed<MoeBindingRow>(Tokens*TopK);
  auto* stages=Managed<StageDesc>(2);stages[0]={};stages[1]={};
  stages[0].kind=TaskKind::kGemm;stages[1].kind=TaskKind::kMoECombine;
  stages[1].group=token_tile;stages[1].width=ChannelTile;stages[1].extent=Width;
  stages[1].moe.top_k=TopK;
  unsigned targets=((Tokens+token_tile-1)/token_tile)*3;
  auto* dep=Managed<StageDependency>(1);*dep={0,1,StageDependency::Map::kCounted,1,0,0,0};
  dep->table_rows=targets;dep->counted_offset=3;dep->counted_threshold_offset=0;
  auto* expected=Managed<unsigned>(targets);
  for(unsigned id=0;id<targets;++id)expected[id]=std::min(token_tile,Tokens-(id/3)*token_tile)*TopK;
  auto* counters=Managed<unsigned long long>(2*(targets+6));
  for(unsigned id=0;id<2*(targets+6);++id)counters[id]=0;
  auto* partials=Managed<int>(Tokens*TopK*Width);auto* output=Managed<int>(Tokens*Width);
  Invocation inv;inv.tile_m=tm;inv.binding=blocks;inv.rows=rows;
  inv.access.b=DmBAccess::kExpertIndirect;inv.access.write.kind=DmWriteKind::kRowScatter;
  inv.access.block_rows=bm;inv.access.binding_blocks=capacity;inv.access.binding_rows=Tokens*TopK;
  inv.access.experts=Experts;inv.access.routing_topk=TopK;
  Params p{};p.dims={Tokens,0,Tokens,1,0};p.stages=stages;p.stage_count=2;
  p.dependencies=dep;p.dependency_count=1;p.counted_dependency_count=targets+6;
  p.counted_thresholds={expected,targets};
  for(unsigned epoch=0;epoch<4;++epoch) {
    for(unsigned v=0;v<capacity;++v)blocks[v]={0xdeadbeef,0xdeadbeef,0xdeadbeef,0};
    std::vector<std::tuple<unsigned,unsigned,unsigned>> sorted;
    for(unsigned t=0;t<Tokens;++t)for(unsigned k=0;k<TopK;++k)
      sorted.emplace_back(bm==1?t*TopK+k:(3*t+k+epoch)%Experts,t,k);
    std::sort(sorted.begin(),sorted.end());
    unsigned used=0;
    for(unsigned begin=0;begin<sorted.size();) {
      unsigned end=begin+1;while(end<sorted.size() && std::get<0>(sorted[end])==std::get<0>(sorted[begin]))++end;
      for(unsigned at=begin;at<end;at+=bm)
        blocks[used++]={bm==1?unsigned((std::get<1>(sorted[at])*3+std::get<2>(sorted[at])+epoch)%Experts):
            std::get<0>(sorted[at]),at,std::min(bm,end-at),1};
      begin=end;
    }
    for(unsigned at=0;at<sorted.size();++at)rows[at]={std::get<1>(sorted[at]),std::get<2>(sorted[at]),0,0,0};
    for(unsigned bank=0;bank<2;++bank) {
      p.counted_dependencies=counters+bank*(targets+6);
      assert(cudaMemset(partials,0xa5,Tokens*TopK*Width*sizeof(int))==cudaSuccess);
      assert(cudaMemset(output,0xa5,Tokens*Width*sizeof(int))==cudaSuccess);
      Pipeline<<<2,threads>>>(p,inv,partials,output,epoch);
      assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
      for(unsigned t=0;t<Tokens;++t)for(unsigned c=0;c<Width;++c) {
        int sum=0;for(unsigned k=0;k<TopK;++k)sum+=Value(t,k,c,epoch);
        assert(output[t*Width+c]==sum);
      }
      for(unsigned id=0;id<targets+6;++id)
        assert(p.counted_dependencies[id]==(id>=3 && id<targets+3?
            std::uint64_t(expected[id-3])*(epoch+1):0));
    }
  }
  for(void* pointer:{static_cast<void*>(blocks),static_cast<void*>(rows),static_cast<void*>(stages),
      static_cast<void*>(dep),static_cast<void*>(expected),static_cast<void*>(counters),
      static_cast<void*>(partials),static_cast<void*>(output)})assert(cudaFree(pointer)==cudaSuccess);
}
int main() {
  unsigned cases=0;
  for(unsigned bm:{1u,16u,32u,64u,128u})for(unsigned tm:{16u,32u,64u,128u})
    for(unsigned token_tile:{1u,16u,32u})for(unsigned threads:{128u,160u}) {
      Check(bm,tm,token_tile,threads);++cases;
    }
  assert(cases==120);
  std::puts("MoE counted publication: 120 slot/group/tail geometries, empty records, 128/160 threads, four epochs and independent banks PASS");
}
