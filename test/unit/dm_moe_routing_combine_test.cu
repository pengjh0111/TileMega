// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/MoETopKTaskBody.h>
#include <tilemega/Codegen/tasks/MoEDispatchTaskBody.h>
#include <tilemega/Codegen/tasks/MoECombineTaskBody.h>
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
template<class T>T* Managed(std::size_t count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(T))==cudaSuccess);return p;
}
void Sync() {assert(cudaGetLastError()==cudaSuccess);assert(cudaDeviceSynchronize()==cudaSuccess);}
using TopK=MoETopKTaskBody<Arch,8,32>;
__global__ void Top(MoeTopKOperands p) {TopK::Run(p,blockIdx.x);}
template<int BM,bool Group>using Dispatch=MoEDispatchTaskBody<Arch,8,BM,Group>;
template<int BM,bool Group>
__global__ void Hist(MoeDispatchOperands p,unsigned* ticket) {
  __shared__ typename Dispatch<BM,Group>::SharedStorage s;
  __shared__ unsigned last;
  Dispatch<BM,Group>::Histogram(p,blockIdx.x,s);
  executor::LastArriver::Run(ticket,Dispatch<BM,Group>::Count(p),&last,[&] {
    Dispatch<BM,Group>::Prefix(p,s);
  });
}
template<int BM,bool Group>
__global__ void Scatter(MoeDispatchOperands p) {
  __shared__ typename Dispatch<BM,Group>::SharedStorage s;
  Dispatch<BM,Group>::Scatter(p,blockIdx.x,s);
}
template<int BM,bool Group>
__global__ void Small(MoeDispatchOperands p) {
  __shared__ typename Dispatch<BM,Group>::SharedStorage s;
  Dispatch<BM,Group>::RunSmall(p,s);
}

template<int BM,bool Group>
unsigned Routing(unsigned tokens,unsigned experts,unsigned tile,unsigned mode) {
  unsigned parts=(experts+tile-1)/tile,part_stride=11,row_stride=parts*part_stride+5;
  auto* logits=Managed<float>(tokens*row_stride);
  auto* candidates=Managed<std::int32_t>(tokens*row_stride);
  auto* indices=Managed<std::int32_t>(tokens*8);
  auto* weights=Managed<E>(tokens*8);
  std::vector<float> full(tokens*experts);
  std::vector<unsigned> expected(tokens*8);
  for(unsigned token=0;token<tokens;++token) {
    for(unsigned e=0;e<experts;++e) {
      float value=mode==0?0:mode==1?float(int((token*31+e*13)%97)-48)*.0625f:
          (e<8?float(8-e):float(-1000-int(e)));
      full[token*experts+e]=float(E(value));
    }
    std::vector<unsigned> sorted(experts);std::iota(sorted.begin(),sorted.end(),0);
    auto less=[&](unsigned a,unsigned b) {
      float x=full[token*experts+a],y=full[token*experts+b];return x>y || (x==y && a<b);
    };
    std::sort(sorted.begin(),sorted.end(),less);
    for(unsigned rank=0;rank<8;++rank)expected[token*8+rank]=sorted[rank];
    for(unsigned part=0;part<parts;++part) {
      std::vector<unsigned> local;
      for(unsigned e=part*tile;e<std::min(experts,(part+1)*tile);++e)local.push_back(e);
      std::sort(local.begin(),local.end(),less);
      for(unsigned rank=0;rank<8;++rank) {
        auto offset=token*row_stride+part*part_stride+rank;
        candidates[offset]=rank<local.size()?local[rank]:INT32_MAX;
        logits[offset]=rank<local.size()?full[token*experts+local[rank]]:-INFINITY;
      }
    }
  }
  unsigned capacity=0;assert(MoeVirtualCapacity(tokens,8,experts,BM,Group,&capacity));
  auto* bindings=Managed<MoeBindingRecord>(capacity);
  auto* rows=Managed<MoeBindingRow>(tokens*8);
  unsigned chunks=(tokens+127)/128;
  auto* histogram=Managed<unsigned>(chunks*experts);
  auto* offsets=Managed<unsigned>(experts+1);auto* blocks=Managed<unsigned>(experts+1);
  auto* ticket=Managed<unsigned>(1);*ticket=0;
  MoeTopKOperands top{logits,candidates,indices,weights,tokens,experts,parts,row_stride,part_stride,1};
  MoeDispatchOperands p{indices,weights,bindings,rows,histogram,offsets,blocks,tokens,experts,capacity};
  for(unsigned epoch=0;epoch<3;++epoch) {
    for(unsigned v=0;v<capacity;++v)bindings[v]={0xdeadbeef,0xdeadbeef,0xdeadbeef,1};
    Top<<<TopK::Count(top),128>>>(top);Sync();
    for(unsigned token=0;token<tokens;++token) {
      double sum=0;
      for(unsigned rank=0;rank<8;++rank)
        sum+=std::exp(double(full[token*experts+expected[token*8+rank]]-full[token*experts+expected[token*8]]));
      for(unsigned rank=0;rank<8;++rank) {
        auto at=token*8+rank;assert(indices[at]==int(expected[at]));
        float ref=float(E(std::exp(double(full[token*experts+expected[at]]-full[token*experts+expected[token*8]]))/sum));
        assert(std::abs(float(weights[at])-ref)<=1e-5f+.008f*std::abs(ref));
      }
    }
    if(tokens*8<=4096 && epoch%2==0)Small<BM,Group><<<1,128>>>(p);
    else {
      if constexpr(Group) {Hist<BM,Group><<<chunks,128>>>(p,ticket);Sync();assert(*ticket==0);}
      Scatter<BM,Group><<<chunks,128>>>(p);
    }
    Sync();
    std::vector<unsigned> assignments(tokens*8);std::iota(assignments.begin(),assignments.end(),0);
    if constexpr(Group) {
      std::stable_sort(assignments.begin(),assignments.end(),[&](auto a,auto b){return expected[a]<expected[b];});
    }
    for(unsigned row=0;row<tokens*8;++row) {
      auto assignment=assignments[row];auto const& actual=rows[row];
      assert(actual.token==assignment/8 && actual.rank==assignment%8);
      assert(actual.weight_bf16==reinterpret_cast<std::uint16_t*>(weights)[assignment]);
      assert(actual.reserved16==0 && actual.reserved32==0);
    }
    unsigned used=0;
    if constexpr(Group) {
      unsigned first=0;
      for(unsigned e=0;e<experts;++e) {
        unsigned end=first;while(end<assignments.size() && expected[assignments[end]]==e)++end;
        assert(offsets[e]==first && blocks[e]==used);
        for(unsigned begin=first;begin<end;begin+=BM) {
          auto const& b=bindings[used++];
          assert(b.valid==1 && b.expert==e && b.row_begin==begin && b.row_count==std::min(unsigned(BM),end-begin));
        }
        first=end;
      }
      assert(offsets[experts]==tokens*8 && blocks[experts]==used);
    }else {
      used=capacity;
      for(unsigned row=0;row<capacity;++row)assert(bindings[row].valid==1 && bindings[row].expert==expected[row] &&
          bindings[row].row_begin==row && bindings[row].row_count==1);
    }
    for(unsigned v=used;v<capacity;++v)assert(bindings[v].valid==0);
  }
  cudaFree(logits);cudaFree(candidates);cudaFree(indices);cudaFree(weights);cudaFree(bindings);cudaFree(rows);
  cudaFree(histogram);cudaFree(offsets);cudaFree(blocks);cudaFree(ticket);return 1;
}

template<int Rows,int Columns>using Combine=MoECombineTaskBody<Arch,8,Rows,Columns>;
template<int Rows,int Columns>
__global__ void Reduce(MoeCombineOperands p) {
  __shared__ typename Combine<Rows,Columns>::SharedStorage s;
  Combine<Rows,Columns>::Run(p,blockIdx.x,s);
}
template<int Rows,int Columns>
__global__ void Weighted(MoeCombineOperands p,E const* source,unsigned* tickets) {
  __shared__ typename Combine<Rows,Columns>::SharedStorage s;
  __shared__ unsigned last;
  unsigned task=blockIdx.x/8,rank=blockIdx.x%8,tiles=(p.channels+Columns-1)/Columns;
  unsigned first_row=task/tiles*Rows,first_column=task%tiles*Columns;
  for(unsigned i=threadIdx.x;i<Rows*Columns;i+=128) {
    unsigned token=first_row+i/Columns,column=first_column+i%Columns;
    if(token<p.tokens && column<p.channels) {
      auto offset=token*p.token_stride+rank*p.rank_stride+column;
      const_cast<E*>(p.partials)[offset]=source[offset];
    }
  }
  unsigned contribution=min(unsigned(Rows),p.tokens-first_row);
  executor::LastArriver::RunWeighted(tickets+task,8*contribution,contribution,&last,[&] {
    Combine<Rows,Columns>::Run(p,task,s);
  });
}
template<int Rows,int Columns>
unsigned Combination(unsigned tokens,unsigned channels) {
  unsigned tiles=(channels+Columns-1)/Columns,row_stride=channels+7;
  unsigned rank_stride=channels+3,token_stride=8*rank_stride+5,stats_stride=tiles*2+4;
  auto* source=Managed<E>(tokens*token_stride);
  auto* partials=Managed<E>(tokens*token_stride);
  auto* residual=Managed<E>(tokens*row_stride);
  auto* weights=Managed<E>(tokens*8);
  auto* output=Managed<E>(tokens*row_stride+16);
  auto* stats=Managed<float>(tokens*stats_stride+16);
  for(unsigned token=0;token<tokens;++token) {
    for(unsigned rank=0;rank<8;++rank) {
      weights[token*8+rank]=E(float(rank+1)/36);
      for(unsigned channel=0;channel<channels;++channel)
        source[token*token_stride+rank*rank_stride+channel]=E(float(int((token*31+rank*17+channel*13)%173)-86)*.03125f);
    }
    for(unsigned channel=0;channel<channels;++channel)
      residual[token*row_stride+channel]=E(float(int((token*19+channel*7)%151)-75)*.125f);
  }
  MoeCombineOperands p{partials,weights,residual,output+8,stats+8,tokens,channels,
                       token_stride,rank_stride,row_stride,stats_stride};
  unsigned count=Combine<Rows,Columns>::Count(p);auto* tickets=Managed<unsigned>(count);
  std::fill(tickets,tickets+count,0);
  std::vector<E> expected(tokens*row_stride,E(-12345));
  for(unsigned token=0;token<tokens;++token)for(unsigned column=0;column<channels;++column) {
    float sum=0;
    for(unsigned rank=0;rank<8;++rank) {
      float product=float(E(float(weights[token*8+rank])*float(source[token*token_stride+rank*rank_stride+column])));
      sum+=product;
    }
    expected[token*row_stride+column]=E(float(E(sum))+float(residual[token*row_stride+column]));
  }
  for(unsigned epoch=0;epoch<3;++epoch) {
    std::fill(partials,partials+tokens*token_stride,E(-12345));
    std::fill(output,output+tokens*row_stride+16,E(-12345));
    std::fill(stats,stats+tokens*stats_stride+16,-12345.f);
    if(epoch==0) {
      assert(cudaMemcpy(partials,source,tokens*token_stride*sizeof(E),cudaMemcpyDefault)==cudaSuccess);
      Reduce<Rows,Columns><<<count,128>>>(p);
    }else Weighted<Rows,Columns><<<count*8,128>>>(p,source,tickets);
    Sync();
    for(unsigned i=0;i<count;++i)assert(tickets[i]==0);
    for(unsigned token=0;token<tokens;++token) {
      for(unsigned column=0;column<row_stride;++column)
        assert(output[8+token*row_stride+column]==expected[token*row_stride+column]);
      for(unsigned tile=0;tile<tiles;++tile) {
        double sum=0,square=0;
        for(unsigned column=tile*Columns;column<std::min(channels,(tile+1)*Columns);++column) {
          double value=float(expected[token*row_stride+column]);sum+=value;square+=value*value;
        }
        float a=stats[8+token*stats_stride+tile*2],b=stats[8+token*stats_stride+tile*2+1];
        assert(std::abs(a-sum)<=1e-5+1e-5*std::abs(sum));
        assert(std::abs(b-square)<=1e-5+1e-5*std::abs(square));
      }
      for(unsigned i=tiles*2;i<stats_stride;++i)assert(stats[8+token*stats_stride+i]==-12345);
    }
    for(unsigned i=0;i<8;++i) {
      assert(output[i]==E(-12345) && output[8+tokens*row_stride+i]==E(-12345));
      assert(stats[i]==-12345 && stats[8+tokens*stats_stride+i]==-12345);
    }
  }
  cudaFree(source);cudaFree(partials);cudaFree(residual);cudaFree(weights);
  cudaFree(output);cudaFree(stats);cudaFree(tickets);return 1;
}
int main() {
  unsigned routing=0,combine=0;
  for(unsigned tokens:{1,2,4,8,16,32,64,128,256,512,1024,2048,4096}) {
    routing+=Routing<16,false>(tokens,128,32,1);
    routing+=Routing<16,true>(tokens,128,32,1);
    routing+=Routing<32,true>(tokens,128,32,1);
    routing+=Routing<64,true>(tokens,128,32,1);
    routing+=Routing<128,true>(tokens,128,32,1);
  }
  for(unsigned experts:{8,16,128})for(unsigned mode:{0,2}) {
    routing+=Routing<16,true>(17,experts,3,mode);
    routing+=Routing<32,true>(513,experts,16,mode);
  }
  for(unsigned tokens:{1,17,65})for(unsigned channels:{1,127,257}) {
    combine+=Combination<1,128>(tokens,channels);
    combine+=Combination<4,64>(tokens,channels);
  }
  combine+=Combination<4,128>(4096,2048);
  std::printf("{\"passed\":true,\"routing_cases\":%u,\"combine_cases\":%u,\"epochs\":3,"
      "\"scope\":\"MoE partial top-k, stable dispatch, BF16 rank combine, histogram LA and weighted combine LA\"}\n",routing,combine);
}
