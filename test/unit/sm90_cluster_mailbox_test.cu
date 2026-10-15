// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_ARCH_ID 900
#include <tilemega/Codegen/executor/ClusterMailbox.cuh>
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <cstdio>
#include <vector>

using Mailbox=tilemega::codegen::executor::ClusterMailbox<tilemega::arch::Sm90,
    TILEMEGA_SERVING_CLUSTER_DIM,1024>;

__global__ void MailboxKernel(float* output,unsigned tiles,unsigned episodes) {
  __shared__ typename Mailbox::Storage mailbox;
  __shared__ float scratch[1024];
  Mailbox::Init(mailbox);
  unsigned const rank=blockIdx.x%TILEMEGA_SERVING_CLUSTER_DIM;
  unsigned const cluster=blockIdx.x/TILEMEGA_SERVING_CLUSTER_DIM;
  unsigned const clusters=gridDim.x/TILEMEGA_SERVING_CLUSTER_DIM;
  for(unsigned step=0;step<episodes;++step)
    for(unsigned tile=cluster;tile<tiles;tile+=clusters) {
      Mailbox::template Run<1024>(mailbox,[&] {
        for(unsigned i=threadIdx.x;i<1024;i+=blockDim.x)
          scratch[i]=float(int((i*17+tile*13+step*7+rank*19)%97)-48)/32;
        __syncthreads();return scratch;
      },[&](float* values) {
        for(unsigned i=threadIdx.x;i<1024;i+=blockDim.x)
          output[(std::size_t(step)*tiles+tile)*1024+i]=values[i];
      });
    }
  Mailbox::Close();
}

int main() {
  using namespace tilemega::codegen::executor;
  int clusters=0;
  if(ServingActiveClusters(&clusters,MailboxKernel,128,0,
       TILEMEGA_SERVING_CLUSTER_DIM*2)!=cudaSuccess || clusters<=0)return 2;
  unsigned const active=unsigned(clusters),c=TILEMEGA_SERVING_CLUSTER_DIM;
  for(unsigned tiles:{1u,active-1,active+1,active*3+1}) {
    if(!tiles)continue;
    unsigned constexpr episodes=13;
    std::vector<float> host(std::size_t(tiles)*episodes*1024);
    float* output=nullptr;
    if(cudaMalloc(&output,host.size()*sizeof(float))!=cudaSuccess)return 3;
    if(LaunchServing(MailboxKernel,active*c,128,0,nullptr,false,output,tiles,episodes)!=cudaSuccess ||
       cudaDeviceSynchronize()!=cudaSuccess)return 4;
    if(cudaMemcpy(host.data(),output,host.size()*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)return 5;
    for(unsigned step=0;step<episodes;++step)for(unsigned tile=0;tile<tiles;++tile)
      for(unsigned i=0;i<1024;++i) {
        float expected=0;
        for(unsigned rank=0;rank<c;++rank)
          expected+=float(int((i*17+tile*13+step*7+rank*19)%97)-48)/32;
        if(host[(std::size_t(step)*tiles+tile)*1024+i]!=expected) {
          std::fprintf(stderr,"mailbox c%u tile%u step%u element%u mismatch\n",c,tile,step,i);
          return 6;
        }
      }
    cudaFree(output);
  }
  std::printf("PASS deterministic mailbox c%u clusters%u double-slot/13 episodes/tail clusters\n",c,active);
}
