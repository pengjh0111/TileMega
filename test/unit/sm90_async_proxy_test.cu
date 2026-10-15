// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/tasks/ClusterSync.cuh>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
#ifndef TILEMEGA_TEST_ARCH_ID
#define TILEMEGA_TEST_ARCH_ID 900
#endif
using TestArch=typename tilemega::arch::ArchFromId<TILEMEGA_TEST_ARCH_ID>::type;
constexpr int kElements=64,kIterations=128;
void Check(cudaError_t error) {
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
void Driver(CUresult error) {
  if(error!=CUDA_SUCCESS) {
    char const* name=nullptr;cuGetErrorName(error,&name);
    throw std::runtime_error(name?name:"CUDA driver error");
  }
}

template<class Arch>
__global__ void GlobalHandoff(int* values,unsigned* ready,unsigned* done,int* failures) {
  using A=tilemega::codegen::executor::Async<Arch>;
  if constexpr(A::Caps::kBulkCopy) {
    __shared__ __align__(16) int tile[kElements];
    __shared__ __align__(8) std::uint64_t full;
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> published(*ready),consumed(*done);
    if(blockIdx.x==1 && threadIdx.x==0){A::Init(&full,1);A::InitFence();}
    __syncthreads();
    for(int iteration=0;iteration<kIterations;++iteration) {
      if(blockIdx.x==0) {
        for(int i=threadIdx.x;i<kElements;i+=blockDim.x)values[i]=iteration*kElements+i;
        __syncthreads();
        if(threadIdx.x==0) {
          published.store(iteration+1,cuda::memory_order_release);
          while(consumed.load(cuda::memory_order_acquire)!=unsigned(iteration+1)){}
        }
        __syncthreads();
      } else {
        if(threadIdx.x==0) {
          while(published.load(cuda::memory_order_acquire)!=unsigned(iteration+1)){}
          A::ProxyAsyncGlobalFence();
          A::ExpectTx(&full,sizeof(tile));A::Bulk(tile,values,sizeof(tile),&full);
        }
        __syncthreads();A::Wait(&full,iteration&1);
        for(int i=threadIdx.x;i<kElements;i+=blockDim.x)
          if(tile[i]!=iteration*kElements+i)atomicAdd(failures,1);
        __syncthreads();
        if(threadIdx.x==0)consumed.store(iteration+1,cuda::memory_order_release);
        __syncthreads();
      }
    }
  }
}

template<int Peers,class Arch>
__global__ void PeerMailbox(int* failures) {
  using A=tilemega::codegen::executor::Async<Arch>;
  if constexpr(A::Caps::kDsmem) {
    using Cluster=tilemega::codegen::ClusterSync<Arch>;
    unsigned rank=Cluster::Rank();
    __shared__ __align__(16) int source[kElements];
    __shared__ __align__(16) int inbox[Peers-1][kElements];
    __shared__ __align__(8) std::uint64_t full[Peers-1],notified[Peers-1];
    if(threadIdx.x==0) {
      for(int r=0;r<Peers-1;++r){A::Init(full+r,1);A::Init(notified+r,1);}
      A::InitFence();
    }
    __syncthreads();Cluster::Sync();
    for(int iteration=0;iteration<kIterations;++iteration) {
      for(int i=threadIdx.x;i<kElements;i+=blockDim.x)
        source[i]=int(rank)*100000+iteration*kElements+i;
      A::FenceProxyAsyncShared();__syncthreads();
      if(rank==0 && threadIdx.x==0)
        for(int r=0;r<Peers-1;++r)A::ExpectTx(full+r,sizeof(source));
      Cluster::Sync();
      if(rank!=0 && threadIdx.x==0) {
        A::BulkSmemToPeer(A::Mapa(inbox[rank-1],0),source,sizeof(source),A::Mapa(full+rank-1,0));
        A::RemoteArrive(A::Mapa(notified+rank-1,0));
      }
      if(rank==0) {
        for(int r=0;r<Peers-1;++r){A::Wait(full+r,iteration&1);A::Wait(notified+r,iteration&1);}
        for(int i=threadIdx.x;i<kElements;i+=blockDim.x) {
          int sum=0,expected=0;
          for(int r=0;r<Peers-1;++r) {
            int value=(r+1)*100000+iteration*kElements+i;
            if(inbox[r][i]!=value)atomicAdd(failures,1);
            sum+=inbox[r][i];expected+=value;
          }
          if(sum!=expected)atomicAdd(failures,1);
        }
      }
      // The diagnostic uses a cluster rendezvous to protect both source and
      // destination reuse; it does not certify an asynchronous mailbox ABI.
      Cluster::Sync();
    }
  }
}

template<int Peers,class Arch>
__global__ void Multicast(CUtensorMap const* map,int* failures) {
  using A=tilemega::codegen::executor::Async<Arch>;
  if constexpr(A::Caps::kTmaMulticast) {
    using Cluster=tilemega::codegen::ClusterSync<Arch>;
    __shared__ __align__(128) int tile[kElements];
    __shared__ __align__(8) std::uint64_t full;
    if(threadIdx.x==0){A::Init(&full,1);A::InitFence();A::ExpectTx(&full,sizeof(tile));}
    __syncthreads();Cluster::Sync();
    if(Cluster::Rank()==0 && threadIdx.x==0) {
      A::PrefetchTensorMap(map);
      A::Tensor2DMulticast(tile,map,0,0,&full,(1u<<Peers)-1);
    }
    __syncthreads();A::Wait(&full,0);
    for(int i=threadIdx.x;i<kElements;i+=blockDim.x)
      if(tile[i]!=i+7)atomicAdd(failures,1);
    Cluster::Sync();
  }
}

template<class Arch>
__global__ void Reduce(float* result) {
  using A=tilemega::codegen::executor::Async<Arch>;
  if constexpr(A::Caps::kBulkReduce) {
    __shared__ __align__(16) float values[kElements];
    for(int i=threadIdx.x;i<kElements;i+=blockDim.x)values[i]=float(blockIdx.x+1);
    A::FenceProxyAsyncShared();__syncthreads();
    if(threadIdx.x==0) {
      A::BulkReduceAddF32(result,values,sizeof(values));A::BulkCommit();
      A::template BulkWait<0,true>();A::template BulkWait<0>();A::ProxyAsyncGlobalFence();
    }
  }
}

template<int Peers> void ClusterTests(CUtensorMap* map,int* failures) {
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim={Peers,1,1};cudaLaunchConfig_t config{};
  config.gridDim=dim3(Peers);config.blockDim=dim3(128);
  config.attrs=&attr;config.numAttrs=1;
  Check(cudaLaunchKernelEx(&config,PeerMailbox<Peers,TestArch>,failures));
  Check(cudaDeviceSynchronize());
  Check(cudaLaunchKernelEx(&config,Multicast<Peers,TestArch>,map,failures));
  Check(cudaDeviceSynchronize());
}
}

int main() {
  try {
    cudaDeviceProp props{};Check(cudaGetDeviceProperties(&props,0));
    if(props.major!=9){std::puts("SKIP: runtime Hopper required");return 77;}
    int *values,*failures;unsigned *ready,*done;float *result;
    Check(cudaMalloc(&values,kElements*sizeof(int)));Check(cudaMalloc(&failures,sizeof(int)));
    Check(cudaMalloc(&ready,sizeof(unsigned)));Check(cudaMalloc(&done,sizeof(unsigned)));
    Check(cudaMalloc(&result,kElements*sizeof(float)));
    Check(cudaMemset(failures,0,sizeof(int)));Check(cudaMemset(ready,0,sizeof(unsigned)));
    Check(cudaMemset(done,0,sizeof(unsigned)));Check(cudaMemset(result,0,kElements*sizeof(float)));
    GlobalHandoff<TestArch><<<2,128>>>(values,ready,done,failures);
    Check(cudaDeviceSynchronize());
    int host[kElements];for(int i=0;i<kElements;++i)host[i]=i+7;
    Check(cudaMemcpy(values,host,sizeof(host),cudaMemcpyHostToDevice));
    CUtensorMap host_map{};cuuint64_t dims[2]={16,4},strides[1]={16*sizeof(int)};
    cuuint32_t box[2]={16,4},element[2]={1,1};
    Driver(cuTensorMapEncodeTiled(&host_map,CU_TENSOR_MAP_DATA_TYPE_INT32,2,values,dims,strides,
        box,element,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUtensorMap* map;Check(cudaMalloc(&map,sizeof(host_map)));
    Check(cudaMemcpy(map,&host_map,sizeof(host_map),cudaMemcpyHostToDevice));
    ClusterTests<2>(map,failures);ClusterTests<4>(map,failures);ClusterTests<8>(map,failures);
    Reduce<TestArch><<<2,128>>>(result);Check(cudaDeviceSynchronize());
    float sums[kElements];Check(cudaMemcpy(sums,result,sizeof(sums),cudaMemcpyDeviceToHost));
    int errors;Check(cudaMemcpy(&errors,failures,sizeof(int),cudaMemcpyDeviceToHost));
    for(float value:sums)if(value!=3.0f)++errors;
    Check(cudaFree(map));Check(cudaFree(result));Check(cudaFree(done));Check(cudaFree(ready));
    Check(cudaFree(failures));Check(cudaFree(values));
    std::printf("async proxy: cross-CTA global %d phases; peer mailboxes and multicast clusters 2/4/8; bulk reduction: %s (%d errors)\n",
        kIterations,errors?"FAIL":"PASS",errors);
    return errors?1:0;
  }catch(std::exception const& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
