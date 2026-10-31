// SPDX-License-Identifier: BSD-3-Clause
// E2 (prompt §4 A2 item 8): the per-CTA local routing must reproduce R4's
// last-arriver routing bit for bit, because the tokens are only unchanged if
// the schedule, the row order and the rounded weights all match.
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace tilemega::backend;

namespace {

// R4's reference: one CTA runs RouterFinish and publishes globally.
template<int Batch,int Experts,int TopK,bool FloatWeights,bool SoftmaxAll>
__global__ void reference_kernel(float const* logits,int* schedule,void* weights,int* index,
                                 std::uint64_t* counter,unsigned* done) {
  extern __shared__ __align__(1024) char shared[];
  // Tasks = 1 so the single CTA is the last arriver.
  using Finish=Sm90DenseRouterFinish<Batch,Experts,TopK,1,FloatWeights,SoftmaxAll>;
  Sm90DenseMoeOperands p{};
  p.output=const_cast<float*>(logits);p.schedule=schedule;p.route_weights=weights;
  p.route_index=index;p.counter=counter;p.gate_up_done=done;
  Finish{}(p,1,shared);
}

// E2: every CTA builds the schedule locally; CTA 0 republishes it.
template<int Batch,int Experts,int TopK,bool FloatWeights,bool SoftmaxAll>
__global__ void local_kernel(float const* logits,int* schedule,void* weights,int* index,
                             int* agreement) {
  using Route=Sm90DenseLocalRoute<Batch,Experts,TopK,FloatWeights,SoftmaxAll>;
  using Local=Sm90DenseLocalSchedule<Batch,Experts,TopK>;
  __shared__ __align__(16) char storage[sizeof(Local)];
  auto& routing=*reinterpret_cast<Local*>(storage);
  Route::Build(logits,routing);
  __syncthreads();
  if(blockIdx.x==0)Route::Publish(routing,schedule,weights,index);
  // Every CTA must agree with CTA 0: the design relies on all of them deriving
  // the same schedule without communicating.
  if(threadIdx.x==0) {
    unsigned long long digest=routing.active;
    for(int a=0;a<routing.active;++a) {
      digest=digest*1000003ull+unsigned(routing.expert[a]);
      digest=digest*1000003ull+routing.count[a];
      for(int r=0;r<routing.count[a];++r)digest=digest*1000003ull+routing.rows[a][r];
    }
    for(int s=0;s<Batch*TopK;++s)
      digest=digest*1000003ull+__float_as_uint(routing.weight[s]);
    agreement[blockIdx.x]=int(digest&0x7fffffff);
  }
}

int failures=0;

template<int Batch,int Experts,int TopK,bool FloatWeights,bool SoftmaxAll>
void compare(char const* label,std::vector<float> const& host_logits) {
  using Schedule=Sm90DenseMoeSchedule<Experts>;
  constexpr int kGrid=8;
  float* logits;int *schedule_a,*schedule_b,*index_a,*index_b,*agreement;
  void *weights_a,*weights_b;std::uint64_t* counter;unsigned* done;
  cudaMalloc(&logits,host_logits.size()*sizeof(float));
  cudaMemcpy(logits,host_logits.data(),host_logits.size()*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(&schedule_a,Schedule::kInts*sizeof(int));
  cudaMalloc(&schedule_b,Schedule::kInts*sizeof(int));
  cudaMalloc(&index_a,Batch*TopK*sizeof(int));
  cudaMalloc(&index_b,Batch*TopK*sizeof(int));
  cudaMalloc(&weights_a,Batch*TopK*sizeof(float));
  cudaMalloc(&weights_b,Batch*TopK*sizeof(float));
  cudaMalloc(&counter,sizeof(std::uint64_t));
  cudaMalloc(&done,Experts*sizeof(unsigned));
  cudaMalloc(&agreement,kGrid*sizeof(int));
  cudaMemset(schedule_a,0,Schedule::kInts*sizeof(int));
  cudaMemset(schedule_b,0,Schedule::kInts*sizeof(int));
  cudaMemset(weights_a,0,Batch*TopK*sizeof(float));
  cudaMemset(weights_b,0,Batch*TopK*sizeof(float));
  cudaMemset(counter,0,sizeof(std::uint64_t));
  // RouterFinish runs in the 128-thread consumer group (ComputeSync is
  // bar.sync 1, 128), so the reference kernel launches exactly that width.
  reference_kernel<Batch,Experts,TopK,FloatWeights,SoftmaxAll><<<1,128,2048>>>(
      logits,schedule_a,weights_a,index_a,counter,done);
  local_kernel<Batch,Experts,TopK,FloatWeights,SoftmaxAll><<<kGrid,128>>>(
      logits,schedule_b,weights_b,index_b,agreement);
  if(cudaDeviceSynchronize()!=cudaSuccess) {
    std::printf("FAIL %s: %s\n",label,cudaGetErrorString(cudaGetLastError()));++failures;return;
  }
  std::vector<int> a(Schedule::kInts),b(Schedule::kInts),ia(Batch*TopK),ib(Batch*TopK);
  std::vector<float> wa(Batch*TopK),wb(Batch*TopK);
  std::vector<int> agree(kGrid);
  cudaMemcpy(a.data(),schedule_a,a.size()*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(),schedule_b,b.size()*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(ia.data(),index_a,ia.size()*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(ib.data(),index_b,ib.size()*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(wa.data(),weights_a,wa.size()*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(wb.data(),weights_b,wb.size()*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(agree.data(),agreement,agree.size()*sizeof(int),cudaMemcpyDeviceToHost);
  int active=a[Schedule::Active()];
  bool ok=active==b[Schedule::Active()];
  for(int slot=0;slot<active && ok;++slot) {
    ok=ok && a[Schedule::Expert(slot)]==b[Schedule::Expert(slot)] &&
        a[Schedule::Count(slot)]==b[Schedule::Count(slot)];
    for(int r=0;r<a[Schedule::Count(slot)] && ok;++r)
      ok=ok && a[Schedule::Row(slot,r)]==b[Schedule::Row(slot,r)];
  }
  for(int s=0;s<Batch*TopK && ok;++s) {
    // Bitwise on the weights: the down epilogue multiplies by them.
    ok=ok && ia[s]==ib[s] && std::memcmp(&wa[s],&wb[s],sizeof(float))==0;
  }
  for(int g=1;g<kGrid && ok;++g)ok=ok && agree[g]==agree[0];
  std::printf("%-44s active=%3d %s\n",label,active,ok?"match":"MISMATCH");
  if(!ok)++failures;
  for(void* p:{(void*)logits,(void*)schedule_a,(void*)schedule_b,(void*)index_a,(void*)index_b,
               weights_a,weights_b,(void*)counter,(void*)done,(void*)agreement})cudaFree(p);
}

// Logits hold BF16 values (the router rounds them), with deliberate ties, +-0
// and equal-value pairs that force the tie-break on the expert number.
template<int Batch,int Experts>
std::vector<float> make_logits(unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-6.f,6.f);
  std::vector<float> values(std::size_t(Batch)*Experts);
  for(auto& v:values)v=float(cutlass::bfloat16_t(uniform(rng)));
  for(int b=0;b<Batch;++b) {
    float* row=values.data()+std::size_t(b)*Experts;
    if(Experts>=8) {
      row[0]=0.0f;row[1]=-0.0f;                  // +-0 fold onto each other
      row[2]=row[3]=float(cutlass::bfloat16_t(3.5f));   // exact tie
      row[4]=row[5]=row[6]=float(cutlass::bfloat16_t(-1.25f));
      row[Experts-1]=row[2];                     // a tie far from its partner
    }
  }
  return values;
}

template<int Experts,int Batch>
void sweep(unsigned seed) {
  char label[96];
  auto logits=make_logits<Batch,Experts>(seed);
  std::snprintf(label,sizeof label,"E%d B%-2d float-weights softmax-selected",Experts,Batch);
  compare<Batch,Experts,8,true,false>(label,logits);
  std::snprintf(label,sizeof label,"E%d B%-2d bf16-weights  softmax-selected",Experts,Batch);
  compare<Batch,Experts,8,false,false>(label,logits);
  std::snprintf(label,sizeof label,"E%d B%-2d float-weights softmax-all",Experts,Batch);
  compare<Batch,Experts,8,true,true>(label,logits);
  std::snprintf(label,sizeof label,"E%d B%-2d bf16-weights  softmax-all",Experts,Batch);
  compare<Batch,Experts,8,false,true>(label,logits);
}

}  // namespace

int main() {
  int count=0;
  if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) {
    std::printf("no CUDA device; skipping\n");return 0;
  }
  for(unsigned seed:{1u,7u,13u}) {
    sweep<128,1>(seed);sweep<128,2>(seed);sweep<128,4>(seed);
    sweep<128,8>(seed);sweep<128,16>(seed);
    sweep<64,1>(seed);sweep<64,2>(seed);sweep<64,4>(seed);
    sweep<64,8>(seed);sweep<64,16>(seed);
  }
  std::printf("%s (%d failures)\n",failures?"FAILED":"PASSED",failures);
  return failures?1:0;
}
