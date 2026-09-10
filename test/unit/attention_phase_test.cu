// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_MODEL_BF16 1
#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionPhasedTaskBody.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace tc = tilemega::codegen;
constexpr int kThreads = tilemega::solver::kTensorBF16Threads;
struct Scratch { float attention[TILEMEGA_ATTENTION_MAX_TOTAL]; };
using Direct = tc::AttentionTaskBody<void, Scratch, kThreads>;
using Phased = tc::AttentionPhasedTaskBody<void, Scratch, kThreads>;
__global__ void DirectKernel(tc::Params p, tc::StageDesc stage) {
  __shared__ Scratch scratch;
  Direct::RunTask(p, stage, scratch, blockIdx.x);
}
__global__ void PhasedKernel(tc::Params p, tc::StageDesc stage) {
  extern __shared__ float storage[];
  Phased::RunTask(p, stage, *reinterpret_cast<Scratch*>(storage), blockIdx.x);
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int main(int argc, char** argv) try {
  std::mt19937 random(argc == 2 ? std::stoul(argv[1]) : 1);
  std::uniform_real_distribution<float> value(-2.0f, 2.0f);
  int cases = 0, exact = 0;
  float max_abs = 0;
  for (int group : {1, 2}) for (int seq : {4, 128}) for (int past : {3, 512}) {
    int heads = 4, width = 128, total = seq + past, queries = seq * heads;
    std::vector<std::size_t> elements{
      std::size_t(queries)*width, std::size_t(heads/group)*total*width,
      std::size_t(heads/group)*total*width, std::size_t(queries)*width};
    std::vector<tc::ModelElement*> buffers(6);
    for (int b = 0; b < 4; ++b) {
      Check(cudaMalloc(&buffers[b], elements[b]*sizeof(tc::ModelElement)));
      std::vector<tc::ModelElement> host(elements[b]);
      for (auto& x : host) x = tc::ModelElement(value(random));
      Check(cudaMemcpy(buffers[b],host.data(),host.size()*sizeof(host[0]),cudaMemcpyHostToDevice));
    }
    Check(cudaMalloc(&buffers[4],std::size_t(queries)*total*sizeof(float)));
    Check(cudaMalloc(&buffers[5],std::size_t(queries)*8*width*sizeof(float)));
    tc::ModelElement** table;
    Check(cudaMalloc(&table,buffers.size()*sizeof(buffers[0])));
    Check(cudaMemcpy(table,buffers.data(),buffers.size()*sizeof(buffers[0]),cudaMemcpyHostToDevice));
    tc::Params params{}; params.dims={seq,past,total}; params.buffers=table;
    tc::StageDesc stage{}; stage.kind=tc::TaskKind::kAttention;
    stage.extent=heads; stage.width=width; stage.group=group;
    for (int i=0;i<6;++i) stage.operand[i]=i;
    DirectKernel<<<queries,kThreads>>>(params,stage);
    Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
    std::vector<tc::ModelElement> expected(elements[3]),actual(elements[3]);
    Check(cudaMemcpy(expected.data(),buffers[3],expected.size()*sizeof(expected[0]),cudaMemcpyDeviceToHost));
    for (int chunks : {1,2,4,8}) {
      stage.operand[6]=chunks;
      for (auto phase : {tc::AttentionPhase::kScores,tc::AttentionPhase::kNormalize,
                        tc::AttentionPhase::kPartialValue,tc::AttentionPhase::kCombine}) {
        stage.operand[7]=static_cast<unsigned>(phase);
        int tasks=tc::AttentionPhaseTasks(phase,queries,chunks);
        std::size_t scratch=phase==tc::AttentionPhase::kScores
            ? std::size_t((total+chunks-1)/chunks)*sizeof(float) : 0;
        PhasedKernel<<<tasks,kThreads,scratch>>>(params,stage);
        Check(cudaGetLastError());
      }
      Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(actual.data(),buffers[3],actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost));
      if (chunks==1) {
        if (std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(actual[0])))
          throw std::runtime_error("chunk=1 differs bitwise from direct attention");
        ++exact;
      }
      int mismatch=0;
      for (std::size_t i=0;i<actual.size();++i) {
        float error=std::abs(float(actual[i])-float(expected[i]));
        max_abs=std::max(max_abs,error);
        if (!std::isfinite(float(actual[i])) || error>1.6e-2f+1.6e-2f*std::abs(float(expected[i]))) ++mismatch;
      }
      if (mismatch) throw std::runtime_error("phased attention violates unchanged BF16 criterion");
      ++cases;
    }
    for (auto* buffer : buffers) Check(cudaFree(buffer));
    Check(cudaFree(table));
  }
  std::printf("ATTENTION_PHASE cases=%d chunk1_bits=%d max_abs=%.9g PASS\n",cases,exact,max_abs);
} catch (std::exception const& e) { std::fprintf(stderr,"attention-phase: %s\n",e.what()); return 1; }
