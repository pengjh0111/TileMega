// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/AttentionMergeTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using Element = cutlass::bfloat16_t;
void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<int D, int Q, int S>
__global__ void Merge(float const* partial, float const* lse, Element* out,
                      int groups, int cap, int extent, int past) {
  tilemega::codegen::AttentionMergeTaskBody<D,Q,S>::Run(
      partial,lse,out,int(blockIdx.x)/groups,int(blockIdx.x)%groups,
      groups,cap,extent,past);
}
template<int D, int Q, int S>
int Cases() {
  constexpr int batches=3,groups=3,cap=1088,rows=Q*S;
  int cases=0;
  for(int extent:{64,256,cap})for(int total:{S,64,65,1087}) {
    int past=total-S;if(past<0)continue;
    int blocks=(cap+extent-1)/extent,live=(total+extent-1)/extent;
    std::vector<float> partial(batches*groups*blocks*rows*D,NAN);
    std::vector<float> lse(batches*groups*blocks*rows,NAN);
    std::vector<Element> expected(batches*S*groups*Q*D),actual(expected.size());
    for(int b=0;b<batches;++b)for(int g=0;g<groups;++g)
      for(int row=0;row<rows;++row) {
        float peak=-INFINITY;
        for(int c=0;c<live;++c) {
          auto base=((b*groups+g)*blocks+c)*rows+row;
          lse[base]=float((row*3+c*7+b+g)%41-20);
          peak=std::max(peak,lse[base]);
          for(int d=0;d<D;++d)
            partial[base*D+d]=float((d*17+base*11)%67-33)/16;
        }
        for(int d=0;d<D;++d) {
          float sum=0,value=0;
          for(int c=0;c<live;++c) {
            auto base=((b*groups+g)*blocks+c)*rows+row;
            float weight=std::exp2(lse[base]-peak);
            sum+=weight;value+=weight*partial[base*D+d];
          }
          int offset=((b*S+row/Q)*groups*Q+g*Q+row%Q)*D+d;
          expected[offset]=Element(value/sum);
        }
      }
    float *dp,*dl;Element* out;
    Check(cudaMalloc(&dp,partial.size()*sizeof(float)));
    Check(cudaMalloc(&dl,lse.size()*sizeof(float)));
    Check(cudaMalloc(&out,actual.size()*sizeof(Element)));
    Check(cudaMemcpy(dp,partial.data(),partial.size()*sizeof(float),cudaMemcpyHostToDevice));
    Check(cudaMemcpy(dl,lse.data(),lse.size()*sizeof(float),cudaMemcpyHostToDevice));
    Merge<D,Q,S><<<batches*groups,128>>>(dp,dl,out,groups,cap,extent,past);
    Check(cudaGetLastError());
    Check(cudaMemcpy(actual.data(),out,actual.size()*sizeof(Element),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<actual.size();++i) {
      float error=std::abs(float(actual[i])-float(expected[i]));
      if(!std::isfinite(float(actual[i])) || error>0.0078125f*std::abs(float(expected[i]))+0.00390625f)
        throw std::runtime_error("vector merge mismatch at "+std::to_string(i));
    }
    Check(cudaFree(dp));Check(cudaFree(dl));Check(cudaFree(out));++cases;
  }
  return cases;
}
int main() {
  try {
    int count=Cases<64,4,1>()+Cases<128,2,1>()+
              Cases<64,4,64>()+Cases<128,2,64>();
    std::printf("serving vector merge: %d/%d pass (inactive blocks poisoned)\n",count,count);
  } catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
