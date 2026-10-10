// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace tilemega::codegen;
void Check(cudaError_t error) {
  if(error!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}

#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
struct Payload {
  ConvDesc conv;
  BufferDesc buffer;
  GemmDesc gemm;
  StageDesc stage;
  ModelSpec model;
  GemmInvocation invocation;
  ServingGemmOperands operands;
};
__global__ void Inspect(Payload const* p,std::uint64_t* out) {
  if(threadIdx.x || blockIdx.x)return;
  out[0]=sizeof(Payload); out[1]=sizeof(GemmDesc); out[2]=sizeof(BufferDesc);
  out[3]=offsetof(GemmDesc,access); out[4]=offsetof(BufferDesc,layout);
  out[5]=(p->conv.h+2*p->conv.pad_h-p->conv.dilation_h*(p->conv.r-1)-1)/p->conv.stride_h+1;
  out[6]=p->buffer.layout.strides[0]*p->buffer.layout.logical[0];
  out[7]=p->gemm.access.a_row_offset+3*p->gemm.access.a_row_stride;
  out[8]=p->gemm.access.expert_stride*127;
  out[9]=p->gemm.chain.operations[0].parameter[0];
  out[10]=unsigned(p->gemm.chain.operations[1].activation);
  out[11]=p->gemm.chain.side[0].buffer;
  out[12]=p->stage.conv; out[13]=p->stage.rows_per_batch;
  out[14]=p->model.convolution_count;
  out[15]=p->invocation.access.rows_per_batch;
  out[16]=p->operands.chain.operations[0].parameter[0];
  out[17]=unsigned(OwnershipOf(p->stage.kind));
  out[18]=unsigned(p->buffer.layout.fill);
  out[19]=p->gemm.access.write.factor;
  out[20]=p->invocation.dm_buffers.count;
  out[21]=p->operands.dm_buffers.dtypes[0];
  out[22]=p->operands.dm_buffers.layouts[0].logical[3];
  out[23]=static_cast<std::uint64_t>(*static_cast<float const*>(p->operands.dm_buffers.data[0])*10);
}
#endif

int main(int argc,char** argv) {
  if(argc==2 && !std::strcmp(argv[1],"--layout")) {
    std::printf("{\"ModelDims\":%zu,\"GemmDesc\":%zu,\"BufferDesc\":%zu,"
                "\"StageDesc\":%zu,\"ModelSpec\":%zu,\"GemmInvocation\":%zu,"
                "\"ServingGemmOperands\":%zu}\n",sizeof(ModelDims),sizeof(GemmDesc),
                sizeof(BufferDesc),sizeof(StageDesc),sizeof(ModelSpec),
                sizeof(GemmInvocation),sizeof(ServingGemmOperands));
    return 0;
  }
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  Payload p{}; p.conv={2,224,224,3,64,7,7,2,2,3,3,1,1,112,112,0,1};
  p.buffer.layout={DmLayout::kNHWC,4,{2,112,112,3},{2,118,118,8},
      {118*118*8,118*8,8,1},3,3,3,3,DmFill::kNegativeInfinity};
  p.gemm.access.a_row_offset=7; p.gemm.access.a_row_stride=128;
  p.gemm.access.expert_stride=std::uint64_t(1)<<33;
  p.gemm.access.write={DmWriteKind::kPixelShuffle,2,1,kDmNoIndex};
  p.gemm.chain.count=2; p.gemm.chain.operations[0].parameter[0]=35;
  p.gemm.chain.operations[1].kind=DmEpilogueKind::kActivation;
  p.gemm.chain.operations[1].activation=DmActivation::kGeluTanh;
  p.gemm.chain.side_count=1; p.gemm.chain.side[0].buffer=73;
  p.stage.kind=TaskKind::kEncoderAttention; p.stage.conv=9; p.stage.rows_per_batch=128;
  p.model.convolution_count=48; p.invocation.access.rows_per_batch=4096;
  p.operands.chain.operations[0].parameter[0]=91;
  float* value; void** addresses; std::uint32_t* dtypes; DmBufferLayout* layouts;
  Check(cudaMalloc(&value,sizeof(float))); Check(cudaMalloc(&addresses,sizeof(void*)));
  Check(cudaMalloc(&dtypes,sizeof(std::uint32_t))); Check(cudaMalloc(&layouts,sizeof(DmBufferLayout)));
  float host_value=12.5f; void* host_address=value; std::uint32_t host_dtype=1;
  Check(cudaMemcpy(value,&host_value,sizeof(host_value),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(addresses,&host_address,sizeof(host_address),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(dtypes,&host_dtype,sizeof(host_dtype),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(layouts,&p.buffer.layout,sizeof(p.buffer.layout),cudaMemcpyHostToDevice));
  p.invocation.dm_buffers=p.operands.dm_buffers={addresses,layouts,dtypes,1};
  std::uint64_t expected[]={sizeof(Payload),sizeof(GemmDesc),sizeof(BufferDesc),
      offsetof(GemmDesc,access),offsetof(BufferDesc,layout),112,222784,391,
      (std::uint64_t(1)<<33)*127,35,3,73,9,128,48,4096,91,0,1,2,1,1,3,125};
  Payload* device; std::uint64_t* output; std::uint64_t actual[24]{};
  Check(cudaMalloc(&device,sizeof(p))); Check(cudaMalloc(&output,sizeof(actual)));
  Check(cudaMemcpy(device,&p,sizeof(p),cudaMemcpyHostToDevice));
  Inspect<<<1,32>>>(device,output); Check(cudaGetLastError());
  Check(cudaMemcpy(actual,output,sizeof(actual),cudaMemcpyDeviceToHost));
  Check(cudaFree(device)); Check(cudaFree(output));
  Check(cudaFree(value)); Check(cudaFree(addresses)); Check(cudaFree(dtypes)); Check(cudaFree(layouts));
  for(unsigned i=0;i<24;++i)if(actual[i]!=expected[i]) {
    std::fprintf(stderr,"descriptor[%u]: %llu != %llu\n",i,
        static_cast<unsigned long long>(actual[i]),static_cast<unsigned long long>(expected[i]));
    return 3;
  }
  std::puts("DM descriptor host/device interpretation: 24 checks passed"); return 0;
#else
  std::fputs("use --layout for a legacy ABI probe\n",stderr);return 2;
#endif
}
