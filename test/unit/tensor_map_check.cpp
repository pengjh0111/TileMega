#include <tilemega/Codegen/executor/TensorMap.h>
#include <cstdio>
int main(){using namespace tilemega::codegen::executor;
 void* memory=nullptr;if(cudaMalloc(&memory,1024*1024)!=cudaSuccess)return 2;
 int encoded=0,unsupported=0;
 for(int cols:{64,128,2048,8192})for(int box_rows:{16,32,64}){
  TensorMapShape shape;shape.columns=cols;shape.rows=64;shape.stride_bytes=cols*2;shape.box_rows=box_rows;
  TensorMap map;auto result=EncodeTensorMap(map,memory,shape);
  if(result==CUDA_SUCCESS)++encoded;else if(result==CUDA_ERROR_NOT_SUPPORTED)++unsupported;else{std::printf("FAIL code=%d\n",int(result));return 1;}
 }
 bool rejected=false;try{TensorMapShape shape;shape.columns=64;shape.rows=1;shape.stride_bytes=127;shape.box_rows=16;shape.Validate(memory);}catch(std::invalid_argument const&){rejected=true;}
 cudaFree(memory);std::printf("encoded=%d unsupported=%d invalid_stride_rejected=%d\n",encoded,unsupported,rejected);return rejected?0:1;
}
