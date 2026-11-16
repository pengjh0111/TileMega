#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
void Check(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
__host__ __device__ float Value(int row,int col) {return float((row*37+col*13)%127-63);}
template<int Cols>
__global__ void Quad(E* out,int stride,int rows,int width) {
  using Side=Sm90F3SideLayout<128,Cols,false,0,false>;
  using G=Sm90F3Gemm<128,Cols,2,1,Sm90F3Mode::kNormal,128,Cols,64,Cols,Sm90F3Chain<Side>>;
  typename G::Mma mma;
  auto owned=mma.get_slice(int(threadIdx.x)).partition_C(
      cute::make_identity_tensor(cute::Shape<cute::_64,cute::Int<Cols>>{}));
  typename G::Fragment tile;
  CUTE_UNROLL
  for(int i=0;i<int(cute::size(tile));++i)tile(i)=Value(cute::get<0>(owned(i)),cute::get<1>(owned(i)));
  CUTE_UNROLL
  for(int group=0;group<Cols/32;++group)G::StoreQuad(tile,owned,out,stride,0,0,0,0,rows,width,group);
}
std::uint64_t output_hash=1469598103934665603ull;
template<int Cols> void Test() {
  for(int rows:{1,7,63,64})for(int width:{1,7,8,9,Cols-1,Cols})
    for(int padding:{4,8})for(int offset:{0,2}) {
      int stride=Cols+padding;
      std::vector<E> reference(66*stride+8,E::bitcast(0xdead)),actual=reference;
      E* out=nullptr;Check(cudaMalloc(&out,actual.size()*sizeof(E)));
      Check(cudaMemcpy(out,actual.data(),actual.size()*sizeof(E),cudaMemcpyHostToDevice));
      Quad<Cols><<<1,128>>>(out+offset,stride,rows,width);Check(cudaGetLastError());
      Check(cudaMemcpy(actual.data(),out,actual.size()*sizeof(E),cudaMemcpyDeviceToHost));
      Check(cudaFree(out));
      for(int row=0;row<rows;++row)for(int col=0;col<width;++col)
        reference[offset+row*stride+col]=E(Value(row,col));
      for(std::size_t i=0;i<actual.size();++i) {
        if(actual[i].raw()!=reference[i].raw()) {
          std::fprintf(stderr,"Cols%d rows%d width%d pad%d offset%d index%zu actual%x expected%x\n",Cols,rows,width,padding,offset,i,actual[i].raw(),reference[i].raw());
          throw std::runtime_error("quad store mismatch or overwritten canary");
        }
        output_hash=(output_hash^actual[i].raw())*1099511628211ull;
      }
    }
}
int main() try {Test<32>();Test<64>();Test<128>();Test<256>();std::printf("384 exact quad-store/tail/canary cases PASS\noutput_hash=%llx\n",(unsigned long long)output_hash);return 0;}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
