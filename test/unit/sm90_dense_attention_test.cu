// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseAttentionStream.cuh>
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <vector>
#include <set>
#include <cstdio>
#include <cmath>
#include <stdexcept>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* device(size_t count){T* p;check(cudaMalloc(&p,count*sizeof(T)));return p;}
template<int D,bool Norm>__global__ void rotate(BF* x,BF* w,BF* c,BF* s,BF* actual,BF* expected){
 Sm90DenseAttentionMath::Rotate<D,Norm>(x,w,c,s,1e-6f,actual);
 tilemega::codegen::FusedAttentionTaskBody<tilemega::arch::Sm90,D,2,1,16,32,Norm>::RotateRow(x,w,c,s,1e-6f,[&](int d,uint4 v){*reinterpret_cast<uint4*>(expected+d)=v;});
}
template<int D,bool Norm>int rotation(){
 std::vector<BF>x(D),w(D),c(D),s(D),a(D),b(D);
 for(int i=0;i<D;++i){x[i]=BF((i%31-15)*.125f);w[i]=BF(.75f+(i%7)*.0625f);c[i]=BF(std::cos(i*.03f));s[i]=BF(std::sin(i*.03f));}
 auto* dx=device<BF>(D);auto* dw=device<BF>(D);auto* dc=device<BF>(D);auto* ds=device<BF>(D);auto* da=device<BF>(D);auto* db=device<BF>(D);
 for(auto pair:{std::make_pair(dx,&x),std::make_pair(dw,&w),std::make_pair(dc,&c),std::make_pair(ds,&s)})check(cudaMemcpy(pair.first,pair.second->data(),D*2,cudaMemcpyHostToDevice));
 rotate<D,Norm><<<1,32>>>(dx,dw,dc,ds,da,db);check(cudaGetLastError());check(cudaDeviceSynchronize());
 check(cudaMemcpy(a.data(),da,D*2,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),db,D*2,cudaMemcpyDeviceToHost));int errors=0;
 for(int d=0;d<D;++d)if(a[d].raw()!=b[d].raw()){++errors;if(errors<3)std::fprintf(stderr,"rotate D%d norm%d d%d got%f expected%f\n",D,Norm,d,float(a[d]),float(b[d]));}
 for(auto* p:{dx,dw,dc,ds,da,db})check(cudaFree(p));return errors;
}
template<int B,int D,int Q,bool Norm,int S,int Work,int Keys=64,int Slots=3>__global__ __launch_bounds__(160)
void evaluate(Sm90DenseAttentionOperands p,std::uint64_t* grid){
 extern __shared__ __align__(1024) char shared[];auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);ring.Initialize();
 for(unsigned repeat=0;repeat<4;++repeat){
  using A=std::conditional_t<(B<=4),Sm90DenseAttention<B,D,Q,Norm,S,Work>,Sm90DenseAttentionStream<B,D,Q,Norm,S,Keys,Slots,Work>>;A::Run(p,shared);
  if(threadIdx.x==0){asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(grid),"l"(1ull):"memory");unsigned long long seen;do{asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(grid):"memory");}while(seen<(repeat+1)*gridDim.x);}__syncthreads();
 }ring.Invalidate();
}
BF position(int bg,int pos,int d){unsigned hash=unsigned(pos+1)*2654435761u ^ unsigned(d+1)*2246822519u ^ unsigned(bg+1)*3266489917u;return BF((int(hash%255)-127)/128.f);}
template<int B,int D,int Q,bool Norm,int S,int Work,int Keys=64,int Slots=3>int run(unsigned& cases){
 using A=std::conditional_t<(B<=4),Sm90DenseAttention<B,D,Q,Norm,S,Work>,Sm90DenseAttentionStream<B,D,Q,Norm,S,Keys,Slots,Work>>;
 constexpr int Groups=B*8,Capacity=1088;
 std::vector<BF>qkv(Groups*(Q+2)*D,BF(0)),value(Groups*Capacity*D),one(Capacity*D,BF(1)),zero(Capacity*D,BF(0)),out(Groups*Q*D);
 for(int bg=0;bg<Groups;++bg)for(int pos=0;pos<Capacity;++pos)for(int d=0;d<D;++d)value[(bg*Capacity+pos)*D+d]=position(bg,pos,d);
 auto* dq=device<BF>(qkv.size());auto* dk=device<BF>(value.size());auto* dv=device<BF>(value.size());auto* output=device<BF>(out.size());
 auto* cosine=device<BF>(one.size());auto* sine=device<BF>(zero.size());auto* norm=device<BF>(D);
 auto* partial=device<float>(Groups*S*Q*D);auto* lse=device<float>(Groups*S*Q);auto* arrivals=device<std::uint64_t>(Groups);auto* grid=device<std::uint64_t>(1);
 check(cudaMemset(dk,0,value.size()*2));check(cudaMemcpy(dv,value.data(),value.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(cosine,one.data(),one.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(sine,zero.data(),zero.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(norm,one.data(),D*2,cudaMemcpyHostToDevice));
 check(cudaFuncSetAttribute(evaluate<B,D,Q,Norm,S,Work,Keys,Slots>,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
 std::set<int> pasts={64,575,1086};for(int total:{S*4,S*16,S*64})for(int delta:{-1,0,1})if(total+delta-1>=64&&total+delta-1<=1086)pasts.insert(total+delta-1);
 int errors=0;
 for(int past:pasts){
  for(int bg=0;bg<Groups;++bg)for(int d=0;d<D;++d)qkv[(bg*(Q+2)+Q+1)*D+d]=position(bg,past,d);
  check(cudaMemcpy(dq,qkv.data(),qkv.size()*2,cudaMemcpyHostToDevice));check(cudaMemset(output,0xff,out.size()*2));check(cudaMemset(arrivals,0,Groups*8));check(cudaMemset(grid,0,8));
  Sm90DenseAttentionOperands p{dq,dk,dv,output,cosine,sine,norm,norm,partial,lse,arrivals,past,1e-6f};
  evaluate<B,D,Q,Norm,S,Work,Keys,Slots><<<132,160,A::kSharedBytes>>>(p,grid);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(out.data(),output,out.size()*2,cudaMemcpyDeviceToHost));
  for(int bg=0;bg<Groups;++bg)for(int d=0;d<D;++d){double sum=0;for(int pos=0;pos<=past;++pos)sum+=float(position(bg,pos,d));float exact=sum/(past+1);BF expected(exact);
   // One BF16 representable spacing allows a round-to-nearest tie to be
   // approached from either side; missing or repeated rows exceed this bound.
   float tolerance=std::max(0x1p-24f,std::ldexp(1.f,std::ilogb(std::fabs(exact))-7));
   for(int q=0;q<Q;++q){float actual=float(out[(bg*Q+q)*D+d]);if(!std::isfinite(actual)||std::fabs(actual-float(expected))>tolerance){if(errors<4)std::fprintf(stderr,"B%d D%d Q%d S%d W%d past%d bg%d d%d got%.9f exp%.9f tol%.9f\n",B,D,Q,S,Work,past,bg,d,actual,float(expected),tolerance);++errors;}}
  }++cases;
 }
 for(auto* p:{dq,dk,dv,output,cosine,sine,norm})check(cudaFree(p));check(cudaFree(partial));check(cudaFree(lse));check(cudaFree(arrivals));check(cudaFree(grid));return errors;
}
int main(){try{check(cudaSetDevice(0));unsigned cases=0;int errors=rotation<64,false>()+rotation<128,false>()+rotation<128,true>();
#define RUN(B,D,Q,N,S,W,Keys,Slots) errors+=run<B,D,Q,N,S,W,Keys,Slots>(cases)
 RUN(1,128,2,true,16,0,64,3);RUN(2,128,2,true,8,8,64,3);RUN(4,128,2,true,4,16,64,3);
 RUN(1,64,4,false,16,0,64,3);RUN(2,64,4,false,8,8,64,3);RUN(4,64,4,false,4,16,64,3);
 RUN(1,128,3,false,16,0,64,3);RUN(2,128,3,false,8,8,64,3);RUN(4,128,3,false,4,16,64,3);
 RUN(1,128,4,false,16,0,64,3);RUN(2,128,4,false,8,8,64,3);RUN(4,128,4,false,4,16,64,3);
 RUN(1,128,2,true,32,0,64,3);RUN(1,128,2,true,64,0,64,3);
 RUN(8,128,2,true,2,0,64,3);RUN(16,128,2,true,1,8,128,2);
 RUN(8,64,4,false,2,0,128,4);RUN(16,64,4,false,1,8,32,2);
 RUN(8,128,3,false,2,0,32,2);RUN(16,128,3,false,1,8,64,3);
 RUN(8,128,4,false,2,0,64,3);RUN(16,128,4,false,1,8,128,2);
 RUN(8,128,2,true,4,16,64,4);RUN(16,64,4,false,8,0,64,3);
#undef RUN
 std::printf("{\"errors\":%d,\"pass\":%s,\"position_cases\":%u,\"persistent_rounds_per_case\":4,\"rotation_cases\":3}\n",errors,errors?"false":"true",cases);return errors?1:0;}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
