// Actual TM32 ldmatrix/mma.sync loop on resident production-layout operands.
// No TMA, epilogue or fitted throughput; normalize 64 K64 iterations directly.
template<int TN> __global__ void Mma32Resident(float* output,U64* stamps,int count) {
  using Math=Sm90ForwardMma32<TN>;
  extern __shared__ __align__(1024) char storage[];
  auto* a=reinterpret_cast<BF*>(storage);auto* b=a+32*64;
  char* scratch=storage+2*(32+TN)*64;
  if(threadIdx.x<128) {
    for(int i=threadIdx.x;i<32*64;i+=128)a[i]=BF(0.25f);
    for(int i=threadIdx.x;i<TN*64;i+=128)b[i]=BF(0.5f);
  }
  __syncthreads();
  if(threadIdx.x<128)for(int sample=0;sample<count;++sample) {
    typename Math::Accumulator accum{};
    codegen::executor::ComputeSync();U64 begin=Cycles();
    #pragma unroll 1
    for(int repeat=0;repeat<64;++repeat)Math::Accumulate(b,a,accum);
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
    auto* values=Math::Materialize(accum,scratch);
    for(int i=threadIdx.x;i<32*TN;i+=128)output[i]=values[i];
    codegen::executor::ComputeSync();
  }
}
template<int TN>void Mma32Case() {
  constexpr int bytes=2*(32+TN)*64+4*32*TN;
  std::string key="tm=32;tn="+std::to_string(TN)+";tk=64;repeat=64;impl=mma_sync";
  Add("mma32_"+std::to_string(TN),"matrix_k64_ns",key,bytes,[=](int count) {
    Buffer<float> output(32*TN);Buffer<U64> stamps(count);
    Shared(Mma32Resident<TN>,bytes);
    Mma32Resident<TN><<<1,160,bytes>>>(output,stamps,count);
    Check(cudaDeviceSynchronize());
    std::vector<float> result(32*TN);Check(cudaMemcpy(result.data(),output.p,result.size()*4,cudaMemcpyDeviceToHost));
    for(float x:result)if(x!=512.0f)throw std::runtime_error("resident MMA result mismatch");
    std::vector<U64> ticks(count);Check(cudaMemcpy(ticks.data(),stamps.p,count*8,cudaMemcpyDeviceToHost));
    std::vector<double> values;for(int i=kDiscard;i<count;++i)values.push_back(double(ticks[i])/cycles_per_ns/64);
    Emit("matrix_k64_ns",key,std::move(values),"ns/k64","production_mma32_resident_operands_64_iterations_no_subtraction");
  });
}
