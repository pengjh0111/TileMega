// This measures the fused body's actual DW plus its cached affine chain,
// rather than substituting standalone Depthwise timing or adding the chain twice.
template<int Live,int Stride>struct FusedDwProbe {
  using Dw=Sm90ForwardElementChain<Sm90ForwardEpilogueStep<Sm90ForwardOp::kScale,0>,
      Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias,1>,
      Sm90ForwardEpilogueStep<Sm90ForwardOp::kActivation,0,codegen::DmActivation::kRelu6>>;
  using Body=Sm90ForwardDwPw<8,8,64,64,Stride,2,Sm90ForwardEpilogueChain<64,64>,8,Dw>;
  static constexpr int filter_offset=(2*Body::kHaloElements+1023)/1024*1024;
  static constexpr int panel_offset=(filter_offset+2*Body::kFilterElements+1023)/1024*1024;
  static constexpr int bytes=panel_offset+128*64;
};
template<int Live,int Stride>
__global__ void FusedDwResident(BF const* input_halo,BF const* input_filter,float const* scale,float const* bias,BF* output,U64* stamps,int count) {
  using P=FusedDwProbe<Live,Stride>;using Body=typename P::Body;
  extern __shared__ __align__(1024) char storage[];
  auto* halo=reinterpret_cast<BF*>(storage);
  auto* weights=reinterpret_cast<BF*>(storage+P::filter_offset);
  auto* panel=reinterpret_cast<BF*>(storage+P::panel_offset);
  if(threadIdx.x<128) {
    for(int i=threadIdx.x;i<Body::kHaloElements;i+=128)halo[i]=input_halo[i];
    for(int i=threadIdx.x;i<Body::kFilterElements;i+=128)weights[i]=input_filter[i];
  }
  __syncthreads();
  if(threadIdx.x<128)for(int sample=0;sample<count;++sample) {
    float const* vectors[4]={scale,bias,nullptr,nullptr};
    codegen::executor::ComputeSync();U64 begin=Cycles();
    Body::Depthwise(halo,weights,panel,64,8,Live,0,vectors);
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
    for(int i=threadIdx.x;i<64*64;i+=128)output[i]=panel[i];
    codegen::executor::ComputeSync();
  }
}
template<int Live,int Stride>void FusedDwCase() {
  using P=FusedDwProbe<Live,Stride>;
  std::string key="tw=8;th=8;ct=64;channels="+std::to_string(Live)+
      ";stride="+std::to_string(Stride)+";pack=8;chain=scale+bias+relu6";
  Add("dwpw_dw_"+std::to_string(Live)+"_s"+std::to_string(Stride),"dwpw_depthwise_ns",key,P::bytes,[=](int count) {
    Buffer<BF> input_halo(P::Body::kHaloElements),input_filter(P::Body::kFilterElements);
    Fill(input_halo,0.25f);Fill(input_filter,0.5f);
    Buffer<float> scale(Live),bias(Live);Fill(scale,1.25f);Fill(bias,0.125f);
    Buffer<BF> output(64*64);Buffer<U64> stamps(count);
    Shared(FusedDwResident<Live,Stride>,P::bytes);
    FusedDwResident<Live,Stride><<<1,160,P::bytes>>>(input_halo,input_filter,scale,bias,output,stamps,count);Check(cudaDeviceSynchronize());
    std::vector<BF> result(64*64);Check(cudaMemcpy(result.data(),output.p,result.size()*2,cudaMemcpyDeviceToHost));
    using Body=typename P::Body;
    for(int row=0;row<64;++row)for(int c=0;c<64;++c) {
      // Host copy of the production SW128 byte mapping; exact synthetic result.
      int index=row*64+(((c/8)^(row%8))*8)+c%8;
      if(float(result[index])!=(c<Live?1.53125f:0.0f))throw std::runtime_error("fused DW result mismatch");
    }
    std::vector<U64> ticks(count);Check(cudaMemcpy(ticks.data(),stamps.p,count*8,cudaMemcpyDeviceToHost));
    std::vector<double> values;for(int i=kDiscard;i<count;++i)values.push_back(double(ticks[i])/cycles_per_ns);
    Emit("dwpw_depthwise_ns",key,std::move(values),"ns/task/channel_block","production_fused_Depthwise_resident_halo_filter_global_affine_included");
  });
}
