template<int TM,int Live>
__global__ void ScaPanel(BF const* input,BF const* scale,BF* output,int m_tile,
    int k_panel,int count,U64* stamps) {
  if(threadIdx.x>=128)return;
  Sm90DenseRoles::Consumer();
  constexpr int Rows=Live==32?65536:16384;
  using Shape=Sm90ForwardInputScale<Rows,Live,Live>;
  using Body=Sm90ForwardGemm<TM,32,64,2,1,Sm90ForwardAMode::kDense,
      Rows,32,64,32,Sm90ForwardEpilogueChain<TM,32>,Sm90ForwardNoConv,true,Shape>;
  extern __shared__ __align__(1024) unsigned char memory[];
  auto* panel=reinterpret_cast<BF*>(memory);
  Sm90ForwardGemmOperands p{};p.input_scale=scale;
  for(int sample=0;sample<count;++sample) {
    for(int i=threadIdx.x;i<TM*64;i+=128)panel[i]=input[i];
    codegen::executor::ComputeSync();U64 begin=Cycles();
    Body::ScalePanel(p,panel,m_tile,k_panel);
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
    codegen::executor::ComputeSync();
  }
  for(int i=threadIdx.x;i<TM*64;i+=128)output[i]=panel[i];
}
template<int TM,int Live>
void ScaCase(std::vector<unsigned short> input_words,std::vector<unsigned short> scale_words,
    std::vector<unsigned short> expected_words) {
  constexpr int SharedBytes=2*TM*64;
  auto key="tm="+std::to_string(TM)+";tk=64;live_k="+std::to_string(Live)+
      ";implementation=typed_cached_per_lane_bf16";
  Add("sca_panel_"+std::to_string(TM)+"_"+std::to_string(Live),"sca_panel_ns",key,
      SharedBytes,[=](int count) {
    if(input_words.size()!=TM*64 || expected_words.size()!=TM*64 || scale_words.size()!=Live)
      throw std::runtime_error("invalid independent CPU SCA oracle size");
    Buffer<BF> input(TM*64),scale(Live),output(TM*64);Buffer<U64> stamps(count);
    Check(cudaMemcpy(input.p,input_words.data(),input_words.size()*2,cudaMemcpyHostToDevice));
    Check(cudaMemcpy(scale.p,scale_words.data(),scale_words.size()*2,cudaMemcpyHostToDevice));
    auto kernel=ScaPanel<TM,Live>;Shared(kernel,SharedBytes,160);
    kernel<<<1,160,SharedBytes>>>(input,scale,output,0,0,count,stamps);
    Check(cudaDeviceSynchronize());std::vector<unsigned short> got(TM*64);
    Check(cudaMemcpy(got.data(),output.p,got.size()*2,cudaMemcpyDeviceToHost));
    if(got!=expected_words)throw std::runtime_error("production SCA ScalePanel differs from CPU PyTorch BF16 oracle");
    std::vector<U64> host(count);Check(cudaMemcpy(host.data(),stamps.p,count*8,cudaMemcpyDeviceToHost));
    std::vector<double> values;
    for(int sample=kDiscard;sample<count;++sample)values.push_back(double(host[sample])/cycles_per_ns);
    Emit("sca_panel_ns",key,std::move(values),"ns/k64/cta",
        "production_ScalePanel_cached_load_bf16_round_sync_fence_sync_clock64;runtime_panel_offsets;128consumer_32idle");
  });
}
