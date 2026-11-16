// Production pool Finish, with Apply and row-statistics explicitly excluded.
template<int M,int N,int Rows,int Parts,int Channels,int TW,int TH,int IW>
__global__ void PoolFinishStore(BF* output,float* partial,int row_base,int live_rows,
    int width,int stride,int count,U64* stamps) {
  extern __shared__ __align__(16) float tile[];
  using Base=Sm90ForwardEpilogueChain<M,N>;
  using Chain=Sm90ForwardPoolEpilogueChain<Rows,Parts,Channels,Base,TW,TH,IW>;
  Sm90ForwardEpilogueOperands p{};p.channel_partial=partial;p.row_base=row_base;
  if constexpr(TW>0) {p.tile_width=TW<IW?TW:IW;p.image_width=IW;}
  // p.row_stats is compile-time null: later row-stat implementation changes
  // are outside this primitive, rather than a silently timed extra branch.
  for(int sample=0;sample<count;++sample) {
    for(int i=threadIdx.x;i<M*N;i+=128)tile[i]=float((i%31)-15)*0.03125f;
    codegen::executor::ComputeSync();U64 begin=Cycles();
    Chain::Finish(tile,live_rows,0,width,p,output+std::size_t(row_base)*stride,stride);
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
  }
}
template<int M,int N,int Rows,int Parts,int Channels,int TW=0,int TH=0,int IW=0>
void PoolFinishCase(std::string id,std::string key,int row_base,int live_rows,
    int width,int stride,int batch) {
  constexpr int SharedBytes=4*M*N+16;
  Add(std::move(id),"pool_finish_store_ns",key,SharedBytes,[=](int count) {
    if(row_base<0 || live_rows<=0 || live_rows>M || width<=0 || width>N ||
       stride<width || batch<=0)throw std::runtime_error("invalid pool Finish geometry");
    std::size_t output_count=std::size_t(batch)*Rows*stride;
    std::size_t partial_count=std::size_t(batch)*Parts*Channels;
    Buffer<BF> output(output_count);Buffer<float> partial(partial_count);Buffer<U64> stamps(count);
    std::vector<BF> expected_output(output_count,BF(-8.0f)),got_output(output_count);
    std::vector<float> expected_partial(partial_count,-8.0f),got_partial(partial_count);
    std::vector<unsigned char> written(partial_count);
    Check(cudaMemcpy(output.p,expected_output.data(),output_count*sizeof(BF),cudaMemcpyHostToDevice));
    Check(cudaMemcpy(partial.p,expected_partial.data(),partial_count*4,cudaMemcpyHostToDevice));
    constexpr int live_tw=TW>0?(TW<IW?TW:IW):0;
    for(int row=0;row<live_rows;++row) {
      int pixel=row_base+(TW>0?(row/live_tw)*IW+row%live_tw:row),image=pixel/Rows;
      int part;
      if constexpr(TW>0) {
        int origin=row_base-image*Rows;
        part=(origin/IW/TH)*((IW+TW-1)/TW)+(origin%IW)/TW;
      }else part=row_base/M-(image*Rows)/M;
      if(image<0 || image>=batch || part<0 || part>=Parts)
        throw std::runtime_error("pool Finish representative exceeds allocated images/parts");
      for(int col=0;col<width;++col) {
        float value=float(((row*N+col)%31)-15)*0.03125f;
        auto index=(std::size_t(image)*Parts+part)*Channels+col;
        if(!written[index]) {expected_partial[index]=0;written[index]=1;}
        expected_partial[index]+=value;
        expected_output[std::size_t(pixel)*stride+col]=BF(value);
      }
    }
    auto kernel=PoolFinishStore<M,N,Rows,Parts,Channels,TW,TH,IW>;
    Shared(kernel,SharedBytes,128);
    kernel<<<1,128,SharedBytes>>>(output,partial,row_base,live_rows,width,stride,count,stamps);
    Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(got_output.data(),output.p,output_count*sizeof(BF),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(got_partial.data(),partial.p,partial_count*4,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<output_count;++i)
      if(got_output[i].raw()!=expected_output[i].raw())throw std::runtime_error("pool Finish output/untouched-region mismatch");
    for(std::size_t i=0;i<partial_count;++i)
      if(got_partial[i]!=expected_partial[i])throw std::runtime_error("pool Finish unique partial/untouched-region mismatch");
    Samples("pool_finish_store_ns",key,stamps,count,1,"production_empty_chain_Finish_pool_and_store_clock64;no_rowstat;actual_geometry");
  });
}
