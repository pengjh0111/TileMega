// Production Finish with local statistics, cross-tile reduction, and final store.
template<int M,int N,int Tiles>
__global__ void RowstatFinishStore(Sm90ForwardEpilogueOperands p,BF* output,
    int live_rows,int stride,U64* stamps) {
  extern __shared__ __align__(16) float tile[];
  p.row_tile=int(blockIdx.x);p.row_tiles=Tiles;
  for(int i=threadIdx.x;i<M*N;i+=128)
    tile[i]=float(((i+int(blockIdx.x)*7)%31)-15)*0.03125f;
  codegen::executor::ComputeSync();U64 begin=Cycles();
  Sm90ForwardEpilogueChain<M,N>::Finish(tile,live_rows,int(blockIdx.x)*N,N,p,
      output,stride);
  codegen::executor::ComputeSync();
  if(threadIdx.x==0)stamps[blockIdx.x]=Cycles()-begin;
}
template<int M,int N,int Tiles>
void RowstatFinishCase(std::string id,std::string key,int live_rows,int stride) {
  constexpr int SharedBytes=4*M*N+16;
  Add(std::move(id),"rowstat_finish_store_ns",key,SharedBytes,[=](int count) {
    if(live_rows<=0 || live_rows>M || stride<N*Tiles)
      throw std::runtime_error("invalid complete rowstat Finish geometry");
    Buffer<float> partial(2*live_rows*Tiles),stats(2*live_rows);
    Buffer<U64> counters(M),stamps(Tiles);Buffer<BF> output(live_rows*stride);
    std::vector<float> expected_partial(2*live_rows*Tiles,-8.0f),expected_stats(2*live_rows),
      got_partial(2*live_rows*Tiles),got_stats(2*live_rows);
    std::vector<BF> expected_output(live_rows*stride,BF(-8.0f)),got_output(live_rows*stride);
    for(int row=0;row<live_rows;++row) {
      float total=0,total_squares=0;
      for(int t=0;t<Tiles;++t) {
        float sum=0,squares=0;
        for(int col=0;col<N;++col) {
          float value=float(((row*N+col+t*7)%31)-15)*0.03125f;
          sum+=value;squares+=value*value;
          expected_output[std::size_t(row)*stride+t*N+col]=BF(value);
        }
        if constexpr(Tiles>1) {
          auto index=(std::size_t(t)*live_rows+row)*2;
          expected_partial[index]=sum;expected_partial[index+1]=squares;
        }
        total+=sum;total_squares+=squares;
      }
      expected_stats[2*row]=total;expected_stats[2*row+1]=total_squares;
    }
    std::vector<float> initial_partial(expected_partial.size(),-8.0f);
    std::vector<BF> initial_output(expected_output.size(),BF(-8.0f));
    Check(cudaMemcpy(partial.p,initial_partial.data(),initial_partial.size()*4,cudaMemcpyHostToDevice));
    Check(cudaMemcpy(output.p,initial_output.data(),initial_output.size()*sizeof(BF),cudaMemcpyHostToDevice));
    Sm90ForwardEpilogueOperands p{};
    p.row_stats=stats;p.row_partials=partial;p.row_counters=counters;p.row_tiles=Tiles;
    auto kernel=RowstatFinishStore<M,N,Tiles>;Shared(kernel,SharedBytes,128);
    std::vector<double> values;std::vector<U64> host(Tiles),zeros(M);
    for(int sample=0;sample<count;++sample) {
      Check(cudaMemset(counters.p,0,M*8));
      kernel<<<Tiles,128,SharedBytes>>>(p,output,live_rows,stride,stamps);
      Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(host.data(),stamps.p,Tiles*8,cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(zeros.data(),counters.p,M*8,cudaMemcpyDeviceToHost));
      if(std::any_of(zeros.begin(),zeros.end(),[](U64 x){return x!=0;}))
        throw std::runtime_error("complete rowstat Finish counter did not reset");
      if(sample>=kDiscard)values.push_back(double(*std::max_element(host.begin(),host.end()))/cycles_per_ns);
    }
    Check(cudaMemcpy(got_output.data(),output.p,got_output.size()*sizeof(BF),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(got_partial.data(),partial.p,got_partial.size()*4,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(got_stats.data(),stats.p,got_stats.size()*4,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<got_output.size();++i)
      if(got_output[i].raw()!=expected_output[i].raw())
        throw std::runtime_error("complete rowstat Finish output or padding mismatch");
    if(got_partial!=expected_partial)throw std::runtime_error("complete rowstat Finish tile-major partial mismatch");
    if(got_stats!=expected_stats)throw std::runtime_error("complete rowstat Finish local/tile-order sum mismatch");
    Emit("rowstat_finish_store_ns",key,std::move(values),"ns",
      "production_empty_chain_Finish_local_statistics_cross_tile_and_store_max_CTA_clock64");
  });
}
