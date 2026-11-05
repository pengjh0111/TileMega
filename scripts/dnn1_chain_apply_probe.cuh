// Exactly one composite Apply; initialization and FP32 output drainage are outside.
template<class Chain,int M,int N,int Rows,int Columns,int Mask,int Stride,int TileW,int ImageW>
__global__ void ChainApplyResident(Sm90ForwardEpilogueOperands p,float* output,U64* stamps,int count) {
  static_assert(Chain::kRegisterPointwise);
  extern __shared__ __align__(16) float tile[];
  for(int sample=0;sample<count;++sample) {
    for(int i=threadIdx.x;i<M*N;i+=128)tile[i]=float((i%31)-15)*0.03125f;
    codegen::executor::ComputeSync();U64 begin=Cycles();
    Chain::ApplyRegister(tile,Rows,0,Columns,p);
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
    for(int i=threadIdx.x;i<M*N;i+=128)output[i]=tile[i];
    codegen::executor::ComputeSync();
  }
}
template<class Chain,int M,int N,int Rows,int Columns,int Mask,int Stride,int TileW,int ImageW,class Oracle>
void ChainApplyCase(int id,char const* key,Oracle expected) {
  constexpr int SharedBytes=4*M*N+16;
  Add("chain_apply_"+std::to_string(id),"epilogue_chain_apply_ns",key,SharedBytes,[=](int count) {
    constexpr int residual_elements=Stride?((TileW?(Rows-1)/TileW*ImageW+(Rows-1)%TileW:Rows-1)*Stride+Columns):1;
    Buffer<float> vectors(4*N),output(M*N);Buffer<BF> residual(residual_elements);Buffer<U64> stamps(count);
    Fill(vectors,0.5f);Fill(residual,0.25f);
    Sm90ForwardEpilogueOperands p{};
    for(int v=0;v<4;++v)p.vectors[v]=(Mask&(1<<v))?vectors.p+v*N:nullptr;
    p.residual=residual;p.residual_stride=Stride;p.tile_width=TileW;p.image_width=ImageW;
    auto kernel=ChainApplyResident<Chain,M,N,Rows,Columns,Mask,Stride,TileW,ImageW>;
    Shared(kernel,SharedBytes,128);kernel<<<1,128,SharedBytes>>>(p,output,stamps,count);Check(cudaDeviceSynchronize());
    std::vector<float> actual(M*N);Check(cudaMemcpy(actual.data(),output.p,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int row=0;row<Rows;++row)for(int col=0;col<Columns;++col) {
      float value=float(((row*N+col)%31)-15)*0.03125f;
      if(actual[row*N+col]!=expected(value))throw std::runtime_error("composite Apply independent CPU oracle mismatch");
    }
    Samples("epilogue_chain_apply_ns",key,stamps,count,1,"production_ApplyRegister_entire_ordered_chain_clock64_no_constituent_sum");
  });
}
