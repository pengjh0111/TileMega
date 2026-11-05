template<int W,int H,int C,int Live=C,int Stride=1>void DepthwiseCase() {
  static_assert(Live==C && Stride==1);
  using Chain=Sm90ForwardEpilogueChain<W*H,C,kDenseWrite,1,0,0,0,
      Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias,0>,
      Sm90ForwardEpilogueStep<Sm90ForwardOp::kGatePair,0,codegen::DmActivation::kRelu,
                            codegen::DmGatePair::kSimpleGate,16>>;
  using Body=Sm90ForwardDepthwise<W,H,C,Stride,2,Chain>;
  auto key="tw="+std::to_string(W)+";th="+std::to_string(H)+";channels="+std::to_string(Live)+
      ";ct="+std::to_string(C)+";stride="+std::to_string(Stride)+";pack=8;gate=simple_gate;bias=1";
  Add("gated_dw_"+std::to_string(W)+"_"+std::to_string(H)+"_"+std::to_string(C)+"_"+
      std::to_string(Live)+"_s"+std::to_string(Stride),"dw_tap_ns",key,
      Body::kSharedBytes,[key](int count) {
    // 32 spatial tiles per CTA; real TMA halo and bulk filters feed Consume.
    constexpr int Width=W*32,Height=H,InputW=Width*Stride,InputH=Height*Stride;
    Buffer<BF> input(InputW*InputH*C),weight(C*9*8),output(Width*Height*Live/2);
    Buffer<float> bias(C);Fill(bias,0.25f);
    Fill(input,0.25f);Fill(weight,0.5f);CUtensorMap map{};
    codegen::executor::EncodeTiledMap<4>(map,input.p,
      {{C,InputW,InputH,1},{2*C,2*C*InputW,2*C*InputW*InputH},{C/2,Body::kHaloW,Body::kHaloH,1}},false);
    Sm90ForwardDepthwiseOperands p{};p.weight=weight;p.output=output;p.width=InputW;p.height=InputH;
    p.channels=Live;p.stride=Stride;p.output_width=Width;p.output_height=Height;p.padded_channels=C;p.output_stride=Live/2;p.epilogue.vectors[0]=bias;
    auto kernel=TiledBody<Body,Sm90ForwardDepthwiseOperands>;Shared(kernel,Body::kSharedBytes);
    std::vector<double> values,epilogue;
    for(int sample=0;sample<count;++sample) {
      ResetTrace();kernel<<<1,160,Body::kSharedBytes>>>(map,p);Check(cudaDeviceSynchronize());
      auto span=ReadTrace(1)[0].segments[unsigned(Segment::kDepthwiseCompute)];
      if(span.count!=32)throw std::runtime_error("depthwise trace task count differs");
      auto ep=ReadTrace(1)[0].segments[unsigned(Segment::kEpilogue)];
      if(ep.count!=32)throw std::runtime_error("gated depthwise epilogue count differs");
      if(sample>=kDiscard) {
        values.push_back(double(span.total_ns)/cycles_per_ns/span.count/(W*H*Live*9));
        epilogue.push_back(double(ep.total_ns)/cycles_per_ns/ep.count);
      }
    }
    std::vector<BF> actual(output.count);Check(cudaMemcpy(actual.data(),output.p,actual.size()*2,cudaMemcpyDeviceToHost));
    for(int y=0;y<Height;++y)for(int x=0;x<Width;++x)for(int c=0;c<Live/2;++c) {
      int valid_y=3-(y==0)-(y==Height-1),valid_x=3-(x==0)-(x==Width-1);
      float sum=valid_y*valid_x*0.125f+0.25f;
      BF expected(sum*sum);
      if(float(actual[(y*Width+x)*(Live/2)+c])!=float(expected))
        throw std::runtime_error("gated depthwise actual output check failed");
    }
    Emit("dw_tap_ns",key,std::move(values),"ns/element/tap","production_Run_compute_clock64_spans");
    Emit("gated_dw_epilogue_diag_ns",key,std::move(epilogue),"ns",
        "production_bias_simplegate_store_only_no_pool_side_output");
  });
}
