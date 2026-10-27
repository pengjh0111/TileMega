// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <tilemega/Target/TargetSpec.h>

namespace tilemega::solver {

// Fixed per-stage formula of prompt §10 N-1 item 8:
//   T = c_bar + L_l2 + max(FLOPs/(eta P_sm min(tasks, G)), bytes/BW) + t_ep
//       + (S_k > 1 ? t_la : 0)
// eta (the WGMMA shape efficiency) and P_sm come from the constants section;
// nothing here is fitted to end-to-end measurements.
struct Sm90ForwardPrediction {
  struct Stage {std::string body;double ns=0;long long tasks=0;json::Value terms;};
  std::vector<Stage> stages;
  double total_ns=0;
  json::Value ToJson() const {
    json::Array rows;
    for(auto const& s:stages)
      rows.emplace_back(json::Object{{"body",s.body},{"tasks",double(s.tasks)},
          {"stage_ns",s.ns},{"terms",s.terms}});
    return json::Object{{"forward_ns",total_ns},{"stages",std::move(rows)}};
  }
};

class Sm90ForwardCostModel {
 public:
  Sm90ForwardCostModel(TargetSpec target,Sm90NativeConstants constants)
      :target_(std::move(target)),constants_(std::move(constants)) {
    if(target_.arch_tag!="sm_90" || target_.res.num_sms<=0)
      throw std::invalid_argument("native forward cost model requires the H800 target");
  }
  int Grid(int residency) const {return target_.res.num_sms*residency;}

  // Tasks of one stage under its coordinate.
  static long long Tasks(Sm90ForwardShape const& shape,Sm90ForwardStage const& c) {
    switch(shape.body) {
      case Sm90ForwardBody::kGemm:
      case Sm90ForwardBody::kConv:
        return (shape.n+c.tile_n-1)/c.tile_n*((shape.m+63)/64)*c.split_k;
      case Sm90ForwardBody::kDepthwise:
        return ((shape.pixels+c.tile_w*c.tile_h-1)/(c.tile_w*c.tile_h))*
            ((shape.channels+c.channel_block-1)/c.channel_block);
      case Sm90ForwardBody::kDwPw:
      case Sm90ForwardBody::kInvertedResidual:
        return ((shape.pixels+c.tile_w*c.tile_h-1)/(c.tile_w*c.tile_h))*
            ((shape.out_channels+c.tile_n-1)/c.tile_n);
      case Sm90ForwardBody::kEncoderAttention:return shape.pixels;
      default:return shape.pixels>0?shape.pixels:1;
    }
  }

  double Stage(Sm90ForwardShape const& shape,Sm90ForwardStage const& c,int residency,
               json::Value* terms=nullptr) const {
    auto const& k=constants_;
    int grid=Grid(residency);
    long long tasks=Tasks(shape,c);
    double waves=std::ceil(double(tasks)/grid);
    double flops=2.0*shape.m*shape.n*shape.k;
    if(shape.body==Sm90ForwardBody::kDepthwise)
      flops=2.0*shape.pixels*shape.channels*shape.kernel*shape.kernel;
    double bytes=double(shape.weight_bytes+shape.activation_bytes+shape.output_bytes);
    // Compute: the busiest CTA's share at the measured WGMMA efficiency.
    double lanes=double(std::min<long long>(tasks,grid));
    // Depthwise and the small bodies run on CUDA cores, not WGMMA; they are
    // bandwidth-bound in the model and take the widest shape's efficiency.
    bool wgmma=shape.body==Sm90ForwardBody::kGemm || shape.body==Sm90ForwardBody::kConv ||
        shape.body==Sm90ForwardBody::kDwPw || shape.body==Sm90ForwardBody::kInvertedResidual;
    double eta=k.WgmmaEfficiency(wgmma?c.tile_n:256);
    double compute=lanes>0?flops/(eta*k.SmFlops()*lanes):0.0;
    double stream=bytes/k.dram_gbps;
    double epilogue=shape.epilogue_ops*k.Barrier(grid)*0.0;  // chain ops are register work
    double total=k.Barrier(grid)+k.l2_latency+std::max(compute,stream)+epilogue;
    if(c.split_k>1)total+=k.Barrier(grid);
    if(terms)*terms=json::Object{{"barrier",k.Barrier(grid)},{"first_byte",k.l2_latency},
        {"compute",compute},{"stream",stream},{"eta",eta},{"tasks",double(tasks)},{"waves",waves},
        {"split_k",double(c.split_k)}};
    return total;
  }

  Sm90ForwardPrediction Predict(std::vector<Sm90ForwardShape> const& shapes,
                                Sm90ForwardCoordinates const& c) const {
    if(shapes.size()!=c.stages.size())
      throw std::invalid_argument("forward coordinates and shapes disagree on the stage count");
    Sm90ForwardPrediction p;
    for(std::size_t i=0;i<shapes.size();++i) {
      Sm90ForwardPrediction::Stage stage{kSm90ForwardBodies[unsigned(shapes[i].body)],0,
          Tasks(shapes[i],c.stages[i]),json::Value()};
      stage.ns=Stage(shapes[i],c.stages[i],c.residency,&stage.terms);
      p.total_ns+=stage.ns;
      p.stages.push_back(std::move(stage));
    }
    return p;
  }
  Sm90NativeConstants const& Constants() const {return constants_;}

 private:
  TargetSpec target_;
  Sm90NativeConstants constants_;
};

// Rule-pruned space of prompt §10 N-1 item 8: TN in {32,64,128,256} capped by
// the padded N, S_k in {1,2,4,8,16} with K/S_k >= 128, spatial tiles in
// {8x8, 8x16, 16x16}, fusion where the plan offers it. The task count should
// land in [G/2, 2G]; when no candidate does, the closest is kept.
struct Sm90ForwardSearchSpace {
  std::vector<Sm90ForwardShape> shapes;
  TargetSpec target;
  int Grid(int residency) const {return target.res.num_sms*residency;}
  std::vector<Sm90ForwardStage> Candidates(std::size_t index,int residency) const;
};

struct Sm90ForwardCandidate {
  Sm90ForwardCoordinates coordinates;
  Sm90ForwardPrediction prediction;
  std::string origin;
};
struct Sm90ForwardSearchResult {
  std::vector<Sm90ForwardCandidate> ranked;
  long long evaluated=0;
  double elapsed_ms=0;
};
Sm90ForwardSearchResult SearchSm90Forward(Sm90ForwardSearchSpace const& space,
    Sm90ForwardCostModel const& cost,int top=4);

}  // namespace tilemega::solver
