// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace tilemega::solver {
double IsolatedNs(TaskPriceParts const& p,double fair_rate) {
  if(!(fair_rate>0) || !std::isfinite(fair_rate))throw std::invalid_argument("invalid fair DRAM rate");
  return p.fixed_ns+std::max(p.compute_ns,p.dram_bytes/fair_rate);
}
TaskPriceParts CostModel::PriceParts(DerivedTaskInput const& input,BackendTraits const& traits,
    Residency residency,ModelDescription const& model,int chunks,
    analysis::ParamBinding const& point,double o,TaskMemoryTraffic const* memory) const {
  if(!options_.regime_a || dtype_!=ScalarType::kBF16)
    throw std::invalid_argument("PriceParts requires the BF16 regime-A path");
  if(!(o>=1 && o<=residency.ctas_per_sm))throw std::invalid_argument("invalid price occupancy");
  auto theta=model.MetricBindings();auto eval=[&](auto const& q){return double(q.BindCoordinates(point).Eval(theta));};
  auto domain=traits.stages<=0 || options_.physical_traffic?analysis::AccessDomain::kPhysicalTensor:analysis::AccessDomain::kNominalTile;
  auto traffic=DeriveTaskMemoryTraffic(input,theta,point,2,options_.fp32_partials && chunks>1 && traits.stages>0?4:2,domain);
  if(memory)traffic=*memory;
  double stream=input.no_producer_read_bytes?input.stream_bytes:model.LiveFootprintBytes();
  double df_np=1-CacheHitProbability(model.LiveFootprintBytes()),df_p=df_np;
  if(stream>calib_->l2_knee_bytes && !options_.sdcm_above_knee) {
    if(!input.no_producer_read_bytes)throw std::invalid_argument("streaming task lacks element-level provenance");
    CacheServiceCurve curve(calib_->l2_curve_bytes,calib_->l2_curve_gbps);
    df_np=1-curve.HitFraction(stream,calib_->l2_gbps,calib_->dram_gbps);
    df_p=input.produced_live_bytes<=calib_->l2_knee_bytes?0:
        1-curve.HitFraction(input.produced_live_bytes,calib_->l2_gbps,calib_->dram_gbps);
  }
  TaskPriceParts result;
  if(input.no_producer_read_bytes && options_.physical_traffic) {
    result.no_producer_dram_bytes=traffic.no_producer_read_bytes*df_np;
    result.dram_bytes=result.no_producer_dram_bytes+traffic.produced_read_bytes*df_p+traffic.external_write_bytes;
  } else result.dram_bytes=traffic.global_read_bytes*df_np;
  double read_dram=input.no_producer_read_bytes?traffic.no_producer_read_bytes*df_np+traffic.produced_read_bytes*df_p:traffic.global_read_bytes*df_np;
  auto const& fit=calib_->task_body;
  if(traits.stages<=0) {
    if(!input.scalar_flow)throw std::invalid_argument("scalar flow not supplied");
    auto [depth,barriers]=input.scalar_flow->MemoryDepthAndBarriers(traits.threads);
    double writes=traffic.global_write_bytes/2,bytes=traffic.global_read_bytes+traffic.global_write_bytes;
    double flops=(input.arithmetic.flops_per_output_element.Eval(theta)+input.scalar_flow->extra_flops_per_output)*writes;
    double transc=input.arithmetic.transcendental_per_output_element.Eval(theta)*writes;
    double structural=depth*calib_->l2_latency_ns+barriers*calib_->syncthreads_ns+traffic.global_write_bytes/l2_bytes_per_ns_per_sm_;
    result.fixed_ns=(fit.samples>0?fit.scalar_fixed_ns:0)+structural;
    result.compute_ns=ScalarInstanceNs(bytes,traffic.global_write_bytes,flops,transc,o,0,input.arithmetic.smem_staged,depth,barriers)-structural;
  } else {
    if(traits.stages<2 || traits.tile_k<=0)throw std::invalid_argument("regime-A collective needs at least two pipeline stages");
    double iters=eval(input.work.nominal_task_reduce_extent)/traits.tile_k;
    if(!(iters>0))throw std::invalid_argument("invalid reduction iterations");
    double reads=eval(input.work.nominal_read_elements)/iters,writes=eval(input.work.nominal_write_elements);
    double nominal_bytes=2*reads,bytes=traffic.global_read_bytes/iters;
    ResourceVector u;u.smem=o*nominal_bytes*fit_.lds_ns;
    if(options_.resource_lanes) {
      double flops=input.arithmetic.flops_per_output_element.Eval(theta)*writes/iters;
      if(input.arithmetic.flops_use_mma)u.tensor_core=o*flops/tc_flops_per_ns_per_sm_;
      else u.cuda_core=o*flops/cuda_flops_per_ns_per_sm_;
      u.sfu=o*input.arithmetic.transcendental_per_output_element.Eval(theta)*writes/iters/sfu_ops_per_ns_per_sm_;
      u.l2=o*bytes/l2_bytes_per_ns_per_sm_;
    }
    for(int i=0;i<ResourceVector::kLaneCount;++i) {
      auto lane=static_cast<ResourceVector::Lane>(i);
      if(lanes_[lane]!=LaneStatus::kLive || options_.disabled_lanes[lane])u[lane]=0;
    }
    double body=0,wait=0;
    if(fit.samples>0) {
      auto const& fixed=options_.physical_fixed && !fit.fixed_physical.empty()?fit.fixed_physical:fit.fixed;
      double output=options_.physical_fixed && !fit.fixed_physical.empty()?eval(input.work.write_elements):writes;
      result.fixed_ns=fixed[0]+fixed[1]*output+fixed[2]*reads*traits.stages+fit.loop_fixed[0]+fit.loop_fixed[1]*writes;
      body=fit.loop_body[0]+fit.loop_body[1]*writes*traits.tile_k;
      wait=fit.loop_wait[0]+fit.loop_wait[1]*reads;
    } else {
      result.fixed_ns=fit_.setup_ns+fit_.setup_per_output_ns*writes+traits.stages*bytes/l2_bytes_per_ns_per_sm_+calib_->l2_latency_ns+traffic.global_write_bytes/l2_bytes_per_ns_per_sm_;
    }
    double iteration=std::max(u.Bottleneck(),body)+wait;
    if(options_.stage_latency) {
      if(!(fit.stage_rate_bytes_per_ns>0))throw std::invalid_argument("regime-A stages require a calibrated transfer rate");
      double h=traffic.global_read_bytes>0?1-read_dram/traffic.global_read_bytes:1;
      double latency=h*calib_->l2_latency_ns+(1-h)*calib_->dram_latency_ns;
      iteration=std::max({u.Bottleneck(),body,(fit.latency_scale*latency+o*bytes/fit.stage_rate_bytes_per_ns)/(traits.stages-1)});
    }
    result.compute_ns=(options_.pipeline_envelope?std::max(iters-(traits.stages-1),0.):iters)*iteration;
  }
  result.dram_rate_cap=std::min(l2_bytes_per_ns_per_sm_,result.compute_ns>0?result.dram_bytes/result.compute_ns:l2_bytes_per_ns_per_sm_);
  for(double v:{result.fixed_ns,result.compute_ns,result.dram_bytes,result.dram_rate_cap})
    if(v<0 || !std::isfinite(v))throw std::runtime_error("invalid task price component");
  return result;
}
} // namespace tilemega::solver
