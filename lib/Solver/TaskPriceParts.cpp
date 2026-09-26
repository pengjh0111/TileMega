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
  auto traffic=memory?*memory:DeriveTaskMemoryTraffic(input,theta,point,2,options_.fp32_partials && chunks>1 && traits.stages>0?4:2,domain);
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
  double serving_flops=0;
  double serving_body_bytes=-1;
  if(traits.stages<=0) {
    if(!input.scalar_flow)throw std::invalid_argument("scalar flow not supplied");
    auto [depth,barriers]=input.scalar_flow->MemoryDepthAndBarriers(traits.threads);
    double writes=traffic.global_write_bytes/2,bytes=traffic.global_read_bytes+traffic.global_write_bytes;
    double flops=(input.arithmetic.flops_per_output_element.Eval(theta)+input.scalar_flow->extra_flops_per_output)*writes;
    serving_flops=flops;
    double transc=input.arithmetic.transcendental_per_output_element.Eval(theta)*writes;
    double structural=depth*calib_->l2_latency_ns+barriers*calib_->syncthreads_ns+traffic.global_write_bytes/l2_bytes_per_ns_per_sm_;
    result.fixed_ns=(fit.samples>0?fit.scalar_fixed_ns:0)+structural;
    // A fused attention task has a scalar control flow but executes its QK/PV
    // arithmetic on tensor cores. Keep its vector loads and online exp2 in
    // the scalar resource vector, and put the declared MMA work on the tensor
    // core lane. The arithmetic declaration, not StageKind, selects the lane.
    double scalar_flops=input.arithmetic.flops_use_mma?0:flops;
    double scalar_ns=ScalarInstanceNs(bytes,traffic.global_write_bytes,
        scalar_flops,transc,o,0,input.arithmetic.smem_staged,depth,barriers)-structural;
    double mma_ns=input.arithmetic.flops_use_mma
        ? o*flops/tc_flops_per_ns_per_sm_:0;
    result.compute_ns=std::max(scalar_ns,mma_ns);
  } else {
    if(traits.stages<2 || traits.tile_k<=0)throw std::invalid_argument("regime-A collective needs at least two pipeline stages");
    double iters=input.collective_k_extent>0
        ? std::ceil(double(input.collective_k_extent)/traits.tile_k)
        : eval(input.work.nominal_task_reduce_extent)/traits.tile_k;
    if(!(iters>0))throw std::invalid_argument("invalid reduction iterations");
    double reads=eval(input.work.nominal_read_elements)/iters,writes=eval(input.work.nominal_write_elements);
    serving_flops=input.arithmetic.flops_per_output_element.Eval(theta)*writes;
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
  if(input.serving_attention) {
    auto const& a=*input.serving_attention;
    if(a.block_count<=0 || a.block_extent<=0 || a.kv_tile<=0 ||
       a.head_dim<=0 || a.queries<=0 || !point.Contains("q"))
      throw std::invalid_argument("invalid serving attention price coordinate");
    int block=int(point.At("q")%a.block_count);
    int active=std::clamp(a.total-block*a.block_extent,0,a.block_extent);
    if(a.prefill && a.query_extent>0 && a.query_heads>0) {
      int query_blocks=(a.query_extent+a.queries-1)/a.queries;
      int qb=int(point.At("q")/a.block_count)%query_blocks;
      int last_query=std::min((qb+1)*a.queries,a.query_extent)-1;
      active=std::min(active,last_query/a.query_heads+1);
    }
    if(active==0) {
      // The TaskBody returns before loading any operand for an empty KV block.
      result.fixed_ns=fit.scalar_fixed_ns;
      result.compute_ns=0;
      result.dram_bytes=0;
      result.no_producer_dram_bytes=0;
    } else {
      // The fitted serving body uses one task's actual KV block: 4D bytes
      // per key/value position and one Q/K/position-table setup.  The generic
      // semantic work also counts replicated tensor-core lanes and is not in
      // the same units as this microbenchmark's flop coefficient.  Feeding
      // that expanded count to the fit inflated one Qwen decode block from
      // ~27 us to 6.7 ms.  Price the executed block in the fit's units.
      if(a.prefill) {
        // K/V rows are shared by this task's query block; each query row is
        // loaded once.  The calibration's full-S query sample is reduced to
        // the actual R_q rows selected by the symbolic task geometry.
        serving_body_bytes=4.0*a.head_dim*active+
            2.0*a.queries*a.head_dim;
        serving_flops=4.0*a.queries*a.head_dim*active;
      } else {
        serving_body_bytes=4.0*a.head_dim*active+
            2.0*a.queries*a.head_dim+4.0*a.head_dim;
        serving_flops=4.0*a.queries*a.head_dim*(active+1);
      }
      double padded_flops=4.0*((a.queries+15)/16*16)*a.head_dim*
          ((active+a.kv_tile-1)/a.kv_tile*a.kv_tile);
      double mma=o*padded_flops/tc_flops_per_ns_per_sm_;
      double exp2=o*a.queries*active/sfu_ops_per_ns_per_sm_;
      result.compute_ns=std::max(mma,exp2);
    }
  }
  if(!input.serving_body_kind.empty() &&
     !(input.serving_attention && result.compute_ns==0 && result.dram_bytes==0)) {
    auto calibrated=fit.serving.find(input.serving_body_kind);
    if(calibrated!=fit.serving.end()) {
      auto const& body=calibrated->second;
      double bytes=serving_body_bytes>=0?serving_body_bytes:
          traffic.global_read_bytes+traffic.global_write_bytes;
      result.fixed_ns=body.fixed_ns;
      result.compute_ns=std::max(result.compute_ns,
          body.byte_ns*bytes+body.flop_ns*serving_flops);
    }
  }
  result.dram_rate_cap=std::min(l2_bytes_per_ns_per_sm_,result.compute_ns>0?result.dram_bytes/result.compute_ns:l2_bytes_per_ns_per_sm_);
  for(double v:{result.fixed_ns,result.compute_ns,result.dram_bytes,result.dram_rate_cap})
    if(v<0 || !std::isfinite(v))throw std::runtime_error("invalid task price component");
  return result;
}
} // namespace tilemega::solver
