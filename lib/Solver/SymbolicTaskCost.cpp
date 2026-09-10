// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <gmpxx.h>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifndef TILEMEGA_SYMBOLIC_TASK_COST
#define TILEMEGA_SYMBOLIC_TASK_COST 1
#endif

namespace tilemega::solver {
namespace {
using Polynomial=analysis::QuasiPolynomial;
Polynomial Expression(std::string const& parameter,long begin,long end,
                      std::array<std::string,3> const& coefficients) {
  return Polynomial::FromIslText("["+parameter+"] -> { ("+coefficients[0]+")+("+
      coefficients[1]+")*"+parameter+"+("+coefficients[2]+")*"+parameter+"^2 : "+
      std::to_string(begin)+"<="+parameter+"<="+std::to_string(end)+" }");
}
mpq_class Exact(double value) {
  if (!std::isfinite(value)) throw std::invalid_argument("nonfinite symbolic price coefficient");
  return mpq_class(value);
}
analysis::ParamBinding FixedExceptSeq(ModelDescription const& model,std::string const& parameter) {
  auto known=model.metric_bindings;
  for (std::string const& name:{std::string("S"),parameter,std::string("L_s")}) known.values.erase(name);
  known.Bind("past",model.dims.past).Bind("P",model.dims.past);
  if (!model.past_metric_parameter.empty()) known.Bind(model.past_metric_parameter,model.dims.past);
  for (auto const& [alias,canonical]:model.metric_aliases) {
    if (canonical==parameter) known.values.erase(alias);
    else if (known.Contains(canonical)) known.Bind(alias,known.At(canonical));
  }
  return known;
}
std::array<std::string,2> Footprint(ModelDescription const& model,int element_bytes) {
  mpq_class base=0,slope=0;
  for (auto const& gemm:model.gemms) {
    base+=mpq_class(element_bytes)*gemm.n*gemm.k;
    slope+=mpq_class(element_bytes)*(gemm.n+gemm.k);
  }
  for (auto const& stage:model.stages) if (stage.kind!=StageKind::kGemm) {
    mpq_class amount=mpq_class(element_bytes)*std::max(stage.extent,1)*std::max(stage.width,1);
    base+=amount*model.dims.past; slope+=amount;
  }
  return {base.get_str(),slope.get_str()};
}
Polynomial Envelope(std::vector<Polynomial> const& values,std::string const& parameter,long first,long last,bool maximum) {
  std::vector<Polynomial> pieces;
  for (auto const& piece:QuadraticEnvelope(values,parameter,first,last,maximum))
    pieces.push_back(Expression(parameter,piece.begin,piece.end,piece.coefficients));
  return Polynomial::Sum(pieces);
}
}

Polynomial CostModel::SymbolicStageNs(ModelDescription const& model,int stage_id,
    GemmConfig const& config,Residency residency,std::string const& parameter,long begin,long end) const {
  analysis::IslReferenceAudit audit(__func__);
  auto const& stage=model.stages.at(stage_id);
  auto selected=model.task_semantics.end();
  for (auto it=model.task_semantics.begin();it!=model.task_semantics.end();++it)
    if (it->stage==stage_id && (stage.kind!=StageKind::kGemm || it->op.kind==analysis::OperatorKind::kMatmul)) {
      if (selected!=model.task_semantics.end()) throw std::invalid_argument("ambiguous symbolic stage semantics");
      selected=it;
    }
  if (selected==model.task_semantics.end()) throw std::invalid_argument("symbolic stage lacks CG semantics");
  auto graph=InstantiateModelTasks(model,std::vector<GemmConfig>(model.gemms.size(),config));
  bool collective=stage.kind==StageKind::kGemm;
  auto input=DeriveModelTaskInput(model,*selected,graph,collective ? &config : nullptr);
  BackendTraits traits;
  if (collective) {
    traits=dtype_==ScalarType::kBF16 ? TensorBF16Traits(config.tile_m,config.tile_n,config.tile_k,config.stages)
                                  : SimtF32Traits(config.tile_m,config.tile_n,config.tile_k,config.stages);
    return SymbolicCollectiveNs(input,traits,residency,model,Chunks(model.gemms.at(stage.gemm),config),parameter,begin,end);
  }
  traits.threads=dtype_==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
  return SymbolicScalarNs(input,traits,residency,model,parameter,begin,end);
}

Polynomial CostModel::SymbolicCombineNs(GemmOp const& gemm,int chunks,
    std::string const& parameter,long begin,long end) const {
  analysis::IslReferenceAudit audit(__func__);
  if (chunks<=1) return Polynomial::Constant(0);
  if (options_.cache_model && !options_.measured_cache_curve)
    throw std::invalid_argument("symbolic combine requires measured cache curve");
  auto elements=Expression(parameter,begin,end,{"0",std::to_string(gemm.n),"0"});
  if (!(options_.fp32_partials && dtype_==ScalarType::kBF16)) {
    auto b=calib_->streamk.empty() ? mpq_class(0) : Exact(calib_->streamk.front().b_ns);
    auto d=calib_->streamk.empty() ? mpq_class(0) : Exact(calib_->streamk.front().d_ns);
    return elements.ScaleRational(mpq_class(b+d*(chunks-1)).get_str()).Add(
        Expression(parameter,begin,end,{Exact(calib_->combine_fixed_ns).get_str(),"0","0"}));
  }
  std::vector<Polynomial::PolynomialInterval> misses{{begin,end,{"1","0","0"}}};
  if (options_.cache_model) {
    if (!cache_service_curve_) throw std::invalid_argument("symbolic combine cache: not_calibrated");
    misses=cache_service_curve_->MissIntervals({"0",mpq_class(mpq_class(4)*chunks*gemm.n).get_str()},
        begin,end,calib_->l2_gbps,calib_->dram_gbps);
  }
  mpq_class fixed,base,l2,dram,extra=0;
  if (options_.measured_partial_combine) {
    auto const& fit=calib_->fp32_partial_combine;
    if (fit.reason!="measured" || !fit.fixed_ns || !fit.base_ns || !fit.d_l2_ns || !fit.d_dram_ns)
      throw std::invalid_argument("symbolic FP32 partial combine: "+fit.reason);
    fixed=Exact(*fit.fixed_ns); base=Exact(*fit.base_ns); l2=Exact(*fit.d_l2_ns); dram=Exact(*fit.d_dram_ns);
  } else {
    fixed=Exact(calib_->combine_fixed_ns);
    base=calib_->streamk.empty() ? mpq_class(0) : Exact(calib_->streamk.front().b_ns);
    l2=calib_->streamk.empty() ? mpq_class(0) : Exact(calib_->streamk.front().d_ns);
    dram=Exact(calib_->combine_d_dram_ns); extra=2*chunks;
  }
  std::vector<Polynomial> pieces;
  for (auto const& region:misses) {
    auto miss=Expression(parameter,region.begin,region.end,region.coefficients);
    auto constant=Expression(parameter,region.begin,region.end,
        {mpq_class(base+l2*(chunks-1)+extra/Exact(calib_->l2_gbps)).get_str(),"0","0"});
    auto per_element=constant.Add(miss.ScaleRational(mpq_class((dram-l2)*(chunks-1)+
        extra*(1/Exact(calib_->dram_gbps)-1/Exact(calib_->l2_gbps))).get_str()));
    pieces.push_back(per_element.Multiply(elements).Add(Expression(parameter,region.begin,region.end,{fixed.get_str(),"0","0"})));
  }
  auto result=Polynomial::Sum(pieces);
  try { result.QuadraticIntervals(parameter,begin,end); }
  catch (std::invalid_argument const& error) {
    throw std::invalid_argument(std::string(error.what())+"; combine="+result.ToString());
  }
  return result;
}

Polynomial CostModel::SymbolicInterfaceEdgeNs(ModelCouplingMetrics const& edge,
    ModelDescription const& model,std::string const& parameter,long begin,long end) const {
  analysis::IslReferenceAudit audit(__func__);
  if (options_.cache_model && !options_.measured_cache_curve)
    throw std::invalid_argument("symbolic interface requires measured cache curve");
  auto known=FixedExceptSeq(model,parameter);
  auto repeated=edge.wait.SumDomain().SubstituteParams(known).Add(
      edge.relation.Reverse().ImageCard().SubstituteParams(known).Scale(-1));
  auto bytes=repeated.Multiply(edge.volume.SubstituteParams(known)).Scale(dtype_==ScalarType::kBF16 ? 2 : 4);
  std::vector<Polynomial::PolynomialInterval> misses{{begin,end,{"1","0","0"}}};
  if (options_.cache_model) {
    if (!cache_service_curve_) throw std::invalid_argument("symbolic interface cache: not_calibrated");
    misses=cache_service_curve_->MissIntervals(Footprint(model,dtype_==ScalarType::kBF16 ? 2 : 4),begin,end,calib_->l2_gbps,calib_->dram_gbps);
  }
  std::vector<Polynomial> pieces;
  for (auto const& region:misses) {
    auto rate=Expression(parameter,region.begin,region.end,{mpq_class(1/Exact(calib_->l2_gbps)).get_str(),"0","0"})
        .Add(Expression(parameter,region.begin,region.end,region.coefficients).ScaleRational(
          mpq_class(1/Exact(calib_->dram_gbps)-1/Exact(calib_->l2_gbps)).get_str()));
    pieces.push_back(bytes.Multiply(rate));
  }
  auto result=Polynomial::Sum(pieces);
  try { result.QuadraticIntervals(parameter,begin,end); }
  catch (std::invalid_argument const& error) {
    throw std::invalid_argument(std::string(error.what())+"; CG interface "+std::to_string(edge.producer)+"->"+
        std::to_string(edge.consumer)+"; repeated="+repeated.ToString()+"; volume="+
        edge.volume.SubstituteParams(known).ToString()+"; price="+result.ToString());
  }
  return result;
}

Polynomial CostModel::SymbolicCollectiveNs(DerivedTaskInput const& input,
    BackendTraits const& traits,Residency residency,ModelDescription const& model,
    int chunks,std::string const& parameter,long begin,long end) const {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_SYMBOLIC_TASK_COST
  throw std::runtime_error("symbolic task pricing disabled");
#endif
  if (parameter.empty() || parameter!=model.seq_metric_parameter ||
      !model.dims.past_parameter.empty() || model.dims.past<0 || begin<=0 || begin>end ||
      end==std::numeric_limits<long>::max() || traits.stages<=0 || traits.tile_k<=0 ||
      residency.ctas_per_sm<=0 || chunks<=0 || model.attention_plan)
    throw std::invalid_argument("symbolic collective requires one CG seq role, fixed past and complete resources");
  if (options_.cache_model && !options_.measured_cache_curve)
    throw std::invalid_argument("symbolic collective requires measured cache curve, not Gaussian CDF");
  auto known=FixedExceptSeq(model,parameter);
  analysis::ParamBinding coordinate;
  for (auto const& name:input.cost_coordinates) coordinate.Bind(name,0);
  auto constant=[&](Polynomial const& quantity) {
    auto reduced=quantity.BindCoordinates(coordinate).SumDomain().SubstituteParams(known);
    if (reduced.IsZero()) return mpq_class(0);
    auto pieces=reduced.QuadraticIntervals(parameter,begin,end);
    if (pieces.empty()) throw std::invalid_argument("empty symbolic collective work domain");
    mpq_class value(pieces.front().coefficients[0]);
    long next=begin;
    std::sort(pieces.begin(),pieces.end(),[](auto const& a,auto const& b) { return a.begin<b.begin; });
    for (auto const& piece:pieces) {
      if (piece.begin!=next || mpq_class(piece.coefficients[0])!=value ||
          mpq_class(piece.coefficients[1])!=0 || mpq_class(piece.coefficients[2])!=0)
        throw std::invalid_argument("collective nominal per-task work is not constant in seq");
      next=piece.end+1;
    }
    if (next!=end+1) throw std::invalid_argument("symbolic collective work has domain gaps");
    return value;
  };
  mpq_class iters=constant(input.work.nominal_task_reduce_extent)/traits.tile_k;
  if (iters<=0 || iters.get_den()!=1) throw std::invalid_argument("invalid symbolic collective iteration count");
  mpq_class reads=constant(input.work.nominal_read_elements)/iters;
  mpq_class writes=constant(input.work.nominal_write_elements);
  mpq_class element_bytes=dtype_==ScalarType::kBF16 ? 2 : 4;
  mpq_class bytes=element_bytes*reads;
  auto arithmetic=[&](analysis::ArithmeticRatio const& work)->mpq_class {
    if (work.denominator<=0) throw std::invalid_argument("invalid arithmetic ratio denominator");
    return constant(work.numerator)/work.denominator;
  };
  mpq_class flops=arithmetic(input.arithmetic.flops_per_output_element)*writes/iters;
  mpq_class transc=arithmetic(input.arithmetic.transcendental_per_output_element)*writes/iters;
  if ((flops>0 && (input.arithmetic.flops_use_mma ? tc_flops_per_ns_per_sm_ : cuda_flops_per_ns_per_sm_)<=0) ||
      (transc>0 && sfu_ops_per_ns_per_sm_<=0))
    throw std::invalid_argument("symbolic arithmetic rate: not_calibrated");
  mpq_class setup=Exact(fit_.setup_per_output_ns);
  for (auto const& tile:input.task.tile) setup*=tile.Eval(known,known);
  mpq_class output_bytes=(options_.fp32_partials && dtype_==ScalarType::kBF16 && chunks>1 ? mpq_class(4) : element_bytes)*writes;
  mpq_class fixed=Exact(fit_.setup_ns)+setup+traits.stages*bytes/Exact(l2_bytes_per_ns_per_sm_)+
      Exact(calib_->l2_latency_ns)+output_bytes/Exact(l2_bytes_per_ns_per_sm_);
  mpq_class effective=iters;
  if (options_.pipeline_envelope) effective=std::max(mpq_class(0),mpq_class(iters-(traits.stages-1)));

  // This is the symbolic form of the existing footprint expression, not a
  // finite-difference fit. A separate scratch plan needs its own QP term.
  std::vector<Polynomial::PolynomialInterval> misses{{begin,end,{"1","0","0"}}};
  if (options_.cache_model) {
    if (!cache_service_curve_) throw std::invalid_argument("symbolic cache curve: not_calibrated");
    misses=cache_service_curve_->MissIntervals(Footprint(model,dtype_==ScalarType::kBF16 ? 2 : 4),
        begin,end,calib_->l2_gbps,calib_->dram_gbps);
  }
  auto counts=input.work.task_count.SubstituteParams(known).QuadraticIntervals(parameter,begin,end);
  std::vector<Polynomial> regions;
  long grid=static_cast<long>(target_->res.num_sms)*residency.ctas_per_sm;
  if (grid<=0) throw std::invalid_argument("symbolic wave grid is empty");
  for (auto const& count:counts) {
    mpq_class tasks(count.coefficients[0]);
    if (mpq_class(count.coefficients[1])!=0 || mpq_class(count.coefficients[2])!=0 ||
        tasks<0 || tasks.get_den()!=1 || !mpz_fits_slong_p(tasks.get_num_mpz_t()))
      throw std::invalid_argument("collective ceil partition did not yield a constant task count");
    long n=tasks.get_num().get_si();
    for (auto const& miss:misses) {
      long first=std::max(count.begin,miss.begin),last=std::min(count.end,miss.end);
      if (first>last) continue;
      auto miss_qp=Expression(parameter,first,last,miss.coefficients);
      std::vector<Polynomial> waves;
      auto wave=[&](long active,long multiplicity) {
        mpq_class o=residency.ctas_per_sm;
        if (options_.wave_tail) o=std::max(mpq_class(1),mpq_class(active,target_->res.num_sms));
        std::vector<Polynomial> lanes(ResourceVector::kLaneCount,Polynomial::Constant(0));
        auto put=[&](ResourceVector::Lane lane,mpq_class value) {
          if (lanes_[lane]==LaneStatus::kLive && !options_.disabled_lanes[lane])
            lanes[lane]=Expression(parameter,first,last,{value.get_str(),"0","0"});
        };
        put(ResourceVector::kSmem,o*(dtype_==ScalarType::kBF16 ? bytes : mpq_class(reads/2))*Exact(fit_.lds_ns));
        if (options_.resource_lanes) {
          if (input.arithmetic.flops_use_mma && tc_flops_per_ns_per_sm_>0)
            put(ResourceVector::kTensorCore,o*flops/Exact(tc_flops_per_ns_per_sm_));
          else if (!input.arithmetic.flops_use_mma && cuda_flops_per_ns_per_sm_>0)
            put(ResourceVector::kCudaCore,o*flops/Exact(cuda_flops_per_ns_per_sm_));
          if (sfu_ops_per_ns_per_sm_>0) put(ResourceVector::kSfu,o*transc/Exact(sfu_ops_per_ns_per_sm_));
          put(ResourceVector::kL2,o*bytes/Exact(l2_bytes_per_ns_per_sm_));
          if (lanes_[ResourceVector::kDram]==LaneStatus::kLive && !options_.disabled_lanes[ResourceVector::kDram])
            lanes[ResourceVector::kDram]=miss_qp.ScaleRational(mpq_class(o*bytes/Exact(dram_bytes_per_ns_per_sm_)).get_str());
        }
        for (auto const& piece:QuadraticEnvelope(lanes,parameter,first,last,true)) {
          auto cost=Expression(parameter,piece.begin,piece.end,piece.coefficients).ScaleRational(effective.get_str())
              .Add(Expression(parameter,piece.begin,piece.end,{fixed.get_str(),"0","0"}));
          waves.push_back(cost.Scale(multiplicity));
        }
      };
      if (n/grid) wave(grid,n/grid);
      if (n%grid) wave(n%grid,1);
      regions.push_back(Polynomial::Sum(waves));
    }
  }
  return Polynomial::Sum(regions);
}

Polynomial CostModel::SymbolicScalarNs(DerivedTaskInput const& input,
    BackendTraits const& traits,Residency residency,ModelDescription const& model,
    std::string const& parameter,long begin,long end) const {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_SYMBOLIC_TASK_COST
  throw std::runtime_error("symbolic task pricing disabled");
#endif
  if (parameter.empty() || parameter!=model.seq_metric_parameter || !model.dims.past_parameter.empty() ||
      model.dims.past<0 || begin<=0 || begin>end || end==std::numeric_limits<long>::max() ||
      traits.stages!=0 || residency.ctas_per_sm<=0 || model.attention_plan ||
      !input.scalar_flow || input.cost_coordinates!=std::vector<std::string>{"q"})
    throw std::invalid_argument("symbolic scalar requires runtime ownership and one fixed-past seq role");
  if (options_.cache_model && !options_.measured_cache_curve)
    throw std::invalid_argument("symbolic scalar requires measured cache curve, not Gaussian CDF");
  auto known=FixedExceptSeq(model,parameter);
  auto counts=input.work.task_count.SubstituteParams(known);
  auto count_pieces=counts.QuadraticIntervals(parameter,begin,end);
  long max_tasks=0;
  for (auto const& piece:count_pieces) {
    if (mpq_class(piece.coefficients[2])!=0)
      throw std::invalid_argument("scalar task count must be affine after ceil partition");
    mpq_class a(piece.coefficients[0]),b(piece.coefficients[1]);
    mpq_class high=std::max(mpq_class(a+b*piece.begin),mpq_class(a+b*piece.end));
    if (high<0 || high.get_den()!=1 || !mpz_fits_slong_p(high.get_num_mpz_t()))
      throw std::invalid_argument("scalar task bound is not a representable integer");
    max_tasks=std::max(max_tasks,high.get_num().get_si());
  }
  long grid=static_cast<long>(target_->res.num_sms)*residency.ctas_per_sm;
  if (grid<=0) throw std::invalid_argument("symbolic scalar wave grid is empty");
  auto [depth,barriers]=input.scalar_flow->MemoryDepthAndBarriers(traits.threads);
  int element_bytes=dtype_==ScalarType::kBF16 ? 2 : 4;
  std::vector<Polynomial::PolynomialInterval> misses{{begin,end,{"1","0","0"}}};
  if (options_.cache_model) {
    if (!cache_service_curve_) throw std::invalid_argument("symbolic cache curve: not_calibrated");
    misses=cache_service_curve_->MissIntervals(Footprint(model,element_bytes),begin,end,calib_->l2_gbps,calib_->dram_gbps);
  }
  std::vector<Polynomial> miss_terms;
  for (auto const& piece:misses) miss_terms.push_back(Expression(parameter,piece.begin,piece.end,piece.coefficients));
  auto miss=Polynomial::Sum(miss_terms);
  analysis::ParamBinding origin;
  for (auto const& axis:input.task.Coordinates()) origin.Bind(axis,0);
  auto arithmetic=[&](analysis::ArithmeticRatio const& ratio) {
    if (ratio.denominator<=0) throw std::invalid_argument("invalid symbolic arithmetic denominator");
    return ratio.numerator.BindCoordinates(origin).SumDomain().SubstituteParams(known)
        .ScaleRational("1/"+std::to_string(ratio.denominator));
  };
  auto f=arithmetic(input.arithmetic.flops_per_output_element),t=arithmetic(input.arithmetic.transcendental_per_output_element);
  if ((!f.IsZero() && lanes_[ResourceVector::kCudaCore]!=LaneStatus::kLive) ||
      (!t.IsZero() && lanes_[ResourceVector::kSfu]!=LaneStatus::kLive))
    throw std::invalid_argument("symbolic scalar arithmetic: not_calibrated");
  std::vector<Polynomial> waves;
  for (long first=0;first<max_tasks;first+=grid) {
    auto occupancy=Expression(parameter,begin,end,{std::to_string(residency.ctas_per_sm),"0","0"});
    if (options_.wave_tail) {
      auto active=Envelope({counts.Add(Polynomial::Constant(-first)),Polynomial::Constant(grid)},parameter,begin,end,false);
      occupancy=Envelope({active.ScaleRational("1/"+std::to_string(target_->res.num_sms)),Polynomial::Constant(1)},parameter,begin,end,true);
    }
    std::map<std::string,Polynomial> unique_tasks;
    for (long q=first;q<std::min(first+grid,max_tasks);++q) {
      analysis::ParamBinding point; point.Bind("q",q);
      auto work=[&](Polynomial const& value) { return value.BindCoordinates(point).SumDomain().SubstituteParams(known); };
      auto writes=work(input.work.write_elements),reads=work(input.work.read_elements);
      auto support=writes.SupportIndicator();
      auto bytes=reads.Add(writes).Scale(element_bytes);
      std::vector<Polynomial> lanes(ResourceVector::kLaneCount,Polynomial::Constant(0));
      auto put=[&](ResourceVector::Lane lane,Polynomial const& work,double rate,bool enabled) {
        if (!enabled || options_.disabled_lanes[lane] || work.IsZero()) return;
        if (rate<=0) throw std::invalid_argument("symbolic scalar resource: not_calibrated");
        lanes[lane]=work.Multiply(occupancy).ScaleRational(mpq_class(1/Exact(rate)).get_str());
      };
      put(ResourceVector::kCudaCore,f.Multiply(writes),cuda_flops_per_ns_per_sm_,options_.resource_lanes);
      put(ResourceVector::kSfu,t.Multiply(writes),sfu_ops_per_ns_per_sm_,options_.resource_lanes);
      put(ResourceVector::kSmem,bytes,calib_->smem_gbps/target_->res.num_sms,input.arithmetic.smem_staged && calib_->smem_gbps>0);
      put(ResourceVector::kL2,bytes,l2_bytes_per_ns_per_sm_,options_.resource_lanes);
      put(ResourceVector::kDram,bytes.Multiply(miss),dram_bytes_per_ns_per_sm_,options_.resource_lanes);
      auto fixed=support.ScaleRational(mpq_class(depth*Exact(calib_->l2_latency_ns)+barriers*Exact(calib_->syncthreads_ns)).get_str())
          .Add(writes.Scale(element_bytes).ScaleRational(mpq_class(1/Exact(l2_bytes_per_ns_per_sm_)).get_str()));
      auto price=Envelope(lanes,parameter,begin,end,true).Add(fixed);
      unique_tasks.emplace(price.ToString(),price);
    }
    std::vector<Polynomial> tasks;
    for (auto const& [key,price]:unique_tasks) tasks.push_back(price);
    waves.push_back(Envelope(tasks,parameter,begin,end,true));
  }
  return Polynomial::Sum(waves);
}
}  // namespace tilemega::solver
