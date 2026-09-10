// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Codegen/tasks/TaskResources.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <sstream>

namespace tilemega::solver {
namespace {

constexpr int kCacheLineBytes = 128;

double CeilDiv(double a, double b) { return std::ceil(a / b); }

double Interpolate(std::vector<double> const& xs, std::vector<double> const& ys,
                   double x) {
  if (xs.empty() || xs.size() != ys.size()) return 0.0;
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  for (std::size_t i = 1; i < xs.size(); ++i) {
    if (x <= xs[i]) {
      double const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys.back();
}

/// Zelen & Severo 26.2.17: the normal tail, to about 7.5e-8 absolute.
double NormalTail(double x) {
  if (x < 0.0) return 1.0 - NormalTail(-x);
  double const t = 1.0 / (1.0 + 0.2316419 * x);
  double const poly =
      t * (0.319381530 +
           t * (-0.356563782 +
                t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI) * poly;
}

/// Scalar ld.shared warp-instructions one CTA issues per mainloop iteration.
/// The SIMT f32 TN collective holds 16x16 threads, so a thread owns tile_m/16
/// A elements and tile_n/16 B elements per k-substep and each is its own
/// instruction.
double LdsInstructions(GemmConfig const& c) {
  return static_cast<double>(c.tile_m + c.tile_n) * c.tile_k / 2.0;
}

/// cp.async bytes one CTA pulls per mainloop iteration.
double ElementBytes(ScalarType dtype) {
  return dtype == ScalarType::kBF16 ? 2.0 : 4.0;
}

double MainloopBytes(GemmConfig const& c, ScalarType dtype) {
  return ElementBytes(dtype) * c.tile_k * (c.tile_m + c.tile_n);
}

double MainloopFlops(GemmConfig const& c) {
  return 2.0 * c.tile_m * c.tile_n * c.tile_k;
}

double EpilogueBytes(GemmConfig const& c, ScalarType dtype, bool fp32_partial = false) {
  return (fp32_partial ? 4.0 : ElementBytes(dtype)) * c.tile_m * c.tile_n;
}

}  // namespace

char const* LaneStatusName(LaneStatus status) {
  switch (status) {
    case LaneStatus::kLive: return "live";
    case LaneStatus::kCapabilityAbsent: return "capability_absent";
    case LaneStatus::kNotCalibrated: return "not_calibrated";
  }
  return "unknown";
}

double ResourceVector::operator[](Lane lane) const {
  return const_cast<ResourceVector*>(this)->operator[](lane);
}

double& ResourceVector::operator[](Lane lane) {
  switch (lane) {
    case kTensorCore: return tensor_core;
    case kCudaCore: return cuda_core;
    case kSfu: return sfu;
    case kTmem: return tmem;
    case kSmem: return smem;
    case kL15: return l1_5;
    case kL2: return l2;
    case kDram: return dram;
    default: return net;
  }
}

char const* ResourceVector::LaneName(Lane lane) {
  switch (lane) {
    case kTensorCore: return "tc";
    case kCudaCore: return "cuda";
    case kSfu: return "sfu";
    case kTmem: return "tmem";
    case kSmem: return "smem";
    case kL15: return "l1_5";
    case kL2: return "l2";
    case kDram: return "ddr";
    default: return "net";
  }
}

double ResourceVector::Bottleneck() const {
  double top = 0.0;
  for (int lane = 0; lane < kLaneCount; ++lane)
    top = std::max(top, (*this)[static_cast<Lane>(lane)]);
  return top;
}

/// Ties go to the earliest lane in skeleton order, so the name is a function
/// of the vector and not of the loop -- an all-zero vector reports `tc`.
char const* ResourceVector::BottleneckName() const {
  double const top = Bottleneck();
  for (int lane = 0; lane < kLaneCount; ++lane)
    if ((*this)[static_cast<Lane>(lane)] == top)
      return LaneName(static_cast<Lane>(lane));
  return LaneName(kTensorCore);
}

CostModel::CostModel(TargetSpec const& target, CostModelOptions options)
    : CostModel(target, ScalarType::kF32, options) {}

CostModel::CostModel(TargetSpec const& target, ScalarType dtype,
                     CostModelOptions options)
    : target_(&target), calib_(&target.CalibrationFor(
                            dtype == ScalarType::kBF16 ? "bf16" : "f32")),
      dtype_(dtype), options_(options) {
  auto const& calib = *calib_;
  if (!calib.calibrated) {
    throw std::runtime_error(
        "cost model needs a calibrated target: run tilemega-calibrate");
  }
  if (options_.cache_model && options_.measured_cache_curve)
    cache_service_curve_.emplace(calib.l2_curve_bytes,calib.l2_curve_gbps);
  // A lane is live only if the target has the pipe *and* something measured
  // its rate.  Anything else is one of the two reasons, never a bare zero:
  // the tmem/l1_5/net lanes are absent on every target measured so far, and
  // an uncalibrated rate would otherwise divide by zero and look infinitely
  // fast rather than unknown.
  auto lane = [&](bool capable, double rate) {
    if (!capable) return LaneStatus::kCapabilityAbsent;
    return rate > 0.0 ? LaneStatus::kLive : LaneStatus::kNotCalibrated;
  };
  using RV = ResourceVector;
  double const tc_rate = dtype == ScalarType::kBF16
                             ? calib.tc_bf16_gflops
                             : calib.tc_fp16_gflops;
  lanes_[RV::kTensorCore] = lane(true, tc_rate);
  lanes_[RV::kCudaCore] = lane(true, calib.cuda_fp32_gflops);
  lanes_[RV::kSfu] = lane(true, calib.sfu_exp2_gops);
  lanes_[RV::kTmem] = lane(target.caps.tcgen05, 0.0);
  // `smem_gbps` is a scalar `ld.shared` throughput, and the SMEM lane's work
  // term is the SIMT mainloop's shared-memory traffic.  A BF16 Tensor Core
  // collective does not feed the MMA that way -- operands arrive by
  // `cp.async` into shared memory and leave it by `ldmatrix` straight into the
  // MMA's register operands -- so nothing in the BF16 profile measures the
  // path this lane is meant to price.  ⚠️ The lane was found harmful by
  // ablation first (removing it moved BF16's Spearman from 0.8246 to 0.8942)
  // and only then explained; the explanation is a micro-architectural
  // argument, not a fit, and must be re-checked on a target where the two can
  // be measured apart.
  lanes_[RV::kSmem] = dtype == ScalarType::kBF16
                          ? LaneStatus::kNotCalibrated
                          : lane(true, calib.smem_gbps);
  lanes_[RV::kL15] = lane(target.caps.l1_5, 0.0);
  lanes_[RV::kL2] = lane(true, calib.l2_gbps);
  lanes_[RV::kDram] = lane(true, calib.dram_gbps);
  lanes_[RV::kNet] = lane(target.caps.net, 0.0);

  double const sms = static_cast<double>(target.res.num_sms);
  l2_bytes_per_ns_per_sm_ = calib.l2_gbps / sms;
  dram_bytes_per_ns_per_sm_ = calib.dram_gbps / sms;
  cuda_flops_per_ns_per_sm_ = calib.cuda_fp32_gflops / sms;
  sfu_ops_per_ns_per_sm_ = calib.sfu_exp2_gops / sms;
  tc_flops_per_ns_per_sm_ = tc_rate / sms;

  // The mainloop slope `c` and the intercept `a` are calibrated at six tile
  // shapes and up to four resident CTA counts each; the model needs them at
  // every legal shape, so two scalars are fitted across those points.
  double lds_num = 0.0, lds_den = 0.0;
  int points = 0;
  for (auto const& shape : calib.streamk) {
    for (std::size_t i = 0; i < shape.occ_per_sm.size(); ++i) {
      GemmConfig cfg{shape.tile_m, shape.tile_n, shape.tile_k, shape.stages, 1};
      double const x = shape.occ_per_sm[i] *
                       (dtype_ == ScalarType::kBF16
                            ? MainloopBytes(cfg, dtype_)
                            : LdsInstructions(cfg));
      lds_num += x * shape.occ_c_ns[i];
      lds_den += x * x;
      ++points;
    }
  }
  if (points == 0 || lds_den <= 0.0) {
    throw std::runtime_error(
        "cost model needs calibrated Stream-K occupancy points");
  }
  fit_.points = points;
  fit_.lds_ns = lds_num / lds_den;

  double lds_sq = 0.0, setup_sum = 0.0;
  std::vector<double> setups, setup_areas;
  for (auto const& shape : calib.streamk) {
    GemmConfig cfg{shape.tile_m, shape.tile_n, shape.tile_k, shape.stages, 1};
    for (std::size_t i = 0; i < shape.occ_per_sm.size(); ++i) {
      double const predicted =
          shape.occ_per_sm[i] *
          (dtype_ == ScalarType::kBF16 ? MainloopBytes(cfg, dtype_)
                                       : LdsInstructions(cfg)) *
          fit_.lds_ns;
      double const e = predicted / shape.occ_c_ns[i] - 1.0;
      lds_sq += e * e;
      double const traffic =
          cfg.stages * (MainloopBytes(cfg, dtype_) / l2_bytes_per_ns_per_sm_) +
          calib.l2_latency_ns +
          EpilogueBytes(cfg, dtype_) / l2_bytes_per_ns_per_sm_;
      // `a` is the intercept of a line in `iters`, so under an envelope with
      // fill depth d it already carries -d*c.  Adding it back is what keeps
      // the envelope from being subtracted twice -- and is also what makes
      // `setup` occupancy-dependent, since c scales with CTAs/SM while a
      // per-CTA setup constant cannot.  Measured on the 12 calibrated points:
      // d=0 leaves setup at 706 ns with 483 ns rms, d=stages-1 at 3562 ns
      // with 1449 ns rms.
      double const fill =
          options_.pipeline_envelope ? (cfg.stages - 1) * shape.occ_c_ns[i] : 0.0;
      double const setup = shape.occ_a_ns[i] + fill - traffic;
      setups.push_back(setup);
      setup_areas.push_back(double(cfg.tile_m) * double(cfg.tile_n));
      setup_sum += setup;
    }
  }
  fit_.lds_rel_rms = std::sqrt(lds_sq / points);
  // A per-CTA setup is not one number.  It is dominated by what the CTA has to
  // materialize before and after the mainloop, which scales with the output
  // tile: on the BF16 profile the calibrated `a` runs from 0 ns at 32x32 to
  // 10112 ns at 256x128, and a single scalar over that range fitted with a
  // residual twice its own value (F-74).  `setup = alpha + beta * tile_m *
  // tile_n` is still two numbers fitted over the same points, so nothing new
  // is measured -- what changes is that the model is allowed to say the
  // epilogue of a large tile costs more than that of a small one.
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < setups.size(); ++i) {
    sx += setup_areas[i];
    sy += setups[i];
    sxx += setup_areas[i] * setup_areas[i];
    sxy += setup_areas[i] * setups[i];
  }
  double const n = static_cast<double>(setups.size());
  double const denominator = n * sxx - sx * sx;
  if (denominator != 0.0) {
    fit_.setup_per_output_ns = (n * sxy - sx * sy) / denominator;
    fit_.setup_ns = (sy - fit_.setup_per_output_ns * sx) / n;
  } else {
    fit_.setup_per_output_ns = 0.0;
    fit_.setup_ns = setup_sum / points;
  }
  double setup_sq = 0.0;
  for (std::size_t i = 0; i < setups.size(); ++i) {
    double const e = fit_.setup_ns + fit_.setup_per_output_ns * setup_areas[i] -
                     setups[i];
    setup_sq += e * e;
  }
  fit_.setup_rms_ns = std::sqrt(setup_sq / points);
}

double CostModel::CacheHitProbability(double footprint_bytes) const {
  if (!options_.cache_model) return 1.0;
  if (options_.measured_cache_curve)
    return cache_service_curve_->HitFraction(footprint_bytes,calib_->l2_gbps,calib_->dram_gbps);
  double const capacity_lines =
      calib_->l2_knee_bytes / static_cast<double>(kCacheLineBytes);
  double const block_lines =
      std::max(1.0, footprint_bytes / static_cast<double>(kCacheLineBytes));
  // §2.2(e): between two touches of a line the stream walks the whole live
  // footprint once, so the reuse distance in distinct lines is that footprint.
  double const distance = block_lines;
  double const p = std::min(1.0, capacity_lines / block_lines);
  double const mu = distance * p;
  double const var = distance * p * (1.0 - p);
  if (var <= 0.0) return capacity_lines >= block_lines ? 1.0 : 0.0;
  double const z = (capacity_lines - 1.0 - mu) / std::sqrt(var);
  return 1.0 - NormalTail(z);
}

ResourceVector CostModel::Steady(GemmConfig const& config, double ctas_per_sm,
                                 double dram_fraction) const {
  ResourceVector u;
  double const o = ctas_per_sm;
  u.smem = o *
           (dtype_ == ScalarType::kBF16 ? MainloopBytes(config, dtype_)
                                        : LdsInstructions(config)) *
           fit_.lds_ns;
  if (options_.resource_lanes) {
    if (dtype_ == ScalarType::kBF16)
      u.tensor_core = o * MainloopFlops(config) / tc_flops_per_ns_per_sm_;
    else
      u.cuda_core = o * MainloopFlops(config) / cuda_flops_per_ns_per_sm_;
    double const bytes = o * MainloopBytes(config, dtype_);
    u.l2 = bytes / l2_bytes_per_ns_per_sm_;
    u.dram = bytes * dram_fraction / dram_bytes_per_ns_per_sm_;
  }
  // The GEMM mainloop this models issues no tcgen05 MMA, touches no L1.5 and
  // crosses no fabric, so those three lanes stay zero on a target that has
  // them too; `lane_status` is what separates "no pipe" from "no work".
  for (int i = 0; i < ResourceVector::kLaneCount; ++i) {
    auto const l = static_cast<ResourceVector::Lane>(i);
    if (lanes_[l] != LaneStatus::kLive || options_.disabled_lanes[l])
      u[l] = 0.0;
  }
  return u;
}

int CostModel::Chunks(GemmOp const& gemm, GemmConfig const& config) const {
  if (!options_.split_k) return 1;
  int const k_tiles = static_cast<int>(CeilDiv(gemm.k, config.tile_k));
  int chunks = config.split_k < k_tiles ? config.split_k : k_tiles;
  return chunks < 1 ? 1 : chunks;
}

double CostModel::GemmStageNs(GemmOp const& gemm, GemmConfig const& config,
                              Residency residency,
                              ModelDescription const& model,
                              int* chunks_out) const {
  int const chunks = Chunks(gemm, config);
  if (chunks_out != nullptr) *chunks_out = chunks;
  double const k_tiles = CeilDiv(gemm.k, config.tile_k);
  double const tiles = CeilDiv(model.dims.seq, config.tile_m) *
                       CeilDiv(gemm.n, config.tile_n);
  double const ctas = tiles * chunks;
  double const iters = CeilDiv(k_tiles, chunks);
  double const grid = static_cast<double>(target_->res.num_sms) *
                      std::max(1, residency.ctas_per_sm);
  double const dram_fraction =
      1.0 - CacheHitProbability(model.LiveFootprintBytes());

  double const per_cta_fixed =
      fit_.setup_ns +
      fit_.setup_per_output_ns * double(config.tile_m) * double(config.tile_n) +
      config.stages * (MainloopBytes(config, dtype_) / l2_bytes_per_ns_per_sm_) +
      calib_->l2_latency_ns +
      EpilogueBytes(config, dtype_, options_.fp32_partials &&
          dtype_ == ScalarType::kBF16 && chunks > 1) / l2_bytes_per_ns_per_sm_;
  // §2.2(b): the first `stages - 1` iterations are covered by the fill.  The
  // second factor of `d = stages * resident_tiles_per_SM - 1` is 1 here: a CTA
  // of this collective owns exactly one output tile per wave, so resident CTAs
  // do not pipeline each other's K iterations -- they contend for the same
  // pipes, which is what `o` in Steady() carries.
  double const effective_iters =
      options_.pipeline_envelope ? std::max(iters - (config.stages - 1), 0.0)
                                 : iters;

  double total = 0.0;
  for (double remaining = ctas; remaining > 0.0; remaining -= grid) {
    double const active = std::min(grid, remaining);
    // §2.2(d): a tail wave that fills a fraction of the device runs at that
    // fraction's occupancy, which is why it is re-evaluated rather than
    // charged as an additive quantization penalty.
    double const o = options_.wave_tail
                         ? std::max(1.0, active / target_->res.num_sms)
                         : std::max(1, residency.ctas_per_sm);
    total += per_cta_fixed +
             effective_iters * Steady(config, o, dram_fraction).Bottleneck();
  }
  return total;
}

double CostModel::CombineStageNs(GemmOp const& gemm, int chunks,
                                 ModelDims const& dims) const {
  if (chunks <= 1) return 0.0;
  auto const& calib = *calib_;
  // §2.3: `b` and `d` are held per output element because at one shape's
  // single output width the reduction moves too little to fit (TargetSpec.h).
  double const b = calib.streamk.empty() ? 0.0 : calib.streamk.front().b_ns;
  double const d = calib.streamk.empty() ? 0.0 : calib.streamk.front().d_ns;
  double const elements = static_cast<double>(dims.seq) * gemm.n;
  if (options_.fp32_partials && dtype_ == ScalarType::kBF16) {
    if (options_.measured_partial_combine) {
      auto const& fit=calib.fp32_partial_combine;
      if (fit.reason!="measured" || !fit.fixed_ns || !fit.base_ns ||
          !fit.d_l2_ns || !fit.d_dram_ns)
        throw std::runtime_error("FP32 partial combine rate: "+fit.reason);
      double miss=1.0-CacheHitProbability(4.0*chunks*elements);
      double peer=(1.0-miss)*(*fit.d_l2_ns)+miss*(*fit.d_dram_ns);
      return *fit.fixed_ns+(*fit.base_ns+peer*(chunks-1))*elements;
    }
    // The existing coefficients measured BF16 partial reads. Add the extra
    // two bytes per partial, not a 2x scale of the entire compute/launch fit.
    // A doubled working set can cross the L2 knee; the measured DRAM peer
    // coefficient must then replace the L2 coefficient explicitly.
    double const miss = 1.0 - CacheHitProbability(4.0 * chunks * elements);
    if (miss > 0.0 && calib.combine_d_dram_ns <= 0.0)
      throw std::runtime_error("FP32 partial combine DRAM rate: not_calibrated");
    double const peer = (1.0-miss)*d + miss*calib.combine_d_dram_ns;
    double const extra = 2.0 * chunks * elements *
        ((1.0-miss)/calib.l2_gbps + miss/calib.dram_gbps);
    return calib.combine_fixed_ns + (b + peer * (chunks - 1)) * elements + extra;
  }
  return calib.combine_fixed_ns + (b + d * (chunks - 1)) * elements;
}

double CostModel::TaskCostNs(DerivedTaskInput const& input, BackendTraits const& traits,
                             Residency residency, ModelDescription const& model,
                             int chunks) const {
  analysis::IslReferenceAudit audit(__func__);
#if defined(TILEMEGA_DERIVED_TASK_COST) && !TILEMEGA_DERIVED_TASK_COST
  throw std::runtime_error("access-derived task pricing is disabled");
#endif
  if (model.dims.IsSymbolic()) throw std::invalid_argument("bind theta before FP64 task evaluation");
  auto known=model.MetricBindings();
  if (traits.stages<=0) {
    if (!input.scalar_flow || input.cost_coordinates!=std::vector<std::string>{"q"})
      throw std::invalid_argument("scalar task latency DAG/ownership has not been supplied");
    auto [depth,barriers]=input.scalar_flow->MemoryDepthAndBarriers(traits.threads);
    double ctas=double(input.work.task_count.SubstituteParams(known).Eval({}));
    double grid=double(target_->res.num_sms)*std::max(1,residency.ctas_per_sm);
    double miss=1.0-CacheHitProbability(model.LiveFootprintBytes());
    double flops_per_output=input.arithmetic.flops_per_output_element.Eval(known);
    double transc_per_output=input.arithmetic.transcendental_per_output_element.Eval(known);
    std::ostringstream cache_key;
    cache_key << input.work.read_elements.ToString() << '\n' << input.work.write_elements.ToString()
              << '\n' << input.work.task_count.ToString() << '\n' << std::hexfloat
              << miss << ':' << flops_per_output << ':' << transc_per_output << ':'
              << depth << ':' << barriers << ':' << traits.threads << ':' << residency.ctas_per_sm
              << ':' << input.arithmetic.smem_staged;
    for (auto const& [name,value]:known.values) cache_key << ':' << name << '=' << value;
    auto cached=scalar_price_cache_.find(cache_key.str());
    if (cached!=scalar_price_cache_.end()) return cached->second;
    auto statuses=lanes_;
    // Unlike a BF16 MMA collective, these bodies issue scalar shared accesses.
    // The measured scalar pipe applies; the GEMM ldmatrix status is unchanged.
    statuses[ResourceVector::kSmem]=calib_->smem_gbps>0 ? LaneStatus::kLive : LaneStatus::kNotCalibrated;
    double total=0;
    long first=0;
    for (double remaining=ctas;remaining>0;remaining-=grid) {
      double active=std::min(grid,remaining);
      double o=options_.wave_tail ? std::max(1.0,active/target_->res.num_sms)
                                 : std::max(1,residency.ctas_per_sm);
      double wave=-std::numeric_limits<double>::infinity();
      for (long q=first;q<first+long(active);++q) {
        analysis::ParamBinding coordinate; coordinate.Bind("q",q);
        auto value=[&](analysis::QuasiPolynomial const& work) {
          return double(work.BindCoordinates(coordinate).SubstituteParams(known).Eval({}));
        };
        double writes=value(input.work.write_elements);
        double bytes=(value(input.work.read_elements)+writes)*ElementBytes(dtype_);
        double flops=flops_per_output*writes,transc=transc_per_output*writes;
        if ((flops>0 && lanes_[ResourceVector::kCudaCore]!=LaneStatus::kLive) ||
            (transc>0 && lanes_[ResourceVector::kSfu]!=LaneStatus::kLive))
          throw std::runtime_error("scalar arithmetic rate: not_calibrated");
        ResourceVector u;
        if (flops>0) u.cuda_core=o*flops/cuda_flops_per_ns_per_sm_;
        if (transc>0) u.sfu=o*transc/sfu_ops_per_ns_per_sm_;
        if (input.arithmetic.smem_staged && statuses[ResourceVector::kSmem]==LaneStatus::kLive)
          u.smem=o*bytes/(calib_->smem_gbps/target_->res.num_sms);
        u.l2=o*bytes/l2_bytes_per_ns_per_sm_;
        u.dram=o*bytes*miss/dram_bytes_per_ns_per_sm_;
        for (int i=0;i<ResourceVector::kLaneCount;++i) {
          auto lane=static_cast<ResourceVector::Lane>(i);
          if (statuses[lane]!=LaneStatus::kLive || options_.disabled_lanes[lane] ||
              (!options_.resource_lanes && lane!=ResourceVector::kSmem)) u[lane]=0;
        }
        // The fitted alpha+beta*tile_area describes a collective accumulator
        // tile's setup. ScalarDataflow has no such initialization phase:
        // scalar arithmetic is charged in u, memory phases in this DAG term.
        // In particular a negative GEMM regression intercept cannot be
        // extrapolated into a small SIMT task then silently clamped to zero.
        double fixed=depth*calib_->l2_latency_ns+barriers*calib_->syncthreads_ns+
            writes*ElementBytes(dtype_)/l2_bytes_per_ns_per_sm_;
        if (!(fixed+u.Bottleneck()>0) || !std::isfinite(fixed+u.Bottleneck()))
          throw std::runtime_error("nonpositive or nonfinite scalar task price");
        wave=std::max(wave,fixed+u.Bottleneck());
      }
      total+=wave;
      first+=long(active);
    }
    scalar_price_cache_.emplace(cache_key.str(),total);
    return total;
  }
  if (traits.tile_k<=0) throw std::invalid_argument("collective reduction tile is missing");
  analysis::ParamBinding point;
  for (auto const& coordinate:input.task.Coordinates()) point.Bind(coordinate,0);
  auto value=[&](analysis::QuasiPolynomial const& quantity) {
    return double(quantity.BindCoordinates(point).SubstituteParams(known).Eval({}));
  };
  double const ctas=double(input.work.task_count.SubstituteParams(known).Eval({}));
  double const iters=value(input.work.nominal_task_reduce_extent)/traits.tile_k;
  if (iters<=0 || iters!=std::floor(iters)) throw std::invalid_argument("invalid collective reduction work");
  double const reads=value(input.work.nominal_read_elements)/iters;
  double const writes=value(input.work.nominal_write_elements);
  double const bytes=ElementBytes(dtype_)*reads;
  double const flops=input.arithmetic.flops_per_output_element.Eval(known)*writes/iters;
  double const transc=input.arithmetic.transcendental_per_output_element.Eval(known)*writes/iters;
  double const dram_fraction=1.0-CacheHitProbability(model.LiveFootprintBytes());
  // Keep the calibrated factorized FP64 setup order. beta*(M*N) is not in
  // general bit-identical to (beta*M)*N, although the integer work is exact.
  double setup=fit_.setup_per_output_ns;
  for (std::size_t axis=0;axis<input.task.tile.size();++axis) {
    setup*=double(input.task.tile[axis].Eval(known,known));
  }
  double const epilogue=(options_.fp32_partials && dtype_==ScalarType::kBF16 && chunks>1
                           ? double(sizeof(float)) : ElementBytes(dtype_))*writes;
  double const fixed=fit_.setup_ns+setup+traits.stages*(bytes/l2_bytes_per_ns_per_sm_)+
      calib_->l2_latency_ns+epilogue/l2_bytes_per_ns_per_sm_;
  double const effective_iters=options_.pipeline_envelope
      ? std::max(iters-(traits.stages-1),0.0) : iters;
  double const grid=double(target_->res.num_sms)*std::max(1,residency.ctas_per_sm);
  double total=0.0;
  for (double remaining=ctas;remaining>0.0;remaining-=grid) {
    double const active=std::min(grid,remaining);
    double const o=options_.wave_tail ? std::max(1.0,active/target_->res.num_sms)
                                    : std::max(1,residency.ctas_per_sm);
    ResourceVector u;
    // The historical scalar 16x16 thread layout issues read_elements/2
    // scalar shared warp instructions; BF16's corresponding lane is disabled.
    u.smem=o*(dtype_==ScalarType::kBF16 ? bytes : reads/2.0)*fit_.lds_ns;
    if (options_.resource_lanes) {
      if (input.arithmetic.flops_use_mma) u.tensor_core=o*flops/tc_flops_per_ns_per_sm_;
      else u.cuda_core=o*flops/cuda_flops_per_ns_per_sm_;
      u.sfu=o*transc/sfu_ops_per_ns_per_sm_;
      u.l2=(o*bytes)/l2_bytes_per_ns_per_sm_;
      u.dram=(o*bytes)*dram_fraction/dram_bytes_per_ns_per_sm_;
    }
    for (int i=0;i<ResourceVector::kLaneCount;++i) {
      auto lane=static_cast<ResourceVector::Lane>(i);
      if (lanes_[lane]!=LaneStatus::kLive || options_.disabled_lanes[lane]) u[lane]=0.0;
    }
    total+=fixed+effective_iters*u.Bottleneck();
  }
  return total;
}

double CostModel::NonGemmStageNs(ModelStage const& stage, ModelDims const& dims,
                                 Residency residency) const {
  if (!options_.non_gemm) return 0.0;
  auto const& calib = *calib_;
  double ctas = 0.0, bytes = 0.0, sfu_ops = 0.0;
  int depth = 0, barriers = 0;
  // The non-GEMM bodies share the megakernel's collective launch width.
  // OFF preserves the historical FP32-family assumption as an ablation.
  int const kSimtThreads = options_.task_body_traits && dtype_ == ScalarType::kBF16
                              ? kTensorBF16Threads : kSimtF32Threads;
  double const width = std::max(stage.width, 1);
  double const extent = std::max(stage.extent, 1);
  switch (stage.kind) {
    case StageKind::kRMSNorm:
      ctas = dims.seq;
      bytes = 3.0 * width * 4.0;
      depth = 2;
      barriers = 2;
      break;
    case StageKind::kRoPE:
      ctas = CeilDiv(dims.seq * extent * (width / 2.0), kSimtThreads);
      bytes = kSimtThreads * 4.0 * 4.0;
      sfu_ops = 2.0 * kSimtThreads;
      depth = 2;
      break;
    case StageKind::kKVAppend:
      ctas = CeilDiv(std::max(dims.seq, dims.past) * extent * width,
                     kSimtThreads);
      bytes = kSimtThreads * 2.0 * 4.0;
      depth = 2;
      break;
    case StageKind::kElementwise:
      ctas = CeilDiv(dims.seq * extent, kSimtThreads);
      bytes = kSimtThreads * 3.0 * 4.0;
      sfu_ops = kSimtThreads;
      depth = 2;
      break;
    case StageKind::kAttention:
      ctas = dims.seq * extent;
      bytes = (2.0 * dims.total * width + 2.0 * width) * 4.0;
      sfu_ops = kSimtThreads;
      depth = 3;
      barriers = 3;
      break;
    case StageKind::kGemm:
      return 0.0;
  }
  double const grid = static_cast<double>(target_->res.num_sms) *
                      std::max(1, residency.ctas_per_sm);
  double total = 0.0;
  for (double remaining = std::max(ctas, 1.0); remaining > 0.0;
       remaining -= grid) {
    double const active = std::min(grid, remaining);
    double const o = std::max(1.0, active / target_->res.num_sms);
    // These stages own a few hundred elements per CTA.  Their cost is a
    // handful of dependent round trips plus that traffic, not a bandwidth:
    // a model built only on rates predicts a few tens of nanoseconds for a
    // stage the profile shows at several microseconds.
    total += depth * calib.l2_latency_ns +
             o * bytes / l2_bytes_per_ns_per_sm_ +
             barriers * calib.syncthreads_ns +
             o * sfu_ops / sfu_ops_per_ns_per_sm_;
  }
  return total;
}

double CostModel::TaskStageNs(ModelDescription const& model,int index,
                              GemmConfig const& requested,Residency residency) const {
  analysis::IslReferenceAudit audit(__func__);
  GemmConfig config=requested;
  if (!options_.split_k) config.split_k=1;
  auto const& stage=model.stages.at(index);
  if (stage.kind!=StageKind::kGemm && !options_.non_gemm) return 0;
  ModelTaskSemantics const* selected=nullptr;
  for (auto const& semantic:model.task_semantics) if (semantic.stage==index) {
    // A native residual epilogue belongs to the existing collective stage
    // envelope. It is not a second runtime task or a newly selected fusion.
    if (stage.kind==StageKind::kGemm && semantic.op.arithmetic=="add") continue;
    if (selected) throw std::invalid_argument("runtime stage needs an explicit mixed-task composition");
    selected=&semantic;
  }
  if (!selected) throw std::invalid_argument("runtime stage lacks a CG semantic cost input");
  BackendTraits traits;
  bool collective=stage.kind==StageKind::kGemm;
  int chunks=collective ? Chunks(model.gemms.at(stage.gemm),config) : 1;
  if (collective) traits=dtype_==ScalarType::kBF16
      ? TensorBF16Traits(config.tile_m,config.tile_n,config.tile_k,config.stages)
      : SimtF32Traits(config.tile_m,config.tile_n,config.tile_k,config.stages);
  else {
    traits.threads=dtype_==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
    traits.smem_bytes=sizeof(float)*codegen::SimtSharedElements(
        static_cast<codegen::TaskKind>(stage.kind),traits.threads,TILEMEGA_ATTENTION_MAX_TOTAL);
  }
  std::ostringstream key;
  key << analysis::EncodeSemanticOp(selected->op) << ':' << selected->element_chunk << ':'
      << stage.width << ':' << stage.extent << ':' << stage.group << ':' << int(model.dtype);
  for (int operand:stage.operands) key << ':' << operand;
  for (auto const& [name,tile]:selected->tiles) key << ':' << name << '=' << tile.ToString();
  if (collective) key << ':' << config.tile_m << ':' << config.tile_n << ':' << config.tile_k
                       << ':' << config.stages << ':' << config.split_k;
  auto found=task_input_cache_.find(key.str());
  if (found==task_input_cache_.end()) {
    auto graph=InstantiateModelTasks(model,std::vector<GemmConfig>(model.gemms.size(),config));
    auto input=std::make_shared<DerivedTaskInput>(DeriveModelTaskInput(model,*selected,graph,collective ? &config : nullptr));
    found=task_input_cache_.emplace(key.str(),std::move(input)).first;
  }
  return TaskCostNs(*found->second,traits,residency,model,chunks);
}

double CostModel::BarrierNs(Residency residency) const {
  if (!options_.sync) return 0.0;
  double const grid = static_cast<double>(target_->res.num_sms) *
                      std::max(1, residency.ctas_per_sm);
  return Interpolate(calib_->grid_barrier_ctas, calib_->grid_barrier_ns, grid);
}

double CostModel::InterfaceEdgeNs(ModelCouplingMetrics const& edge,
                                  ModelDescription const& model) const {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_CG_INTERFACE_COST
  throw std::runtime_error("CG interface pricing is disabled");
#endif
  auto known=model.MetricBindings();
  long waits=edge.wait.SumDomain().SubstituteParams(known).Eval({});
  // Only consumers in domain(C) owe a first read. Subtracting |T_c| would
  // produce negative work for consumers outside a partial writer's domain.
  long consumers=edge.relation.Reverse().ImageCard().SubstituteParams(known).Eval({});
  long volume=edge.volume.SubstituteParams(known).Eval({});
  if (waits<consumers || volume<0) throw std::invalid_argument("invalid physical interface incidence");
  if (calib_->l2_gbps<=0 || calib_->dram_gbps<=0)
    throw std::runtime_error("interface bandwidth: not_calibrated");
  double miss=1.0-CacheHitProbability(model.LiveFootprintBytes());
  return double(waits-consumers)*double(volume)*ElementBytes(dtype_)*
      ((1.0-miss)/calib_->l2_gbps+miss/calib_->dram_gbps);
}

double CostModel::EventNs(ModelDescription const& model, std::vector<GemmConfig> const& configs,
                         Residency residency, int stage_count) const {
  if (!options_.sync || !options_.l2_events) return 0.0;
  if (!model.coupling_metrics.runtime)
    throw std::invalid_argument("L2 price requires exact CG runtime event metrics");
  auto const& metrics=*model.coupling_metrics.runtime;
  int threads=dtype_==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
  if (metrics.grid!=target_->res.num_sms*residency.ctas_per_sm || metrics.threads!=threads ||
      metrics.kappa!=options_.kappa || metrics.stage_count!=stage_count || metrics.gemms.size()!=configs.size())
    throw std::invalid_argument("event metrics do not match runtime residency, kappa or stage rewrite");
  for (std::size_t i=0;i<configs.size();++i) {
    auto const& a=configs[i]; auto const& b=metrics.gemms[i];
    if (a.tile_m!=b.tile_m || a.tile_n!=b.tile_n || a.tile_k!=b.tile_k ||
        a.stages!=b.stages || a.split_k!=b.split_k)
      throw std::invalid_argument("event metrics belong to a different GEMM variant");
  }
  auto const& calibration=target_->EventCalibrationFor(dtype_==ScalarType::kBF16 ? "bf16" : "f32");
  auto rate=[](TargetSpec::EventRate const& r,char const* unit) {
    if (!r.ns || r.reason!="measured") throw std::runtime_error("structured event rate: "+r.reason);
    if (r.unit!=unit || !std::isfinite(*r.ns) || *r.ns<0)
      throw std::invalid_argument("structured event rate has invalid units or value");
    return *r.ns;
  };
  auto known=model.MetricBindings();
  auto count=[&](analysis::QuasiPolynomial const& q) {
    long value=q.SubstituteParams(known).Eval({});
    if (value<0) throw std::invalid_argument("negative runtime event work");
    return static_cast<double>(value);
  };
  double notify=rate(calibration.notify,"ns/runtime_task_ref");
  double poll=rate(calibration.poll,"ns/runtime_wait_entry");
  double total=count(metrics.task_refs)*notify+count(metrics.wait_entries)*poll;
  total+=metrics.stage_count*(rate(calibration.notify_stage,"ns/runtime_stage")+
                             rate(calibration.poll_stage,"ns/runtime_stage"));
  total+=count(metrics.max_worker_task_refs)*
      (rate(calibration.notify_longest_worker,"ns/max_worker_task_ref")+
       rate(calibration.poll_longest_worker,"ns/max_worker_task_ref"));
  if (calibration.fence.ns)
    total-=count(metrics.fence_free_producers)*rate(calibration.fence,"ns/fence_free_producer");
  total-=count(metrics.fused_edges)*(notify+poll);
  if (total<0) throw std::invalid_argument("event rebates exceed the priced event work");
  return total;
}

double CostModel::InterfaceNs(ModelDescription const& model,
                               std::vector<GemmConfig> const& configs) const {
  double total=0;
  for (auto const& edge:InstantiateModelCouplings(model,configs)) total+=InterfaceEdgeNs(edge,model);
  return total;
}

CostBreakdown CostModel::Evaluate(ModelDescription const& model,
                                  std::vector<GemmConfig> const& configs,
                                  Residency residency) const {
  if (model.dims.IsSymbolic())
    throw std::invalid_argument("bind model dimensions before FP64 evaluation");
  if (configs.size() != model.gemms.size()) {
    throw std::invalid_argument("one GemmConfig per model GEMM is required");
  }
  CostBreakdown out;
  for (std::size_t i=0;i<model.stages.size();++i) {
    auto const& stage=model.stages[i];
    ++out.stage_count;
    if (stage.kind != StageKind::kGemm) {
      if (options_.unified_task_cost)
        out.task_ns_sum+=TaskStageNs(model,int(i),configs.empty() ? GemmConfig{} : configs.front(),residency);
      else out.other_ns += NonGemmStageNs(stage, model.dims, residency);
      continue;
    }
    GemmOp const& gemm = model.gemms[stage.gemm];
    GemmConfig const& config = configs[stage.gemm];
    int chunks = 1;
    if (options_.unified_task_cost) {
      chunks=Chunks(gemm,config);
      out.task_ns_sum+=TaskStageNs(model,int(i),config,residency);
    } else out.gemm_ns += GemmStageNs(gemm, config, residency, model, &chunks);
    if (chunks > 1) {
      // The split rewrite appends a combiner stage, which the megakernel pays
      // for with its own grid barrier: split-K buys arithmetic parallelism and
      // spends synchronization.
      out.combine_ns += CombineStageNs(gemm, chunks, model.dims);
      ++out.stage_count;
    }
  }
  out.barrier_ns = options_.l2_events ? 0.0 : out.stage_count * BarrierNs(residency);
  out.event_ns=EventNs(model,configs,residency,out.stage_count);
  out.total_ns = options_.unified_task_cost ? out.task_ns_sum+out.combine_ns+out.barrier_ns
      : out.gemm_ns + out.combine_ns + out.other_ns + out.barrier_ns;
  if (options_.l2_events) out.total_ns+=out.event_ns;
  if (options_.cg_interface) {
    out.interface_ns=InterfaceNs(model,configs);
    out.total_ns+=out.interface_ns;
  }
  return out;
}

CostBreakdown CostModel::Evaluate(ModelDescription const& model,
                                  GemmConfig config,
                                  Residency residency) const {
  return Evaluate(model, std::vector<GemmConfig>(model.gemms.size(), config),
                  residency);
}

}  // namespace tilemega::solver
