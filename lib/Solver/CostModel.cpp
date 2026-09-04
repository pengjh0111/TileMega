// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CostModel.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tilemega::solver {
namespace {

constexpr int kSimtThreads = 256;
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
double MainloopBytes(GemmConfig const& c) {
  return 4.0 * c.tile_k * (c.tile_m + c.tile_n);
}

double MainloopFlops(GemmConfig const& c) {
  return 2.0 * c.tile_m * c.tile_n * c.tile_k;
}

double EpilogueBytes(GemmConfig const& c) {
  return 4.0 * c.tile_m * c.tile_n;
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
    : target_(&target), options_(options) {
  auto const& calib = target.calib;
  if (!calib.calibrated) {
    throw std::runtime_error(
        "cost model needs a calibrated target: run tilemega-calibrate");
  }
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
  lanes_[RV::kTensorCore] = lane(true, calib.tc_fp16_gflops);
  lanes_[RV::kCudaCore] = lane(true, calib.cuda_fp32_gflops);
  lanes_[RV::kSfu] = lane(true, calib.sfu_exp2_gops);
  lanes_[RV::kTmem] = lane(target.caps.tcgen05, 0.0);
  lanes_[RV::kSmem] = lane(true, calib.smem_gbps);
  lanes_[RV::kL15] = lane(target.caps.l1_5, 0.0);
  lanes_[RV::kL2] = lane(true, calib.l2_gbps);
  lanes_[RV::kDram] = lane(true, calib.dram_gbps);
  lanes_[RV::kNet] = lane(target.caps.net, 0.0);

  double const sms = static_cast<double>(target.res.num_sms);
  l2_bytes_per_ns_per_sm_ = calib.l2_gbps / sms;
  dram_bytes_per_ns_per_sm_ = calib.dram_gbps / sms;
  cuda_flops_per_ns_per_sm_ = calib.cuda_fp32_gflops / sms;
  sfu_ops_per_ns_per_sm_ = calib.sfu_exp2_gops / sms;
  tc_flops_per_ns_per_sm_ = calib.tc_fp16_gflops / sms;

  // The mainloop slope `c` and the intercept `a` are calibrated at six tile
  // shapes and up to four resident CTA counts each; the model needs them at
  // every legal shape, so two scalars are fitted across those points.
  double lds_num = 0.0, lds_den = 0.0;
  int points = 0;
  for (auto const& shape : calib.streamk) {
    for (std::size_t i = 0; i < shape.occ_per_sm.size(); ++i) {
      GemmConfig cfg{shape.tile_m, shape.tile_n, shape.tile_k, shape.stages, 1};
      double const x = shape.occ_per_sm[i] * LdsInstructions(cfg);
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
  std::vector<double> setups;
  for (auto const& shape : calib.streamk) {
    GemmConfig cfg{shape.tile_m, shape.tile_n, shape.tile_k, shape.stages, 1};
    for (std::size_t i = 0; i < shape.occ_per_sm.size(); ++i) {
      double const predicted =
          shape.occ_per_sm[i] * LdsInstructions(cfg) * fit_.lds_ns;
      double const e = predicted / shape.occ_c_ns[i] - 1.0;
      lds_sq += e * e;
      double const traffic =
          cfg.stages * (MainloopBytes(cfg) / l2_bytes_per_ns_per_sm_) +
          calib.l2_latency_ns + EpilogueBytes(cfg) / l2_bytes_per_ns_per_sm_;
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
      setup_sum += setup;
    }
  }
  fit_.lds_rel_rms = std::sqrt(lds_sq / points);
  fit_.setup_ns = setup_sum / points;
  double setup_sq = 0.0;
  for (double s : setups) {
    double const e = fit_.setup_ns - s;
    setup_sq += e * e;
  }
  fit_.setup_rms_ns = std::sqrt(setup_sq / points);
}

double CostModel::CacheHitProbability(double footprint_bytes) const {
  if (!options_.cache_model) return 1.0;
  double const capacity_lines =
      target_->calib.l2_knee_bytes / static_cast<double>(kCacheLineBytes);
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
  u.smem = o * LdsInstructions(config) * fit_.lds_ns;
  if (options_.resource_lanes) {
    u.cuda_core = o * MainloopFlops(config) / cuda_flops_per_ns_per_sm_;
    double const bytes = o * MainloopBytes(config);
    u.l2 = bytes / l2_bytes_per_ns_per_sm_;
    u.dram = bytes * dram_fraction / dram_bytes_per_ns_per_sm_;
  }
  // The GEMM mainloop this models issues no tcgen05 MMA, touches no L1.5 and
  // crosses no fabric, so those three lanes stay zero on a target that has
  // them too; `lane_status` is what separates "no pipe" from "no work".
  for (int i = 0; i < ResourceVector::kLaneCount; ++i) {
    auto const l = static_cast<ResourceVector::Lane>(i);
    if (lanes_[l] != LaneStatus::kLive) u[l] = 0.0;
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
      config.stages * (MainloopBytes(config) / l2_bytes_per_ns_per_sm_) +
      target_->calib.l2_latency_ns +
      EpilogueBytes(config) / l2_bytes_per_ns_per_sm_;
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
  auto const& calib = target_->calib;
  // §2.3: `b` and `d` are held per output element because at one shape's
  // single output width the reduction moves too little to fit (TargetSpec.h).
  double const b = calib.streamk.empty() ? 0.0 : calib.streamk.front().b_ns;
  double const d = calib.streamk.empty() ? 0.0 : calib.streamk.front().d_ns;
  double const elements = static_cast<double>(dims.seq) * gemm.n;
  return calib.combine_fixed_ns + (b + d * (chunks - 1)) * elements;
}

double CostModel::NonGemmStageNs(ModelStage const& stage, ModelDims const& dims,
                                 Residency residency) const {
  if (!options_.non_gemm) return 0.0;
  auto const& calib = target_->calib;
  double ctas = 0.0, bytes = 0.0, sfu_ops = 0.0;
  int depth = 0, barriers = 0;
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

double CostModel::BarrierNs(Residency residency) const {
  if (!options_.sync) return 0.0;
  double const grid = static_cast<double>(target_->res.num_sms) *
                      std::max(1, residency.ctas_per_sm);
  return Interpolate(target_->calib.grid_barrier_ctas,
                     target_->calib.grid_barrier_ns, grid);
}

CostBreakdown CostModel::Evaluate(ModelDescription const& model,
                                  std::vector<GemmConfig> const& configs,
                                  Residency residency) const {
  if (configs.size() != model.gemms.size()) {
    throw std::invalid_argument("one GemmConfig per model GEMM is required");
  }
  CostBreakdown out;
  for (auto const& stage : model.stages) {
    ++out.stage_count;
    if (stage.kind != StageKind::kGemm) {
      out.other_ns += NonGemmStageNs(stage, model.dims, residency);
      continue;
    }
    GemmOp const& gemm = model.gemms[stage.gemm];
    GemmConfig const& config = configs[stage.gemm];
    int chunks = 1;
    out.gemm_ns += GemmStageNs(gemm, config, residency, model, &chunks);
    if (chunks > 1) {
      // The split rewrite appends a combiner stage, which the megakernel pays
      // for with its own grid barrier: split-K buys arithmetic parallelism and
      // spends synchronization.
      out.combine_ns += CombineStageNs(gemm, chunks, model.dims);
      ++out.stage_count;
    }
  }
  out.barrier_ns = out.stage_count * BarrierNs(residency);
  out.total_ns = out.gemm_ns + out.combine_ns + out.other_ns + out.barrier_ns;
  return out;
}

CostBreakdown CostModel::Evaluate(ModelDescription const& model,
                                  GemmConfig config,
                                  Residency residency) const {
  return Evaluate(model, std::vector<GemmConfig>(model.gemms.size(), config),
                  residency);
}

}  // namespace tilemega::solver
