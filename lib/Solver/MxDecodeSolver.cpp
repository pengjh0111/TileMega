// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Solver/MxDecodeSolver.cpp: see the header.
#include "tilemega/Solver/MxDecodeSolver.h"

#include "tilemega/Backend/Mx/DecodeShapes.h"
#include "tilemega/Support/Json.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
namespace {

using codegen::MxDecodeClass;
using codegen::MxDecodeConfig;
using codegen::MxDecodeModel;

struct ClassShape {
  char const* name;
  char const* op;
  int norm;
};
constexpr ClassShape kClasses[] = {{"qkv", "kQkvRope", 1},
                                   {"o", "kResidual", 0},
                                   {"gate_up", "kSwiGLU", 1},
                                   {"down", "kResidual", 0},
                                   {"head", "kArgmax", 1}};

void Shape(MxDecodeModel const& m, std::string const& cls, int& n, int& k) {
  if (cls == "qkv") n = (m.q_per_kv + 2) * m.hkv * m.head_dim, k = m.hidden;
  else if (cls == "o") n = m.hidden, k = m.heads * m.head_dim;
  else if (cls == "gate_up") n = 2 * m.inter, k = m.hidden;
  else if (cls == "down") n = m.hidden, k = m.inter;
  else if (cls == "head") n = m.vocab, k = m.hidden;
  else throw std::invalid_argument("mx decode solver: unknown class " + cls);
}

/// §4.3 features of one stage. The terms are those of
///   T = t_bar + t_A + max_c sum_seg (t_seg + bytes_seg / r(TN, D))
///       + [last segment across CTAs] t_last + t_epi(Op, B*TN)
/// for the busiest CTA (most weight bytes): [1, segments, bytes per (TN, D)
/// (15 rates), cross-CTA, B*TN per Op (4)].
constexpr int kTnD = 15, kFeatures = 3 + kTnD + 4;
int TnDIndex(int tn, int d) {
  int const t = tn == 16 ? 0 : tn == 32 ? 1 : tn == 64 ? 2 : 3;
  return t * 4 + (d - 1) - (t == 3 && d == 4 ? 1000 : 0);  // TN128 has D <= 3
}
int OpIndex(std::string const& op) {
  return op == "kQkvRope" ? 0 : op == "kResidual" ? 1 : op == "kSwiGLU" ? 2 : 3;
}
std::vector<double> Features(int n, int k, int b, std::string const& op, int tn, int d, bool stream_k, int grid) {
  std::vector<double> x(kFeatures, 0.0);
  long long const kc = k / 32, tiles = n / tn, units = tiles * kc;
  long long best = 0, segments = 0;
  bool cross = false;
  for (long long c = 0; c < grid; ++c) {
    long long begin, end;
    if (stream_k) begin = c * units / grid, end = (c + 1) * units / grid;
    else begin = c * tiles / grid * kc, end = (c + 1) * tiles / grid * kc;
    if (end - begin <= best) continue;
    best = end - begin;
    segments = (end - 1) / kc - begin / kc + 1;
    cross = stream_k && (begin % kc != 0 || end % kc != 0);
  }
  double const bytes = double(best) * tn * 32 * 2;
  x[0] = 1;
  x[1] = double(segments);
  x[2] = cross ? 1 : 0;
  int const r = TnDIndex(tn, d);
  if (r >= 0 && r < kTnD) x[3 + r] = bytes;
  x[3 + kTnD + OpIndex(op)] = double(b) * tn;
  return x;
}

/// Least squares by normal equations with a small ridge (Gauss-Jordan).
std::vector<double> LeastSquares(std::vector<std::vector<double>> const& rows, std::vector<double> const& y) {
  int const p = kFeatures;
  std::vector<std::vector<double>> a(p, std::vector<double>(p + 1, 0.0));
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (int r = 0; r < p; ++r) {
      for (int c = 0; c < p; ++c) a[r][c] += rows[i][r] * rows[i][c];
      a[r][p] += rows[i][r] * y[i];
    }
  for (int r = 0; r < p; ++r) a[r][r] += 1e-9 * (a[r][r] + 1);
  for (int c = 0; c < p; ++c) {
    int pivot = c;
    for (int r = c + 1; r < p; ++r)
      if (std::abs(a[r][c]) > std::abs(a[pivot][c])) pivot = r;
    std::swap(a[c], a[pivot]);
    if (std::abs(a[c][c]) < 1e-30) continue;
    for (int r = 0; r < p; ++r)
      if (r != c) {
        double const f = a[r][c] / a[c][c];
        for (int k = c; k <= p; ++k) a[r][k] -= f * a[c][k];
      }
  }
  std::vector<double> coef(p, 0.0);
  for (int r = 0; r < p; ++r) coef[r] = std::abs(a[r][r]) < 1e-30 ? 0.0 : a[r][p] / a[r][r];
  return coef;
}

}  // namespace

std::vector<MxDecodeClass> MxDecodeDomain(MxDecodeModel const& m, std::string const& cls, int batch, int grid) {
  using namespace backend::mx;
  int n, k;
  Shape(m, cls, n, k);
  std::vector<MxDecodeClass> out;
  for (int tn : {16, 32, 64, 128}) {
    if (n % tn || k % 32) continue;
    if (cls == "gate_up" && tn < 32) continue;
    if (cls == "qkv" && !((tn == m.head_dim || tn == 2 * m.head_dim) && tn <= 128 &&
                          ((m.q_per_kv + 2) * m.head_dim) % tn == 0))
      continue;
    if (!MxReduceRounds(batch, tn)) continue;
    for (int d = 1; d <= 4; ++d) {
      if (!MxDepthFits(tn, d)) continue;
      for (bool stream_k : {false, true}) {
        if (stream_k && (cls == "head" || (long long)(n / tn) * (k / 32) < grid)) continue;
        out.push_back({tn, d, stream_k});
      }
    }
  }
  return out;
}

MxDecodeSolver::MxDecodeSolver(std::string const& table) {
  auto doc = json::ParseFile(table);
  for (auto const& r : doc.At("gemm").AsArray("gemm"))
    gemm_.push_back({int(r.At("n").AsNumber("n")), int(r.At("k").AsNumber("k")), int(r.At("b").AsNumber("b")),
                     int(r.At("tn").AsNumber("tn")), int(r.At("d").AsNumber("d")), r.At("op").AsString("op"),
                     int(r.At("norm").AsNumber("norm")), r.At("part").AsString("part") == "kStreamK",
                     r.At("median_ns").AsNumber("median_ns"), int(r.At("spill_bytes").AsNumber("spill"))});
  for (auto const& r : doc.At("attention").AsArray("attention"))
    attn_.push_back({int(r.At("dim").AsNumber("dim")), int(r.At("q").AsNumber("q")), int(r.At("hkv").AsNumber("hkv")),
                     int(r.At("b").AsNumber("b")), int(r.At("length").AsNumber("length")),
                     int(r.At("tmin").AsNumber("tmin")), r.At("median_ns").AsNumber("median_ns")});
  for (auto const& r : doc.At("embed").AsArray("embed"))
    embed_.push_back({{int(r.At("hidden").AsNumber("hidden")), int(r.At("b").AsNumber("b"))},
                      r.At("median_ns").AsNumber("median_ns")});
  Fit();
}

void MxDecodeSolver::Fit() {
  std::vector<std::vector<double>> rows;
  std::vector<double> y;
  for (auto const& r : gemm_) {
    rows.push_back(Features(r.n, r.k, r.b, r.op, r.tn, r.d, r.stream_k, 104));
    y.push_back(r.ns);
  }
  if (rows.empty()) return;
  coef_ = LeastSquares(rows, y);
  double mean = 0;
  for (double v : y) mean += v / y.size();
  double ss_res = 0, ss_tot = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    double f = 0;
    for (int c = 0; c < kFeatures; ++c) f += coef_[c] * rows[i][c];
    ss_res += (y[i] - f) * (y[i] - f);
    ss_tot += (y[i] - mean) * (y[i] - mean);
    max_rel_ = std::max(max_rel_, std::abs(f - y[i]) / y[i]);
  }
  r2_ = ss_tot > 0 ? 1 - ss_res / ss_tot : 0;
}

MxDecodeCandidate MxDecodeSolver::Price(int n, int k, int b, std::string const& op, int norm, MxDecodeClass const& c,
                                        int grid) const {
  for (auto const& r : gemm_)
    if (r.n == n && r.k == k && r.b == b && r.op == op && r.norm == norm && r.tn == c.tn && r.d == c.d &&
        r.stream_k == c.stream_k)
      return {c, r.ns, true, r.spill};
  auto x = Features(n, k, b, op, c.tn, c.d, c.stream_k, grid);
  double f = 0;
  for (int i = 0; i < kFeatures && i < int(coef_.size()); ++i) f += coef_[i] * x[i];
  return {c, f, false, -1};
}

double MxDecodeSolver::Attention(MxDecodeModel const& m, int batch, int tmin, int grid, int past_lo,
                                 int past_hi) const {
  (void)grid;
  std::vector<std::pair<int, double>> points;
  for (auto const& r : attn_)
    if (r.dim == m.head_dim && r.q == m.q_per_kv && r.hkv == m.hkv && r.b == batch && r.tmin == tmin)
      points.push_back({r.length, r.ns});
  std::sort(points.begin(), points.end());
  if (points.size() < 2) throw std::invalid_argument("mx decode solver: no attention rows for this shape");
  auto at = [&](int length) {
    if (length <= points.front().first) return points.front().second;
    for (std::size_t i = 1; i < points.size(); ++i)
      if (length <= points[i].first) {
        auto const& [l0, v0] = points[i - 1];
        auto const& [l1, v1] = points[i];
        return v0 + (v1 - v0) * double(length - l0) / double(l1 - l0);
      }
    auto const& [l0, v0] = points[points.size() - 2];
    auto const& [l1, v1] = points.back();
    return v1 + (v1 - v0) * double(length - l1) / double(l1 - l0);
  };
  // The mean envelope over the plan's decode steps: L = past + 1.
  double sum = 0;
  for (int past = past_lo; past <= past_hi; ++past) sum += at(past + 1);
  return sum / double(past_hi - past_lo + 1);
}

double MxDecodeSolver::Embed(int hidden, int batch) const {
  for (auto const& [key, ns] : embed_)
    if (key.first == hidden && key.second == batch) return ns;
  return 0;
}

double MxDecodeSolver::Predict(MxDecodeModel const& m, int batch, MxDecodeConfig const& config, int grid,
                               int past_lo, int past_hi, std::map<std::string, double>* stages) const {
  std::map<std::string, double> s;
  MxDecodeClass const* chosen[] = {&config.qkv, &config.o, &config.gate_up, &config.down, &config.head};
  for (int i = 0; i < 5; ++i) {
    int n, k;
    Shape(m, kClasses[i].name, n, k);
    s[kClasses[i].name] = Price(n, k, batch, kClasses[i].op, kClasses[i].norm, *chosen[i], grid).ns;
  }
  s["attn"] = Attention(m, batch, config.tmin, grid, past_lo, past_hi);
  s["embed"] = Embed(m.hidden, batch);
  double const step = s["embed"] + s["head"] + m.layers * (s["qkv"] + s["attn"] + s["o"] + s["gate_up"] + s["down"]);
  if (stages) *stages = s;
  return step;
}

MxDecodeSelection MxDecodeSolver::Solve(MxDecodeModel const& m, int batch, int tmin, int grid, int past_lo,
                                        int past_hi) const {
  MxDecodeSelection sel;
  sel.fit_r2 = r2_;
  sel.fit_max_rel = max_rel_;
  sel.fit_rows = int(gemm_.size());
  MxDecodeClass* first[] = {&sel.first.qkv, &sel.first.o, &sel.first.gate_up, &sel.first.down, &sel.first.head};
  std::string swap;
  double swap_gap = std::numeric_limits<double>::infinity();
  MxDecodeClass swap_to{};
  for (int i = 0; i < 5; ++i) {
    int n, k;
    Shape(m, kClasses[i].name, n, k);
    auto& list = sel.candidates[kClasses[i].name];
    for (auto const& c : MxDecodeDomain(m, kClasses[i].name, batch, grid))
      list.push_back(Price(n, k, batch, kClasses[i].op, kClasses[i].norm, c, grid));
    if (list.empty()) throw std::invalid_argument(std::string("mx decode solver: empty domain for ") + kClasses[i].name);
    std::stable_sort(list.begin(), list.end(), [](auto const& a, auto const& b) { return a.ns < b.ns; });
    *first[i] = list[0].k;
    if (list.size() > 1) {
      double const gap = (list[1].ns - list[0].ns) / list[0].ns;
      if (gap < swap_gap) swap_gap = gap, swap = kClasses[i].name, swap_to = list[1].k;
    }
  }
  sel.first.tmin = tmin;
  sel.second = sel.first;
  sel.swapped = swap;
  MxDecodeClass* second[] = {&sel.second.qkv, &sel.second.o, &sel.second.gate_up, &sel.second.down,
                             &sel.second.head};
  for (int i = 0; i < 5; ++i)
    if (swap == kClasses[i].name) *second[i] = swap_to;
  sel.first_ns = Predict(m, batch, sel.first, grid, past_lo, past_hi, &sel.stage_ns);
  sel.second_ns = Predict(m, batch, sel.second, grid, past_lo, past_hi);
  return sel;
}

}  // namespace tilemega::solver
