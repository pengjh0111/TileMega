// SPDX-License-Identifier: BSD-3-Clause
#include "tilemega/Solver/MxDecodeSolverV2.h"
#include "MxDecodeCompilerProbeV2.h"

#include "tilemega/Backend/Mx/DecodeGeometryV2.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tilemega::solver {
namespace {
using codegen::MxDecodeClassV2;
using codegen::MxDecodeConfigV2;
using codegen::MxDecodeModel;
using json::Value;
using Key = std::map<std::string, int>;
using Ranks = std::array<std::size_t, 6>;
constexpr char const* kClasses[] = {"qkv", "o", "gate_up", "down", "head", "attention"};
constexpr int kGrid = 104;

[[noreturn]] void Fail(std::string const& message) {
  throw std::invalid_argument("mx decode solver v2: " + message);
}
int Integer(Value const& value, char const* name, int minimum = 0) {
  double v = value.AsNumber(name);
  if (!std::isfinite(v) || v < minimum || v > std::numeric_limits<int>::max() || std::floor(v) != v)
    Fail(std::string("invalid integer ") + name);
  return int(v);
}
int I(Value const& v, char const* name, int minimum = 0) { return Integer(v.At(name), name, minimum); }
double Positive(Value const& v, char const* name) {
  double x = v.At(name).AsNumber(name);
  if (!std::isfinite(x) || x <= 0) Fail(std::string("nonpositive/nonfinite ") + name);
  return x;
}
std::string S(Value const& v, char const* name) {
  auto s = v.At(name).AsString(name);
  if (s.empty()) Fail(std::string("empty ") + name);
  return s;
}
void Hash(Value const& v, char const* field) {
  auto s = S(v, field);
  if (s.size() != 64 || s.find_first_not_of("0123456789abcdef") != std::string::npos)
    Fail(std::string("invalid provenance hash ") + field);
}
void Identity(Value const& p, std::string const& kind) {
  if (S(p, "executor") != "mx_static_stage_bench" || S(p, "implementation") != "mx5_" + kind + "_v2")
    Fail("wrong executor/implementation provenance");
  Hash(p, "source_sha256");
  Hash(p, "binary_sha256");
  S(p, "base_head");
}
MxDecodeResourcesV2 Resources(Value const& v, int threads, bool require_fit = true) {
  MxDecodeResourcesV2 r{I(v, "mtreg", 1), I(v, "local_bytes"), I(v, "static_smem_bytes"), I(v, "resident_ctas")};
  if (require_fit && (r.mtreg > (threads == 576 ? 227 : 255) || r.local_bytes || r.static_smem_bytes || r.resident_ctas != 1))
    Fail("resource rejection (registers, stack, shared memory or resident CTAs)");
  return r;
}
bool SameResources(MxDecodeResourcesV2 const& a, MxDecodeResourcesV2 const& b) {
  return std::tie(a.mtreg, a.local_bytes, a.static_smem_bytes, a.resident_ctas) ==
         std::tie(b.mtreg, b.local_bytes, b.static_smem_bytes, b.resident_ctas);
}
bool SameIdentity(Value const& a, Value const& b, bool binary = true) {
  for (char const* field : {"executor", "implementation", "source_sha256", "binary_sha256", "base_head"}) {
    if (!binary && std::string(field) == "binary_sha256") continue;
    if (S(a, field) != S(b, field)) return false;
  }
  return true;
}
Key ReadKey(Value const& value, std::string const& kind, bool measurement = false) {
  std::vector<std::string> fields = kind == "attention"
      ? std::vector<std::string>{"dim", "q", "hkv", "b", "length", "b_min", "qk_norm", "capacity", "threads", "prefetch_kind"}
      : std::vector<std::string>{"n", "k", "b", "tn", "wn", "depth", "threads", "prefetch"};
  if (kind == "gemm") fields.insert(fields.end(), {"op", "norm", "part"});
  Key key;
  for (auto const& field : fields) {
    auto* found = value.Find(field);
    if (!found && measurement && field == "prefetch") found = value.Find("pw");
    if (!found) Fail("missing key coordinate " + field);
    key[field] = Integer(*found, field.c_str());
  }
  for (auto const& [name, unused] : value.AsObject("key")) {
    (void)unused;
    if (!key.count(name) && !(measurement && (name == "library" || name == "pw"))) Fail("unknown key coordinate " + name);
  }
  if (auto* pw = value.Find("pw"))
    if (Integer(*pw, "pw") != key.at("prefetch")) Fail("conflicting prefetch/pw");
  if (key.at("threads") != 512 && key.at("threads") != 576) Fail("invalid threads");
  if (key.at("b") != 1 && key.at("b") != 2 && key.at("b") != 4 && key.at("b") != 8 && key.at("b") != 16)
    Fail("invalid batch");
  if (kind == "attention") {
    if ((key.at("dim") != 64 && key.at("dim") != 128) || (key.at("q") != 2 && key.at("q") != 3 && key.at("q") != 4 && key.at("q") != 8)
        || key.at("hkv") < 1 || key.at("length") < 1 || key.at("length") > 1088 || key.at("capacity") != 1088
        || (key.at("b_min") != 1 && key.at("b_min") != 2 && key.at("b_min") != 4 && key.at("b_min") != 8)
        || key.at("prefetch_kind") > 2 || key.at("qk_norm") > 1 || (key.at("qk_norm") && key.at("dim") != 128))
      Fail("illegal attention key");
  } else {
    using namespace backend::mx;
    auto op = kind == "head" ? MxOpV2::kStore : static_cast<MxOpV2>(key.at("op"));
    auto part = kind == "head" ? MxPart::kTileRange : static_cast<MxPart>(key.at("part"));
    if ((kind == "gemm" && (key.at("op") > 2 || key.at("norm") > 1 || key.at("part") > 1))
        || !MxGemmLegalV2(key.at("n"), key.at("k"), key.at("b"), key.at("tn"), key.at("wn"), key.at("depth"), op, part)
        || (kind == "head" && key.at("wn") != 4 && key.at("wn") != 8)) Fail("illegal GEMM/head key");
    int pw = key.at("prefetch");
    if ((pw && pw < key.at("depth")) || (key.at("depth") - 1) * (key.at("tn") / 16 + 1) + pw * (key.at("tn") / 16) > 63)
      Fail("illegal self-prefetch/wait count");
  }
  return key;
}
void Shape(MxDecodeModel const& m, std::string const& cls, int& n, int& k, int& op, int& norm) {
  norm = cls == "qkv" || cls == "gate_up" || cls == "head";
  op = cls == "gate_up" ? 2 : cls == "o" || cls == "down" ? 1 : 0;
  if (cls == "qkv") n = (m.q_per_kv + 2) * m.hkv * m.head_dim, k = m.hidden;
  else if (cls == "o") n = m.hidden, k = m.heads * m.head_dim;
  else if (cls == "gate_up") n = 2 * m.inter, k = m.hidden;
  else if (cls == "down") n = m.hidden, k = m.inter;
  else if (cls == "head") n = m.vocab, k = m.hidden;
  else Fail("unknown class " + cls);
}
template <int TN, int WN>
int First(MxDecodeClassV2 const& c, int n, int k, int grid) {
  using namespace backend::mx;
  if (c.stream_k) {
    if constexpr (WN == 1) return MxFirstChunksV2<TN, WN, MxPart::kStreamK>(n, k, grid);
    Fail("StreamK WN must be one");
  }
  return MxFirstChunksV2<TN, WN, MxPart::kTileRange>(n, k, grid);
}
template <int TN>
int First(MxDecodeClassV2 const& c, int n, int k, int grid) {
  switch (c.wn) {
    case 1: return First<TN, 1>(c, n, k, grid);
    case 2: return First<TN, 2>(c, n, k, grid);
    case 4: return First<TN, 4>(c, n, k, grid);
    case 8: return First<TN, 8>(c, n, k, grid);
  }
  Fail("invalid WN");
}
int Width(MxDecodeClassV2 const& c, int n, int k, int threads, int grid) {
  int first = 0;
  switch (c.tn) {
    case 16: first = First<16>(c, n, k, grid); break;
    case 32: first = First<32>(c, n, k, grid); break;
    case 64: first = First<64>(c, n, k, grid); break;
    case 128: first = First<128>(c, n, k, grid); break;
    default: Fail("invalid TN");
  }
  return std::max(c.d, std::min(first, std::min(64, (threads == 576 ? 220 : 248) - c.r_next) / (c.tn / 4)));
}
MxDecodeClassV2 Class(Key const& key) {
  return {key.at("tn"), key.at("wn"), key.at("depth"), key.count("part") && key.at("part") == 0,
          key.at("prefetch"), -1};
}
void SelectedResource(Value const& row, Key const& key, std::string const& kind, int initial) {
  char const* pf = kind == "attention" ? "prefetch_kind" : "prefetch";
  if (key.at(pf) == initial) return;
  if (kind == "attention") Fail("attention PF differs from measured R_next rule");
  auto* selection = row.Find("resource_selection");
  if (!selection) Fail("reduced prefetch requires resource_selection evidence");
  int step = I(*selection, "selected_resource_step");
  if (step < 1 || step > 3) Fail("resource selection exceeds three halvings");
  auto const& attempts = selection->At("resource_attempts").AsArray("resource_attempts");
  if (attempts.size() != std::size_t(step + 1)) Fail("resource attempts must end at selected step");
  int width = initial;
  for (int i = 0; i <= step; ++i) {
    auto const& a = attempts[i];
    if (i) {
      int next = std::max(key.at("depth"), width / 2);
      if (next >= width) Fail("resource halving must decrease prefetch");
      width = next;
    }
    auto expected = key;
    expected[pf] = width;
    if (I(a, "resource_step") != i || I(a, "prefetch") != width || ReadKey(a.At("key"), kind, true) != expected
        || I(a, "query_status") != 0 || !a.At("query_only").AsBool("query_only")) Fail("invalid resource attempt coordinates/status");
    bool fits = a.At("fits").AsBool("fits");
    if (fits != (i == step)) Fail("resource attempt must select first passing build");
    auto resource = Resources(a.At("resources"), key.at("threads"), fits);
    bool actual_fit = resource.mtreg <= (key.at("threads") == 576 ? 227 : 255) && !resource.local_bytes
        && !resource.static_smem_bytes && resource.resident_ctas == 1;
    if (actual_fit != fits) Fail("resource attempt fit flag contradicts resources");
    if (!fits && resource.mtreg <= (key.at("threads") == 576 ? 227 : 255) && !resource.local_bytes)
      Fail("prefetch halving requires register or stack failure");
    Identity(a.At("provenance"), kind);
    if (!SameIdentity(a.At("provenance"), row.At("provenance"), false)) Fail("resource attempt source/base identity mismatch");
    if (i == step && (!SameResources(resource, Resources(row.At("resources"), key.at("threads")))
                     || !SameIdentity(a.At("provenance"), row.At("provenance")))) Fail("selected attempt differs from measured build");
  }
  if (width != key.at(pf)) Fail("selected prefetch is not prescribed halving");
}
bool SameClass(MxDecodeClassV2 const& a, MxDecodeClassV2 const& b) {
  return std::tie(a.tn, a.wn, a.d, a.stream_k, a.prefetch, a.r_next) ==
         std::tie(b.tn, b.wn, b.d, b.stream_k, b.prefetch, b.r_next);
}
bool CandidateLess(MxDecodeCandidateV2 const& a, MxDecodeCandidateV2 const& b) {
  return std::tie(a.ns, a.k.tn, a.k.wn, a.k.d, a.k.stream_k, a.k.prefetch, a.b_min, a.attn_prefetch) <
         std::tie(b.ns, b.k.tn, b.k.wn, b.k.d, b.k.stream_k, b.k.prefetch, b.b_min, b.attn_prefetch);
}
struct MergeNode { double cost; Ranks ranks; };
struct MergeGreater {
  bool operator()(MergeNode const& a, MergeNode const& b) const {
    return a.cost != b.cost ? a.cost > b.cost : a.ranks > b.ranks;
  }
};
std::vector<Ranks> Merge(std::array<std::vector<double>, 6> const& costs, std::size_t count,
                         std::function<bool(Ranks const&)> accept) {
  for (auto const& list : costs) {
    if (list.empty()) Fail("empty k-best list");
    for (std::size_t j = 0; j < list.size(); ++j)
      if (!std::isfinite(list[j]) || list[j] < 0 || (j && list[j] < list[j - 1])) Fail("invalid/unsorted k-best costs");
  }
  auto sum = [&](Ranks const& r) { double x = 0; for (int i = 0; i < 6; ++i) x += costs[i][r[i]]; return x; };
  std::priority_queue<MergeNode, std::vector<MergeNode>, MergeGreater> queue;
  std::set<Ranks> seen;
  Ranks first{};
  queue.push({sum(first), first});
  seen.insert(first);
  std::vector<Ranks> out;
  while (!queue.empty() && out.size() < count) {
    auto node = queue.top(); queue.pop();
    if (accept(node.ranks)) out.push_back(node.ranks);
    for (int i = 0; i < 6; ++i) {
      auto next = node.ranks;
      if (++next[i] < costs[i].size() && seen.insert(next).second) queue.push({sum(next), next});
    }
  }
  return out;
}
bool Waits(MxDecodeConfigV2 const& c, int dim) {
  auto loads = [](MxDecodeClassV2 const& x) { return x.prefetch * (x.tn / 16); };
  auto wait = [](MxDecodeClassV2 const& x, int next) { return (x.d - 1) * (x.tn / 16 + 1) + next <= 63; };
  return wait(c.qkv, c.attn_prefetch * (dim / 16)) && wait(c.o, loads(c.gate_up))
      && wait(c.gate_up, loads(c.down)) && wait(c.down, loads(c.qkv)) && wait(c.down, loads(c.head));
}
void Arguments(MxDecodeModel const& m, int batch, int threads, int grid, int lo, int hi) {
  if (grid != kGrid || (threads != 512 && threads != 576) || m.layers < 1 || m.capacity != 1088
      || lo < 64 || hi > 1087 || hi < lo || (batch != 1 && batch != 2 && batch != 4 && batch != 8 && batch != 16))
    Fail("invalid model, batch, threads, grid or attention length range");
}
// The MX-4 §4.3 feature basis is retained for table-external diagnostics.
constexpr int kFeatures = 22;
std::vector<double> Features(int n, int k, int b, int op, MxDecodeClassV2 const& c, int grid) {
  std::vector<double> x(kFeatures, 0);
  long long kc = k / 32, tiles = n / c.tn, units = tiles * kc, best = 0, segments = 0;
  bool cross = false;
  for (long long cta = 0; cta < grid; ++cta) {
    long long begin = c.stream_k ? cta * units / grid : cta * tiles / grid * kc;
    long long end = c.stream_k ? (cta + 1) * units / grid : (cta + 1) * tiles / grid * kc;
    if (end - begin <= best) continue;
    best = end - begin; segments = (end - 1) / kc - begin / kc + 1;
    cross = c.stream_k && (begin % kc || end % kc);
  }
  int t = c.tn == 16 ? 0 : c.tn == 32 ? 1 : c.tn == 64 ? 2 : 3;
  int rate = t * 4 + c.d - 1;
  x[0] = 1; x[1] = double(segments); x[2] = cross;
  if (rate < 15) x[3 + rate] = double(best) * c.tn * 32 * 2;
  x[18 + op] = double(b) * c.tn;
  return x;
}
std::vector<double> LeastSquares(std::vector<std::vector<double>> const& rows, std::vector<double> const& y) {
  std::vector<std::vector<double>> a(kFeatures, std::vector<double>(kFeatures + 1));
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (int r = 0; r < kFeatures; ++r) {
      for (int c = 0; c < kFeatures; ++c) a[r][c] += rows[i][r] * rows[i][c];
      a[r][kFeatures] += rows[i][r] * y[i];
    }
  for (int r = 0; r < kFeatures; ++r) a[r][r] += 1e-9 * (a[r][r] + 1);
  for (int c = 0; c < kFeatures; ++c) {
    int pivot = c;
    for (int r = c + 1; r < kFeatures; ++r) if (std::abs(a[r][c]) > std::abs(a[pivot][c])) pivot = r;
    std::swap(a[c], a[pivot]);
    if (std::abs(a[c][c]) < 1e-30) continue;
    for (int r = 0; r < kFeatures; ++r) if (r != c) {
      double f = a[r][c] / a[c][c];
      for (int j = c; j <= kFeatures; ++j) a[r][j] -= f * a[c][j];
    }
  }
  std::vector<double> coef(kFeatures);
  for (int r = 0; r < kFeatures; ++r) if (std::abs(a[r][r]) >= 1e-30) coef[r] = a[r][kFeatures] / a[r][r];
  return coef;
}
double Dot(std::vector<double> const& a, std::vector<double> const& b) {
  double sum = 0; for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i]; return sum;
}
}  // namespace

std::vector<Ranks> MxDecodeKBestV2(std::array<std::vector<double>, 6> const& costs, std::size_t count) {
  return Merge(costs, count, [](Ranks const&) { return true; });
}

std::vector<MxDecodeClassV2> MxDecodeDomainV2(MxDecodeModel const& model, std::string const& cls, int batch, int grid) {
  using namespace backend::mx;
  if (grid != kGrid) Fail("grid must be 104");
  int n, k, op, norm; Shape(model, cls, n, k, op, norm);
  std::vector<MxDecodeClassV2> out;
  for (int tn : {16, 32, 64, 128}) for (int wn : {1, 2, 4, 8}) for (int d = 1; d <= 4; ++d)
    for (bool stream : {false, true}) {
      if (cls == "head" && (stream || (wn != 4 && wn != 8))) continue;
      if (MxGemmLegalV2(n, k, batch, tn, wn, d, static_cast<MxOpV2>(op), stream ? MxPart::kStreamK : MxPart::kTileRange, grid))
        out.push_back({tn, wn, d, stream, -1, -1});
    }
  if (out.empty()) Fail("empty legal domain for " + cls);
  return out;
}

MxDecodeSolverV2::MxDecodeSolverV2(std::string const& table) : MxDecodeSolverV2(json::ParseFile(table)) {}
MxDecodeSolverV2::MxDecodeSolverV2(Value const& document) {
  int schema = I(document, "schema_version");
  outcomes_v3_ = schema == 3;
  if ((schema != 2 && schema != 3) || S(document, "table_kind") != "mx5_decode_stage_costs"
      || !document.At("qualified").AsBool("qualified")) Fail("requires qualified schema-v2 stage costs");
  S(document, "qualification_scope");
  if (outcomes_v3_) {
    if (S(document,"qualification_scope")!="complete_outcomes_single_request_v1"
        || !document.At("outcome_domain_complete").AsBool("outcome_domain_complete")
        || document.At("full_table").AsBool("full_table")
        || document.At("execution_512_approved").AsBool("execution_512_approved")
        || document.At("global_dt576_policy_approved").AsBool("global_dt576_policy_approved")
        || document.At("target_defaults_changed").AsBool("target_defaults_changed")) Fail("invalid schema3 qualification/policy claims");
    scope_=document.At("scope"); coverage_=document.At("coverage");
  }
  if (!document.At("diagnostic_reasons").AsArray("diagnostic_reasons").empty()) Fail("table has diagnostic reasons");
  document.At("provenance").AsObject("table provenance");
  std::set<std::pair<std::string, Key>> seen;
  for (auto const& v : document.At("rows").AsArray("rows")) {
    Row r;
    r.kind = S(v, "kind");
    if (r.kind != "gemm" && r.kind != "head" && r.kind != "attention") Fail("unknown row kind");
    r.key = ReadKey(v.At("key"), r.kind);
    if (!seen.insert({r.kind, r.key}).second) Fail("duplicate measured key");
    r.ns = Positive(v, "median_ns"); r.p90 = Positive(v, "p90_ns");
    if (r.p90 < r.ns || I(v, "processes") != 3) Fail("invalid p90/process count");
    int threads = r.key.at("threads");
    r.resources = Resources(v.At("resources"), threads);
    Identity(v.At("provenance"), r.kind);
    r.r_next = I(v, "r_next", 1);
    auto const& reference = v.At("r_next_reference");
    auto expected = r.key;
    expected[r.kind == "attention" ? "prefetch_kind" : "prefetch"] = 0;
    if (S(reference, "kind") != r.kind || ReadKey(reference.At("key"), r.kind) != expected)
      Fail("R_next reference does not match exact zero-prefetch body");
    bool const compiler_probe = reference.Find("compiler_probe_only") != nullptr;
    if(outcomes_v3_)compiler_probe_v2::Schema3LegacyReference(reference);
    if (compiler_probe) compiler_probe_v2::Validate(reference, r.kind, threads, false, outcomes_v3_);
    auto rr = Resources(reference.At("resources"), threads, !compiler_probe);
    if (rr.mtreg != r.r_next) Fail("R_next differs from reference MTreg");
    Identity(reference.At("provenance"), r.kind);
    for (char const* field : {"executor", "implementation", "source_sha256"})
      if (S(reference.At("provenance"), field) != S(v.At("provenance"), field)) Fail("R_next provenance identity mismatch");
    auto const& rounds = v.At("rounds").AsArray("rounds");
    if (rounds.size() != 3) Fail("requires three measured rounds");
    std::set<int> numbers;
    std::set<std::string> paths;
    std::vector<double> medians, p90s;
    for (auto const& round : rounds) {
      int id = I(round, "round", 1);
      if (id > 3 || !numbers.insert(id).second) Fail("invalid/duplicate round");
      auto const& input = round.At("input");
      if (!paths.insert(S(input, "path")).second) Fail("rounds must use distinct process input files");
      Hash(input, "sha256"); I(input, "line", 1);
      auto const& m = round.At("measurement");
      if (I(m, "round") != id || ReadKey(m.At("key"), r.kind, true) != r.key) Fail("round coordinates mismatch");
      for (auto const& [name, coordinate] : r.key)
        if (auto* field = m.Find(name)) if (Integer(*field, name.c_str()) != coordinate) Fail("round coordinate alias mismatch");
      Identity(m, r.kind);
      for (char const* field : {"executor", "implementation", "source_sha256", "binary_sha256", "base_head"})
        if (S(m, field) != S(v.At("provenance"), field)) Fail("round provenance mismatch");
      auto mr = Resources(m, threads);
      if (mr.mtreg != r.resources.mtreg || mr.local_bytes != r.resources.local_bytes || mr.static_smem_bytes != r.resources.static_smem_bytes
          || mr.resident_ctas != r.resources.resident_ctas) Fail("round resources mismatch");
      if (I(m, "iterations", 1) <= I(m, "copies", 1)) Fail("iterations must visit all copies");
      auto* working_set = m.Find("active_working_set_bytes");
      if (!working_set) working_set = m.Find("working_set_bytes");
      if (!working_set || !std::isfinite(working_set->AsNumber("working_set_bytes"))
          || working_set->AsNumber("working_set_bytes") < 256.0 * 1024 * 1024) Fail("working set below 256 MiB");
      medians.push_back(Positive(m, "median_ns")); p90s.push_back(Positive(m, "p90_ns"));
      if (p90s.back() < medians.back()) Fail("round p90 below median");
    }
    std::sort(medians.begin(), medians.end()); std::sort(p90s.begin(), p90s.end());
    if (medians[1] != r.ns || p90s[1] != r.p90) Fail("aggregate differs from process medians");
    double spread = v.At("spread").AsNumber("spread");
    if (!std::isfinite(spread) || std::abs(spread - (medians[2] - medians[0]) / r.ns) > 0.000050001) Fail("aggregate spread mismatch");
    int initial;
    if (r.kind == "attention") {
      int budget = std::min(64, (threads == 576 ? 220 : 248) - r.r_next);
      initial = budget >= 64 ? 2 : budget >= 32 ? 1 : 0;
    } else {
      auto c = Class(r.key); c.r_next = r.r_next;
      initial = Width(c, r.key.at("n"), r.key.at("k"), threads, kGrid);
    }
    SelectedResource(v, r.key, r.kind, initial);
    r.evidence = v; rows_.push_back(std::move(r));
  }
  if (outcomes_v3_) for (auto const& v:document.At("resource_rejections").AsArray("resource_rejections")) {
    if(S(v,"disposition")!="primary_resource_rejected")Fail("invalid rejection disposition");
    for(char const* field:{"median_ns","p90_ns","rounds","processes","spread","ns"})
      if(v.Find(field))Fail("resource rejection must not contain costs");
    Rejection r;r.kind=S(v,"kind");
    if(r.kind!="gemm" && r.kind!="head" && r.kind!="attention")Fail("invalid rejection kind");
    r.key=ReadKey(v.At("key"),r.kind);int threads=r.key.at("threads");
    compiler_probe_v2::Validate(v,r.kind,threads,true);Identity(v.At("provenance"),r.kind);
    r.resources=Resources(v.At("resources"),threads,false);r.r_next=I(v,"r_next",1);
    auto const& reference=v.At("r_next_reference");auto expected=r.key;
    expected[r.kind=="attention"?"prefetch_kind":"prefetch"]=0;
    if(S(reference,"kind")!=r.kind || ReadKey(reference.At("key"),r.kind)!=expected)Fail("rejection R_next coordinates differ");
    bool probe=reference.Find("compiler_probe_only")!=nullptr;
    if(!probe)Fail("primary rejection requires modern witnessed P0 reference");
    if(probe)compiler_probe_v2::Validate(reference,r.kind,threads,false,true);
    auto rr=Resources(reference.At("resources"),threads,!probe);
    if(rr.mtreg!=r.r_next)Fail("rejection R_next differs from measured MTreg");
    Identity(reference.At("provenance"),r.kind);
    for(char const* field:{"executor","implementation","source_sha256"})
      if(S(reference.At("provenance"),field)!=S(v.At("provenance"),field))Fail("rejection R_next source differs");
    int budget=std::min(64,(threads==576?220:248)-r.r_next);
    auto c=Class(r.kind=="attention"?Key{{"tn",16},{"wn",1},{"depth",1},{"prefetch",0}}:r.key);c.r_next=r.r_next;
    int primary=r.kind=="attention"?(budget>=64?2:budget>=32?1:0):Width(c,r.key.at("n"),r.key.at("k"),threads,kGrid);
    if(r.key.at(r.kind=="attention"?"prefetch_kind":"prefetch")!=primary)Fail("rejection does not use prescribed primary width");
    r.evidence=v;rejected_.push_back(std::move(r));
  }
  if (outcomes_v3_ && document.At("measured_domain_complete").AsBool("measured_domain_complete")!=rejected_.empty())
    Fail("schema3 measured completeness contradicts rejections");
  if (rows_.empty() && !outcomes_v3_) Fail("empty measured table");
  Fit();
}


namespace {
Key BaseKey(Key key,std::string const& kind) {key.erase(kind=="attention"?"prefetch_kind":"prefetch");return key;}
}
bool MxDecodeSolverV2::Rejected(std::string const& kind,Key const& key) const {
  auto base=BaseKey(key,kind);
  return std::any_of(rejected_.begin(),rejected_.end(),[&](auto const& r){return r.kind==kind && BaseKey(r.key,kind)==base;});
}
json::Value MxDecodeSolverV2::ValidateScope(MxDecodeModel const& m,int batch,int threads,int grid,int lo,int hi) const {
  std::set<std::string> fields{"model_id","model","batch","threads","grid","past_lo","past_hi","source_sha256"};
  if(scope_.AsObject("scope").size()!=fields.size())Fail("schema3 scope fields differ");
  for(auto const& [field,unused]:scope_.AsObject("scope")) { (void)unused;if(!fields.count(field))Fail("unknown scope field"); }
  S(scope_,"model_id");auto const& model=scope_.At("model");
  std::map<std::string,int> expected_model{{"layers",m.layers},{"hidden",m.hidden},{"heads",m.heads},{"hkv",m.hkv},
      {"head_dim",m.head_dim},{"q_per_kv",m.q_per_kv},{"inter",m.inter},{"vocab",m.vocab},{"capacity",m.capacity}};
  if(model.AsObject("model").size()!=expected_model.size()+1)Fail("scope model fields differ");
  for(auto const& [field,value]:expected_model)if(I(model,field.c_str(),1)!=value)Fail("scope model descriptor mismatch: "+field);
  if(model.At("qk_norm").AsBool("qk_norm")!=m.qk_norm || I(scope_,"batch")!=batch || I(scope_,"threads")!=threads
     || I(scope_,"grid")!=grid || I(scope_,"past_lo")!=lo || I(scope_,"past_hi")!=hi || lo!=64 || hi!=1086)Fail("scope request mismatch");
  auto const& sources=scope_.At("source_sha256");if(sources.AsObject("source_sha256").size()!=3)Fail("scope requires three stage sources");
  for(char const* kind:{"gemm","head","attention"})Hash(sources,kind);
  using Slot=std::pair<std::string,Key>;std::set<Slot> expected,seen;
  for(int i=0;i<5;++i) {
    std::string cls=kClasses[i],kind=cls=="head"?"head":"gemm";int n,k,op,norm;Shape(m,cls,n,k,op,norm);
    for(auto const& c:MxDecodeDomainV2(m,cls,batch,grid)) {
      Key key{{"n",n},{"k",k},{"b",batch},{"threads",threads},{"tn",c.tn},{"wn",c.wn},{"depth",c.d}};
      if(kind=="gemm")key.insert({{"op",op},{"norm",norm},{"part",c.stream_k?0:1}});
      expected.insert({kind,key});
    }
  }
  for(int bmin:{1,2,4,8})for(int length:{65,256,512,768,1088})
    expected.insert({"attention",Key{{"dim",m.head_dim},{"q",m.q_per_kv},{"hkv",m.hkv},{"b",batch},{"length",length},
      {"b_min",bmin},{"qk_norm",int(m.qk_norm)},{"capacity",m.capacity},{"threads",threads}}});
  auto check_outcomes=[&](auto const& list) {for(auto const& r:list) {
    Slot slot{r.kind,BaseKey(r.key,r.kind)};
    if(!expected.count(slot))Fail("outcome outside exact requested model/B/threads domain");
    if(!seen.insert(slot).second)Fail("duplicate or contradictory cost/rejection outcome");
    if(S(r.evidence.At("provenance"),"source_sha256")!=S(sources,r.kind.c_str()))Fail("outcome source differs from scope");
  }};check_outcomes(rows_);check_outcomes(rejected_);
  if(seen!=expected)Fail("incomplete schema3 outcomes; missing="+std::to_string(expected.size()-seen.size()));
  if(I(coverage_,"expected_keys")!=int(expected.size()) || I(coverage_,"costed_keys")!=int(rows_.size())
     || I(coverage_,"rejected_keys")!=int(rejected_.size()) || I(coverage_,"missing_keys")!=0)Fail("coverage counts disagree with actual outcomes");
  int modern=0,legacy=0;
  for(auto const& r:rows_)r.evidence.At("r_next_reference").Find("compiler_probe_only")?++modern:++legacy;
  modern+=int(rejected_.size());
  json::Array rejections;for(auto const& r:rejected_)rejections.push_back(r.evidence);
  return json::Object{{"scope",scope_},{"coverage",coverage_},{"resource_rejections",rejections},
    {"r_next_policy_audit",json::Object{{"modern_normal_checked",modern},{"legacy_cost_only_unverifiable",legacy}}},
    {"ranking_claim","no_top4_claim"},{"execution_512_approved",false},
    {"fit_status",fit_rows_?"measured_cost_rows_only":"no_measured_rows"},{"final",false},{"default_enabled",false}};
}

std::vector<MxDecodeCandidateV2> MxDecodeSolverV2::Candidates(MxDecodeModel const& m, std::string const& cls,
                                                            int batch, int threads, int grid) const {
  int n, k, op, norm; Shape(m, cls, n, k, op, norm);
  std::string kind = cls == "head" ? "head" : "gemm";
  std::vector<MxDecodeCandidateV2> out;
  for (auto c : MxDecodeDomainV2(m, cls, batch, grid)) {
    Row const* match = nullptr;
    for (auto const& r : rows_) {
      auto const& key = r.key;
      if (r.kind != kind || key.at("n") != n || key.at("k") != k || key.at("b") != batch || key.at("threads") != threads
          || key.at("tn") != c.tn || key.at("wn") != c.wn || key.at("depth") != c.d) continue;
      if (kind == "gemm" && (key.at("op") != op || key.at("norm") != norm || (key.at("part") == 0) != c.stream_k)) continue;
      if (match) Fail("ambiguous selected prefetch rows for " + cls);
      match = &r;
    }
    if (!match && outcomes_v3_) {
      Key key{{"n",n},{"k",k},{"b",batch},{"threads",threads},{"tn",c.tn},{"wn",c.wn},{"depth",c.d},{"prefetch",0}};
      if(kind=="gemm")key.insert({{"op",op},{"norm",norm},{"part",c.stream_k?0:1}});
      if(Rejected(kind,key))continue;
    }
    if (!match) Fail("missing qualified legal coordinate for " + cls + " N=" + std::to_string(n) + " K=" + std::to_string(k)
                     + " B=" + std::to_string(batch) + " TN=" + std::to_string(c.tn) + " WN=" + std::to_string(c.wn)
                     + " D=" + std::to_string(c.d) + " Part=" + (c.stream_k ? "StreamK" : "TileRange")
                     + "; a partial covered domain cannot use the model");
    c.prefetch = match->key.at("prefetch"); c.r_next = match->r_next;
    MxDecodeCandidateV2 candidate;
    candidate.k = c; candidate.ns = match->ns; candidate.p90_ns = match->p90;
    candidate.r_next = match->r_next; candidate.resources = match->resources; candidate.evidence = {match->evidence};
    out.push_back(std::move(candidate));
  }
  std::sort(out.begin(), out.end(), CandidateLess);
  return out;
}

std::vector<MxDecodeCandidateV2> MxDecodeSolverV2::Attention(MxDecodeModel const& m, int batch, int threads, int lo, int hi) const {
  std::vector<MxDecodeCandidateV2> out;
  for (int bmin : {1, 2, 4, 8}) {
    std::vector<Row const*> points;
    bool executable=true;
    for (int length : {65, 256, 512, 768, 1088}) {
      Row const* match = nullptr;
      for (auto const& r : rows_) {
        if (r.kind != "attention") continue;
        auto const& k = r.key;
        if (k.at("dim") != m.head_dim || k.at("q") != m.q_per_kv || k.at("hkv") != m.hkv || k.at("b") != batch
            || k.at("qk_norm") != int(m.qk_norm) || k.at("capacity") != m.capacity || k.at("threads") != threads
            || k.at("b_min") != bmin || k.at("length") != length) continue;
        if (match) Fail("ambiguous attention prefetch rows");
        match = &r;
      }
      if (!match && outcomes_v3_) {
        Key key{{"dim",m.head_dim},{"q",m.q_per_kv},{"hkv",m.hkv},{"b",batch},{"qk_norm",int(m.qk_norm)},
                {"capacity",m.capacity},{"threads",threads},{"b_min",bmin},{"length",length},{"prefetch_kind",0}};
        if(Rejected("attention",key)) {executable=false;continue;}
      }
      if (!match) Fail("missing qualified attention grid coordinate b_min=" + std::to_string(bmin) + " L=" + std::to_string(length));
      points.push_back(match);
    }
    if(!executable)continue;
    MxDecodeCandidateV2 candidate;
    candidate.b_min = bmin; candidate.attn_prefetch = points[0]->key.at("prefetch_kind");
    candidate.r_next = points[0]->r_next; candidate.resources = points[0]->resources;
    for (auto* r : points) {
      if (r->key.at("prefetch_kind") != candidate.attn_prefetch || r->r_next != candidate.r_next
          || !SameResources(r->resources, candidate.resources)
          || !SameIdentity(r->evidence.At("provenance"), points[0]->evidence.At("provenance")))
        Fail("attention runtime length grid changes compiled PF/R_next/resources/provenance");
      candidate.evidence.push_back(r->evidence);
    }
    for (int past = lo; past <= hi; ++past) {
      int length = past + 1;
      std::size_t i = 1;
      while (i + 1 < points.size() && length > points[i]->key.at("length")) ++i;
      auto* a = points[i - 1]; auto* b = points[i];
      double f = double(length - a->key.at("length")) / (b->key.at("length") - a->key.at("length"));
      candidate.ns += a->ns + f * (b->ns - a->ns);
      candidate.p90_ns += a->p90 + f * (b->p90 - a->p90);
    }
    candidate.ns /= hi - lo + 1; candidate.p90_ns /= hi - lo + 1;
    out.push_back(std::move(candidate));
  }
  std::sort(out.begin(), out.end(), CandidateLess);
  return out;
}

MxDecodeSelectionV2 MxDecodeSolverV2::Solve(MxDecodeModel const& m, int batch, int threads, int grid, int lo, int hi) const {
  if(outcomes_v3_)Fail("schema3 requires explicit SolveQualifiedDomain; production route is not connected");
  return SolveCore(m,batch,threads,grid,lo,hi);
}
MxDecodeSelectionV2 MxDecodeSolverV2::SolveQualifiedDomain(MxDecodeModel const& m,int batch,int threads,int grid,int lo,int hi) const {
  if(!outcomes_v3_)Fail("SolveQualifiedDomain requires explicit schema3 outcomes");
  return SolveCore(m,batch,threads,grid,lo,hi);
}
MxDecodeSelectionV2 MxDecodeSolverV2::SolveCore(MxDecodeModel const& m,int batch,int threads,int grid,int lo,int hi) const {
  Arguments(m, batch, threads, grid, lo, hi);
  MxDecodeSelectionV2 result;
  result.fit_r2 = fit_r2_; result.fit_rows = fit_rows_;
  if(outcomes_v3_)result.domain_report=ValidateScope(m,batch,threads,grid,lo,hi);
  for (int i = 0; i < 5; ++i) result.candidates[kClasses[i]] = Candidates(m, kClasses[i], batch, threads, grid);
  result.candidates["attention"] = Attention(m, batch, threads, lo, hi);
  if(outcomes_v3_) {
    json::Array empty;json::Object counts;
    for(char const* cls:kClasses) {
      auto size=result.candidates.at(cls).size();counts.emplace_back(cls,int(size));if(!size)empty.emplace_back(cls);
    }
    result.domain_report.Set("executable_candidates_by_class",counts);
    if(!empty.empty()) {
      result.status="no_executable_plan";result.domain_report.Set("empty_classes",empty);
      result.domain_report.Set("required_review","D7 and per-plan 512 eligibility; no fallback authorized");return result;
    }
  }
  std::array<std::vector<double>, 6> costs;
  for (int i = 0; i < 6; ++i)
    for (auto const& candidate : result.candidates.at(kClasses[i])) costs[i].push_back(candidate.ns * (i == 4 ? 1 : m.layers));
  auto prediction = [&](Ranks const& ranks) {
    MxDecodePredictionV2 p; p.ranks = ranks; p.config.threads = threads;
    MxDecodeClassV2* classes[] = {&p.config.qkv, &p.config.o, &p.config.gate_up, &p.config.down, &p.config.head};
    for (int i = 0; i < 6; ++i) {
      auto const& c = result.candidates.at(kClasses[i])[ranks[i]];
      if (i < 5) *classes[i] = c.k;
      else p.config.b_min = c.b_min, p.config.attn_prefetch = c.attn_prefetch;
      p.stage_ns[kClasses[i]] = c.ns;
      p.tpot_ns += costs[i][ranks[i]];
    }
    return p;
  };
  auto ranks = Merge(costs, 4, [&](Ranks const& r) { return Waits(prediction(r).config, m.head_dim); });
  for (auto const& r : ranks) result.top.push_back(prediction(r));
  if(outcomes_v3_) {
    result.status=result.top.empty()?"no_executable_plan":result.top.size()<4?"insufficient_top4":"predicted_top4_resource_review_required";
    if(result.top.size()==4)result.domain_report.Set("ranking_claim","exact_top4_of_certified_executable_stage_domain");
    if(result.top.size()!=4)result.domain_report.Set("required_review","complete executable top4 unavailable; no fallback authorized");
    return result;
  }
  if (result.top.size() != 4) Fail("fewer than four qualified combinations satisfy next-stage wait counts");
  return result;
}

MxDecodePredictionV2 MxDecodeSolverV2::Predict(MxDecodeModel const& m, int batch, MxDecodeConfigV2 const& config,
                                              int grid, int lo, int hi) const {
  if(outcomes_v3_)Fail("schema3 Predict requires a separately reviewed lineage-aware interface");
  Arguments(m, batch, config.threads, grid, lo, hi);
  if (!Waits(config, m.head_dim)) Fail("configuration violates next-stage wait counts");
  MxDecodePredictionV2 prediction; prediction.config = config;
  MxDecodeClassV2 const* classes[] = {&config.qkv, &config.o, &config.gate_up, &config.down, &config.head};
  for (int i = 0; i < 6; ++i) {
    auto list = i < 5 ? Candidates(m, kClasses[i], batch, config.threads, grid) : Attention(m, batch, config.threads, lo, hi);
    auto found = std::find_if(list.begin(), list.end(), [&](MxDecodeCandidateV2 const& c) {
      return i < 5 ? SameClass(c.k, *classes[i]) : c.b_min == config.b_min && c.attn_prefetch == config.attn_prefetch;
    });
    if (found == list.end()) Fail("configuration lacks qualified candidate evidence");
    prediction.ranks[i] = std::size_t(found - list.begin()); prediction.stage_ns[kClasses[i]] = found->ns;
    prediction.tpot_ns += found->ns * (i == 4 ? 1 : m.layers);
  }
  return prediction;
}

MxDecodeMeasuredSelectionV2 SelectMxDecodeMeasuredV2(MxDecodeSelectionV2 const& selection,
                                                     std::vector<std::array<double, 2>> const& measurements) {
  if (selection.top.size() != 4 || measurements.size() != 4) Fail("selection requires exactly four plans with two process measurements each");
  MxDecodeMeasuredSelectionV2 out;
  std::size_t fastest = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    if (!std::isfinite(selection.top[i].tpot_ns) || selection.top[i].tpot_ns <= 0) Fail("invalid predicted TPOT");
    for (double x : measurements[i]) if (!std::isfinite(x) || x <= 0) Fail("invalid measured TPOT");
    double ns = measurements[i][0] / 2 + measurements[i][1] / 2;
    out.measured_ns.push_back(ns); out.pred_over_meas.push_back(selection.top[i].tpot_ns / ns);
    if (ns < out.measured_ns[fastest]) fastest = i;
  }
  if (out.measured_ns[fastest] <= out.measured_ns[0] * 0.99) out.selected = fastest;
  return out;
}

void MxDecodeSolverV2::Fit() {
  std::array<std::vector<std::vector<double>>, 2> x;
  std::array<std::vector<double>, 2> y;
  for (auto const& r : rows_) if (r.kind != "attention") {
    int idx = r.key.at("threads") == 576;
    int op = r.kind == "head" ? 3 : r.key.at("op");
    auto c = Class(r.key);
    x[idx].push_back(Features(r.key.at("n"), r.key.at("k"), r.key.at("b"), op, c, kGrid));
    y[idx].push_back(r.ns);
  }
  double residual = 0, total = 0;
  for (int t = 0; t < 2; ++t) {
    if (x[t].empty()) continue;
    coefficients_[t] = LeastSquares(x[t], y[t]);
    double mean = 0; for (double value : y[t]) mean += value / y[t].size();
    for (std::size_t i = 0; i < x[t].size(); ++i) {
      double error = y[t][i] - Dot(x[t][i], coefficients_[t]);
      residual += error * error; total += (y[t][i] - mean) * (y[t][i] - mean);
    }
    fit_rows_ += int(x[t].size());
  }
  fit_r2_ = total > 0 ? 1 - residual / total : 0;
}

double MxDecodeSolverV2::EstimateUncovered(MxDecodeModel const& m, std::string const& cls, int batch,
                                          MxDecodeClassV2 const& candidate, int threads, int grid) const {
  if (grid != kGrid || (threads != 512 && threads != 576)) Fail("invalid estimate threads/grid");
  int n, k, op, norm; Shape(m, cls, n, k, op, norm);
  std::string kind = cls == "head" ? "head" : "gemm";
  for (auto const& r : rows_) if (r.kind == kind && r.key.at("n") == n && r.key.at("k") == k && r.key.at("b") == batch
                                 && r.key.at("threads") == threads
                                 && (kind == "head" || (r.key.at("op") == op && r.key.at("norm") == norm)))
    Fail("covered shape cannot use least-squares fallback");
  for(auto const& r:rejected_)if(r.kind==kind && r.key.at("n")==n && r.key.at("k")==k && r.key.at("b")==batch
      && r.key.at("threads")==threads && (kind=="head" || (r.key.at("op")==op && r.key.at("norm")==norm)))
    Fail("rejected covered shape cannot use least-squares fallback");
  auto domain = MxDecodeDomainV2(m, cls, batch, grid);
  if (std::none_of(domain.begin(), domain.end(), [&](MxDecodeClassV2 const& c) {
    return std::tie(c.tn, c.wn, c.d, c.stream_k) == std::tie(candidate.tn, candidate.wn, candidate.d, candidate.stream_k);
  })) Fail("estimate candidate outside legal domain");
  auto const& coef = coefficients_[threads == 576];
  if (coef.empty()) Fail("no least-squares rows for requested threads");
  double ns = Dot(Features(n, k, batch, cls == "head" ? 3 : op, candidate, grid), coef);
  if (!std::isfinite(ns) || ns <= 0) Fail("least-squares estimate is not positive/finite");
  return ns;
}
}  // namespace tilemega::solver
