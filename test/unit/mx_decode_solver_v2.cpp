// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxDecodeSolverV2.h>
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace tilemega;
using namespace tilemega::solver;
using json::Value;
using codegen::MxDecodeModel;
namespace {
Value Obj(std::initializer_list<std::pair<std::string, Value>> fields) { return Value(json::Object(fields)); }
Value Identity(std::string const& kind, bool reference = false) {
  return Obj({{"executor", "mx_static_stage_bench"}, {"implementation", "mx5_" + kind + "_v2"},
              {"source_sha256", std::string(64, 'a')}, {"binary_sha256", std::string(64, reference ? 'b' : 'c')}, {"base_head", "fixture"}});
}
Value Resource(int mtreg = 200) { return Obj({{"mtreg", mtreg}, {"local_bytes", 0}, {"static_smem_bytes", 0}, {"resident_ctas", 1}}); }
void Copy(Value& to, Value const& from) { for (auto const& [key, value] : from.AsObject("object")) to.Set(key, value); }
int Number(Value const& v, char const* key) { return int(v.At(key).AsNumber(key)); }
Value Row(std::string const& kind, Value key, double ns, int rnext = 160) {
  auto identity = Identity(kind), res = Resource();
  auto refkey = key;
  refkey.Set(kind == "attention" ? "prefetch_kind" : "prefetch", 0);
  auto reference = Obj({{"kind", kind}, {"key", refkey}, {"resources", Resource(rnext)}, {"provenance", Identity(kind, true)}});
  json::Array rounds;
  for (int round = 1; round <= 3; ++round) {
    auto measurement = Obj({{"key", key}, {"round", round}, {"median_ns", ns}, {"p90_ns", ns + 1},
                            {"iterations", 5}, {"copies", 2}, {"active_working_set_bytes", 268435456}});
    Copy(measurement, identity); Copy(measurement, res);
    rounds.push_back(Obj({{"round", round}, {"measurement", measurement},
                         {"input", Obj({{"path", "process" + std::to_string(round)}, {"sha256", std::string(64, 'd')}, {"line", 1}})}}));
  }
  return Obj({{"kind", kind}, {"key", key}, {"median_ns", ns}, {"p90_ns", ns + 1}, {"spread", 0}, {"processes", 3},
              {"resources", res}, {"provenance", identity}, {"rounds", Value(rounds)}, {"r_next", rnext}, {"r_next_reference", reference}});
}
template <int TN, int WN>
int First(codegen::MxDecodeClassV2 const& c, int n, int k) {
  using namespace backend::mx;
  if (c.stream_k) { if constexpr (WN == 1) return MxFirstChunksV2<TN, WN, MxPart::kStreamK>(n, k, 104); }
  return MxFirstChunksV2<TN, WN, MxPart::kTileRange>(n, k, 104);
}
template <int TN>
int First(codegen::MxDecodeClassV2 const& c, int n, int k) {
  if (c.wn == 1) return First<TN, 1>(c, n, k);
  if (c.wn == 2) return First<TN, 2>(c, n, k);
  if (c.wn == 4) return First<TN, 4>(c, n, k);
  return First<TN, 8>(c, n, k);
}
int Width(codegen::MxDecodeClassV2 const& c, int n, int k) {
  int first = c.tn == 16 ? First<16>(c,n,k) : c.tn == 32 ? First<32>(c,n,k) : c.tn == 64 ? First<64>(c,n,k) : First<128>(c,n,k);
  return std::max(c.d, std::min(first, 60 / (c.tn / 4)));
}
MxDecodeModel Model() {
  MxDecodeModel m;
  m.layers = 2; m.hidden = 1664; m.heads = 26; m.hkv = 13; m.head_dim = 64; m.q_per_kv = 2;
  m.inter = 1664; m.vocab = 16384; m.capacity = 1088;
  return m;
}
Value Table() {
  auto m = Model();
  json::Array rows;
  std::set<std::string> seen;
  int ci = 0;
  for (std::string cls : {"qkv", "o", "gate_up", "down", "head"}) {
    int n = cls == "qkv" ? 3328 : cls == "gate_up" ? 3328 : cls == "head" ? 16384 : 1664;
    int k = 1664, op = cls == "o" || cls == "down" ? 1 : cls == "gate_up" ? 2 : 0;
    int norm = cls == "qkv" || cls == "gate_up" || cls == "head";
    int rank = 0;
    for (auto const& c : MxDecodeDomainV2(m, cls, 1)) {
      auto key = Obj({{"n", n}, {"k", k}, {"b", 1}, {"tn", c.tn}, {"wn", c.wn}, {"depth", c.d}, {"threads", 576}, {"prefetch", Width(c,n,k)}});
      if (cls != "head") { key.Set("op", op); key.Set("norm", norm); key.Set("part", c.stream_k ? 0 : 1); }
      std::string kind = cls == "head" ? "head" : "gemm";
      // o and down share the exact same stage key and therefore the same cost.
      if (seen.insert(kind + key.Dump(0)).second) rows.push_back(Row(kind, key, 100 + ci * 10 + rank * (ci == 0 ? 0.01 : ci == 1 ? 0.02 : 5)));
      ++rank;
    }
    ++ci;
  }
  for (int bmin : {1, 2, 4, 8}) for (int length : {65, 256, 512, 768, 1088}) {
    auto key = Obj({{"dim", 64}, {"q", 2}, {"hkv", 13}, {"b", 1}, {"length", length}, {"b_min", bmin}, {"qk_norm", 0},
                    {"capacity", 1088}, {"threads", 576}, {"prefetch_kind", 1}});
    rows.push_back(Row("attention", key, 2.0 * length + 1000 * bmin));
  }
  return Obj({{"schema_version", 2}, {"table_kind", "mx5_decode_stage_costs"}, {"qualified", true},
              {"qualification_scope", "measured_rows_only"}, {"full_table", false}, {"diagnostic_reasons", Value(json::Array{})},
              {"provenance", Obj({{"card", "synthetic CPU fixture"}})}, {"rows", Value(rows)}});
}
void Throws(std::function<void()> action, std::string const& expected) {
  try { action(); } catch (std::exception const& e) {
    if (std::string(e.what()).find(expected) == std::string::npos) { std::cerr << e.what() << " expected " << expected << '\n'; std::abort(); }
    return;
  }
  std::cerr << "missing rejection: " << expected << '\n'; std::abort();
}
Value ChangeRow(Value const& table, std::function<void(Value&)> action, std::size_t index = 0) {
  auto changed = table;
  auto rows = table.At("rows").AsArray("rows"); action(rows[index]); changed.Set("rows", Value(rows)); return changed;
}
void TestMerge() {
  using Ranks = std::array<std::size_t, 6>;
  std::array<std::vector<double>, 6> lists{{{0, 1, 2}, {0, 1, 3}, {0, 7}, {0}, {0, 8}, {0, 9}}};
  std::vector<std::pair<double,Ranks>> brute;
  Ranks r{};
  std::function<void(int,double)> visit = [&](int i, double sum) {
    if (i == 6) { brute.push_back({sum,r}); return; }
    for (r[i] = 0; r[i] < lists[i].size(); ++r[i]) visit(i+1, sum+lists[i][r[i]]);
  };
  visit(0,0); std::sort(brute.begin(),brute.end());
  auto top = MxDecodeKBestV2(lists, 20);
  assert(top.size() == 20);
  for (std::size_t i = 0; i < top.size(); ++i) assert(top[i] == brute[i].second);
  assert(top[3][0] + top[3][1] == 2);  // A second replacement is required among the best four.
  lists[0] = {2,1}; Throws([&] { MxDecodeKBestV2(lists); }, "unsorted");
}
void TestSolve(Value const& table) {
  MxDecodeSolverV2 solver(table);
  auto m = Model(); auto selected = solver.Solve(m,1);
  assert(selected.top.size() == 4);
  assert(selected.top[0].config.threads == 576 && selected.top[0].config.b_min == 1);
  // An affine attention curve has exact discrete mean at mean L=576.
  assert(std::abs(selected.top[0].stage_ns.at("attention") - 2152) < 1e-9);
  auto p = solver.Predict(m,1,selected.top[0].config);
  assert(p.tpot_ns == selected.top[0].tpot_ns);
  assert(solver.FitRows() == int(table.At("rows").AsArray("rows").size()) - 20 && std::isfinite(solver.FitR2()));
  std::array<std::vector<double>,6> costs;
  char const* names[] = {"qkv","o","gate_up","down","head","attention"};
  for (int i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < std::min<std::size_t>(4, selected.candidates.at(names[i]).size()); ++j)
      costs[i].push_back(selected.candidates.at(names[i])[j].ns * (i == 4 ? 1 : m.layers));
  using Ranks = std::array<std::size_t,6>;
  std::vector<std::pair<double,Ranks>> oracle;
  Ranks ranks{};
  std::function<void(int,double)> enumerate = [&](int i, double sum) {
    if (i == 6) { oracle.push_back({sum,ranks}); return; }
    for (ranks[i] = 0; ranks[i] < costs[i].size(); ++ranks[i]) enumerate(i+1,sum+costs[i][ranks[i]]);
  };
  enumerate(0,0); std::sort(oracle.begin(),oracle.end());
  for (int i = 0; i < 4; ++i) assert(selected.top[i].ranks == oracle[i].second);
  std::vector<std::array<double,2>> measured{{{100,100}},{{99.1,99.1}},{{110,110}},{{120,120}}};
  assert(SelectMxDecodeMeasuredV2(selected,measured).selected == 0);
  measured[1] = {99,99}; auto chosen = SelectMxDecodeMeasuredV2(selected,measured);
  assert(chosen.selected == 1 && chosen.pred_over_meas[1] == selected.top[1].tpot_ns / 99);
  measured[2] = {98,98}; assert(SelectMxDecodeMeasuredV2(selected,measured).selected == 2);
  Throws([&] { solver.EstimateUncovered(m,"qkv",1,selected.top[0].config.qkv); }, "covered shape");
  m.hidden += 32;
  auto domain = MxDecodeDomainV2(m,"qkv",1);
  assert(std::isfinite(solver.EstimateUncovered(m,"qkv",1,domain[0])));
  Throws([&] { solver.Solve(Model(),1,512); }, "missing qualified legal coordinate");
}
void TestPiecewiseAttention(Value const& table) {
  auto rows = table.At("rows").AsArray("rows");
  int lengths[] = {65,256,512,768,1088};
  double values[] = {100,400,200,900,500};
  for (auto& row : rows) {
    if (row.At("kind").AsString("kind") != "attention" || Number(row.At("key"),"b_min") != 1) continue;
    int length = Number(row.At("key"),"length");
    auto i = std::find(std::begin(lengths),std::end(lengths),length) - std::begin(lengths);
    row = Row("attention",row.At("key"),values[i]);
  }
  auto changed = table; changed.Set("rows",Value(rows)); MxDecodeSolverV2 solver(changed);
  double expected = 0;
  for (int i = 0; i < 4; ++i) {
    int first = lengths[i] + (i != 0), last = std::min(1087,lengths[i+1]);
    double a = values[i] + (values[i+1]-values[i]) * double(first-lengths[i]) / (lengths[i+1]-lengths[i]);
    double b = values[i] + (values[i+1]-values[i]) * double(last-lengths[i]) / (lengths[i+1]-lengths[i]);
    expected += (last-first+1) * (a+b) / 2;
  }
  expected /= 1023;
  assert(std::abs(solver.Solve(Model(),1).top[0].stage_ns.at("attention") - expected) < 1e-9);
  for (int i = 0; i < 5; ++i)
    assert(solver.Solve(Model(),1,576,104,lengths[i]-1,lengths[i]-1).top[0].stage_ns.at("attention") == values[i]);
}
void TestRejections(Value const& table) {
  auto bad = table; bad.Set("schema_version",1); Throws([&] { MxDecodeSolverV2 s(bad); }, "schema-v2");
  bad = table; bad.Set("qualified",false); Throws([&] { MxDecodeSolverV2 s(bad); }, "qualified");
  bad = ChangeRow(table,[](Value& r) { auto res = r.At("resources"); res.Set("local_bytes",8); r.Set("resources",res); });
  Throws([&] { MxDecodeSolverV2 s(bad); }, "resource rejection");
  bad = ChangeRow(table,[](Value& r) { auto p = r.At("provenance"); p.Set("source_sha256","invalid"); r.Set("provenance",p); });
  Throws([&] { MxDecodeSolverV2 s(bad); }, "provenance hash");
  bad = ChangeRow(table,[](Value& r) { r.Set("r_next",159); });
  Throws([&] { MxDecodeSolverV2 s(bad); }, "reference MTreg");
  bad = ChangeRow(table,[](Value& r) { auto ref = r.At("r_next_reference"); auto p = ref.At("provenance"); p.Set("source_sha256",std::string(64,'e')); ref.Set("provenance",p); r.Set("r_next_reference",ref); });
  Throws([&] { MxDecodeSolverV2 s(bad); }, "identity mismatch");
  bad = ChangeRow(table,[](Value& r) { auto rounds = r.At("rounds").AsArray("rounds"); rounds.pop_back(); r.Set("rounds",Value(rounds)); });
  Throws([&] { MxDecodeSolverV2 s(bad); }, "three measured rounds");
  bad = table; auto rows = table.At("rows").AsArray("rows"); rows.erase(rows.begin()); bad.Set("rows",Value(rows));
  MxDecodeSolverV2 incomplete(bad); Throws([&] { incomplete.Solve(Model(),1); }, "partial covered domain");
  Throws([&] { incomplete.EstimateUncovered(Model(),"qkv",1,MxDecodeDomainV2(Model(),"qkv",1)[0]); }, "covered shape");
  bad = table; rows = table.At("rows").AsArray("rows"); rows.pop_back(); bad.Set("rows",Value(rows));
  MxDecodeSolverV2 missing_attention(bad); Throws([&] { missing_attention.Solve(Model(),1); }, "attention grid coordinate");
  bad = table; rows = table.At("rows").AsArray("rows"); rows.push_back(rows[0]); bad.Set("rows",Value(rows));
  Throws([&] { MxDecodeSolverV2 s(bad); }, "duplicate measured key");
}
void TestReduction(Value const& table) {
  auto rows = table.At("rows").AsArray("rows");
  std::size_t idx = 0;
  while (idx < rows.size() && (rows[idx].At("kind").AsString("kind") == "attention"
         || Number(rows[idx].At("key"),"prefetch") <= Number(rows[idx].At("key"),"depth"))) ++idx;
  assert(idx < rows.size());
  auto old = rows[idx], key = old.At("key");
  int initial = Number(key,"prefetch"), depth = Number(key,"depth");
  key.Set("prefetch",std::max(depth,initial/2));
  std::string kind = old.At("kind").AsString("kind");
  rows[idx] = Row(kind,key,old.At("median_ns").AsNumber("median_ns"));
  auto bad = table; bad.Set("rows",Value(rows));
  Throws([&] { MxDecodeSolverV2 s(bad); }, "resource_selection evidence");
  auto attempt = [&](int step, bool fits, Value const& akey) {
    auto identity = Identity(kind).AsObject("identity"); std::reverse(identity.begin(),identity.end());
    auto resource = Resource(fits ? 200 : 228).AsObject("resources"); std::reverse(resource.begin(),resource.end());
    return Obj({{"resource_step",step},{"prefetch",Number(akey,"prefetch")},{"key",akey},{"query_status",0},{"query_only",true},
                {"fits",fits},{"resources",Value(resource)},{"provenance",Value(identity)}});
  };
  auto selection = Obj({{"selected_resource_step",1},{"resource_attempts",Value(json::Array{attempt(0,false,old.At("key")),attempt(1,true,key)})}});
  rows[idx].Set("resource_selection",selection);
  auto good = table; good.Set("rows",Value(rows)); MxDecodeSolverV2 valid(good);
  assert(valid.Solve(Model(),1).top.size() == 4);
  auto attempts = selection.At("resource_attempts").AsArray("attempts"); attempts[0].Set("fits",true); selection.Set("resource_attempts",Value(attempts));
  rows[idx].Set("resource_selection",selection); bad.Set("rows",Value(rows));
  Throws([&] { MxDecodeSolverV2 s(bad); }, "first passing build");
}
}  // namespace
int main() {
  TestMerge(); auto table = Table(); TestSolve(table); TestPiecewiseAttention(table); TestRejections(table); TestReduction(table);
  std::cout << "mx_decode_solver_v2: CPU synthetic qualification, interpolation, k-best and selection checks passed\n";
}
