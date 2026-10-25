// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxDecodeSolverV2.h>
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include "../../lib/Solver/MxDecodeCompilerProbeV2.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

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
std::string TarMember(std::string const& name, std::string const& bytes, char type) {
  std::string header(512, '\0');
  std::copy(name.begin(),name.end(),header.begin());
  std::snprintf(&header[100],8,"%07o",0644);
  std::snprintf(&header[124],12,"%011lo",static_cast<unsigned long>(bytes.size()));
  std::fill(header.begin()+148,header.begin()+156,' ');header[156]=type;
  std::copy_n("ustar",5,header.begin()+257);
  unsigned long sum=0;for(unsigned char c:header)sum+=c;
  std::snprintf(&header[148],8,"%06lo",sum);header[154]='\0';header[155]=' ';
  return header+bytes+std::string((512-bytes.size()%512)%512,'\0');
}
Value ProofFile(std::filesystem::path const& path) {
  std::ifstream stream(path,std::ios::binary);std::string text((std::istreambuf_iterator<char>(stream)),{});
  return Obj({{"path",path.string()},{"sha256",solver::compiler_probe_v2::Digest(text)}});
}
void Put(std::filesystem::path const& path,std::string const& text) {std::ofstream(path,std::ios::binary)<<text;}
void TestCompilerOnlyReference(Value const& table) {
  namespace fs=std::filesystem;
  auto dir=fs::temp_directory_path()/("mx5_p0_cpu_fixture_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(dir);
  auto binary=dir/"synthetic_p0.so",metadata_path=dir/"synthetic_p0.json",archive=dir/"sources.tar.gz";
  Put(binary,"CPU synthetic compiler-probe binary fixture, never executed\n");
  std::map<std::string,std::string> files{{"experiments/mx5/stage_bench.cu","// CPU synthetic bench fixture\n"},
                                       {"include/tilemega/Backend/Mx/P0Fixture.h","// CPU synthetic body fixture\n"}};
  std::string canonical,tar;
  for(auto const& [name,body]:files) {
    canonical+=name+'\0'+body+'\0';
    std::string record=" path="+name+"\n";std::size_t length=record.size()+1;
    while(std::to_string(length).size()+record.size()!=length)length=std::to_string(length).size()+record.size();
    tar+=TarMember("PaxHeader",std::to_string(length)+record,'x');tar+=TarMember("content",body,'0');
  }
  tar+=std::string(1024,'\0');gzFile gzip=gzopen(archive.c_str(),"wb");assert(gzip);
  assert(gzwrite(gzip,tar.data(),unsigned(tar.size()))==int(tar.size()));assert(gzclose(gzip)==Z_OK);
  auto source_sha=solver::compiler_probe_v2::Digest(canonical),binary_sha=ProofFile(binary).At("sha256").AsString("binary");
  auto metadata=Obj({{"body","gemm"},{"bench",true},{"returncode",0},{"tn",16},{"depth",1},{"threads",576},{"pw",0},
    {"base_head","CPU_FIXTURE"},{"source_sha256",source_sha},{"binary_sha256",binary_sha},
    {"command",Value(json::Array{"/cpu/fixture/compiler","-DMX5_TEST_THREADS=576","-DMX5_TEST_TN=16","-DMX5_TEST_DEPTH=1",
                                 "-DMX5_BENCH_PW=0","experiments/mx5/stage_bench.cu"})}});
  Put(metadata_path,metadata.Dump());
  auto key=table.At("rows").AsArray("rows")[0].At("key");key.Set("prefetch",0);
  auto provenance=Identity("gemm",true);provenance.Set("source_sha256",source_sha);provenance.Set("binary_sha256",binary_sha);provenance.Set("base_head","CPU_FIXTURE");
  auto res=Resource(168);res.Set("local_bytes",616);
  auto reference=Obj({{"kind","gemm"},{"key",key},{"provenance",provenance},{"resources",res},{"compiler_probe_only",true},{"fits",false},
    {"probe",Obj({{"query_only",true},{"iterations",0},{"query_status",0},{"binary",ProofFile(binary)},
                   {"build_metadata",ProofFile(metadata_path)},{"source_archive",ProofFile(archive)}})}});
  auto over_cap=reference;auto attributes=reference.At("resources");
  attributes.Set("mtreg",268);attributes.Set("static_smem_bytes",16);attributes.Set("resident_ctas",0);over_cap.Set("resources",attributes);
  solver::compiler_probe_v2::Validate(over_cap,"gemm",576);
  std::vector<Value> invalid_attributes{Value(),Value(-1),Value(true),Value("not-an-attribute"),Value(1.5)};
  for (auto const& seed:std::vector<Value>{over_cap,reference}) {
    for (char const* field:{"local_bytes","static_smem_bytes","resident_ctas"}) {
      for (auto const& invalid:invalid_attributes) {
        auto malformed=seed;auto values=seed.At("resources");values.Set(field,invalid);malformed.Set("resources",values);
        Throws([&]{solver::compiler_probe_v2::Validate(malformed,"gemm",576);},field);
      }
      auto malformed=seed;Value incomplete=json::Object{};
      for(char const* name:{"mtreg","local_bytes","static_smem_bytes","resident_ctas"})
        if(std::string(name)!=field)incomplete.Set(name,seed.At("resources").At(name));
      malformed.Set("resources",incomplete);
      Throws([&]{solver::compiler_probe_v2::Validate(malformed,"gemm",576);},field);
    }
  }
  auto changed=ChangeRow(table,[&](Value& row) {
    row.Set("r_next",168);row.Set("r_next_reference",reference);
    auto p=row.At("provenance");p.Set("source_sha256",source_sha);row.Set("provenance",p);
    auto rounds=row.At("rounds").AsArray("rounds");
    for(auto& round:rounds) {auto m=round.At("measurement");m.Set("source_sha256",source_sha);round.Set("measurement",m);}
    row.Set("rounds",rounds);
  });
  MxDecodeSolverV2 accepted(changed);auto selected=accepted.Solve(Model(),1);
  assert(selected.top[0].config.qkv.r_next==168 && selected.top[0].config.qkv.prefetch==6);
  auto const& actual=selected.candidates.at("qkv")[0].evidence[0].At("r_next_reference");
  assert(actual.At("resources").At("local_bytes").AsNumber("stack")==616 && !actual.At("fits").AsBool("fits"));
  for(auto const& [field,value]:std::vector<std::pair<std::string,Value>>{{"query_only",false},{"iterations",1},{"query_status",1}}) {
    auto bad=ChangeRow(changed,[&](Value& row) {auto r=row.At("r_next_reference"),q=r.At("probe");q.Set(field,value);r.Set("probe",q);row.Set("r_next_reference",r);});
    Throws([&]{MxDecodeSolverV2 solver(bad);},"compiler probe");
  }
  auto bad=ChangeRow(changed,[](Value& row){auto r=row.At("resources");r.Set("local_bytes",4);row.Set("resources",r);});
  Throws([&]{MxDecodeSolverV2 solver(bad);},"resource rejection");
  bad=ChangeRow(changed,[](Value& row){auto r=row.At("r_next_reference"),p=r.At("provenance");p.Set("source_sha256",std::string(64,'f'));r.Set("provenance",p);row.Set("r_next_reference",r);});
  Throws([&]{MxDecodeSolverV2 solver(bad);},"compiler probe");
  bad=ChangeRow(changed,[](Value& row){auto r=row.At("r_next_reference"),q=r.At("probe");q.Set("query_status",Value());r.Set("probe",q);row.Set("r_next_reference",r);});
  Throws([&]{MxDecodeSolverV2 solver(bad);},"query_status");
  bad=ChangeRow(changed,[](Value& row){auto r=row.At("r_next_reference"),v=r.At("resources");v.Set("mtreg",Value());r.Set("resources",v);row.Set("r_next_reference",r);});
  Throws([&]{MxDecodeSolverV2 solver(bad);},"mtreg");
  bad=ChangeRow(changed,[](Value& row){auto r=row.At("r_next_reference");r.Set("fits",true);row.Set("r_next_reference",r);});
  Throws([&]{MxDecodeSolverV2 solver(bad);},"diagnostic fits");
  bad=ChangeRow(changed,[](Value& row){auto r=row.At("r_next_reference");row.Set("r_next_reference",Obj({{"kind",r.At("kind")},{"key",r.At("key")},{"resources",r.At("resources")},{"provenance",r.At("provenance")}}));});
  Throws([&]{MxDecodeSolverV2 solver(bad);},"resource rejection");
  Put(binary,"changed synthetic proof\n");Throws([&]{MxDecodeSolverV2 solver(changed);},"proof file hash mismatch");
  fs::remove_all(dir);
}
// CPU synthetic proof bundles; no real costs or resource queries.
struct OutcomeProofs {
  std::filesystem::path dir,compiler;
  std::map<std::string,std::string> sources;
  int sequence=0;
  OutcomeProofs() {
    dir=std::filesystem::temp_directory_path()/("mx5_outcomes_cpu_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);compiler=dir/"CPU_fixture_compiler";Put(compiler,"CPU synthetic compiler identity, never executed\n");std::filesystem::permissions(compiler,std::filesystem::perms::owner_exec,std::filesystem::perm_options::add);
    for(std::string kind:{"gemm","head","attention"}) {
      std::string name="experiments/mx5/"+(kind=="gemm"?std::string("stage_bench.cu"):kind+"_stage_bench.cu"),body="// CPU synthetic "+kind+" bench fixture\n";
      sources[kind]=compiler_probe_v2::Digest(name+'\0'+body+'\0');
      auto tar=TarMember(name,body,'0')+std::string(1024,'\0');auto path=dir/(kind+".tar.gz");auto gz=gzopen(path.c_str(),"wb");assert(gz);
      assert(gzwrite(gz,tar.data(),unsigned(tar.size()))==int(tar.size()));assert(gzclose(gz)==Z_OK);
    }
  }
  ~OutcomeProofs(){std::filesystem::remove_all(dir);}
  Value Proof(std::string const& kind,Value key,bool rejection) {
    auto stem=dir/("proof"+std::to_string(sequence++)),binary=stem;binary+=".so";
    Put(binary,"CPU synthetic binary "+std::to_string(sequence));
    auto prov=Identity(kind);prov.Set("source_sha256",sources[kind]);prov.Set("binary_sha256",ProofFile(binary).At("sha256"));
    std::string field=kind=="attention"?"prefetch_kind":"prefetch";
    json::Array command{compiler.string(),"-std=c++17","-O3","-shared","-Xcompiler","-fPIC,-resource-usage","-fno-strict-aliasing","-DTILEMEGA_SYNC_V3=1","-DTILEMEGA_WATCHDOG=0","-DMX5_TEST_THREADS=576"};
    auto meta=Obj({{"body",kind},{"bench",true},{"returncode",0},{"threads",576},{"source_sha256",sources[kind]},
                   {"binary_sha256",prov.At("binary_sha256")},{"base_head",prov.At("base_head")}});
    auto coordinate=[&](char const* name,char const* dimension,char const* macro) {
      int value=Number(key,dimension);meta.Set(name,value);command.emplace_back(std::string("-D")+macro+"="+std::to_string(value));
    };
    if(kind=="attention") {coordinate("dim","dim","MX5_TEST_DIM");coordinate("query","q","MX5_TEST_Q");coordinate("prefetch_kind","prefetch_kind","MX5_BENCH_PREFETCH_KIND");}
    else {coordinate("tn","tn","MX5_TEST_TN");coordinate("depth","depth","MX5_TEST_DEPTH");coordinate("pw","prefetch","MX5_BENCH_PW");}
    command.emplace_back("-Iinclude");command.emplace_back("experiments/mx5/"+(kind=="gemm"?std::string("stage_bench.cu"):kind+"_stage_bench.cu"));command.emplace_back("-o");command.emplace_back(binary.string());meta.Set("command",command);
    auto metadata=stem;metadata+=".json";Put(metadata,meta.Dump());
    auto resources=Resource(160);if(rejection)resources.Set("local_bytes",8);
    auto raw=Obj({{"library",binary.string()},{"kind",kind},{"key",key},{"resources",resources},{"provenance",prov},{"fits",!rejection},
                   {"query_only",true},{"iterations",0},{"query_status",0}});
    auto rawpath=stem;rawpath+=".raw.json";Put(rawpath,raw.Dump());
    auto proof=Obj({{"query_only",true},{"iterations",0},{"query_status",0},{"binary",ProofFile(binary)},
                   {"build_metadata",ProofFile(metadata)},{"source_archive",ProofFile(dir/(kind+".tar.gz"))},
                   {"compiler",ProofFile(compiler)},{"raw_query",ProofFile(rawpath)}});
    auto record=Obj({{"kind",kind},{"key",key},{"resources",resources},{"provenance",prov},{"fits",!rejection},{"probe",proof}});
    if(rejection) {
      record.Set("compiler_binding","retained_executable_at_qualification");record.Set("original_compiler_identity_recorded",false);record.Set("original_build_compiler_sha256",Value());record.Set("build_policy","mx5_build_bodies_default_v1");record.Set("primary_resource_rejection",true);record.Set("disposition","primary_resource_rejected");record.Set("r_next",160);
      key.Set(field,0);record.Set("r_next_reference",Proof(kind,key,false));
    } else record.Set("compiler_probe_only",true);
    return record;
  }
  Value Envelope(Value const& table,json::Array rows,json::Array rejected) {
    for(auto& row:rows) {
      auto kind=row.At("kind").AsString("kind");auto prov=row.At("provenance");prov.Set("source_sha256",sources[kind]);row.Set("provenance",prov);
      if(auto* selection=row.Find("resource_selection")) {
        auto value=*selection;
        if(auto* attempts=value.Find("resource_attempts")) {
          auto history=attempts->AsArray("resource_attempts");
          for(auto& attempt:history){auto identity=attempt.At("provenance");identity.Set("source_sha256",sources[kind]);attempt.Set("provenance",identity);}
          value.Set("resource_attempts",history);row.Set("resource_selection",value);
        }
      }
      auto ref=row.At("r_next_reference"),rp=ref.At("provenance");rp.Set("source_sha256",sources[kind]);ref.Set("provenance",rp);row.Set("r_next_reference",ref);
      auto rounds=row.At("rounds").AsArray("rounds");for(auto& round:rounds){auto m=round.At("measurement");m.Set("source_sha256",sources[kind]);round.Set("measurement",m);}row.Set("rounds",rounds);
    }
    auto m=Model();auto descriptor=Obj({{"layers",m.layers},{"hidden",m.hidden},{"heads",m.heads},{"hkv",m.hkv},
      {"head_dim",m.head_dim},{"q_per_kv",m.q_per_kv},{"inter",m.inter},{"vocab",m.vocab},{"capacity",m.capacity},{"qk_norm",m.qk_norm}});
    auto scope=Obj({{"model_id","CPU_SYNTHETIC_MODEL"},{"model",descriptor},{"batch",1},{"threads",576},{"grid",104},{"past_lo",64},{"past_hi",1086},
      {"source_sha256",Obj({{"gemm",sources["gemm"]},{"head",sources["head"]},{"attention",sources["attention"]}})}});
    auto doc=table;doc.Set("schema_version",3);doc.Set("qualification_scope","complete_outcomes_single_request_v1");doc.Set("scope",scope);
    doc.Set("rows",rows);doc.Set("resource_rejections",rejected);doc.Set("outcome_domain_complete",true);doc.Set("measured_domain_complete",rejected.empty());
    doc.Set("execution_512_approved",false);doc.Set("global_dt576_policy_approved",false);doc.Set("target_defaults_changed",false);
    doc.Set("coverage",Obj({{"expected_keys",int(table.At("rows").AsArray("rows").size())},{"costed_keys",int(rows.size())},{"rejected_keys",int(rejected.size())},{"missing_keys",0}}));return doc;
  }
};
void OutcomeOracle(MxDecodeSelectionV2 const& result) {
  char const* names[]={"qkv","o","gate_up","down","head","attention"};using Ranks=std::array<std::size_t,6>;
  std::vector<std::pair<double,Ranks>> brute;Ranks ranks{};
  std::function<void(int,double)> visit=[&](int i,double sum) {
    if(i==6){brute.push_back({sum,ranks});return;}
    auto const& list=result.candidates.at(names[i]);
    for(ranks[i]=0;ranks[i]<std::min<std::size_t>(4,list.size());++ranks[i])visit(i+1,sum+list[ranks[i]].ns*(i==4?1:Model().layers));
  };
  visit(0,0);std::sort(brute.begin(),brute.end());assert(result.top.size()==4);
  for(int i=0;i<4;++i) {assert(result.top[i].ranks==brute[i].second);assert(result.top[i].tpot_ns==brute[i].first);}
}
void TestOutcomeDomain(Value const& table) {
  OutcomeProofs proof;auto rows=table.At("rows").AsArray("rows");
  auto all=proof.Envelope(table,rows,{});MxDecodeSolverV2 solver(all);
  auto result=solver.SolveQualifiedDomain(Model(),1);OutcomeOracle(result);
  auto old=MxDecodeSolverV2(table).Solve(Model(),1);assert(result.top[0].tpot_ns==old.top[0].tpot_ns);
  Throws([&]{solver.Solve(Model(),1);},"explicit SolveQualifiedDomain");
  Throws([&]{solver.Predict(Model(),1,result.top[0].config);},"lineage-aware");
  auto wrong=Model();wrong.head_dim=128;Throws([&]{solver.SolveQualifiedDomain(wrong,1);},"model descriptor mismatch");
  Throws([&]{solver.SolveQualifiedDomain(Model(),2);},"scope request mismatch");
  auto wrong_source=all;auto scope=wrong_source.At("scope"),sources=scope.At("source_sha256");sources.Set("gemm",std::string(64,'f'));scope.Set("source_sha256",sources);wrong_source.Set("scope",scope);
  Throws([&]{MxDecodeSolverV2(wrong_source).SolveQualifiedDomain(Model(),1);},"source differs from scope");
  auto missing=rows;missing.erase(missing.begin());auto document=proof.Envelope(table,missing,{});
  Throws([&]{MxDecodeSolverV2(document).SolveQualifiedDomain(Model(),1);},"incomplete schema3 outcomes");
  auto rejected=proof.Proof("gemm",rows[0].At("key"),true);
  document=proof.Envelope(table,missing,{rejected});result=MxDecodeSolverV2(document).SolveQualifiedDomain(Model(),1);OutcomeOracle(result);
  assert(result.candidates.at("qkv").size()+1==old.candidates.at("qkv").size());
  auto conflict=proof.Envelope(table,rows,{rejected});
  Throws([&]{MxDecodeSolverV2(conflict).SolveQualifiedDomain(Model(),1);},"contradictory");
  auto bad=ChangeRow(document,[](Value& row){auto resource=row.At("resources");resource.Set("local_bytes",4);row.Set("resources",resource);});
  Throws([&]{MxDecodeSolverV2 s(bad);},"resource rejection");
  auto bad_rejection=rejected;bad_rejection.Set("median_ns",0);auto badrows=document.At("rows").AsArray("rows");
  Throws([&]{MxDecodeSolverV2 s(proof.Envelope(table,badrows,{bad_rejection}));},"must not contain costs");
  // One rejected point invalidates its whole b_min, without interpolating across it.
  json::Array costs,exclusions;for(auto const& row:rows) {
    if(row.At("kind").AsString("kind")=="attention" && Number(row.At("key"),"b_min")==1 && Number(row.At("key"),"length")==256)
      exclusions.push_back(proof.Proof("attention",row.At("key"),true));else costs.push_back(row);
  }
  result=MxDecodeSolverV2(proof.Envelope(table,costs,exclusions)).SolveQualifiedDomain(Model(),1);OutcomeOracle(result);
  assert(result.candidates.at("attention").size()==3 && result.top[0].config.b_min==2 && result.top[0].stage_ns.at("attention")==3152);
  costs.clear();exclusions.clear();for(auto const& row:rows) {
    if(row.At("kind").AsString("kind")=="head")exclusions.push_back(proof.Proof("head",row.At("key"),true));else costs.push_back(row);
  }
  MxDecodeSolverV2 empty(proof.Envelope(table,costs,exclusions));result=empty.SolveQualifiedDomain(Model(),1);
  assert(result.status=="no_executable_plan" && result.top.empty() && result.candidates.at("head").empty());
  assert(!result.domain_report.At("execution_512_approved").AsBool("512"));
  Throws([&]{empty.EstimateUncovered(Model(),"head",1,MxDecodeDomainV2(Model(),"head",1)[0]);},"rejected covered shape");
  // A single costed choice per class cannot be duplicated to manufacture top4.
  costs.clear();exclusions.clear();std::set<std::string> kept;
  for(auto const& row:rows) {
    auto kind=row.At("kind").AsString("kind");auto const& key=row.At("key");
    std::string shape=kind=="attention"?"attention":kind+std::to_string(Number(key,"n"))+"/"+std::to_string(Number(key,"k"))+(kind=="gemm"?"/"+std::to_string(Number(key,"op"))+"/"+std::to_string(Number(key,"norm")):"");
    bool keep=kind=="attention"?Number(key,"b_min")==1:kept.insert(shape).second;
    if(keep)costs.push_back(row);else exclusions.push_back(proof.Proof(kind,key,true));
  }
  result=MxDecodeSolverV2(proof.Envelope(table,costs,exclusions)).SolveQualifiedDomain(Model(),1);
  assert(result.status=="insufficient_top4" && result.top.size()==1);
  costs.clear();exclusions.clear();for(auto const& row:rows)exclusions.push_back(proof.Proof(row.At("kind").AsString("kind"),row.At("key"),true));
  MxDecodeSolverV2 all_rejected(proof.Envelope(table,costs,exclusions));result=all_rejected.SolveQualifiedDomain(Model(),1);
  assert(result.status=="no_executable_plan" && result.top.empty() && result.fit_rows==0);
  assert(result.domain_report.At("ranking_claim").AsString("ranking_claim")=="no_top4_claim");
  assert(result.domain_report.At("fit_status").AsString("fit_status")=="no_measured_rows");
  for(char const* field:{"local_bytes","static_smem_bytes","resident_ctas"}) {
    auto broken=rejected;auto attrs=broken.At("resources");attrs.Set("mtreg",268);attrs.Set(field,Value());broken.Set("resources",attrs);
    Throws([&]{MxDecodeSolverV2 s(proof.Envelope(table,missing,{broken}));},field);
  }
  auto line_bound=rejected;auto line_probe=line_bound.At("probe");
  auto raw=compiler_probe_v2::File(line_probe.At("raw_query"));auto rawpath=proof.dir/"original-attempts.jsonl";
  Put(rawpath,"{}\n"+raw+"\n");auto descriptor=ProofFile(rawpath);descriptor.Set("line",2);line_probe.Set("raw_query",descriptor);line_bound.Set("probe",line_probe);
  // The raw fixture was pretty-printed; retain a single genuine JSONL record.
  Put(rawpath,"{}\n"+json::Parse(raw).Dump(0)+"\n");descriptor=ProofFile(rawpath);descriptor.Set("line",2);line_probe.Set("raw_query",descriptor);line_bound.Set("probe",line_probe);
  auto normal_meta=json::Parse(compiler_probe_v2::File(line_probe.At("build_metadata")));
  auto observed=normal_meta.At("command").AsArray("command");observed.insert(observed.begin()+1,"-mllvm");observed.insert(observed.begin()+2,"-print-after=stack-frame-layout");normal_meta.Set("command",observed);
  auto normal_path=proof.dir/"normal-mir.json";Put(normal_path,normal_meta.Dump());line_probe.Set("build_metadata",ProofFile(normal_path));line_bound.Set("probe",line_probe);
  assert(MxDecodeSolverV2(proof.Envelope(table,missing,{line_bound})).SolveQualifiedDomain(Model(),1).top.size()==4);
  auto forced=rejected;auto attachment=forced.At("probe");
  auto meta=json::Parse(compiler_probe_v2::File(attachment.At("build_metadata")));
  auto command=meta.At("command").AsArray("command");command.insert(command.begin()+1,"--maxrregcount=227");meta.Set("command",command);
  auto path=proof.dir/"forced-diagnostic.json";Put(path,meta.Dump());attachment.Set("build_metadata",ProofFile(path));forced.Set("probe",attachment);
  Throws([&]{MxDecodeSolverV2 s(proof.Envelope(table,missing,{forced}));},"forced-register flags");
  for(char const* field:{"query_only","iterations","query_status"}) {
    auto broken=rejected;auto probe=broken.At("probe");probe.Set(field,Value());broken.Set("probe",probe);
    Throws([&]{MxDecodeSolverV2 s(proof.Envelope(table,missing,{broken}));},field);
  }
}

void TestSchema3ReviewRegressions(Value const& table) {
  OutcomeProofs proof;auto rows=table.At("rows").AsArray("rows");auto missing=rows;missing.erase(missing.begin());
  auto rejected=proof.Proof("gemm",rows[0].At("key"),true);
  auto alter=[&](Value reference,json::Array extra,bool trailing,std::string const& stem) {
    auto probe=reference.At("probe");auto meta=json::Parse(compiler_probe_v2::File(probe.At("build_metadata")));
    auto command=meta.At("command").AsArray("command");
    command.insert(trailing?command.end():command.begin()+1,extra.begin(),extra.end());meta.Set("command",command);
    auto path=proof.dir/(stem+".json");Put(path,meta.Dump());probe.Set("build_metadata",ProofFile(path));reference.Set("probe",probe);return reference;
  };
  for(bool trailing:{false,true})for(bool p0:{false,true}) {
    auto row=rejected;
    if(p0)row.Set("r_next_reference",alter(row.At("r_next_reference"),{"-mllvm","-print-after=stack-frame-layout"},trailing,"p0-mir"));
    else row=alter(row,{"-mllvm","-print-after=stack-frame-layout"},trailing,"primary-mir");
    auto result=MxDecodeSolverV2(proof.Envelope(table,missing,{row})).SolveQualifiedDomain(Model(),1);
    assert(result.top.size()==4);
  }
  auto forced=alter(rejected.At("r_next_reference"),{"-mllvm","--max-mtreg-number=255"},true,"forced-p0");
  // Schema2 keeps its pre-existing permissive compiler-only probe contract.
  compiler_probe_v2::Validate(forced,"gemm",576);
  auto broken=rejected;broken.Set("r_next_reference",forced);
  Throws([&]{MxDecodeSolverV2 solver(proof.Envelope(table,missing,{broken}));},"forced-register flags");
  auto cost_rows=rows;cost_rows[0].Set("r_next_reference",forced);
  auto document=proof.Envelope(table,cost_rows,{});
  Throws([&]{MxDecodeSolverV2 solver(document);},"forced-register flags");
  document.Set("schema_version",2);assert(MxDecodeSolverV2(document).Solve(Model(),1).top.size()==4);
  Value unmarked=json::Object{};for(auto const& [field,value]:forced.AsObject("reference"))if(field!="compiler_probe_only")unmarked.Set(field,value);
  cost_rows[0].Set("r_next_reference",unmarked);document=proof.Envelope(table,cost_rows,{});
  Throws([&]{MxDecodeSolverV2 solver(document);},"contains build evidence");
  auto flat=table.At("rows").AsArray("rows")[0].At("r_next_reference");compiler_probe_v2::Schema3LegacyReference(flat);
  // A legal reduced measured width needs every prescribed, source-bound attempt.
  auto key=rows[0].At("key");int initial=Number(key,"prefetch");assert(initial==6);key.Set("prefetch",3);
  cost_rows=rows;cost_rows[0]=Row("gemm",key,rows[0].At("median_ns").AsNumber("ns"));
  cost_rows[0].Set("resource_selection",Obj({{"selected_resource_step",0}}));
  Throws([&]{MxDecodeSolverV2 solver(proof.Envelope(table,cost_rows,{}));},"resource selection exceeds three halvings");
  auto primary_key=key;primary_key.Set("prefetch",initial);
  auto attempt=[&](int step,bool fits,Value actual) {
    return Obj({{"resource_step",step},{"prefetch",Number(actual,"prefetch")},{"key",actual},{"query_status",0},{"query_only",true},
                {"fits",fits},{"resources",Resource(fits?200:228)},{"provenance",Identity("gemm")}});
  };
  auto selection=Obj({{"selected_resource_step",1},{"resource_attempts",json::Array{attempt(0,false,primary_key),attempt(1,true,key)}}});
  cost_rows[0].Set("resource_selection",selection);document=proof.Envelope(table,cost_rows,{});
  auto result=MxDecodeSolverV2(document).SolveQualifiedDomain(Model(),1);assert(result.top.size()==4);
  assert(result.domain_report.At("r_next_policy_audit").At("legacy_cost_only_unverifiable").AsNumber("legacy")>0);
  auto attempts=selection.At("resource_attempts").AsArray("attempts");attempts[0].Set("fits",true);selection.Set("resource_attempts",attempts);cost_rows[0].Set("resource_selection",selection);
  Throws([&]{MxDecodeSolverV2 solver(proof.Envelope(table,cost_rows,{}));},"first passing build");
  std::cout<<"schema3 review regressions: P0 override rejected, MIR positions accepted, cost-width/halving parity and schema2 legacy preserved\n";
}
}  // namespace
int main() {
  TestMerge(); auto table = Table(); TestSolve(table); TestPiecewiseAttention(table); TestRejections(table); TestReduction(table); TestCompilerOnlyReference(table); TestOutcomeDomain(table); TestSchema3ReviewRegressions(table);
  std::cout << "mx_decode_solver_v2: CPU synthetic qualification, interpolation, k-best and selection checks passed\n";
  return 0;
}
