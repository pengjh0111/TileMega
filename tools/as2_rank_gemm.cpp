#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/AscendImplementation.h>
#include <tilemega/Support/Json.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <cmath>

using namespace tilemega;
using namespace tilemega::solver;
namespace fs=std::filesystem;
static fs::path Controlled(fs::path path) {
  auto root=fs::path("/home/d50064638/junhuip/TileMega/runs/as2");
  path=fs::canonical(path);auto r=path.lexically_relative(root);
  if(r.empty() || *r.begin()==".." || path==root)throw std::invalid_argument("Use existing controlled AS-2 evidence");
  return path;
}
static json::Value Read(fs::path path) {return json::ParseFile(Controlled(path).string());}
class Prices final:public AscendGemmPriceProvider {
 public:
  GemmProblem shape;
  std::map<std::string,AscendGemmEstimate> values;
  AscendGemmEstimate Estimate(BackendCandidate const& candidate,GemmProblem const& problem) const override {
    if(problem.m!=shape.m || problem.n!=shape.n || problem.k!=shape.k)
      throw std::invalid_argument("No exact-shape native empirical profile");
    return values.at(candidate.traits().ascend_implementation->impl_id);
  }
};
static void Write(fs::path path,std::string const& text) {
  std::ofstream out(path);out<<text<<'\n';if(!out)throw std::runtime_error("Write failed");
}
int main(int argc,char** argv) {
  try {
    if(argc!=4 || (std::string(argv[1])!="--pilot" && std::string(argv[1])!="--final"))
      throw std::invalid_argument("usage: as2_rank_gemm --pilot|--final evidence_index.json new_output_directory");
    bool pilot=std::string(argv[1])=="--pilot";
    auto index=Read(argv[2]);auto output=fs::weakly_canonical(argv[3]);
    auto timer=Read("/home/d50064638/junhuip/TileMega/runs/as2/jobs/timer_differential/stdout.log");
    if(timer.At("status").AsString("status")!="passed" || timer.At("counter_hz").AsNumber("frequency")!=1e9)
      throw std::invalid_argument("Native pricing requires the successful 1 GHz counter probe");
    Controlled(output.parent_path());if(fs::exists(output))throw std::invalid_argument("Output already exists");
    auto prices=std::make_shared<Prices>();std::vector<BackendCandidate> candidates;
    auto target_json=json::ParseFile(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
    auto impls=target_json.At("implementations").AsArray("implementations");
    json::Array evidence;
    for(auto const& entry:index.At("profiles").AsArray("profiles")) {
      auto job_dir=Controlled(entry.At("job").AsString("job"));
      auto numeric=Read(fs::path(entry.At("output").AsString("output"))/"numeric.json");
      auto config=Read(fs::path(entry.At("input").AsString("input"))/"config.json");
      auto job=Read(job_dir/"job.json"),measurement=Read(job_dir/"stdout.log");
      if(job.At("status").AsString("status")!="passed" || job.At("exit_code").AsNumber("exit")!=0 ||
          job.At("device_locked").AsBool("locked") || numeric.At("status").AsString("status")!="passed" ||
          measurement.At("tilemega_pipeline").AsBool("pipeline") || measurement.At("groups").AsNumber("groups")!=2)
        throw std::invalid_argument("Require passed numerical/launch receipts from the bounded G2 body probe");
      bool contended=job.At("timing_status").AsString("timing_status")=="contended";
      if(contended && !pilot)throw std::invalid_argument("Contended body profiles are pilot-only");
      unsigned implementation=measurement.At("implementation").AsNumber("implementation");
      auto traits=As2Bf16GemmTraits(implementation);
      auto const& mnk=config.At("shape_mnk").AsArray("shape_mnk");
      if(config.At("shape_mnk").Dump()!=measurement.At("shape_mnk").Dump())
        throw std::invalid_argument("Measurement/input shape identity mismatch");
      GemmProblem shape{int(mnk.at(0).AsNumber("M")),int(mnk.at(1).AsNumber("N")),int(mnk.at(2).AsNumber("K"))};
      if(shape.m<1 || shape.n<1 || shape.k<1)throw std::invalid_argument("Invalid empirical shape");
      if(!candidates.empty() && (shape.m!=prices->shape.m || shape.n!=prices->shape.n || shape.k!=prices->shape.k))
        throw std::invalid_argument("Compare candidates on the same actual GEMM problem");
      prices->shape=shape;
      auto const& ticks=measurement.At("tile_service_ticks_3_repeats").AsArray("ticks");
      const int tiles=((shape.m+traits.tile_m-1)/traits.tile_m)*((shape.n+traits.tile_n-1)/traits.tile_n);
      if(ticks.size()!=unsigned(tiles))throw std::invalid_argument("Measurement does not cover the candidate's tile domain");
      double lanes[2]={0,0};
      for(unsigned t=0;t<ticks.size();++t) {
        auto value=ticks[t].AsNumber("ticks");
        if(!std::isfinite(value) || value<=0)throw std::invalid_argument("Missing body service time");
        lanes[t%2]+=value/3;
      }
      const auto id=traits.ascend_implementation->impl_id;
      if(!prices->values.emplace(id,AscendGemmEstimate{std::max(lanes[0],lanes[1]),2,job_dir.string()}).second)
        throw std::invalid_argument("Duplicate implementation profile");
      candidates.emplace_back(traits);impls.emplace_back(json::Object{{"impl_id",id},{"task_kind",0}});
      evidence.emplace_back(json::Object{{"impl_id",id},{"job",job_dir.string()},
        {"binary_sha256",job.At("executable_sha256").AsString("hash")},
        {"timing_status",contended?"contended":"load_not_fully_characterized"},
        {"task_service_makespan_ns",std::max(lanes[0],lanes[1])}});
    }
    if(candidates.size()<2)throw std::invalid_argument("This comparison requires two measured legal native implementations");
    target_json.Set("implementations",std::move(impls));
    target_json.Set("selection","native_empirical_provider");
    target_json.Set("cost_provider_schema","tilemega.ascend.empirical_gemm.v1");
    fs::create_directory(output);
    Write(output/"target.json",target_json.Dump());auto target=TargetSpec::FromJson((output/"target.json").string());
    CandidateGenerator generator(target,ScalarType::kBF16);generator.SetAscendDomain(candidates,prices);
    auto legal=generator.Enumerate();auto selected=generator.TopK(legal,{prices->shape},1).at(0);
    json::Value result(json::Object{{"status","verified_component_ranking_only"},{"pilot_only",pilot},
      {"cost_unit","ns; 1 GHz counter verified by timer_differential"},
      {"cost_scope","sum of measured hot tile service times per worker; excludes initialization, dispatch and inter-task handoff"},
      {"profiles",std::move(evidence)},{"selected_impl",selected.traits().ascend_implementation->impl_id},
      {"full_model_plan_selected",false},{"G_PIPELINE","in_progress"}});
    Write(output/"search.json",result.Dump());std::cout<<result.Dump()<<std::endl;return 0;
  }catch(std::exception const& error){std::cerr<<error.what()<<std::endl;return 1;}
}
