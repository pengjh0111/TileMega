// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Solver/CompilerSearch.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/IntervalSegments.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <llvm/ADT/StringExtras.h>

#include <exception>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <optional>
#include <cmath>
#include <limits>

namespace {
std::string quote(std::string const& value) {
  std::string result = "'";
  for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
  return result + "'";
}

std::string modelFingerprint(std::string const& path) {
  // This LLVM version keeps SHA256's byte count in uint32_t and shifts it
  // before widening. torch.export archives can exceed its 512 MiB limit.
  if(std::filesystem::file_size(path)>=(std::uintmax_t(1)<<29)) {
    std::string script="import hashlib,sys; h=hashlib.sha256(); "
        "f=open(sys.argv[1],'rb'); "
        "[h.update(b) for b in iter(lambda:f.read(1048576),b'')]; print(h.hexdigest())";
    auto command="python3 -c "+quote(script)+" "+quote(path);
    auto* pipe=popen(command.c_str(),"r");
    if(!pipe)throw std::runtime_error("cannot hash model archive");
    char buffer[66]={};auto* got=fgets(buffer,sizeof(buffer),pipe);int status=pclose(pipe);
    if(!got || status || std::string(buffer).size()!=65)
      throw std::runtime_error("model archive hash failed");
    return std::string(buffer,64);
  }
  auto file=llvm::MemoryBuffer::getFile(path);
  if(!file)throw std::runtime_error("cannot read model fingerprint input");
  llvm::SHA256 digest;digest.update(file.get()->getBuffer());
  return llvm::toHex(digest.final(),true);
}


int queryResidency(mlir::ModuleOp module,int kappa,
    tilemega::solver::CompilerSearchOptions const& options,
    std::filesystem::path const& directory,std::filesystem::path const& library) {
  std::filesystem::create_directories(directory);
  auto const free_mib=std::filesystem::space(directory).available/(1024*1024);
  std::cerr << "DISK NEED_MIB=8192 FREE_MIB=" << free_mib << '\n';
  if (free_mib<8192) throw std::runtime_error("insufficient disk for resource probe");
  std::vector<tilemega::codegen::RuntimeVariantModule> variants{{module,1,
      static_cast<std::uint32_t>(options.placement.dims.seq)}};
  auto source=directory/"candidate.cu",probe=directory/"query.cu",binary=directory/"query";
  std::ofstream(source) << "#define TILEMEGA_EVENT_KAPPA " << kappa << '\n'
      << tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(variants);
  std::ofstream wrapper(probe);
  wrapper << "#define main tilemega_fixture_main\n#include \"candidate.cu\"\n#undef main\n"
      << "int main() { using namespace tilemega::codegen;\n"
      << "auto target=tilemega::TargetSpec::Probe();\n"
      << "if (target.arch_tag!=" << std::quoted(options.placement.target.arch_tag)
      << " || target.res.num_sms!=" << options.placement.target.res.num_sms << ") return 3;\n"
      << "cudaFuncAttributes a{},b{};\n"
      << "TILEMEGA_CUDA_CHECK(cudaFuncGetAttributes(&a,tilemega_l1_kernel));\n"
      << "TILEMEGA_CUDA_CHECK(cudaFuncGetAttributes(&b,tilemega_l2_kernel));\n"
      << "TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l1_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TaskSmem)));\n"
      << "TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TaskSmem)));\n"
      << "int l1=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l1_kernel),kHarnessThreads,sizeof(TaskSmem));\n"
      << "int l2=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l2_kernel),kHarnessThreads,sizeof(TaskSmem));\n"
      << "std::printf(\"{\\\"resident\\\":%d,\\\"l1\\\":%d,\\\"l2\\\":%d,\\\"registers_l1\\\":%d,\\\"registers_l2\\\":%d,\\\"dynamic_shared\\\":%zu,\\\"threads\\\":%d}\\n\",std::min(l1,l2),l1,l2,a.numRegs,b.numRegs,sizeof(TaskSmem),kHarnessThreads);\n}\n";
  wrapper.close();
  std::string root=TILEMEGA_SOURCE_DIR;
  std::string nvcc=std::getenv("CUDACXX") ? std::getenv("CUDACXX") : "/usr/local/cuda/bin/nvcc";
  std::string command=quote(nvcc)+" -std=c++17 -O2 -lineinfo -Xptxas=-v -DTILEMEGA_MIDPOINT_REFINE=0 -arch="+
      quote(options.placement.target.NvccArch());
  for (char const* sub:{"include","third_party/cutlass/include","third_party/cutlass/tools/util/include","third_party/cutlass/test"})
    command+=" -I"+quote(root+"/"+sub);
  command+=" "+quote(probe.string())+" "+quote(library.string())+
      " -L/usr/local/cuda/lib64 -lcudart -o "+quote(binary.string());
  std::ofstream(directory/"build_command.txt") << command << '\n';
  if (std::system((command+" >"+quote((directory/"build.log").string())+" 2>&1").c_str()))
    throw std::runtime_error("resource probe compilation failed: "+directory.string());
  if (std::system((quote(binary.string())+" >"+quote((directory/"resources.json").string())+
      " 2>"+quote((directory/"query.log").string())).c_str()))
    throw std::runtime_error("resource probe does not match target or failed: "+directory.string());
  auto file=llvm::MemoryBuffer::getFile((directory/"resources.json").string());
  if (!file) throw std::runtime_error("missing resource query result");
  auto value=llvm::json::parse(file.get()->getBuffer());
  auto* object=value ? value->getAsObject() : nullptr;
  auto count=object ? object->getInteger("resident") : std::nullopt;
  if (!count || *count<=0) throw std::runtime_error("invalid resident limit from CUDA");
  std::cerr << "RESOURCE_QUERY directory=" << directory << " resident=" << *count << '\n';
  return int(*count);
}

struct VariantRequest {
  std::uint32_t seq_begin = 0;
  std::uint32_t seq_end = 0;
  tilemega::frontend::ImportOptions options;
};

std::int64_t requiredInteger(llvm::json::Object const& object,
                             llvm::StringRef name) {
  auto value = object.getInteger(name);
  if (!value) throw std::runtime_error("variant plan lacks integer " + name.str());
  return *value;
}

tilemega::frontend::GemmGranularity readGemm(llvm::json::Object const& object) {
  tilemega::frontend::GemmGranularity result;
  result.tile_m = requiredInteger(object, "tile_m");
  result.tile_n = requiredInteger(object, "tile_n");
  result.tile_k = requiredInteger(object, "tile_k");
  result.stages = requiredInteger(object, "stages");
  result.split_k = requiredInteger(object, "split_k");
  return result;
}

std::vector<VariantRequest> readVariants(std::string const& path,
                                         std::size_t gemm_count) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file) throw std::runtime_error("cannot read runtime variant JSON: " + path);
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) throw std::runtime_error("invalid runtime variant JSON: " + path);
  auto* root = parsed->getAsObject();
  if (!root || root->getString("schema") != "tilemega.runtime_variants.v1")
    throw std::runtime_error("unsupported runtime variant schema");
  auto* array = root->getArray("variants");
  if (!array || array->empty()) throw std::runtime_error("variant plan is empty");
  std::vector<VariantRequest> result;
  for (auto const& value : *array) {
    auto* object = value.getAsObject();
    if (!object) throw std::runtime_error("runtime variant is not an object");
    VariantRequest request;
    request.seq_begin = requiredInteger(*object, "seq_begin");
    request.seq_end = requiredInteger(*object, "seq_end");
    if (auto* attention = object->getArray("attention")) {
      for (auto const& value : *attention) {
        auto* choice = value.getAsObject();
        if (!choice) throw std::runtime_error("attention choice must be an object");
        auto stage = requiredInteger(*choice,"stage");
        auto chunks = requiredInteger(*choice,"chunks");
        auto extent = requiredInteger(*choice,"chunk_extent");
        if (stage < 0 || stage > INT_MAX || chunks <= 0 || chunks > INT_MAX ||
            extent <= 0 || extent > INT_MAX)
          throw std::runtime_error("invalid attention stage/chunks/scratch extent");
        request.options.attention.push_back({static_cast<int>(stage),
            {static_cast<std::uint32_t>(chunks),static_cast<std::uint32_t>(extent)}});
      }
    }
    if (auto own = object->getBoolean("rope_tile_per_block"))
      request.options.rope_tile_per_block = *own;
    if (auto own = object->getBoolean("kv_tile_per_block"))
      request.options.kv_tile_per_block = *own;
    if (auto own = object->getBoolean("activation_tile_per_block"))
      request.options.activation_tile_per_block = *own;
    if (auto own = object->getBoolean("combiner_tile_per_block"))
      request.options.combiner_tile_per_block = *own;
    if (auto balanced = object->getBoolean("balanced_placement"))
      request.options.balanced_placement = *balanced;
    if (auto separate = object->getBoolean("separate_residual_tasks"))
      request.options.separate_residual_tasks = *separate;
    if (auto* uniform = object->getObject("uniform")) {
      request.options.gemms.assign(gemm_count, readGemm(*uniform));
    } else if (auto* gemms = object->getArray("gemms")) {
      for (auto const& item : *gemms) {
        auto* gemm = item.getAsObject();
        if (!gemm) throw std::runtime_error("variant GEMM is not an object");
        request.options.gemms.push_back(readGemm(*gemm));
      }
      if (request.options.gemms.size() != gemm_count)
        throw std::runtime_error("variant GEMM list has wrong length");
    } else {
      throw std::runtime_error("variant needs either uniform or gemms plan");
    }
    result.push_back(std::move(request));
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> normalized;
  for(int i=0;i<argc;++i) {
    std::string value=argv[i];auto equals=value.find('=');
    if(value.rfind("--",0)==0 && equals!=std::string::npos) { normalized.push_back(value.substr(0,equals));normalized.push_back(value.substr(equals+1)); }
    else normalized.push_back(value);
  }
  std::vector<char*> pointers;for(auto& value:normalized)pointers.push_back(value.data());
  argc=int(pointers.size());argv=pointers.data();
  tilemega::analysis::IslContext isl_context;
  if (argc < 3 || argc % 2 == 0) {
    std::cerr << "usage: tilemega-compile {EXPORTED_PROGRAM.pt2|STABLE_EXPORT.json|CG.mlir} "
                 "{OUTPUT.cu|OUTPUT.so} [--variants PLAN.json] [--solve TARGET.json --seq N --past N\n"
                 " --solver legacy|skeleton --legacy-seed CG.mlir --k-base 4|8|16|W\n"
                 " --search-passes 1..3 --search-jobs 1 --variant-cache DIR --flow-fixture DIR --flow-search-only 0|1\n"
                 " --serving-pruning 0|1 --incremental-prepare 0|1\n"
                 " --search-capacity N --per-stage-kappa 0|1 --stage-kappa CSV\n"
                 " --segments 1|2 --segment-candidates N\n"
                 " --dump-cg FILE.mlir\n"
                 " --hop-curve FILE.tsv --seq-begin N\n"
                 " --prefetch-page-bytes N]\n";
    return 2;
  }
  try {
    mlir::MLIRContext context;
    context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  context.getOrLoadDialect<tilemega::dialect::ExecDialect>();
    tilemega::frontend::ImportSummary summary;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::filesystem::path input(argv[1]);
    std::string variants_path,solve_target,dump_cg,hop_path,domain_path,rejections_path,evaluation_cases_path;
    std::string serving_phase, emit_mode,measure_command;
    int serving_capacity=1088,serving_batch=1,serving_past_lo=64,
        serving_past_hi=1086,serving_kv_block=256,
        serving_query_rows=64,serving_argmax_tile_n=128;
    bool resource_probes=true;bool dump_evaluated=false;
    std::string solver_mode="skeleton",legacy_seed,variant_cache,flow_fixture;
    int skeleton_k=8,search_passes=3,search_jobs=1,search_top_m=8;
    bool all_workers=false,flow_search_only=false,incremental_prepare=true,
         serving_pruning=true;
    tilemega::solver::SolverTiming solver_timing;
    int interval_begin=0,segments=1,segment_candidates=3;
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> variant_modules;
    tilemega::solver::CompilerSearchOptions solve_options;
    solve_options.placement.dims={4,3,7};
    for (int i=3;i<argc;i+=2) {
      std::string flag=argv[i],value=argv[i+1];
      if (flag=="--variants") variants_path=value;
      else if (flag=="--solver") { solver_mode=value; if(value!="legacy" && value!="skeleton") throw std::runtime_error("unknown solver"); }
      else if (flag=="--legacy-seed") legacy_seed=value;
      else if (flag=="--flow-fixture") flow_fixture=value;
      else if (flag=="--variant-cache") variant_cache=value;
      else if (flag=="--k-base") {all_workers=value=="W";if(!all_workers)skeleton_k=std::stoi(value);}
      else if (flag=="--flow-search-only") flow_search_only=std::stoi(value)!=0;
      else if (flag=="--incremental-prepare") incremental_prepare=std::stoi(value)!=0;
      else if (flag=="--serving-pruning") serving_pruning=std::stoi(value)!=0;
      else if (flag=="--search-passes") search_passes=std::stoi(value);
      else if (flag=="--top-m") search_top_m=std::stoi(value);
      else if (flag=="--search-jobs") search_jobs=std::stoi(value);
      else if (flag=="--solve") solve_target=value;
      else if (flag=="--serving") serving_phase=value;
      else if (flag=="--emit") emit_mode=value;
      else if (flag=="--measure-cmd") measure_command=value;
      else if (flag=="--capacity") serving_capacity=std::stoi(value);
      else if (flag=="--batch") serving_batch=std::stoi(value);
      else if (flag=="--past-range") {
        auto colon=value.find(':');
        if(colon==std::string::npos)throw std::runtime_error("past range needs lo:hi");
        serving_past_lo=std::stoi(value.substr(0,colon));
        serving_past_hi=std::stoi(value.substr(colon+1));
      }
      else if (flag=="--serve-kv-block") serving_kv_block=std::stoi(value);
      else if (flag=="--serve-query-rows") serving_query_rows=std::stoi(value);
      else if (flag=="--serve-argmax-tile-n") serving_argmax_tile_n=std::stoi(value);
      else if (flag=="--seq-begin") interval_begin=std::stoi(value);
      else if (flag=="--segments") segments=std::stoi(value);
      else if (flag=="--segment-candidates") segment_candidates=std::stoi(value);
      else if (flag=="--seq") solve_options.placement.dims.seq=std::stoi(value);
      else if (flag=="--past") solve_options.placement.dims.past=std::stoi(value);
      else if (flag=="--search-capacity") solve_options.capacity=std::stoul(value);
      else if (flag=="--per-stage-kappa") solve_options.per_stage_kappa=std::stoi(value)!=0;
      // Pins the table instead of searching for it, so a chosen coarsening can
      // be built and run as evidence even when the search prefers uniform.
      else if (flag=="--stage-kappa") {
        solve_options.stage_kappa.clear();
        for (std::size_t at=0;at<value.size();) {
          std::size_t const comma=value.find(',',at);
          solve_options.stage_kappa.push_back(std::stoi(value.substr(at,comma-at)));
          at=comma==std::string::npos ? value.size() : comma+1;
        }
      }
      // Zero prices no prefetch at all, which is the pipelining dimension of
      // sigma switched off: every credit and every queue-edge discount is then
      // exactly zero, so a solve can be repeated without it.
      else if (flag=="--prefetch-page-bytes") solve_options.placement.prefetch_page_bytes=std::stoi(value);
      else if (flag=="--dump-cg") dump_cg=value;
      else if (flag=="--hop-curve") hop_path=value;
      else if (flag=="--resource-probes") resource_probes=std::stoi(value)!=0;
      else if (flag=="--dump-evaluated") dump_evaluated=std::stoi(value)!=0;
      else if (flag=="--search-domain") domain_path=value;
      else if (flag=="--evaluate-configs") evaluation_cases_path=value;
      else if (flag=="--numerical-rejections") rejections_path=value;
      else throw std::runtime_error("unknown option: "+flag);
    }
    if(input.extension()==".pt2") {
      auto bridge=std::filesystem::absolute(std::string(argv[2])+".export.json");
      std::string python_path=std::string(TILEMEGA_SOURCE_DIR)+"/python";
      if(auto inherited=std::getenv("PYTHONPATH"))python_path+=":"+std::string(inherited);
      std::string command="PYTHONPATH="+quote(python_path)+" python3 -m tilemega.export_bridge "+
          quote(std::filesystem::absolute(input).string())+" --out "+quote(bridge.string());
      if(std::system(command.c_str())!=0)throw std::runtime_error("torch.export bridge failed");
      input=bridge;
      std::cerr<<"EXPORT_BRIDGE source="<<std::quoted(argv[1])<<" output="<<std::quoted(input.string())<<'\n';
    }
    bool has_variants=!variants_path.empty();
    bool serving=!serving_phase.empty();
    if(serving && serving_phase!="decode" && serving_phase!="prefill")
      throw std::runtime_error("--serving expects decode or prefill");
    if(!emit_mode.empty() && emit_mode!="serving")
      throw std::runtime_error("--emit expects serving");
    if(serving != (emit_mode=="serving"))
      throw std::runtime_error("--serving and --emit serving must be paired");
    if(serving && (serving_batch<1 || serving_batch>64 ||
                   serving_past_lo<0 || serving_past_hi<serving_past_lo))
      throw std::runtime_error("invalid serving batch or past range");
    if(search_top_m<1 || search_top_m>8)
      throw std::runtime_error("--top-m must be in 1..8");
    if(serving && !solve_target.empty() && !flow_search_only &&
       measure_command.empty())
      throw std::runtime_error("serving solve requires --measure-cmd for the top-3 decision");
    if (!solve_target.empty() && has_variants)
      throw std::runtime_error("--solve chooses variants; cannot combine with --variants");
    std::string source,selected_serving_mode,selected_serving_binary;
    if(serving && solve_target.empty()) {
      if(has_variants)throw std::runtime_error("serving needs one exported model or solved CG");
      if(input.extension()==".mlir") {
        module=mlir::parseSourceFile<mlir::ModuleOp>(input.string(),&context);
        if(!module || !(*module)->hasAttr("tilemega.serving"))
          throw std::runtime_error("serving CG is missing its serving schema");
        source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(
            {{*module,static_cast<std::uint32_t>(serving_phase=="decode"?1:64),
                       static_cast<std::uint32_t>(serving_phase=="decode"?1:64)}});
      }else {
      auto bridge=tilemega::frontend::ReadExportBridge(input.string());
      tilemega::frontend::ServingOptions options;
      options.phase=serving_phase=="decode"
          ? tilemega::frontend::ServingOptions::Phase::kDecode
          : tilemega::frontend::ServingOptions::Phase::kPrefill;
      options.seq=serving_phase=="decode" ? 1 : 64;
      options.capacity=serving_capacity;
      options.kv_block=serving_kv_block;
      options.query_rows=serving_query_rows;
      options.argmax_tile_n=serving_argmax_tile_n;
      auto plan=tilemega::frontend::BuildModelPlan(
          bridge.nodes,bridge.inputs,bridge.outputs,options);
      tilemega::frontend::ImportOptions import;
      import.gemms.assign(plan.gemms.size(),{16,128,128,2,1});
      module=tilemega::frontend::TorchExportImporter{}.ImportPlan(
          input.string(),plan,context,&summary,import);
      source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(
          {{*module,static_cast<std::uint32_t>(options.seq),
                     static_cast<std::uint32_t>(options.seq)}});
      std::cerr<<"SERVING_SEED phase="<<serving_phase<<" batch="<<serving_batch
               <<" past="<<serving_past_lo<<':'<<serving_past_hi
               <<" stages="<<summary.stages<<'\n';
      }
    } else if (!solve_target.empty()) {
      if(serving && solver_mode!="skeleton")
        throw std::runtime_error("serving plans require the skeleton solver");
      if (input.extension()==".mlir")
        throw std::runtime_error("automatic geometry search requires export JSON; use tilemega-opt for placement-only CG solving");
      solve_options.placement.target=tilemega::TargetSpec::FromJson(solve_target);
      if(serving) {
        auto& d=solve_options.placement.dims;
        d.batch=serving_batch;d.seq=serving_phase=="decode"?1:64;
        d.past=serving_phase=="decode"?(serving_past_lo+serving_past_hi)/2:0;
        d.total=d.seq+d.past;
      }
      if (!domain_path.empty()) {
        auto file=llvm::MemoryBuffer::getFile(domain_path);
        if (!file) throw std::runtime_error("cannot read calibration search domain");
        auto value=llvm::json::parse(file.get()->getBuffer());
        auto* object=value ? value->getAsObject() : nullptr;
        auto* shapes=object ? object->getArray("geometries") : nullptr;
        if (!shapes || shapes->empty()) throw std::runtime_error("search domain needs calibrated geometries");
        for (auto const& shape:*shapes) {
          auto* o=shape.getAsObject();if (!o) throw std::runtime_error("invalid geometry domain entry");
          solve_options.geometry_domain.push_back({int(requiredInteger(*o,"tile_m")),int(requiredInteger(*o,"tile_n")),
              int(requiredInteger(*o,"tile_k")),int(requiredInteger(*o,"stages")),1});
        }
      }
      if(!rejections_path.empty()) {
        auto file=llvm::MemoryBuffer::getFile(rejections_path);
        if(!file)throw std::runtime_error("cannot read numerical exclusions");
        auto value=llvm::json::parse(file.get()->getBuffer());auto* object=value ? value->getAsObject() : nullptr;
        auto* entries=object ? object->getArray("rejected") : nullptr;
        if(!entries || requiredInteger(*object,"seq")!=solve_options.placement.dims.seq ||
            requiredInteger(*object,"past")!=solve_options.placement.dims.past)
          throw std::runtime_error("numerical exclusion theta does not match the request");
        auto fingerprint=object->getString("model_sha256");
        if(!fingerprint)throw std::runtime_error("numerical exclusion requires its model fingerprint");
        if(modelFingerprint(argv[1])!=*fingerprint)
          throw std::runtime_error("numerical exclusion model fingerprint does not match the request");
        for(auto const& entry:*entries) {
          auto* o=entry.getAsObject();if(!o || !o->getString("reason") || o->getString("reason")->empty())
            throw std::runtime_error("numerical exclusion requires its raw-evidence reason");
          solve_options.numerical_rejections.push_back({{int(requiredInteger(*o,"tile_m")),int(requiredInteger(*o,"tile_n")),
              int(requiredInteger(*o,"tile_k")),int(requiredInteger(*o,"stages")),int(requiredInteger(*o,"split_k"))},o->getString("reason")->str()});
        }
      }
      auto& dims=solve_options.placement.dims;dims.total=dims.seq+dims.past;
      if (!hop_path.empty()) {
        std::string error;
        if (!tilemega::solver::HopCurve::FromTsv(hop_path,&solve_options.placement.hop,&error))
          throw std::runtime_error(error);
      }
      std::ofstream evidence(std::string(argv[2])+".search.tsv");
      if (!evidence) throw std::runtime_error("cannot open search evidence");
      auto resource_root=std::filesystem::absolute(std::string(argv[2])+".resources") /
          std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
      auto library=std::filesystem::canonical(argv[0]).parent_path().parent_path()/"libtilemega.a";
      int probe_index=0;
      solve_options.keep_evaluated=dump_evaluated;
      if (resource_probes) solve_options.query_residency=[&](mlir::ModuleOp m,int kappa) {
        return queryResidency(m,kappa,solve_options,resource_root/std::to_string(probe_index++),library);
      };
      solve_options.timing=&solver_timing;
      tilemega::solver::CompilerSearchResult solved;
      if(solver_mode=="legacy")solved=tilemega::solver::SolveExport(input.string(),context,solve_options,&summary,evidence);
      else {
        tilemega::solver::SkeletonSearchOptions skeleton;skeleton.common=solve_options;
        skeleton.k_base=skeleton_k;skeleton.all_workers=all_workers;
        skeleton.passes=search_passes;skeleton.jobs=search_jobs;
        skeleton.artifact_prefix=argv[2];skeleton.fixture=flow_fixture;
        skeleton.search_only=flow_search_only;
        skeleton.incremental_prepare=incremental_prepare;
        skeleton.serving_pruning=serving_pruning;
        skeleton.top_m=search_top_m;
        if(!evaluation_cases_path.empty()) {
          if(!flow_search_only)throw std::runtime_error("--evaluate-configs requires --flow-search-only 1");
          auto file=llvm::MemoryBuffer::getFile(evaluation_cases_path);
          if(!file)throw std::runtime_error("cannot read evaluation cases");
          auto parsed=llvm::json::parse(file.get()->getBuffer());
          auto* object=parsed?parsed->getAsObject():nullptr;
          auto* cases=object?object->getArray("cases"):nullptr;
          if(!cases || cases->empty())throw std::runtime_error("evaluation cases must be nonempty");
          for(auto const& item:*cases) {
            auto* entry=item.getAsObject();
            auto* geometry=entry?entry->getArray("geometries"):nullptr;
            if(!geometry || geometry->empty())throw std::runtime_error("evaluation case needs geometries");
            tilemega::solver::SkeletonEvaluationCase test;
            test.kappa=int(requiredInteger(*entry,"kappa"));
            test.residency=int(requiredInteger(*entry,"residency"));
            for(auto const& value:*geometry) {
              auto* g=value.getAsObject();
              if(!g)throw std::runtime_error("invalid evaluation geometry");
              test.config.push_back({int(requiredInteger(*g,"tile_m")),
                  int(requiredInteger(*g,"tile_n")),int(requiredInteger(*g,"tile_k")),
                  int(requiredInteger(*g,"stages")),int(requiredInteger(*g,"split_k"))});
            }
            skeleton.evaluation_cases.push_back(std::move(test));
          }
        }
        if(serving && serving_phase=="decode") {
          skeleton.serving_past_lo=serving_past_lo;
          skeleton.serving_past_hi=serving_past_hi;
        }
        std::optional<tilemega::frontend::ImportedSemantics> serving_imported;
        mlir::OwningOpRef<mlir::ModuleOp> seed;
        if(serving) {
          if(!legacy_seed.empty())throw std::runtime_error("serving search does not use a legacy seed");
          auto bridge=tilemega::frontend::ReadExportBridge(input.string());
          tilemega::frontend::ServingOptions options;
          options.phase=serving_phase=="decode"
              ? tilemega::frontend::ServingOptions::Phase::kDecode
              : tilemega::frontend::ServingOptions::Phase::kPrefill;
          options.seq=dims.seq;options.capacity=serving_capacity;
          options.kv_block=serving_kv_block;options.query_rows=serving_query_rows;
          options.argmax_tile_n=serving_argmax_tile_n;
          auto plan=tilemega::frontend::BuildModelPlan(bridge.nodes,bridge.inputs,
              bridge.outputs,options);
          serving_imported.emplace(tilemega::frontend::TorchExportImporter{}.
              ImportSemantics(input.string(),plan,context));
          skeleton.seed={16,128,128,2,1};skeleton.kappa=1;
        }else {
          if(!legacy_seed.empty()) {
            seed=mlir::parseSourceFile<mlir::ModuleOp>(legacy_seed,&context);
            if(!seed)throw std::runtime_error("cannot read legacy seed CG");
          } else {
            tilemega::solver::SolverTiming seed_timing;auto seed_options=solve_options;seed_options.timing=&seed_timing;
            std::ofstream seed_evidence(std::string(argv[2])+".seed.search.tsv");
            auto legacy=tilemega::solver::SolveExport(input.string(),context,seed_options,&summary,seed_evidence);
            seed=std::move(legacy.module);
            std::ofstream seed_times(std::string(argv[2])+".seed.timing.tsv");
            seed_timing.Write(seed_times,"legacy-seed",input.stem().string(),dims.seq);
          }
          auto runtime=tilemega::codegen::ReadRuntimePlan(*seed);
          if(runtime.gemms.empty())throw std::runtime_error("legacy seed has no GEMM geometry");
          auto const& g=runtime.gemms.front();
          for(auto const& other:runtime.gemms)
            if(std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k)!=std::tie(other.tile_m,other.tile_n,other.tile_k,other.stages,other.split_k))
              throw std::runtime_error("coordinate descent needs a uniform legacy seed");
          skeleton.seed={g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k};
          auto kappa=(*seed)->getAttrOfType<mlir::IntegerAttr>("tmexec.solved_kappa");
          skeleton.kappa=kappa ? int(kappa.getInt()):1;
          if(auto r=(*seed)->getAttrOfType<mlir::IntegerAttr>("tmexec.solved_residency"))skeleton.seed_residency=r.getInt();
        }
        if(variant_cache.empty())variant_cache=(resource_root.parent_path()/"variant_cache").string();
        if(serving && solve_options.geometry_domain.empty()) {
          // Exact per-variant ptxas resources remain part of the search.  A
          // cold serial first class scan otherwise spends its entire budget
          // compiling wrappers; populate the shared cache concurrently first.
          int max_m=16;
          while(max_m<dims.batch*dims.seq && max_m<128)max_m*=2;
          auto prewarm_dir=resource_root/"prewarm";
          std::filesystem::create_directories(resource_root);
          std::string command="python3 "+quote(std::string(TILEMEGA_SOURCE_DIR)+
              "/tools/prewarm_serving_variants.py")+
              " --target "+quote(solve_target)+
              " --cache "+quote(variant_cache)+
              " --output "+quote(prewarm_dir.string())+
              " --max-m "+std::to_string(max_m)+" --jobs 8";
          std::ofstream(resource_root/"prewarm.command.txt")<<command<<'\n';
          if(std::system((command+" >"+quote((resource_root/"prewarm.log").string())+
              " 2>&1").c_str()))
            throw std::runtime_error("serving variant prewarm failed: "+
                (resource_root/"prewarm.log").string());
        }
        int variant_index=0;
        std::map<std::tuple<int,int,int,int,int>,tilemega::solver::VariantResources> probed_bodies;
        skeleton.variant_probe=[&](std::string const&,tilemega::solver::GemmConfig const* tile,tilemega::solver::ScalarType dtype) {
          // The compiled TaskBody template has no class or split-K parameter.
          // Keep logical variant keys above, but reuse its identical probe.
          auto body=std::make_tuple(tile?tile->tile_m:0,tile?tile->tile_n:0,
              tile?tile->tile_k:0,tile?tile->stages:0,int(dtype));
          if(auto found=probed_bodies.find(body);found!=probed_bodies.end()) {
            auto reused=found->second;reused.compiled=false;return reused;
          }
          auto output=resource_root/("variant_"+std::to_string(variant_index++)+".json");
          auto log=output;log.replace_extension("log");std::filesystem::create_directories(resource_root);
          std::string command="python3 "+quote(std::string(TILEMEGA_SOURCE_DIR)+"/tools/probe_variant.py")+
            " --cache "+quote(variant_cache)+" --output "+quote(output.string())+" --arch "+quote(solve_options.placement.target.NvccArch())+
            " --dtype "+std::string(dtype==tilemega::solver::ScalarType::kBF16 ? "bf16":"f32");
          if(serving) {
            command+=" --serving";
            if(!tile) {
              auto found=std::find_if(serving_imported->plan.stages.begin(),
                  serving_imported->plan.stages.end(),[](auto const& stage){
                    return stage.kind==tilemega::frontend::PlanTaskKind::kFusedAttention;});
              if(found==serving_imported->plan.stages.end())
                throw std::runtime_error("serving resource probe lacks attention stage");
              command+=" --head-dim "+std::to_string(found->width)+
                  " --qperkv "+std::to_string(found->group);
            }
          }
          if(tile)command+=" --tile "+quote(std::to_string(tile->tile_m)+","+std::to_string(tile->tile_n)+","+std::to_string(tile->tile_k)+","+std::to_string(tile->stages));
          else command+=" --nongemm";
          if(std::system((command+" >"+quote(log.string())+" 2>&1").c_str()))throw std::runtime_error("variant probe failed: "+log.string());
          auto file=llvm::MemoryBuffer::getFile(output.string());if(!file)throw std::runtime_error("missing variant resource result");
          auto value=llvm::json::parse(file.get()->getBuffer());auto* object=value?value->getAsObject():nullptr;
          if(!object)throw std::runtime_error("invalid variant resource JSON");
          auto resource=tilemega::solver::VariantResources{int(requiredInteger(*object,"registers")),int(requiredInteger(*object,"shared_bytes")),int(requiredInteger(*object,"threads")),object->getBoolean("compiled").value_or(false)};
          probed_bodies.emplace(body,resource);return resource;
        };
        auto result=serving
            ? tilemega::solver::SolveSkeletonImported(*serving_imported,context,skeleton,&summary,evidence)
            : tilemega::solver::SolveSkeletonExport(input.string(),context,skeleton,&summary,evidence);
        if(flow_search_only) {
          std::ofstream events(std::string(argv[2])+".phases.tsv");solver_timing.WriteEvents(events);
          std::ofstream times(std::string(argv[2])+".timing.tsv");solver_timing.Write(times,solver_mode,input.stem().string(),dims.seq);
          std::ofstream ranked(std::string(argv[2])+".flow_ranked.tsv");ranked<<std::setprecision(17)<<"rank\tkey\tflow_ns\tresidency\tkappa\terror\n";
          for(std::size_t i=0;i<result.evaluated.size();++i){auto const& c=result.evaluated[i];ranked<<i+1<<'\t'<<c.key<<'\t'<<c.score<<'\t'<<c.residency<<'\t'<<c.kappa<<'\t'<<c.error<<'\n';}
          return 0;
        }
        solved=std::move(result.compiled);
        std::ofstream events(std::string(argv[2])+".phases.tsv");solver_timing.WriteEvents(events);
      }
      std::ofstream timing_file(std::string(argv[2])+".timing.tsv");
      solver_timing.Write(timing_file,solver_mode,input.stem().string(),dims.seq);
      std::ofstream shortlist(std::string(argv[2])+".top3.tsv");
      shortlist << "rank\tkey\tplacement\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\tkappa\tresidency\tfloor_ns\tpredicted_ns\tsource\tcg\n";
      for (std::size_t i=0;i<solved.shortlist.size();++i) {
        auto const& entry=solved.shortlist[i];auto const& e=entry.evaluation;
        auto const& g=e.candidate.config;
        std::string stem=std::string(argv[2])+".top"+std::to_string(i+1);
        std::vector<tilemega::codegen::RuntimeVariantModule> variants{{*entry.module,1u,
            static_cast<std::uint32_t>(dims.seq)}};
        std::ofstream(stem+".cu") << tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(variants);
        std::error_code ec;llvm::raw_fd_ostream cg(stem+".mlir",ec);
        if (ec) throw std::runtime_error("cannot write shortlisted CG");
        (*entry.module).print(cg);
        shortlist << i+1 << '\t' << e.candidate.key << '\t' << e.placement << '\t'
            << g.tile_m << '\t' << g.tile_n << '\t' << g.tile_k << '\t' << g.stages << '\t' << g.split_k << '\t'
            << e.candidate.kappa << '\t' << e.candidate.ctas_per_sm << '\t' << e.floor_ns << '\t' << e.makespan_ns
            << '\t' << stem << ".cu\t" << stem << ".mlir\n";
      }
      if(serving && !measure_command.empty()) {
        double fastest=std::numeric_limits<double>::infinity();
        std::size_t fastest_index=0;
        std::ofstream selected(std::string(argv[2])+".top3_measured.tsv");
        selected<<"rank\tmode\tmean_ms\tso\n";
        std::vector<std::string> candidate_sos;
        for(std::size_t i=0;i<solved.shortlist.size();++i) {
          std::string stem=std::string(argv[2])+".top"+std::to_string(i+1);
          std::string candidate_so=stem+".candidate.so";
          std::string compile=quote(std::filesystem::canonical(argv[0]).string())+
              " "+quote(stem+".mlir")+" "+quote(candidate_so)+
              " --serving "+quote(serving_phase)+" --emit serving"+
              " --batch "+std::to_string(serving_batch)+
              " --past-range "+quote(std::to_string(serving_past_lo)+":"+
                                    std::to_string(serving_past_hi))+
              " --capacity "+std::to_string(serving_capacity);
          if(std::system((compile+" >"+quote(stem+".build.stdout")+
              " 2>"+quote(stem+".build.stderr")).c_str()))
            throw std::runtime_error("top-3 serving candidate compilation failed: "+stem);
          candidate_sos.push_back(candidate_so);
        }
        // Compile every candidate before timing any of them.  Each round
        // rotates the order, so a warm/cool device does not systematically
        // favor a particular rank.  Keep all 32-step raw CUDA-event samples.
        std::map<std::pair<std::size_t,std::string>,std::vector<double>> samples;
        std::ofstream rounds(std::string(argv[2])+".top3_measure_rounds.tsv");
        rounds<<"round\trank\tmode\tmean_ms\tartifact\n";
        for(int round=0;round<3;++round)for(std::size_t position=0;
            position<candidate_sos.size();++position) {
          std::size_t i=(position+std::size_t(round))%candidate_sos.size();
          std::string stem=std::string(argv[2])+".top"+std::to_string(i+1);
          std::string artifact=stem+".measurement.r"+std::to_string(round);
          std::string measure=measure_command+" --so "+quote(candidate_sos[i])+
              " --batch "+std::to_string(serving_batch)+
              " --past-mid "+std::to_string(dims.past)+
              " --out "+quote(artifact);
          if(round&1)measure+=" --reverse-modes";
          if(std::system((measure+" >"+quote(artifact+".stdout")+
              " 2>"+quote(artifact+".stderr")).c_str()))
            throw std::runtime_error("top-3 serving candidate measurement failed: "+artifact);
          auto measured_file=llvm::MemoryBuffer::getFile(artifact+"/measurements.json");
          if(!measured_file)throw std::runtime_error("missing top-3 measurement output");
          auto measured=llvm::json::parse(measured_file.get()->getBuffer());
          auto* object=measured?measured->getAsObject():nullptr;
          auto* modes=object?object->getObject("modes"):nullptr;
          if(!modes)throw std::runtime_error("top-3 measurement has no mode table");
          for(auto const& mode:{"L1","L2"})
            if(auto* item=modes->getObject(mode))if(auto mean=item->getNumber("mean_ms")) {
              samples[{i,mode}].push_back(*mean);
              rounds<<round<<'\t'<<i+1<<'\t'<<mode<<'\t'<<*mean<<'\t'
                    <<artifact<<"/measurements.json\n";
            }
          rounds.flush();
        }
        for(auto& [key,values]:samples) {
          if(values.size()!=3)throw std::runtime_error("top-3 mode lacks three measurement rounds");
          std::sort(values.begin(),values.end());
          double median=values[1];
          selected<<key.first+1<<'\t'<<key.second<<'\t'<<median<<'\t'
                  <<candidate_sos[key.first]<<'\n';
          if(median<fastest) {
            fastest=median;fastest_index=key.first;
            selected_serving_mode=key.second;
          }
        }
        if(!std::isfinite(fastest))throw std::runtime_error("top-3 measurements contain no timing");
        solved.module=mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::cast<mlir::ModuleOp>(solved.shortlist[fastest_index].module->clone()));
        solved.winner=solved.shortlist[fastest_index].evaluation.candidate;
        selected_serving_binary=std::string(argv[2])+".top"+
            std::to_string(fastest_index+1)+".candidate.so";
        std::cerr<<"SERVING_TOP3_WINNER rank="<<fastest_index+1<<" mode="
                 <<selected_serving_mode<<" mean_ms="<<fastest<<'\n';
      }
      if (dump_evaluated) {
        // §6 C1-c measures the search's choice against everything the search
        // priced, so every evaluated candidate needs a buildable source, not
        // just the three the shortlist keeps.
        std::ofstream table(std::string(argv[2])+".evaluated.tsv");
        table << "index\tkey\tplacement\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\tkappa\tresidency\tfloor_ns\tpredicted_ns\tsource\tcg\n";
        for (std::size_t i=0;i<solved.evaluated.size();++i) {
          auto const& entry=solved.evaluated[i];auto const& e=entry.evaluation;
          auto const& g=e.candidate.config;
          std::string stem=std::string(argv[2])+".cand"+std::to_string(i);
          std::vector<tilemega::codegen::RuntimeVariantModule> variants{{*entry.module,1u,
              static_cast<std::uint32_t>(dims.seq)}};
          std::ofstream(stem+".cu") << tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(variants);
          std::error_code ec;llvm::raw_fd_ostream cg(stem+".mlir",ec);
          if (ec) throw std::runtime_error("cannot write evaluated CG");
          (*entry.module).print(cg);
          table << i << '\t' << e.candidate.key << '\t' << e.placement << '\t'
              << g.tile_m << '\t' << g.tile_n << '\t' << g.tile_k << '\t' << g.stages << '\t'
              << g.split_k << '\t' << e.candidate.kappa << '\t' << e.candidate.ctas_per_sm << '\t'
              << e.floor_ns << '\t' << e.makespan_ns << '\t' << stem << ".cu\t" << stem << ".mlir\n";
        }
      }
      std::ofstream outer(std::string(argv[2])+".bounds.tsv");
      outer<<"candidate\twork_lb_ns\tcp_lb_ns\tqueue_lb_lb_ns\tpriority_ns\n";
      outer<<std::setprecision(17);
      for(auto const& candidate:solved.outer_candidates)
        outer<<candidate.key<<'\t'<<candidate.work_lb_ns<<'\t'<<candidate.cp_lb_ns
             <<'\t'<<candidate.queue_lb_lb_ns<<'\t'<<candidate.priority_ns<<'\n';
      module=std::move(solved.module);
      if(interval_begin) {
        auto interval_options=solve_options.placement;
        auto integer=[&](char const* key){return int((*module)->getAttrOfType<mlir::IntegerAttr>(key).getInt());};
        interval_options.residency=integer("tmexec.solved_residency");
        interval_options.verified_resident_limit=interval_options.residency;
        interval_options.kappa=integer("tmexec.solved_kappa");
        interval_options.requested_grid=integer("tmexec.solved_grid");
        std::ofstream interval_evidence(std::string(argv[2])+".interval.tsv");
        interval_evidence<<"seq\tplacement\tfloor_ns\tpredicted_ns\terror\n";
        tilemega::dialect::SolveAndWritePlacementInterval(*module,interval_options,interval_begin,dims.seq,&interval_evidence);
        if (segments>1) {
          // The fixed geometry the search chose is written beside the segmented
          // build from the same solve, so the comparison is one invocation.
          std::vector<tilemega::codegen::RuntimeVariantModule> fixed{{*module,
              1u,static_cast<std::uint32_t>(dims.seq)}};
          std::ofstream(std::string(argv[2])+".fixed.cu")
              << tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(fixed);
          std::ofstream segment_evidence(std::string(argv[2])+".segments.tsv");
          auto cut=tilemega::solver::SolveIntervalSegments(input.string(),context,
              solve_options,solved,*module,interval_begin,dims.seq,segments,
              std::size_t(segment_candidates),segment_evidence);
          for (auto const& [geometry,reason]:cut.refused)
            std::cerr << "SEGMENT_REFUSED " << tilemega::solver::GeometryKey(geometry)
                      << ' ' << reason << "\n";
          std::cerr << "SEGMENT_SUMMARY candidates=" << cut.candidates.size()
              << " points=" << dims.seq-interval_begin+1
              << " winner=" << tilemega::solver::GeometryKey(solved.winner.config)
              << " winner_ns=" << cut.winner_ns
              << " fixed=" << tilemega::solver::GeometryKey(cut.fixed_config)
              << " fixed_ns=" << cut.fixed_ns
              << " segmented_ns=" << cut.segmented_ns << " cut=" << cut.cut
              << " segments=" << cut.segments.size() << "\n";
          module=std::move(cut.segments.front().module);
          for (std::size_t s=1;s<cut.segments.size();++s)
            variant_modules.push_back(std::move(cut.segments[s].module));
          std::vector<tilemega::codegen::RuntimeVariantModule> pieces{{*module,1u,
              static_cast<std::uint32_t>(cut.segments.front().end)}};
          for (std::size_t s=1;s<cut.segments.size();++s)
            pieces.push_back({*variant_modules[s-1],
                static_cast<std::uint32_t>(cut.segments[s].begin),
                static_cast<std::uint32_t>(cut.segments[s].end)});
          source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(pieces);
        }
      }
      if (source.empty()) {
        std::vector<tilemega::codegen::RuntimeVariantModule> inputs{{*module,
            1u,static_cast<std::uint32_t>(dims.seq)}};
        source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(inputs);
      }
      std::cerr << "SOLVE_SUMMARY evaluated=" << solved.stats.evaluated
          << " deferred=" << solved.stats.capacity_deferred
          << " residency_scope=" << (resource_probes ? "compiled" : "1_degraded") << " hop_calibrated=" << !hop_path.empty() << "\n";
      if (solve_options.per_stage_kappa || !solve_options.stage_kappa.empty()) {
        std::cerr << "SOLVE_STAGE_KAPPA stages=" << solved.stage_count
                  << " moves=" << solved.stage_kappa_moves
            << " uniform_ns=" << solved.uniform_ns
            << " per_stage_ns=" << solved.per_stage_ns << " table=";
        for (std::size_t i=0;i<solved.stage_kappa.size();++i)
          std::cerr << (i ? "," : "") << solved.stage_kappa[i];
        std::cerr << (solved.stage_kappa.empty() ? "uniform" : "") << "\n";
      }
    } else if (input.extension() == ".mlir") {
      if (has_variants)
        throw std::runtime_error("runtime variants require stable export JSON input");
      module = mlir::parseSourceFile<mlir::ModuleOp>(input.string(), &context);
      if (!module) throw std::runtime_error("cannot parse CG MLIR input");
      for (auto task : module->getOps<tilemega::dialect::TileSpaceOp>()) {
        ++summary.task_spaces;
        summary.stages = std::max(summary.stages,
            static_cast<std::size_t>(task.getStage() + 1));
      }
      for (auto coupling : module->getOps<tilemega::dialect::CouplingOp>())
        ++summary.couplings;
      if (auto guards = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
              "tilemega.guard_count"))
        summary.guards = guards.getInt();
      source = tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
    } else if (!has_variants) {
      module = tilemega::frontend::TorchExportImporter{}.Import(input.string(), context, &summary);
      source = tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
    } else {
      auto bridge = tilemega::frontend::ReadExportBridge(input.string());
      auto plan = tilemega::frontend::BuildModelPlan(
          bridge.nodes, bridge.inputs, bridge.outputs);
      auto requests = readVariants(variants_path, plan.gemms.size());
      std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
      std::vector<tilemega::codegen::RuntimeVariantModule> inputs;
      modules.reserve(requests.size());
      inputs.reserve(requests.size());
      for (std::size_t i = 0; i < requests.size(); ++i) {
        modules.push_back(tilemega::frontend::TorchExportImporter{}.Import(
            input.string(), context, i == 0 ? &summary : nullptr,
            requests[i].options));
        inputs.push_back({*modules.back(), requests[i].seq_begin,
                          requests[i].seq_end});
      }
      source = tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(inputs);
    }
    if (!dump_cg.empty()) {
      if (!module) throw std::runtime_error("--dump-cg requires a single module");
      std::error_code error;llvm::raw_fd_ostream dump(dump_cg,error);
      if (error) throw std::runtime_error("cannot write CG dump: "+error.message());
      module->print(dump);dump << "\n";
      for (std::size_t v=0;v<variant_modules.size();++v) {
        std::filesystem::path stem(dump_cg);stem.replace_extension();
        std::string name=stem.string()+".segment"+std::to_string(v+1)+".mlir";
        llvm::raw_fd_ostream piece(name,error);
        if (error) throw std::runtime_error("cannot write segment CG: "+error.message());
        (*variant_modules[v]).print(piece);piece << "\n";
      }
    }
    if (serving) {
      if (serving_phase == "prefill")
        serving_past_lo = serving_past_hi = 0;
      source = "#define TILEMEGA_SERVING_BATCH_LO " +
          std::to_string(serving_batch) + "\n" +
          "#define TILEMEGA_SERVING_BATCH_HI " +
          std::to_string(serving_batch) + "\n" +
          "#define TILEMEGA_SERVING_PAST_LO " +
          std::to_string(serving_past_lo) + "\n" +
          "#define TILEMEGA_SERVING_PAST_HI " +
          std::to_string(serving_past_hi) + "\n" + source;
    }
    std::filesystem::path requested(argv[2]);
    bool shared = requested.extension() == ".so";
    std::filesystem::path cuda = shared
        ? std::filesystem::path(requested.string() + ".cu") : requested;
    std::ofstream output(cuda);
    if (!output) throw std::runtime_error("cannot open generated CUDA output");
    output << source;
    output.close();
    if (shared) {
      // The winner was already compiled for the GPU timing gate.  Reuse that
      // exact binary only when the final generated CUDA is byte-identical.
      // This saves the otherwise redundant full megakernel nvcc invocation.
      bool reused=false;
      if(!selected_serving_binary.empty() &&
         std::filesystem::exists(selected_serving_binary) &&
         std::filesystem::exists(selected_serving_binary+".cu")) {
        std::ifstream built(selected_serving_binary+".cu",std::ios::binary);
        std::string compiled_source((std::istreambuf_iterator<char>(built)),
                                    std::istreambuf_iterator<char>());
        if(compiled_source==source) {
          std::filesystem::copy_file(selected_serving_binary,requested,
              std::filesystem::copy_options::overwrite_existing);
          for(auto const& suffix:{".ptxas.log",".build_command.txt"})
            if(std::filesystem::exists(selected_serving_binary+suffix))
              std::filesystem::copy_file(selected_serving_binary+suffix,
                  requested.string()+suffix,
                  std::filesystem::copy_options::overwrite_existing);
          std::ofstream(requested.string()+".binary_reuse.txt")
              <<"source_byte_identical="<<selected_serving_binary
              <<".cu\ncompiled_binary="<<selected_serving_binary<<'\n';
          reused=true;
        }
      }
      if(!reused) {
      std::string root = TILEMEGA_SOURCE_DIR;
      std::string nvcc = std::getenv("CUDACXX") ? std::getenv("CUDACXX") :
                                                 "/usr/local/cuda/bin/nvcc";
      std::string arch = tilemega::TargetSpec::Probe().NvccArch();
      std::string command = quote(nvcc) +
          " -std=c++17 -O3 -DTILEMEGA_MIDPOINT_REFINE=0 -arch="+
          quote(arch)+" --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -cudart shared -Xptxas=-v -x cu" +
          " -I" + quote(root + "/include") +
          " -I" + quote(root + "/third_party/cutlass/include") +
          " -I" + quote(root + "/third_party/cutlass/tools/util/include") +
          " -I" + quote(root + "/third_party/cutlass/test") + " " +
          quote(cuda.string()) + " -x cu " +
          quote(root + "/lib/Target/TargetSpec.cpp") +
          " -x cu " + quote(root + "/lib/Support/Json.cpp") +
          " -x cu " + quote(root + "/lib/Codegen/RuntimeTaskGraph.cpp") +
          " -x cu " + quote(root + "/lib/Solver/PlanMaterialize.cpp") +
          " -x cu " + quote(root + "/lib/Dialect/CouplingGraph/PlacementPlan.cpp") +
          " -x cu " + quote(root + "/lib/Solver/BalancedPlacement.cpp") +
          " -x cu " + quote(root + "/lib/Solver/ListScheduler.cpp") +
          " -L/usr/local/cuda/lib64 -lcudart -o " + quote(requested.string());
      std::ofstream(requested.string()+".build_command.txt") << command << '\n';
      int status = std::system((command+" >"+quote(requested.string()+".ptxas.log")+
          " 2>&1").c_str());
      if (status != 0) throw std::runtime_error("nvcc failed while building shared object");
      }
    }
    if(serving && module) {
      auto runtime=tilemega::codegen::ReadRuntimePlan(*module);
      auto integer=[&](char const* key,int fallback) {
        if(auto value=(*module)->getAttrOfType<mlir::IntegerAttr>(key))
          return int(value.getInt());
        return fallback;
      };
      std::ofstream manifest(requested.string()+".plan.json");
      manifest<<"{\n  \"model\": "<<std::quoted(input.stem().string())
              <<",\n  \"phase\": "<<std::quoted(serving_phase)
              <<",\n  \"batch_lo\": "<<serving_batch
              <<",\n  \"batch_hi\": "<<serving_batch
              <<",\n  \"past_lo\": "<<serving_past_lo
              <<",\n  \"past_hi\": "<<serving_past_hi
              <<",\n  \"seq\": "<<(serving_phase=="decode"?1:64)
              <<",\n  \"capacity\": "<<serving_capacity
              <<",\n  \"mode\": "<<std::quoted(selected_serving_mode.empty()?"unmeasured":selected_serving_mode)
              <<",\n  \"grid\": "<<integer("tmexec.solved_grid",0)
              <<",\n  \"residency\": "<<integer("tmexec.solved_residency",0)
              <<",\n  \"kappa\": "<<integer("tmexec.solved_kappa",1)
              <<",\n  \"gemms\": [\n";
      for(std::size_t i=0;i<runtime.gemms.size();++i) {
        auto const& g=runtime.gemms[i];
        manifest<<"    {\"index\": "<<i<<", \"tile_m\": "<<g.tile_m
                <<", \"tile_n\": "<<g.tile_n<<", \"tile_k\": "<<g.tile_k
                <<", \"stages\": "<<g.stages<<", \"split_k\": "<<g.split_k<<"}"
                <<(i+1==runtime.gemms.size()?"\n":",\n");
      }
      manifest<<"  ]\n}\n";
    }
    std::cerr << "CODEGEN_SUMMARY tasks=" << summary.task_spaces
              << " couplings=" << summary.couplings
              << " stages=" << summary.stages
              << " symbolic_windows=" << summary.symbolic_windows
              << " fallback_windows=" << summary.fallback_windows
              << " output=" << requested.string() << "\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-compile: " << error.what() << "\n";
    return 1;
  }
}
