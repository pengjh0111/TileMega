#include <tilemega/Codegen/ServingPages.h>
#include "Toolchain.h"
// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Solver/CompilerSearch.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/IntervalSegments.h>
#include <tilemega/Solver/DmSharedWeights.h>
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
#include <future>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <optional>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

namespace tilemega::commands::compile {

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


struct PreparedResidencyProbe {
  std::filesystem::path directory;
  std::string command;
};

PreparedResidencyProbe prepareResidency(mlir::ModuleOp module,int kappa,
    tilemega::solver::CompilerSearchOptions const& options,
    std::filesystem::path const& directory,std::filesystem::path const&,
    std::string const& runtime_flags) {
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
      << "TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l1_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kServingSharedBytes));\n"
      << "TILEMEGA_CUDA_CHECK(cudaFuncSetAttribute(tilemega_l2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kServingSharedBytes));\n"
      << "int l1=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l1_kernel),kServingThreads,kServingSharedBytes);\n"
      << "int l2=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_l2_kernel),kServingThreads,kServingSharedBytes);\n"
      << "std::printf(\"{\\\"resident\\\":%d,\\\"l1\\\":%d,\\\"l2\\\":%d,\\\"registers_l1\\\":%d,\\\"registers_l2\\\":%d,\\\"dynamic_shared\\\":%zu,\\\"threads\\\":%d}\\n\",std::min(l1,l2),l1,l2,a.numRegs,b.numRegs,kServingSharedBytes,kServingThreads);\n}\n";
  wrapper.close();
  std::string root=TILEMEGA_SOURCE_DIR;
  std::string nvcc=tilemega::commands::NvccPath();
  std::string command=quote(nvcc)+" -std=c++17 -O2 -lineinfo -Xptxas=-v -DTILEMEGA_MIDPOINT_REFINE=0 -arch="+
      quote(options.placement.target.NvccArch())+runtime_flags;
  for (char const* sub:{"include","third_party/cutlass/include","third_party/cutlass/tools/util/include","third_party/cutlass/test"})
    command+=" -I"+quote(root+"/"+sub);
  // The generated runtime uses a small host support set. Linking the whole
  // analysis archive pulls in ISL and LLVM, which nvcc does not link here.
  command+=" "+quote(probe.string())+
      " -x cu "+quote(root+"/lib/Target/TargetSpec.cpp")+
      " -x cu "+quote(root+"/lib/Support/Json.cpp")+
      " -x cu "+quote(root+"/lib/Codegen/RuntimeTaskGraph.cpp")+
      " -x cu "+quote(root+"/lib/Solver/PlanMaterialize.cpp")+
      " -x cu "+quote(root+"/lib/Dialect/CouplingGraph/PlacementPlan.cpp")+
      " -x cu "+quote(root+"/lib/Solver/BalancedPlacement.cpp")+
      " -x cu "+quote(root+"/lib/Solver/ListScheduler.cpp")+
      " -L"+quote(tilemega::commands::CudaLibraryDirectory())+" -lcudart -o "+quote(binary.string());
  std::ofstream(directory/"build_command.txt") << command << '\n';
  return {directory,std::move(command)};
}

int executeResidency(PreparedResidencyProbe const& prepared) {
  auto const& directory=prepared.directory;
  auto binary=directory/"query";
  if (std::system((prepared.command+" >"+quote((directory/"build.log").string())+" 2>&1").c_str()))
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

int queryResidency(mlir::ModuleOp module,int kappa,
    tilemega::solver::CompilerSearchOptions const& options,
    std::filesystem::path const& directory,std::filesystem::path const& library,
    std::string const& runtime_flags) {
  return executeResidency(prepareResidency(module,kappa,options,directory,library,runtime_flags));
}

std::vector<int> queryResidencies(
    std::vector<std::pair<mlir::ModuleOp,int>> const& probes,
    tilemega::solver::CompilerSearchOptions const& options,
    std::filesystem::path const& directory,std::filesystem::path const& library,
    std::string const& runtime_flags) {
  std::vector<PreparedResidencyProbe> prepared;
  prepared.reserve(probes.size());
  // LowerVariants uses MLIR and ISL state owned by the caller's thread.
  for(std::size_t i=0;i<probes.size();++i)
    prepared.push_back(prepareResidency(probes[i].first,probes[i].second,
        options,directory/std::to_string(i),library,runtime_flags));
  std::vector<std::future<int>> running;
  running.reserve(prepared.size());
  for(auto const& job:prepared)
    running.push_back(std::async(std::launch::async,[job]{return executeResidency(job);}));
  std::vector<int> result;
  result.reserve(running.size());
  for(auto& job:running)result.push_back(job.get());
  return result;
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

int RunCompile(int argc, char** argv) {
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
                 " --serving-pruning 0|1 --incremental-prepare 0|1 --serving-warm-start PREVIOUS.plan.json\n"
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
    std::string serving_phase, emit_mode,measure_command,serving_warm_start,paged_seed_from,artifact_cache;
    std::string frontend_mode="decoder";
    std::string shared_weight_layout;
    std::string sync_policy="calibrated",runtime_target,runtime_flags,pg_mode="off";
    std::string arch_paths="auto",pdl="auto",handoff_mode="off",weight_layout="tiled";
    int page_bytes=16384,lookahead_bytes=-1,prefetch_depth=1,prefetch_stride=0;
    int kphase_mask=31,v3_poll_ns=0,watchdog=0,l2_slim=0,page_loop_split=0,evict_first=0,evict_last=1;
    int deferred_norm=1,paged_la=1,paged_la_splitk=1,candidate_guard_wait_s=300,candidate_loop=0;
    int nonpaged_la=0,moe_dynamic=0,moe_opaque=0,moe_gemv=0;
    std::string candidate_mode="L1",nonpaged_weight_layout="row";
    bool page_bytes_pinned=false,nonpaged_la_pinned=false;
    bool event_solo=false,event_red=false,barrier_v2=false;
    if(auto* cache=std::getenv("TILEMEGA_ARTIFACT_CACHE"))artifact_cache=cache;
    int serving_capacity=1088,serving_batch=1,serving_past_lo=64,
        serving_past_hi=1086,serving_kv_block=256,
        serving_query_rows=64,serving_argmax_tile_n=128;
    bool resource_probes=true;bool dump_evaluated=false;
    std::string solver_mode="skeleton",legacy_seed,variant_cache,flow_fixture;
    int skeleton_k=8,search_passes=3,search_jobs=1,search_top_m=8,
        search_budget_ms=0;
    bool all_workers=false,flow_search_only=false,incremental_prepare=true,
         serving_pruning=true;
    tilemega::solver::SolverTiming solver_timing;
    int interval_begin=0,segments=1,segment_candidates=3;
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> variant_modules;
    tilemega::solver::CompilerSearchOptions solve_options;
    std::string moe_binding="auto";unsigned moe_bm=16;
    std::string memory_reuse="none",search_selection="measure";
    std::string dnn_deferred_ln="auto",dnn_dwpw_fuse="auto",global_la="auto";
    bool sequence_pinned=false;
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
      else if (flag=="--search-budget-ms") search_budget_ms=std::stoi(value);
      else if (flag=="--solve") solve_target=value;
      else if (flag=="--serving") serving_phase=value;
      else if (flag=="--shared-weight-layout") shared_weight_layout=value;
      else if (flag=="--frontend") frontend_mode=value;
      else if (flag=="--moe-dynamic") moe_dynamic=std::stoi(value);
      else if (flag=="--moe-opaque") moe_opaque=std::stoi(value);
      else if (flag=="--moe-gemv") moe_gemv=std::stoi(value);
      else if (flag=="--moe-binding") moe_binding=value;
      else if (flag=="--reuse") memory_reuse=value;
      else if (flag=="--deferred-ln") dnn_deferred_ln=value;
      else if (flag=="--dwpw-fuse") dnn_dwpw_fuse=value;
      else if (flag=="--global-la") global_la=value;
      else if (flag=="--selection") search_selection=value;
      else if (flag=="--moe-bm") moe_bm=value=="auto"?16:std::stoul(value);
      else if (flag=="--emit") emit_mode=value;
      else if (flag=="--measure-cmd") measure_command=value;
      else if (flag=="--serving-warm-start") serving_warm_start=value;
      else if (flag=="--paged-seed-from") paged_seed_from=value;
      else if (flag=="--sync") sync_policy=value;
      else if (flag=="--arch-paths") arch_paths=value;
      else if (flag=="--pdl") pdl=value;
      else if (flag=="--pg") pg_mode=value;
      else if (flag=="--weight-layout") weight_layout=value;
      else if (flag=="--nonpaged-weight-layout") nonpaged_weight_layout=value;
      else if (flag=="--handoff") handoff_mode=value;
      else if (flag=="--page-bytes") {page_bytes=std::stoi(value);page_bytes_pinned=true;}
      else if (flag=="--lookahead-bytes") lookahead_bytes=std::stoi(value);
      else if (flag=="--kphase-mask") kphase_mask=std::stoi(value);
      else if (flag=="--deferred-norm") deferred_norm=std::stoi(value);
      else if (flag=="--paged-la") paged_la=std::stoi(value);
      else if (flag=="--paged-la-splitk") paged_la_splitk=std::stoi(value);
      else if (flag=="--nonpaged-la") {nonpaged_la=std::stoi(value);nonpaged_la_pinned=true;}
      else if (flag=="--candidate-guard-wait-s") candidate_guard_wait_s=std::stoi(value);
      else if (flag=="--candidate-mode") candidate_mode=value;
      else if (flag=="--candidate-loop") candidate_loop=std::stoi(value);
      else if (flag=="--watchdog") watchdog=std::stoi(value);
      else if (flag=="--l2-slim") l2_slim=std::stoi(value);
      else if (flag=="--page-loop-split") page_loop_split=std::stoi(value);
      else if (flag=="--evict-first") evict_first=std::stoi(value);
      else if (flag=="--evict-last") evict_last=std::stoi(value);
      else if (flag=="--v3-poll-ns") v3_poll_ns=std::stoi(value);
      else if (flag=="--l2-prefetch-depth") prefetch_depth=std::stoi(value);
      else if (flag=="--l2-prefetch-stride") prefetch_stride=std::stoi(value);
      else if (flag=="--runtime-target") runtime_target=value;
      else if (flag=="--event-solo") event_solo=std::stoi(value)!=0;
      else if (flag=="--event-red-publish") event_red=std::stoi(value)!=0;
      else if (flag=="--barrier-v2") barrier_v2=std::stoi(value)!=0;
      else if (flag=="--artifact-cache") artifact_cache=value;
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
      else if (flag=="--seq") {solve_options.placement.dims.seq=std::stoi(value);sequence_pinned=true;}
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
    bool const forward=serving_phase=="forward";
    int forward_seq=0;
    if(serving && serving_phase!="decode" && serving_phase!="prefill" && !forward)
      throw std::runtime_error("--serving expects decode, prefill or forward");
    if(frontend_mode!="decoder" && frontend_mode!="dnn")
      throw std::runtime_error("--frontend expects decoder or dnn");
    if(frontend_mode=="dnn" && !forward)
      throw std::runtime_error("DNN frontend requires --emit serving --serving forward");
    if(moe_binding!="auto" && moe_binding!="slot" && moe_binding!="group")
      throw std::runtime_error("--moe-binding expects auto, slot or group");
    if(moe_bm!=16 && moe_bm!=32 && moe_bm!=64 && moe_bm!=128)
      throw std::runtime_error("--moe-bm expects auto, 16, 32, 64 or 128");
    if((moe_gemv!=0 && moe_gemv!=1) ||
       (moe_gemv && (!serving || frontend_mode!="decoder")))
      throw std::invalid_argument("--moe-gemv expects 0 or 1 on a decoder serving plan");
    if(moe_gemv && !solve_target.empty())
      throw std::invalid_argument("GEMV search requires calibrated family pricing; use a fixed build");
    if(forward)serving_capacity=serving_past_lo=serving_past_hi=0;
    if(!emit_mode.empty() && emit_mode!="serving")
      throw std::runtime_error("--emit expects serving");
    if(serving != (emit_mode=="serving"))
      throw std::runtime_error("--serving and --emit serving must be paired");
    if(serving && (serving_batch<1 || serving_batch>64 ||
                   serving_past_lo<0 || serving_past_hi<serving_past_lo))
      throw std::runtime_error("invalid serving batch or past range");
    if(search_top_m<1 || search_top_m>8)
      throw std::runtime_error("--top-m must be in 1..8");
    if(search_jobs<1)
      throw std::runtime_error("--search-jobs must be positive");
    if(search_budget_ms<0)
      throw std::runtime_error("--search-budget-ms must be nonnegative");
    if(search_selection!="measure" && search_selection!="predicted")
      throw std::runtime_error("--selection expects measure or predicted");
    if(search_selection=="predicted" && !measure_command.empty())
      throw std::runtime_error("predicted selection cannot run a measurement command");
    if(memory_reuse!="none" && memory_reuse!="greedy" && memory_reuse!="l2" && memory_reuse!="auto")
      throw std::runtime_error("--reuse expects auto, none, greedy or l2");
    if(memory_reuse!="none" && frontend_mode!="dnn")
      throw std::runtime_error("buffer reuse currently requires the DNN frontend");
    if(global_la!="auto" && global_la!="0")
      throw std::invalid_argument("--global-la expects auto or 0");
    if(dnn_dwpw_fuse!="auto" && dnn_dwpw_fuse!="0")
      throw std::runtime_error("--dwpw-fuse expects auto or 0");
    if(dnn_deferred_ln!="auto" && dnn_deferred_ln!="0")
      throw std::runtime_error("--deferred-ln expects auto or 0");
    if(serving && !solve_target.empty() && !flow_search_only &&
       search_selection=="measure" && measure_command.empty())
      throw std::runtime_error("serving solve requires --measure-cmd for the top-3 decision");
    if (!solve_target.empty() && has_variants)
      throw std::runtime_error("--solve chooses variants; cannot combine with --variants");
    if(arch_paths!="auto" && arch_paths!="sm80")throw std::runtime_error("--arch-paths must be auto or sm80");
    if(pdl!="auto" && pdl!="off")throw std::runtime_error("--pdl must be auto or off");
    if(pg_mode!="off" && pg_mode!="pages" && pg_mode!="l2" && pg_mode!="auto")
      throw std::runtime_error("--pg must be off, l2, pages or auto");
    if(handoff_mode!="off" && handoff_mode!="auto")
      throw std::runtime_error("--handoff must be off or auto");
    bool use_pages=serving && (pg_mode=="pages" || (serving_phase=="decode" && pg_mode=="auto"));
    if(moe_dynamic!=0 && moe_dynamic!=1)
      throw std::invalid_argument("--moe-dynamic must be 0 or 1");
    if(moe_dynamic && (!serving || use_pages || pg_mode!="l2" || frontend_mode!="decoder"))
      throw std::invalid_argument("--moe-dynamic requires a decoder MoE plan with --pg l2");
    if(moe_opaque!=0 && moe_opaque!=1)
      throw std::invalid_argument("--moe-opaque must be 0 or 1");
    if(moe_opaque && (!serving || frontend_mode!="decoder" || (pg_mode!="l2" && !use_pages)))
      throw std::invalid_argument("--moe-opaque requires a decoder MoE plan with --pg l2 or pages");
    if(pg_mode=="pages" && !use_pages)throw std::runtime_error("paged execution requires a serving phase");
    if(handoff_mode=="auto" && use_pages)
      throw std::runtime_error("paged decode reductions use fixed last-arriver; --handoff auto is unsupported");
    if(handoff_mode=="auto" && !use_pages)
      throw std::runtime_error("handoff=auto currently requires paged decode");
    if(sync_policy!="legacy" && sync_policy!="calibrated")
      throw std::runtime_error("--sync must be calibrated or legacy");
    if(nonpaged_weight_layout!="row" && nonpaged_weight_layout!="tiled")
      throw std::invalid_argument("--nonpaged-weight-layout must be row or tiled");
    bool const use_nonpaged_tiled=serving &&
        !use_pages && nonpaged_weight_layout=="tiled";
    if(!shared_weight_layout.empty() && (!serving ||
        (use_pages?weight_layout!="tiled":!use_nonpaged_tiled)))
      throw std::invalid_argument("shared weight layout requires packed serving weights");
    if(!shared_weight_layout.empty() && (input.extension()==".mlir" || moe_gemv))
      throw std::invalid_argument("shared weight layout requires export input and the planned GEMM family");
    if(nonpaged_la!=0 && nonpaged_la!=1)
      throw std::invalid_argument("--nonpaged-la must be 0 or 1");
    if(nonpaged_la && !serving)
      throw std::invalid_argument("--nonpaged-la requires a serving plan");
    if(serving) {
      runtime_flags+=" -DTILEMEGA_NONPAGED_TILED="+std::to_string(use_nonpaged_tiled);
      if(runtime_target.empty())runtime_target=solve_target;
      if(runtime_target.empty())runtime_target=std::string(TILEMEGA_SOURCE_DIR)+
          "/configs/targets/"+tilemega::TargetSpec::Probe().arch_tag+".json";
      auto target=tilemega::TargetSpec::FromJson(runtime_target);
      if(sync_policy=="calibrated") {
        runtime_flags+=" -DTILEMEGA_SYNC_V3=1 -DTILEMEGA_EVENT_RED_PUBLISH=1";
      }
      if(kphase_mask<0 || kphase_mask>31 || v3_poll_ns<0)
        throw std::runtime_error("invalid K-phase mask or V3 poll interval");
      if(watchdog!=0 && watchdog!=1)throw std::runtime_error("--watchdog must be 0 or 1");
      if(l2_slim!=0 && l2_slim!=1)throw std::runtime_error("--l2-slim must be 0 or 1");
      if(page_loop_split!=0 && page_loop_split!=1)throw std::runtime_error("--page-loop-split must be 0 or 1");
      if((evict_first!=0 && evict_first!=1) || (evict_last!=0 && evict_last!=1))
        throw std::runtime_error("eviction policy switches must be 0 or 1");
      runtime_flags+=" -DTILEMEGA_EVICT_FIRST="+std::to_string(evict_first)+
          " -DTILEMEGA_EVICT_LAST="+std::to_string(evict_last);
      runtime_flags+=" -DTILEMEGA_PAGE_LOOP_SPLIT="+std::to_string(page_loop_split);
      runtime_flags+=" -DTILEMEGA_L2_SLIM="+std::to_string(l2_slim);
      runtime_flags+=" -DTILEMEGA_WATCHDOG="+std::to_string(watchdog);
      runtime_flags+=" -DTILEMEGA_KPHASE_CLASS_MASK="+std::to_string(kphase_mask)+
          " -DTILEMEGA_V3_POLL_NS="+std::to_string(v3_poll_ns);
      runtime_flags+=" -DTILEMEGA_PDL="+std::to_string(pdl=="auto")+
          " -DTILEMEGA_ARCH_PATH_SM80="+std::to_string(arch_paths=="sm80");
      if(sync_policy!="calibrated")
        runtime_flags+=" -DTILEMEGA_EVENT_SOLO="+std::to_string(event_solo)+
            " -DTILEMEGA_EVENT_RED_PUBLISH="+std::to_string(event_red)+
            " -DTILEMEGA_BARRIER_V2="+std::to_string(barrier_v2);
    }
    if((candidate_mode!="L1" && candidate_mode!="L2") ||
       (candidate_loop!=0 && candidate_loop!=1))
      throw std::invalid_argument("candidate mode/loop is invalid");
    if((deferred_norm!=0 && deferred_norm!=1) || (paged_la!=0 && paged_la!=1) ||
       (paged_la_splitk!=0 && paged_la_splitk!=1) || candidate_guard_wait_s<0)
      throw std::runtime_error("invalid serving ablation option");
    std::string source,selected_serving_mode,selected_serving_binary;
    std::optional<tilemega::analysis::ScopedExactAnalysisMemo> dm_memo;
    std::optional<tilemega::frontend::ModelPlan> dnn_plan;
    if(frontend_mode=="dnn" && input.extension()!=".mlir") {
      auto bridge=tilemega::frontend::ReadExportBridge(input.string());
      tilemega::frontend::DnnPlanOptions options;options.batch=serving_batch;
      options.deferred_layernorm=dnn_deferred_ln=="auto";
      options.dwpw_fuse=dnn_dwpw_fuse=="auto";
      options.memory_reuse=memory_reuse=="auto"?"l2":memory_reuse;
      if(!runtime_target.empty()) {
        auto target=tilemega::TargetSpec::FromJson(runtime_target);
        options.workspace_budget_bytes=target.res.max_dynamic_smem_per_cta;
        options.memory_l2_budget_bytes=target.res.l2_bytes;
        if(target.calib.l2_knee_bytes>0)
          options.memory_l2_budget_bytes=std::min(options.memory_l2_budget_bytes,
              std::uint64_t(target.calib.l2_knee_bytes));
      }
      dnn_plan=tilemega::frontend::BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
      forward_seq=dnn_plan->serving_seq;
    }else if(forward && input.extension()!=".mlir") {
      if(serving_batch!=1)throw std::runtime_error("MoE forward regions require batch one");
      auto bridge=tilemega::frontend::ReadExportBridge(input.string());
      tilemega::frontend::MoeRegionOptions options;
      options.tokens=sequence_pinned?solve_options.placement.dims.seq:1;options.block_rows=moe_bm;
      options.grouped=moe_binding=="group" || (moe_binding=="auto" && options.tokens>2);
      dnn_plan=tilemega::frontend::BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
      forward_seq=dnn_plan->serving_seq;
    }
    double selected_serving_ms=std::numeric_limits<double>::infinity();
    if(serving && solve_target.empty()) {
      if(has_variants)throw std::runtime_error("serving needs one exported model or solved CG");
      if(input.extension()==".mlir") {
        module=mlir::parseSourceFile<mlir::ModuleOp>(input.string(),&context);
        if(!module || !(*module)->hasAttr("tilemega.serving"))
          throw std::runtime_error("serving CG is missing its serving schema");
        auto info=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
        auto phase=info.getAs<mlir::IntegerAttr>("phase");
        bool cg_forward=phase && phase.getInt()==2;
        if(forward!=cg_forward)
          throw std::runtime_error("serving request disagrees with CG forward phase");
        if(forward) {
          auto seq=info.getAs<mlir::IntegerAttr>("seq");
          auto capacity=info.getAs<mlir::IntegerAttr>("capacity");
          auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
          auto dm=plan?plan.getAs<mlir::BoolAttr>("dm"):mlir::BoolAttr{};
          auto region=plan?plan.getAs<mlir::BoolAttr>("forward"):mlir::BoolAttr{};
          if(!seq || seq.getInt()<=0 || seq.getInt()>std::numeric_limits<int>::max() ||
              !capacity || capacity.getInt()!=0 || !dm || !dm.getValue() ||
              !region || !region.getValue())
            throw std::runtime_error("invalid forward CG runtime schema");
          forward_seq=int(seq.getInt());
          auto token_axis=plan.getAs<mlir::BoolAttr>("forward_token_axis");
          if(token_axis && token_axis.getValue() && serving_batch!=1)
            throw std::runtime_error("forward token regions require batch one");
        }
        auto seq=forward?forward_seq:(serving_phase=="decode"?1:64);
        source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(
            {{*module,static_cast<std::uint32_t>(seq),static_cast<std::uint32_t>(seq)}});
      }else {
      auto bridge=tilemega::frontend::ReadExportBridge(input.string());
      tilemega::frontend::ServingOptions options;
      options.deferred_norm=deferred_norm!=0;
      options.moe_batch=serving_batch;
      options.moe_grouped=moe_binding=="group" ||
          (moe_binding=="auto" && serving_batch*(serving_phase=="decode"?1:64)>2);
      options.moe_block_rows=moe_bm;
      options.phase=serving_phase=="decode"
          ? tilemega::frontend::ServingOptions::Phase::kDecode
          : tilemega::frontend::ServingOptions::Phase::kPrefill;
      options.seq=serving_phase=="decode" ? 1 : 64;
      options.capacity=serving_capacity;
      options.kv_block=serving_kv_block;
      options.query_rows=serving_query_rows;
      options.argmax_tile_n=serving_argmax_tile_n;
      auto plan=dnn_plan?*dnn_plan:tilemega::frontend::BuildModelPlan(
          bridge.nodes,bridge.inputs,bridge.outputs,options);
      if(plan.dm && !tilemega::analysis::active_exact_memo)dm_memo.emplace();
      if(forward)options.seq=forward_seq;
      tilemega::frontend::ImportOptions import;
      if (plan.forward || plan.dm) import.phase_batch = serving_batch;
      import.gemms.assign(plan.gemms.size(),forward?
          tilemega::frontend::GemmGranularity{128,64,16,3,1}:
          tilemega::frontend::GemmGranularity{16,128,128,2,1});
      for(auto const& stage:plan.stages)
        if(stage.kind==tilemega::frontend::PlanTaskKind::kDwPwFused)
          import.gemms.at(stage.gemm).tile_m=16;
      if(!shared_weight_layout.empty())
        for(auto const& [id,layout]:tilemega::solver::SharedDmWeightLayouts(plan,
            tilemega::json::ParseFile(shared_weight_layout))) {
          import.gemms.at(id).tile_n=layout.tile_n;
          import.gemms.at(id).tile_k=layout.tile_k;
        }
      if(moe_gemv) {
        bool expert=false;
        for(auto const& g:plan.gemms)expert|=g.access.b==tilemega::codegen::DmBAccess::kExpertIndirect;
        if(!expert)throw std::invalid_argument("--moe-gemv requires a MoE plan");
        for(auto const& stage:plan.stages)if(tilemega::frontend::IsGemmStage(stage.kind)) {
          auto const& g=plan.gemms.at(stage.gemm);
          auto rows=g.access.b==tilemega::codegen::DmBAccess::kExpertIndirect?g.access.block_rows:
              unsigned(serving_batch)*(g.access.rows_per_batch?g.access.rows_per_batch:
                  (stage.batch_rows?1:options.seq));
          if(rows && rows<=4 && g.access.a!=tilemega::codegen::DmAAccess::kIm2Col)
            import.gemms.at(stage.gemm)={16,32,64,2,1};
        }
      }
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
        d.batch=serving_batch;d.seq=forward?forward_seq:(serving_phase=="decode"?1:64);
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
        return queryResidency(m,kappa,solve_options,resource_root/std::to_string(probe_index++),library,runtime_flags);
      };
      if(resource_probes && serving && solver_mode=="skeleton")
        solve_options.query_residencies=[&](std::vector<std::pair<mlir::ModuleOp,int>> const& probes) {
          return queryResidencies(probes,solve_options,
              resource_root/std::to_string(probe_index++),library,runtime_flags);
        };
      if(serving) {
        auto const& coefficients=sync_policy=="calibrated"
            ? solve_options.placement.target.serving_hop_coefficients
            : solve_options.placement.target.serving_legacy_hop_coefficients;
        if(coefficients.size()==3)
          solve_options.placement.hop={coefficients[0],coefficients[1],coefficients[2]};
        else if(sync_policy=="calibrated" || hop_path.empty())
          throw std::runtime_error("serving search requires the selected sync policy's hop calibration");
      }
      solve_options.timing=&solver_timing;
      tilemega::solver::CompilerSearchResult solved;
      if(solver_mode=="legacy")solved=tilemega::solver::SolveExport(input.string(),context,solve_options,&summary,evidence);
      else {
        tilemega::solver::SkeletonSearchOptions skeleton;skeleton.common=solve_options;
        skeleton.k_base=skeleton_k;skeleton.all_workers=all_workers;
        skeleton.passes=search_passes;skeleton.jobs=search_jobs;
        skeleton.search_budget_ms=search_budget_ms;
        skeleton.artifact_prefix=argv[2];skeleton.fixture=flow_fixture;
        skeleton.search_only=flow_search_only;
        skeleton.incremental_prepare=incremental_prepare;
        skeleton.serving_pruning=serving_pruning;
        skeleton.top_m=search_top_m;
        skeleton.pg_pages=use_pages;
        skeleton.handoff_auto=handoff_mode=="auto";
        skeleton.page_bytes=page_bytes;
        if(use_pages)skeleton.page_choices=page_bytes_pinned
            ?std::vector<int>{page_bytes}:std::vector<int>{8192,16384};
        if(use_pages && lookahead_bytes>=0)
          skeleton.lookahead_choices={lookahead_bytes};
        if(!paged_seed_from.empty()) {
          if(!use_pages)throw std::invalid_argument("paged seed requires pages");
          auto file=llvm::MemoryBuffer::getFile(paged_seed_from);
          if(!file)throw std::invalid_argument("cannot read paged seed manifest");
          auto parsed=llvm::json::parse(file.get()->getBuffer());
          auto* object=parsed?parsed->getAsObject():nullptr;
          auto* gemms=object?object->getArray("gemms"):nullptr;
          if(!gemms)throw std::invalid_argument("paged seed lacks GEMMs");
          for(auto const& item:*gemms) {
            auto* g=item.getAsObject();if(!g)throw std::invalid_argument("invalid seed GEMM");
            auto value=[&](char const* key) {
              auto n=g->getInteger(key);if(!n)throw std::invalid_argument("seed lacks geometry");
              return int(*n);
            };
            skeleton.paged_seed_gemms.push_back({value("tile_m"),value("tile_n"),
                value("tile_k"),value("stages"),value("split_k")});
          }
        }
        if(!serving_warm_start.empty()) {
          if(!serving)throw std::runtime_error("warm start needs a serving plan");
          auto file=llvm::MemoryBuffer::getFile(serving_warm_start);
          if(!file)throw std::runtime_error("cannot read serving warm start");
          auto parsed=llvm::json::parse(file.get()->getBuffer());
          auto* object=parsed?parsed->getAsObject():nullptr;
          auto* gemms=object?object->getArray("gemms"):nullptr;
          if(!gemms || gemms->empty())
            throw std::runtime_error("serving warm start has no GEMM table");
          auto integer=[](llvm::json::Object const& item,char const* name)->int {
            auto value=item.getInteger(name);
            if(!value)throw std::runtime_error(std::string("warm start lacks ")+name);
            return int(*value);
          };
          for(auto const& item:*gemms) {
            auto* g=item.getAsObject();
            if(!g)throw std::runtime_error("warm start GEMM entry is not an object");
            skeleton.serving_warm_gemms.push_back({
                integer(*g,"tile_m"),integer(*g,"tile_n"),
                integer(*g,"tile_k"),integer(*g,"stages"),
                integer(*g,"split_k")});
          }
          skeleton.serving_warm_kappa=integer(*object,"kappa");
          skeleton.serving_warm_residency=integer(*object,"residency");
          skeleton.serving_warm_kv_block=int(object->getInteger("attention_kv_block")
              .value_or(serving_kv_block));
          skeleton.serving_warm_query_rows=int(object->getInteger("attention_query_rows")
              .value_or(serving_query_rows));
        }
        if(!evaluation_cases_path.empty()) {
          // A single explicit case also provides a reproducible way to
          // materialize and validate a previously selected serving plan.
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
          options.deferred_norm=deferred_norm!=0;
          options.moe_batch=serving_batch;
          options.moe_grouped=moe_binding=="group" ||
              (moe_binding=="auto" && serving_batch*dims.seq>2);
          options.moe_block_rows=moe_bm;
          options.phase=serving_phase=="decode"
              ? tilemega::frontend::ServingOptions::Phase::kDecode
              : tilemega::frontend::ServingOptions::Phase::kPrefill;
          options.seq=dims.seq;options.capacity=serving_capacity;
          options.kv_block=serving_kv_block;options.query_rows=serving_query_rows;
          options.argmax_tile_n=serving_argmax_tile_n;
          auto plan=dnn_plan?*dnn_plan:tilemega::frontend::BuildModelPlan(bridge.nodes,bridge.inputs,
              bridge.outputs,options);
          if(plan.dm && !tilemega::analysis::active_exact_memo)dm_memo.emplace();
          if(!shared_weight_layout.empty())
            skeleton.dm_shared_weights=tilemega::solver::SharedDmWeightLayouts(plan,
                tilemega::json::ParseFile(shared_weight_layout));
          if(plan.dm && plan.serving)
            skeleton.dm_structure_rebuild=[](auto const& previous,auto const& bridge,
                auto const& lift,int kv_block,int query_rows,int argmax_tile_n) {
              auto binding=std::find_if(previous.stages.begin(),previous.stages.end(),[](auto const& s) {
                return s.moe.step!=tilemega::codegen::DmMoeStep::kNone;
              });
              if(binding==previous.stages.end())
                throw std::invalid_argument("MoE decoder rebuild has no binding geometry");
              auto const& cfg=binding->moe;
              tilemega::frontend::ServingOptions requested;
              requested.seq=previous.serving_seq;requested.capacity=previous.serving_capacity;
              requested.phase=requested.seq==1?tilemega::frontend::ServingOptions::Phase::kDecode:
                  tilemega::frontend::ServingOptions::Phase::kPrefill;
              requested.deferred_norm=std::any_of(previous.gemms.begin(),previous.gemms.end(),
                  [](auto const& g){return g.norm_ss!=tilemega::codegen::kDmNoIndex;});
              requested.moe_batch=cfg.row_capacity/cfg.top_k/requested.seq;
              requested.moe_grouped=cfg.grouped;requested.moe_block_rows=cfg.block_rows;
              requested.kv_block=kv_block;requested.query_rows=query_rows;
              requested.argmax_tile_n=argmax_tile_n;
              auto rebuilt=tilemega::frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,requested);
              auto semantics=tilemega::frontend::LiftSemantics(rebuilt,lift);
              return std::make_pair(std::move(rebuilt),std::move(semantics));
            };
          serving_imported.emplace(tilemega::frontend::TorchExportImporter{}.
              ImportSemantics(input.string(),plan,context));
          // A paged seed must satisfy the selected page's single-stage
          // constraint before any fixed evaluation case or coordinate pass.
          skeleton.seed=use_pages
              ? tilemega::solver::GemmConfig{16,page_bytes>=16384?128:64,64,2,1}
              : tilemega::solver::GemmConfig{16,128,128,2,1};
          skeleton.kappa=1;
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
          std::string command="CUDACXX="+quote(tilemega::commands::NvccPath())+" python3 "+quote(std::string(TILEMEGA_SOURCE_DIR)+
              "/python/tilemega/build/variants/prewarm.py")+
              " --target "+quote(solve_target)+
              " --cache "+quote(variant_cache)+
              " --output "+quote(prewarm_dir.string())+
              " --max-m "+std::to_string(max_m)+" --jobs 8";
          // Paged decode uses the fixed two-stage local MMA mainloop;
          // additional stage templates cannot occur in its search domain.
          if(use_pages)command+=" --stages 2";
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
          std::string command="CUDACXX="+quote(tilemega::commands::NvccPath())+" python3 "+quote(std::string(TILEMEGA_SOURCE_DIR)+"/python/tilemega/build/variants/probe.py")+
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
      shortlist << "rank\tkey\tplacement\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\tkappa\tresidency\tfloor_ns\tpredicted_ns\tsource\tcg\torigin\n";
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
            << '\t' << stem << ".cu\t" << stem << ".mlir\t" << entry.origin << "\n";
      }
      if(serving && !measure_command.empty()) {
        double fastest=std::numeric_limits<double>::infinity();
        std::size_t fastest_index=0;
        std::ofstream selected(std::string(argv[2])+".top3_measured.tsv");
        selected<<"rank\tmode\tloop\tmean_ms\tso\n";
        std::vector<std::string> candidate_sos;
        std::vector<std::pair<std::string,std::string>> compile_commands;
        for(std::size_t i=0;i<solved.shortlist.size();++i) {
          std::string stem=std::string(argv[2])+".top"+std::to_string(i+1);
          std::string candidate_so=stem+".candidate.so";
          int candidate_page_bytes=page_bytes;
          if(use_pages) {
            auto pages=(*solved.shortlist[i].module)->getAttrOfType<mlir::DictionaryAttr>("tmexec.pages");
            if(!pages)throw std::runtime_error("paged shortlist lacks its selected page layout");
            candidate_page_bytes=int(mlir::cast<mlir::IntegerAttr>(pages.get("page_bytes")).getInt());
          }
          std::string compile=quote(std::filesystem::canonical(argv[0]).string())+" compile"+
              " "+quote(stem+".mlir")+" "+quote(candidate_so)+
              " --serving "+quote(serving_phase)+" --emit serving"+
              " --batch "+std::to_string(serving_batch)+
              " --past-range "+quote(std::to_string(serving_past_lo)+":"+
                                    std::to_string(serving_past_hi))+
              " --capacity "+std::to_string(serving_capacity)+
              " --sync "+quote(sync_policy)+" --runtime-target "+quote(runtime_target)+
              " --arch-paths "+quote(arch_paths)+" --pdl "+quote(pdl)+
              " --pg "+quote(pg_mode)+" --page-bytes "+std::to_string(candidate_page_bytes)+
              " --weight-layout "+quote(weight_layout)+
              " --nonpaged-weight-layout "+quote(nonpaged_weight_layout)+
              " --kphase-mask "+std::to_string(kphase_mask)+
              " --v3-poll-ns "+std::to_string(v3_poll_ns)+
              " --watchdog "+std::to_string(watchdog)+
              " --l2-slim "+std::to_string(l2_slim)+
              " --page-loop-split "+std::to_string(page_loop_split)+
              " --evict-first "+std::to_string(evict_first)+
              " --evict-last "+std::to_string(evict_last)+
              " --paged-la "+std::to_string(paged_la)+
              " --paged-la-splitk "+std::to_string(paged_la_splitk)+
              (nonpaged_la?" --nonpaged-la 1":"")+
              " --l2-prefetch-depth "+std::to_string(prefetch_depth)+" --l2-prefetch-stride "+std::to_string(prefetch_stride)+
              " --event-solo "+std::to_string(event_solo)+" --event-red-publish "+std::to_string(event_red)+
              " --barrier-v2 "+std::to_string(barrier_v2)+
              (artifact_cache.empty()?"":" --artifact-cache "+quote(artifact_cache));
          compile_commands.emplace_back(stem,compile+" >"+quote(stem+".build.stdout")+
              " 2>"+quote(stem+".build.stderr"));
          candidate_sos.push_back(candidate_so);
        }
        // Top-3 compilation is independent and does not touch the GPU timing
        // session. Preserve the default serial build; --search-jobs explicitly
        // allows at most that many concurrent nvcc processes.
        for(std::size_t begin=0;begin<compile_commands.size();begin+=std::max(1,search_jobs)) {
          std::vector<std::future<int>> jobs;
          auto end=std::min(compile_commands.size(),begin+std::size_t(std::max(1,search_jobs)));
          for(std::size_t i=begin;i<end;++i) {
            auto command=compile_commands[i].second;
            jobs.push_back(std::async(std::launch::async,[command]{return std::system(command.c_str());}));
          }
          bool failed=false;
          for(auto& job:jobs)failed|=job.get()!=0;
          if(failed)throw std::runtime_error("top-3 serving candidate compilation failed; see topN.build.stderr");
        }
        // Compile every candidate before timing any of them.  Each round
        // rotates the order, so a warm/cool device does not systematically
        // favor a particular rank.  Keep all 32-step raw CUDA-event samples.
        std::map<std::pair<std::size_t,std::string>,std::vector<double>> samples;
        std::vector<bool> rejected(candidate_sos.size(),false);
        std::ofstream rounds(std::string(argv[2])+".top3_measure_rounds.tsv");
        rounds<<"round\trank\tmode\tloop\tmean_ms\tartifact\n";
        for(int round=0;round<3;++round)for(std::size_t position=0;
            position<candidate_sos.size();++position) {
          std::size_t i=(position+std::size_t(round))%candidate_sos.size();
          if(rejected[i])continue;
          std::string stem=std::string(argv[2])+".top"+std::to_string(i+1);
          std::string artifact=stem+".measurement.r"+std::to_string(round);
          std::string measure=measure_command+" --so "+quote(candidate_sos[i])+
              " --batch "+std::to_string(serving_batch)+
              " --past-mid "+std::to_string(dims.past)+
              " --out "+quote(artifact)+" --mode "+candidate_mode+
              " --loop "+std::to_string(use_pages && candidate_mode=="L2"?candidate_loop:0)+
              " --guard-wait-s "+std::to_string(candidate_guard_wait_s);
          if(round==0)measure+=" --smoke-steps 16";
          // A deadlocked device kernel otherwise holds the GPU indefinitely.
          // Normal candidate timing takes seconds; a timeout is a failed
          // candidate measurement, never a performance observation.
          int status=std::system(("timeout --signal=TERM --kill-after=5s "+std::to_string(candidate_guard_wait_s+600)+"s "+measure+
              " >"+quote(artifact+".stdout")+
              " 2>"+quote(artifact+".stderr")).c_str());
          if(status==75*256)return 75;
          if(status) {
            rejected[i]=true;
            selected<<i+1<<"\trejected\t0\t"<<status<<'\t'<<artifact<<".stderr\n";
            continue;
          }
          auto measured_file=llvm::MemoryBuffer::getFile(artifact+"/measurements.json");
          if(!measured_file)throw std::runtime_error("missing top-3 measurement output");
          auto measured=llvm::json::parse(measured_file.get()->getBuffer());
          auto* object=measured?measured->getAsObject():nullptr;
          auto* modes=object?object->getObject("modes"):nullptr;
          if(!modes)throw std::runtime_error("top-3 measurement has no mode table");
          for(auto const& mode:{candidate_mode})
            if(auto* item=modes->getObject(mode))if(auto mean=item->getNumber("mean_ms")) {
              samples[{i,mode}].push_back(*mean);
              rounds<<round<<'\t'<<i+1<<'\t'<<mode<<'\t'
                    <<(use_pages && candidate_mode=="L2"?candidate_loop:0)<<'\t'<<*mean<<'\t'
                    <<artifact<<"/measurements.json\n";
            }
          rounds.flush();
        }
        for(auto& [key,values]:samples) {
          if(rejected[key.first])continue;
          if(values.size()!=3)throw std::runtime_error("top-3 mode lacks three measurement rounds");
          std::sort(values.begin(),values.end());
          double median=values[1];
          selected<<key.first+1<<'\t'<<key.second<<'\t'
                  <<(use_pages && candidate_mode=="L2"?candidate_loop:0)<<'\t'<<median<<'\t'
                  <<candidate_sos[key.first]<<'\n';
          if(median<fastest) {
            fastest=median;fastest_index=key.first;
            selected_serving_mode=candidate_mode;
          }
        }
        if(!std::isfinite(fastest))throw std::runtime_error("top-3 measurements contain no timing");
        solved.module=mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::cast<mlir::ModuleOp>(solved.shortlist[fastest_index].module->clone()));
        solved.winner=solved.shortlist[fastest_index].evaluation.candidate;
        selected_serving_binary=std::string(argv[2])+".top"+
            std::to_string(fastest_index+1)+".candidate.so";
        selected_serving_ms=fastest;
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
      if(use_pages) {
        auto pages=(*module)->getAttrOfType<mlir::DictionaryAttr>("tmexec.pages");
        if(!pages)throw std::runtime_error("paged winner lacks its selected page layout");
        page_bytes=int(mlir::cast<mlir::IntegerAttr>(pages.get("page_bytes")).getInt());
      }
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
    bool use_l2=serving && (pg_mode=="l2" || (pg_mode=="auto" && !use_pages));
    if(moe_opaque)(*module)->setAttr("tmexec.moe_opaque",mlir::BoolAttr::get(&context,true));
    if(serving && !use_pages && nonpaged_la) {
      mlir::OpBuilder handoffs(module->getContext());
      (*module)->setAttr("tmexec.nonpaged_la",handoffs.getBoolAttr(true));
      auto reductions=tilemega::dialect::SelectServingHandoffs(*module,2|4);
      std::cerr<<"NONPAGED_LAST_ARRIVER selected="<<reductions.last_arriver<<'\n';
      handoff_mode="last_arriver";
      source=tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
    }
    if(use_l2) {
      auto target=tilemega::TargetSpec::FromJson(runtime_target);
      tilemega::codegen::ConfigureServingPrefetch(*module,target,prefetch_depth,prefetch_stride);
      auto seq=mlir::cast<mlir::IntegerAttr>((*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving").get("seq")).getInt();
      auto model=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
      auto dm=model.getAs<mlir::BoolAttr>("dm");
      // Existing nonpaged prefill dispatches through index 1; its token
      // geometry is carried separately by TILEMEGA_SERVING_SEQ.
      if(!(dm && dm.getValue()) && !nonpaged_la && !use_nonpaged_tiled)seq=1;
      source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,unsigned(seq),unsigned(seq)}});
    }
    if(use_pages) {
      auto target=tilemega::TargetSpec::FromJson(runtime_target);
      if(lookahead_bytes>=0)
        (*module)->setAttr("tmexec.lookahead_bytes",
            mlir::IntegerAttr::get(mlir::IntegerType::get(&context,64),lookahead_bytes));
      tilemega::codegen::ConfigureServingPages(*module,target,page_bytes);
      if(weight_layout=="tiled")tilemega::codegen::ResolveServingWeightPacking(*module);
      else if(weight_layout!="row")throw std::invalid_argument("weight layout must be row or tiled");
      // Decode paging always lowers reduction handoffs. The selected IR owns
      // the decision; there is no R11 recompute coordinate in this regime.
      if(paged_la) {
      auto r12_reductions=tilemega::dialect::SelectServingHandoffs(*module,2|(paged_la_splitk?4:0));
      std::cerr<<"R12_LAST_ARRIVER selected="<<r12_reductions.last_arriver<<'\n';
      handoff_mode="last_arriver";
      }else handoff_mode="off";
      auto seq=mlir::cast<mlir::IntegerAttr>((*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving").get("seq")).getInt();
      source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,unsigned(seq),unsigned(seq)}});
    }
    if(use_nonpaged_tiled) {
      mlir::OpBuilder packing(module->getContext());
      (*module)->setAttr("tmexec.nonpaged_weight_layout_tiled",packing.getBoolAttr(true));
      tilemega::codegen::ResolveServingWeightPacking(*module);
      auto seq=mlir::cast<mlir::IntegerAttr>((*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving").get("seq")).getInt();
      source=use_l2 ? tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,unsigned(seq),unsigned(seq)}})
                    : tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
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
      if (serving_phase == "prefill" || forward)
        serving_past_lo = serving_past_hi = 0;
      source = "#define TILEMEGA_PDL " +std::to_string(pdl=="auto")+"\n"+
          "#define TILEMEGA_ARCH_PATH_SM80 "+std::to_string(arch_paths=="sm80")+"\n"+
          "#define TILEMEGA_SERVING_BATCH_LO " +
          std::to_string(serving_batch) + "\n" +
          "#define TILEMEGA_SERVING_BATCH_HI " +
          std::to_string(serving_batch) + "\n" +
          "#define TILEMEGA_SERVING_PAST_LO " +
          std::to_string(serving_past_lo) + "\n" +
          "#define TILEMEGA_SERVING_PAST_HI " +
          std::to_string(serving_past_hi) + "\n" + source;
    }
    bool dm_pool_la=false,dm_moe_la=false;
    if(serving) {
      auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
      auto dm=plan?plan.getAs<mlir::BoolAttr>("dm"):mlir::BoolAttr{};
      if(dm && dm.getValue()) {
        bool enabled=!moe_opaque && (use_pages?paged_la:(!nonpaged_la_pinned || nonpaged_la));
        dm_pool_la=enabled && global_la=="auto";
        dm_moe_la=enabled;
        if(dm_pool_la || dm_moe_la)source="#define TILEMEGA_DM_REDUCTIONS 1\n#define TILEMEGA_DM_POOL_LA "+
            std::to_string(dm_pool_la)+"\n#define TILEMEGA_DM_MOE_LA "+std::to_string(dm_moe_la)+"\n"+source;
      }
    }
    if(moe_dynamic || moe_opaque || moe_gemv) {
      auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
      auto dm=plan?plan.getAs<mlir::BoolAttr>("dm"):mlir::BoolAttr{};
      bool experts=false;
      if(dm && dm.getValue())for(auto entry:plan.getAs<mlir::ArrayAttr>("stages"))
        experts|=bool(mlir::cast<mlir::DictionaryAttr>(entry).get("dm_moe"));
      if(!experts)throw std::invalid_argument("MoE execution controls require virtual MoE stages");
      if(moe_dynamic)source="#define TILEMEGA_MOE_DYNAMIC 1\n"+source;
      if(moe_gemv)source="#define TILEMEGA_MOE_GEMV 1\n"+source;
      if(moe_opaque) {
        source="#define TILEMEGA_MOE_OPAQUE 1\n"+source;
      }
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
        auto macros=[](std::string const& command) {
          std::set<std::string> result;std::istringstream words(command);std::string word;
          while(words>>word)if(word.rfind("-D",0)==0)result.insert(word);
          return result;
        };
        std::ifstream built_command(selected_serving_binary+".build_command.txt");
        std::string built_line;std::getline(built_command,built_line);
        bool const flags_match=built_command.good() &&
            macros(built_line)==macros("-DTILEMEGA_MIDPOINT_REFINE=0 "+runtime_flags);
        if(compiled_source==source && flags_match) {
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
      std::string nvcc = tilemega::commands::NvccPath();
      std::string arch = tilemega::TargetSpec::Probe().NvccArch();
      std::string command = quote(nvcc) +
          " -std=c++17 -O3 -DTILEMEGA_MIDPOINT_REFINE=0 -arch="+
          quote(arch)+runtime_flags+" --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -cudart shared -Xptxas=-v -x cu" +
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
          " -L"+quote(tilemega::commands::CudaLibraryDirectory())+" -lcudart -o " + quote(requested.string());
      std::ofstream(requested.string()+".build_command.txt") << command << '\n';
      std::string invocation=command+" >"+quote(requested.string()+".ptxas.log")+" 2>&1";
      if(!artifact_cache.empty()) {
        std::string python_path=root+"/python";
        if(auto* inherited=std::getenv("PYTHONPATH"))python_path+=":"+std::string(inherited);
        invocation="PYTHONPATH="+quote(python_path)+
            " python3 -m tilemega.build.artifacts --command "+quote(command)+
            " --cache "+quote(artifact_cache)+" --log "+quote(requested.string()+".ptxas.log")+
            " >"+quote(requested.string()+".cache.stdout")+" 2>"+quote(requested.string()+".cache.stderr");
      }
      int status = std::system(invocation.c_str());
      if (status != 0) throw std::runtime_error("nvcc failed while building shared object");
      }
    }
    if(serving && module) {
      auto runtime=tilemega::codegen::ReadRuntimePlan(*module);
      // classes.tsv describes the predicted winner. Preserve its class/GEMM
      // mapping but take every geometry from the actual measured winner.
      std::ifstream class_input(std::string(argv[2])+".classes.tsv");
      if(class_input) {
        std::ofstream chosen(std::string(argv[2])+".selected_classes.tsv");
        std::string line;std::getline(class_input,line);chosen<<line<<'\n';
        while(std::getline(class_input,line)) {
          std::istringstream row(line);unsigned cls,index;std::string op;
          if(!(row>>cls>>index>>op) || index>=runtime.gemms.size())continue;
          auto const& g=runtime.gemms[index];
          chosen<<cls<<'\t'<<index<<'\t'<<op<<'\t'<<g.tile_m<<'\t'<<g.tile_n
                <<'\t'<<g.tile_k<<'\t'<<g.stages<<'\t'<<g.split_k<<'\n';
        }
      }

      auto integer=[&](char const* key,int fallback) {
        if(auto value=(*module)->getAttrOfType<mlir::IntegerAttr>(key))
          return int(value.getInt());
        return fallback;
      };
      bool manifest_deferred_norm=serving_phase=="decode";
      if(auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan"))
        if(auto stages=plan.getAs<mlir::ArrayAttr>("stages"))for(auto entry:stages)
          if(auto stage=llvm::dyn_cast<mlir::DictionaryAttr>(entry))
            if(auto kind=stage.getAs<mlir::StringAttr>("kind");kind && kind.getValue()=="kRMSNorm")
              manifest_deferred_norm=false;
      int attention_kv_block=0,attention_query_rows=0;
      if(auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan"))
        if(auto stages=llvm::dyn_cast_or_null<mlir::ArrayAttr>(plan.get("stages")))
          for(auto entry:stages)
            if(auto stage=llvm::dyn_cast<mlir::DictionaryAttr>(entry))
              if(auto kind=stage.getAs<mlir::StringAttr>("kind");
                 kind && kind.getValue()=="kFusedAttention") {
                attention_kv_block=int(stage.getAs<mlir::IntegerAttr>(
                    "attention_kv_block").getInt());
                attention_query_rows=int(stage.getAs<mlir::IntegerAttr>(
                    "attention_query_rows").getInt());
                break;
              }
      std::string model_name=input.parent_path().filename().string();
      if(auto file=llvm::MemoryBuffer::getFile(
             (input.parent_path()/"manifest.json").string())) {
        auto parsed=llvm::json::parse(file.get()->getBuffer());
        if(auto* manifest_object=parsed?parsed->getAsObject():nullptr)
          if(auto* config=manifest_object->getObject("config"))
            if(auto type=config->getString("model_type"))
              model_name=type->str();
      }
      if(model_name.empty())model_name=input.stem().string();
      std::ofstream manifest(requested.string()+".plan.json");
      std::string pages_json="null";
      if(auto pages=(*module)->getAttrOfType<mlir::DictionaryAttr>("tmexec.pages")) {
        std::ostringstream json;json<<'{';bool comma=false;
        for(auto field:pages) {if(comma)json<<',';comma=true;
          json<<std::quoted(field.getName().str())<<':'<<mlir::cast<mlir::IntegerAttr>(field.getValue()).getInt();}
        json<<'}';pages_json=json.str();
      }
      std::string prefetch_json="null";
      if(auto prefetch=(*module)->getAttrOfType<mlir::DictionaryAttr>("tmexec.prefetch")) {
        std::ostringstream json;json<<'{';bool comma=false;
        for(auto key:{"depth","stride","worker_bytes"}) {
          if(comma)json<<',';comma=true;
          json<<std::quoted(key)<<':'<<prefetch.getAs<mlir::IntegerAttr>(key).getInt();
        }
        json<<",\"history_proofs\":"<<prefetch.getAs<mlir::ArrayAttr>("history_proofs").size()<<'}';
        prefetch_json=json.str();
      }
      manifest<<"{\n  \"model\": "<<std::quoted(model_name)
              <<",\n  \"phase\": "<<std::quoted(serving_phase)
              <<",\n  \"batch_lo\": "<<serving_batch
              <<",\n  \"batch_hi\": "<<serving_batch
              <<",\n  \"past_lo\": "<<serving_past_lo
              <<",\n  \"past_hi\": "<<serving_past_hi
              <<",\n  \"seq\": "<<(forward?forward_seq:(serving_phase=="decode"?1:64))
              <<",\n  \"capacity\": "<<serving_capacity
              <<",\n  \"sync\": "<<std::quoted(sync_policy)
              <<",\n  \"pg\": "<<std::quoted(use_pages?"pages":pg_mode)
              <<",\n  \"watchdog\": "<<watchdog
              <<",\n  \"l2_slim\": "<<l2_slim
              <<",\n  \"nonpaged_weight_layout\": "<<std::quoted(nonpaged_weight_layout)
              <<",\n  \"page_loop_split\": "<<page_loop_split
              <<",\n  \"evict_first\": "<<evict_first
              <<",\n  \"evict_last\": "<<evict_last
              <<",\n  \"deferred_norm\": "<<(manifest_deferred_norm?"true":"false")
              <<",\n  \"paged_la\": "<<(use_pages && paged_la?"true":"false")
              <<",\n  \"paged_la_splitk\": "<<(use_pages && paged_la && paged_la_splitk?"true":"false")
              <<",\n  \"nonpaged_la\": "<<(!use_pages && nonpaged_la?"true":"false")
              <<",\n  \"handoff\": "<<std::quoted(handoff_mode)
              <<",\n  \"pages\": "<<pages_json
              <<",\n  \"prefetch\": "<<prefetch_json
              <<",\n  \"arch_paths\": "<<std::quoted(arch_paths)
              <<",\n  \"pdl\": "<<std::quoted(pdl)
              <<",\n  \"runtime_target\": "<<std::quoted(runtime_target)
              <<",\n  \"event_solo\": "<<(event_solo?"true":"false")
              <<",\n  \"event_red_publish\": "<<(event_red?"true":"false")
              <<",\n  \"barrier_v2\": "<<(barrier_v2?"true":"false")
              <<",\n  \"mode\": "<<std::quoted(selected_serving_mode.empty()?"unmeasured":selected_serving_mode)
              <<",\n  \"grid\": "<<integer("tmexec.solved_grid",0)
              <<",\n  \"residency\": "<<integer("tmexec.solved_residency",0)
              <<",\n  \"kappa\": "<<integer("tmexec.solved_kappa",1)
              <<",\n  \"attention_kv_block\": "<<attention_kv_block
              <<",\n  \"attention_query_rows\": "<<attention_query_rows;
      auto manifest_plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
      auto manifest_dm=manifest_plan?manifest_plan.getAs<mlir::BoolAttr>("dm"):mlir::BoolAttr{};
      if(forward || (manifest_dm && manifest_dm.getValue())) {
        auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
        auto token_axis=plan?plan.getAs<mlir::BoolAttr>("forward_token_axis"):mlir::BoolAttr{};
        manifest<<",\n  \"frontend\": "<<std::quoted(frontend_mode);
        manifest<<",\n  \"moe_dynamic\": "<<(moe_dynamic?"true":"false");
        manifest<<",\n  \"moe_opaque\": "<<(moe_opaque?"true":"false");
        manifest<<",\n  \"moe_gemv\": "<<(moe_gemv?"true":"false");
        manifest<<",\n  \"global_la\": "<<std::quoted(global_la);
        manifest<<",\n  \"dm_pool_la\": "<<(dm_pool_la?"true":"false");
        manifest<<",\n  \"dm_moe_la\": "<<(dm_moe_la?"true":"false");
        if(!shared_weight_layout.empty())
          manifest<<",\n  \"shared_weight_layout\": "<<std::quoted(shared_weight_layout)
                  <<",\n  \"shared_weight_layout_sha256\": "<<std::quoted(modelFingerprint(shared_weight_layout));
        if(frontend_mode=="dnn")manifest<<",\n  \"deferred_ln\": "<<std::quoted(dnn_deferred_ln);
        if(frontend_mode=="dnn")manifest<<",\n  \"dwpw_fuse\": "<<std::quoted(dnn_dwpw_fuse);
        if(search_selection=="predicted")manifest<<",\n  \"selection\": \"predicted\"";
        if(auto reuse=plan.getAs<mlir::StringAttr>("dm_memory_reuse"))
          manifest<<",\n  \"reuse\": "<<std::quoted(reuse.getValue().str())
                  <<",\n  \"memory_arena_bytes\": "<<plan.getAs<mlir::IntegerAttr>("dm_memory_arena_bytes").getInt()
                  <<",\n  \"memory_retained_internal_bytes\": "<<integer("tilemega.memory_retained_internal_bytes",0)
                  <<",\n  \"memory_total_internal_bytes\": "<<integer("tilemega.memory_total_internal_bytes",0)
                  <<",\n  \"memory_hazard_count\": "<<integer("tilemega.memory_hazard_count",0);
        llvm::json::Array buffer_records;
        for(auto entry:plan.getAs<mlir::ArrayAttr>("buffers")) {
          auto buffer=mlir::cast<mlir::DictionaryAttr>(entry);
          llvm::json::Object record;
          for(auto name:{"name","role","dtype","source"})
            if(auto field=buffer.getAs<mlir::StringAttr>(name))record[name]=field.getValue().str();
          for(auto name:{"constant","per_seq","per_past","per_total","per_batch","dm_arena_offset"})
            if(auto field=buffer.getAs<mlir::IntegerAttr>(name))record[name]=field.getInt();
          if(auto field=buffer.getAs<mlir::StringAttr>("pack_json");field && !field.getValue().empty()) {
            auto recipe=llvm::json::parse(field.getValue());
            if(!recipe)throw std::invalid_argument("manifest weight recipe is invalid JSON");
            record["recipe"]=std::move(*recipe);
          }
          buffer_records.push_back(std::move(record));
        }
        manifest<<",\n  \"buffers\": "<<llvm::formatv("{0}",llvm::json::Value(std::move(buffer_records))).str();
        if((token_axis && token_axis.getValue()) || !forward) {
          for(auto entry:plan.getAs<mlir::ArrayAttr>("stages")) {
            auto stage=mlir::cast<mlir::DictionaryAttr>(entry);
            if(!stage.get("dm_moe"))continue;
            auto moe=tilemega::frontend::DecodeDmMoeStage(stage.get("dm_moe"));
            manifest<<",\n  \"moe_binding\": "<<std::quoted(moe.grouped?"group":"slot")
                    <<",\n  \"moe_bm\": "<<moe.block_rows
                    <<",\n  \"moe_experts\": "<<moe.experts
                    <<",\n  \"moe_top_k\": "<<moe.top_k
                    <<",\n  \"moe_binding_capacity\": "<<moe.binding_capacity;
            break;
          }
        }
      }
      manifest<<",\n  \"gemms\": [\n";
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

  return 0;
}

}  // namespace tilemega::commands::compile
