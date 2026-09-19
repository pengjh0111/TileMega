// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Solver/CompilerSearch.h>
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
  std::string command=quote(nvcc)+" -std=c++17 -O2 -lineinfo -Xptxas=-v -arch="+
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
  tilemega::analysis::IslContext isl_context;
  if (argc < 3 || argc % 2 == 0) {
    std::cerr << "usage: tilemega-compile {EXPORTED_PROGRAM.pt2|STABLE_EXPORT.json|CG.mlir} "
                 "{OUTPUT.cu|OUTPUT.so} [--variants PLAN.json] [--solve TARGET.json --seq N --past N\n"
                 " --search-capacity N --per-stage-kappa 0|1 --dump-cg FILE.mlir\n"
                 " --hop-curve FILE.tsv --seq-begin N\n"
                 " --prefetch-page-bytes N]\n";
    return 2;
  }
  try {
    mlir::MLIRContext context;
    context.getOrLoadDialect<tilemega::dialect::CGDialect>();
    tilemega::frontend::ImportSummary summary;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::filesystem::path input(argv[1]);
    std::string variants_path,solve_target,dump_cg,hop_path,domain_path,rejections_path;
    bool resource_probes=true;
    int interval_begin=0;
    tilemega::solver::CompilerSearchOptions solve_options;
    solve_options.placement.dims={4,3,7};
    for (int i=3;i<argc;i+=2) {
      std::string flag=argv[i],value=argv[i+1];
      if (flag=="--variants") variants_path=value;
      else if (flag=="--solve") solve_target=value;
      else if (flag=="--seq-begin") interval_begin=std::stoi(value);
      else if (flag=="--seq") solve_options.placement.dims.seq=std::stoi(value);
      else if (flag=="--past") solve_options.placement.dims.past=std::stoi(value);
      else if (flag=="--search-capacity") solve_options.capacity=std::stoul(value);
      else if (flag=="--per-stage-kappa") solve_options.per_stage_kappa=std::stoi(value)!=0;
      // Zero prices no prefetch at all, which is the pipelining dimension of
      // sigma switched off: every credit and every queue-edge discount is then
      // exactly zero, so a solve can be repeated without it.
      else if (flag=="--prefetch-page-bytes") solve_options.placement.prefetch_page_bytes=std::stoi(value);
      else if (flag=="--dump-cg") dump_cg=value;
      else if (flag=="--hop-curve") hop_path=value;
      else if (flag=="--resource-probes") resource_probes=std::stoi(value)!=0;
      else if (flag=="--search-domain") domain_path=value;
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
    if (!solve_target.empty() && has_variants)
      throw std::runtime_error("--solve chooses variants; cannot combine with --variants");
    std::string source;
    if (!solve_target.empty()) {
      if (input.extension()==".mlir")
        throw std::runtime_error("automatic geometry search requires export JSON; use tilemega-opt for placement-only CG solving");
      solve_options.placement.target=tilemega::TargetSpec::FromJson(solve_target);
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
      if (resource_probes) solve_options.query_residency=[&](mlir::ModuleOp m,int kappa) {
        return queryResidency(m,kappa,solve_options,resource_root/std::to_string(probe_index++),library);
      };
      auto solved=tilemega::solver::SolveExport(input.string(),context,solve_options,&summary,evidence);
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
        interval_options.residency=integer("tilemega.solved_residency");
        interval_options.verified_resident_limit=interval_options.residency;
        interval_options.kappa=integer("tilemega.solved_kappa");
        interval_options.requested_grid=integer("tilemega.solved_grid");
        std::ofstream interval_evidence(std::string(argv[2])+".interval.tsv");
        interval_evidence<<"seq\tplacement\tfloor_ns\tpredicted_ns\terror\n";
        tilemega::dialect::SolveAndWritePlacementInterval(*module,interval_options,interval_begin,dims.seq,&interval_evidence);
      }
      std::vector<tilemega::codegen::RuntimeVariantModule> inputs{{*module,
          1u,static_cast<std::uint32_t>(dims.seq)}};
      source=tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(inputs);
      std::cerr << "SOLVE_SUMMARY evaluated=" << solved.stats.evaluated
          << " deferred=" << solved.stats.capacity_deferred
          << " residency_scope=" << (resource_probes ? "compiled" : "1_degraded") << " hop_calibrated=" << !hop_path.empty() << "\n";
      if (solve_options.per_stage_kappa) {
        std::cerr << "SOLVE_STAGE_KAPPA moves=" << solved.stage_kappa_moves
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
      for (auto task : module->getOps<tilemega::dialect::TaskSpaceOp>()) {
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
      std::string root = TILEMEGA_SOURCE_DIR;
      std::string nvcc = std::getenv("CUDACXX") ? std::getenv("CUDACXX") :
                                                 "/usr/local/cuda/bin/nvcc";
      std::string command = quote(nvcc) +
          " -std=c++17 -O2 -arch=native -shared -Xcompiler=-fPIC -x cu" +
          " -I" + quote(root + "/include") +
          " -I" + quote(root + "/third_party/cutlass/include") +
          " -I" + quote(root + "/third_party/cutlass/tools/util/include") +
          " -I" + quote(root + "/third_party/cutlass/test") + " " +
          quote(cuda.string()) + " -x cu " +
          quote(root + "/lib/Target/TargetSpec.cpp") +
          " -L/usr/local/cuda/lib64 -lcudart -o " + quote(requested.string());
      int status = std::system(command.c_str());
      if (status != 0) throw std::runtime_error("nvcc failed while building shared object");
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
