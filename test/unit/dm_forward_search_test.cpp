// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace tilemega::tests::dm_forward_search_test {
int TestDmForwardSearch(int,char**) {
  using namespace tilemega;
  using namespace frontend;
  analysis::IslContext isl;
  for(bool token_axis:{false,true}) {
    mlir::MLIRContext context;
    llvm::json::Array shape = token_axis ? llvm::json::Array{"s0", "128"}
                                        : llvm::json::Array{"s0", "17", "128"};
    auto node = [](int index, char const* name, char const* op, char const* target,
                   llvm::json::Array inputs, llvm::json::Array dimensions) {
      return llvm::json::Object{{"index", index}, {"name", name}, {"op", op},
          {"target", target}, {"inputs", std::move(inputs)},
          {"shape", std::move(dimensions)}, {"dtype", "torch.bfloat16"}};
    };
    llvm::json::Value json = llvm::json::Object{
        {"schema", "tilemega.exported_program.v1"}, {"guards", llvm::json::Array{}},
        {"range_constraints", llvm::json::Object{{"s0", "VR[1, 4096]"}}},
        {"nodes", llvm::json::Array{
            node(0, "x", "placeholder", "x", {}, std::move(shape)),
            node(1, "weight", "placeholder", "weight", {}, llvm::json::Array{"64", "128"}),
            node(2, "linear", "call_function", "aten.linear.default", llvm::json::Array{"x", "weight"},
                 token_axis ? llvm::json::Array{"s0", "64"}
                            : llvm::json::Array{"s0", "17", "64"})}},
        {"signature", llvm::json::Object{
            {"inputs", llvm::json::Array{
                llvm::json::Object{{"name", "x"}, {"kind", "USER_INPUT"}, {"target", ""}},
                llvm::json::Object{{"name", "weight"}, {"kind", "PARAMETER"}, {"target", "weight"}}}},
            {"outputs", llvm::json::Array{
                llvm::json::Object{{"name", "linear"}, {"kind", "USER_OUTPUT"}}}}}}};
    int fd;
    llvm::SmallString<128> filename;
    assert(!llvm::sys::fs::createTemporaryFile("dm-forward", "json", fd, filename));
    { llvm::raw_fd_ostream out(fd, true); out << llvm::formatv("{0:2}", json); }
    ModelPlan plan;
    plan.dm = plan.forward = true; plan.forward_token_axis = token_axis;
    plan.dtype = "bf16"; plan.serving_seq = token_axis ? 35 : 1;
    plan.buffers.resize(3);
    for (unsigned i = 0; i < 3; ++i) {
      auto& buffer = plan.buffers[i];
      buffer.name = i == 0 ? "x" : i == 1 ? "weight" : "out";
      buffer.external_name = buffer.name; buffer.role = "external";
      if (i == 1) buffer.constant = 64 * 128;
      else if (token_axis) buffer.per_seq = i == 0 ? 128 : 64;
      else buffer.per_batch = 17 * (i == 0 ? 128 : 64);
    }
    PlanGemm gemm; gemm.n = 64; gemm.k = 128; gemm.b = 1; gemm.d = 2;
    gemm.access.rows_per_batch = token_axis ? 0 : 17;
    plan.gemms.push_back(gemm);
    PlanStage stage; stage.representative = "linear"; stage.representative_index = 2;
    plan.stages.push_back(stage); plan.outputs.push_back({2, ""});

    auto imported=TorchExportImporter{}.ImportSemantics(filename.str().str(),plan,context);
    assert(!imported.plan.serving && imported.plan.forward);
    for(bool paged:{false,true}) {
      solver::SkeletonSearchOptions search;
      search.common.placement.target=TargetSpec::FromJson(
          std::string(TILEMEGA_SOURCE_DIR)+
          "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
      search.common.placement.target.res.num_sms=4;
      search.common.placement.dims={plan.serving_seq,0,plan.serving_seq};
      search.common.placement.dims.batch=token_axis?1:8;
      search.seed={16,16,16,2,1};
      search.common.geometry_domain={search.seed,{16,32,16,2,1},{32,16,16,2,1}};
      search.dm_shared_weights={{0,{16,16}}};
      // Host search mechanics use synthetic resource callbacks. Native
      // compiled resources and fitted latency accuracy are separate gates.
      search.variant_probe=[](auto const&,auto const* g,auto) {
        return solver::VariantResources{32,g?solver::DmServingBF16SmemBytes(
            g->tile_m,g->tile_n,g->tile_k,g->stages):512,128,false};
      };
      bool rebuilt=false;
      search.dm_structure_rebuild=[&](auto const&,auto const&,auto const&,int,int,int)
          ->std::pair<ModelPlan,LiftedModel> {
        rebuilt=true;throw std::runtime_error("forward region acquired a decoder coordinate");
      };
      search.pg_pages=paged;search.page_choices={8192};search.lookahead_choices={0};
      search.passes=1;search.search_only=true;
      auto folder=std::filesystem::temp_directory_path()/
          ("tilemega-dm-forward-search-"+std::to_string(token_axis)+"-"+std::to_string(paged));
      std::filesystem::create_directories(folder);
      search.artifact_prefix=(folder/"search").string();
      std::ostringstream evidence;
      auto result=solver::SolveSkeletonImported(imported,context,search,nullptr,evidence);
      assert(!rebuilt && !result.evaluated.empty() && result.top.empty());
      for(auto const& candidate:result.evaluated) {
        if(!candidate.error.empty())std::cerr<<candidate.key<<": "<<candidate.error<<'\n';
        assert(candidate.error.empty() && std::isfinite(candidate.score));
        assert(candidate.attention_kv_block==0 && candidate.attention_query_rows==0);
        assert(candidate.config[0].tile_n==16 && candidate.config[0].tile_k==16);
        assert(candidate.shared_bytes>0 && candidate.shared_bytes<=
            search.common.placement.target.res.max_dynamic_smem_per_cta);
        if(paged)assert(candidate.actual_limit<=1 && candidate.estimated_limit==1);
      }
      assert(evidence.str().find("ATTENTION_COORDINATE")==std::string::npos);
      std::cout<<"DM forward search token_axis="<<token_axis<<" paged="<<paged
          <<" evaluations="<<result.evaluated.size()<<" PASS\n";
      std::filesystem::remove_all(folder);
    }
    assert(!llvm::sys::fs::remove(filename));
  }
  return 0;
}
} // namespace tilemega::tests::dm_forward_search_test
