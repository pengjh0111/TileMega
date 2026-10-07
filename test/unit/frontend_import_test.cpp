// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Solver/ModelDescription.h>

#include <mlir/IR/MLIRContext.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <set>
#include <stdexcept>
#include <string>

namespace tilemega::tests::frontend_import_test {

int TestFrontendImport(int argc, char** argv) {
  {
    using namespace tilemega::frontend;
    auto bridge=ReadExportBridge(std::string(TILEMEGA_SOURCE_DIR)+
                                 "/test/fixtures/export_structured_args.json");
    auto const& node=bridge.tasks.at(0);using Kind=FxArgument::Kind;
    assert(node.has_scalars && node.scalars==std::vector<double>{1e-6});
    assert(node.has_arguments && node.args.size()==13);
    assert(node.args[0].kind==Kind::kNode && node.args[0].text=="x");
    assert(node.args[2].kind==Kind::kNone);
    assert(node.args[3].kind==Kind::kList && node.args[3].items[0].integer==2);
    assert(node.args[5].kind==Kind::kBool && !node.args[5].boolean);
    assert(node.args[6].kind==Kind::kString && node.args[6].text=="tanh");
    assert(std::isinf(node.args[7].real) && node.args[7].real<0);
    assert(node.args[8].kind==Kind::kDtype && node.args[8].text=="torch.bfloat16");
    assert(node.args[9].kind==Kind::kDevice && node.args[9].text=="cpu");
    assert(node.args[10].kind==Kind::kLayout);
    assert(node.args[11].kind==Kind::kMemoryFormat);
    assert(node.args[12].items[0].items[0].integer==3);
    assert(node.kwargs.at("scale").real==0.5 && !node.kwargs.at("enabled").boolean);
    auto const& constant=bridge.tasks.at(1).constant;
    assert(constant.present && constant.has_all_true && constant.all_true);
    assert(constant.has_all_equal && constant.all_equal && constant.equal_value.boolean);
    assert((constant.shape==std::vector<std::int64_t>{1}));
    assert(constant.data_base64=="AQ==" && constant.byte_order=="little");
    auto const& shape=bridge.tasks.at(2);
    assert(shape.shape_constant_symbols==std::vector<std::string>{"s0"});
    assert(shape.shape_constant_bindings.at("s0")==8 && !shape.shape_constant_fragment_json.empty());
  }
  tilemega::analysis::IslContext isl_context;
  mlir::MLIRContext context;
  tilemega::frontend::ImportSummary summary;
  auto module = tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) +
          "/docs/experiments/E2E_GEN/raw/export_bridge.json",
      context, &summary);
  // Operator granularity: one task space per L-task node of the instantiated
  // OperatorGraph (34 for the two-layer fixture), one coupling per
  // (consumer, in-graph operand) pair.  The previous 179/222 counted FX nodes
  // and the placeholder edges between them.
  assert(summary.task_spaces == 34 && summary.couplings == 42);
  assert(summary.stages == 30 && summary.guards == 4);
  assert(module->getOperation()->getAttr("tilemega.model_plan"));
  {
    auto source=llvm::MemoryBuffer::getFile(std::string(TILEMEGA_SOURCE_DIR)+
                   "/docs/experiments/E2E_GEN/raw/export_bridge.json");
    assert(source);auto json=llvm::json::parse(source.get()->getBuffer());assert(json);
    for(auto& value:*json->getAsObject()->getArray("nodes")) {
      auto* node=value.getAsObject();
      (*node)["args"]=llvm::json::Object{{"t","list"},{"v",llvm::json::Array{}}};
      (*node)["kwargs"]=llvm::json::Object{};
    }
    int fd;llvm::SmallString<128> filename;
    assert(!llvm::sys::fs::createTemporaryFile("dm1-enriched-bridge","json",fd,filename));
    {llvm::raw_fd_ostream out(fd,true);out<<llvm::formatv("{0:2}",*json);}
    auto enriched=tilemega::frontend::TorchExportImporter{}.Import(filename.str().str(),context);
    std::string original_text,enriched_text;
    {llvm::raw_string_ostream out(original_text);module->print(out);}
    {llvm::raw_string_ostream out(enriched_text);enriched->print(out);}
    assert(original_text==enriched_text);
    auto old_bridge=tilemega::frontend::ReadExportBridge(std::string(TILEMEGA_SOURCE_DIR)+
                   "/docs/experiments/E2E_GEN/raw/export_bridge.json");
    assert(!old_bridge.nodes.front().has_arguments && !old_bridge.nodes.front().constant.present);
    auto* first=json->getAsObject()->getArray("nodes")->front().getAsObject();
    auto rejects=[&] {
      {std::ofstream out(filename.str().str());out<<llvm::formatv("{0:2}",*json).str();}
      bool rejected=false;
      try {(void)tilemega::frontend::ReadExportBridge(filename.str().str());}
      catch(std::invalid_argument const&) {rejected=true;}
      assert(rejected);
    };
    (*first)["args"]=llvm::json::Object{{"t","list"},{"v",llvm::json::Array{
        llvm::json::Object{{"t","bool"},{"v",0}}}}};
    rejects();
    (*first)["args"]=llvm::json::Object{{"t","list"},{"v",llvm::json::Array{}}};
    (*first)["kwargs"]=llvm::json::Array{};rejects();
    (*first)["kwargs"]=llvm::json::Object{};
    (*first)["constant"]=llvm::json::Object{{"dtype","torch.bool"},
        {"shape",llvm::json::Array{1025,1025}}};rejects();
    (*first)["constant"]=llvm::json::Object{{"dtype","torch.bool"},
        {"shape",llvm::json::Array{1048577,0}}};
    {std::ofstream out(filename.str().str());out<<llvm::formatv("{0:2}",*json).str();}
    auto empty=tilemega::frontend::ReadExportBridge(filename.str().str());
    assert(empty.nodes.front().constant.present);
    assert(!llvm::sys::fs::remove(filename));
  }
  auto aliases = module->getOperation()->getAttrOfType<mlir::DictionaryAttr>(
      "tilemega.symbol_aliases");
  assert(aliases.getAs<mlir::StringAttr>("s61").getValue() == "s14");
  assert(aliases.getAs<mlir::StringAttr>("s65").getValue() == "s14");
  // A task space now names the role the semantic lifting recognised, not the
  // FX target it came from, so `view`/`transpose` no longer appear as kinds.
  // What replaces that coverage: every recognised role must be present and
  // none may have degraded to `generic` on a model the frontend claims to
  // cover.
  std::set<std::string> kinds;
  for (auto task : module->getOps<tilemega::dialect::TileSpaceOp>()) {
    auto kind = task.getWriteMap().getFields().getAs<mlir::StringAttr>("kind").getValue();
    assert(kind != "generic");
    kinds.insert(kind.str());
  }
  assert((kinds == std::set<std::string>{"attention", "elementwise", "gemm",
                                         "kvappend", "rmsnorm", "rope"}));
  // The gap this whole change closes: C used to be the constant
  // `{ [0] -> [0] }` for every edge.  No edge may carry a one-point relation.
  for (auto coupling : module->getOps<tilemega::dialect::CouplingOp>()) {
    std::string relation = coupling.getRelation().getMap().ToString();
    assert(relation.find("[0] -> [0]") == std::string::npos);
  }

  std::string cuda =
      tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
  assert(cuda.find("constexpr ModelSpec kModel") != std::string::npos);
  assert(cuda.find("kStages, 30u") != std::string::npos);
  assert(cuda.find("TILEMEGA_GENERATED_WAIT_global") != std::string::npos);
  assert(cuda.find("TILEMEGA_GENERATED_NOTIFY_global") != std::string::npos);
  assert(cuda.find("TILEMEGA_GENERATED_RESIDENT_GRID") != std::string::npos);
  assert(cuda.find("ModelHarness.cuh") != std::string::npos);
  assert(cuda.find("GeneratedLlamaRuntime.cuh") == std::string::npos);
  assert(cuda.find("TILEMEGA_DM_SUPPORT")==std::string::npos);
  {
    using namespace tilemega::frontend;
    using namespace tilemega::codegen;
    auto bridge=ReadExportBridge(std::string(TILEMEGA_SOURCE_DIR)+
                                "/docs/experiments/E2E_GEN/raw/export_bridge.json");
    auto plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
    mlir::Builder builder(&context);
    auto original=module->getOperation()->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
    assert(!original.get("dm") && !original.get("dm_convolutions"));
    plan.dm=true;
    plan.convolutions.push_back({2,224,224,3,64,7,7,2,2,3,3,1,1,112,112,0,1});
    auto& gemm=plan.gemms.front(); gemm.access.rows_per_batch=128;
    gemm.access.a_row_stride=128; gemm.access.a_scale=2;
    gemm.access.write={DmWriteKind::kPixelShuffle,2,1,kDmNoIndex};
    gemm.chain.count=3;
    gemm.chain.operations[0].kind=DmEpilogueKind::kBias;
    gemm.chain.operations[0].parameter[0]=2;
    gemm.chain.operations[1].kind=DmEpilogueKind::kResidual;
    gemm.chain.operations[1].parameter[0]=3;
    gemm.chain.operations[2].kind=DmEpilogueKind::kActivation;
    gemm.chain.operations[2].activation=DmActivation::kRelu;
    gemm.chain.side_count=1;
    gemm.chain.side[0]={DmSideOutputKind::kRowStats,4,kDmNoIndex,0};
    plan.stages.front().rows_per_batch=128;
    auto dm_attr=EncodeModelPlan(builder,plan,{});
    assert(dm_attr.getAs<mlir::BoolAttr>("dm").getValue());
    auto g=mlir::cast<mlir::DictionaryAttr>(dm_attr.getAs<mlir::ArrayAttr>("gemms")[0]);
    assert(DecodeDmAccess(g.get("dm_access")).a_row_stride==128);
    assert(DecodeDmChain(g.get("dm_chain")).operations[1].parameter[0]==3);
    mlir::OwningOpRef<mlir::ModuleOp> dm_module(module->clone());
    dm_module->getOperation()->setAttr("tilemega.model_plan",dm_attr);
    auto emitted=CouplingGraphToCUDA{}.Lower(*dm_module);
    auto description=tilemega::solver::ModelDescription::FromCouplingGraph(
        *dm_module,{1,0,1},"descriptor-codegen-fixture");
    assert(description.dm && description.convolutions.size()==1);
    assert(description.gemm_access[0].a_row_stride==128);
    assert(description.epilogue_chains[0].count==3);
    assert(description.stages.front().rows_per_batch==128);
    assert(emitted.find("#define TILEMEGA_DM_SUPPORT 1\n")!=std::string::npos);
    assert(emitted.find("constexpr ConvDesc kConvolutions[]")!=std::string::npos);
    assert(emitted.find("using DmChain0 = DmEpilogueKinds<")!=std::string::npos);
    assert(emitted.find("kConvolutions, 1u}")!=std::string::npos);
    if(argc==3 && std::string(argv[1])=="--emit-dm") {
      std::ofstream output(argv[2]); output<<emitted; assert(output.good());
    }
    plan.gemms.front().access.a_scale=plan.buffers.size();
    bool rejected=false;
    try {(void)EncodeModelPlan(builder,plan,{});} catch(std::invalid_argument const&) {rejected=true;}
    assert(rejected);
    dm_module->getOperation()->setAttr("tilemega.model_plan",original);
    assert(CouplingGraphToCUDA{}.Lower(*dm_module)==cuda);
  }
  {
    tilemega::frontend::ImportOptions separate;
    separate.separate_residual_tasks=true;
    tilemega::frontend::ImportSummary expanded_summary;
    auto expanded=tilemega::frontend::TorchExportImporter{}.Import(
        std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/E2E_GEN/raw/export_bridge.json",
        context,&expanded_summary,separate);
    assert(expanded_summary.stages==summary.stages+4);
    auto expanded_cuda=tilemega::codegen::CouplingGraphToCUDA{}.Lower(*expanded);
    assert(expanded_cuda.find("TaskKind::kAdd")!=std::string::npos);
    assert(expanded_cuda.find(".gemm_product")!=std::string::npos);
  }
  assert(cuda.find("% 12") == std::string::npos);
  assert(cuda.find("wait_table=") == std::string::npos);
  assert(cuda.find("kRuntimeVariants") != std::string::npos);
  assert(cuda.find("constexpr ScheduleStageDesc kSchedule0[]") !=
         std::string::npos);
  assert(cuda.find("kDependencyOffsets0, kSchedule0, 30u") !=
         std::string::npos);
  {
    mlir::OwningOpRef<mlir::ModuleOp> resident(module->clone());
    for (auto placement:resident->getOps<tilemega::dialect::PlacementOp>())
      placement->setAttr("resident_only",mlir::BoolAttr::get(&context,true));
    auto emitted=tilemega::codegen::CouplingGraphToCUDA{}.Lower(*resident);
    assert(emitted.find(", nullptr, true}")!=std::string::npos);
    auto first=*resident->getOps<tilemega::dialect::PlacementOp>().begin();
    first->removeAttr("resident_only");
    int before=isl_context.ReferenceCount();
    bool rejected=false;
    try { (void)tilemega::codegen::CouplingGraphToCUDA{}.Lower(*resident); }
    catch (std::invalid_argument const&) { rejected=true; }
    assert(rejected && isl_context.ReferenceCount()==before);
  }
  {
    auto balanced=module->clone();
    for (auto placement:balanced.getOps<tilemega::dialect::PlacementOp>()) {
      placement->setAttr("resident_only",mlir::BoolAttr::get(&context,true));
      placement->setAttr("mapping_mode",mlir::StringAttr::get(&context,"balanced"));
    }
    auto emitted=tilemega::codegen::CouplingGraphToCUDA{}.Lower(balanced);
    assert(emitted.find(", nullptr, true, true}")!=std::string::npos);
    balanced.erase();
  }

  // Phase-5 prerequisite: two independently instantiated granularities are
  // fused into one binary, and ModelSpec -- not an external -D plan -- binds
  // each interval to both its GEMM implementation and its dependency table.
  tilemega::frontend::ImportOptions coarse;
  coarse.gemms.assign(14, {16, 64, 16, 2, 16});
  auto coarse_module = tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) +
          "/docs/experiments/E2E_GEN/raw/export_bridge.json",
      context, nullptr, coarse);
  std::string multi = tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(
      {{*module, 1, 3}, {*coarse_module, 4, 2048}});
  assert(multi.find("#define TILEMEGA_GEMM_VARIANT_COUNT 2") !=
         std::string::npos);
  assert(multi.find("{kRuntimeGemms0, kDependencies0") != std::string::npos);
  assert(multi.find("{kRuntimeGemms1, kDependencies1") != std::string::npos);
  assert(multi.find("constexpr ScheduleStageDesc kSchedule0[]") !=
         std::string::npos);
  assert(multi.find("constexpr ScheduleStageDesc kSchedule1[]") !=
         std::string::npos);
  assert(multi.find("table[s] = 1u") != std::string::npos);
  assert(multi.find("wait_table=") == std::string::npos);

  // P4.7: the cluster shape is a property of the whole launch, so the
  // generator's contract is all-or-nothing.  Flipping every coupling and every
  // placement together must produce a clustered kernel; flipping only one side
  // must be rejected rather than silently resolved to the other.
  {
    auto clustered = module->clone();
    auto sync = tilemega::dialect::SyncKindAttr::get(
        &context, mlir::StringAttr::get(&context, "cluster"));
    for (auto coupling : clustered.getOps<tilemega::dialect::CouplingOp>())
      coupling.setSyncKindAttr(sync);
    std::string half;
    try {
      half = tilemega::codegen::CouplingGraphToCUDA{}.Lower(clustered);
    } catch (std::invalid_argument const&) {
      half = "rejected";
    }
    assert(half == "rejected");
    for (auto placement : clustered.getOps<tilemega::dialect::PlacementOp>())
      placement.setCluster(2);
    std::string whole = tilemega::codegen::CouplingGraphToCUDA{}.Lower(clustered);
    assert(whole.find("#define TILEMEGA_GENERATED_CLUSTER_DIM 2") !=
           std::string::npos);
    // The flat kernel must keep saying nothing about clusters, so a default
    // build cannot pick the macro up by accident.
    assert(cuda.find("TILEMEGA_GENERATED_CLUSTER_DIM") == std::string::npos);
    clustered->erase();
  }

  // An uncovered operator degrades instead of being rejected; `degradation_test`
  // holds the rest of that contract.
  tilemega::frontend::ImportSummary degraded;
  (void)tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) +
          "/test/fixtures/export_unsupported.json", context, &degraded);
  assert(degraded.degraded.size() == 1 &&
         degraded.degraded.front() == "aten.imaginary.default");

  auto domain = tilemega::frontend::SymbolicShapeBridge{}.Parse(
      {{"s0", "VR[1, 1024]"}},
      {"L['flat_args'][0].size()[0] % 128 == 0",
       "L['flat_args'][0].size()[0] <= 512"},
      {{"s0"}});
  assert(domain.constraints.size() == 2);
  assert(domain.constraints[0].predicate ==
         tilemega::frontend::ShapeConstraint::Predicate::kDivisible);
  assert(domain.constraints[1].predicate ==
         tilemega::frontend::ShapeConstraint::Predicate::kLessEqual);
  return 0;

  return 0;
}

}  // namespace tilemega::tests::frontend_import_test
