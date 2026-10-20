// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/AscendCodegen.h>
#include <mlir/IR/Verifier.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tilemega::codegen {
std::string LowerAscend(mlir::ModuleOp module,solver::SymbolicProblem const& p,solver::AscendPlan const& plan) {
  if(!module || mlir::failed(mlir::verify(module)) || p.model.dtype!=solver::ScalarType::kBF16 ||
     p.model.gemms.size()!=1 || p.model.stages.size()!=1 || p.runtime.gemms.size()!=1 ||
     p.model.stages[0].kind!=solver::StageKind::kGemm || p.runtime.gemms[0].split_k!=1 ||
     plan.stage_implementations.size()!=1 || plan.stage_implementations[0].impl_id!="catlass_950_bf16_32x64x64_gm")
    throw std::invalid_argument("Native codegen requires the verified single BF16 GEMM implementation");
  auto model=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto gemms=model?model.getAs<mlir::ArrayAttr>("gemms"):mlir::ArrayAttr{};
  auto outputs=model?model.getAs<mlir::ArrayAttr>("outputs"):mlir::ArrayAttr{};
  if(!gemms || gemms.size()!=1 || !outputs || outputs.size()!=1)
    throw std::invalid_argument("Native GEMM requires exactly one model output");
  auto g=mlir::cast<mlir::DictionaryAttr>(gemms[0]);
  auto output=mlir::cast<mlir::DictionaryAttr>(outputs[0]);
  auto beta=g.getAs<mlir::FloatAttr>("beta");
  auto epilogue=g.getAs<mlir::StringAttr>("epilogue");
  if(!beta || beta.getValueAsDouble()!=0 || (epilogue && epilogue.getValue()!="store") ||
     g.getAs<mlir::IntegerAttr>("d")!=output.getAs<mlir::IntegerAttr>("buffer") ||
     g.get("norm_ss") || g.get("ss_out"))
    throw std::invalid_argument("GEMM residual, reduction side output or epilogue is unsupported");
  if(p.model.gemm_access.size()!=1 || p.model.epilogue_chains.size()!=1 || p.model.epilogue_chains[0].count ||
     p.model.gemm_access[0].a!=DmAAccess::kDense || p.model.gemm_access[0].b!=DmBAccess::kDense ||
     p.model.gemm_access[0].a_row_offset || p.model.gemm_access[0].a_row_stride>1 ||
     p.model.gemm_access[0].write.kind!=DmWriteKind::kDense)
    throw std::invalid_argument("Unsupported GEMM access/epilogue rejected before native emission");
  auto const& geom=p.runtime.gemms[0];auto const& gemm=p.model.gemms[0];
  if(geom.tile_m!=32 || geom.tile_n!=64 || geom.tile_k!=64 || geom.stages!=2)
    throw std::invalid_argument("Plan geometry disagrees with native CATLASS body");
  unsigned m=p.model.gemm_access[0].rows_per_batch*p.model.dims.batch,n=gemm.n,k=gemm.k;
  if(!((m==64&&n==128&&k==256)||(m==33&&n==96&&k==129)) ||
      plan.tasks.size()!=((m+31)/32)*((n+63)/64))
    throw std::invalid_argument("Native candidate shape or CTG tile coverage is outside the verified scope");
  std::ostringstream tables;
  tables<<"static constexpr unsigned kM="<<m<<",kN="<<n<<",kK="<<k<<",kPhysicalK="<<((k+15)/16*16)
    <<",kGroups="<<plan.metadata.groups<<",kTaskCount="<<plan.tasks.size()<<",kScheduleCount="<<plan.schedule.size()<<";\n";
  auto metadata=plan.metadata;
  if(metadata.buffers!=3)throw std::invalid_argument("Native E-0 requires exactly three model buffers");
  tables<<"static constexpr abi::Metadata kMetadata={"<<metadata.version<<','<<metadata.groups<<','<<metadata.workers
    <<','<<metadata.tasks<<','<<metadata.dependencies<<','<<metadata.epochs<<','<<metadata.stages<<','<<metadata.buffers<<"};\n";
  tables<<"static constexpr abi::TaskRef kTasks[]={";
  for(auto const& t:plan.tasks)tables<<'{'<<t.stage<<','<<t.logical_task<<','<<t.dependency_begin<<','<<t.dependency_count
    <<','<<t.wait_begin<<','<<t.wait_count<<','<<t.flags<<"},";
  tables<<"};\nstatic constexpr abi::TaskMeta kOwners[]={";
  for(auto const& t:plan.task_metadata)tables<<'{'<<t.owner<<','<<t.event<<"},";
  tables<<"};\n";
  auto array=[&](char const* name,std::vector<uint32_t> const& values) {
    tables<<"static constexpr uint32_t "<<name<<"[]={";
    if(values.empty())tables<<0;else for(auto v:values)tables<<v<<',';
    tables<<"};\n";
  };
  array("kOffsets",plan.schedule_offsets);array("kSchedule",plan.schedule);array("kDependencies",plan.dependencies);
  std::ifstream input(std::string(TILEMEGA_SOURCE_DIR)+"/lib/Codegen/Ascend/PersistentGemm.cpp.in");
  if(!input)throw std::runtime_error("Missing native codegen template");
  std::string source((std::istreambuf_iterator<char>(input)),{}),marker="@AS1_PLAN_TABLES@";
  auto pos=source.find(marker);
  if(pos==std::string::npos || source.find(marker,pos+marker.size())!=std::string::npos)
    throw std::runtime_error("Native codegen template has ambiguous plan insertion");
  source.replace(pos,marker.size(),tables.str());return source;
}
} // namespace tilemega::codegen
