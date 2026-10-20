#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Solver/AscendImplementation.h>
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/ImplementationContract.h>
#include <tilemega/Support/Json.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

template<class F> static void Reject(F f) {
  bool failed=false;try{f();}catch(std::invalid_argument const&){failed=true;}
  assert(failed);
}
int main() {
  using namespace tilemega;
  analysis::IslContext isl;
  solver::ImplementationContract exact;
  exact.name="logical_NK_fixture";
  auto read=analysis::CouplingRelation::FromIslText("{ [m,n] -> [j,k] : 0<=m<2 and 0<=n<2 and n<=j<n+1 and 0<=k<2 }");
  exact.exact_operands={read};std::string reason;
  assert(solver::VerifyExactAccess(exact,{read},&reason));
  auto transposed=analysis::CouplingRelation::FromIslText("{ [m,n] -> [k,j] : 0<=m<2 and 0<=n<2 and n<=j<n+1 and 0<=k<2 }");
  assert(!solver::VerifyExactAccess(exact,{transposed},&reason));
  auto clipped=analysis::CouplingRelation::FromIslText("{ [m,n] -> [j,k] : 0<=m<2 and 0<=n<2 and n<=j<n+1 and 0<=k<1 }");
  assert(!solver::VerifyExactAccess(exact,{clipped},&reason));
  auto source=std::filesystem::path(TILEMEGA_SOURCE_DIR)/"configs/targets/ascend_3510.json";
  auto target=TargetSpec::FromJson(source.string());
  assert(target.arch_tag=="ascend_3510" && target.ascend_capability);
  assert(target.ascend_capability->TrackCount(ascend::TrackRole::kAic)==28);
  assert(target.ascend_capability->TrackCount(ascend::TrackRole::kAiv)==56);
  assert(!target.ascend_capability->same_group_l0c_to_ub && target.ascend_capability->gm_event);
  assert(!target.res.num_sms && !target.caps.cluster && !target.caps.tma);
  auto serialized=json::Parse(target.ToJson());
  assert(serialized.At("calibration_status").AsString("calibration_status")=="not_calibrated");
  solver::CandidateGenerator generator(target,solver::ScalarType::kBF16);
  solver::EnumerationStats stats;auto candidates=generator.Enumerate(&stats);
  assert(candidates.size()==1 && stats.touched==1 && stats.cartesian==1 && stats.legal==1);
  Reject([&]{generator.TopK(candidates,{{64,128,256}},1);});
  Reject([&]{generator.Analyze(candidates[0],{64,128,256});});
  Reject([&]{target.CalibrationFor("bf16");});
  Reject([&]{target.ActiveBlocksPerSM(nullptr,1);});
  auto impl=*solver::As1Bf16GemmTraits().ascend_implementation;
  impl.Validate(*target.ascend_capability);
  impl.kind=codegen::TaskKind::kRMSNorm;
  Reject([&]{impl.Validate(*target.ascend_capability);});
  auto temp=std::filesystem::path("/home/d50064638/junhuip/tilemega-as1/tmp")/
      ("target_contract_"+std::to_string(getpid())+".json");
  if(std::filesystem::exists(temp))throw std::runtime_error("Use a fresh test output");
  auto invalid=json::ParseFile(source.string());invalid.Set("product_sku","unsupported_sku");
  {std::ofstream out(temp);out<<invalid.Dump();}
  Reject([&]{TargetSpec::FromJson(temp.string());});
  invalid=json::ParseFile(source.string());invalid.Set("arch_tag","unsupported_arch");
  auto invalid_arch=temp;invalid_arch+=".arch";
  {std::ofstream out(invalid_arch);out<<invalid.Dump();}
  Reject([&]{TargetSpec::FromJson(invalid_arch.string());});
  std::cout<<"{\"status\":\"passed\",\"checks\":[\"native_target_parse\",\"one_fixed_candidate\",\"reject_uncalibrated_ranking\",\"reject_unsupported_kind_arch_sku\",\"exact_access_reject_transpose_and_tail\"],\"npu_initialized\":false}\n";
}
