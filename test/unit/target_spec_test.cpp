#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Target/TargetSpec.h>

#include <cassert>
#include <string>

int main() {
  using tilemega::TargetSpec;
  auto sm80 = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_80.json");
  auto sm120 = TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_120.json");
  assert(sm80.caps.cp_async && !sm80.caps.cluster);
  assert(sm120.caps.cluster && sm120.caps.tma && !sm120.caps.tcgen05);
  assert(TargetSpec::ComputeStages(100, 30, 10, 16) == 3);
  assert(TargetSpec::ComputeStages(8, 16, 0, 16) == 0);
  assert(sm120.ToJson().find("\"tcgen05\": false") != std::string::npos);
  static_assert(!tilemega::arch::Caps<tilemega::arch::Sm120>::kTcgen05);
  return 0;
}
