// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Support/Json.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace tilemega::tests::dm_target_memory_test {
int TestDmTargetMemory(int argc,char** argv) {
  auto path=std::filesystem::temp_directory_path()/
      ("tilemega-dm-target-memory-"+std::to_string(getpid())+".json");
  auto read=[&](json::Value const& value) {
    {std::ofstream stream(path);stream<<value.Dump();assert(stream.good());}
    return TargetSpec::FromJson(path.string());
  };
  if(argc==2 && std::string(argv[1])=="--probe") {
    auto target=TargetSpec::Probe();
    cudaDeviceProp properties{};assert(cudaGetDeviceProperties(&properties,0)==cudaSuccess);
    assert(target.res.dram_capacity_bytes==properties.totalGlobalMem);
    assert(target.res.dram_capacity_bytes>0);
    auto restored=read(json::Parse(target.ToJson()));
    assert(restored.res.dram_capacity_bytes==target.res.dram_capacity_bytes);
    std::cout<<"DM_TARGET_MEMORY probe_bytes="<<target.res.dram_capacity_bytes<<" PASS\n";
    std::filesystem::remove(path);return 0;
  }
  unsigned checks=0,rejections=0;
  for(int arch:{80,89,90,100,120}) {
    auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
        "/configs/targets/sm_"+std::to_string(arch)+".json");
    assert(target.res.dram_capacity_bytes==0);
    auto legacy=target.ToJson();
    assert(!json::Parse(legacy).At("resources").Find("dram_capacity_bytes"));
    assert(read(json::Parse(legacy)).ToJson()==legacy);++checks;
    for(std::uint64_t bytes:{std::uint64_t(1),std::uint64_t(24)<<30,
                             std::uint64_t(80)<<30,std::uint64_t(9007199254740991ULL)}) {
      target.res.dram_capacity_bytes=bytes;
      auto encoded=target.ToJson();
      auto restored=read(json::Parse(encoded));
      assert(restored.res.dram_capacity_bytes==bytes && restored.ToJson()==encoded);++checks;
    }
    for(auto value:{json::Value(-1),json::Value(1.5),json::Value(9007199254740992.),
                    json::Value("25769803776"),json::Value(true)}) {
      auto input=json::Parse(legacy),resources=input.At("resources");
      resources.Set("dram_capacity_bytes",value);input.Set("resources",resources);
      bool caught=false;try{(void)read(input);}catch(std::exception const&){caught=true;}
      assert(caught);++rejections;
    }
    auto input=json::Parse(legacy),resources=input.At("resources");
    resources.Set("dram_capacity_bytes",0);input.Set("resources",resources);
    assert(read(input).ToJson()==legacy);++checks;
    target.res.dram_capacity_bytes=9007199254740992ULL;
    bool caught=false;try{(void)target.ToJson();}catch(std::invalid_argument const&){caught=true;}
    assert(caught);++rejections;
  }
  std::filesystem::remove(path);
  assert(checks==30 && rejections==30);
  std::cout<<"DM_TARGET_MEMORY host_checks="<<checks<<" rejections="<<rejections<<" PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_target_memory_test
