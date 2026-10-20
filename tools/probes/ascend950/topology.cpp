#include <kernel_operator.h>
#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

// Each record occupies 512 bytes to separate workers' scalar stores.
static constexpr uint32_t kWords = 64;

__global__ __aicore__ __mix__(1, 2) __schedmode__(1)
void as1_topology_kernel(GM_ADDR memory, uint32_t groups, uint32_t tiles) {
  const uint32_t block = AscendC::GetBlockIdx();
  uint32_t role = 0, worker = 0;
  if ASCEND_IS_AIC {
    role = 1;
    worker = block;
  } else if ASCEND_IS_AIV {
    role = 2;
    worker = groups + block;
  }
  if (!role || worker >= 3 * groups) return;
  for (uint32_t epoch = 0; epoch < tiles; ++epoch) {
    AscendC::GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer((__gm__ uint64_t*)memory + (worker * tiles + epoch) * kWords);
    const auto start = AscendC::GetSystemCycle();
    output.SetValue(0, 0x415331);
    output.SetValue(1, role);
    output.SetValue(2, block);
    output.SetValue(3, AscendC::GetSubBlockIdx());
    output.SetValue(4, AscendC::GetBlockNum());
    output.SetValue(5, AscendC::GetSubBlockNum());
    output.SetValue(6, worker);
    output.SetValue(7, epoch);
    output.SetValue(8, start);
    output.SetValue(9, AscendC::GetSystemCycle());
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(output);
  }
}

static void Check(aclError error, const char* call) {
  if (error) throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: as1_topology groups tiles");
    const int groups = std::stoi(argv[1]), tiles = std::stoi(argv[2]);
    if (groups < 1 || groups > 4 || tiles < 1 || tiles > 16)
      throw std::runtime_error("Bounded probe requires groups=1..4 and tiles=1..16");
    Check(aclInit(nullptr), "aclInit");
    uint32_t visible = 0;
    Check(aclrtGetDeviceCount(&visible), "aclrtGetDeviceCount");
    if (visible != 1) throw std::runtime_error("Use the single-device wrapper");
    Check(aclrtSetDevice(0), "aclrtSetDevice");
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950PR") != 0)
      throw std::runtime_error("Expected the inventoried Ascend950PR target");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    const size_t words = size_t(groups) * 3 * tiles * kWords;
    std::vector<uint64_t> host(words, 0);
    uint8_t* memory = nullptr;
    Check(aclrtMalloc((void**)&memory, words * 8, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    Check(aclrtMemset(memory, words * 8, 0, words * 8), "aclrtMemset");
    as1_topology_kernel<<<groups, nullptr, stream>>>(memory, groups, tiles);
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    Check(aclrtMemcpy(host.data(), words * 8, memory, words * 8, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy");
    std::cout << "{\"groups\":" << groups << ",\"tiles_per_worker\":" << tiles
              << ",\"host_kernel_launches\":1,\"records\":[";
    bool first = true;
    for (int worker = 0; worker < 3 * groups; ++worker) {
      for (int epoch = 0; epoch < tiles; ++epoch) {
        auto* data = host.data() + (worker * tiles + epoch) * kWords;
        if (data[0] != 0x415331 || data[6] != unsigned(worker) || data[7] != unsigned(epoch))
          throw std::runtime_error("Missing, duplicate, or misaddressed worker/epoch record");
        const bool aic = worker < groups;
        const unsigned expected_block = aic ? worker : worker-groups;
        if (data[1] != unsigned(aic ? 1 : 2) || data[2] != expected_block ||
            data[3] != (aic ? 0 : expected_block % 2) || data[4] != unsigned(groups) ||
            data[5] != unsigned(aic ? 1 : 2) || data[9] < data[8])
          throw std::runtime_error("Unexpected MIX ID mapping or nonmonotone timer");
        if (!first) std::cout << ',';
        first = false;
        std::cout << '[';
        for (int field = 1; field <= 9; ++field) {
          if (field != 1) std::cout << ',';
          std::cout << data[field];
        }
        std::cout << ']';
      }
    }
    std::cout << "],\"status\":\"passed\",\"timer_unit\":\"cycles_unscaled\"}" << std::endl;
    Check(aclrtFree(memory), "aclrtFree");
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclFinalize(), "aclFinalize");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
