#include <kernel_operator.h>
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ __aicore__ __mix__(1, 2) __schedmode__(1)
void as2_timer(GM_ADDR output, uint32_t groups, uint64_t ticks) {
  uint32_t worker = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV { worker += groups; }
  const uint64_t begin = AscendC::GetSystemCycle();
  uint64_t now = begin;
  uint32_t polls = 0;
  // The iteration cap is a second escape hatch, not a conversion to seconds.
  while (now - begin < ticks && polls < 100000000) {
    AscendC::Nop<8>();
    now = AscendC::GetSystemCycle();
    ++polls;
  }
  AscendC::GlobalTensor<uint64_t> record;
  record.SetGlobalBuffer((__gm__ uint64_t*)output + worker * 64);
  record.SetValue(0, worker);
  record.SetValue(1, begin);
  record.SetValue(2, now);
  record.SetValue(3, polls);
  AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(record);
}

static void Check(aclError code, const char* call) {
  if (code) throw std::runtime_error(std::string(call) + ": " + std::to_string(code));
}
int main() {
  try {
    Check(aclInit(nullptr), "aclInit");
    uint32_t count = 0;
    Check(aclrtGetDeviceCount(&count), "device count");
    if (count != 1) throw std::runtime_error("Require one visible device");
    Check(aclrtSetDevice(0), "set device");
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950PR") != 0)
      throw std::runtime_error("Require inventoried 950PR");
    aclrtStream stream;
    aclrtEvent start, end;
    Check(aclrtCreateStream(&stream), "stream");
    Check(aclrtCreateEvent(&start), "start event");
    Check(aclrtCreateEvent(&end), "end event");
    constexpr uint32_t groups = 2, workers = 3 * groups;
    std::vector<uint64_t> host(workers * 64);
    void* device = nullptr;
    Check(aclrtMalloc(&device, host.size() * 8, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    as2_timer<<<groups, nullptr, stream>>>((uint8_t*)device, groups, 250000);
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    std::cout << "{\"soc\":\"" << soc << "\",\"groups\":2,\"samples\":[";
    bool first = true;
    float elapsed[2] = {};
    unsigned sample = 0;
    for (uint64_t ticks : {4000000ULL, 16000000ULL}) {
      Check(aclrtMemset(device, host.size() * 8, 0, host.size() * 8), "zero");
      Check(aclrtRecordEvent(start, stream), "record start");
      as2_timer<<<groups, nullptr, stream>>>((uint8_t*)device, groups, ticks);
      Check(aclrtRecordEvent(end, stream), "record end");
      Check(aclrtSynchronizeStream(stream), "synchronize");
      float ms = 0;
      Check(aclrtEventElapsedTime(&ms, start, end), "elapsed event");
      elapsed[sample++] = ms;
      Check(aclrtMemcpy(host.data(), host.size() * 8, device, host.size() * 8,
                        ACL_MEMCPY_DEVICE_TO_HOST), "read records");
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"requested_ticks\":" << ticks << ",\"acl_elapsed_ms\":" << ms
                << ",\"worker_deltas\":[";
      for (uint32_t w = 0; w < workers; ++w) {
        const auto* r = host.data() + w * 64;
        if (r[0] != w || r[2] < r[1] || r[2] - r[1] < ticks || !r[3])
          throw std::runtime_error("Timer did not advance to bounded target");
        if (w) std::cout << ',';
        std::cout << r[2] - r[1];
      }
      std::cout << "]}";
    }
    // Subtract the two event durations to remove fixed dispatch/event cost.
    // The 20% interval was set before this longer differential experiment.
    const double delta_ms = elapsed[1] - elapsed[0];
    const bool admitted = delta_ms >= 9.6 && delta_ms <= 14.4;
    Check(aclrtFree(device), "free");
    Check(aclrtDestroyEvent(start), "destroy start");
    Check(aclrtDestroyEvent(end), "destroy end");
    Check(aclrtDestroyStream(stream), "destroy stream");
    Check(aclFinalize(), "finalize");
    std::cout << "],\"differential_ms\":" << delta_ms
              << ",\"status\":\"" << (admitted ? "passed" : "inconclusive")
              << "\",\"counter_hz\":" << (admitted ? "1000000000" : "null")
              << ",\"host_kernel_launches\":3,\"tilemega_pipeline\":false}" << std::endl;
    return admitted ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
