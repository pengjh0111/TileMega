#include <kernel_operator.h>
#include <acl/acl.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef AS1_UB_BYTES
#define AS1_UB_BYTES (212*1024)
#endif

__aicore__ inline void Record(GM_ADDR output, uint32_t role, uint32_t first, uint32_t last) {
  auto* words = (__gm__ uint32_t*)output;
  AscendC::WriteGmByPassDCache(words, role);
  AscendC::WriteGmByPassDCache(words+1, uint32_t(AscendC::GetBlockIdx()));
  AscendC::WriteGmByPassDCache(words+2, uint32_t(AscendC::GetBlockNum()));
  AscendC::WriteGmByPassDCache(words+3, uint32_t(AscendC::GetSubBlockIdx()));
  AscendC::WriteGmByPassDCache(words+4, uint32_t(AscendC::GetSubBlockNum()));
  AscendC::WriteGmByPassDCache(words+5, first);
  AscendC::WriteGmByPassDCache(words+6, last);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}
__global__ __cube__ void as1_cube_entry(GM_ADDR output) { Record(output, 1, 42, 173); }

__global__ __vector__ void as1_vector_entry(GM_ADDR output) {
  __ubuf__ volatile uint8_t scratch[AS1_UB_BYTES];
  for (unsigned i = 0; i < AS1_UB_BYTES; i += 32) scratch[i] = uint8_t(i/32);
  scratch[0] = 42; scratch[AS1_UB_BYTES-1] = 173;
  Record(output, 2, scratch[0], scratch[AS1_UB_BYTES-1]);
}
static void Check(aclError e, const char* name) {
  if (e) throw std::runtime_error(std::string(name)+": "+std::to_string(e));
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: as1_entry_resources cube|vector");
    std::string kind = argv[1];
    if (kind != "cube" && kind != "vector") throw std::runtime_error("Unknown role");
    if (AS1_UB_BYTES > 216*1024) throw std::runtime_error("Host preflight rejects UB above inventoried ACL budget");
    Check(aclInit(nullptr), "init");
    uint32_t count = 0; Check(aclrtGetDeviceCount(&count), "device count");
    if (count != 1) throw std::runtime_error("Use single-device wrapper");
    Check(aclrtSetDevice(0), "set device");
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950PR") != 0) throw std::runtime_error("Wrong SKU");
    aclrtStream stream = nullptr; Check(aclrtCreateStream(&stream), "stream");
    uint8_t* output = nullptr; Check(aclrtMalloc((void**)&output, 512, ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
    Check(aclrtMemset(output, 512, 0, 512), "clear");
    if (kind == "cube") as1_cube_entry<<<1, nullptr, stream>>>(output);
    else as1_vector_entry<<<1, nullptr, stream>>>(output);
    Check(aclrtSynchronizeStream(stream), "synchronize");
    std::vector<uint32_t> host(128);
    Check(aclrtMemcpy(host.data(), 512, output, 512, ACL_MEMCPY_DEVICE_TO_HOST), "read");
    if (host[0] != (kind == "cube" ? 1u : 2u) || host[1] != 0 || host[2] != 1 ||
        host[5] != 42 || host[6] != 173) throw std::runtime_error("Entry or UB boundary mismatch");
    Check(aclrtFree(output), "free"); Check(aclrtDestroyStream(stream), "destroy"); Check(aclFinalize(), "finalize");
    std::cout << "{\"status\":\"passed\",\"entry\":\"" << kind << "\",\"block\":" << host[1]
              << ",\"blocks\":" << host[2] << ",\"subblock\":" << host[3] << ",\"subblocks\":" << host[4]
              << ",\"ub_bytes\":" << (kind == "vector" ? AS1_UB_BYTES : 0)
              << ",\"host_kernel_launches\":1}" << std::endl;
    return 0;
  } catch (std::exception const& e) { std::cerr << e.what() << std::endl; return 1; }
}
