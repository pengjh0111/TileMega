#include <acl/acl.h>
#include <runtime/dev.h>
#include <tiling/platform/platform_ascendc.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

static void Check(aclError err, const char *call) {
  if (err != ACL_SUCCESS)
    throw std::runtime_error(std::string(call) + ": " + std::to_string(err));
}

int main() {
  try {
    const char *expected = std::getenv("AS1_PHYSICAL_DEVICE_ID");
    if (!expected) throw std::runtime_error("Use the as1_run.py wrapper");
    Check(aclInit(nullptr), "aclInit");
    uint32_t count = 0, physical = ~0u;
    Check(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount");
    if (count != 1) throw std::runtime_error("Expected exactly one visible NPU");
    Check(rtGetDevicePhyIdByIndex(0, &physical), "rtGetDevicePhyIdByIndex");
    if (physical != std::stoul(expected))
      throw std::runtime_error("Physical/logical mapping mismatch");
    Check(aclrtSetDevice(0), "aclrtSetDevice(0)");
    const char *soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950PR") != 0)
      throw std::runtime_error("This project requires Ascend950PR");
    int64_t aic = 0, aiv = 0, ub = 0;
    Check(aclrtGetDeviceInfo(0, ACL_DEV_ATTR_CUBE_CORE_NUM, &aic), "cube count");
    Check(aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv), "vector count");
    Check(aclrtGetDeviceInfo(0, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub), "UB capacity");
    size_t free = 0, total = 0;
    Check(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total), "aclrtGetMemInfo");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (!platform) throw std::runtime_error("PlatformAscendC unavailable");
    using platform_ascendc::CoreMemType;
    std::cout << "{\"evidence\":\"verified\",\"physical_device_id\":" << physical
              << ",\"logical_device_id\":0,\"visible_count\":" << count
              << ",\"soc\":\"" << soc << "\",\"aic\":" << aic << ",\"aiv\":" << aiv
              << ",\"acl_ub_bytes\":" << ub << ",\"hbm_free_bytes\":" << free
              << ",\"hbm_total_bytes\":" << total << ",\"platform_bytes\":{";
    bool first = true;
    for (auto item : {std::pair{CoreMemType::L1, "l1"}, {CoreMemType::L0_A, "l0a"},
                      {CoreMemType::L0_B, "l0b"}, {CoreMemType::L0_C, "l0c"},
                      {CoreMemType::UB, "ub"}, {CoreMemType::L2, "l2"}}) {
      uint64_t size = 0;
      platform->GetCoreMemSize(item.first, size);
      if (!first) std::cout << ',';
      first = false;
      std::cout << '"' << item.second << "\":" << size;
    }
    std::cout << "},\"device_kernel_launches\":0}" << std::endl;
    // Finalize this process's ACL state. Do not call device reset APIs.
    Check(aclFinalize(), "aclFinalize");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
