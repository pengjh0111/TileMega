#include <tilemega/Target/Ascend/ProgressPrimitives.h>
#include <tilemega/Backend/Ascend/GemmTile.h>
#include <acl/acl.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr uint32_t kElements = 1024, kEpochs = 16, kCell = 128;
constexpr uint64_t kDeadlineTicks = 2000000000ULL;
struct ControlCells {
  __gm__ uint32_t* state;
  uint32_t workers;
  __aicore__ inline __gm__ uint32_t* operator()(uint32_t field, uint32_t worker) const {
    return state + (field * workers + worker) * kCell;
  }
};

__aicore__ inline float Expected(uint32_t worker, uint32_t epoch, uint32_t i) {
  return float(64 * (worker + 1) + epoch + i % 4);
}

__aicore__ inline void Phase(GM_ADDR identity, GM_ADDR values, GM_ADDR payload,
                            GM_ADDR controls, uint32_t groups, uint32_t direction,
                            uint32_t elements) {
  const uint32_t worker = GetWorkerId(groups), workers = 3 * groups;
  const bool aic = worker < groups;
  const uint32_t group = aic ? worker : (worker - groups) / 2;
  const uint32_t lane = aic ? 0 : (worker - groups) % 2;
  const uint32_t prev = (group + groups - 1) % groups, next = (group + 1) % groups;
  uint32_t inputs[2], outputs[2], ni = 0, no = 0;
  if (direction == 0 && aic) { inputs[ni++] = prev; outputs[no++] = next; }
  if (direction == 1) {
    if (aic) { outputs[no++] = groups + 2 * next; outputs[no++] = groups + 2 * next + 1; }
    else inputs[ni++] = prev;
  }
  if (direction == 2) {
    if (aic) { inputs[ni++] = groups + 2 * prev; inputs[ni++] = groups + 2 * prev + 1; }
    else outputs[no++] = next;
  }
  if (direction == 3 && !aic) {
    inputs[ni++] = groups + 2 * prev + 1 - lane;
    outputs[no++] = groups + 2 * next + 1 - lane;
  }
  if (!ni && !no) return;
  auto* state = (__gm__ uint32_t*)controls;
  ControlCells cell{state, workers};
  auto* abort = cell(6, 0);
  AscendC::GlobalTensor<float> data, source;
  data.SetGlobalBuffer((__gm__ float*)payload);
  source.SetGlobalBuffer((__gm__ float*)values + worker * kEpochs * kElements);
  uint32_t mismatches = 0;

  if ASCEND_IS_AIC {
    GemmTile<float, float> tile;
    for (uint32_t epoch = 1; epoch <= kEpochs; ++epoch) {
      if (ni) {
        float old = 0;
        for (uint32_t j = 0; j < ni; ++j)
          for (uint32_t i = 0; i < elements; ++i) old += data.GetValue(inputs[j] * kElements + i);
        WriteControl(cell(5, worker), uint32_t(old));
        PublishSingleOwnerEvent(cell(2, worker), epoch);
      }
      for (uint32_t j = 0; j < no; ++j)
        if (!WaitProgress(cell(1, outputs[j]), epoch - 1, abort, kDeadlineTicks) ||
            !WaitProgress(cell(2, outputs[j]), epoch, abort, kDeadlineTicks)) return;
      if (no) {
        tile.RunTask(identity, values + ((worker * kEpochs + epoch - 1) * kElements) * 4,
                     payload + worker * kElements * 4, 32, 32, 32, 0, 0);
        CompleteDma<AscendC::HardEvent::FIX_S>();
        PublishSingleOwnerEvent(cell(0, worker), epoch);
      }
      for (uint32_t j = 0; j < ni; ++j)
        if (!WaitProgress(cell(0, inputs[j]), epoch, abort, kDeadlineTicks)) return;
      if (ni) {
        AcquireData(data);
        for (uint32_t j = 0; j < ni; ++j)
          for (uint32_t i = 0; i < elements; ++i)
            if (data.GetValue(inputs[j] * kElements + i) != Expected(inputs[j], epoch, i)) ++mismatches;
        PublishSingleOwnerEvent(cell(1, worker), epoch);
      }
      WriteControl(cell(3, worker), epoch);
    }
  } else if ASCEND_IS_AIV {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kElements * 4);
    auto local = buffer.Get<float>();
    for (uint32_t epoch = 1; epoch <= kEpochs; ++epoch) {
      if (ni) {
        float old = 0;
        for (uint32_t j = 0; j < ni; ++j)
          for (uint32_t i = 0; i < elements; ++i) old += data.GetValue(inputs[j] * kElements + i);
        WriteControl(cell(5, worker), uint32_t(old));
        PublishSingleOwnerEvent(cell(2, worker), epoch);
      }
      for (uint32_t j = 0; j < no; ++j)
        if (!WaitProgress(cell(1, outputs[j]), epoch - 1, abort, kDeadlineTicks) ||
            !WaitProgress(cell(2, outputs[j]), epoch, abort, kDeadlineTicks)) return;
      if (no) {
        AscendC::DataCopy(local, source[(epoch - 1) * kElements], elements);
        CompleteDma<AscendC::HardEvent::MTE2_S>();
        CompleteDma<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(data[worker * kElements], local, elements);
        CompleteDma<AscendC::HardEvent::MTE3_S>();
        PublishSingleOwnerEvent(cell(0, worker), epoch);
      }
      for (uint32_t j = 0; j < ni; ++j)
        if (!WaitProgress(cell(0, inputs[j]), epoch, abort, kDeadlineTicks)) return;
      if (ni) {
        AcquireData(data);
        for (uint32_t j = 0; j < ni; ++j)
          for (uint32_t i = 0; i < elements; ++i)
            if (data.GetValue(inputs[j] * kElements + i) != Expected(inputs[j], epoch, i)) ++mismatches;
        PublishSingleOwnerEvent(cell(1, worker), epoch);
      }
      WriteControl(cell(3, worker), epoch);
    }
  }
  WriteControl(cell(4, worker), mismatches);
}

__global__ __aicore__ __mix__(1, 2) __schedmode__(1)
void as2_progress(GM_ADDR identity, GM_ADDR values, GM_ADDR payload,
                   GM_ADDR controls, uint32_t groups) {
  for (uint32_t phase = 0; phase < 4; ++phase)
    Phase(identity, values, payload + phase * 3 * groups * kElements * 4,
          controls + phase * 7 * 3 * groups * kCell * 4, groups, phase,
          phase % 2 ? 32 : 1024);
}
static void Check(aclError code, const char* call) {
  if (code) throw std::runtime_error(std::string(call) + ": " + std::to_string(code));
}
int main(int argc, char** argv) {
  try {
    const unsigned groups = argc == 2 ? std::stoul(argv[1]) : 0;
    if (groups != 2 && groups != 4 && groups != 8 && groups != 14 && groups != 28)
      throw std::runtime_error("Require one explicit bounded group candidate: 2/4/8/14/28");
    Check(aclInit(nullptr), "init");
    uint32_t count = 0; Check(aclrtGetDeviceCount(&count), "device count");
    if (count != 1) throw std::runtime_error("Use one visible device");
    Check(aclrtSetDevice(0), "set device");
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950PR") != 0) throw std::runtime_error("Wrong SKU");
    const unsigned workers = 3 * groups;
    std::vector<float> identity(kElements, 0), values(workers * kEpochs * kElements);
    for (unsigned i = 0; i < 32; ++i) identity[32 * i + i] = 1;
    for (unsigned w = 0; w < workers; ++w)
      for (unsigned e = 1; e <= kEpochs; ++e)
        for (unsigned i = 0; i < kElements; ++i)
          values[(w * kEpochs + e - 1) * kElements + i] = 64 * (w + 1) + e + i % 4;
    std::vector<uint32_t> states(4 * 7 * workers * kCell, 0);
    std::vector<void*> memory;
    auto allocate = [&](size_t bytes) {
      void* p = nullptr; Check(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
      memory.push_back(p); Check(aclrtMemset(p, bytes, 0, bytes), "zero"); return (uint8_t*)p;
    };
    auto* a = allocate(identity.size() * 4);
    auto* v = allocate(values.size() * 4);
    auto* p = allocate(4 * workers * kElements * 4);
    auto* s = allocate(states.size() * 4);
    Check(aclrtMemcpy(a, identity.size() * 4, identity.data(), identity.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE), "copy identity");
    Check(aclrtMemcpy(v, values.size() * 4, values.data(), values.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE), "copy values");
    aclrtStream stream; Check(aclrtCreateStream(&stream), "stream");
    as2_progress<<<groups, nullptr, stream>>>(a, v, p, s, groups);
    Check(aclrtSynchronizeStream(stream), "synchronize");
    Check(aclrtMemcpy(states.data(), states.size() * 4, s, states.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST), "read state");
    for (unsigned phase = 0; phase < 4; ++phase) {
      auto* st = states.data() + phase * 7 * workers * kCell;
      if (st[6 * workers * kCell]) throw std::runtime_error("Device deadline/abort in phase " + std::to_string(phase));
      for (unsigned w = 0; w < workers; ++w) {
        const bool active = phase == 0 ? w < groups : phase == 3 ? w >= groups : true;
        if (st[4 * workers * kCell + w * kCell] ||
            st[3 * workers * kCell + w * kCell] != (active ? kEpochs : 0))
          throw std::runtime_error("Payload mismatch/missing worker: phase=" + std::to_string(phase) + " worker=" + std::to_string(w));
      }
    }
    for (void* m : memory) Check(aclrtFree(m), "free");
    Check(aclrtDestroyStream(stream), "destroy stream"); Check(aclFinalize(), "finalize");
    std::cout << "{\"status\":\"passed\",\"soc\":\"Ascend950PR_9579\",\"groups\":" << groups
              << ",\"workers\":" << workers << ",\"epochs\":16,\"directions\":4,\"stale_cache_preread\":true,"
                 "\"both_aiv_lanes_compute\":true,\"aic_payload\":\"CATLASS identity GEMM FIX output\","
                 "\"aiv_payload\":\"MTE2/MTE3\",\"mismatches\":0,\"wait_deadline_ticks\":2000000000,"
                 "\"host_kernel_launches\":1,\"tilemega_pipeline\":false}" << std::endl;
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
