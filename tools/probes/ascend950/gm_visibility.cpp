#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <tilemega/Backend/Ascend/GemmTile.h>
#include <acl/acl.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr unsigned kElements = 32*32;
constexpr unsigned kEpochs = 16;
constexpr unsigned kCellWords = 128;

__aicore__ inline void RunPhase(GM_ADDR matrix_a, GM_ADDR epochs_b, GM_ADDR payload,
                   GM_ADDR controls, unsigned direction, unsigned elements) {
  const unsigned block = AscendC::GetBlockIdx();
  constexpr bool aic = g_coreType == AscendC::AIC;
  const bool producer_aic = direction < 2;
  const bool consumer_aic = direction == 0 || direction == 2;
  const bool producer = aic == producer_aic && block == 0;
  const bool consumer = aic == consumer_aic && block == (aic ? 1u : 2u);
  if (!producer && !consumer) return;
  auto* state = (__gm__ uint32_t*)controls;
  auto* ready = state;
  auto* ack = state+kCellWords;
  auto* preloaded = state+2*kCellWords;
  auto* abort = state+3*kCellWords;
  auto* error = state+(producer ? 4 : 5)*kCellWords;
  AscendC::GlobalTensor<uint32_t> data, expected;
  data.SetGlobalBuffer((__gm__ uint32_t*)payload);
  expected.SetGlobalBuffer((__gm__ uint32_t*)epochs_b);
  if (producer) {
    if ASCEND_IS_AIC {
      GemmTile<float, float> tile;
      for (unsigned epoch = 1; epoch <= kEpochs; ++epoch) {
        if (!WaitEvent(ack, epoch-1, abort) || !WaitEvent(preloaded, epoch, abort)) {
          WriteControl(error, epoch); return;
        }
        tile.RunTask(matrix_a, epochs_b+(epoch-1)*kElements*4, payload, 32, 32, 32, 0, 0);
        CompleteDma<AscendC::HardEvent::FIX_S>();
        PublishSingleOwnerEvent(ready, epoch);
      }
    } else if ASCEND_IS_AIV {
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
      pipe.InitBuffer(buffer, kElements*4);
      auto local = buffer.Get<uint32_t>();
      for (unsigned epoch = 1; epoch <= kEpochs; ++epoch) {
        if (!WaitEvent(ack, epoch-1, abort) || !WaitEvent(preloaded, epoch, abort)) {
          WriteControl(error, epoch); return;
        }
        AscendC::DataCopy(local, expected[(epoch-1)*kElements], elements);
        CompleteDma<AscendC::HardEvent::MTE2_S>();
        CompleteDma<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(data, local, elements);
        CompleteDma<AscendC::HardEvent::MTE3_S>();
        PublishSingleOwnerEvent(ready, epoch);
      }
    }
  } else {
    uint32_t mismatches = 0;
    for (unsigned epoch = 1; epoch <= kEpochs; ++epoch) {
      uint32_t old = 0;
      for (unsigned i = 0; i < elements; ++i) old ^= data.GetValue(i);
      // Make the pre-read externally observable before allowing the overwrite.
      WriteControl(state+(6+epoch)*kCellWords, old);
      PublishSingleOwnerEvent(preloaded, epoch);
      if (!WaitEvent(ready, epoch, abort)) { WriteControl(error, epoch); return; }
      AcquireData(data);
      for (unsigned i = 0; i < elements; ++i)
        if (data.GetValue(i) != expected.GetValue((epoch-1)*kElements+i)) ++mismatches;
      WriteControl(state+(24+epoch)*kCellWords, mismatches);
      PublishSingleOwnerEvent(ack, epoch);
    }
    WriteControl(error, mismatches);
  }
}

__global__ __aicore__ __mix__(1, 2) __schedmode__(1)
void gm_visibility(GM_ADDR matrix_a, GM_ADDR epochs_b, GM_ADDR payload,
                   GM_ADDR controls, unsigned direction, unsigned elements) {
  if (direction < 4) RunPhase(matrix_a, epochs_b, payload, controls, direction, elements);
  else {
    for (unsigned phase = 0; phase < 4; ++phase)
      RunPhase(matrix_a, epochs_b, payload+phase*kElements*4,
               controls+phase*64*kCellWords*4, phase, elements);
  }
}

static void Check(aclError error, char const* name) {
  if (error) throw std::runtime_error(std::string(name)+": "+std::to_string(error));
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: as1_gm_visibility direction(0..3,4=all) elements(32/1024)");
    unsigned direction = std::stoul(argv[1]), elements = std::stoul(argv[2]);
    if (direction > 4 || (elements != 32 && elements != kElements)) throw std::runtime_error("Bounded probe shape required");
    Check(aclInit(nullptr), "init");
    uint32_t visible = 0; Check(aclrtGetDeviceCount(&visible), "device count");
    if (visible != 1) throw std::runtime_error("Use single-device wrapper");
    Check(aclrtSetDevice(0), "set device");
    auto* soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950PR") != 0) throw std::runtime_error("Wrong SKU");
    aclrtStream stream = nullptr; Check(aclrtCreateStream(&stream), "stream");
    std::vector<float> a(kElements, 0), b(kElements*kEpochs);
    for (unsigned i = 0; i < 32; ++i) a[i*32+i] = 1;
    for (unsigned epoch = 0; epoch < kEpochs; ++epoch)
      for (unsigned i = 0; i < kElements; ++i) b[epoch*kElements+i] = float(epoch+1+i%4);
    const unsigned phases = direction == 4 ? 4 : 1;
    std::vector<uint32_t> states(phases*64*kCellWords, 0);
    uint8_t *da = nullptr, *db = nullptr, *data = nullptr, *state = nullptr;
    auto allocate = [&](uint8_t** p, size_t size) {
      Check(aclrtMalloc((void**)p, size, ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
      Check(aclrtMemset(*p, size, 0, size), "zero");
    };
    allocate(&da, a.size()*4); allocate(&db, b.size()*4);
    allocate(&data, phases*kElements*4); allocate(&state, states.size()*4);
    Check(aclrtMemcpy(da, a.size()*4, a.data(), a.size()*4, ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
    Check(aclrtMemcpy(db, b.size()*4, b.data(), b.size()*4, ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
    gm_visibility<<<2, nullptr, stream>>>(da, db, data, state, direction, elements);
    Check(aclrtSynchronizeStream(stream), "synchronize");
    Check(aclrtMemcpy(states.data(), states.size()*4, state, states.size()*4, ACL_MEMCPY_DEVICE_TO_HOST), "read states");
    for (unsigned phase = 0; phase < phases; ++phase) {
      auto* s = states.data()+phase*64*kCellWords;
      if (s[3*kCellWords] || s[4*kCellWords] || s[5*kCellWords] ||
          s[0] != kEpochs || s[kCellWords] != kEpochs || s[2*kCellWords] != kEpochs)
        throw std::runtime_error("Bounded message passing failed: phase="+std::to_string(phase)+
                                 " abort="+std::to_string(s[3*kCellWords])+
                                 " producer="+std::to_string(s[4*kCellWords])+" consumer="+std::to_string(s[5*kCellWords]));
      for (unsigned epoch = 1; epoch <= kEpochs; ++epoch)
        if (s[(24+epoch)*kCellWords]) throw std::runtime_error("Stale epoch observed");
    }
    for (auto p : {da, db, data, state}) Check(aclrtFree(p), "free");
    Check(aclrtDestroyStream(stream), "destroy stream"); Check(aclFinalize(), "finalize");
    std::cout << "{\"status\":\"passed\",\"direction\":" << direction
              << ",\"groups\":2,\"epochs\":16,\"elements\":" << elements
              << ",\"pre_read_old_cache\":true,\"mismatches\":0,\"host_kernel_launches\":1,\"tilemega_pipeline\":false}" << std::endl;
    return 0;
  } catch (std::exception const& error) { std::cerr << error.what() << std::endl; return 1; }
}
