// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
#if __CCE_AICORE__ == 310
inline constexpr AscendC::MicroAPI::CastTrait kRegisterSwiGluWiden={AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,AscendC::MicroAPI::MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
inline constexpr AscendC::MicroAPI::CastTrait kRegisterSwiGluRound={AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,AscendC::MicroAPI::MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
#endif
__aicore__ inline bool RegisterSwiGlu(AscendC::LocalTensor<bfloat16_t> gate,
    AscendC::LocalTensor<bfloat16_t> up,AscendC::LocalTensor<bfloat16_t> output,unsigned count) {
#if __CCE_AICORE__ == 310
  if(!count || count>1024)return false;
  auto* g=(__ubuf__ bfloat16_t*)gate.GetPhyAddr();
  auto* u=(__ubuf__ bfloat16_t*)up.GetPhyAddr();
  auto* o=(__ubuf__ bfloat16_t*)output.GetPhyAddr();
  const uint16_t loops=(count+63)/64;
  __VEC_SCOPE__ {
    using namespace AscendC::MicroAPI;
    RegTensor<bfloat16_t> rg,ru,rounded;
    RegTensor<float> x,y,value;
    uint32_t left=count;
    for(uint16_t i=0;i<loops;++i) {
      auto mask=UpdateMask<float>(left);
      DataCopy<bfloat16_t,LoadDist::DIST_UNPACK_B16>(rg,g+i*64);
      DataCopy<bfloat16_t,LoadDist::DIST_UNPACK_B16>(ru,u+i*64);
      Cast<float,bfloat16_t,kRegisterSwiGluWiden>(x,rg,mask);Cast<float,bfloat16_t,kRegisterSwiGluWiden>(y,ru,mask);
      Muls(value,x,-1.0f,mask);Exp(value,value,mask);Adds(value,value,1.0f,mask);
      Div(value,x,value,mask);
      // BF16 SiLU is a semantic graph boundary, even without a UB roundtrip.
      Cast<bfloat16_t,float,kRegisterSwiGluRound>(rounded,value,mask);
      Cast<float,bfloat16_t,kRegisterSwiGluWiden>(value,rounded,mask);
      Mul(value,value,y,mask);Cast<bfloat16_t,float,kRegisterSwiGluRound>(rounded,value,mask);
      DataCopy<bfloat16_t,StoreDist::DIST_PACK_B32>(o+i*64,rounded,mask);
    }
  }
  return true;
#else
  return false;
#endif
}
} // namespace tilemega::ascend
